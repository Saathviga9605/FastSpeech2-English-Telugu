"""Read Hugging Face Parquet shards directly, inspect them, select clips and export audio.

Reading Parquet with pyarrow avoids any dependency on the `datasets` audio decoder.
"""
import argparse
import collections
import glob
import io
import json
import os
import random

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from ..config import load_configs
from ..text import get_frontend
from .storage import cache_problems, write_marker
from .validation import audio_properties, text_problem


def remote_state(preprocess_config):
    """Current revision and Parquet sizes on the Hub, using the logged-in token if there is one."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(preprocess_config["hf_repo"], files_metadata=True)
    files = {
        f.rfilename: f.size for f in info.siblings
        if f.rfilename.startswith("data/") and f.rfilename.endswith(".parquet")
    }
    return info.sha, files


def download(preprocess_config):
    """Fetch Parquet shards unless a complete copy exists. Gated datasets need a login first."""
    from huggingface_hub import snapshot_download

    target = preprocess_config["path"]["parquet_dir"]
    repo = preprocess_config["hf_repo"]
    try:
        revision, expected = remote_state(preprocess_config)
    except Exception as error:  # offline or no access: an already complete copy is still usable
        print("Could not reach the Hub ({}); checking the local copy.".format(type(error).__name__))
        if cache_problems(target, repo):
            raise
        return target
    if not cache_problems(target, repo, expected):
        write_marker(target, repo, revision, expected)
        print("Using complete cached dataset:", target)
        return target
    snapshot_download(
        repo, repo_type="dataset", local_dir=target, revision=revision,
        allow_patterns=["data/*.parquet", "README.md"],
    )
    problems = cache_problems(target, repo, expected)
    if problems:
        raise RuntimeError("Download incomplete: {}".format(problems))
    write_marker(target, repo, revision, expected)
    return target


def iter_rows(preprocess_config, with_audio=True):
    """Yield row dictionaries with a stable row number across all shards."""
    audio_col = preprocess_config["columns"]["audio"]
    shards = sorted(glob.glob(os.path.join(preprocess_config["path"]["parquet_dir"], "data", "*.parquet")))
    if not shards:
        raise FileNotFoundError("No Parquet shards in {}".format(preprocess_config["path"]["parquet_dir"]))
    index = 0
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                row["_row"] = index
                row["_audio"] = row.pop(audio_col)
                index += 1
                yield row


def speaker_of(row, preprocess_config):
    """Speaker label: dataset-specific. Metadata-derived labels are only approximate identities."""
    if preprocess_config["language"] == "telugu":
        return {0: "female", 1: "male"}.get(row["gender"], str(row["gender"]))
    explicit = row.get("speaker_id") or row.get("speaker")
    return str(explicit) if explicit else "svarah"


def inspect(preprocess_config, inventory_path):
    """Write one JSON line per clip with measured properties. Bad rows are recorded, not skipped."""
    language = preprocess_config["language"]
    frontend = get_frontend(language, preprocess_config)
    options = {"resolve_homographs": False} if language == "english" else {}
    text_column = preprocess_config["columns"]["text"]
    os.makedirs(os.path.dirname(inventory_path), exist_ok=True)
    tmp = inventory_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as out:
        for row in iter_rows(preprocess_config):
            text = row[text_column]
            text_error = text_problem(text, language)
            if text_error is None:
                try:
                    if not frontend(text, strict=True, **options).phones:
                        text_error = "transcript has nothing pronounceable"
                except ValueError as error:
                    text_error = str(error)
            cell = row["_audio"]
            record = {
                "row": row["_row"], "speaker": speaker_of(row, preprocess_config), "text": text,
                "audio_path": cell.get("path") if cell else None, "text_error": text_error,
                "audio_error": None, "seconds": 0.0, "sample_rate": None,
            }
            try:
                record.update(audio_properties(cell))
            except ValueError as error:
                record["audio_error"] = str(error)
            record["meta"] = {k: v for k, v in row.items() if not k.startswith("_") and k != text_column}
            out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, inventory_path)
    return inventory_path


def read_inventory(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def select(inventory, selection):
    """Seeded, disjoint train and validation selection by duration."""
    speakers = selection.get("speakers")
    pool = [
        r for r in inventory
        if selection["min_seconds"] <= r["seconds"] <= selection["max_seconds"]
        and not r["text_error"]
        and not r.get("audio_error")
        and (speakers is None or r["speaker"] in speakers)
    ]
    pool.sort(key=lambda r: r["row"])
    random.Random(selection["seed"]).shuffle(pool)
    val, train, val_s, train_s = [], [], 0.0, 0.0
    # Svarah repeats some sentences across recordings; optionally keep them on one side only.
    disjoint_text = selection.get("disjoint_text", False)
    val_texts = set()
    val_target = selection["val_hours"] * 3600
    train_target = selection["train_hours"] * 3600
    for record in pool:
        if val_s < val_target:
            val.append(record)
            val_s += record["seconds"]
            if disjoint_text:
                val_texts.add(record["text"])
        elif disjoint_text and record["text"] in val_texts:
            continue
        elif train_s < train_target:
            train.append(record)
            train_s += record["seconds"]
        else:
            break
    if train_s < train_target * 0.95 or val_s < val_target * 0.95:
        raise ValueError(
            "Only {:.2f} h train and {:.2f} h validation are available".format(train_s / 3600, val_s / 3600)
        )
    return train, val


def write_manifests(preprocess_config, train, val):
    directory = preprocess_config["path"]["manifest_dir"]
    os.makedirs(directory, exist_ok=True)
    language = preprocess_config["language"]
    for split, records in (("train", train), ("val", val)):
        path = os.path.join(directory, "{}_{}.jsonl".format(language, split))
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({
                    "id": "{}_{:06d}".format(language, r["row"]), "row": r["row"],
                    "speaker": r["speaker"], "seconds": round(r["seconds"], 3), "text": r["text"],
                }, ensure_ascii=False) + "\n")


def read_manifests(preprocess_config):
    directory = preprocess_config["path"]["manifest_dir"]
    language = preprocess_config["language"]
    out = {}
    for split in ("train", "val"):
        with open(os.path.join(directory, "{}_{}.jsonl".format(language, split)), encoding="utf-8") as f:
            out[split] = [json.loads(line) for line in f]
    return out


def export_audio(preprocess_config):
    """Write selected clips as mono PCM at the configured sample rate, peak-normalized as upstream."""
    manifests = read_manifests(preprocess_config)
    wanted = {r["row"]: r for records in manifests.values() for r in records}
    rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    root = os.path.join(preprocess_config["path"]["interim_path"], "wav")
    os.makedirs(root, exist_ok=True)
    written = 0
    for row in iter_rows(preprocess_config):
        record = wanted.get(row["_row"])
        if record is None:
            continue
        wav, _ = librosa.load(io.BytesIO(row["_audio"]["bytes"]), sr=rate, mono=True)
        peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
        if peak == 0.0:
            raise ValueError("Silent audio in row {}".format(row["_row"]))
        sf.write(os.path.join(root, record["id"] + ".wav"), wav / peak * 0.999, rate, subtype="PCM_16")
        written += 1
    if written != len(wanted):
        raise RuntimeError("Exported {} of {} clips".format(written, len(wanted)))
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=["english", "telugu"])
    parser.add_argument("action", choices=["download", "select", "export"])
    args = parser.parse_args()
    preprocess, _, _ = load_configs(args.language)
    inventory_path = os.path.join(preprocess["path"]["interim_path"], "inventory.jsonl")
    if args.action == "download":
        print(download(preprocess))
    elif args.action == "select":
        train, val = select(read_inventory(inventory_path), preprocess["selection"])
        write_manifests(preprocess, train, val)
        print("train clips", len(train), "hours", sum(r["seconds"] for r in train) / 3600)
        print("val clips", len(val), "hours", sum(r["seconds"] for r in val) / 3600)
        print("speakers", collections.Counter(r["speaker"] for r in train + val))
    else:
        print("exported", export_audio(preprocess))


if __name__ == "__main__":
    main()
