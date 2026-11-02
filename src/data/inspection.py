"""Verify access, inspect a dataset and write a report. All values are measured, none assumed.

load_dataset runs in streaming mode, which checks access and schema without a second full copy
of the data. Full statistics come from the Parquet files the pipeline itself reads.
"""
import argparse
import collections
import glob
import json
import os

import pyarrow.parquet as pq
import soundfile as sf

from ..config import load_configs
from .corpus import download, inspect, read_inventory, read_manifests, select
from .validation import speaker_columns, text_notes


def verify_access(preprocess, token=None, samples=3):
    """Run a real authenticated load and return splits, features and a few text samples."""
    from datasets import Audio, load_dataset

    kwargs = {"streaming": True}
    if token is not None:
        kwargs["token"] = token
    loaded = load_dataset(preprocess["hf_repo"], **kwargs)
    audio_column = preprocess["columns"]["audio"]
    text_column = preprocess["columns"]["text"]
    report = {"repo": preprocess["hf_repo"], "splits": {}}
    for split, data in loaded.items():
        features = data.features
        report["config_name"] = data.info.config_name
        report["splits"][split] = {
            "declared_examples": data.info.splits[split].num_examples if data.info.splits else None,
            "features": {k: str(v) for k, v in features.items()},
        }
        raw = data.cast_column(audio_column, Audio(decode=False))
        report["splits"][split]["samples"] = [
            {k: v for k, v in row.items() if k != audio_column} for row in raw.take(samples)
        ]
        if text_column not in features or audio_column not in features:
            report.setdefault("incompatibilities", []).append(
                "split {} lacks column {} or {}".format(split, text_column, audio_column)
            )
    return report


def parquet_schema(preprocess):
    shards = sorted(glob.glob(os.path.join(preprocess["path"]["parquet_dir"], "data", "*.parquet")))
    if not shards:
        return None
    files = [pq.ParquetFile(s) for s in shards]
    schema = files[0].schema_arrow
    return {
        "shards": len(files),
        "rows": sum(f.metadata.num_rows for f in files),
        "columns": {name: str(schema.field(name).type) for name in schema.names},
    }


def _hours(records):
    return round(sum(r["seconds"] for r in records) / 3600, 3)


def speaker_summary(records, selection):
    groups = collections.defaultdict(list)
    for record in records:
        groups[record["speaker"]].append(record)
    out = {}
    for speaker, rows in sorted(groups.items()):
        valid = [r for r in rows if not r.get("audio_error")]
        usable = [
            r for r in valid
            if selection["min_seconds"] <= r["seconds"] <= selection["max_seconds"] and not r["text_error"]
        ]
        out[speaker] = {
            "clips": len(rows),
            "hours": _hours(valid),
            "sample_rates": dict(collections.Counter(str(r["sample_rate"]) for r in valid)),
            "usable_clips": len(usable),
            "usable_hours": _hours(usable),
        }
    return out


def duration_column_check(records, column="duration"):
    """Compare a dataset's own duration column, if any, with the duration measured from audio."""
    pairs = [
        (r["meta"][column], r["seconds"]) for r in records
        if isinstance(r.get("meta", {}).get(column), (int, float)) and not r.get("audio_error")
    ]
    if not pairs:
        return None
    return {
        "column": column,
        "compared": len(pairs),
        "over_100ms_apart": sum(1 for stated, measured in pairs if abs(stated - measured) > 0.1),
    }


def build_report(preprocess, inventory, schema=None, access=None):
    selection = preprocess["selection"]
    needed = selection["train_hours"] + selection["val_hours"]
    speakers = speaker_summary(inventory, selection)
    chosen = selection.get("speakers") or list(speakers)
    usable_hours = sum(speakers[s]["usable_hours"] for s in chosen if s in speakers)
    notes = collections.Counter(
        note for r in inventory if isinstance(r["text"], str) for note in text_notes(r["text"])
    )
    columns = list(schema["columns"]) if schema else sorted(inventory[0].get("meta", {}))
    required = [preprocess["columns"]["audio"], preprocess["columns"]["text"]]
    report = {
        "language": preprocess["language"],
        "access": access,
        "parquet_schema": schema,
        "examples": len(inventory),
        "missing_or_invalid_audio": collections.Counter(
            r["audio_error"] for r in inventory if r.get("audio_error")
        ),
        "invalid_transcripts": collections.Counter(
            r["text_error"].split("'")[0] for r in inventory if r["text_error"]
        ).most_common(15),
        "transcript_notes": notes,
        "speaker_like_columns": speaker_columns(columns),
        "speaker_groups": speakers,
        "duration_column": duration_column_check(inventory),
        "selected_speakers": selection.get("speakers"),
        "required_hours": needed,
        "usable_hours_for_selection": round(usable_hours, 3),
        "enough_audio": usable_hours >= needed,
        "missing_required_columns": [c for c in required if schema and c not in schema["columns"]],
    }
    if schema and not report["speaker_like_columns"] and preprocess["language"] == "english":
        report["speaker_note"] = (
            "No speaker column. Clips form one group, and no identity was inferred from metadata."
        )
    try:
        select(inventory, selection)
        report["selection_check"] = "ok"
    except ValueError as error:
        report["selection_check"] = str(error)
    return report


def verify_export(preprocess, inventory):
    """Check an existing manifest and audio export against the inventory, without changing them."""
    rate = preprocess["preprocessing"]["audio"]["sampling_rate"]
    interim = preprocess["path"]["interim_path"]
    by_row = {r["row"]: r for r in inventory}
    try:
        manifests = read_manifests(preprocess)
    except OSError:
        return {"status": "no export found"}
    # A restored archive has features but not the exported audio, which is then not checked here.
    audio_present = os.path.isdir(os.path.join(interim, "wav"))
    problems = collections.Counter()
    examples = []
    total = 0

    def note(kind, uid):
        problems[kind] += 1
        if len(examples) < 5:
            examples.append("{}: {}".format(kind, uid))

    train_rows = {r["row"] for r in manifests["train"]}
    val_rows = {r["row"] for r in manifests["val"]}
    if train_rows & val_rows:
        problems["train and validation overlap"] += len(train_rows & val_rows)
    for records in manifests.values():
        for record in records:
            total += 1
            source = by_row.get(record["row"])
            if source is None:
                note("row not in inventory", record["id"])
                continue
            if source["text"] != record["text"]:
                note("transcript differs from dataset", record["id"])
            if abs(source["seconds"] - record["seconds"]) > 0.01:
                note("duration differs from dataset", record["id"])
            path = os.path.join(interim, "wav", record["id"] + ".wav")
            if not audio_present:
                continue
            if not os.path.exists(path):
                note("exported audio missing", record["id"])
                continue
            info = sf.info(path)
            if info.samplerate != rate or info.channels != 1:
                note("wrong audio format", record["id"])
            if abs(info.frames / info.samplerate - source["seconds"]) > 0.05:
                note("exported duration differs", record["id"])
            if not os.path.exists(os.path.join(interim, "mfa_corpus", record["speaker"], record["id"] + ".lab")):
                note("label file missing", record["id"])
    return {
        "status": "valid" if not problems else "problems found",
        "clips_checked": total,
        "audio_files_checked": audio_present,
        "train_hours": round(sum(r["seconds"] for r in manifests["train"]) / 3600, 3),
        "val_hours": round(sum(r["seconds"] for r in manifests["val"]) / 3600, 3),
        "problems": dict(problems),
        "examples": examples,
    }


def print_summary(report):
    print("Dataset:", report["language"], "| examples:", report["examples"])
    if report["access"]:
        print("Load verified:", report["access"]["repo"], "| config:", report["access"].get("config_name"))
        for split, data in report["access"]["splits"].items():
            print(" split", split, "| declared examples:", data["declared_examples"])
            print("  features:", data["features"])
            for sample in data["samples"]:
                print("  sample:", sample)
    print("Parquet:", report["parquet_schema"])
    print("Invalid audio:", dict(report["missing_or_invalid_audio"]))
    print("Invalid transcripts:", report["invalid_transcripts"])
    print("Transcript notes:", dict(report["transcript_notes"]))
    print("Speaker-like columns:", report["speaker_like_columns"], report.get("speaker_note", ""))
    print("Duration column check:", report["duration_column"])
    for speaker, data in report["speaker_groups"].items():
        print(" group", speaker, data)
    print(
        "Selection {}: usable {} h of {} h needed; enough: {}; check: {}".format(
            report["selected_speakers"], report["usable_hours_for_selection"],
            report["required_hours"], report["enough_audio"], report["selection_check"],
        )
    )
    if report.get("export"):
        print("Existing export:", report["export"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=["english", "telugu"])
    parser.add_argument("--refresh", action="store_true", help="rescan all audio even if an inventory exists")
    parser.add_argument("--skip_access_check", action="store_true")
    args = parser.parse_args()
    preprocess, _, _ = load_configs(args.language)

    access = None if args.skip_access_check else verify_access(preprocess)
    download(preprocess)
    inventory_path = os.path.join(preprocess["path"]["interim_path"], "inventory.jsonl")
    needs_scan = args.refresh or not os.path.exists(inventory_path)
    if not needs_scan:
        schema = parquet_schema(preprocess)
        needs_scan = sum(1 for _ in open(inventory_path, encoding="utf-8")) != schema["rows"]
    if not needs_scan and "audio_error" not in read_inventory(inventory_path)[0]:
        needs_scan = True  # written by an earlier version without these checks
    if needs_scan:
        inspect(preprocess, inventory_path)
    inventory = read_inventory(inventory_path)
    report = build_report(preprocess, inventory, parquet_schema(preprocess), access)
    report["export"] = verify_export(preprocess, inventory)
    print_summary(report)
    directory = os.path.join(os.environ["FS2_EXP_ROOT"], args.language)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "dataset_inspection.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print("Report:", path)


if __name__ == "__main__":
    main()
