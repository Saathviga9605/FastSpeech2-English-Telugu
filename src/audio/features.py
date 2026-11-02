"""Mel, F0, energy and duration extraction, adapted from upstream preprocessor/preprocessor.py.

Differences from upstream: normalization statistics come from the training split only,
clips that fail are listed with a reason, and a missing alignment cannot reuse stale values.
"""
import argparse
import collections
import json
import os

import librosa
import numpy as np
import pyworld as pw
import torch
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

from ..alignment.prepare import SILENCE_LABELS, expected_phones, read_alignment, read_lexicon
from ..config import load_configs, load_symbols
from ..data.corpus import read_manifests
from .stft import TacotronSTFT


def build_stft(preprocess_config):
    p = preprocess_config["preprocessing"]
    return TacotronSTFT(
        p["stft"]["filter_length"], p["stft"]["hop_length"], p["stft"]["win_length"],
        p["mel"]["n_mel_channels"], p["audio"]["sampling_rate"], p["mel"]["mel_fmin"], p["mel"]["mel_fmax"],
    )


def mel_and_energy(wav, stft):
    audio = torch.clip(torch.FloatTensor(wav).unsqueeze(0), -1, 1)
    with torch.no_grad():
        mel, energy = stft.mel_spectrogram(audio)
    return mel.squeeze(0).numpy().astype(np.float32), energy.squeeze(0).numpy().astype(np.float32)


def extract_f0(wav, sampling_rate, hop_length):
    wav = wav.astype(np.float64)
    f0, times = pw.dio(wav, sampling_rate, frame_period=hop_length / sampling_rate * 1000)
    return pw.stonemask(wav, f0, times, sampling_rate)


def phoneme_average(values, durations):
    out, position = np.zeros(len(durations), dtype=np.float32), 0
    for i, d in enumerate(durations):
        if d > 0:
            out[i] = np.mean(values[position : position + d])
        position += d
    return out


def interpolate_unvoiced(f0):
    voiced = np.where(f0 != 0)[0]
    fn = interp1d(
        voiced, f0[voiced], fill_value=(f0[voiced[0]], f0[voiced[-1]]), bounds_error=False,
    )
    return fn(np.arange(len(f0)))


def process_clip(wav_path, textgrid_path, stft, cfg, valid_phones):
    rate = cfg["audio"]["sampling_rate"]
    hop = cfg["stft"]["hop_length"]
    phones, durations, start, end = read_alignment(textgrid_path, rate, hop, valid_phones)
    if not phones or end <= start:
        raise ValueError("empty alignment")
    wav, _ = librosa.load(wav_path, sr=rate)
    wav = wav[int(rate * start) : int(rate * end)].astype(np.float32)
    total = sum(durations)
    mel, energy = mel_and_energy(wav, stft)
    if mel.shape[1] < total:
        raise ValueError("alignment has {} frames, audio has {}".format(total, mel.shape[1]))
    f0 = extract_f0(wav, rate, hop)
    if len(f0) < total:
        raise ValueError("F0 has {} frames, alignment has {}".format(len(f0), total))
    f0 = f0[:total]
    if np.sum(f0 != 0) <= 1:
        raise ValueError("fewer than two voiced frames")
    mel, energy = mel[:, :total], energy[:total]
    if cfg["pitch"]["feature"] == "phoneme_level":
        f0 = phoneme_average(interpolate_unvoiced(f0), durations)
    if cfg["energy"]["feature"] == "phoneme_level":
        energy = phoneme_average(energy, durations)
    return phones, np.array(durations), f0.astype(np.float32), energy.astype(np.float32), mel.T


def inliers(values):
    low, high = np.percentile(values, [25, 75])
    span = 1.5 * (high - low)
    # Inclusive bounds: a clip with one repeated value has zero spread and must not come back empty.
    return values[(values >= low - span) & (values <= high + span)]


def process_corpus(preprocess_config, symbols):
    cfg = preprocess_config["preprocessing"]
    out_dir = preprocess_config["path"]["preprocessed_path"]
    interim = preprocess_config["path"]["interim_path"]
    for kind in ("mel", "pitch", "energy", "duration"):
        os.makedirs(os.path.join(out_dir, kind), exist_ok=True)
    stft = build_stft(preprocess_config)
    valid_phones = set(symbols)
    scalers = {"pitch": StandardScaler(), "energy": StandardScaler()}
    manifests = read_manifests(preprocess_config)
    lexicon = read_lexicon(os.path.join(interim, "lexicon.txt"))
    missing_audio = [
        r["id"] for rows in manifests.values() for r in rows
        if not os.path.exists(os.path.join(interim, "mfa_corpus", r["speaker"], r["id"] + ".wav"))
    ]
    if missing_audio:
        raise FileNotFoundError(
            "{} clips have no audio, so features cannot be extracted (first: {}). A restored archive "
            "holds finished features and labels only; rerun export and prepare for a full dataset.".format(
                len(missing_audio), missing_audio[0]
            )
        )
    kept, failures, frames = {"train": [], "val": []}, [], {"train": 0, "val": 0}

    for split, records in manifests.items():
        for record in records:
            uid = record["id"]
            wav_path = os.path.join(interim, "mfa_corpus", record["speaker"], uid + ".wav")
            tg_path = os.path.join(interim, "textgrids", record["speaker"], uid + ".TextGrid")
            try:
                if not os.path.exists(tg_path):
                    raise ValueError("no alignment")
                phones, durations, f0, energy, mel = process_clip(wav_path, tg_path, stft, cfg, valid_phones)
                if [p for p in phones if p not in SILENCE_LABELS] != expected_phones(record, interim, lexicon):
                    raise ValueError("phones differ from lexicon")
                if not len(phones) == len(durations) == len(f0) == len(energy) or mel.shape[0] != durations.sum():
                    raise ValueError("feature lengths disagree")
            except ValueError as error:
                failures.append({"id": uid, "split": split, "reason": str(error)})
                continue
            for kind, data in (("duration", durations), ("pitch", f0), ("energy", energy), ("mel", mel)):
                np.save(os.path.join(out_dir, kind, uid + ".npy"), data)
            if split == "train":
                scalers["pitch"].partial_fit(inliers(f0).reshape(-1, 1))
                scalers["energy"].partial_fit(inliers(energy).reshape(-1, 1))
            frames[split] += int(durations.sum())
            kept[split].append({"id": uid, "speaker": record["speaker"], "phones": phones, "text": record["text"]})

    stats = {}
    for kind in ("pitch", "energy"):
        normalize = cfg[kind]["normalization"]
        mean = float(scalers[kind].mean_[0]) if normalize else 0.0
        std = float(scalers[kind].scale_[0]) if normalize else 1.0
        low, high = np.inf, -np.inf
        for split, rows in kept.items():
            for row in rows:
                path = os.path.join(out_dir, kind, row["id"] + ".npy")
                values = (np.load(path) - mean) / std
                np.save(path, values.astype(np.float32))
                if split == "train":
                    low, high = min(low, float(values.min())), max(high, float(values.max()))
        stats[kind] = [low, high, mean, std]

    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f)
    speakers = sorted({row["speaker"] for rows in kept.values() for row in rows})
    with open(os.path.join(out_dir, "speakers.json"), "w", encoding="utf-8") as f:
        json.dump({s: i for i, s in enumerate(speakers)}, f)
    for split, rows in kept.items():
        with open(os.path.join(out_dir, split + ".jsonl"), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "excluded.jsonl"), "w", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")
    frame_seconds = cfg["stft"]["hop_length"] / cfg["audio"]["sampling_rate"]
    phone_frames = {}
    zero_duration = total_phones = 0
    for rows in kept.values():
        for row in rows:
            durations = np.load(os.path.join(out_dir, "duration", row["id"] + ".npy"))
            zero_duration += int((durations == 0).sum())
            total_phones += len(durations)
            for phone, frames_count in zip(row["phones"], durations):
                phone_frames.setdefault(phone, []).append(int(frames_count))
    report = {
        "train_clips": len(kept["train"]), "val_clips": len(kept["val"]), "excluded": len(failures),
        "train_hours": round(frames["train"] * frame_seconds / 3600, 3),
        "val_hours": round(frames["val"] * frame_seconds / 3600, 3),
        "speakers": speakers, "stats": stats,
        "zero_duration_phones": zero_duration, "total_phones": total_phones,
        "mean_phone_ms": {
            p: round(float(np.mean(v)) * frame_seconds * 1000, 1) for p, v in sorted(phone_frames.items())
        },
        "exclusion_reasons": collections.Counter(f["reason"].split(":")[0] for f in failures),
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=["english", "telugu"])
    args = parser.parse_args()
    preprocess, _, _ = load_configs(args.language)
    print(json.dumps(process_corpus(preprocess, load_symbols(args.language)), indent=2))


if __name__ == "__main__":
    main()
