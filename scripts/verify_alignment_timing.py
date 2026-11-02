"""Test whether alignment boundaries fit the audio, using acoustics rather than phone labels.

Each clip's measures are computed twice: with its own alignment, and with a control that
stretches another clip's alignment over it. The control shows what these measures give when
boundaries have no real link to the audio. Contrasts well above it are evidence that timing is
broadly sensible, but they are not boundary accuracy: that needs hand-labelled reference
boundaries, which this project does not have. The script also draws spectrograms with phone
boundaries for inspection by eye.
"""
import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import tgt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.features import extract_f0  # noqa: E402
from src.config import load_configs  # noqa: E402
from src.data.corpus import read_manifests  # noqa: E402

VOWELS = {"a", "aa", "i", "ii", "u", "uu", "e", "ee", "o", "oo", "ai", "au"}
FRICATIVES = {"s", "sh", "ss"}
UNVOICED_STOPS = {"k", "t", "tt", "p", "c"}
ARPABET_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
ARPABET_FRICATIVES = {"S", "SH"}
ARPABET_UNVOICED_STOPS = {"K", "T", "P", "CH"}
SILENCE = {"", "sil", "sp", "spn"}
HOP = 256
RATE = 22050


def intervals(path):
    tier = tgt.io.read_textgrid(path).get_tier_by_name("phones")
    out, cursor = [], 0.0
    for item in tier.intervals:
        start, end = float(item.start_time), float(item.end_time)
        if start > cursor:
            out.append((cursor, start, "sil"))
        out.append((start, end, item.text.strip()))
        cursor = end
    if float(tier.end_time) > cursor:
        out.append((cursor, float(tier.end_time), "sil"))
    return out


def frame_features(wav):
    spec = np.abs(librosa.stft(wav, n_fft=1024, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=RATE, n_fft=1024)
    energy = 10 * np.log10(spec.sum(axis=0) + 1e-10)
    high = spec[freqs >= 4000].sum(axis=0) / (spec.sum(axis=0) + 1e-10)
    f0 = extract_f0(wav, RATE, HOP)
    n = min(len(energy), len(f0))
    return energy[:n], high[:n], (f0[:n] > 0)


def category(label):
    if label in SILENCE:
        return "sil"
    # Telugu labels are lower case and ARPAbet upper case, so the sets cannot collide.
    label = label.rstrip("012")
    if label in VOWELS or label in ARPABET_VOWELS:
        return "vowel"
    if label in FRICATIVES or label in ARPABET_FRICATIVES:
        return "fricative"
    if label in UNVOICED_STOPS or label in ARPABET_UNVOICED_STOPS:
        return "stop"
    return None


def score(features, spans, stretch=1.0):
    energy, high, voiced = features
    groups = collections.defaultdict(list)
    for start, end, label in spans:
        a, b = int(start * stretch * RATE / HOP), int(end * stretch * RATE / HOP)
        if b <= a or a >= len(energy):
            continue
        key = category(label)
        if key:
            groups[key].append((energy[a:b].mean(), high[a:b].mean(), voiced[a:b].mean()))
    summary = {}
    if groups["vowel"] and groups["sil"]:
        summary["vowel_minus_silence_db"] = float(np.mean([g[0] for g in groups["vowel"]]) - np.mean([g[0] for g in groups["sil"]]))
    if groups["vowel"] and groups["fricative"]:
        summary["fricative_minus_vowel_high_band"] = float(np.mean([g[1] for g in groups["fricative"]]) - np.mean([g[1] for g in groups["vowel"]]))
    if groups["vowel"] and groups["stop"]:
        summary["vowel_minus_stop_voicing"] = float(np.mean([g[2] for g in groups["vowel"]]) - np.mean([g[2] for g in groups["stop"]]))
    return summary


def draw(uid, wav, spans, text, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=wav, sr=RATE, n_fft=1024, hop_length=HOP, n_mels=80))
    seconds = len(wav) / RATE
    figure, axis = plt.subplots(figsize=(max(12, seconds * 3.2), 4.2))
    axis.imshow(mel, origin="lower", aspect="auto", extent=[0, seconds, 0, 80], cmap="magma")
    for start, end, label in spans:
        axis.axvline(start, color="white", linewidth=0.6)
        axis.text((start + end) / 2, 74, label or "sil", color="cyan", ha="center", fontsize=9)
    axis.set_xlabel("seconds")
    axis.set_title(uid + " (phones over mel spectrogram)")
    figure.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    figure.savefig(os.path.join(out_dir, uid + ".png"), dpi=90)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=["english", "telugu"])
    parser.add_argument("--clips", type=int, default=300)
    parser.add_argument("--plots", nargs="*", default=[])
    parser.add_argument("--out", default="output/alignment_check")
    args = parser.parse_args()
    preprocess, _, _ = load_configs(args.language)
    interim = preprocess["path"]["interim_path"]
    records = [r for rows in read_manifests(preprocess).values() for r in rows]
    by_id = {r["id"]: r for r in records}

    def paths(record):
        return (
            os.path.join(interim, "mfa_corpus", record["speaker"], record["id"] + ".wav"),
            os.path.join(interim, "textgrids", record["speaker"], record["id"] + ".TextGrid"),
        )

    for uid in args.plots:
        record = by_id[uid]
        wav_path, grid = paths(record)
        wav, _ = librosa.load(wav_path, sr=RATE)
        draw(uid, wav, intervals(grid), record["text"], args.out)

    usable = [r for r in records if os.path.exists(paths(r)[1])]
    rng = random.Random(1234)
    sample = rng.sample(usable, min(args.clips, len(usable)))
    real, control = collections.defaultdict(list), collections.defaultdict(list)
    interval_counts = collections.Counter()
    for record in sample:
        wav_path, grid = paths(record)
        wav, _ = librosa.load(wav_path, sr=RATE)
        features = frame_features(wav)
        other = rng.choice(usable)
        while other["id"] == record["id"]:
            other = rng.choice(usable)
        other_spans = intervals(paths(other)[1])
        own_spans = intervals(grid)
        interval_counts.update(category(label) or "not_compared" for _, _, label in own_spans)
        for key, value in score(features, own_spans).items():
            real[key].append(value)
        stretch = (len(wav) / RATE) / max(other_spans[-1][1], 1e-3)
        for key, value in score(features, other_spans, stretch).items():
            control[key].append(value)
    result = {
        key: {
            "alignment_median": round(float(np.median(real[key])), 3),
            "control_median": round(float(np.median(control[key])), 3),
            "alignment_positive_fraction": round(float(np.mean(np.array(real[key]) > 0)), 3),
            "clips": len(real[key]),
        }
        for key in real
    }
    print(json.dumps({"intervals_per_category": interval_counts, "measures": result}, indent=1))
    print("These are plausibility contrasts against a control, not ground-truth alignment accuracy.")


if __name__ == "__main__":
    main()
