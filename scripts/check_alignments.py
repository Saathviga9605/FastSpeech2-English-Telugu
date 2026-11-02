"""Report real alignment coverage and quality for a person to review.

Some rejected clips are normal, so rejections alone do not fail this tool. It exits nonzero
only when the required training and validation hours are no longer available.

This reads TextGrids, labels and the lexicon, so it also works on a restored archive without
audio. It therefore cannot show that boundaries fit the audio; verify_alignment_timing.py does.
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np
import tgt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alignment.prepare import SILENCE_LABELS, expected_phones, read_alignment, read_lexicon  # noqa: E402
from src.config import load_configs, load_symbols  # noqa: E402
from src.data.corpus import read_manifests  # noqa: E402

SHORT_MS = 15
LONG_MS = 500


def check_clip(record, interim, lexicon, valid, rate, hop):
    """Return (phones, frames) for a clip, or raise ValueError with the reason."""
    path = os.path.join(interim, "textgrids", record["speaker"], record["id"] + ".TextGrid")
    if not os.path.exists(path):
        raise ValueError("no TextGrid")
    try:
        phones, frames, _, _ = read_alignment(path, rate, hop, valid)
    except ValueError:
        raise
    except Exception as error:  # a damaged file can raise many parser errors
        raise ValueError("unreadable TextGrid: {}".format(type(error).__name__)) from None
    if not phones:
        raise ValueError("empty alignment")
    spoken = [p for p in phones if p not in SILENCE_LABELS]
    if spoken != expected_phones(record, interim, lexicon):
        raise ValueError("phones differ from lexicon")
    return phones, frames


def check_language(preprocess, symbols):
    cfg = preprocess["preprocessing"]
    rate, hop = cfg["audio"]["sampling_rate"], cfg["stft"]["hop_length"]
    frame_ms = hop / rate * 1000
    interim = preprocess["path"]["interim_path"]
    valid = set(symbols)
    lexicon = read_lexicon(os.path.join(interim, "lexicon.txt"))
    wavs_present = any(
        name.endswith(".wav")
        for _, _, names in os.walk(os.path.join(interim, "mfa_corpus")) for name in names
    )
    result = {"audio_available": wavs_present, "timing_checked_by_this_tool": False, "splits": {}}
    reasons, examples = collections.Counter(), collections.defaultdict(list)
    durations = collections.defaultdict(list)
    labels = collections.Counter()
    grids = 0
    for split, records in read_manifests(preprocess).items():
        good, seconds = 0, 0.0
        for record in records:
            grids += os.path.exists(os.path.join(interim, "textgrids", record["speaker"], record["id"] + ".TextGrid"))
            try:
                phones, frames = check_clip(record, interim, lexicon, valid, rate, hop)
            except (ValueError, KeyError, OSError) as error:
                reason = str(error) if isinstance(error, ValueError) else "missing label or lexicon entry"
                reasons[reason] += 1
                examples[reason].append(record["id"])
                continue
            good += 1
            seconds += record["seconds"]
            labels.update(phones)
            for phone, count in zip(phones, frames):
                durations[phone].append(count * frame_ms)
        result["splits"][split] = {
            "clips": len(records), "valid": good, "rejected": len(records) - good,
            "coverage_percent": round(100 * good / max(len(records), 1), 2),
            "valid_hours": round(seconds / 3600, 3),
        }
    speech = np.array([d for p, v in durations.items() if p not in SILENCE_LABELS for d in v] or [0.0])
    result.update({
        "textgrids": grids,
        "rejection_reasons": dict(reasons),
        "rejected_examples": {k: v[:5] for k, v in examples.items()},
        "phones_not_in_vocabulary": "reported as rejections above",
        "phone_intervals": int(sum(len(v) for v in durations.values())),
        "speech_phone_ms_percentiles_5_25_50_75_95": [round(float(x), 1) for x in np.percentile(speech, [5, 25, 50, 75, 95])],
        "speech_phone_ms_mean": round(float(speech.mean()), 1),
        "zero_duration_speech_phones": int((speech == 0).sum()),
        "speech_phones_under_{}ms".format(SHORT_MS): int((speech < SHORT_MS).sum()),
        "speech_phones_over_{}ms".format(LONG_MS): int((speech > LONG_MS).sum()),
        "silence_counts": {s: labels[s] for s in sorted(SILENCE_LABELS) if s and labels[s]},
        "vocabulary_never_aligned": sorted(valid - set(labels) - {"_"}),
        "mean_ms_per_phone": {p: round(float(np.mean(v)), 1) for p, v in sorted(durations.items())},
    })
    return result


def show(preprocess, ids):
    interim = preprocess["path"]["interim_path"]
    by_id = {r["id"]: r for rows in read_manifests(preprocess).values() for r in rows}
    for uid in ids:
        record = by_id[uid]
        grid = tgt.io.read_textgrid(os.path.join(interim, "textgrids", record["speaker"], uid + ".TextGrid"))
        print("---", uid, "seconds", record["seconds"], "text:", record["text"])
        for tier in ("words", "phones"):
            print(" ", tier, [(round(float(i.start_time), 2), i.text) for i in grid.get_tier_by_name(tier).intervals])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=["english", "telugu"])
    parser.add_argument("--show", nargs="*", default=[], help="clip IDs to print")
    args = parser.parse_args()
    preprocess, _, _ = load_configs(args.language)
    show(preprocess, args.show)
    result = check_language(preprocess, load_symbols(args.language))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    selection = preprocess["selection"]
    enough = (
        result["splits"]["train"]["valid_hours"] >= selection["train_hours"] * 0.95
        and result["splits"]["val"]["valid_hours"] >= selection["val_hours"] * 0.95
    )
    print("Enough valid audio for about {} h + {} h: {}".format(selection["train_hours"], selection["val_hours"], enough))
    if not result["audio_available"]:
        print("No audio present: boundary timing has NOT been checked against audio here.")
    print("Review the rejection reasons and representative alignments before extracting durations.")
    sys.exit(0 if enough else 1)


if __name__ == "__main__":
    main()
