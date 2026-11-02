"""Median F0 across the row range of each speaker, with listening samples.

F0 can reveal a change of speaker but cannot prove there is only one, so listening is still needed.
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.features import extract_f0  # noqa: E402
from src.config import load_configs  # noqa: E402
from src.data.corpus import iter_rows, read_inventory  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language")
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--per_segment", type=int, default=12)
    parser.add_argument("--out", default="output/speaker_check")
    args = parser.parse_args()
    preprocess, _, _ = load_configs(args.language)
    inventory = read_inventory(os.path.join(preprocess["path"]["interim_path"], "inventory.jsonl"))
    wanted = {}
    for speaker in sorted({r["speaker"] for r in inventory}):
        rows = [r for r in inventory if r["speaker"] == speaker and 2 <= r["seconds"] <= 15]
        for segment, part in enumerate(np.array_split(np.arange(len(rows)), args.segments)):
            for position in part[np.linspace(0, len(part) - 1, args.per_segment).astype(int)]:
                wanted[rows[position]["row"]] = (speaker, segment)
    medians = {}
    os.makedirs(args.out, exist_ok=True)
    saved = set()
    for row in iter_rows(preprocess):
        if row["_row"] not in wanted:
            continue
        speaker, segment = wanted[row["_row"]]
        wav, _ = librosa.load(io.BytesIO(row["_audio"]["bytes"]), sr=22050, mono=True)
        f0 = extract_f0(wav, 22050, 256)
        voiced = f0[f0 > 0]
        if len(voiced):
            medians.setdefault(speaker, {}).setdefault(segment, []).append(float(np.median(voiced)))
        if (speaker, segment) not in saved:
            saved.add((speaker, segment))
            sf.write(os.path.join(args.out, "{}_segment{}_row{}.wav".format(speaker, segment, row["_row"])), wav, 22050)
    report = {
        speaker: {str(s): [round(float(np.median(v)), 1), round(float(np.min(v)), 1), round(float(np.max(v)), 1)] for s, v in sorted(segments.items())}
        for speaker, segments in medians.items()
    }
    print("speaker: segment: [median of clip median F0, min, max] in Hz")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
