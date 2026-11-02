"""Check whether a validation gap is real: score one checkpoint in eval mode on both splits, against simple baselines.

Baselines use training targets only. "Zero" predicts the normalized mean; "per-phone" predicts each
phone's training mean, which is the least a text-driven predictor should beat.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from src.config import load_configs, load_symbols  # noqa: E402
from src.data.dataset import Dataset  # noqa: E402
from src.models import FastSpeech2, FastSpeech2Loss  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.train import LOSS_NAMES, evaluate  # noqa: E402

KINDS = ("pitch", "energy", "duration")


def load_target(path, kind, item):
    values = np.load(os.path.join(path, kind, item["id"] + ".npy")).astype(float)
    return np.log(values + 1) if kind == "duration" else values


def phone_means(path, items):
    totals = {kind: defaultdict(list) for kind in KINDS}
    for item in items:
        for kind in KINDS:
            for phone, value in zip(item["phones"], load_target(path, kind, item)):
                totals[kind][phone].append(value)
    return {kind: {p: float(np.mean(v)) for p, v in totals[kind].items()} for kind in KINDS}


def baselines(path, items, means, train_duration_mean):
    report = {}
    for kind in KINDS:
        targets, per_phone, clip_means = [], [], []
        for item in items:
            values = load_target(path, kind, item)
            targets.append(values)
            per_phone.append([means[kind].get(p, 0.0) for p in item["phones"]])
            clip_means.append(np.full(len(values), values.mean()))
        y = np.concatenate(targets)
        constant = train_duration_mean if kind == "duration" else 0.0
        report[kind] = {
            "constant_mse": round(float(np.mean((y - constant) ** 2)), 4),
            "per_phone_mse": round(float(np.mean((y - np.concatenate(per_phone)) ** 2)), 4),
            "within_clip_variance": round(float(np.mean((y - np.concatenate(clip_means)) ** 2)), 4),
            "target_variance": round(float(y.var()), 4),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=["english", "telugu"])
    parser.add_argument("checkpoint")
    parser.add_argument("--clips", type=int, default=300)
    parser.add_argument("--out")
    args = parser.parse_args()

    preprocess, model_config, train_config = load_configs(args.language)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = FastSpeech2(preprocess, model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    loss_fn = FastSpeech2Loss(preprocess, model_config).to(device)
    symbols = load_symbols(args.language)

    report = {"language": args.language, "step": checkpoint["step"], "clips_per_split": args.clips, "splits": {}}
    path = preprocess["path"]["preprocessed_path"]
    items = {}
    for split in ("train", "val"):
        dataset = Dataset(split, preprocess, train_config, symbols)
        items[split] = dataset.items
        rng = np.random.default_rng(1234)
        chosen = rng.permutation(len(dataset))[:args.clips].tolist()
        loader = DataLoader(Subset(dataset, chosen), batch_size=dataset.batch_size, collate_fn=dataset.collate_fn)
        report["splits"][split] = {"model_eval_mode": dict(zip(LOSS_NAMES, evaluate(model, loss_fn, loader, device)))}
    means = phone_means(path, items["train"])
    train_duration = np.concatenate([load_target(path, "duration", i) for i in items["train"]]).mean()
    for split in ("train", "val"):
        report["splits"][split]["baselines"] = baselines(path, items[split], means, float(train_duration))

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
