"""Loss curves from real training logs and per-utterance variance plots."""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import load_configs  # noqa: E402

LOSSES = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_losses(train_csv, val_csv, out_path, title):
    train = read_csv(train_csv)
    if not train:
        raise ValueError("No logged training losses in {}".format(train_csv))
    val = read_csv(val_csv) if os.path.exists(val_csv) else []
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for axis, name in zip(axes.flat, LOSSES):
        axis.plot([int(r["step"]) for r in train], [float(r[name]) for r in train], label="training", linewidth=1)
        if val:
            axis.plot([int(r["step"]) for r in val], [float(r[name]) for r in val], label="validation", linewidth=1.5)
        axis.set_title(name)
        axis.set_xlabel("step")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle(title)
    figure.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def plot_loss_file(csv_path, out_path, title):
    rows = read_csv(csv_path)
    if not rows:
        raise ValueError("No logged losses in {}".format(csv_path))
    steps = [int(r["step"]) for r in rows]
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for axis, name in zip(axes.flat, LOSSES):
        axis.plot(steps, [float(r[name]) for r in rows], linewidth=1)
        axis.set_title(name)
        axis.set_xlabel("step")
        axis.grid(alpha=0.3)
    figure.suptitle(title)
    figure.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def plot_variance(phones, values, ylabel, title, out_path):
    """Bar plot of one variance value per input phone."""
    figure, axis = plt.subplots(figsize=(max(8, len(phones) * 0.28), 3.6))
    axis.bar(range(len(phones)), values, color="#3b6fb6")
    axis.set_xticks(range(len(phones)))
    axis.set_xticklabels(phones, rotation=90, fontsize=8)
    axis.set_xlabel("phone")
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=10)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=["english", "telugu"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    _, _, train_config = load_configs(args.language)
    log = train_config["path"]["log_path"]
    plot_losses(
        os.path.join(log, "train_loss.csv"), os.path.join(log, "val_loss.csv"), args.out,
        "{} FastSpeech 2 loss".format(args.language.capitalize()),
    )
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
