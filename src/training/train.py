"""Train one language model. Rerunning the same command resumes from the newest checkpoint."""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import load_configs, load_symbols
from ..data.dataset import Dataset
from ..models import FastSpeech2, FastSpeech2Loss, ScheduledOptim
from ..utils import to_device
from .checkpoint import latest_valid_checkpoint, restore_rng, save_checkpoint

LOSS_NAMES = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]
GROUP_SIZE = 4  # as upstream: load 4 batches at once so length sorting works


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_csv(path, step, values, seconds=None, lr=None):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["step", *LOSS_NAMES, "seconds_since_log", "learning_rate"])
        writer.writerow([step, *["{:.6f}".format(v) for v in values], seconds if seconds is not None else "", lr if lr is not None else ""])
        f.flush()
        os.fsync(f.fileno())


def trim_csv(path, step):
    """After resuming, drop rows newer than the checkpoint so curves show one real history."""
    if not os.path.exists(path):
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    kept = rows[:1] + [r for r in rows[1:] if int(r[0]) <= step]
    if len(kept) != len(rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(kept)


def build_model(preprocess_config, model_config, device):
    return FastSpeech2(preprocess_config, model_config).to(device)


@torch.no_grad()
def evaluate(model, loss_fn, loader, device):
    was_training = model.training
    model.eval()
    totals, count = [0.0] * len(LOSS_NAMES), 0
    for groups in loader:
        for batch in groups:
            batch = to_device(batch, device)
            losses = loss_fn(batch, model(*batch[2:]))
            size = len(batch[0])
            totals = [t + l.item() * size for t, l in zip(totals, losses)]
            count += size
    model.train(was_training)
    return [t / count for t in totals]


def train(language, max_steps=None):
    preprocess_config, model_config, train_config = load_configs(language)
    configs = (preprocess_config, model_config, train_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(train_config["seed"])

    symbols = load_symbols(language)
    train_set = Dataset("train", preprocess_config, train_config, symbols, sort=True, drop_last=True)
    val_set = Dataset("val", preprocess_config, train_config, symbols)
    batch_size = train_config["optimizer"]["batch_size"]
    if batch_size * GROUP_SIZE >= len(train_set):
        raise ValueError("Training set of {} clips is too small".format(len(train_set)))

    model = build_model(preprocess_config, model_config, device)
    loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)

    ckpt_dir = train_config["path"]["ckpt_path"]
    log_dir = train_config["path"]["log_path"]
    os.makedirs(log_dir, exist_ok=True)
    train_csv = os.path.join(log_dir, "train_loss.csv")
    val_csv = os.path.join(log_dir, "val_loss.csv")

    path, checkpoint = latest_valid_checkpoint(ckpt_dir, device)
    step, epoch = 0, 1
    optimizer = ScheduledOptim(model, train_config, model_config, 0)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, epoch = checkpoint["step"], checkpoint["epoch"]
        optimizer.current_step = step
        restore_rng(checkpoint["rng"])
        trim_csv(train_csv, step)
        trim_csv(val_csv, step)
        print("Resumed from {} at step {}".format(path, step), flush=True)
    else:
        print("Starting a new run", flush=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size * GROUP_SIZE,
        shuffle=True,
        collate_fn=train_set.collate_fn,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=val_set.collate_fn)

    cfg = train_config["step"]
    optim_cfg = train_config["optimizer"]
    total_step = cfg["total_step"] if max_steps is None else min(cfg["total_step"], step + max_steps)
    model.train()
    last_log = time.time()
    while step < total_step:
        for groups in train_loader:
            for batch in groups:
                if step >= total_step:
                    break
                batch = to_device(batch, device)
                losses = loss_fn(batch, model(*batch[2:]))
                (losses[0] / optim_cfg["grad_acc_step"]).backward()
                step += 1
                if step % optim_cfg["grad_acc_step"] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg["grad_clip_thresh"])
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                values = [l.item() for l in losses]
                if not np.isfinite(values[0]):
                    raise RuntimeError("Non-finite loss at step {}: {}".format(step, values))
                if step % cfg["log_step"] == 0:
                    now = time.time()
                    append_csv(train_csv, step, values, round(now - last_log, 2), optimizer._optimizer.param_groups[0]["lr"])
                    print(
                        "step {} total {:.4f} mel {:.4f} post {:.4f} pitch {:.4f} energy {:.4f} dur {:.4f} ({:.2f} steps/s)".format(
                            step, *values, cfg["log_step"] / max(now - last_log, 1e-6)
                        ),
                        flush=True,
                    )
                    last_log = now
                if step % cfg["val_step"] == 0:
                    val = evaluate(model, loss_fn, val_loader, device)
                    append_csv(val_csv, step, val)
                    print("val step {} total {:.4f} mel {:.4f}".format(step, val[0], val[1]), flush=True)
                if step % cfg["save_step"] == 0:
                    print("saved", save_checkpoint(ckpt_dir, step, epoch, model, optimizer, configs, cfg["keep_last"]), flush=True)
            else:
                continue
            break
        epoch += 1

    # Always keep the final state, whether or not it falls on a save interval.
    print("saved", save_checkpoint(ckpt_dir, step, epoch, model, optimizer, configs, cfg["keep_last"]), flush=True)
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=["english", "telugu"])
    parser.add_argument("--max_steps", type=int, default=None, help="stop after this many more steps")
    args = parser.parse_args()
    train(args.language, args.max_steps)


if __name__ == "__main__":
    main()
