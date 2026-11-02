"""Atomic, resumable checkpoints that survive a Colab disconnect mid-save."""
import json
import os
import random
import re
import tempfile

import numpy as np
import torch

_NAME = re.compile(r"^(\d+)\.pth\.tar$")


def list_checkpoints(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return []
    found = []
    for name in os.listdir(ckpt_dir):
        match = _NAME.match(name)
        if match:
            found.append((int(match.group(1)), os.path.join(ckpt_dir, name)))
    return sorted(found)


def rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(ckpt_dir, step, epoch, model, optimizer, configs, keep_last=3):
    """Write to a temp file first, so an interruption cannot leave a corrupt checkpoint."""
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": rng_state(),
        "configs": json.dumps(configs),
    }
    path = os.path.join(ckpt_dir, "{}.pth.tar".format(step))
    handle, tmp = tempfile.mkstemp(dir=ckpt_dir, suffix=".tmp")
    os.close(handle)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    for _, old in list_checkpoints(ckpt_dir)[:-keep_last]:
        os.remove(old)
    return path


def load_checkpoint(path, device):
    # Our own files; they hold Python RNG state, so full unpickling is required.
    return torch.load(path, map_location=device, weights_only=False)


def latest_valid_checkpoint(ckpt_dir, device):
    """Newest readable checkpoint, skipping files damaged by a storage failure."""
    for _, path in reversed(list_checkpoints(ckpt_dir)):
        try:
            return path, load_checkpoint(path, device)
        except Exception as error:  # a damaged file must not block resuming from an older one
            print("Skipping unreadable checkpoint {}: {}".format(path, error))
    return None, None


def export_for_inference(checkpoint, out_path):
    """Model weights only (about 134 MB rather than 399 MB) for the deliverable."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {"step": checkpoint["step"], "model": checkpoint["model"], "configs": checkpoint["configs"]},
        out_path,
    )
