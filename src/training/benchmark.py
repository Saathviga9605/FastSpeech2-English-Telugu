"""Environment report and throughput benchmark on SYNTHETIC tensors. It trains nothing useful."""
import argparse
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time

import torch

from ..config import load_configs
from ..models import FastSpeech2, FastSpeech2Loss, ScheduledOptim
from ..models.vocoder import load_vocoder
from .checkpoint import latest_valid_checkpoint, save_checkpoint

PACKAGES = ["torch", "numpy", "scipy", "librosa", "pyworld", "tgt", "g2p-en", "inflect", "pyarrow", "soundfile", "scikit-learn"]


def environment():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_build": torch.version.cuda,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        report["gpu"] = properties.name
        report["vram_gb"] = round(properties.total_memory / 2**30, 2)
    if shutil.which("nvidia-smi"):
        report["nvidia_driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip()
    return report


def synthetic_batch(batch_size, n_phones, n_frames, n_symbols, device):
    per_phone = n_frames // n_phones
    frames = per_phone * n_phones
    return (
        None,
        None,
        torch.zeros(batch_size, dtype=torch.long, device=device),
        torch.randint(1, n_symbols, (batch_size, n_phones), device=device),
        torch.full((batch_size,), n_phones, device=device),
        n_phones,
        torch.randn(batch_size, frames, 80, device=device),
        torch.full((batch_size,), frames, device=device),
        frames,
        torch.randn(batch_size, n_phones, device=device),
        torch.randn(batch_size, n_phones, device=device),
        torch.full((batch_size, n_phones), per_phone, device=device),
    )


def build(language, stats_dir, device):
    preprocess, model_config, train_config = load_configs(language)
    preprocess["path"]["preprocessed_path"] = stats_dir
    model = FastSpeech2(preprocess, model_config).to(device)
    loss = FastSpeech2Loss(preprocess, model_config).to(device)
    optimizer = ScheduledOptim(model, train_config, model_config, 0)
    return model, loss, optimizer, (preprocess, model_config, train_config)


def measure(language, stats_dir, device, batch_size, n_phones, n_frames, steps):
    model, loss_fn, optimizer, configs = build(language, stats_dir, device)
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(steps):
        batch = synthetic_batch(batch_size, n_phones, n_frames, configs[1]["n_symbols"], device)
        start = time.perf_counter()
        losses = loss_fn(batch, model(*batch[2:]))
        losses[0].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step_and_update_lr()
        optimizer.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    warm = timings[min(3, len(timings) - 1):]
    result = {
        "batch_size": batch_size,
        "n_phones": n_phones,
        "n_frames": n_frames,
        "steps_per_second": round(len(warm) / sum(warm), 3),
        "loss_finite": bool(torch.isfinite(losses[0])),
    }
    if device.type == "cuda":
        result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return result


def checkpoint_roundtrip(language, stats_dir, device):
    model, _, optimizer, configs = build(language, stats_dir, device)
    with tempfile.TemporaryDirectory() as folder:
        save_checkpoint(folder, 7, 1, model, optimizer, configs)
        path, loaded = latest_valid_checkpoint(folder, device)
        size = os.path.getsize(path)
        same = all(torch.equal(v, loaded["model"][k]) for k, v in model.state_dict().items())
    return {"size_mb": round(size / 2**20, 1), "weights_identical": same, "step": loaded["step"]}


def run(out_path, steps, language="english", batch_sizes=(16, 24, 32)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"note": "Synthetic tensors only; not a training result.", "environment": environment()}
    with tempfile.TemporaryDirectory() as stats_dir:
        with open(os.path.join(stats_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump({"pitch": [-3.0, 3.0, 0.0, 1.0], "energy": [-3.0, 3.0, 0.0, 1.0]}, f)

        model, _, _, configs = build(language, stats_dir, device)
        report["parameters"] = sum(p.numel() for p in model.parameters())
        del model
        report["checkpoint"] = checkpoint_roundtrip(language, stats_dir, device)

        report["throughput"] = []
        for batch_size in batch_sizes:
            # 1000 frames is the longest clip the selection allows, so this is the memory worst case.
            for phones, frames in ((70, 600), (100, 1000)):
                try:
                    result = measure(language, stats_dir, device, batch_size, phones, frames, steps)
                except torch.cuda.OutOfMemoryError:
                    result = {"batch_size": batch_size, "n_frames": frames, "error": "out of GPU memory"}
                    torch.cuda.empty_cache()
                report["throughput"].append(result)
                print(result, flush=True)

        vocoder = load_vocoder(configs[1], device)
        with torch.no_grad():
            start = time.perf_counter()
            wav = vocoder(torch.randn(1, 80, 430, device=device))
            report["vocoder"] = {"output_shape": list(wav.shape), "seconds": round(time.perf_counter() - start, 3)}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Wrote", out_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="output/benchmark.json")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(environment(), indent=2))
    run(args.out, args.steps)


if __name__ == "__main__":
    main()
