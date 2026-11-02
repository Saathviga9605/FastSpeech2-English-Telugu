"""Copy real training outputs into the folder layout the assignment requires, and report gaps."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.config import load_configs  # noqa: E402
from src.inference.synthesize import DEFAULT_SENTENCES, RESULT_DIR  # noqa: E402
from src.training.checkpoint import export_for_inference, latest_valid_checkpoint  # noqa: E402
from src.visualization.plots import plot_loss_file  # noqa: E402


def collect(language, source_root):
    preprocess, _, train_config = load_configs(language)
    folder = ROOT / language.capitalize()
    results = ROOT / RESULT_DIR[language]
    log = train_config["path"]["log_path"]

    path, checkpoint = latest_valid_checkpoint(train_config["path"]["ckpt_path"], torch.device("cpu"))
    if checkpoint is None:
        print(language, "has no trained checkpoint; nothing exported")
    else:
        export_for_inference(checkpoint, str(folder / "checkpoints" / "final.pth.tar"))
        print(language, "exported step", checkpoint["step"], "from", path)
        assets = folder / "model_assets"
        assets.mkdir(exist_ok=True)
        for name in ("stats.json", "speakers.json"):
            shutil.copy2(os.path.join(preprocess["path"]["preprocessed_path"], name), assets / name)

    for kind in ("training", "validation"):
        csv_name = {"training": "train_loss.csv", "validation": "val_loss.csv"}[kind]
        csv_path = os.path.join(log, csv_name)
        if os.path.exists(csv_path):
            plot_loss_file(csv_path, str(results / "{}_loss.png".format(kind)), "{} {} loss".format(language.capitalize(), kind))
            shutil.copy2(csv_path, results / csv_name)

    generated = Path(source_root) / RESULT_DIR[language]
    if generated.resolve() != results.resolve() and generated.exists():
        shutil.copytree(generated, results, dirs_exist_ok=True)

    with open(DEFAULT_SENTENCES, encoding="utf-8") as f:
        ids = [s["id"] for s in json.load(f)[language]]
    required = [folder / "checkpoints" / "final.pth.tar", results / "training_loss.png", results / "validation_loss.png"]
    for uid in ids:
        required.append(results / "test_generated_audios" / (uid + ".wav"))
        for kind in ("duration", "pitch", "energy"):
            required.append(results / "variance_adaptor_plots" / kind / (uid + ".png"))
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    print(language, "missing deliverables:", missing or "none")
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", default=os.environ.get("FS2_OUTPUT_ROOT", str(ROOT)))
    args = parser.parse_args()
    missing = [item for language in ("english", "telugu") for item in collect(language, args.source_root)]
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
