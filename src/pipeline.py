"""Entry points behind English/ and Telugu/ preprocess.py, train.py and inference.py."""
import argparse
import os
import subprocess
import sys

from .config import REPO_ROOT, load_configs, load_symbols

STEPS = ["inspect", "select", "export", "prepare", "align", "features"]


def preprocess(language):
    parser = argparse.ArgumentParser(description="Prepare {} data for training".format(language))
    parser.add_argument("--start", choices=STEPS, default=STEPS[0], help="first step to run")
    parser.add_argument("--stop", choices=STEPS, default=STEPS[-1], help="last step to run")
    args = parser.parse_args()
    config, _, _ = load_configs(language)
    inventory = os.path.join(config["path"]["interim_path"], "inventory.jsonl")

    def run_inspect():
        subprocess.run([sys.executable, "-m", "src.data.inspection", language], cwd=REPO_ROOT, check=True)

    def run_select():
        from .data.corpus import read_inventory, select, write_manifests

        train, val = select(read_inventory(inventory), config["selection"])
        write_manifests(config, train, val)
        print("train hours", sum(r["seconds"] for r in train) / 3600, "val hours", sum(r["seconds"] for r in val) / 3600)

    def run_export():
        from .data.corpus import export_audio

        print("exported", export_audio(config))

    def run_prepare():
        from .alignment.prepare import prepare

        prepare(config)

    def run_align():
        subprocess.run(["bash", "scripts/run_mfa.sh", language], cwd=REPO_ROOT, check=True)

    def run_features():
        import json

        from .audio.features import process_corpus

        print(json.dumps(process_corpus(config, load_symbols(language)), indent=2))

    actions = dict(zip(STEPS, [run_inspect, run_select, run_export, run_prepare, run_align, run_features]))
    for step in STEPS[STEPS.index(args.start) : STEPS.index(args.stop) + 1]:
        print("==", step, flush=True)
        actions[step]()


def train(language):
    from .training.train import train as run

    parser = argparse.ArgumentParser(description="Train or resume the {} model".format(language))
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()
    run(language, args.max_steps)


def inference(language):
    from .inference.synthesize import DEFAULT_SENTENCES, synthesize

    parser = argparse.ArgumentParser(description="Synthesize the {} test sentences".format(language))
    parser.add_argument("--sentences", default=str(DEFAULT_SENTENCES))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out_root", default=".")
    parser.add_argument("--speaker", type=int, default=0)
    args = parser.parse_args()
    try:
        records = synthesize(language, args.sentences, args.checkpoint, args.out_root, args.speaker)
    except FileNotFoundError as error:
        sys.exit(str(error))
    for record in records:
        print(record["id"], record["status"], record["issues"])
