"""Configuration loading. English and Telugu live in separate config directories."""
import json
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _expand(value):
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def set_default_roots():
    """FS2_DATA_ROOT holds data; FS2_EXP_ROOT holds checkpoints and logs (Drive on Colab)."""
    os.environ.setdefault("FS2_DATA_ROOT", str(REPO_ROOT / "data"))
    os.environ.setdefault("FS2_EXP_ROOT", str(REPO_ROOT / "output"))
    os.environ.setdefault("FS2_REPO_ROOT", str(REPO_ROOT))
    os.environ.setdefault("FS2_RAW_ENGLISH", str(REPO_ROOT / "data" / "raw" / "svarah"))
    os.environ.setdefault("FS2_RAW_TELUGU", str(REPO_ROOT / "data" / "raw" / "indictts_telugu"))
    os.environ.setdefault("FS2_CKPT_ROOT", str(REPO_ROOT / "output" / "checkpoints"))
    os.environ.setdefault("FS2_OUTPUT_ROOT", str(REPO_ROOT / "output" / "outputs"))


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return _expand(yaml.safe_load(f))


def language_dir(language):
    return REPO_ROOT / language.capitalize()


def load_symbols(language):
    with open(language_dir(language) / "symbols.json", encoding="utf-8") as f:
        return json.load(f)


def load_configs(language):
    set_default_roots()
    config = load_yaml(language_dir(language) / "config.yaml")
    preprocess, model, train = config["preprocess"], config["model"], config["train"]
    if preprocess["language"] != language:
        raise ValueError("Config language mismatch for {}".format(language))
    model["n_symbols"] = len(load_symbols(language))
    return preprocess, model, train
