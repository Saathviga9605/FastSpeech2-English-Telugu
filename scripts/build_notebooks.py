"""Generate the Colab notebooks, keeping shared setup in one place."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETUP = f'''# Run this cell at the start of every Colab session. Runtime type must be GPU.
import os, subprocess, sys
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/MyDrive/FastSpeech2-IIITH'
REPO = '/content/FastSpeech2-IIITH'
REPO_URL = ''  # optional: GitHub URL once the repository is public
os.makedirs(DRIVE, exist_ok=True)
if not os.path.isdir(REPO):
    if REPO_URL:
        subprocess.run(['git', 'clone', REPO_URL, REPO], check=True)
    else:
        subprocess.run(['unzip', '-q', f'{{DRIVE}}/FastSpeech2-IIITH-code.zip', '-d', REPO], check=True)
os.chdir(REPO)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
sys.path.insert(0, REPO)
from src.data.storage import configure_environment
# HF_HOME is deliberately left alone, so Hugging Face's token file stays off Drive.
configure_environment(DRIVE)
os.environ.update(FS2_REPO_ROOT=REPO, PYTHONUNBUFFERED='1')
!nvidia-smi
import torch
print('PyTorch', torch.__version__, 'CUDA build', torch.version.cuda, 'GPU available', torch.cuda.is_available())
assert torch.cuda.is_available(), 'Choose Runtime > Change runtime type > GPU'
'''

HF_LOGIN = '''# Add a Read token first: key icon in the left sidebar > Add new secret > name HF_TOKEN > allow this notebook.
from src.data.hf_auth import authenticate
authenticate()
'''

HF_LOGIN_OPTIONAL = '''# IndicTTS Telugu is public, so a login is optional here. It only raises rate limits.
from src.data.hf_auth import authenticate
try:
    authenticate()
except RuntimeError as error:
    print('Continuing without a login:', error)
'''


def language_cells(language):
    pretty = language.capitalize()
    cells = [
        ("markdown", f"# {pretty} FastSpeech 2: data, alignment, training\n\nRun the setup cell first in each session. Cells can be rerun safely; complete downloads on Drive are reused."),
        ("code", SETUP),
        ("code", HF_LOGIN if language == "english" else HF_LOGIN_OPTIONAL),
        ("markdown", f"## 1. Verify access and inspect\n\nRuns a real load, measures every clip and checks any existing export. It saves `experiments/{language}/dataset_inspection.json` on Drive. Never commit that file, because it holds transcript samples."),
        ("code", f"!python {pretty}/preprocess.py --stop inspect"),
        ("markdown", "## 2. Restore prepared data and review alignment\n\nUse this if a `prepared.tar` is on Drive under `processed/" + language + "/`. **Stop and review before going on:** read the coverage, rejection reasons, enough-audio line and printed report. 100% coverage is not required, but a high or systematic rejection rate needs investigating. An archive has TextGrids but no WAV files, so this cell has not checked timing against audio. Timing evidence comes only from `scripts/verify_alignment_timing.py` and spectrogram inspection where the audio exists."),
        ("code", f"!python scripts/drive_sync.py restore {language}\n!python scripts/check_alignments.py {language}; echo checker exit $?\n!cat \"$FS2_DATA_ROOT/processed/{language}/report.json\""),
        ("markdown", "## 3. Select, align and extract features\n\nAlignment uses the CPU. Review the inspection summary first."),
        ("code", f"!python {pretty}/preprocess.py --start select --stop align\n!python scripts/check_alignments.py {language}; echo checker exit $?\n!python scripts/verify_alignment_timing.py {language} --clips 200"),
        ("markdown", "**Stop and review the alignment results above.** Run the next cell only if the rejection rate and representative alignments look acceptable."),
        ("code", f"!python {pretty}/preprocess.py --start features\n!python scripts/drive_sync.py save {language}"),
        ("markdown", "## 4. Train\n\nAfter a disconnect, rerun setup, restore, then this cell. It resumes from the newest checkpoint on Drive."),
        ("code", f"!python {pretty}/train.py 2>&1 | tee -a \"$FS2_EXP_ROOT/{language}_train_console.log\""),
    ]
    return cells


NOTEBOOKS = {
    "01_setup_and_benchmark.ipynb": [
        ("markdown", "# Setup and GPU benchmark\n\nUses synthetic tensors only, so its output is not a training result. Share `benchmark.json` from Drive when finished."),
        ("code", SETUP),
        ("markdown", "## Hugging Face access\n\nConfirms the `HF_TOKEN` secret works. The token is never printed or saved to Drive."),
        ("code", HF_LOGIN),
        ("code", "!python -m pytest -q tests"),
        ("code", "!python -m src.training.benchmark --out \"$FS2_EXP_ROOT/benchmark.json\" --steps 30\n!cat \"$FS2_EXP_ROOT/benchmark.json\""),
    ],
    "02_english.ipynb": language_cells("english"),
    "03_telugu.ipynb": language_cells("telugu"),
    "04_inference.ipynb": [
        ("markdown", "# Generate test sentences and collect deliverables\n\nNeeds real trained checkpoints on Drive. The final cell lists anything still missing."),
        ("code", SETUP),
        ("code", "!python scripts/drive_sync.py restore english\n!python scripts/drive_sync.py restore telugu"),
        ("code", "!python English/inference.py --out_root \"$FS2_OUTPUT_ROOT\"\n!python Telugu/inference.py --out_root \"$FS2_OUTPUT_ROOT\""),
        ("code", "!python scripts/collect_deliverables.py\n!cp -r English/checkpoints English/model_assets \"$FS2_OUTPUT_ROOT/English/\"\n!cp -r Telugu/checkpoints Telugu/model_assets \"$FS2_OUTPUT_ROOT/Telugu/\"\n!cp -r Results \"$FS2_OUTPUT_ROOT/\""),
    ],
}


def notebook(cells):
    out = []
    for index, (kind, text) in enumerate(cells):
        cell = {"id": "cell-{}".format(index), "cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


for name, cells in NOTEBOOKS.items():
    path = ROOT / "notebooks" / name
    path.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", path)
