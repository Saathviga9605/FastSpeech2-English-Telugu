"""Synthesize the assignment's test sentences and save audio, variance values and plots.

Sentence text is never edited: the frontend works on a copy, and logs every change.
Results are written from a real checkpoint only; a missing checkpoint is an error.
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch

from ..config import REPO_ROOT, language_dir, load_configs, load_symbols
from ..models import FastSpeech2
from ..models.vocoder import load_vocoder, vocode
from ..text import Vocabulary, get_frontend
from ..training.checkpoint import latest_valid_checkpoint, load_checkpoint
from ..visualization.plots import plot_variance

DEFAULT_SENTENCES = REPO_ROOT / "data" / "test_utterances" / "sentences.json"
RESULT_DIR = {"english": "Results/English", "telugu": "Results/Telugu"}


def read_sentences(path, language):
    with open(path, encoding="utf-8") as f:
        sentences = json.load(f)[language]
    if not sentences:
        raise ValueError("{} has no {} sentences".format(path, language))
    return sentences


def predict(model, phone_ids, speaker, device):
    length = len(phone_ids)
    with torch.no_grad():
        output = model(
            torch.tensor([speaker], device=device),
            torch.tensor([phone_ids], device=device),
            torch.tensor([length], device=device),
            length,
        )
    return {
        "mel": output[1],
        "pitch": output[2][0].cpu().numpy(),
        "energy": output[3][0].cpu().numpy(),
        "duration": output[5][0].cpu().numpy(),
        "mel_len": int(output[9][0]),
    }


def synthesize(language, sentence_path=DEFAULT_SENTENCES, checkpoint_path=None, out_root=".", speaker=0):
    preprocess, model_config, train_config = load_configs(language)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # A checkout without the dataset uses the normalization values saved with the final model.
    assets = language_dir(language) / "model_assets"
    if not os.path.exists(os.path.join(preprocess["path"]["preprocessed_path"], "stats.json")):
        preprocess["path"]["preprocessed_path"] = str(assets)
    if checkpoint_path is None and (language_dir(language) / "checkpoints" / "final.pth.tar").exists() and not os.path.isdir(train_config["path"]["ckpt_path"]):
        checkpoint_path = str(language_dir(language) / "checkpoints" / "final.pth.tar")
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path, device)
    else:
        checkpoint_path, checkpoint = latest_valid_checkpoint(train_config["path"]["ckpt_path"], device)
    if checkpoint is None:
        raise FileNotFoundError("No trained checkpoint for {}".format(language))

    model = FastSpeech2(preprocess, model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    vocoder = load_vocoder(model_config, device)
    vocab = Vocabulary(load_symbols(language))
    frontend = get_frontend(language, preprocess)
    with open(os.path.join(preprocess["path"]["preprocessed_path"], "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)

    cfg = preprocess["preprocessing"]
    frame_ms = cfg["stft"]["hop_length"] / cfg["audio"]["sampling_rate"] * 1000
    root = os.path.join(out_root, RESULT_DIR[language])
    audio_dir = os.path.join(root, "test_generated_audios")
    os.makedirs(audio_dir, exist_ok=True)
    records = []
    for sentence in read_sentences(sentence_path, language):
        name = sentence["id"]
        result = frontend(sentence["text"])
        record = {"id": name, **result.as_dict(), "checkpoint_step": checkpoint["step"]}
        if not result.phones:
            record["status"] = "no audio: the input contains nothing pronounceable"
            records.append(record)
            continue
        output = predict(model, vocab.encode(result.phones), speaker, device)
        wav = vocode(
            output["mel"].transpose(1, 2), vocoder, cfg["audio"]["max_wav_value"],
            [output["mel_len"] * cfg["stft"]["hop_length"]],
        )[0]
        sf.write(os.path.join(audio_dir, name + ".wav"), wav, cfg["audio"]["sampling_rate"], subtype="PCM_16")

        duration_ms = output["duration"] * frame_ms
        pitch_hz = output["pitch"] * stats["pitch"][3] + stats["pitch"][2]
        energy = output["energy"] * stats["energy"][3] + stats["energy"][2]
        phones = result.phones
        title = "{} {}, step {}".format(language.capitalize(), name, checkpoint["step"])
        plots = os.path.join(root, "variance_adaptor_plots")
        plot_variance(phones, duration_ms, "predicted duration (ms)", title, os.path.join(plots, "duration", name + ".png"))
        plot_variance(phones, pitch_hz, "predicted F0 (Hz)", title, os.path.join(plots, "pitch", name + ".png"))
        plot_variance(phones, energy, "predicted energy", title, os.path.join(plots, "energy", name + ".png"))
        record.update({
            "status": "generated",
            "audio_seconds": round(len(wav) / cfg["audio"]["sampling_rate"], 3),
            "duration_ms": np.round(duration_ms, 1).tolist(),
            "pitch_hz": np.round(pitch_hz, 2).tolist(),
            "energy": np.round(energy, 4).tolist(),
        })
        records.append(record)
    report_path = os.path.join(root, "synthesis_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": checkpoint_path, "utterances": records}, f, ensure_ascii=False, indent=2)
    return records
