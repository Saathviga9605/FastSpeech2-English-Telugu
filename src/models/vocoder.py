"""Load the pretrained HiFi-GAN generator bundled with upstream."""
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch

from .hifigan.models import Generator


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def load_vocoder(model_config, device):
    cfg = model_config["vocoder"]
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    with zipfile.ZipFile(cfg["zip_path"]) as archive:
        member = archive.namelist()[0]
        path = os.path.join(cfg["cache_dir"], member)
        if not os.path.exists(path):
            archive.extract(member, cfg["cache_dir"])
    with open(Path(__file__).parent / "hifigan" / "config.json", encoding="utf-8") as f:
        config = AttrDict(json.load(f))
    vocoder = Generator(config)
    vocoder.load_state_dict(torch.load(path, map_location="cpu")["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder.to(device)


def vocode(mels, vocoder, max_wav_value, lengths=None):
    """mels: (batch, n_mels, frames). Returns a list of int16 arrays."""
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)
    wavs = (wavs.cpu().numpy() * max_wav_value).clip(-32768, 32767).astype(np.int16)
    if lengths is None:
        return list(wavs)
    return [wav[: int(length)] for wav, length in zip(wavs, lengths)]
