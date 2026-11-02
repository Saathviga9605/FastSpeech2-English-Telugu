"""Reconstruct real clips through mel and HiFi-GAN: the best quality the vocoder can give this data."""
import argparse
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.features import build_stft, mel_and_energy  # noqa: E402
from src.config import load_configs  # noqa: E402
from src.data.corpus import read_manifests  # noqa: E402
from src.models.vocoder import load_vocoder, vocode  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=["english", "telugu"])
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--out", default="output/vocoder_check")
    args = parser.parse_args()
    preprocess, model_config, _ = load_configs(args.language)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocoder = load_vocoder(model_config, device)
    stft = build_stft(preprocess)
    rate = preprocess["preprocessing"]["audio"]["sampling_rate"]
    os.makedirs(args.out, exist_ok=True)
    for record in read_manifests(preprocess)["val"][: args.count]:
        wav, _ = sf.read(os.path.join(preprocess["path"]["interim_path"], "wav", record["id"] + ".wav"))
        mel, _ = mel_and_energy(wav.astype(np.float32), stft)
        output = vocode(torch.from_numpy(mel).unsqueeze(0).to(device), vocoder, 32768.0)[0]
        n = min(len(output), len(wav))
        mel_out = librosa.feature.melspectrogram(y=output[:n] / 32768.0, sr=rate, n_mels=80)
        mel_in = librosa.feature.melspectrogram(y=wav[:n], sr=rate, n_mels=80)
        distance = np.mean(np.abs(np.log(mel_out + 1e-5) - np.log(mel_in + 1e-5)))
        sf.write(os.path.join(args.out, record["id"] + "_original.wav"), wav, rate)
        sf.write(os.path.join(args.out, record["id"] + "_reconstructed.wav"), output, rate)
        print(record["id"], "seconds", round(n / rate, 2), "log-mel L1", round(float(distance), 3))


if __name__ == "__main__":
    main()
