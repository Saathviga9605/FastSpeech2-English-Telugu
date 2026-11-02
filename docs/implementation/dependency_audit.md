# Dependency Audit (Initial)

Nothing was installed during this audit. Compatibility has not been tested, so the assessments below are expectations to verify.

## Upstream pins and the current environment

| Package | Upstream pin | Installed now | Used for |
|---|---|---|---|
| torch | 1.7.0 | 2.14.0+cpu | Model and training |
| numpy | 1.19.0 | 2.2.2 | Arrays |
| scipy | 1.5.0 | 1.18.1 | Signal processing and I/O |
| librosa | 0.7.2 | missing | Audio features |
| numba | 0.48 | not checked | Not imported directly; listed by upstream |
| soundfile | 0.10.3.post1 | 0.14.0 | Not imported directly; listed by upstream |
| matplotlib | 3.2.2 | 3.11.2 | Plots |
| PyYAML | 5.4.1 | 6.0.3 | Configs |
| tqdm | 4.46.1 | 4.70.1 | Progress bars |
| tensorboard | 2.2.2 | not checked | Training logs; upstream code does not import TensorFlow |
| pyworld | 0.2.10 | missing | F0 extraction |
| tgt | 1.4.4 | missing | TextGrid parsing |
| g2p-en | 2.1.0 | missing | English phonemization |
| inflect | 4.1.0 | missing | English number text |
| unidecode | 1.1.1 | missing | English text cleaning |
| pypinyin | 0.39.0 | not checked | Mandarin only |
| scikit-learn | 0.23.2 | not checked | Pitch and energy normalization |

Also installed but not in upstream's list: torchaudio 2.11.0+cpu and huggingface_hub 0.36.2.

## Risks to resolve

1. **Pins and interpreter:** Upstream's pins predate Python 3.13 by several years, so installing them on the current interpreter is expected to fail.
2. **Newer libraries:** If the pins are relaxed, upstream code will need testing against newer APIs, especially librosa, numpy and torch.
3. **Torch mismatch:** The installed torch (2.14.0) and torchaudio (2.11.0) versions differ. Check this if torchaudio is ever needed.
4. **Upstream checkpoints:** Loading the bundled HiFi-GAN generators in the chosen torch version needs testing.

## Next step

Create an isolated environment on the chosen compute platform, select a Python version, and test imports plus one model forward pass. Then replace `requirements.txt` with the verified pins. Do not change the system Python.
