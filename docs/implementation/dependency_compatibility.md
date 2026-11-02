# Dependency Compatibility

Audit date: 2026-10-31. **Nothing was installed.**

**Tested on synthetic data, on CPU:** upstream's model (forward, loss, backward and inference) and universal HiFi-GAN loading and synthesis work on PyTorch 2.14.0 and NumPy 2.2.2. `unidecode` and `inflect` were stubbed. Everything else here is untested, including librosa, pyworld, tgt, g2p-en, MFA and any CUDA build.

## Recommendation

**Port to a modern stack in an isolated Linux environment rather than reproducing the 2020 pins:** Python 3.10 or 3.11, a current PyTorch CUDA build matching the chosen GPU, and librosa 0.10 or later. Apply the small code changes below in project code, then verify with the tests listed at the end. Keep MFA in its own environment.

Choose exact versions on the GPU platform in the next phase and only then fill in `requirements.txt`.

## Why not use the upstream pins

- `torch==1.7.0` publishes wheels only for Python 3.6, 3.7 and 3.8 (PyPI), so a 3.8 or older environment would be required. Those versions are past end of life.
- CUDA builds for 1.7.0 could not be confirmed from the PyTorch site in this audit. Older CUDA builds may not support newer GPUs, so this must be checked against the GPU if the legacy route is ever taken.
- The current local Python, 3.13.14, cannot use these pins.

Use the legacy environment only as a diagnostic if the port produces suspicious results.

## Package review

| Package | Upstream pin | Used by | Assessment |
|---|---|---|---|
| torch | 1.7.0 | Everything | **Port.** Newest listed stable is 2.14.1, with CUDA 12.6, 13.0 and 13.2 builds |
| librosa | 0.7.2 | Audio, preprocessing | **Code changes needed**, see below |
| numba | 0.48 | Pulled in for librosa | Do not pin separately; let librosa resolve it |
| numpy | 1.19.0 | Everything | Port; test with the selected version |
| scipy | 1.5.0 | Preprocessing, tools | Port; low risk |
| scikit-learn | 0.23.2 | `StandardScaler` only | Port; low risk |
| pyworld | 0.2.10 | F0 extraction | **Verify a wheel or build on the platform.** A newer version may change nothing, but compare F0 on a sample |
| tgt | 1.4.4 | TextGrid parsing | Test against MFA's current TextGrid output |
| g2p-en | 2.1.0 | English inference | Needs NLTK data at runtime; test it. May need a fixed version of NLTK data |
| inflect, unidecode | 4.1.0, 1.1.1 | English cleaners | Port; low risk |
| pypinyin | 0.39.0 | Imported unconditionally in `synthesize.py:10` | Mandarin only. Remove the import in project code |
| soundfile | 0.10.3.post1 | Not imported directly | Audio backend for librosa; needs libsndfile (bundled in wheels on most platforms) |
| tensorboard | 2.2.2 | Logging | Port. **TensorFlow is not imported and not required** |
| matplotlib, PyYAML, tqdm | 3.2.2, 5.4.1, 4.46.1 | Plots, configs, progress | Port; low risk |

## Code changes expected for a modern stack

| Location | Issue | Status |
|---|---|---|
| `preprocessor/ljspeech.py:28` and matching dataset scripts | `librosa.load(path, sampling_rate)` passes the rate positionally | Expected to fail on librosa 0.10+, where it became keyword-only |
| `audio/stft.py:145` | `librosa_mel_fn(...)` positional | Expected to fail on librosa 0.10+ |
| `audio/stft.py:42` | `pad_center(window, length)` positional | Expected to fail on librosa 0.10+ |
| `utils/model.py:20`, `:63`, `:65` | `torch.load` with no `map_location` or `weights_only` | The universal generator loaded under the PyTorch 2.14 default. Test full checkpoints with optimizer state, and add `map_location` |
| `hifigan/*.pth.tar.zip` | Generators are stored zipped, while code loads `.pth.tar` | Unzip outside `vendor/` |
| `preprocessor/preprocessor.py:172` | `librosa.load` uses the default rate instead of the configured one | Works only at 22050 Hz. Pass the rate explicitly |
| Text file reads | No UTF-8 encoding set | Fix for Telugu |

`audio/stft.py` uses its own `conv1d` STFT, so it does not rely on `torch.stft` return-type changes. Numerical equivalence with the old librosa mel filter should still be checked.

## Alignment tool

- **MFA:** Conda from conda-forge is the documented route, with Docker and pip as alternatives. Conda is not installed on this machine.
- The docs do not state Windows support outright, and describe building G2P support from source for Mac and Linux only. **Run MFA on Linux.**
- Keep MFA in a separate environment because of its Kaldi and Conda requirements.
- Upstream's README uses old commands (`mfa_align`, `mfa_train_and_align`). Use current MFA commands instead.
- English (India) MFA dictionaries exist at v2.2.1, v3.0.0 and v3.1.0. Pretrained models have no Telugu option.

## Vocoder

- Bundled HiFi-GAN needs only PyTorch. Its config is 22050 Hz, 80 mels, FFT 1024, hop 256, window 1024, and `fmax` 8000. Upstream's `LJSpeech` config matches.
- Upstream says to use the `universal` generator for any dataset other than LJSpeech.
- **MelGAN should not be used:** it downloads through `torch.hub`, needs `fmax` null, and is not needed.
- The universal generator loads and runs under PyTorch 2.14 on CPU (13.9M parameters after weight-norm removal). Its quality on these datasets is untested.

## Audio system requirements

- FFmpeg is not installed locally. It is needed only for formats libsndfile cannot read; Telugu samples are WAV. Svarah's format is unknown.
- eSpeak NG is needed only for the comparison step.
- Hugging Face access can use the installed `huggingface_hub` library or the CLI on the GPU platform.

## Verification to run in the next phase

1. Import every module used from upstream.
2. Build the model and run one forward and backward pass on synthetic data.
3. Load the universal HiFi-GAN and vocode a random mel.
4. Compute mel, F0 and energy for one licensed sample, and compare them against upstream's formula.
5. Parse a real MFA TextGrid.
6. Record the frozen environment in `requirements.txt`.
