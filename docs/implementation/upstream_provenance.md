# Upstream Provenance

| Field | Value |
|---|---|
| Repository | https://github.com/ming024/FastSpeech2 |
| Local path | `vendor/FastSpeech2/` (own Git history, ignored by this repository) |
| Branch | `master` |
| Commit | `d4e79eb52e8b01d24703b2dfc0385544092958f3` |
| Commit date and message | 2021-07-08, "update README and demo page" |
| Cloned | 2026-10-31 (33 commits, 137 files, about 508 MB including history) |
| Local modifications | None at clone time (`git status` clean) |

## Licenses

- FastSpeech2: MIT, copyright (c) 2020 Chung-Ming Chien (`LICENSE`)
- Bundled HiFi-GAN: MIT, copyright (c) 2020 Jungil Kong (`hifigan/LICENSE`)
- The upstream README also credits [xcmyz's FastSpeech](https://github.com/xcmyz/FastSpeech) as its basis. Check that project's license before reusing any code attributed to it.
- The upstream README asks users to cite Chien et al., ICASSP 2021.

## Python version and dependencies

Upstream does not declare a Python version. There is no `setup.py`, `pyproject.toml` or environment file, and the README only uses `python3` and `pip3`. See the [dependency audit](dependency_audit.md).

## Original structure

```text
audio/              STFT and audio processing
config/             AISHELL3, LJSpeech, LJSpeech_paper, LibriTTS (model, preprocess, train YAML)
demo/               Demo audio
hifigan/            Vocoder code, config and two zipped generators (about 52 MB each)
img/                README images
lexicon/            Lexicons
model/              fastspeech2, modules, loss, optimizer
preprocessed_data/  Per-dataset preprocessing outputs and metadata
preprocessor/       Dataset preparation and feature extraction
text/               Cleaners, cmudict, numbers, pinyin, symbols
transformer/        Encoder and decoder layers
utils/              Model and plotting helpers
dataset.py, evaluate.py, prepare_align.py, preprocess.py, synthesize.py, train.py
```

## Restoring the clone

```bash
git clone https://github.com/ming024/FastSpeech2 vendor/FastSpeech2
git -C vendor/FastSpeech2 checkout d4e79eb52e8b01d24703b2dfc0385544092958f3
```
