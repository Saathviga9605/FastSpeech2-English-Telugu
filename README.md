# FastSpeech 2 for English and Telugu

Two separate FastSpeech 2 text-to-speech models, one for English and one for Telugu, with generated test speech and an analysis of the duration, pitch and energy predictions.

Built as an assignment for an Intern Position at LTRC, IIIT Hyderabad.

**[Read the technical report](Report/Technical_Report.pdf)** for the full method, results and analysis.

## Results at a glance

| | English | Telugu |
|---|---|---|
| Dataset | [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) | [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) |
| Audio selected (train + validation) | 5 h + 1 h | 5 h + 1 h |
| Clips used (train + validation) | 3,499 + 689 | 2,524 + 507 |
| Submitted checkpoint | Step 27,500 | Step 25,000 |
| Validation mel loss | 0.873 | 0.580 |
| Pitch, energy and duration vs per-phone baseline | Only energy is better | All three are better |

Training stopped before the planned 50,000 steps in both cases, when the Colab GPU limit was reached. Validation loss had already stopped improving by then.

| Language | Audio | Loss curves | Variance plots |
|---|---|---|---|
| English | [Test audio](Results/English/test_generated_audios/) | [Training](Results/English/training_loss.png), [validation](Results/English/validation_loss.png) | [Duration](Results/English/variance_adaptor_plots/duration/), [pitch](Results/English/variance_adaptor_plots/pitch/), [energy](Results/English/variance_adaptor_plots/energy/) |
| Telugu | [Test audio](Results/Telugu/test_generated_audios/) | [Training](Results/Telugu/training_loss.png), [validation](Results/Telugu/validation_loss.png) | [Duration](Results/Telugu/variance_adaptor_plots/duration/), [pitch](Results/Telugu/variance_adaptor_plots/pitch/), [energy](Results/Telugu/variance_adaptor_plots/energy/) |

## Architecture

```text
Text -> language-specific frontend -> phoneme sequence
     -> Transformer encoder
     -> variance adaptor (duration, pitch, energy, length regulator)
     -> Transformer decoder and Post-Net -> mel-spectrogram
     -> HiFi-GAN vocoder -> waveform
```

The model is based on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), and keeps that implementation's phoneme-level F0 and energy and its Post-Net. The main additions are:

- **Telugu:** a rule-based grapheme-to-phoneme frontend, and number, ordinal and time expansion.
- **English:** homograph handling by part of speech, number and date expansion, and a three-state minimum per phone during alignment.

## Quick start

Generate the test sentences from the trained checkpoints. This needs no dataset and runs on CPU.

```bash
# Install Git LFS first (https://git-lfs.com), so the checkpoints download
git clone https://github.com/Saathviga9605/FastSpeech2-English-Telugu.git
cd FastSpeech2-English-Telugu

# Install PyTorch and NumPy first (https://pytorch.org)
pip install -r requirements.txt

python English/inference.py    # writes to Results/English/
python Telugu/inference.py     # writes to Results/Telugu/
```

To synthesize your own sentences, add them to [data/test_utterances/sentences.json](data/test_utterances/sentences.json).

## Train from scratch

A GPU is needed for practical training. I used a Colab T4, and the notebooks in [notebooks/](notebooks/) run the whole process on Colab. The commands are the same for `Telugu/`:

```bash
python English/preprocess.py    # --start and --stop choose steps
python English/train.py         # rerunning resumes from the newest checkpoint
python English/inference.py
python scripts/collect_deliverables.py
```

- **Svarah access:** the dataset is gated. Accept its terms on Hugging Face, then set `HF_TOKEN` in your environment or as a Colab secret. Never commit a token.
- **Alignment:** Montreal Forced Aligner 3.4.3 runs in its own environment, created by [scripts/run_mfa.sh](scripts/run_mfa.sh).
- **Configuration:** each language has its own file, [English/config.yaml](English/config.yaml) and [Telugu/config.yaml](Telugu/config.yaml).

## Tests

```bash
python -m pytest -q tests
```

## Project structure

```text
English/, Telugu/   train.py, preprocess.py, inference.py, config.yaml, checkpoints/
Results/            Loss plots, test audio and variance plots for each language
Report/             Technical report
src/                Model, text frontends, audio features, training and inference
scripts/            Alignment, checks, report building and result collection
notebooks/          Colab notebooks
data/               Test sentences; dataset audio is not included
docs/               Dataset notes, implementation log and experiment records
tests/              Tests
```

## Known limitations

- Both models overfit, English severely. English pitch on unseen clips is worse than a constant guess.
- Svarah has no speaker labels here, so English trains on an unlabelled mix of voices. This limits any comparison between the two models.
- Some sentences sound a bit robotic, and the last word is sometimes cut off.
- English `AM` is read as "am", and Telugu `ny` is predicted at a single frame.

The report covers these and other limitations, with possible improvements.

## Credits and licenses

- [FastSpeech2](https://github.com/ming024/FastSpeech2) by Chung-Ming Chien and [HiFi-GAN](https://github.com/jik876/hifi-gan) by Jungil Kong, both under the MIT license. See [third_party_licenses/](third_party_licenses/).
- Paper: Ren et al., [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558).
- Datasets: Svarah by AI4Bharat, and the Indic TTS Database by the Speech Technology Consortium, IIT Madras. Dataset audio and transcripts are not redistributed here. See the [Indic TTS notice](third_party_licenses/IndicTTS_NOTICE.md).
