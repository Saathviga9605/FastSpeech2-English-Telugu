# FastSpeech 2 for English and Telugu

**STATUS: COMPLETE, WITH KNOWN LIMITATIONS.** Both models stopped early at the Colab GPU limit: English at step 27,500 and Telugu at step 25,000, of a planned 50,000. Both overfit, English severely. Read the [technical report](Report/Technical_Report.pdf) for measured results.

## Context

This is an implementation assignment from the Speech Processing Lab (SLFI), LTRC, IIIT Hyderabad.

## Research objective

Implement FastSpeech 2 text-to-speech for English and Telugu, and produce trained checkpoints, generated test audio, variance adaptor plots (duration, pitch, energy) and a report for each language as the assignment requires.

## References

| Item | Source | Local location |
|---|---|---|
| Paper | Ren et al., *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech*, [arXiv:2006.04558](https://arxiv.org/abs/2006.04558) | [docs/research_paper/](docs/research_paper/) (v8) |
| Reference implementation | [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), MIT license | `vendor/FastSpeech2/` (not tracked here) |
| English dataset | [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) | [docs/datasets/svarah.md](docs/datasets/svarah.md) |
| Telugu dataset | [SPRINGLab/IndicTTS_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) | [docs/datasets/indictts_telugu.md](docs/datasets/indictts_telugu.md) |

The pinned upstream commit and instructions for restoring the clone are in [upstream_provenance.md](docs/implementation/upstream_provenance.md).

## Architecture

```text
Text -> language-specific frontend -> phoneme sequence
     -> Transformer encoder
     -> variance adaptor (duration, pitch, energy; length regulator)
     -> Transformer decoder -> mel-spectrogram
     -> HiFi-GAN vocoder -> waveform
```

During training, the variance adaptor uses ground-truth duration, pitch and energy. During inference, it uses predicted values. The models keep upstream's phoneme-level F0 and Post-Net; see section 1 of the report.

## Project structure

```text
English/, Telugu/  train.py, preprocess.py, inference.py, config.yaml, symbols.json, checkpoints/
Results/           Loss plots, test audio and variance adaptor plots, per language
Report/            Technical report
src/               Modified upstream model and project code
scripts/           Helpers, including collect_deliverables.py and run_mfa.sh
data/test_utterances/sentences.json   Assignment test sentences, unedited
notebooks/         Colab notebooks, the supported way to run on a GPU
docs/              Audits, checklist and implementation log
tests/             Tests
```

## Commands

```bash
python English/preprocess.py   # or Telugu/; options --start and --stop choose steps
python English/train.py        # rerunning resumes from the newest checkpoint
python English/inference.py    # writes Results/English/
python scripts/collect_deliverables.py
```

## Results

| | English | Telugu |
|---|---|---|
| Dataset | Svarah | IndicTTS Telugu |
| Submitted checkpoint | `English/checkpoints/final.pth.tar`, step 27,500 | `Telugu/checkpoints/final.pth.tar`, step 25,000 |
| Validation mel loss | 0.873 | 0.580 |

Clone with [Git LFS](https://git-lfs.com) installed to get the checkpoints. Run records are in [docs/experiments/](docs/experiments/).

Known limitations: both models stopped before the planned 50,000 steps, English has no speaker IDs, and Telugu speaker consistency is unverified. The report covers these and others.

## Dataset access on Colab

1. Create a fresh Read token at https://huggingface.co/settings/tokens. While logged in, accept the terms at https://huggingface.co/datasets/ai4bharat/Svarah.
2. In Colab, open Secrets (key icon), add `HF_TOKEN`, and allow the notebook to access it. Never put a token in code, Drive or Git.
3. Upload the code zip from `scripts/package_for_colab.py` to `MyDrive/FastSpeech2-IIITH/`, then run `notebooks/01_setup_and_benchmark.ipynb`.
4. Run setup, login and section 1 in `notebooks/02_english.ipynb` and `notebooks/03_telugu.ipynb` to verify each dataset.

Drive layout: `datasets/` holds raw downloads, `processed/` prepared archives, `checkpoints/` training state, `experiments/` logs and reports, `outputs/` generated results. Dataset audio and transcripts must not go in the public repository, as redistribution rights are unconfirmed. The Indic TTS terms are noted in `third_party_licenses/IndicTTS_NOTICE.md`.

## Attribution

Upstream FastSpeech2 is copyright (c) 2020 Chung-Ming Chien, and its bundled HiFi-GAN code is copyright (c) 2020 Jungil Kong. Both are under the MIT license.
