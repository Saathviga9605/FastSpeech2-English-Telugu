# FastSpeech 2: Paper Versus Implementation

Audit date: 2026-10-31. Upstream commit: `d4e79eb`. Findings come from reading the papers and code. Nothing was trained or run.

## Recommendation

Use **upstream's `LJSpeech` configuration as the baseline: F0 pitch, phoneme-level pitch and energy, normalized values, and the Post-Net**. Report every difference from the paper explicitly. Do not implement CWT unless the assignment requires it.

This means the project is correctly described as "FastSpeech 2 based on the ming024 implementation, closest to arXiv v1 for pitch, with documented deviations". It should not be described as a reproduction of arXiv v8.

**Status:** The project owner approved F0 pitch and a configurable Post-Net on 2026-10-31, provided the assignment does not explicitly require wavelet-based pitch. Verify this against the official PDF. If it requires frame-level variance or strict paper settings, use the fallback below.

## Sources compared

- **v8:** `docs/research_paper/FastSpeech2_Original_Paper.pdf`, the latest version and the one saved in this project.
- **v1:** https://arxiv.org/abs/2006.04558v1, which upstream's README says it most resembles.
- **Upstream `LJSpeech`:** upstream's suggested configuration.
- **Upstream `LJSpeech_paper`:** described by upstream as close to the original paper.

## Comparison

| Topic | arXiv v8 | arXiv v1 | Upstream `LJSpeech` | Upstream `LJSpeech_paper` |
|---|---|---|---|---|
| Pitch target | CWT pitch spectrogram, plus mean and variance | Frame-level F0 | Phoneme-averaged F0 | Frame-level F0 |
| Pitch inversion | iCWT at inference | None | None | None |
| Pitch and energy level | Frame | Frame | Phoneme | Frame |
| Normalization | Per-utterance for the CWT pitch contour | Not established in this audit | Global z-score | None |
| Pitch quantization | 256 bins, log scale | 256 values | 256 bins, linear | 256 bins, log |
| Energy | L2 norm of STFT frame; 256 uniform bins | L2 norm of STFT frame; 256 values | Same source; phoneme-averaged | Same source; frame-level |
| Duration targets | MFA | MFA | MFA TextGrids | MFA TextGrids |
| Encoder and decoder layers | 4 and 4 | 4 and 4 | 4 and 6 | 4 and 4 |
| Encoder and decoder dropout | 0.1 | 0.2 | 0.2 | 0.2 |
| **Post-Net** | **Not mentioned** | **Not mentioned** | **Yes** | **Yes** |
| Mel loss | MAE | Not re-read in detail | MAE, before and after Post-Net | Same |
| Batch size | 48 | 48 | 16 | 48 |
| Training steps | 160k | Not re-read in detail | 900k | 160k |
| Vocoder | Parallel WaveGAN | Parallel WaveGAN | HiFi-GAN | HiFi-GAN |

Text was extracted from the PDFs with a basic script, and the Post-Net result relies on keyword searches, so confirm these rows in a PDF viewer before quoting them in the report.

## The pitch discrepancy

The pitch representation differs in kind, not just in settings.

- **v8** predicts a pitch spectrogram from a 10-scale CWT plus utterance mean and variance, then inverts it for the pitch embedding.
- **Upstream has no CWT code.** It uses PyWorld DIO and StoneMask F0 (`preprocessor/preprocessor.py:182-187`), interpolates unvoiced frames only for phoneme-level features, and predicts F0 directly (`model/modules.py:80-89`).
- Even `LJSpeech_paper` uses F0, so it matches v1 on pitch and is not a v8 reproduction.

### Options

| Option | For | Against |
|---|---|---|
| **A. Phoneme-level F0 (recommended)** | Upstream code exists; upstream reports more natural prosody; pitch plots are easy to interpret | Differs from both paper versions on feature level |
| B. Frame-level F0 | Matches v1; existing config | Upstream reports worse prosody; unvoiced handling needs review; needs no new model code but changes the plots |
| C. CWT from v8 | Matches the latest paper | New, unverified code; a new failure source on data already at risk from alignment and small size |

Phoneme-level F0 also depends more heavily on alignment quality, because every phoneme average uses the alignment boundaries. That is a real risk for Telugu and is why alignment needs its own validation.

## The Post-Net discrepancy

Upstream's README says it adds a Tacotron-2-style Post-Net that the original FastSpeech 2 does not use. The code confirms it:

- It is always built and applied (`model/fastspeech2.py:27`, `model/fastspeech2.py:97`). **No config option disables it.**
- It adds an extra mel loss (`model/loss.py:75`).
- It measured 4.35M of 34.65M trainable parameters.

**Decision:** Keep it in the baseline because upstream's reported results use it, but make it switchable in project code. The comparison experiment is out of scope under the deadline, so the report must make no claim about its benefit.

## Other deviations to report

- 6-layer decoder rather than 4. Upstream says this works better, especially for multi-speaker TTS.
- HiFi-GAN rather than Parallel WaveGAN.
- Gradient clipping and a different learning-rate annealing schedule.
- Linear instead of log quantization, which upstream says is required once values are normalized.

## Fallback if the assignment demands paper settings

Start from `LJSpeech_paper` and add the Post-Net switch, set off. State that pitch still follows v1, or add CWT as a separate work item if v8 is explicitly required.

## Upstream defects found while reading

| Location | Defect | Effect |
|---|---|---|
| `model/modules.py:124`, `:146` | Energy path receives `p_control`, not `e_control` | `--energy_control` has no effect, and pitch control also scales energy. No effect at the default 1.0 |
| `text/__init__.py:67`, `:75` | Unknown symbols are silently dropped | **Telugu phones not in the symbol table would vanish without an error** |
| `preprocessor/preprocessor.py:85-90` | Statistics update sits outside the TextGrid check | Reuses stale values when an alignment is missing, or raises `NameError` if the first file has none |
| `model/modules.py:230` | Second predictor convolution hard-codes `padding=1` | Only correct for kernel size 3 |
| `utils/model.py:33` | `model.requires_grad_ = False` overwrites a method instead of freezing | Harmless in eval use, but misleading |
