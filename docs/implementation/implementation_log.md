# Implementation Log

Only work that actually ran is recorded as done.

## 2026-10-31: Phase 1 foundation

**Ported upstream code to `src/`, with `vendor/` unchanged.** Modern-stack and project changes:

| Change | Reason |
|---|---|
| Energy path uses `e_control` | Upstream bug: it used `p_control` |
| Unknown symbols raise an error | Upstream drops them silently |
| Per-language vocabulary, `n_symbols` from config | Upstream symbol table is global |
| Training raises if a clip exceeds `max_seq_len` | Upstream truncates silently |
| Decoder protected from all-zero predicted durations | Upstream crashes; reproduced in 3 of 12 random untrained models |
| Post-Net config switch, its loss is zero when off | Paper has no Post-Net |
| Keyword arguments for librosa; removed hard-coded `.cuda()` from STFT; no global device | Compatibility |
| Second predictor convolution padding follows kernel size | Upstream hard-coded it |
| Statistics from training split only; failures listed with reasons | Prevents validation leakage and stale values |
| Atomic checkpoints with RNG state; damaged files skipped; log rows after a resume point trimmed | Colab disconnects |
| Homographs resolved by part of speech at inference | Assignment's ambiguous-word case |

**Added:** Telugu rule-based G2P, Parquet reader, inspection, selection, audio export, MFA preparation and wrapper, feature extraction, training, plots, synthesis, Drive sync, and four Colab notebooks.

**Decisions:**

- Indic NLP is not used. Unicode NFC does the same vowel-sign canonicalization, with one less dependency.
- MFA word keys for Telugu are ASCII phone strings, so MFA's tokenizer cannot alter Telugu text.
- Learning-rate anneal steps are scaled from upstream's for a 50k run. This is a judgment call, not a tuned value.

**Verified on CPU with synthetic data, on Python 3.13, PyTorch 2.14.0, NumPy 2.2.2, librosa 1.0.0:** 37 tests pass, including forward, loss, backward, update, checkpoint and resume, feature extraction on a synthetic tone, and full synthesis to WAV and plots.

**Not yet verified:** anything on Colab or a GPU, MFA, real datasets, or model quality.

## 2026-10-31: Telugu dataset measured (full download, all 8,576 clips)

| | Female label | Male label |
|---|---|---|
| Clips | 4,043 (rows 0 to 4042) | 4,533 (rows 4043 to 8575) |
| Total hours | 23.92 | 12.79 |
| Sample rate | 16 kHz | 48 kHz |
| Usable hours (1 to 11.5 s) | **2.09** | **6.58** |

All clips are mono 16-bit WAV. The card's 8.74 hours is wrong. All text passes the strict Telugu frontend, with no unpronounceable characters.

**Provisional selection:** male label only, 5.00 h training (2,564 clips) and 1.00 h validation (514 clips), seed 1234, because it is the only single speaker with enough usable audio. It leaves only 0.58 h spare for alignment exclusions.

**Speaker consistency is not verified.** Median F0 for the male label moves gradually between about 160 and 209 Hz across the rows, with no sharp step. The level is high for a label of male, and close to the female label's 180 to 217 Hz. F0 cannot settle whether this is one speaker or whether labels are correct, so listening is required. Samples are in `output/speaker_check/` (not committed).

**Vocoder check on 4 real validation clips:** runs, with log-mel L1 of 0.44 to 0.53. This number does not establish perceived quality; listening samples are in `output/vocoder_check/`.

**Corpus preparation ran on real data:** 3,078 labs, 12,316 lexicon entries, 47 phones used. Rare phones include `jh` (9 words) and `ddh` (17 words). MFA has not run yet because it needs Colab's Linux.

Manifests contain dataset transcripts, so they are ignored by Git until dataset terms are reviewed. They can be regenerated from the seed.

## 2026-10-31: Authentication, dataset inspection, Drive layout

- Added `src/data/hf_auth.py`, `validation.py`, `storage.py` and `inspection.py`. Extended `corpus.py` so bad rows are recorded instead of crashing the scan, and complete downloads are reused.
- `load_dataset` runs with `streaming=True`. A full load would store a second copy of each dataset on Drive; full statistics come from the Parquet files the pipeline reads.
- `HF_HOME` is never moved to Drive, and login refuses to run if it is, because `login()` writes a token file there. On Colab that file is on temporary disk.
- Checkpoints moved to `checkpoints/<language>`; earlier ones are copied, never deleted. No speaker identity is inferred from metadata.

## 2026-10-31: Official PDF, required layout

Read the PDF in full (`docs/assignment/requirements_checklist.md`). Restructured to its layout: per-language `train.py`, `preprocess.py`, `inference.py`, `config.yaml`. The PDF stays out of Git until publishing it is confirmed.

Test sentences come from the rendered page and all versions are stored in `data/test_utterances/sentences.json`. The PDF text layer and the owner's message both differ from the printed text. Selected text matches the text layer apart from spaces (tested). Every frontend transformation is logged.

## 2026-10-31: Telugu alignment on real audio (local CPU, MFA 3.4.3)

**Command:** `mfa train --clean -j 8 --single_speaker --output_directory data/interim/telugu/textgrids data/interim/telugu/mfa_corpus data/interim/telugu/lexicon.txt data/interim/telugu/acoustic_model.zip`

**Result:** exit 0 in 83m58s. Produced `acoustic_model.zip` (60 MB), which `mfa model inspect` reads, and exported final-stage (`sat_3_ali`) TextGrids. No separate `mfa align` was run.

| | Clips | Valid | Rejected | Coverage | Clip hours kept | Hours without edge silence |
|---|---|---|---|---|---|---|
| Train | 2,564 | 2,524 | 40 | 98.44% | 4.904 | 4.104 |
| Validation | 514 | 507 | 7 | 98.64% | 0.985 | 0.821 |

All 47 rejections have no TextGrid: MFA reported them as errors in the final stage (3 errors in the previous reported round). They cluster in rows 7161 to 7382 and 8057 to 8277. **The cause is unknown.** Abbreviations are not a sufficient explanation: 5 of 47 failed clips have them, against 153 of 3,031 kept. All are excluded with a recorded reason.

**Quality:** no phone mismatches against the lexicon or vocabulary. Speech phones: mean 66.3 ms, percentiles 5/25/50/75/95 of 23/46/58/81/139 ms. Of 248,527 intervals, 318 speech phones have zero frames, 2,133 are under 15 ms, and 1 is over 500 ms. Zero-frame phones are kept, as upstream does.

**Timing plausibility (300 clips, own alignment vs shuffled control, medians):** vowel minus silence 41.4 vs 10.9 dB; fricative minus vowel high-band ratio 0.593 vs -0.017; vowel minus stop voicing 0.587 vs -0.011. This is not boundary accuracy: no hand-labelled reference exists.

**Spectrograms inspected by eye (not listened to):** short clip `006562`, long `007973`, punctuation `007554`, consonant cluster `007703`, number word `007217`, Latin typo `006008`. Boundaries generally follow visible onsets, closures and silences; very short phones in fast passages are hard to judge. No selected clip contains digits, so digit expansion cannot be tested against alignment.

**Data defect:** 7 kept clips contain Latin letters. In `006008`, a Latin `o` sits inside a Telugu word and is read as a letter name. They are kept for now and are a limitation for the report.

**Duration features:** 3,031 clips extracted (492 MB). No length mismatches, non-finite values, clips over `max_seq_len`, or train/validation overlap.

**Archive:** `prepared.tar` (0.59 GB) restored with 3,078 labels, 3,031 TextGrids and no WAVs. The checker works and states timing was not checked; feature extraction refuses with a clear error.

**Tests:** 77 passed, 0 failed. All four notebooks validate. Nothing has run on Colab.

**Blockers:** Colab benchmark, Svarah access and English alignment, Indic TTS license, speaker-consistency listening check, public-repository permission.

**Next action:** run the Colab steps: benchmark, Svarah inspection, Telugu restore. No full training until both data pipelines are valid.

## 2026-10-31: Svarah access and inspection (local, authenticated)

Access verified with a real `load_dataset` call using a token supplied by the owner in an environment variable; it was not written to any file. One `test` split: 6,656 clips, 9.614 h, all 16 kHz mono PCM. No invalid audio; the dataset's `duration` column agrees with measured audio.

**Speakers:** no speaker column. Filename fields have 2,938 and 3,232 unique values with a median of one clip, so they are not speaker IDs. There are 115 distinct metadata profiles, close to the paper's 117 speakers, but this is not proof and is not used. All clips form one group, without speaker embeddings, pending the assignment owner.

**Filtering:** 1,215 clips are under 1 s (0.197 h) and 538 over 11.5 s (2.456 h). 69 transcripts have symbols without a defined pronunciation (`%`, `&`, `+`, `@`, `_`, and a quote form), recorded and excluded. Usable: 4,841 clips, 6.85 h. No bracketed noise markers; 8 clips contain filler words.

**Selection:** seeded and disjoint by transcript as well as by clip, because some sentences repeat across recordings.

## 2026-10-31: English alignment on real audio (local CPU, MFA 3.4.3)

**First run rejected.** `mfa train` with default topology exited 0 in 104m53s with 4,220 of 4,231 TextGrids, and timing contrasts were well above control. But MFA's default lets a phone last one 10 ms frame, and 7.2% of intervals did (Telugu: 0.9%): `TH` 99.9%, `G` 98.1%, `R` 85.3%, `HH` 44.4%. The timing contrasts did not reveal this. Why these phones collapse is unknown. Kept as `textgrids_default_topology` and `acoustic_model_default_topology.zip`.

**Change:** `English/mfa_topology.yaml` requires three states, so 30 ms, per phone. Used by `scripts/run_mfa.sh` for English only. This is an alignment setting; the model architecture is unchanged.

**Second run:** exit 0 in 83m55s. Command as the Telugu run, plus `--topology_path English/mfa_topology.yaml`.

| | Clips | Valid | Rejected | Coverage | Clip hours kept |
|---|---|---|---|---|---|
| Train | 3,533 | 3,501 | 32 | 99.09% | 4.938 |
| Validation | 698 | 689 | 9 | 98.71% | 0.984 |

All 41 rejections have no TextGrid; 31 had one in the first run and 1 gained one. The cause is unknown. A three-state minimum makes one-frame phones impossible, so their absence alone does not show the boundaries are right.

**Quality:** median speech phone `TH` 80 ms, `G` 70 ms, `R` 50 ms (raw TextGrid). Speech phones percentiles 5/25/50/75/95 of 34.8/46.4/81.3/116.1/197.4 ms in 11.6 ms feature frames, mean 89.7 ms, none zero-frame, 174 over 500 ms (not inspected individually).

**Timing plausibility (300 clips, own alignment vs shuffled control, medians):** vowel minus silence 24.6 vs 4.4 dB; fricative minus vowel high-band ratio 0.389 vs -0.011; vowel minus stop voicing 0.442 vs -0.013. Not boundary accuracy.

**Spectrograms inspected by eye (not listened to):** `english_000000`, `english_000046`. Boundaries generally follow visible events; the fast passage near the end of `english_000046` is too dense to judge at this scale. Two clips are too small a sample to generalize.

**Known defects, not fixed:** the lexicon step keeps the first listed pronunciation (`good` becomes `G IH0 D`), and 1,018 of 6,571 corpus words have several listed. `CE` in `english_000000` is treated as the word `S`, though probably spoken as letters. Telugu `ny` collapses to one frame in 87.7% of 1,458 cases; Telugu has not been realigned.

**Duration features:** fixed a crash from upstream's outlier filter, whose strict bounds return nothing when at least half of a clip's values are identical. Telugu features predate the fix. Extracted 3,499 train and 689 validation clips (548 MB); 41 lack alignment and 2 have under two voiced frames, both recorded in `excluded.jsonl`. No length mismatches, non-finite values, clips over `max_seq_len`, or train/validation overlap. Mel frames without edge silence: 4.590 h train, 0.908 h validation.

**Not verified:** English on Colab, speaker identity, listening quality.

## 2026-11-01: Real-data smoke run and report builder

Ten training steps, a checkpoint and synthesis of all ten test sentences ran on real features for both languages, on CPU, with outputs in temporary storage. This shows the code path works, not model quality. CPU speed was about 15 s per step including the checkpoint save, so 50,000 steps would take about 208 hours per model. Training therefore needs Colab.

`scripts/build_report.py` builds the report from `Report/report_source.html`. While anything is pending it writes only `output/Technical_Report_DRAFT.pdf`.

## 2026-11-01: First Colab run on English

GPU benchmark ran on a Tesla T4: 1.17 to 2.06 synthetic steps/s at batch 16. Running every notebook cell exposed two bugs. `git archive` under `core.autocrlf` gave `run_mfa.sh` CRLF endings, which fail on Linux; `.sh` files are now fixed to LF. `drive_sync.py save` replaced the good archive after feature extraction failed; it now refuses without `train.jsonl`. The local archives were unaffected.
