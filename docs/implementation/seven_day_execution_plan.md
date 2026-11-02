# Seven-Day Execution Plan

Created Saturday 2026-10-31, evening IST. No step here has started, and upstream code remains unmodified.

## Goal and rules

**Goal: one reproducible, genuinely trained FastSpeech 2 model per language, with every required deliverable.** Quality improvements come only after that.

- **Working deadline: Friday, November 6, 18:00 IST.** "November 7, midnight" could mean the start or the end of that day, and the time zone is not confirmed. Plan for the earlier reading.
- `vendor/` stays untouched. A copy of the needed upstream code goes into `src/` with its MIT notices, and fixes are made there.
- All results reported must come from real runs. If a gate fails, record it and report the fallback in the technical report.

## What the deadline changes

| Area | Earlier proposal | Now | Why |
|---|---|---|---|
| Telugu alignment | MFA trained on the corpus | **Keep, time-boxed to Tuesday night** | No pretrained option, and it needs no new model code |
| Telugu G2P validation | Full comparison with two tools | **Rule tests plus 200-word comparison with whichever tool installs within 1 hour** | Comparison must not hold up training |
| Boundary checks | 20 to 50 utterances | **20 per language** | Enough to catch systematic errors; report the sample size |
| English alignment | Dictionary to be chosen | **Upstream's ARPAbet lexicon, with an acoustic model trained on the selected corpus** | Keeps alignment phones, `g2p_en` and upstream's symbol table consistent |
| Learned-aligner fallback | Third fallback | **Removed** | Too much new code for this deadline |
| Steps | Up to 160k | **Target 50k per language, minimum 20k** | See below |
| Dependencies | Port to a modern stack | **Keep.** Model and vocoder already run on current PyTorch | See benchmark; librosa changes are small |

### Removed from scope

CWT pitch, Post-Net comparison experiments, vocoder fine-tuning, HMM-GD-DNN alignment, and legacy environment reproduction. The Post-Net config switch stays because it is a small change. These can be done only if everything else is finished.

## Step budget

Measured on this CPU, 12 to 22 seconds per step means about 33 to 61 hours for only 10k steps, so **local training is ruled out**.

GPU speed is still **unmeasured**. The planning rate comes from upstream's report of at least about 2.8 steps per second on a GTX 1080 Ti, with an allowance for up to 3 times slower.

| Steps per language | At upstream's rate | At 3x slower | Both languages, at 3x slower |
|---|---|---|---|
| 20k (minimum) | 2 hours | 6 hours | 12 hours |
| **50k (target)** | **5 hours** | **15 hours** | **30 hours** |
| 100k (only with spare time) | 10 hours | 30 hours | 60 hours |

- The target fits the training window of Wednesday morning to Friday morning under both rates, but at 3x slower it would use all of Kaggle's reported 30-hour weekly quota, which is itself unconfirmed.
- **Replace this table with the Sunday GPU measurement.**
- Whether 50k is enough for these datasets is unknown. With about 5 hours of data, it is several hundred epochs. Decide from validation curves and listening.
- Set checkpoints every 5k steps with optimizer state (about 400 MB each), keep only the latest few, and test resume before long runs.

## Daily plan

### Saturday, October 31: unblock

- **You:** add the assignment PDF and paste the five test utterances per language exactly as given.
- **You:** contact the assignment owner about Svarah speaker metadata, and whether training without speaker embeddings is acceptable. Also confirm the deadline's date and time zone.
- **You:** accept Svarah's terms, read the Indic TTS license, and choose the GPU.
- Commit the audit and this plan.

### Sunday, November 1: environment and data

- Create the GPU environment and freeze versions.
- Run the [verification list](dependency_compatibility.md), and repeat the synthetic benchmark on the GPU.
- Download both datasets to the platform. Measure exact durations, sample rates and transcripts.
- Inspect Svarah for any speaker identity in file paths or metadata.
- Verify Telugu: usable duration per speaker at or below the length limit, and speaker consistency through listening across the row range.

**Gate 1, end of day:** a GPU is working with a measured step rate. Otherwise use a paid GPU on Monday.

### Monday, November 2: selection and frontends

- Decide the Telugu speaker setup from Sunday's measurements.
- Decide the English speaker setup. If there is no answer from the owner and no IDs were found, use the default below.
- Create seeded manifests with about 5 hours training and 1 hour validation per language, with no shared clips.
- Check vocoder reconstruction on real audio in both languages.
- Copy upstream code into `src/` and apply: UTF-8 reads, librosa fixes, control bug fix, Post-Net switch, missing-alignment fix, per-language symbols and seeds.
- Build and test both frontends, including the malformed-input and ambiguous-word test cases.

**Gate 2, end of day:** Every character in manifests and test sentences has defined handling, and vocoder reconstruction is intelligible. Otherwise, fix these before alignment.

### Tuesday, November 3: alignment and features

- Install MFA in its own environment on Linux.
- Align English and Telugu, then inspect 20 utterances per language.
- Extract mel, F0, energy and duration. Check statistics and exclusions.

**Gate 3, end of day:** at least 5 hours of training data and 1 hour of validation data per language have usable alignments.

If MFA cannot be installed on the platform, move only alignment to a rented Linux machine. If Telugu alignment quality is poor, tune settings and exclude bad utterances, drawing on the extra audio if available.

### Wednesday, November 4: start real training

- Overfit a tiny batch in each language.
- Test checkpoint, resume and loss-plot generation.
- Start English and Telugu training. Run them in parallel if the platform allows; otherwise alternate sessions.

**Gate 4, midday:** Both pipelines can overfit a tiny batch and resume. Otherwise debug these before using more quota.

### Thursday, November 5: prove the full pipeline

- **Using intermediate real checkpoints, produce every deliverable once:** test audio, loss plots, and duration, pitch and energy plots. Label these as intermediate.
- Listen to the output, and watch for overfitting.
- Draft the technical report's methods, deviations and data sections.
- Decide whether to keep training toward the target or stop.

### Friday, November 6: freeze and submit

- **Noon:** stop training, and select checkpoints using validation behavior, recording the reasons.
- Generate final audio and plots from those checkpoints.
- Write results and limits from real outputs.
- Review public repository contents: license notices, no credentials, no raw data, and manifests allowed by dataset terms.
- With your approval, create the public repository, push with Git LFS, and verify from a clean clone.
- **18:00: working deadline, submit.**

### Saturday, November 7: reserve only

Keep this free for problems. Do not plan work here.

## Default decisions if answers do not arrive

| Topic | Default | Must be reported as |
|---|---|---|
| Svarah speaker IDs, by Monday | Train without speaker embeddings (`multi_speaker: False`) | A model averaged over mixed speakers, a limitation |
| Svarah IDs found in the data | Multi-speaker, with one documented speaker for test audio | Selection method and speaker used |
| Telugu speaker | Decide on Monday from measurements; a single speaker is preferred if 5 + 1 usable hours exist | Selection evidence |
| Target steps not reached | Use the best real checkpoint | Actual steps and effect on quality |

## Minimum acceptable submission

If time runs out, submit this and clearly state what is missing:

- Both models trained on the real data for at least 20k steps, or as far as was actually reached
- Separate configurations, source code and checkpoints
- Real loss curves
- All five utterances per language, and all three variance plots

Weak audio quality is an acceptable result to report. A fabricated or undisclosed result is not.
