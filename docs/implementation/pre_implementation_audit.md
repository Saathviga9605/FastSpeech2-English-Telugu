# Pre-Implementation Audit and Feasibility Report

Audit date: 2026-10-31, updated the same day with confirmed assignment details and benchmark results. Upstream commit: `d4e79eb`.

**Conclusion: two genuinely trained models by the deadline looks achievable with a working GPU by November 1 or 2, but it is not yet proven.** GPU speed and both alignments are still untested. English is the larger scientific risk. The plan is in [seven_day_execution_plan.md](seven_day_execution_plan.md).

**Evidence labels:** *Read* means checked in code, a paper or official metadata. *Sampled* means measured on a small non-random sample. *Estimated* means calculated with stated assumptions. *Not verified* means no evidence was obtained.

No datasets were downloaded, no packages were installed, and no real-data training was run. WAV headers of 54 Telugu clips were read with small range requests. An approved benchmark ran upstream's model on synthetic data only. Upstream code is unmodified.

## 1. Assignment verification

The assignment PDF is **still missing** from the workspace. The earlier planning context was not available to this audit, so established items below rest on the project owner's message of 2026-10-31.

### Established from the assignment instructions, as confirmed by the project owner

| Requirement | Detail |
|---|---|
| Deadline | November 7, 2026, midnight |
| Models | Two separate FastSpeech 2 models |
| Datasets | English: Svarah. Telugu: IndicTTS Telugu |
| Data amount | Approximately 5 hours training and 1 hour validation, per language |
| Test utterances | Five specified per language, including ambiguous-word and malformed-input examples |
| Analysis | Duration, pitch/F0 and energy predictions from the variance adaptor |
| Submission | Source code, separate configurations, trained checkpoints, training and validation loss plots, generated audio, variance-adaptor plots, public GitHub repository, technical report |

This reconciles the earlier figures: "six hours" is 5 + 1 per language, and "ten utterances" is five per language.

### Still to verify against the official PDF

| Item | Why it matters |
|---|---|
| **Exact text of all ten test utterances** | **Not available to this audit. Blocks frontend tests.** Must be copied verbatim |
| Deadline date meaning and time zone | Start or end of November 7 changes the time available by a full day |
| Whether wavelet pitch is required | Your F0 decision depends on this |
| Whether the Post-Net or a specific vocoder is required or banned | Affects the configuration and pretrained HiFi-GAN use |
| Whether "approximately" sets a tolerance | Affects filtering |
| Whether a speaker subset is acceptable | Affects both datasets, including Telugu's speaker choice |
| Required code structure, and plot format | Affects packaging |
| Report format and required sections | Affects Friday's writing |
| Submission method, and repository contents expected | Checkpoints of about 134 MB or more exceed GitHub's normal file limit, so require LFS or external links |

### Consequences of the confirmed requirements

- **Malformed input must be handled, not rejected.** The earlier advice to raise errors on unknown symbols now applies only to training data. At inference, the frontend must apply a defined, logged policy, and must not crash or silently lose text.
- **A public repository needs a review** of dataset terms for manifests and audio, the arXiv PDF, and permission to publish the assignment PDF. The local Git rule against pushing stands until you approve the push.
- **Loss plots need new code**, as upstream logs losses but does not export these figures. Duration plots also need new code.

## 2. Architecture findings

Full comparison: [paper_vs_implementation.md](../architecture/paper_vs_implementation.md).

- The saved paper (v8) uses CWT pitch. Upstream has no CWT code and predicts F0. *Read*
- Upstream always includes a Post-Net that neither paper mentions. *Read*
- **Recommended target:** upstream's `LJSpeech` configuration, with deviations reported.

### Upstream implementation map

| Function | Files | Reuse |
|---|---|---|
| Text processing | `text/__init__.py`, `text/symbols.py`, `text/cleaners.py`, `text/cmudict.py`, `text/numbers.py` | English: reuse. Telugu: **adapt**; symbol table is global and unknown symbols are dropped |
| Inference text | `synthesize.py:32-84` | English: reuse. Telugu: **new** |
| Dataset preparation | `prepare_align.py`, `preprocessor/ljspeech.py`, `preprocessor/libritts.py`, `dataset.py` | **New** for both datasets; loaders adapt the `dataset.py` format. Selection must be by duration, because upstream uses a fixed count and an unseeded shuffle (`preprocessor/preprocessor.py:142-151`) |
| Alignment | External MFA; read at `preprocessor/preprocessor.py:163-166`, `:253-291` | Parsing: reuse. Telugu lexicon and model: **new** |
| Pitch extraction | `preprocessor/preprocessor.py:182-217` | Reuse, with the rate fix. Check F0 on both datasets |
| Energy extraction | `audio/stft.py`, `audio/tools.py`, `preprocessor/preprocessor.py:194-228` | Reuse, with librosa fixes |
| Model architecture | `model/fastspeech2.py`, `transformer/*` | Reuse; add a Post-Net switch |
| Variance adaptor | `model/modules.py` | Reuse; fix the energy control bug |
| Training | `train.py`, `model/loss.py`, `model/optimizer.py` | Mostly reuse. Add seeds and more frequent checkpoints; `save_step` is 100000 in the suggested config |
| Validation | `evaluate.py` | Reuse. Loss is measured with ground-truth variance inputs, so also listen to and plot fully predicted output |
| Inference | `synthesize.py`, `utils/tools.py:164-211` | Adapt. Saves mel, pitch and energy figures, but **duration plots need new code** |
| Vocoder | `hifigan/`, `utils/model.py:42-71` | Reuse the `universal` generator after unzipping |

Upstream code remains in `vendor/`. Changes belong in `src/`, `scripts/` and `configs/`.

## 3. Dataset feasibility

### Summary

| | Svarah | IndicTTS Telugu |
|---|---|---|
| Access | **Gated: login and accepting terms required** | Public, but subject to Indic TTS terms |
| Splits | `test` only | `train` only |
| Rows | 6,656 | 8,576 |
| Stated duration | 9.6 hours | ~8.74 hours (**conflicts with the evidence below**) |
| Download size | About 1.09 GB | About 6.44 GB |
| Schema | `audio_filepath`, `duration`, `text`, `gender`, `age-group`, `primary_language`, `native_place_state`, `native_place_district`, `highest_qualification`, `job_category`, `occupation_domain` | `audio`, `text`, `gender` |
| Speakers | 117 by the paper; **no speaker ID column in the schema** | 2 by the card |
| Audio | Not verified. Size per hour is consistent with about 16 kHz, 16-bit audio (*Estimated*) | **Mixed in the sample:** female 16 kHz, male 48 kHz, all 16-bit mono PCM |
| License | No license in Hub metadata. AIKosh lists CC BY 4.0. Card unread | Indic TTS license. **Could not be read: the site timed out** |
| 5 + 1 hours possible? | **Numerically yes, but about 63% of all audio** | **Likely, possibly from one speaker. Must be measured** |

### IndicTTS Telugu: the card's duration looks wrong

The card says about 8.74 hours, 48 kHz. Other evidence disagrees:

- Public statistics cover 6,432 of 8,576 rows, with a mean of 17.38 seconds. That alone is about 31 hours. *Read*
- WAV headers of 54 sampled clips show the female speaker at 16 kHz, with a mean of 19.2 seconds, and the male at 48 kHz, with a mean of 8.6 seconds. *Sampled*
- The transcript rate is plausible for this duration: about 9 to 12.5 characters per second. *Sampled*

| Estimate | Female | Male | Total |
|---|---|---|---|
| Combining statistics, sample and byte sizes (*Estimated*) | About 25 hours | About 9 to 11 hours | **About 34 to 36 hours** |

One possible explanation is that the card's figure treats all data as 48 kHz, 32-bit audio: 6.04 GB at that rate is almost exactly 8.74 hours. This is speculation. The card also contains a copied sentence that calls it the "Hindi" portion, so it should not be trusted over the audio. **Compute exact durations after download.**

**Long clips matter.** Upstream's limit is 1000 frames, or about 11.6 seconds. Of 30 sampled male clips, 28 were under this limit, suggesting about 8 to 9 hours of usable male audio. *Estimated from a small sample.* Sampled female clips ran as long as 79 seconds, so the female speaker would need filtering or segmentation.

**Status: not finalized, by project owner's decision.** The male speaker remains the leading option, because of its higher sample rate and shorter clips. The choice waits until usable duration and speaker consistency are measured after download.

### Svarah: possible, but not designed for TTS

Svarah is an ASR benchmark. Its paper describes 9.6 hours from 117 non-native English speakers with read and spontaneous speech.

- **About 4.9 minutes per speaker on average** (*Estimated*), so a good single-speaker model is not possible.
- **Speaker embeddings would need speaker IDs, and the public schema has none.** IDs may be derivable from file names or metadata combinations, but this cannot be checked until access is granted. The project owner will ask the assignment owner. Training without embeddings is the fallback, and would produce a model averaged over speakers.
- The required 5 + 1 hours is about 63% of the audio, leaving about 3.6 hours of room to filter out noise, disfluencies or poor alignments.
- Spontaneous speech and accent variation are likely to make alignment and prosody harder. *Not verified.*
- Test utterances need a chosen speaker, or speakers, for inference.

**Expected outcome:** intelligibility is achievable, but quality may be limited by per-speaker data and accent variety. Report this as a dataset limitation rather than hiding it.

## 4. Telugu adaptation feasibility

Full analysis: [telugu_feasibility.md](telugu_feasibility.md).

- **No pretrained Telugu MFA model or dictionary is listed.** Train one on the corpus. *Read*
- Indic NLP has a `TeluguNormalizer` but lists no G2P. *Read*
- eSpeak NG lists Telugu, but quality is unknown. Unified Parser needs a licence check. *Read*
- **Strategy:** own rule-based G2P, validated against both tools, then MFA trained on the corpus with hand-checked boundaries.

## 5. Compute feasibility

**GPU training time is still not benchmarked.** The approved benchmark ran on this machine's CPU with synthetic data, so it tests compatibility and CPU speed only. It says nothing about training quality, or GPU memory.

### Benchmark results (*Measured, synthetic data, CPU*)

Environment: PyTorch 2.14.0+cpu, NumPy 2.2.2, 8 threads. `unidecode` and `inflect` were stubbed because they are not installed. Adam was used without upstream's scheduler.

| Test | Result |
|---|---|
| Forward, loss and backward, batch 16, 600 frames | **About 12 seconds per step**, 2.33 GB peak process memory |
| Same, batch 16, 1000 frames | **About 21.5 seconds per step**, 4.14 GB peak process memory |
| Inference without targets | Works |
| Load universal HiFi-GAN with default `torch.load` | Works |
| HiFi-GAN on about 5 seconds of random mel | 3.4 seconds, output shape correct |
| Telugu text through upstream `text_to_sequence` | **Returns an empty list, confirming silent loss** |
| Training input of 1320 frames | **Silently truncated to 1000, while `mel_lens` still reports 1320** |
| `audio` module import | Fails here because librosa is not installed, so **feature extraction is untested** |

**Model size:**

| Configuration | Total parameters | Trainable | Model only | With Adam state |
|---|---|---|---|---|
| Upstream `LJSpeech` (6-layer decoder, Post-Net) | 35.16M | 34.65M | 134.2 MB | 398.7 MB |
| 4-layer decoder, Post-Net | 29.39M | 28.87M | Not measured | Not measured |

For comparison, v8 states 27M parameters for its 4-layer model without Post-Net. **Final checkpoints can be saved without optimizer state to reduce LFS use, but resumable checkpoints need it.**

### Storage (*Estimated*)

| Item | Size |
|---|---|
| Datasets as downloaded | About 7.5 GB |
| 22.05 kHz WAVs for 12 hours | About 2 GB |
| Mel features for 12 hours | About 1.2 GB |
| Checkpoints, keeping about 10 per language | About 8 GB |
| MFA models and working files | Not measured; allow several GB |
| **Plan for** | **30 GB or more** |

This machine has enough disk space. Cloud session storage must be checked.

### Training time

Upstream reports under 10k steps, under 1 hour, for acceptable output on a GTX 1080 Ti, which implies at least about 2.8 steps per second on LJSpeech. Applying that rate, with a 1x to 3x allowance for unknowns:

| Steps per language | At that rate | With allowance | Both languages |
|---|---|---|---|
| 50k | About 5 hours | 5 to 15 hours | 10 to 30 hours |
| 100k | About 10 hours | 10 to 30 hours | 20 to 60 hours |
| 160k (v8) | About 16 hours | 16 to 48 hours | 32 to 96 hours |

With about 5 hours of training data at batch size 16, 100k steps would be several hundred epochs, so stop based on validation behavior and listening rather than matching upstream's 900k. **How many steps these datasets need is unknown.** The plan targets 50k per language, with a 20k minimum.

**Batch size:** v8 used 48 on a V100. Upstream's suggested 16 is the sensible start. Filtering to 11.6 seconds also limits memory use. Confirm with a benchmark.

### Environments

| Environment | Assessment |
|---|---|
| This machine | **Ruled out for training.** At 12 to 22 seconds per step, 10k steps takes about 33 to 61 hours. Suitable for docs, text frontend work and tests |
| Kaggle | **Recommended starting point** if quota is confirmed. Third-party sources report about 30 GPU hours per week with T4 or P100 and 9 to 12-hour sessions. Not confirmed from Kaggle |
| Google Colab free | Backup. Google does not publish limits, and disconnection is a risk |
| Rented GPU, or Colab paid | **Use if the deadline is tight.** Prices were not verified |
| IIITH or institutional GPU | Best if you have access |

**Recommended GPU:** one NVIDIA GPU with at least 12 to 16 GB memory. Checkpoints must go to persistent storage, and saves should happen at least hourly on interruptible services.

### Can both languages finish before the deadline?

**Likely yes at 50k steps per language, if a GPU works by Monday and alignment finishes by Tuesday.** That is about 10 to 30 hours of GPU time for both, inside the planned Wednesday to Friday window. It is not yet proven, because GPU speed is unmeasured. At 3x slower it would use all of Kaggle's reported weekly quota, leaving no room for reruns, so **line up a paid GPU as a backup now**. 100k steps per language is not a realistic plan on free quota.

### Reassessment against the deadline

| Item | Verdict | Change |
|---|---|---|
| Alignment approach | **Achievable, not yet tested** | Keep MFA, time-boxed to Tuesday. Use upstream's ARPAbet lexicon with a corpus-trained model for English, and drop the learned-aligner fallback |
| Training steps | **Reduce** | 50k target and 20k minimum per language, based on a measurement taken Sunday |
| Dependency setup | **Achievable, risk reduced** | The model and HiFi-GAN already run on current PyTorch and NumPy. Librosa, pyworld, tgt and MFA remain untested |

For English, a pretrained English (India) MFA option exists, but its phone set differs from the ARPAbet symbols used by upstream's lexicon, `g2p_en` and symbol table. Using it would require frontend changes, so it is not the plan.

## 6. Dependency compatibility

Full analysis: [dependency_compatibility.md](dependency_compatibility.md).

- Upstream pins require Python 3.8 or older for `torch==1.7.0`. *Read*
- **Recommended:** port to a modern stack, in isolated Linux environments, with MFA kept separate.
- Librosa call changes are expected. TensorFlow is not needed.
- Nothing has been tested.

## 7. Risks

| Level | Risk | Mitigation |
|---|---|---|
| **Critical** | Exact test utterances and official PDF not in the workspace | Supply them on October 31 |
| **Critical** | About six working days, with GPU speed and alignment unproven | Dated gates, a Friday freeze, and deliverables dry run on Thursday |
| **Critical** | No GPU environment confirmed | Decide today, benchmark on Sunday, line up a paid backup |
| **High** | Malformed test inputs could crash or silently corrupt synthesis | Defined inference policy and tests |
| **High** | Public repository could expose content that dataset terms restrict | Review before pushing, with your approval |
| **High** | Svarah may offer no reliable speaker IDs | Inspect immediately after access; ask the assignment owner if needed |
| **High** | Svarah may give weak TTS quality (few minutes per speaker, spontaneous speech, accents) | Filter by quality, select speakers, report limits |
| **High** | Telugu alignment quality is unproven, with no pretrained models | Validation gates and fallbacks |
| **High** | Indic TTS licence unread | Read it before using the data |
| **High** | Telugu card metadata is unreliable | Measure all audio after download |
| **Medium** | Upstream silently drops unknown symbols | Raise errors; add tests |
| **Medium** | Clips over 11.6 seconds silently truncated (confirmed by benchmark) | Filter by duration |
| **Medium** | Mixed Telugu sample rates | Prefer one speaker; resample to 22.05 kHz; record bandwidth |
| **Medium** | Sources at 16 kHz, if Svarah is, hold nothing above 8 kHz | Matches HiFi-GAN `fmax` 8000, so likely tolerable; listen and report |
| **Medium** | Universal HiFi-GAN may perform poorly on Telugu or Indian English | Check vocoder-only reconstruction first |
| **Medium** | Stack port changes behavior | Verification list; compare features with upstream's formula |
| **Medium** | Session interruptions lose training | Frequent persistent checkpoints; test resume |
| **Medium** | Overfitting with small data over many epochs | Validation tracking and listening |
| **Medium** | Paper and implementation mismatch questioned in review | Documented deviations |
| **Low** | Energy control bug | Fix in project code; no effect at defaults |
| **Low** | Checkpoints exceed LFS limits | Save model-only final checkpoints; use external storage if needed |
| **Low** | Dataset license limits redistribution in manifests | Check before pushing |

## 8. Recommended architecture decisions

| Decision | Recommendation |
|---|---|
| Base | Upstream `LJSpeech` config: 4-layer encoder, 6-layer decoder |
| Pitch | **Approved by project owner, unless the assignment requires wavelets:** phoneme-level F0, no CWT |
| Energy | Phoneme-level, normalized |
| Post-Net | **Approved by project owner:** keep, with a config switch; report as a deviation |
| Vocoder | Pretrained `universal` HiFi-GAN at 22.05 kHz; check reconstruction first |
| English | Multi-speaker if speaker identity is recovered or confirmed; otherwise no speaker embeddings, as a reported limitation |
| Telugu | **Not finalized.** Single speaker preferred if measurements support it; resample to 22.05 kHz |
| Telugu text | Project rule-based phonemes, with per-language symbol tables |
| Duration | MFA for both languages; Telugu model trained on the corpus |
| Selection | Seeded manifests chosen by duration, in `data/manifests/` |
| Code | `vendor/` unchanged; project code holds fixes |

All of these remain subject to the assignment.

## 9. Recommended implementation sequence

The dated plan, gates and fallbacks are in [seven_day_execution_plan.md](seven_day_execution_plan.md). In brief:

1. **October 31:** resolve blockers.
2. **November 1:** GPU environment, GPU benchmark and dataset inspection.
3. **November 2:** manifests, vocoder check, code copy with fixes, and frontends.
4. **November 3:** alignment and features.
5. **November 4:** overfit test, then start real training.
6. **November 5:** produce every deliverable once from real intermediate checkpoints.
7. **November 6:** freeze, generate final outputs, finish the report and submit by 18:00 IST.
8. **November 7:** reserve only.

Doing Telugu's frontend and alignment before long training is deliberate: errors there invalidate all later training.

## 10. Critical blockers

| Blocker | Owner | Blocks |
|---|---|---|
| Assignment PDF and exact test utterances | You | Final requirement verification and frontend tests |
| GPU environment | You | GPU benchmarking and training |
| Svarah access and terms | You | English inspection and speaker ID check |
| Indic TTS licence review | You, or retry when the site responds | Legitimate Telugu use |

## Audit limitations

- The benchmark covers CPU and synthetic data only, not GPU speed, real data or feature extraction.
- Svarah's card and rows require authentication.
- The IIT Madras license, database and Unified Parser pages timed out, including a retry for the license.
- The Telugu sample of 54 clips was not random, covering clusters at nine offsets.
- CUDA builds for `torch==1.7.0` and Kaggle quotas could not be confirmed from official pages.

## Sources

- [FastSpeech 2 paper](https://arxiv.org/abs/2006.04558) and [v1](https://arxiv.org/abs/2006.04558v1)
- [Svarah on Hugging Face](https://huggingface.co/datasets/ai4bharat/Svarah), [Svarah paper](https://arxiv.org/abs/2305.15760), [AIKosh listing](https://aikosh.indiaai.gov.in/home/datasets/details/svarah.html)
- [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) and [license link from its card](https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf) (unread)
- [MFA acoustic models](https://mfa-models.readthedocs.io/en/latest/acoustic/index.html), [MFA dictionaries](https://mfa-models.readthedocs.io/en/latest/dictionary/index.html), [MFA installation](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html)
- [eSpeak NG languages](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md), [Telugu rules](https://raw.githubusercontent.com/espeak-ng/espeak-ng/master/dictsource/te_list)
- [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library)
- [Unified Parser paper](https://www.iitm.ac.in/donlab/indictts/downloads/unified.pdf)
- [Prakash et al. on 13 Indian languages](https://arxiv.org/abs/2210.17153)
- [PyPI torch 1.7.0](https://pypi.org/project/torch/1.7.0/), [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)

## Decisions requiring your input

In priority order:

Resolved: F0 pitch with a switchable Post-Net (conditional on the assignment), asking the assignment owner about Svarah speaker IDs, and the synthetic benchmark. Telugu speaker choice is deferred to measurements.

Still needed, in priority order:

1. **Provide the five test utterances per language verbatim, and the PDF.** The frontends cannot be tested without them.
2. **Choose the GPU today.** Recommended: an institutional GPU if available, otherwise Kaggle with a paid GPU as backup.
3. **Send the assignment owner one message** covering Svarah speaker metadata, training without embeddings, and the deadline's exact time and time zone.
4. **Accept Svarah's terms and read the Indic TTS license.** Both need your own account or agreement.
5. **Approve a paid GPU budget** in case Sunday's measurement or Kaggle's quota falls short.
6. **Approve the public push on Friday,** after the content review. I will not push before you approve it.
