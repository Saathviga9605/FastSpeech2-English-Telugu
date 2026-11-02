# Telugu Feasibility

Audit date: 2026-10-31. No tool was installed or run on Telugu text, so nothing here proves quality. It records documented support and a plan for testing it.

## Recommendation

- **Text representation:** phonemes from a project-owned, rule-based Telugu grapheme-to-phoneme table, with Indic NLP's `TeluguNormalizer` applied first.
- **Duration supervision:** train a Montreal Forced Aligner (MFA) acoustic model from scratch on the selected Telugu corpus, using a lexicon generated from those rules.
- **Validation under the deadline:** before any Telugu training, hand-check boundaries on 20 utterances, and compare the rules on 200 words with whichever of Unified Parser or eSpeak NG installs within 1 hour.
- **Speaker:** not finalized. Decide after measuring usable duration and speaker consistency (see [the audit](pre_implementation_audit.md)).
- **Time limit:** alignment must pass by Tuesday, November 3, under the [seven-day plan](seven_day_execution_plan.md).

This keeps the paper's MFA-based duration approach and needs no pretrained Telugu assets, because none were found.

## Tool support

| Tool | Documented support | What is not established | Role |
|---|---|---|---|
| **Indic NLP Library** (MIT) | `te` Unicode range; `TeluguNormalizer` for poorna virama, vowel-sign canonicalization (U+0C46 U+0C56 to U+0C48) and colon to visarga; tokenizer and syllabifier modules listed | No G2P module is listed. Per-operation Telugu quality untested | Normalization |
| **eSpeak NG** | `te` listed in `languages.md`; rules in `dictsource/te_list`. A 2013 report says Telugu phonemes were inherited from Hindi and later refined | No quality rating found. Current inherent-vowel handling unverified. Not installed | Comparison only |
| **Unified Parser** (IIT Madras) | Text to common label set with language rules; applied to Telugu in a 13-language FastSpeech 2 study | Telugu rules not read. Official download requires accepting a license. The site timed out. A PyPI package and community repository exist but were not checked | Comparison, or replacement if the licence allows |
| **MFA** | Can train on your own corpus and dictionary. Conda is the main install route; Docker and pip are also documented | No Telugu run tested. Windows support is not stated outright | Alignment |
| **MFA pretrained models** | **No Telugu acoustic model or dictionary listed.** Tamil has acoustic and dictionary models; Hindi has a dictionary only | Whether a Tamil model could transfer. Not recommended: different phone set and unknown fit | Not used |

## Why a project-owned frontend

- It is deterministic, testable and fully documented, as the project rules require.
- Sources describe Dravidian languages as mostly written as spoken, with schwa deletion applied to the Indo-Aryan group. This suggests rules may be practical for Telugu. **It is a hypothesis to validate, not a finding.**
- The rule table can be checked against two independent tools.

## Upstream changes required

All go in project code, not `vendor/`.

| Issue | Location | Change |
|---|---|---|
| No Telugu symbols | `text/symbols.py` | Per-language symbol tables, because the table is global and also sets the embedding size |
| **Unknown symbols silently dropped** (confirmed by benchmark) | `text/__init__.py:67`, `:75` | Raise an error for training data. At inference, apply a defined, logged policy, because the test set includes malformed input |
| No Telugu inference path | `synthesize.py:205-208` handles only `en` and `zh` | Add a `te` frontend |
| No UTF-8 on text reads | `preprocessor/preprocessor.py:178`, `synthesize.py:22` | Set `encoding="utf-8"`; otherwise this may fail or corrupt text on Windows |
| Missing alignment handling | `preprocessor/preprocessor.py:85-90` | Fix, because Telugu out-of-vocabulary items make missing alignments more likely |
| English cleaners | config `text_cleaners` | Add a Telugu cleaner |

## Evidence from the transcripts

From the first 100 public rows only:

- No Latin letters or digits appeared. The only non-Telugu characters were spaces, commas and periods. **The full corpus still needs a scan.**
- Long story passages run together with little punctuation, and some commas look misplaced. Pause placement therefore cannot rely on punctuation, so MFA's optional silences need to be inspected.
- Normalization for numbers, abbreviations and Latin text is needed if the test sentences contain them. Those sentences are not available yet, so this cannot be scoped.

## Dataset points affecting Telugu

- **Long clips:** the sampled female clips reach about 79 seconds, while upstream's `max_seq_len` of 1000 frames is about 11.6 seconds. Upstream silently truncates longer input in training (`transformer/Models.py`, decoder forward). Filter by duration.
- **Mixed sample rates:** sampled female clips are 16 kHz and male clips 48 kHz, despite the card stating 48 kHz.
- **Precedent:** [Prakash et al.](https://arxiv.org/abs/2210.17153) built Telugu FastSpeech 2 systems from about 10 hours per speaker from Indic TTS. Their alignment figures are from Hindi and do not transfer directly. Their HMM-GD-DNN method is a possible improvement, not part of this plan.

## Validation gates

1. **Normalization:** the same text gives the same output, with tests for vowel signs and the virama.
2. **Rules:** record agreement and every disagreement with Unified Parser and eSpeak NG on a sample of at least 200 words.
3. **Coverage:** every character in the selected corpus and test sentences maps to a symbol, or an error is raised.
4. **Alignment:** report the fraction of utterances aligned and out-of-vocabulary items, then inspect boundaries by hand.
5. **Features:** spot-check phoneme-level pitch and energy against the audio.

## Fallbacks

Listed in order, each to be used only if the gate before it fails:

1. Use Unified Parser labels, if its licence permits.
2. Tune MFA settings and exclude badly aligned utterances, provided the required duration is still met.

A learned aligner, such as [the NVIDIA method](https://arxiv.org/abs/2108.10447), is out of scope for this deadline because it needs new model code.
