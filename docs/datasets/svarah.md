# Svarah (English)

- **Official URL:** https://huggingface.co/datasets/ai4bharat/Svarah
- **Planned role:** English dataset for this project
- **Status:** not downloaded or inspected

Downloading and inspection will happen in a later phase. Do not commit dataset files or Hugging Face tokens to Git.

## Hub metadata (API, checked 2026-10-31)

| Field | Value |
|---|---|
| Visibility | Public |
| Gated | `auto`: expect to log in and accept the access terms before downloading |
| Last modified | 2025-03-10 |
| License in card metadata | None declared: check the dataset card and record the terms |

**Warning:** The public schema has no speaker ID column, and there is only a `test` split. With about 4.9 minutes per speaker on average, single-speaker training is not realistic. See the [pre-implementation audit](../implementation/pre_implementation_audit.md).

## To verify during inspection

- License and terms of use
- Number of speakers, utterances and hours
- Sample rate, audio format and recording quality
- Transcript quality and normalization
- Suitability for TTS training, including how to handle multiple speakers
- Dataset revision hash to pin for reproducibility
