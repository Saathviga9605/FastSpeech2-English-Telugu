# IndicTTS Telugu

- **Official URL:** https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu
- **Planned role:** Telugu dataset for this project
- **Status:** used for training

Do not commit dataset files or Hugging Face tokens to Git.

## Hub metadata (API, checked 2026-10-31)

| Field | Value |
|---|---|
| Visibility | Public |
| Gated | No |
| Last modified | 2025-01-27 |
| License in card metadata | None declared: check the dataset card and record the terms |

**Warning:** The card's duration and sample rate conflict with public statistics and sampled WAV headers. Mean clip length is about 17 seconds, and estimated total duration is about 34 to 36 hours rather than 8.74. Sampled female clips are 16 kHz and male clips 48 kHz. See the [pre-implementation audit](../implementation/pre_implementation_audit.md).

## License

The dataset card says use is subject to the IIT Madras License For Use of Indic TTS. The license PDF could not be downloaded on 2026-11-03 (connection timed out), so its terms have not been read here. See `third_party_licenses/IndicTTS_NOTICE.md`.

## To verify during inspection

- License and terms of use
- Number of speakers, utterances and hours
- Sample rate, audio format and recording quality
- Script, transcript quality and normalization
- Any metadata that would help with phonemization or alignment
- Dataset revision hash to pin for reproducibility
