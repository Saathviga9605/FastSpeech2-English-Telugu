# Architecture Research Checklist

These are topics to investigate. Nothing here is a conclusion. Check each item against the paper and the upstream code (commit `d4e79eb`), and cite the source when closing it.

## Model

- [ ] **FastSpeech 2 architecture:** How does the overall design differ between arXiv v1 and v8? Which version should this project follow?
- [ ] **Transformer encoder:** Layers, heads, dimensions and positional encoding in the paper and upstream
- [ ] **Variance adaptor:** Order of predictors, conditioning and how embeddings are added
- [ ] **Duration predictor:** Targets, log scale, and the role of alignment
- [ ] **Pitch predictor:** F0 or CWT features, phoneme or frame level, normalization, quantization
- [ ] **Energy predictor:** Definition, level, normalization, quantization
- [ ] **Length regulator:** Expansion behavior and duration control
- [ ] **Transformer decoder:** Structure and differences from the encoder
- [ ] **Post-Net:** Upstream's README says it is not used in the original. Should it stay, and what does the assignment require?
- [ ] **Mel-spectrogram generation:** Which parameters must match the vocoder?
- [ ] **HiFi-GAN vocoder:** Are the bundled generators suitable for each dataset's sample rate and for Telugu? Is fine-tuning needed?

## Training and inference

- [ ] **Training losses:** Terms, weights and masking in the paper and upstream
- [ ] **Ground-truth variance information:** How targets are extracted and used during training
- [ ] **Inference-time variance prediction:** How predictions replace targets, and how control ratios work

## Language frontends and data

- [ ] **English phonemization:** Are upstream's `g2p_en` and lexicon suitable for the English dataset's speech and transcripts?
- [ ] **Telugu grapheme-to-phoneme conversion:** Which options exist, what phoneme set should be used, and how can it be validated?
- [ ] **Telugu forced alignment:** Which aligner, acoustic model and lexicon are available, or must a model be trained? How will quality be checked?
- [ ] **Dataset compatibility:** Sample rate, speakers, transcripts, licenses, and the changes needed for upstream's preprocessing

## Observed so far (facts, not decisions)

- The upstream LJSpeech config and bundled HiFi-GAN config both use 22050 Hz, 80 mel channels, FFT 1024, hop 256 and window 1024.
- The upstream LJSpeech config sets pitch and energy to `phoneme_level` with normalization on.
