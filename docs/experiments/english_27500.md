# English run, submitted checkpoint step 27,500

- **Config:** `English/config.yaml`, seed 1234. Planned 50,000 steps.
- **Upstream:** ming024/FastSpeech2 `d4e79eb`.
- **Code commit:** not recorded in the run logs. Commit history was consolidated before publication, so the training commit is not in this repository.
- **Dataset:** Svarah, 3,499 train and 689 validation clips after alignment and feature checks. See `data/processed/english/report.json` (not in Git). Dataset revision hash: not recorded; this is a known reproducibility gap.
- **Environment:** Google Colab, T4 GPU, about 1.5 steps/s at batch 16.
- **Stopped:** Colab usage limit at about step 28,400. Last saved checkpoint is 27,500; later steps were lost. Not resumed, because validation had stopped improving and the GPU quota is needed for Telugu.
- **Exported:** `English/checkpoints/final.pth.tar`, step 27,500.
- **Generalization check:** `english_27500_generalization.json`, from `scripts/audit_generalization.py`, run on CPU over 300 clips per split.
- **Audio and plots:** synthesized on CPU from the step 27,500 checkpoint on 2026-11-02.
- **Listening notes:** from the owner, in section 5 of the report. Informal, one listener.

## Measured

Validation mel was lowest at step 7,000 (0.847) and 0.862 at step 28,000. On validation clips, pitch MSE is 1.151 against 1.079 for a constant, so the pitch predictor does not generalize.
