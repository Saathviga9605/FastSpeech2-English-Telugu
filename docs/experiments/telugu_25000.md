# Telugu run, submitted checkpoint step 25,000

- **Config:** `Telugu/config.yaml`, seed 1234. Planned 50,000 steps.
- **Upstream:** ming024/FastSpeech2 `d4e79eb`.
- **Code commit:** not recorded in the run logs. Commit history was consolidated before publication, so the training commit is not in this repository.
- **Dataset:** IndicTTS Telugu, 2,524 train and 507 validation clips. See `data/processed/telugu/report.json` (not in Git). Dataset revision hash: not recorded; this is a known reproducibility gap.
- **Environment:** Google Colab, T4 GPU, about 1.5 steps/s at batch 16.
- **Stopped:** at about step 25,200, at the Colab usage limit, as reported by the owner. Last saved checkpoint is 25,000; later steps were lost. This is 2,500 steps fewer than English.
- **Exported:** `Telugu/checkpoints/final.pth.tar`, step 25,000.
- **Generalization check:** `telugu_25000_generalization.json`, from `scripts/audit_generalization.py`, run on CPU over 300 clips per split.
- **Audio and plots:** synthesized on CPU from the step 25,000 checkpoint on 2026-11-02.
- **Listening notes:** from the owner, in section 5 of the report. Informal, one listener.

## Measured

Validation mel was lowest at step 12,000 (0.576) and 0.582 at step 25,000. On validation clips, pitch, energy and duration all beat the per-phone baseline. Every `ny` in the test sentences is predicted at one frame (12 ms), reproducing the known alignment defect.
