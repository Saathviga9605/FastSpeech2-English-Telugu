# Requirement checklist

| Requirement | Where | Status |
|---|---|---|
| Separate English and Telugu models | `English/`, `Telugu/` | Done |
| Changes documented, especially Telugu | Report section 1, `docs/implementation/` | Done |
| About 5 h train and 1 h validation, no overlap | Report section 2 | Selected 5 + 1 h; less after alignment and trimming |
| Test sentences preserved | `data/test_utterances/sentences.json` | Done |
| Code, configs and checkpoints per language | `English/`, `Telugu/`, `src/` | Done; checkpoints use Git LFS |
| Loss plots | `Results/*/training_loss.png`, `validation_loss.png` | Done |
| Audio for all test sentences | `Results/*/test_generated_audios/` | Done |
| Duration, pitch and energy plots per sentence | `Results/*/variance_adaptor_plots/` | Done |
| Technical report | `Report/Technical_Report.pdf` | Done; quality notes are overall, not per clip |
| Public repository and submitted URL | GitHub | Pending |
