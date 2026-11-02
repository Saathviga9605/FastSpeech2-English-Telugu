# Artifact Tracking Strategy

| Artifact | Location | Tracking |
|---|---|---|
| Final checkpoints | `English/checkpoints/`, `Telugu/checkpoints/` | Git LFS through `.gitattributes` |
| Intermediate checkpoints | `output/` | Ignored |
| Test audio and plots | `Results/` | Regular Git |
| Report | `Report/` | Regular Git; LaTeX build files ignored |
| Manifests | `data/manifests/` | Regular Git |
| Raw, interim and processed data | `data/raw/`, `data/interim/`, `data/processed/` | Ignored |
| Training logs | `logs/` | Ignored; summarize key results in `docs/experiments/` |

## Before pushing

1. Check the size of the final checkpoints. Git LFS storage and bandwidth limits depend on the host and account, so confirm them first.
2. If the checkpoints exceed those limits, or the assignment requires a specific submission method, keep them in external storage. Then add a `CHECKPOINTS.md` beside them with the URL, SHA-256 hash, training commit and config.
3. Run `git lfs ls-files` to confirm the checkpoints are stored as LFS objects.
4. Check that manifests contain only what the dataset's license permits you to redistribute.
5. The arXiv paper PDF is tracked for local reference. Check its redistribution terms before making the repository public.
