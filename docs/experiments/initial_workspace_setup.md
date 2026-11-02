# Initial Workspace Setup Report

**Date:** 2026-10-31
**Status:** Complete, except for the assignment PDF, which must be added manually.

**No model implementation, dataset download, preprocessing or training has started.** No packages were installed and no system-level changes were made.

## Directories created

All directories in the requested structure were created, with `.gitkeep` files in those otherwise empty so Git can track them. The upstream clone is the exception: it has no `.gitkeep` and is ignored by this repository.

## Files downloaded

| File | Source | Verification |
|---|---|---|
| `docs/research_paper/FastSpeech2_Original_Paper.pdf` | https://arxiv.org/pdf/2006.04558 | 1,273,673 bytes; `%PDF-1.5` header and `%%EOF` trailer present; all 167 compressed streams decompress; identifier inside the file is `arXiv:2006.04558v8` |

SHA-256: `cc63cbd90dbc90107603cd064969f23a9b6b00b986c3bda3c27dc63613147e2a`

No PDF viewer or parser was available, so these are structural checks. The page was not rendered visually.

## Git repository status

- Project repository initialized on branch `main`, with the initial commit created using the configured Git identity. The commit containing this report is the first entry in `git log`.
- No remote was added and nothing was pushed.
- Git LFS hooks were enabled for this repository only (`git lfs install --local`).
- `vendor/FastSpeech2/` is ignored and is not a submodule or gitlink.

## Upstream

| Field | Value |
|---|---|
| Repository | https://github.com/ming024/FastSpeech2 |
| Branch | `master` |
| Commit | `d4e79eb52e8b01d24703b2dfc0385544092958f3` |
| License | MIT; bundled HiFi-GAN is also MIT |
| Working tree | Clean, with no modifications |

Full details: [upstream_provenance.md](../implementation/upstream_provenance.md).

## Environment

Windows 11 Pro, Python 3.13.14, Git 2.55.0, 15.3 GB RAM, about 284 GB free on D:, AMD integrated graphics, PyTorch 2.14.0+cpu, no CUDA. Full details: [environment_audit.md](../implementation/environment_audit.md).

## Dependencies found

Upstream pins 17 packages, including `torch==1.7.0`, `numpy==1.19.0` and `librosa==0.7.2`, and does not declare a Python version. `requirements.txt` is deliberately pending, with all pins commented out. Full details: [dependency_audit.md](../implementation/dependency_audit.md).

## Missing tools

- NVIDIA GPU and CUDA
- FFmpeg
- espeak-ng
- Montreal Forced Aligner
- Hugging Face CLI (the Python library is present)
- Conda

Git LFS is installed.

## Failed operations

| Operation | Result | Effect |
|---|---|---|
| Copy assignment PDF | Not in the workspace, and no attachment was available in this session | `docs/assignment/README.md` created instead |
| Render the paper's first page | Failed because `pdftoppm` is not installed | Replaced with the structural checks above |

## Verification

| Check | Result |
|---|---|
| Required directories and files | Present |
| Paper PDF | Passed structural checks |
| Upstream repository and license | Commit matches; working tree clean; MIT license read |
| README, .gitignore | Present; ignore rules tested with sample paths |
| Deliverable paths | Checkpoints, audio, plots and report paths are not ignored; checkpoints resolve to LFS |
| Excluded paths | Raw and processed data, `.env`, token file, logs, intermediate checkpoints and vendor clone are ignored |
| Staged content | No datasets, audio, checkpoints or gitlinks; token pattern scan found nothing |

## Outstanding manual actions

1. Add the assignment PDF at `docs/assignment/IIITH_FastSpeech2_Assignment.pdf`.
2. Choose a GPU environment for training.
3. Accept the Svarah access terms on Hugging Face and log in before the dataset phase. Do not store the token in this repository.
4. Confirm that the configured Git identity is the one to use for this project.

## Open decisions for the next phase

- **Which FastSpeech 2 to follow:** The saved paper is arXiv v8, while upstream says it is closer to v1. Upstream also adds a Post-Net that its README says is not in the original. The assignment should settle this.
- **Python and library versions:** The upstream pins need a compatible older environment, or the code must be tested on newer libraries.
