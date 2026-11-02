# Environment Audit

Audited on 2026-10-31 with read-only commands. Nothing was installed or reconfigured, and no tokens or secret environment variables were read.

## Key finding

**This machine has no NVIDIA GPU and no CUDA, so it is not suited to full training.** Upstream reports needing a GTX 1080 Ti for less than an hour to reach about 10k steps, and its pretrained checkpoints are at 600k to 900k steps. CPU training time here has not been measured.

## System

| Item | Result |
|---|---|
| Operating system | Windows 11 Pro, version 10.0.26300, build 26300, 64-bit |
| CPU | AMD Ryzen 7 PRO 5850U (8 cores, 16 threads) |
| RAM | 15.3 GB total, 5.5 GB free at audit time |
| Disk | D: 293.0 GB total, 284.2 GB free before the upstream clone; C: 182.0 GB total, 62.4 GB free |
| GPU | AMD Radeon Graphics (integrated), driver 31.0.21923.3003 |
| NVIDIA GPU and driver | None found; `nvidia-smi` not available |
| CUDA | Not available |

## Software

| Tool | Result |
|---|---|
| Python | 3.13.14 (Microsoft Store install) |
| pip | 26.1.2 |
| PyTorch | 2.14.0+cpu, `torch.cuda.is_available()` is False |
| torchaudio | 2.11.0+cpu |
| TensorFlow | Not installed (upstream code does not import it) |
| Git | 2.55.0.windows.3; global identity is configured |
| Git LFS | Installed, 3.7.1 |
| FFmpeg | **Not found** |
| espeak-ng | **Not found** |
| Montreal Forced Aligner | **Not found** |
| Hugging Face CLI | **Not found** (`huggingface-cli` and `hf`); the `huggingface_hub` 0.36.2 Python library is installed |
| Conda | **Not found** |

Package details are in the [dependency audit](dependency_audit.md).

## Actions for later phases

- Choose a GPU environment for training. This environment will need its own audit.
- Install FFmpeg, the Hugging Face CLI and the aligner in that environment as needed.
- Decide whether espeak-ng is needed, based on the frontend research.
