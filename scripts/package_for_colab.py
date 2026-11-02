"""Zip all committed files, so the code can reach Colab through Drive before any GitHub push."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "FastSpeech2-IIITH-code.zip"
OUT.parent.mkdir(exist_ok=True)
subprocess.run(["git", "archive", "--format=zip", "-o", str(OUT), "HEAD"], cwd=ROOT, check=True)
print("Upload to Google Drive folder MyDrive/FastSpeech2-IIITH/:", OUT)
