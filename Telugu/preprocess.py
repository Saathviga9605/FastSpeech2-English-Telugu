"""Telugu preprocess. See README.md for usage."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import preprocess  # noqa: E402

if __name__ == "__main__":
    preprocess("telugu")
