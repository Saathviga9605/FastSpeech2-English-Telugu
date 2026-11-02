"""Write the separate English and Telugu symbol tables to configs/."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.text import telugu  # noqa: E402
from src.text.cmudict import valid_symbols  # noqa: E402
from src.text.vocab import PAD, SILENCES  # noqa: E402

TABLES = {
    "telugu": telugu.phone_symbols(),
    "english": [PAD] + sorted(valid_symbols) + SILENCES,
}
for language, symbols in TABLES.items():
    path = ROOT / language.capitalize() / "symbols.json"
    path.write_text(json.dumps(symbols, indent=0), encoding="utf-8")
    print(language, len(symbols))
