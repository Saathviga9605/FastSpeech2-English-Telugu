"""Checks on individual dataset rows. They report problems and never change the data."""
import io
import unicodedata

import soundfile as sf

TELUGU = (0x0C00, 0x0C7F)
MOJIBAKE = ("à°", "à±", "Ã", "â€")
SPEAKER_HINTS = ("speaker", "spk", "client", "voice", "person", "user")


def text_problem(text, language):
    """Return why a transcript is unusable, or None. Only unusable text is flagged here."""
    if text is None:
        return "missing transcript"
    if not isinstance(text, str):
        return "transcript has type {}".format(type(text).__name__)
    if not text.strip():
        return "empty transcript"
    if "�" in text:
        return "contains the replacement character U+FFFD"
    if any(0xD800 <= ord(c) <= 0xDFFF for c in text):
        return "contains a lone surrogate"
    if any(marker in text for marker in MOJIBAKE):
        return "looks wrongly decoded"
    if language == "telugu" and not any(TELUGU[0] <= ord(c) <= TELUGU[1] for c in text):
        return "contains no Telugu characters"
    return None


def text_notes(text):
    """Harmless properties worth counting, such as text that is not in NFC form."""
    notes = []
    if unicodedata.normalize("NFC", text) != text:
        notes.append("not_nfc")
    if any(unicodedata.category(c) == "Cc" and c not in "\t\n\r" for c in text):
        notes.append("control_character")
    if any(unicodedata.category(c) == "Cf" for c in text):
        notes.append("format_character")
    return notes


def audio_properties(cell):
    """Read audio properties from a Parquet audio cell. Raises ValueError if unusable."""
    if cell is None or not cell.get("bytes"):
        raise ValueError("missing audio")
    try:
        info = sf.info(io.BytesIO(cell["bytes"]))
    except Exception as error:  # soundfile raises several types for damaged files
        raise ValueError("unreadable audio: {}".format(type(error).__name__)) from None
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("audio has no samples")
    return {
        "seconds": info.frames / info.samplerate,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "format": info.format,
    }


def describe_feature(feature):
    return str(feature)


def speaker_columns(column_names):
    """Columns whose names suggest speaker identity. Candidates only, never assumed correct."""
    return [c for c in column_names if any(hint in c.lower() for hint in SPEAKER_HINTS)]
