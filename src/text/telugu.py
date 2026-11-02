"""Rule-based Telugu text normalization and grapheme-to-phoneme conversion.

Each consonant carries the inherent vowel unless a vowel sign or virama follows.
The rules are a project design, checked by tests but not against a reference
lexicon. Nothing is removed silently: every removal or replacement is recorded in
FrontendResult.issues, and strict=True (training data) rejects text containing
anything without a defined pronunciation.
"""
import re
import unicodedata

from .frontend import FrontendResult
from .vocab import PAD, SILENCES

VOWELS = {
    "అ": "a", "ఆ": "aa", "ఇ": "i", "ఈ": "ii", "ఉ": "u", "ఊ": "uu", "ఋ": "ru", "ౠ": "ruu",
    "ఌ": "lu", "ౡ": "luu", "ఎ": "e", "ఏ": "ee", "ఐ": "ai", "ఒ": "o", "ఓ": "oo", "ఔ": "au",
}
MATRAS = {
    "ా": "aa", "ి": "i", "ీ": "ii", "ు": "u", "ూ": "uu", "ృ": "ru", "ౄ": "ruu", "ౢ": "lu",
    "ౣ": "luu", "ె": "e", "ే": "ee", "ై": "ai", "ొ": "o", "ో": "oo", "ౌ": "au",
}
CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "c", "ఛ": "ch", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tt", "ఠ": "tth", "డ": "dd", "ఢ": "ddh", "ణ": "nn",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "rr", "ల": "l", "ళ": "ll", "ఴ": "zh", "వ": "v",
    "శ": "sh", "ష": "ss", "స": "s", "హ": "h", "ౘ": "ts", "ౙ": "dz",
}
VIRAMA = "్"
ANUSVARA = "ం"
HALF_ANUSVARA = "ఁ"
VISARGA = "ః"
AVAGRAHA = "ఽ"
LENGTH_MARK = "ౕ"

NASAL_FOR = {}
for _nasal, _group in {
    "ng": "k kh g gh",
    "ny": "c ch j jh",
    "nn": "tt tth dd ddh",
    "n": "t th d dh",
    "m": "p ph b bh",
}.items():
    for _c in _group.split():
        NASAL_FOR[_c] = _nasal

# Used for digits outside a recognized number, such as non-ASCII digit forms.
DIGIT_WORDS = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది"]
LATIN_NAMES = dict(zip(
    "abcdefghijklmnopqrstuvwxyz",
    "ఏ బీ సీ డీ ఈ ఎఫ్ జీ హెచ్ ఐ జే కే ఎల్ ఎం ఎన్ ఓ పీ క్యూ ఆర్ ఎస్ టీ యూ వీ డబ్ల్యూ ఎక్స్ వై జెడ్".split(),
))
PAUSE_MARKS = set(",.;:!?।॥…—–-()[]{}\"'‘’“”/")
INVISIBLE = {"‌", "‍", "​", "﻿", "­"}
_SIGNS = (VIRAMA, ANUSVARA, HALF_ANUSVARA, VISARGA, AVAGRAHA, LENGTH_MARK)


_ONES = ["", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది", "పది",
         "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పందొమ్మిది"]
_TENS = ["", "", "ఇరవై", "ముప్పై", "నలభై", "యాభై", "అరవై", "డెబ్భై", "ఎనభై", "తొంభై"]
_HUNDREDS = ["", "నూట", "రెండు వందల", "మూడు వందల", "నాలుగు వందల", "ఐదు వందల", "ఆరు వందల", "ఏడు వందల", "ఎనిమిది వందల", "తొమ్మిది వందల"]
_EXACT_HUNDREDS = {1: "వంద"}
_THOUSANDS = ["", "వెయ్యి", "రెండు వేల", "మూడు వేల", "నాలుగు వేల", "ఐదు వేల", "ఆరు వేల", "ఏడు వేల", "ఎనిమిది వేల", "తొమ్మిది వేల"]
_ONE_THOUSAND_PREFIX = "వెయ్యిన్ని"
_TIME_RE = re.compile(r"(?<![0-9])([0-9]{1,2}):([0-9]{2})(?![0-9])")
_ORDINAL_RE = re.compile(r"(?<![0-9])([0-9]{1,4})వ(?![\u0c00-\u0c7f])")
_NUMBER_RE = re.compile(r"(?<![0-9])[0-9]{1,4}(?![0-9])")


def number_to_words(value):
    """Read 0 to 9999 as Telugu words. Longer digit runs are read digit by digit elsewhere."""
    if value == 0:
        return DIGIT_WORDS[0]
    words = []
    thousands, rest = value // 1000, value % 1000
    if thousands:
        words.append(_ONE_THOUSAND_PREFIX if thousands == 1 and rest else _THOUSANDS[thousands])
    hundreds, rest = rest // 100, rest % 100
    if hundreds:
        words.append(_EXACT_HUNDREDS[hundreds] if hundreds == 1 and not rest else _HUNDREDS[hundreds])
    if rest < 20:
        if rest:
            words.append(_ONES[rest])
    else:
        words.append(_TENS[rest // 10])
        if rest % 10:
            words.append(_ONES[rest % 10])
    return " ".join(words)


def expand_numbers(text, issues):
    """Replace ASCII numbers with words in the working copy; each change is logged."""

    def time(match):
        words = number_to_words(int(match.group(1))) + " " + number_to_words(int(match.group(2)))
        issues.append("read time {!r} as {}".format(match.group(0), words))
        return " " + words + " "

    def ordinal(match):
        words = number_to_words(int(match.group(1)))
        if words.endswith("\u0c41"):
            words = words[:-1]  # the final vowel sign elides before the ordinal suffix
        words += "వ"
        issues.append("read ordinal {!r} as {}".format(match.group(0), words))
        return " " + words + " "

    def number(match):
        words = number_to_words(int(match.group(0)))
        issues.append("read number {!r} as {}".format(match.group(0), words))
        return " " + words + " "

    return _NUMBER_RE.sub(number, _ORDINAL_RE.sub(ordinal, _TIME_RE.sub(time, text)))


def phone_symbols():
    phones = set(VOWELS.values()) | set(CONSONANTS.values()) | set(NASAL_FOR)
    return [PAD] + sorted(phones) + SILENCES


def word_to_phones(word, lenient=False):
    """Convert one cleaned Telugu word.

    A vowel sign with no consonant before it is unusual. With lenient=True it is read as
    its vowel and a stray virama has no sound; the caller logs this. Otherwise it is an error.
    """
    phones = []
    i = 0
    while i < len(word):
        ch = word[i]
        following = word[i + 1] if i + 1 < len(word) else ""
        if ch in CONSONANTS:
            phones.append(CONSONANTS[ch])
            if following == VIRAMA:
                i += 1
            elif following in MATRAS:
                phones.append(MATRAS[following])
                i += 1
            elif following == LENGTH_MARK:
                phones.append("aa")
                i += 1
            else:
                phones.append("a")
        elif ch in VOWELS:
            phones.append(VOWELS[ch])
        elif ch in (ANUSVARA, HALF_ANUSVARA):
            phones.append(NASAL_FOR.get(CONSONANTS.get(following), "m"))
        elif ch == VISARGA:
            phones.append("h")
        elif ch == AVAGRAHA:
            pass  # marks elision and has no sound of its own
        elif lenient and ch in MATRAS:
            phones.append(MATRAS[ch])
        elif lenient and ch in (VIRAMA, LENGTH_MARK):
            pass
        else:
            raise ValueError(
                "misplaced sign U+{:04X} in {!r}".format(ord(ch), word)
            )
        i += 1
    return phones


def _is_telugu_letter(ch):
    return ch in VOWELS or ch in CONSONANTS or ch in MATRAS or ch in _SIGNS


def _digit_value(ch):
    try:
        return unicodedata.digit(ch)
    except ValueError:
        return None


def tokenize(text, issues, strict):
    """Split into words and pauses (None), expanding digits and Latin letters."""
    tokens, word = [], []

    def flush():
        if word:
            tokens.append("".join(word))
            word.clear()

    for ch in text:
        if _is_telugu_letter(ch):
            word.append(ch)
            continue
        if ch.isspace():
            flush()
            continue
        if ch in INVISIBLE:
            issues.append("removed invisible character U+{:04X}".format(ord(ch)))
            continue
        flush()
        if ch in PAUSE_MARKS:
            tokens.append(None)
            continue
        digit = _digit_value(ch)
        if digit is not None:
            issues.append("read digit {!r} as {}".format(ch, DIGIT_WORDS[digit]))
            tokens.append(DIGIT_WORDS[digit])
        elif ch.lower() in LATIN_NAMES:
            issues.append("read Latin letter {!r} by its name".format(ch))
            tokens.append(LATIN_NAMES[ch.lower()])
        else:
            issue = "no pronunciation for {!r} (U+{:04X})".format(ch, ord(ch))
            if strict:
                raise ValueError(issue)
            issues.append(issue + "; replaced with a pause")
            tokens.append(None)
    flush()
    return tokens


def process(text, strict=False):
    """Text to phones. strict=True raises on text with no defined pronunciation."""
    result = FrontendResult(original=text)
    normalized = unicodedata.normalize("NFC", text)
    if normalized != text:
        result.issues.append("applied Unicode NFC normalization")
    if re.search(r"\S\s{2,}\S", normalized):
        result.issues.append("a run of spaces is treated as one word break, with no added pause")
    working = expand_numbers(normalized, result.issues)
    result.normalized = working

    for token in tokenize(working, result.issues, strict):
        if token is None:
            if result.phones and result.phones[-1] != "sp":
                result.phones.append("sp")
            continue
        try:
            phones = word_to_phones(token)
        except ValueError as error:
            if strict:
                raise
            phones = word_to_phones(token, lenient=True)
            result.issues.append("{}; read the sign on its own".format(error))
        result.words.append((token, phones))
        result.phones.extend(phones)
    while result.phones and result.phones[-1] == "sp":
        result.phones.pop()
    return result
