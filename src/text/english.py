"""English text frontend: upstream's lexicon-then-g2p_en approach with ARPAbet phones.

Changes from upstream: homographs are resolved from the sentence's part-of-speech
tags instead of always taking the lexicon's first entry, and input the frontend
cannot read is reported rather than passed on or dropped silently.
"""
import re

from .cleaners import english_cleaners
from .cmudict import valid_symbols
from .frontend import FrontendResult
from .vocab import PAD, SILENCES

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")
_PAUSE_RE = re.compile(r"[,;:.!?()\-\"]")
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*|\S")
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_MERIDIEM_RE = re.compile(r"(\d|\b)\s*\b([AP])\.?M\b\.?")


def expand_time(text):
    """Read 10:45 AM as ten forty five A M, before the general number rules run."""
    text = _MERIDIEM_RE.sub(lambda m: "{} {} M".format(m.group(1), m.group(2)), text)

    def time(match):
        minutes = match.group(2)
        if minutes == "00":
            return match.group(1)
        return "{} {}{}".format(match.group(1), "oh " if minutes[0] == "0" else "", int(minutes))

    return _TIME_RE.sub(time, text)


def phone_symbols():
    return [PAD] + sorted(valid_symbols) + SILENCES


def read_lexicon(path):
    lexicon = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[0].lower() not in lexicon:
                lexicon[parts[0].lower()] = parts[1:]
    return lexicon


class EnglishFrontend:
    def __init__(self, lexicon_path):
        self.lexicon = read_lexicon(lexicon_path)
        self._g2p = None
        self.valid = set(valid_symbols)

    @property
    def g2p(self):
        if self._g2p is None:
            import nltk

            for package in ("averaged_perceptron_tagger_eng", "cmudict"):
                nltk.download(package, quiet=True)
            from g2p_en import G2p

            self._g2p = G2p()
        return self._g2p

    def oov_phones(self, word):
        phones = [p for p in self.g2p(word) if p in self.valid]
        if not phones:
            raise ValueError("g2p_en gave no phones for {!r}".format(word))
        return phones

    def _homograph_choice(self, tokens, index):
        table = self.g2p.homograph2features
        word = tokens[index]
        if word not in table:
            return None
        import nltk

        tag = nltk.pos_tag(tokens)[index][1]
        first, second, pos = table[word]
        return list(first if tag.startswith(pos) else second)

    def process(self, text, strict=False, resolve_homographs=True):
        result = FrontendResult(original=text)
        timed = expand_time(text)
        if timed != text:
            result.issues.append("read time expression: {!r}".format(timed))
        cleaned = english_cleaners(timed)  # unidecode, lowercase, number expansion
        if cleaned != timed:
            result.issues.append("normalized for pronunciation: {!r}".format(cleaned))
        if "  " in text:
            result.issues.append("a run of spaces is treated as one word break, with no added pause")
        result.normalized = cleaned
        tokens = _TOKEN_RE.findall(cleaned)
        for index, token in enumerate(tokens):
            if _PAUSE_RE.fullmatch(token):
                if result.phones and result.phones[-1] != "sp":
                    result.phones.append("sp")
                continue
            if not _WORD_RE.fullmatch(token):
                issue = "no pronunciation for {!r}".format(token)
                if strict:
                    raise ValueError(issue)
                result.issues.append(issue + "; replaced with a pause")
                if result.phones and result.phones[-1] != "sp":
                    result.phones.append("sp")
                continue
            phones = None
            if resolve_homographs:
                phones = self._homograph_choice(tokens, index)
                if phones is not None:
                    result.issues.append(
                        "homograph {!r} read as {}".format(token, " ".join(phones))
                    )
            if phones is None:
                if token in self.lexicon:
                    phones = self.lexicon[token]
                else:
                    phones = self.oov_phones(token)
                    result.issues.append("{!r} not in lexicon; used g2p_en".format(token))
            result.words.append((token, phones))
            result.phones.extend(phones)
        while result.phones and result.phones[-1] == "sp":
            result.phones.pop()
        return result
