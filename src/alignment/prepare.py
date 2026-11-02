"""Prepare a corpus and lexicon for Montreal Forced Aligner, and read its output.

Lab files use ASCII word keys built from phones (for example k_a_m_a). MFA therefore
sees no Telugu Unicode or punctuation it might tokenize differently, and each key maps
to exactly one pronunciation. English keeps ordinary lowercase words.
"""
import argparse
import os
import shutil

import tgt

from ..config import load_configs, load_symbols
from ..data.corpus import read_manifests
from ..text import get_frontend

SILENCE_LABELS = {"", "sil", "sp", "spn"}


def word_key(token, phones, language):
    return "_".join(phones).lower() if language == "telugu" else token


def prepare(preprocess_config):
    """Create <interim>/mfa_corpus/*.{wav,lab} and <interim>/lexicon.txt."""
    language = preprocess_config["language"]
    interim = preprocess_config["path"]["interim_path"]
    corpus = os.path.join(interim, "mfa_corpus")
    frontend = get_frontend(language, preprocess_config)
    lexicon = {}
    records = [r for rows in read_manifests(preprocess_config).values() for r in rows]
    for record in records:
        speaker_dir = os.path.join(corpus, record["speaker"])
        os.makedirs(speaker_dir, exist_ok=True)
        # Homographs need one pronunciation per word during alignment.
        if language == "english":
            result = frontend(record["text"], strict=True, resolve_homographs=False)
        else:
            result = frontend(record["text"], strict=True)
        words = []
        for token, phones in result.words:
            key = word_key(token, phones, language)
            lexicon.setdefault(key, phones)
            words.append(key)
        with open(os.path.join(speaker_dir, record["id"] + ".lab"), "w", encoding="utf-8") as f:
            f.write(" ".join(words))
        target = os.path.join(speaker_dir, record["id"] + ".wav")
        if not os.path.exists(target):
            shutil.copyfile(os.path.join(interim, "wav", record["id"] + ".wav"), target)
    path = os.path.join(interim, "lexicon.txt")
    with open(path, "w", encoding="utf-8") as f:
        for key in sorted(lexicon):
            f.write("{}\t{}\n".format(key, " ".join(lexicon[key])))
    print("clips", len(records), "lexicon entries", len(lexicon))
    return corpus, path


def read_lexicon(path):
    lexicon = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            key, phones = line.rstrip("\n").split("\t")
            lexicon[key] = phones.split()
    return lexicon


def expected_phones(record, interim, lexicon):
    """The non-silent phones the label file implies for this clip."""
    path = os.path.join(interim, "mfa_corpus", record["speaker"], record["id"] + ".lab")
    with open(path, encoding="utf-8") as f:
        return [phone for word in f.read().split() for phone in lexicon[word]]


def read_alignment(textgrid_path, sampling_rate, hop_length, valid_phones):
    """Return (phones, frame durations, start seconds, end seconds), trimming edge silence.

    MFA can leave silence as empty or absent intervals. Both become explicit `sil`, so frame
    counts always add up. Unknown labels raise an error rather than being dropped.
    """
    tier = tgt.io.read_textgrid(textgrid_path).get_tier_by_name("phones")
    intervals, cursor = [], float(tier.start_time)
    for item in tier.intervals if hasattr(tier, "intervals") else tier._objects:
        if item.start_time > cursor:
            intervals.append((cursor, item.start_time, "sil"))
        label = item.text.strip()
        intervals.append((item.start_time, item.end_time, "sil" if label == "" else label))
        cursor = item.end_time
    if tier.end_time > cursor:
        intervals.append((cursor, tier.end_time, "sil"))

    phones, durations, start, end, end_index = [], [], 0.0, 0.0, 0
    for s, e, label in ((float(a), float(b), c) for a, b, c in intervals):
        if not phones and label in SILENCE_LABELS:
            continue
        if not phones:
            start = s
        if label not in valid_phones:
            raise ValueError("Alignment label {!r} is not in the vocabulary".format(label))
        phones.append(label)
        durations.append(
            int(round(e * sampling_rate / hop_length) - round(s * sampling_rate / hop_length))
        )
        if label not in SILENCE_LABELS:
            end = e
            end_index = len(phones)
    return phones[:end_index], durations[:end_index], start, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=["english", "telugu"])
    args = parser.parse_args()
    preprocess, _, _ = load_configs(args.language)
    load_symbols(args.language)
    prepare(preprocess)


if __name__ == "__main__":
    main()
