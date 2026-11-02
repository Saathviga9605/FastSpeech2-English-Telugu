"""Training dataset, adapted from upstream dataset.py to JSONL metadata and per-language vocabularies."""
import json
import os

import numpy as np
from torch.utils.data import Dataset as TorchDataset

from ..text.vocab import Vocabulary
from ..utils import pad_1D, pad_2D


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Dataset(TorchDataset):
    def __init__(self, split, preprocess_config, train_config, symbols, sort=False, drop_last=False):
        self.path = preprocess_config["path"]["preprocessed_path"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.vocab = Vocabulary(symbols)
        self.items = read_jsonl(os.path.join(self.path, "{}.jsonl".format(split)))
        with open(os.path.join(self.path, "speakers.json"), encoding="utf-8") as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.items)

    def _load(self, kind, item):
        return np.load(os.path.join(self.path, kind, "{}.npy".format(item["id"])))

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "id": item["id"],
            "speaker": self.speaker_map[item["speaker"]],
            "text": np.array(self.vocab.encode(item["phones"])),
            "raw_text": item["text"],
            "mel": self._load("mel", item),
            "pitch": self._load("pitch", item),
            "energy": self._load("energy", item),
            "duration": self._load("duration", item),
        }

    def reprocess(self, data, idxs):
        texts = [data[i]["text"] for i in idxs]
        mels = [data[i]["mel"] for i in idxs]
        text_lens = np.array([t.shape[0] for t in texts])
        mel_lens = np.array([m.shape[0] for m in mels])
        return (
            [data[i]["id"] for i in idxs],
            [data[i]["raw_text"] for i in idxs],
            np.array([data[i]["speaker"] for i in idxs]),
            pad_1D(texts),
            text_lens,
            int(max(text_lens)),
            pad_2D(mels),
            mel_lens,
            int(max(mel_lens)),
            pad_1D([data[i]["pitch"] for i in idxs]),
            pad_1D([data[i]["energy"] for i in idxs]),
            pad_1D([data[i]["duration"] for i in idxs]),
        )

    def collate_fn(self, data):
        size = len(data)
        if self.sort:
            order = np.argsort(-np.array([d["text"].shape[0] for d in data]))
        else:
            order = np.arange(size)
        usable = size - size % self.batch_size
        groups = order[:usable].reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and usable < size:
            groups.append(order[usable:].tolist())
        return [self.reprocess(data, g) for g in groups]
