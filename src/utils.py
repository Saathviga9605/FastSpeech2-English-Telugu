"""Small helpers ported from upstream utils/tools.py, without a global device."""
import numpy as np
import torch
import torch.nn.functional as F


def to_device(batch, device):
    ids, raw_texts, speakers, texts, src_lens, max_src_len = batch[:6]
    out = [
        ids,
        raw_texts,
        torch.from_numpy(speakers).long().to(device),
        torch.from_numpy(texts).long().to(device),
        torch.from_numpy(src_lens).to(device),
        max_src_len,
    ]
    if len(batch) == 12:
        mels, mel_lens, max_mel_len, pitches, energies, durations = batch[6:]
        out += [
            torch.from_numpy(mels).float().to(device),
            torch.from_numpy(mel_lens).to(device),
            max_mel_len,
            torch.from_numpy(pitches).float().to(device),
            torch.from_numpy(energies).float().to(device),
            torch.from_numpy(durations).long().to(device),
        ]
    return tuple(out)


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device)
    ids = ids.unsqueeze(0).expand(batch_size, -1)
    return ids >= lengths.unsqueeze(1).expand(-1, max_len)


def pad_1D(inputs, PAD=0):
    max_len = max(len(x) for x in inputs)
    return np.stack(
        [
            np.pad(x, (0, max_len - x.shape[0]), mode="constant", constant_values=PAD)
            for x in inputs
        ]
    )


def pad_2D(inputs):
    max_len = max(np.shape(x)[0] for x in inputs)
    return np.stack(
        [
            np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode="constant")
            for x in inputs
        ]
    )


def pad(input_ele, mel_max_length=None):
    max_len = mel_max_length or max(x.size(0) for x in input_ele)
    out = []
    for batch in input_ele:
        if batch.dim() == 1:
            out.append(F.pad(batch, (0, max_len - batch.size(0)), "constant", 0.0))
        else:
            out.append(F.pad(batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0))
    return torch.stack(out)
