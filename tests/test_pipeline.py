"""End-to-end checks on synthetic data. They verify the code runs, not model quality."""
import json
import os

import numpy as np
import pytest
import soundfile as sf
import torch

from src.alignment.prepare import read_alignment
from src.audio import features
from src.config import load_configs, load_symbols
from src.data.corpus import select
from src.models import FastSpeech2, FastSpeech2Loss, ScheduledOptim
from src.text.english import EnglishFrontend
from src.training import checkpoint as ckpt
from src.training import train as trainer
from src.training.benchmark import synthetic_batch
from src.visualization.plots import plot_losses, plot_variance


@pytest.fixture(params=["english", "telugu"])
def setup(request, tmp_path):
    preprocess, model_config, train_config = load_configs(request.param)
    (tmp_path / "stats.json").write_text(
        json.dumps({"pitch": [-3, 3, 0, 1], "energy": [-3, 3, 0, 1]}), encoding="utf-8"
    )
    preprocess["path"]["preprocessed_path"] = str(tmp_path)
    return request.param, preprocess, model_config, train_config, tmp_path


def batch_for(model_config, n=2):
    return synthetic_batch(n, 8, 40, model_config["n_symbols"], torch.device("cpu"))


def test_forward_backward_update(setup):
    _, preprocess, model_config, train_config, _ = setup
    model = FastSpeech2(preprocess, model_config)
    loss_fn = FastSpeech2Loss(preprocess, model_config)
    optimizer = ScheduledOptim(model, train_config, model_config, 0)
    before = model.encoder.src_word_emb.weight.clone()
    batch = batch_for(model_config)
    losses = loss_fn(batch, model(*batch[2:]))
    losses[0].backward()
    optimizer.step_and_update_lr()
    assert torch.isfinite(losses[0])
    assert not torch.equal(before, model.encoder.src_word_emb.weight)


def test_vocabularies_are_separate():
    assert load_symbols("english") != load_symbols("telugu")


def test_postnet_switch(setup):
    _, preprocess, model_config, _, _ = setup
    config = {**model_config, "use_postnet": False}
    model = FastSpeech2(preprocess, config)
    batch = batch_for(config)
    losses = FastSpeech2Loss(preprocess, config)(batch, model(*batch[2:]))
    assert model.postnet is None
    assert losses[2].item() == 0


def test_energy_control_is_independent_of_pitch_control(setup):
    _, preprocess, model_config, _, _ = setup
    torch.manual_seed(0)
    model = FastSpeech2(preprocess, model_config).eval()
    # A long duration control keeps untrained predictions from rounding to zero frames.
    ids = torch.randint(1, model_config["n_symbols"], (1, 6))
    args = (torch.zeros(1, dtype=torch.long), ids, torch.tensor([6]), 6)
    with torch.no_grad():
        base = model(*args)[3]
        changed_energy = model(*args, e_control=2.0)[3]
        changed_pitch = model(*args, p_control=2.0)[2]
        base_pitch = model(*args)[2]
    assert torch.allclose(changed_energy, base * 2.0, atol=1e-5)
    assert torch.allclose(changed_pitch, base_pitch * 2.0, atol=1e-5)


def test_all_zero_predicted_durations_do_not_crash(setup):
    _, preprocess, model_config, _, _ = setup
    model = FastSpeech2(preprocess, model_config).eval()
    args = (torch.zeros(1, dtype=torch.long), torch.ones(1, 4, dtype=torch.long), torch.tensor([4]), 4)
    with torch.no_grad():
        output = model(*args, d_control=0.0)
    assert output[9].item() == 4


def test_training_refuses_to_truncate_long_clips(setup):
    _, preprocess, model_config, _, _ = setup
    model = FastSpeech2(preprocess, model_config)
    batch = synthetic_batch(1, 10, 1200, model_config["n_symbols"], torch.device("cpu"))
    with pytest.raises(ValueError):
        model(*batch[2:])


def test_checkpoint_resume_restores_everything(setup):
    _, preprocess, model_config, train_config, tmp_path = setup
    model = FastSpeech2(preprocess, model_config)
    loss_fn = FastSpeech2Loss(preprocess, model_config)
    optimizer = ScheduledOptim(model, train_config, model_config, 0)
    batch = batch_for(model_config)
    loss_fn(batch, model(*batch[2:]))[0].backward()
    optimizer.step_and_update_lr()
    folder = str(tmp_path / "ckpt")
    ckpt.save_checkpoint(folder, 10, 2, model, optimizer, ["cfg"])
    (tmp_path / "ckpt" / "20.pth.tar").write_bytes(b"damaged by a disconnect")
    path, loaded = ckpt.latest_valid_checkpoint(folder, torch.device("cpu"))
    assert path.endswith("10.pth.tar") and loaded["step"] == 10 and loaded["epoch"] == 2
    fresh = FastSpeech2(preprocess, model_config)
    fresh.load_state_dict(loaded["model"])
    for key, value in model.state_dict().items():
        assert torch.equal(value, fresh.state_dict()[key])
    resumed = ScheduledOptim(fresh, train_config, model_config, 10)
    resumed.load_state_dict(loaded["optimizer"])
    assert resumed.state_dict()["state"]


def test_checkpoint_retention(setup):
    _, preprocess, model_config, train_config, tmp_path = setup
    model = FastSpeech2(preprocess, model_config)
    optimizer = ScheduledOptim(model, train_config, model_config, 0)
    folder = str(tmp_path / "keep")
    for step in (1, 2, 3, 4):
        ckpt.save_checkpoint(folder, step, 1, model, optimizer, [], keep_last=2)
    assert [s for s, _ in ckpt.list_checkpoints(folder)] == [3, 4]


def test_selection_is_seeded_disjoint_and_limited():
    inventory = [
        {"row": i, "seconds": 8.0, "speaker": "a", "text_error": None} for i in range(4000)
    ] + [{"row": 9000, "seconds": 90.0, "speaker": "a", "text_error": None}]
    settings = {"train_hours": 5.0, "val_hours": 1.0, "min_seconds": 1.0, "max_seconds": 11.5, "seed": 1, "speakers": None}
    train, val = select(inventory, settings)
    again, _ = select(inventory, settings)
    assert [r["row"] for r in train] == [r["row"] for r in again]
    assert not {r["row"] for r in train} & {r["row"] for r in val}
    assert all(r["seconds"] <= 11.5 for r in train + val)
    assert 4.99 <= sum(r["seconds"] for r in train) / 3600 <= 5.01
    with pytest.raises(ValueError):
        select(inventory[:100], settings)


TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 5
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = ""
        intervals [2]:
            xmin = 0.1
            xmax = 0.4
            text = "k"
        intervals [3]:
            xmin = 0.4
            xmax = 0.5
            text = ""
        intervals [4]:
            xmin = 0.5
            xmax = 0.9
            text = "a"
        intervals [5]:
            xmin = 0.9
            xmax = 1.0
            text = "sil"
"""


def test_alignment_keeps_internal_silence_and_trims_edges(tmp_path):
    path = tmp_path / "x.TextGrid"
    path.write_text(TEXTGRID, encoding="utf-8")
    phones, durations, start, end = read_alignment(str(path), 22050, 256, {"k", "a", "sil"})
    assert phones == ["k", "sil", "a"]
    assert start == pytest.approx(0.1) and end == pytest.approx(0.9)
    assert sum(durations) == round(0.9 * 22050 / 256) - round(0.1 * 22050 / 256)
    with pytest.raises(ValueError):
        read_alignment(str(path), 22050, 256, {"sil", "a"})


def test_features_on_a_synthetic_tone(tmp_path):
    preprocess, _, _ = load_configs("telugu")
    rate = 22050
    t = np.arange(rate) / rate
    wav_path = tmp_path / "x.wav"
    sf.write(wav_path, 0.5 * np.sin(2 * np.pi * 150 * t), rate, subtype="PCM_16")
    grid = tmp_path / "x.TextGrid"
    grid.write_text(TEXTGRID, encoding="utf-8")
    stft = features.build_stft(preprocess)
    phones, durations, f0, energy, mel = features.process_clip(
        str(wav_path), str(grid), stft, preprocess["preprocessing"], {"k", "a", "sil"}
    )
    assert mel.shape == (int(durations.sum()), 80)
    assert len(f0) == len(energy) == len(phones) == 3
    assert abs(f0[0] - 150) < 5
    assert energy[0] > 0


def test_inliers_keep_constant_values_and_drop_outliers():
    assert len(features.inliers(np.full(4, 120.0))) == 4
    assert 1000.0 not in features.inliers(np.array([100.0, 101.0, 102.0, 103.0, 1000.0]))


def test_english_frontend_lexicon_and_malformed_input():
    preprocess, _, _ = load_configs("english")
    frontend = EnglishFrontend(preprocess["path"]["lexicon_path"])
    result = frontend.process("Hello, world", resolve_homographs=False)
    assert result.phones == ["HH", "AH0", "L", "OW1", "sp", "W", "ER1", "L", "D"]
    assert frontend.process("Dr. Smith has 2 cats", resolve_homographs=False).phones
    assert frontend.process("", resolve_homographs=False).phones == []
    odd = frontend.process("#### hello @@", resolve_homographs=False)
    assert odd.original == "#### hello @@"
    assert odd.issues


def test_plots_write_files(tmp_path):
    for name, rows in (("t.csv", 3), ("v.csv", 1)):
        path = tmp_path / name
        for step in range(rows):
            trainer.append_csv(str(path), step + 1, [1.0, 0.5, 0.5, 0.1, 0.1, 0.1], 1.0)
    plot_losses(str(tmp_path / "t.csv"), str(tmp_path / "v.csv"), str(tmp_path / "out" / "l.png"), "x")
    plot_variance(["a", "b"], [1, 2], "y", "t", str(tmp_path / "out" / "v.png"))
    assert os.path.getsize(tmp_path / "out" / "l.png") > 0


def test_resume_trims_rows_after_checkpoint(tmp_path):
    path = str(tmp_path / "l.csv")
    for step in (100, 200, 300):
        trainer.append_csv(path, step, [1.0] * 6)
    trainer.trim_csv(path, 200)
    with open(path, encoding="utf-8") as f:
        assert len(f.readlines()) == 3


def test_selection_can_keep_repeated_text_on_one_side():
    inventory = [
        {"row": i, "seconds": 8.0, "speaker": "a", "text_error": None, "text": "t{}".format(i % 1500)}
        for i in range(4000)
    ]
    settings = {
        "train_hours": 5.0, "val_hours": 1.0, "min_seconds": 1.0, "max_seconds": 11.5,
        "seed": 1, "speakers": None, "disjoint_text": True,
    }
    train, val = select(inventory, settings)
    assert not {r["text"] for r in train} & {r["text"] for r in val}
    settings["disjoint_text"] = False
    train, val = select(inventory, settings)
    assert {r["text"] for r in train} & {r["text"] for r in val}
