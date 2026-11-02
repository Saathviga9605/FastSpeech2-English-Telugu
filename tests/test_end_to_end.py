"""Train, interrupt, resume and synthesize on synthetic features. Tests the code path only."""
import json
import os

import numpy as np

from src.config import load_symbols
from src.inference import synthesize as synth
from src.text import telugu
from src.training import train as trainer
from src.training.checkpoint import list_checkpoints
from src.visualization.plots import plot_losses


def write_features(root):
    os.makedirs(root, exist_ok=True)
    for kind in ("mel", "pitch", "energy", "duration"):
        os.makedirs(os.path.join(root, kind), exist_ok=True)
    rng = np.random.default_rng(0)
    symbols = load_symbols("telugu")[1:-3]
    for split, count in (("train", 70), ("val", 4)):
        with open(os.path.join(root, split + ".jsonl"), "w", encoding="utf-8") as f:
            for i in range(count):
                uid = "{}_{}".format(split, i)
                phones = list(rng.choice(symbols, 6))
                durations = np.full(6, 3)
                np.save(os.path.join(root, "duration", uid + ".npy"), durations)
                np.save(os.path.join(root, "pitch", uid + ".npy"), rng.normal(size=6).astype(np.float32))
                np.save(os.path.join(root, "energy", uid + ".npy"), rng.normal(size=6).astype(np.float32))
                np.save(os.path.join(root, "mel", uid + ".npy"), rng.normal(size=(18, 80)).astype(np.float32))
                f.write(json.dumps({"id": uid, "speaker": "s", "phones": phones, "text": "x"}) + "\n")
    with open(os.path.join(root, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"pitch": [-3, 3, 150, 30], "energy": [-3, 3, 10, 3]}, f)
    with open(os.path.join(root, "speakers.json"), "w", encoding="utf-8") as f:
        json.dump({"s": 0}, f)


def test_train_resume_plot_synthesize(tmp_path, monkeypatch):
    monkeypatch.setenv("FS2_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("FS2_EXP_ROOT", str(tmp_path / "exp"))
    monkeypatch.setenv("FS2_CKPT_ROOT", str(tmp_path / "ckpt"))
    write_features(str(tmp_path / "data" / "processed" / "telugu"))

    real_load = trainer.load_configs

    def short_config(language):
        preprocess, model, train = real_load(language)
        train["step"].update({"log_step": 1, "val_step": 2, "save_step": 2, "keep_last": 2})
        return preprocess, model, train

    monkeypatch.setattr(trainer, "load_configs", short_config)
    assert trainer.train("telugu", max_steps=2) == 2
    ckpt_dir = tmp_path / "ckpt" / "telugu"
    assert [s for s, _ in list_checkpoints(str(ckpt_dir))] == [2]

    assert trainer.train("telugu", max_steps=2) == 4  # resumes rather than restarting
    assert [s for s, _ in list_checkpoints(str(ckpt_dir))] == [2, 4]
    log = tmp_path / "exp" / "telugu" / "log"
    with open(log / "train_loss.csv", encoding="utf-8") as f:
        assert [line.split(",")[0] for line in f.read().splitlines()[1:]] == ["1", "2", "3", "4"]
    plot_losses(str(log / "train_loss.csv"), str(log / "val_loss.csv"), str(tmp_path / "loss.png"), "test")

    utterances = tmp_path / "u.json"
    utterances.write_text(json.dumps({"telugu": [{"id": "T1", "text": "తెలుగు భాష, ☃"}, {"id": "T2", "text": "@@@"}]}), encoding="utf-8")
    records = synth.synthesize("telugu", str(utterances), out_root=str(tmp_path))
    assert records[0]["original"] == "తెలుగు భాష, ☃"
    assert records[0]["status"] == "generated"
    assert any("U+2603" in issue for issue in records[0]["issues"])
    assert records[1]["status"].startswith("no audio")
    results = tmp_path / "Results" / "Telugu"
    assert (results / "test_generated_audios" / "T1.wav").stat().st_size > 44
    for kind in ("duration", "pitch", "energy"):
        assert (results / "variance_adaptor_plots" / kind / "T1.png").exists()
    assert telugu.process("తెలుగు").phones
