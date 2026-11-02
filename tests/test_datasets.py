"""Dataset access, inspection and storage logic. No test needs credentials or network access."""
import glob
import io
import json
import os
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from src.config import load_configs
from src.data import corpus, hf_auth, inspection, storage, validation

FAKE = "unit-test-value"


def wav_bytes(seconds=2.0, rate=16000):
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(seconds * rate)), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def write_shard(directory, rows, audio_column="audio", extra=None):
    os.makedirs(os.path.join(directory, "data"), exist_ok=True)
    table = pa.table({
        audio_column: pa.array([{"bytes": b, "path": "a.wav"} if b is not None else None for b, _ in rows],
                               type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "text": [t for _, t in rows],
        **(extra or {}),
    })
    pq.write_table(table, os.path.join(directory, "data", "train-00000.parquet"))


@pytest.fixture
def telugu_config(tmp_path):
    preprocess, _, _ = load_configs("telugu")
    preprocess = json.loads(json.dumps(preprocess))
    preprocess["path"]["parquet_dir"] = str(tmp_path / "raw")
    preprocess["path"]["interim_path"] = str(tmp_path / "interim")
    preprocess["path"]["manifest_dir"] = str(tmp_path / "manifests")
    preprocess["selection"]["speakers"] = None
    return preprocess


# Authentication


def test_missing_secret_gives_clear_error():
    with pytest.raises(RuntimeError, match="HF_TOKEN is missing"):
        hf_auth.read_secret(lambda name: None)


def test_inaccessible_secret_gives_clear_error():
    class SecretNotFoundError(Exception):
        pass

    def getter(name):
        raise SecretNotFoundError("details")

    with pytest.raises(RuntimeError, match="Colab Secrets.*SecretNotFoundError") as caught:
        hf_auth.read_secret(getter)
    assert caught.value.__cause__ is None


def test_authentication_uses_token_without_exposing_it(capsys):
    calls = {}
    result = hf_auth.authenticate(
        lambda name: FAKE, lambda **kw: calls.update(kw), forbidden_root="/not/the/token/path"
    )
    assert calls == {"token": FAKE, "add_to_git_credential": False}
    assert result is None
    captured = capsys.readouterr()
    assert FAKE not in captured.out + captured.err


def test_rejected_login_does_not_expose_token():
    def fail(**kw):
        raise ValueError("bad token " + kw["token"])

    with pytest.raises(RuntimeError) as caught:
        hf_auth.authenticate(lambda name: FAKE, fail, forbidden_root="/not/the/token/path")
    assert FAKE not in str(caught.value) and caught.value.__cause__ is None


def test_refuses_to_store_token_on_drive(monkeypatch, tmp_path):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(tmp_path / "drive" / "token"))
    with pytest.raises(RuntimeError, match="Drive"):
        hf_auth.authenticate(lambda name: FAKE, lambda **kw: None, forbidden_root=str(tmp_path / "drive"))


def test_tracked_project_files_hold_no_token_or_notebook_output():
    root = Path(__file__).resolve().parents[1]
    for path in glob.glob(str(root / "notebooks" / "*.ipynb")):
        text = Path(path).read_text(encoding="utf-8")
        assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)
        for cell in json.loads(text)["cells"]:
            assert not cell.get("outputs")


# Row validation


@pytest.mark.parametrize(
    "text,language,expected",
    [
        (None, "english", "missing"),
        ("   ", "english", "empty"),
        ("bad �", "english", "U+FFFD"),
        ("à°¤à±†", "telugu", "wrongly decoded"),
        ("hello", "telugu", "no Telugu"),
        ("Dr. Rao, isn't it?", "english", None),
        ("తెలుగు", "telugu", None),
    ],
)
def test_text_problem(text, language, expected):
    result = validation.text_problem(text, language)
    assert (result is None) if expected is None else (expected in result)


def test_missing_and_corrupt_audio_are_detected():
    with pytest.raises(ValueError, match="missing audio"):
        validation.audio_properties(None)
    with pytest.raises(ValueError, match="missing audio"):
        validation.audio_properties({"bytes": None})
    with pytest.raises(ValueError, match="unreadable"):
        validation.audio_properties({"bytes": b"not audio"})
    assert validation.audio_properties({"bytes": wav_bytes()})["sample_rate"] == 16000


def test_speaker_columns_are_only_candidates():
    columns = ["audio_filepath", "text", "gender", "age-group", "primary_language", "native_place_state"]
    assert validation.speaker_columns(columns) == []
    assert validation.speaker_columns(["text", "speaker_id"]) == ["speaker_id"]


# Inventory and report


def test_inspection_keeps_going_and_reports_bad_rows(telugu_config, tmp_path):
    original = "తెలుగు‌, భాష!"
    write_shard(
        telugu_config["path"]["parquet_dir"],
        [(wav_bytes(), original), (None, "తెలుగు"), (b"garbage", "తెలుగు"), (wav_bytes(), None), (wav_bytes(), "hello")],
        extra={"gender": [1, 1, 1, 0, 0]},
    )
    path = str(tmp_path / "interim" / "inventory.jsonl")
    corpus.inspect(telugu_config, path)
    inventory = corpus.read_inventory(path)

    assert inventory[0]["text"] == original  # transcripts are preserved byte for byte
    assert not inventory[0]["text_error"] and not inventory[0]["audio_error"]
    assert inventory[1]["audio_error"] == "missing audio"
    assert "unreadable" in inventory[2]["audio_error"]
    assert inventory[3]["text_error"] == "missing transcript"
    assert "no Telugu" in inventory[4]["text_error"]

    report = inspection.build_report(telugu_config, inventory, inspection.parquet_schema(telugu_config))
    assert report["examples"] == 5
    assert sum(report["missing_or_invalid_audio"].values()) == 2
    assert report["speaker_groups"]["male"]["usable_clips"] == 1
    assert report["enough_audio"] is False
    assert report["parquet_schema"]["rows"] == 5
    assert "gender" in report["parquet_schema"]["columns"]


def test_non_nfc_text_is_noted_not_rewritten(telugu_config, tmp_path):
    decomposed = "కై"
    write_shard(telugu_config["path"]["parquet_dir"], [(wav_bytes(), decomposed)], extra={"gender": [1]})
    path = str(tmp_path / "interim" / "inventory.jsonl")
    corpus.inspect(telugu_config, path)
    inventory = corpus.read_inventory(path)
    assert inventory[0]["text"] == decomposed
    report = inspection.build_report(telugu_config, inventory)
    assert report["transcript_notes"]["not_nfc"] == 1


def test_english_without_speaker_column_is_one_group_not_inferred():
    preprocess, _, _ = load_configs("english")
    row = {"gender": "female", "native_place_state": "Kerala", "primary_language": "Malayalam"}
    assert corpus.speaker_of(row, preprocess) == "svarah"
    assert corpus.speaker_of({**row, "speaker_id": "s7"}, preprocess) == "s7"


def test_duration_column_is_compared_with_measurement():
    records = [
        {"seconds": 2.0, "meta": {"duration": 2.0}},
        {"seconds": 2.0, "meta": {"duration": 3.0}},
    ]
    assert inspection.duration_column_check(records)["over_100ms_apart"] == 1
    assert inspection.duration_column_check([{"seconds": 1.0, "meta": {}}]) is None


def test_export_verification_finds_changed_transcript(telugu_config, tmp_path):
    preprocess = telugu_config
    inventory = [{"row": 0, "seconds": 2.0, "text": "తెలుగు", "speaker": "male"}]
    manifest = {"id": "telugu_000000", "row": 0, "speaker": "male", "seconds": 2.0, "text": "తెలుగు!"}
    os.makedirs(preprocess["path"]["manifest_dir"])
    for split, rows in (("train", [manifest]), ("val", [])):
        with open(os.path.join(preprocess["path"]["manifest_dir"], "telugu_{}.jsonl".format(split)), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    os.makedirs(os.path.join(preprocess["path"]["interim_path"], "wav"))
    result = inspection.verify_export(preprocess, inventory)
    assert result["status"] == "problems found"
    assert result["problems"]["transcript differs from dataset"] == 1
    assert result["problems"]["exported audio missing"] == 1


# Drive and cache


def test_drive_layout_is_created_and_repeatable(tmp_path):
    root = str(tmp_path / "MyDrive" / "FastSpeech2-IIITH")
    for _ in range(2):
        storage.create_drive_layout(root)
    for relative in (
        "datasets/english", "datasets/telugu", "processed/english", "processed/telugu",
        "checkpoints/english", "checkpoints/telugu", "experiments", "outputs",
    ):
        assert os.path.isdir(os.path.join(root, relative))


def test_environment_points_at_drive_and_never_moves_hf_home(tmp_path, monkeypatch):
    for name in ("FS2_DRIVE_ROOT", "FS2_RAW_ENGLISH", "FS2_RAW_TELUGU", "FS2_CKPT_ROOT", "FS2_EXP_ROOT", "FS2_OUTPUT_ROOT", "FS2_DATA_ROOT"):
        monkeypatch.setenv(name, "unset")
    monkeypatch.delenv("HF_HOME", raising=False)
    root = str(tmp_path / "drive")
    storage.configure_environment(root)
    assert os.environ["FS2_RAW_TELUGU"] == os.path.join(root, "datasets", "telugu")
    assert "HF_HOME" not in os.environ


def test_legacy_checkpoints_are_copied_not_moved(tmp_path, monkeypatch):
    root = tmp_path / "drive"
    old = root / "experiments" / "telugu" / "ckpt"
    old.mkdir(parents=True)
    (old / "5.pth.tar").write_bytes(b"x")
    storage.create_drive_layout(str(root))
    storage.copy_legacy_checkpoints(str(root))
    assert (old / "5.pth.tar").exists()
    assert (root / "checkpoints" / "telugu" / "5.pth.tar").exists()


def test_cache_validation(tmp_path):
    folder = str(tmp_path / "cache")
    assert "no record" in storage.cache_problems(folder, "a/b")[0]
    write_shard(folder, [(wav_bytes(), "తెలుగు")])
    size = os.path.getsize(os.path.join(folder, "data", "train-00000.parquet"))
    expected = {"data/train-00000.parquet": size}
    storage.write_marker(folder, "a/b", "abc", expected)
    assert storage.cache_problems(folder, "a/b") == []
    assert storage.cache_problems(folder, "other/repo")
    assert "missing" in storage.cache_problems(folder, "a/b", {**expected, "data/x.parquet": 1})[0]

    path = os.path.join(folder, "data", "train-00000.parquet")
    with open(path, "r+b") as f:  # same size, damaged contents
        f.seek(-8, os.SEEK_END)
        f.write(b"\0" * 8)
    assert "unreadable" in storage.cache_problems(folder, "a/b")[0]
    with open(path, "ab") as f:
        f.write(b"more")
    assert "bytes" in storage.cache_problems(folder, "a/b")[0]


def test_valid_cache_avoids_a_second_download(telugu_config, monkeypatch):
    folder = telugu_config["path"]["parquet_dir"]
    write_shard(folder, [(wav_bytes(), "తెలుగు")])
    size = os.path.getsize(os.path.join(folder, "data", "train-00000.parquet"))
    monkeypatch.setattr(corpus, "remote_state", lambda p: ("rev", {"data/train-00000.parquet": size}))

    def fail(*args, **kwargs):
        raise AssertionError("should not download")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    assert corpus.download(telugu_config) == folder
    assert storage.cache_problems(folder, telugu_config["hf_repo"]) == []


# Alignment checker and restored archives

GRID = """File type = "ooTextFile"
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
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = "k"
        intervals [2]:
            xmin = 0.5
            xmax = 1.0
            text = "a"
"""


def _aligned_corpus(preprocess, grids):
    import importlib.util

    interim = preprocess["path"]["interim_path"]
    os.makedirs(preprocess["path"]["manifest_dir"])
    os.makedirs(os.path.join(interim, "mfa_corpus", "male"))
    os.makedirs(os.path.join(interim, "textgrids", "male"))
    with open(os.path.join(interim, "lexicon.txt"), "w", encoding="utf-8") as f:
        f.write("k_a\tk a\n")
    rows = []
    for uid, grid in grids.items():
        rows.append({"id": uid, "row": len(rows), "speaker": "male", "seconds": 1.0, "text": "క"})
        with open(os.path.join(interim, "mfa_corpus", "male", uid + ".lab"), "w", encoding="utf-8") as f:
            f.write("k_a")
        if grid is not None:
            with open(os.path.join(interim, "textgrids", "male", uid + ".TextGrid"), "w", encoding="utf-8") as f:
                f.write(grid)
    for split, part in (("train", rows), ("val", [])):
        with open(os.path.join(preprocess["path"]["manifest_dir"], "telugu_{}.jsonl".format(split)), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
    spec = importlib.util.spec_from_file_location(
        "check_alignments", Path(__file__).resolve().parents[1] / "scripts" / "check_alignments.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checker_separates_valid_missing_and_invalid_alignments(telugu_config):
    module = _aligned_corpus(telugu_config, {
        "good": GRID,
        "missing": None,
        "wrong_phone": GRID.replace('"k"', '"zz"'),
        "wrong_order": GRID.replace('"k"', '"X"').replace('"a"', '"k"').replace('"X"', '"a"'),
        "garbage": "not a textgrid",
    })
    result = module.check_language(telugu_config, ["_", "k", "a", "sil", "sp", "spn"])
    assert result["splits"]["train"]["valid"] == 1
    assert result["splits"]["train"]["rejected"] == 4
    assert result["rejection_reasons"]["no TextGrid"] == 1
    assert result["rejection_reasons"]["phones differ from lexicon"] == 1
    assert any("zz" in reason for reason in result["rejection_reasons"])


def test_restored_archive_is_not_reported_as_timing_checked(telugu_config):
    module = _aligned_corpus(telugu_config, {"good": GRID})
    result = module.check_language(telugu_config, ["_", "k", "a", "sil", "sp", "spn"])
    assert result["audio_available"] is False
    assert result["timing_checked_by_this_tool"] is False


def test_features_refuse_a_restored_archive_without_audio(telugu_config, tmp_path):
    from src.audio import features

    _aligned_corpus(telugu_config, {"good": GRID})
    telugu_config["path"]["preprocessed_path"] = str(tmp_path / "processed")
    with pytest.raises(FileNotFoundError, match="restored archive"):
        features.process_corpus(telugu_config, ["_", "k", "a", "sil", "sp", "spn"])
