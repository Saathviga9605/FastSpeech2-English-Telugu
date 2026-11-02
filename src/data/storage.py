"""Google Drive layout and validation of downloaded dataset files."""
import json
import os
import shutil

import pyarrow.parquet as pq

LANGUAGES = ("english", "telugu")
MARKER = "download_complete.json"


def create_drive_layout(root):
    """Create the persistent directories and return their paths."""
    paths = {"root": root}
    for group in ("datasets", "processed", "checkpoints"):
        for language in LANGUAGES:
            paths["{}_{}".format(group, language)] = os.path.join(root, group, language)
    paths["experiments"] = os.path.join(root, "experiments")
    paths["outputs"] = os.path.join(root, "outputs")
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def configure_environment(root, local_data="/content/data"):
    """Point the pipeline at Drive. Features stay on the faster local disk during a session."""
    paths = create_drive_layout(root)
    os.environ.update(
        FS2_DRIVE_ROOT=root,
        FS2_RAW_ENGLISH=paths["datasets_english"],
        FS2_RAW_TELUGU=paths["datasets_telugu"],
        FS2_CKPT_ROOT=os.path.join(root, "checkpoints"),
        FS2_EXP_ROOT=paths["experiments"],
        FS2_OUTPUT_ROOT=paths["outputs"],
        FS2_DATA_ROOT=local_data,
    )
    copy_legacy_checkpoints(root)
    return paths


def copy_legacy_checkpoints(root):
    """Earlier notebooks saved under experiments/<language>/ckpt. Copy, never delete, those files."""
    for language in LANGUAGES:
        old = os.path.join(root, "experiments", language, "ckpt")
        new = os.path.join(root, "checkpoints", language)
        if os.path.isdir(old) and os.listdir(old) and not os.listdir(new):
            for name in os.listdir(old):
                shutil.copy2(os.path.join(old, name), new)
            print("Copied earlier checkpoints from", old, "to", new)


def local_files(directory):
    found = {}
    data = os.path.join(directory, "data")
    if os.path.isdir(data):
        for name in sorted(os.listdir(data)):
            if name.endswith(".parquet"):
                found["data/" + name] = os.path.getsize(os.path.join(data, name))
    return found


def write_marker(directory, repo, revision, expected):
    path = os.path.join(directory, MARKER)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"repo": repo, "revision": revision, "files": expected}, f, indent=1)
    os.replace(path + ".tmp", path)


def cache_problems(directory, repo, expected=None):
    """Return the reasons a cached download cannot be reused; an empty list means it can.

    `expected` holds current file sizes from the Hub, when reachable. Otherwise the sizes
    recorded after the last complete download are used.
    """
    if expected is None:
        try:
            with open(os.path.join(directory, MARKER), encoding="utf-8") as f:
                marker = json.load(f)
        except (OSError, ValueError):
            return ["no record of a completed download"]
        if marker.get("repo") != repo:
            return ["cache belongs to {}".format(marker.get("repo"))]
        expected = marker["files"]
    if not expected:
        return ["no expected files"]
    actual = local_files(directory)
    problems = []
    for name, size in expected.items():
        if name not in actual:
            problems.append("missing {}".format(name))
        elif actual[name] != size:
            problems.append("{} is {} bytes, expected {}".format(name, actual[name], size))
        else:
            try:
                pq.ParquetFile(os.path.join(directory, name))
            except Exception as error:  # any read failure means the file cannot be trusted
                problems.append("{} is unreadable: {}".format(name, type(error).__name__))
    return problems
