"""Save or restore one language's prepared data between Colab's local disk and Drive.

Training reads many small files, so features live on the local disk during a session
and are kept on Drive as a single archive.
"""
import argparse
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import set_default_roots  # noqa: E402


def members(language, data_root):
    candidates = [
        "processed/" + language,
        "manifests/{}_train.jsonl".format(language),
        "manifests/{}_val.jsonl".format(language),
        "interim/{}/inventory.jsonl".format(language),
        "interim/{}/lexicon.txt".format(language),
        "interim/{}/textgrids".format(language),
        "interim/{}/acoustic_model.zip".format(language),
        "interim/{}/mfa_validate.log".format(language),
    ]
    return [c for c in candidates if os.path.exists(os.path.join(data_root, c))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["save", "restore"])
    parser.add_argument("language", choices=["english", "telugu"])
    parser.add_argument("--drive_dir", default=os.environ.get("FS2_DRIVE_ROOT"), required="FS2_DRIVE_ROOT" not in os.environ)
    args = parser.parse_args()
    set_default_roots()
    data_root = os.environ["FS2_DATA_ROOT"]
    archive = os.path.join(args.drive_dir, "processed", args.language, "prepared.tar")
    legacy = os.path.join(args.drive_dir, "prepared_{}.tar".format(args.language))
    if args.action == "restore" and not os.path.exists(archive) and os.path.exists(legacy):
        archive = legacy
    if args.action == "save":
        # Never let a failed preprocessing run replace a good archive on Drive.
        finished = os.path.join(data_root, "processed", args.language, "train.jsonl")
        if not os.path.exists(finished):
            sys.exit("Not saving: {} is missing, so features are incomplete. Drive is unchanged.".format(finished))
        tmp = archive + ".tmp"
        os.makedirs(os.path.dirname(archive), exist_ok=True)
        with tarfile.open(tmp, "w") as tar:
            for member in members(args.language, data_root):
                tar.add(os.path.join(data_root, member), arcname=member)
            # Label files let alignments be rechecked; the much larger audio is not archived.
            corpus = os.path.join("interim", args.language, "mfa_corpus")
            for folder, _, names in os.walk(os.path.join(data_root, corpus)):
                for name in names:
                    if name.endswith(".lab"):
                        path = os.path.join(folder, name)
                        tar.add(path, arcname=os.path.relpath(path, data_root))
        os.replace(tmp, archive)
        print("Saved", archive, round(os.path.getsize(archive) / 2**30, 2), "GB")
    else:
        os.makedirs(data_root, exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(data_root, filter="data")
        print("Restored", archive, "to", data_root)


if __name__ == "__main__":
    main()
