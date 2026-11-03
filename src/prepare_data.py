#!/usr/bin/env python3
"""Verify bundled task archives and materialize the exact JSON task files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = ROOT / "data" / "tasks-gz"
TASKS_DIR = ROOT / "data" / "tasks"

RAW_SHA256 = {
    "level_0_0k.json": "73ffbc3b9065bf640212b97cacecbc65a416a6e22cd1f42dbc293ed5da712c2f",
    "level_0_16k.json": "b4c2ce6d8e3759a97e852ca60777e6324330eb294840c388f8420a90246df8ef",
    "level_0_64k.json": "42c29891f70f08ec680f40976013a7adcda249d8c5ef65babc3c1f1d7812d635",
    "level_1_0k.json": "09c47a18a828d7e2710ac0dc21ef935f6b043a945a62d5fe20bc4546a4230f89",
    "level_1_16k.json": "71cf1213940203958e2421b3c1a289c5149d7c0844bb98808091fb069cf3e08d",
    "level_1_64k.json": "e3f0715bf7b3a057c580dea77e6ea7f7e6c12c73d3ea6de9a7cb368a31d66e0b",
    "level_2_0k.json": "bd0ea160f0baa9fe2cfe989767f4bedfec6c34df29571273496fb70beecb1237",
    "level_2_16k.json": "a9dc88a23601980c1cc97d7710f8b749e3be3cd9289a47e4a2cacf2b91f7958a",
    "level_2_64k.json": "1399e28545aed8dab2309b890942e3ea6ba033ac1ba3680a3802c41662f01617",
    "level_3_0k.json": "f677e4c2774d81db7cfa4a9eea124010cf8435caf3909ec001814afe500f28b0",
    "level_3_16k.json": "94476165a1d3a86791433129368d75d8ae1fc1c8e55d5a0349f92f1029da7e10",
    "level_3_64k.json": "6e746d9fd7dc902d4e81f9b36c93c5a02485e6b605da22cf9b89e2a6ae296bac",
}

ARCHIVE_SHA256 = {
    "level_0_0k.json.gz": "41159f30dcfe21982b0f08080fdeb65875f23da940ae103c1e5d826b8bb7e90f",
    "level_0_16k.json.gz": "53be5bfe5ff2b8894b27ff362ea612b979c5d7f848398b1f3ef7cfd3ea83aad2",
    "level_0_64k.json.gz": "f3d4a71f1e3934844ab262baf194fce8e8d64abcccfbe8b7430485f4f5699082",
    "level_1_0k.json.gz": "4cac31ebb535c4e775df9d1a627b472d11c8119a5facb56df352d2bff94f8221",
    "level_1_16k.json.gz": "b360f87326467fbdf8d92de3baa94d5224a29717d78b7e41782a54914f8c7fcd",
    "level_1_64k.json.gz": "fccbf0f5cfb6a5c5a71bbf5f44754fae3054aefbb04f726fb18a2a5ebb33ba13",
    "level_2_0k.json.gz": "58f65874cf89d74192903cb0f2e28a4431cb1fb92e84a8ca5746e16f304a0a1c",
    "level_2_16k.json.gz": "beb074d73a841d19a5852927e62a42cc83e9e05b29596260c6fd25a76583c433",
    "level_2_64k.json.gz": "f4656c2a33cd2a31e2b27a6348bbc387fdfde0b6de71345b4f2a0ba3906dd5d8",
    "level_3_0k.json.gz": "14fb9ac1aaa51933f477e3d7bb0069ee5d612ff88716c7a7f7012454e91be06b",
    "level_3_16k.json.gz": "00cd968647bd378674e0f0781432213e0bbd8e6369b00211b60bfa7442bfd072",
    "level_3_64k.json.gz": "b05e691baa0b3d1c3e2231951aef9d3db6abdaab8b0bbdf8898f63cea477e5e4",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify archives and any extracted files without extracting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    for archive_name, archive_hash in ARCHIVE_SHA256.items():
        archive = ARCHIVE_DIR / archive_name
        if not archive.exists():
            raise FileNotFoundError(f"missing task archive: {archive}")
        actual_archive_hash = sha256_file(archive)
        if actual_archive_hash != archive_hash:
            raise ValueError(
                f"{archive}: expected sha256={archive_hash}, "
                f"found {actual_archive_hash}"
            )

        target_name = archive_name[:-3]
        target = TASKS_DIR / target_name
        expected_raw_hash = RAW_SHA256[target_name]
        if target.exists() and sha256_file(target) == expected_raw_hash:
            print(f"[ok] {target_name}")
            continue
        if args.verify_only:
            raise FileNotFoundError(
                f"{target} is absent or invalid; rerun without --verify-only"
            )

        temporary = target.with_suffix(target.suffix + ".tmp")
        with gzip.open(archive, "rb") as source, temporary.open("wb") as sink:
            shutil.copyfileobj(source, sink, length=1024 * 1024)
        actual_raw_hash = sha256_file(temporary)
        if actual_raw_hash != expected_raw_hash:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"{archive}: decompressed sha256 mismatch; expected "
                f"{expected_raw_hash}, found {actual_raw_hash}"
            )
        temporary.replace(target)
        print(f"[extract] {target_name}")

    print(f"[complete] exact task files are ready in {TASKS_DIR}")


if __name__ == "__main__":
    main()
