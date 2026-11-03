#!/usr/bin/env python3
"""Materialize checksum-verified historical data without making API calls."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def contained_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        raise ValueError(f"Path escapes repository: {value}") from None
    return path


def write_verified(path: Path, payload: bytes, overwrite: bool) -> None:
    if path.exists():
        if path.read_bytes() == payload:
            return
        if not overwrite:
            raise FileExistsError(
                f"Refusing to replace modified data: {path}. "
                "Use a fresh clone or explicitly pass --overwrite."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(payload)
    temporary.replace(path)


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256(destination.read_bytes()) != sha256(source.read_bytes()):
            raise FileExistsError(f"Conflicting legacy input: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(os.path.relpath(source, destination.parent))
    except OSError:
        shutil.copyfile(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores-only', action='store_true',
                        help='Extract evaluator scores only; enough for offline paper tables.')
    parser.add_argument('--verify-only', action='store_true',
                        help='Verify all selected archives without writing files.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'data/reference_manifest.json').read_text())
    count = 0
    for entry in manifest['entries']:
        if args.scores_only and not Path(entry['destination']).name.startswith('eval_'):
            continue
        archive = contained_path(entry['archive'])
        destination = contained_path(entry['destination'])
        compressed = archive.read_bytes()
        if sha256(compressed) != entry['sha256']:
            raise ValueError(f'Archive checksum mismatch: {archive}')
        payload = gzip.decompress(compressed)
        if sha256(payload) != entry['raw_sha256']:
            raise ValueError(f'Payload checksum mismatch: {archive}')
        json.loads(payload)
        if not args.verify_only:
            write_verified(destination, payload, args.overwrite)
        count += 1
    if not args.verify_only and not args.scores_only:
        for source in (ROOT / 'data/source').glob('*.json'):
            link_or_copy(source, ROOT / 'paper/results' / source.name)
        for source in (ROOT / 'data/tasks').glob('level_*.json'):
            link_or_copy(source, ROOT / 'paper/generated_tasks/gemini-2.0-flash-001' / source.name)
    print(f'[complete] {count} reference archives verified' +
          ('' if args.verify_only else ' and materialized'))


if __name__ == '__main__':
    main()

