#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Build the deterministic source handoff from the pinned original Linux tar."""
import argparse
import gzip
import io
import os
from pathlib import Path
import tarfile
import tempfile
import sys

sys.dont_write_bytecode = True

from bundle import (EPOCH, HELPERS, NAME, ROOT, SECRET_PATTERN, SOURCE_SHA256, SOURCE_OVERRIDES,
                    apply_test_override, file_hash, json_bytes, record, source_members)


def create(source, output):
    if file_hash(source) != SOURCE_SHA256:
        raise ValueError("original source archive SHA-256 mismatch; refusing substitute source")
    output.mkdir(parents=True, exist_ok=True)
    final = output / (NAME + ".tar.gz")
    entries = {}
    payload = {}
    with tarfile.open(source, "r:gz") as original:
        members = source_members(original)
        for name, item in members.items():
            dest = "source" + ("/" + name if name else "")
            data = original.extractfile(item).read() if item.isfile() else None
            if data is not None:
                data = apply_test_override(ROOT, name, data)
            if data is not None and SECRET_PATTERN.search(data):
                raise ValueError(f"private path/credential pattern in {dest}; review required")
            entries[dest] = record(dest, item.mode, data)
            payload[dest] = data
        for name in HELPERS:
            path = ROOT / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"missing or symlinked helper: {name}")
            data = path.read_bytes()
            # The scanner's own pattern strings intentionally name excluded paths.
            if name != "scripts/bundle.py" and SECRET_PATTERN.search(data):
                raise ValueError(f"private path/credential pattern in helper {name}")
            if b"\r" in data:
                raise ValueError(f"helper must have LF line endings: {name}")
            mode = 0o755 if path.suffix in (".sh", ".py") else 0o644
            entries[name] = record(name, mode, data)
            payload[name] = data
            for parent in path.relative_to(ROOT).parents:
                if str(parent) != ".":
                    key = parent.as_posix()
                    entries[key] = record(key, 0o755)
                    payload[key] = None
    manifest = json_bytes({
        "format": 1, "package": NAME, "source_archive_sha256": SOURCE_SHA256,
        "source_overrides": SOURCE_OVERRIDES,
        "source_revision_label": "git:c92cc08 (installed snapshot; not upstream identity)",
        "source_regular_files": sum(e["type"] == "file" and n.startswith("source/")
                                    for n, e in entries.items()),
        "excluded": ["__pycache__", "*.pyc", ".build-*", "git metadata", "local handoff records"],
        "entries": [entries[n] for n in sorted(entries)],
    })
    payload["manifest.json"] = manifest
    entries["manifest.json"] = record("manifest.json", 0o644, manifest)
    sums = "".join(f"{entries[n]['sha256']}  {n}\n" for n in sorted(entries)
                   if entries[n]["type"] == "file").encode()
    payload["SHA256SUMS"] = sums
    entries["SHA256SUMS"] = record("SHA256SUMS", 0o644, sums)
    # SHA256SUMS cannot hash itself. The adjacent archive digest covers every byte.
    with tempfile.NamedTemporaryFile(prefix=".package-", dir=output, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with temporary.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                                       mtime=EPOCH, compresslevel=9) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for name in ["", *sorted(entries)]:
                    entry = entries.get(name, {"type": "dir", "mode": 0o755})
                    item = tarfile.TarInfo(NAME + ("/" + name if name else ""))
                    item.mode = entry["mode"]
                    item.mtime = EPOCH
                    item.uid = item.gid = 0
                    item.uname = item.gname = ""
                    data = payload.get(name)
                    if entry["type"] == "dir":
                        item.type = tarfile.DIRTYPE
                        tar.addfile(item)
                    else:
                        item.size = len(data)
                        tar.addfile(item, io.BytesIO(data))
        os.replace(temporary, final)
    finally:
        temporary.unlink(missing_ok=True)
    checksum = file_hash(final)
    final.with_name(final.name + ".sha256").write_text(
        f"{checksum}  {final.name}\n", encoding="utf-8", newline="\n")
    print(f"{checksum}  {final}")
    print(f"source: {sum(n.startswith('source/') and e['type'] == 'file' for n, e in entries.items())} regular files; all modes preserved; one documented test-only overlay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tar", type=Path, default=ROOT.parent / "inputs/source-stage6-target.tar.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    create(args.source_tar.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, tarfile.TarError) as error:
        raise SystemExit(f"ERROR: {error}")
