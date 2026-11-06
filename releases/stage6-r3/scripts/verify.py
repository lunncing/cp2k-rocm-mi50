#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Verify payload inventory, hashes, modes, source provenance and shell syntax offline."""
import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True

from bundle import (NAME, ROOT, REQUIRED, SOURCE_SHA256, SOURCE_OVERRIDES, apply_test_override,
                    digest, file_hash, record, safe_name, source_members)


def manifest_entries(manifest):
    if manifest.get("format") != 1 or manifest.get("package") != NAME:
        raise ValueError("unsupported package manifest")
    if manifest.get("source_archive_sha256") != SOURCE_SHA256:
        raise ValueError("manifest has wrong original source provenance")
    if manifest.get("source_overrides") != SOURCE_OVERRIDES:
        raise ValueError("manifest test overlay provenance mismatch")
    entries = {}
    for entry in manifest["entries"]:
        name = entry["path"]
        safe_name(name)
        if name in entries or name in ("manifest.json", "SHA256SUMS"):
            raise ValueError(f"duplicate or recursive manifest entry: {name}")
        if entry["type"] not in ("file", "dir") or entry["mode"] & ~0o777:
            raise ValueError(f"unsupported manifest entry: {name}")
        entries[name] = entry
    if manifest["source_regular_files"] != sum(
            name.startswith("source/") and e["type"] == "file" for name, e in entries.items()):
        raise ValueError("source count mismatch")
    for path in REQUIRED:
        if path not in entries:
            raise ValueError(f"required package path missing: {path}")
    return entries


def check_sums(sums, entries, manifest_data):
    actual = {}
    for line in sums.decode("utf-8").splitlines():
        sha, name = line.split("  ", 1)
        safe_name(name)
        if name in actual:
            raise ValueError("duplicate SHA256SUMS entry")
        actual[name] = sha
    expected = {name: e["sha256"] for name, e in entries.items() if e["type"] == "file"}
    expected["manifest.json"] = digest(manifest_data)
    if actual != expected:
        raise ValueError("SHA256SUMS coverage/hash mismatch")


def verify_tree(root):
    root = root.resolve()
    manifest_data = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_data)
    entries = manifest_entries(manifest)
    check_sums((root / "SHA256SUMS").read_bytes(), entries, manifest_data)
    actual_names = set()
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        actual_names.add(name)
        if path.is_symlink():
            raise ValueError(f"unexpected symlink in package: {name}")
        st = path.stat()
        if not (stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode)):
            raise ValueError(f"special file in package: {name}")
        if name in ("manifest.json", "SHA256SUMS"):
            if not path.is_file():
                raise ValueError(f"not a regular file: {name}")
            if os.name != "nt" and stat.S_IMODE(st.st_mode) != 0o644:
                raise ValueError(f"metadata mode mismatch: {name}")
            continue
        expected = entries.get(name)
        if expected is None:
            raise ValueError(f"unmanifested package path: {name}")
        if (expected["type"] == "dir") != path.is_dir():
            raise ValueError(f"type mismatch: {name}")
        if os.name != "nt" and stat.S_IMODE(st.st_mode) != expected["mode"]:
            raise ValueError(f"extracted mode mismatch: {name}")
        if path.is_file() and (st.st_size != expected["size"] or file_hash(path) != expected["sha256"]):
            raise ValueError(f"extracted hash/size mismatch: {name}")
    if actual_names != set(entries) | {"manifest.json", "SHA256SUMS"}:
        raise ValueError("extracted inventory differs from manifest")
    for name in entries:
        if name.endswith(".sh") and not name.startswith("source/"):
            data = (root / name).read_bytes()
            if b"\r" in data:
                raise ValueError(f"non-LF shell helper: {name}")
            subprocess.run(["bash", "-n", str(root / name)], check=True)
    print(f"PASS extracted inventory/hashes and bash -n: {manifest['source_regular_files']} source files", flush=True)
    if os.name == "nt":
        print("NOTE Windows extracted modes are not authoritative; verify tar modes and a Linux extraction")
    return entries


def verify_archive(archive, root, extracted):
    actual = {}
    metadata = {}
    with tarfile.open(archive, "r:gz") as tar:
        seen = set()
        for member in tar:
            name = member.name
            safe_name(name)
            if name in seen:
                raise ValueError(f"duplicate archive member: {name}")
            seen.add(name)
            if not (member.isdir() or member.isfile()) or member.mode & ~0o777:
                raise ValueError(f"unsafe archive member: {name}")
            if name == NAME:
                if not member.isdir() or member.mode != 0o755:
                    raise ValueError("invalid package root")
                continue
            if not name.startswith(NAME + "/"):
                raise ValueError("unexpected archive root")
            name = name[len(NAME) + 1:]
            data = tar.extractfile(member).read() if member.isfile() else None
            entry = record(name, member.mode, data)
            if name in ("manifest.json", "SHA256SUMS"):
                if data is None or member.mode != 0o644:
                    raise ValueError("invalid archive metadata")
                metadata[name] = data
            else:
                actual[name] = entry
        if NAME not in seen:
            raise ValueError("archive lacks package root")
    if metadata.get("manifest.json") != (root / "manifest.json").read_bytes():
        raise ValueError("archive/extracted manifest differs")
    if metadata.get("SHA256SUMS") != (root / "SHA256SUMS").read_bytes():
        raise ValueError("archive/extracted SHA256SUMS differs")
    if actual != extracted:
        raise ValueError("archive hashes/modes/inventory differ from extracted manifest")
    print(f"PASS archive vs extraction: {file_hash(archive)}", flush=True)


def verify_original(source_tar, entries, root=ROOT):
    if file_hash(source_tar) != SOURCE_SHA256:
        raise ValueError("original source tar SHA-256 mismatch")
    original = {}
    with tarfile.open(source_tar, "r:gz") as tar:
        for name, member in source_members(tar).items():
            data = tar.extractfile(member).read() if member.isfile() else None
            if data is not None:
                data = apply_test_override(root, name, data)
            name = "source" + ("/" + name if name else "")
            original[name] = record(name, member.mode, data)
    if original != {n: e for n, e in entries.items() if n == "source" or n.startswith("source/")}:
        raise ValueError("source bytes/modes/inventory differ from pinned original tar")
    print("PASS original source tar: all physics bytes and all modes match; one pinned test-only overlay and its diff verified", flush=True)


def run_mocks(root, selection):
    suites = ["pw_gpu_residency", "elpa_gpu"] if selection == "all" else [
        {"pw": "pw_gpu_residency", "elpa": "elpa_gpu"}[selection]]
    # Fail before invoking suites rather than silently skipping missing tools.
    required = ["gfortran"]
    if "pw_gpu_residency" in suites:
        required += ["gcc", "g++"]
        subprocess.run([sys.executable, "-c", "import fypp"], check=True)
    for tool in required:
        if shutil.which(tool) is None:
            raise ValueError(f"mock prerequisite missing: {tool}")
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", CC="gcc", CXX="g++", FC="gfortran")
    failed = []
    # Existing suites create .build-* below tests. Keep the distributed source untouched.
    with tempfile.TemporaryDirectory(prefix="cp2k-package-mocks-") as temporary:
        copied = Path(temporary) / "source"
        shutil.copytree(root / "source", copied)
        for suite in suites:
            print(f"RUN existing {suite} on a disposable source copy", flush=True)
            status = subprocess.call([sys.executable, "-B", str(copied / "tests" / suite / "run_tests.py")],
                                     cwd=copied, env=env)
            print(f"{'FAIL' if status else 'PASS'} {suite}: exit {status}", flush=True)
            if status:
                failed.append(suite)
    if failed:
        raise ValueError("existing mock suite failure: " + ", ".join(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--source-tar", type=Path)
    parser.add_argument("--mock", choices=("pw", "elpa", "all"),
                        help="all selects only PW residency + ELPA, not all CP2K tests")
    parser.add_argument("--launcher-tests", action="store_true")
    args = parser.parse_args()
    entries = verify_tree(args.root)
    if args.archive:
        verify_archive(args.archive, args.root, entries)
    if args.source_tar:
        verify_original(args.source_tar, entries, args.root)
    if args.launcher_tests:
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts/test_packaging.py")], check=True)
    if args.mock:
        run_mocks(args.root, args.mock)


if __name__ == "__main__":
    # Imports must not add __pycache__ to the verified package on subsequent runs.
    try:
        main()
    except (ValueError, OSError, KeyError, tarfile.TarError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ERROR: {error}")
