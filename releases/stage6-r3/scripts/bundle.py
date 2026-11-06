#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Shared, offline archive format. Only regular files and directories are allowed."""
import hashlib
import difflib
import json
from pathlib import Path, PurePosixPath
import re

NAME = "cp2k-2026.2-mi50-stage6-source-20261106-r3"
SOURCE_SHA256 = "d45fe70edbf96c571c3100c13fc65f6f581056efa9224fa8fbe40119e5dd2fd0"
EPOCH = 1793923200  # 2026-11-06 00:00:00 UTC
ROOT = Path(__file__).resolve().parents[1]
HELPERS = (
    "README.md", "LICENSE", "NOTICE", "recipe/README.md", "recipe/dependencies.json",
    "recipe/build_inside_container.sh", "scripts/build.sh", "scripts/run.sh",
    "scripts/verify.sh", "scripts/package.py", "scripts/bundle.py",
    "scripts/launcher.py", "scripts/verify.py", "scripts/test_packaging.py",
    "overrides/source/tests/elpa_gpu/test_elpa.F", "patches/elpa-test-expectation.patch",
)
SOURCE_OVERRIDES = {
    "tests/elpa_gpu/test_elpa.F": {
        "original_sha256": "560da0f8cc5629c62cd26c0674029b56b35305a41ae189a454c59dcd8410a885",
        "replacement_sha256": "5c9833bf8ea45cffc1af1722dba8ddc682d7a8b4905f12703cabceb5a717dadb",
        "reason": "Test-only repair of stale HIP GPU expectations; explicit CPU stays CPU, with AUTO assertions. No production source change.",
        "diff": "patches/elpa-test-expectation.patch",
    },
}
REQUIRED = (
    *HELPERS, "source/CMakeLists.txt", "source/REVISION", "source/LICENSE",
    "source/cmake", "source/src/start/cp2k.F", "source/data/BASIS_MOLOPT",
    "source/data/POTENTIAL", "source/tools", "source/tests",
    "source/tests/pw_gpu_residency/run_tests.py", "source/tests/elpa_gpu/run_tests.py",
)
SOURCE_TOPS = set("""CODEOWNERS INSTALL.md CONTRIBUTING.md benchmarks REVISION
README.md .gitignore .git-blame-ignore-revs CMakeLists.txt LICENSE .packit.yaml
.gitmodules .gitattributes make_cp2k.sh make_pretty.sh .dockerignore .fmf
ci cmake data docs src tests tools""".split())
SECRET_PATTERN = re.compile(
    rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|"
    rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,}|AKIA[0-9A-Z]{16})"
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def safe_name(name):
    if not name or "\\" in name or any(ord(c) < 32 for c in name):
        raise ValueError(f"unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError(f"unsafe archive path: {name!r}")
    return path


def excluded(name):
    parts = PurePosixPath(name).parts
    return any(p in ("__pycache__", ".git", "SPEC.md", "REPORT.md", "AGENTS.md")
               or p.startswith(".build-") or p.endswith(".pyc") for p in parts)


def source_members(tar):
    """Validate before extraction/packaging; return relative-name -> TarInfo."""
    members = {}
    for item in tar.getmembers():
        name = item.name[2:] if item.name.startswith("./") else item.name
        if name == ".":
            name = ""
        if not name:
            if not item.isdir():
                raise ValueError("source root must be a directory")
            members[name] = item
            continue
        path = safe_name(name)
        if name in members:
            raise ValueError(f"duplicate source member: {name}")
        if not (item.isfile() or item.isdir()) or item.mode & 0o7000:
            raise ValueError(f"unsupported source member or special mode: {name}")
        if path.parts[0] not in SOURCE_TOPS:
            raise ValueError(f"unreviewed source top-level path: {name}")
        if excluded(name):
            continue
        members[name] = item
    return members


def record(name, mode, data=None):
    entry = {"path": name, "mode": mode, "type": "dir" if data is None else "file"}
    if data is not None:
        entry.update(size=len(data), sha256=digest(data))
    return entry


def apply_test_override(root, name, original):
    if name not in SOURCE_OVERRIDES:
        return original
    change = SOURCE_OVERRIDES[name]
    replacement = (root / "overrides/source" / name).read_bytes()
    if digest(original) != change["original_sha256"] or digest(replacement) != change["replacement_sha256"]:
        raise ValueError(f"allowlisted test overlay hash mismatch: {name}")
    diff = "".join(difflib.unified_diff(original.decode().splitlines(True), replacement.decode().splitlines(True),
                                      fromfile="a/source/" + name, tofile="b/source/" + name)).encode()
    if diff != (root / change["diff"]).read_bytes():
        raise ValueError(f"test overlay public diff mismatch: {name}")
    return replacement
