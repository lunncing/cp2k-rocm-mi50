# Stage 8 maintenance: deployment and rollback

The two canonical Stage 8 installations were deployed and numerically accepted
on 2026-11-14. Daily users should follow `STAGE8-MANUAL.md`; no rebuild, rename or
certification is required to keep using those installations. This document
records the reviewed adapter's maintenance contract and the original controlled
Stage 7 to Stage 8 upgrade sequence. It does not authorize running that sequence
against an active installation.

The former RX validation build has been retired after acceptance. Historical
validation files alone cannot satisfy a future build: recreate a runnable,
source-identical candidate below the canonical project root and validate it
before a new RX canonical build. Keep `stage7.py`, `cp2k_env.sh` and both dependency
JSON files at their original contents. Preserve the original Stage 7 backups.
Do not rename any root while a job uses it. The adapter does not adjust clocks,
enable a peer pool, or set a GPU cap or host-fallback default.

Install the reviewed `stage8.py` at `/home/nowr/cp2k-hip/runtime/stage8.py` and the
unchanged supplied `CMakeCache-rx.txt` at `/home/nowr/cp2k-hip/CMakeCache-rx.txt`.
The cache SHA-256 is pinned in the adapter. Python 3.11+ is required. No Git
metadata is required in source snapshots. No package fetching is enabled.

## Prepare and build

After the reference job finishes and both devices are idle, archive each original
root exactly once. The commands fail if a backup already exists:

```sh
cd /home/nowr/cp2k-hip
test ! -e build-mi50-stage7backup && mv -T -- build-mi50 build-mi50-stage7backup
test ! -e build-9070xt-stage7backup && mv -T -- build-9070xt build-9070xt-stage7backup
```

For each selected root, coordinator stages the complete identical frozen Stage8
tree in `source/` and creates owned empty `build/`, `install/`, `logs/`, `cache/`,
`container-home/`, and `bin/` directories. Source and root ancestors must not be
symlinks. MI50 needs the original dependency tree copied into new canonical
`build-mi50/deps/`; do not symlink it into the backup. RX needs a complete copy of
the original `recipe/` tree in each RX root, including
`recipe/local-packages/Libint2`. Common ROCm, `/opt/dbcsr-rocm101`, host Spack
libraries and shared `/home/nowr/cp2k-hip/data` stay at their original locations.
Do not copy an old build cache or compilation outputs.

Only these roots are supported; there is no arbitrary-root option:

| GPU selector | Root | Execution |
| --- | --- | --- |
| `mi50` | `/home/nowr/cp2k-hip/build-mi50` | Original digest-pinned Docker image |
| `rx-validation` | `/home/nowr/cp2k-hip/build-9070xt-stage8-validation` | Host, RX UUID |
| `rx` | `/home/nowr/cp2k-hip/build-9070xt` | Host, RX UUID |

```sh
python3 -B /home/nowr/cp2k-hip/runtime/stage8.py build --gpu mi50 --jobs 16
python3 -B /home/nowr/cp2k-hip/runtime/stage8.py build --gpu rx-validation --jobs 16
/home/nowr/cp2k-hip/build-mi50/bin/cp2k.psmp --check
/home/nowr/cp2k-hip/build-9070xt-stage8-validation/bin/cp2k.psmp --check
```

Each build uses `build/cp2k` and `install/cp2k`. It creates only the selected
root's `bin/cp2k.psmp`, which handles `--check` and calls `stage8.py`. A differing
existing entry is rejected; coordinator must archive it first. Entries remain
unusable until the successful inspection creates `stage8-build.json`.

## Certify the already completed trusted MI50 build

For the coordinator's completed Stage8 source build at the exact canonical
`/home/nowr/cp2k-hip/build-mi50`, retain its existing `source/`, `build/cp2k`,
`install/cp2k`, `stage7-build.json` and original logs. Do not repeat the archival,
empty-directory preparation or compilation steps above for this build. The
original `build-mi50-stage7backup` remains immutable. Run:

```sh
python3 -B /home/nowr/cp2k-hip/runtime/stage8.py certify-mi50
/home/nowr/cp2k-hip/build-mi50/bin/cp2k.psmp --check
```

This explicit import accepts only a trusted legacy receipt from the pinned
original `stage7.py` (SHA-256
`ec1b70d8b6471e49ee67959d431082a16d78939f73e02c0e51fe3aaa0ba55157`).
Both dependency recipe hashes must match. The receipt's schema-1 identity must
exactly match a fresh legacy digest of every current source file, canonical root,
MI50 flags and image; its artifact hashes must match the installed ELFs. The
configured cache must satisfy all current MI50 flags, paths, gfx906 and profiling
checks. Source aliases must already be materialized as identical regular files;
the command neither edits sources nor accepts backup paths or arbitrary roots.

There must be no Stage8 attempt or success manifest. Certification uses
`install_entry` for the final wrapper, binds the normal Stage8 identity, and runs
a new `_inspect` through `docker` and `phase` in the pinned container with no GPU
devices. It never invokes configure, build or install. Both inspected artifact
hashes and the resolved `ldd` map must exactly match the old receipt. Inputs are
rechecked before publishing the normal complete `stage8-build.json`, including
the installed tree and cache digest; normal `installed_state`/`checked` apply.

The attempt and success manifests add `certification` with
`kind: trusted-stage7-mi50`, `receipt_sha256` and `builder_sha256`. Inspection
logs and a failure record remain in `logs/stage8-certify-*` after failure, with
no success manifest. Existing Stage7 receipts and logs are preserved. There is
no certification resume or RX import option; RX still requires fresh validation
and build. Certification establishes source/build/load identity only. Independent
scientific and numerical acceptance remains required.

## Configuration and RX validation

RX configure arguments are explicitly reconstructed from the frozen cache's
49 CP2K BOOL fields, supported CP2K STRING/PATH fields, compiler/optimization
settings, selected CMake controls and known dependency directories. INTERNAL
cache entries, computed MPI/pkg-config results and old build output directories
are not replayed. The original Spack `CMAKE_PREFIX_PATH` is preserved. Only
canonical RX `recipe/`, `include/` and `install/` prefix occurrences are relocated.
All four architecture flags use `gfx1201`; `CP2K_WITH_GPU=NONE`,
`CP2K_USE_ACCEL=HIP` and profiling OFF remain authoritative. MI50 preserves its
original feature recipe and `gfx906` architecture. Exact selected flags, paths
and all RX feature booleans must match the newly configured cache.

The cache scanner permits read-only shared-library soname aliases below the
selected root's `deps/`, `build/cp2k/{src,lib,lib64}/` and
`install/cp2k/{lib,lib64}/`. Every link hop must stay within the same listed
subtree, use a `lib*.so` or numeric-version soname, and end at a regular file.
Ancestors through the selected root must be owned directories without symlinks;
lexically ambiguous paths, directory links, dangling/cyclic links, backups,
`/media`, other roots and external targets fail. Keep accepted internal soname
links in place; no dependency duplication or link materialization is needed.
Source/recipe trees and build/install/output directories retain strict checks.
Generated output paths may still be absent immediately after configure, but
existing output links fail. Cache validation returns the unchanged cache file's
SHA-256; global `safe()` and `tree_manifest()` policies are unchanged.

Coordinator next runs independent gfx1201 numerical/capability validation using
the validation wrapper, in a dedicated external research case or
`build-9070xt-stage8-validation/validation/<case>`. Retain the existing 49-feature
capability evidence and comparison results. `--check` only inspects ELF hashes
and the resolved loader map; it never runs CP2K or selects/exposes a GPU.
It is not a numerical or runtime capability test.

After independent validation passes, coordinator writes
`build-9070xt-stage8-validation/stage8-validation.json`:

```json
{
  "status": "PASS",
  "build_manifest_sha256": "<sha256 of validation root/stage8-build.json>",
  "evidence": {
    "validation/gfx1201.log": "<sha256 of retained validation log>"
  }
}
```

The evidence paths must name files below that root's `validation/` directory.
This is coordinator attestation of the results, not automatic interpretation of
scientific logs. The canonical RX build rechecks the validation installation,
binds the attestation to its inspected manifest and exact source digest, and
rejects changed evidence. Then:

```sh
python3 -B /home/nowr/cp2k-hip/runtime/stage8.py build --gpu rx --jobs 16
/home/nowr/cp2k-hip/build-9070xt/bin/cp2k.psmp --check
cd /absolute/dedicated/research/case
/home/nowr/cp2k-hip/build-9070xt/bin/cp2k.psmp -i input.inp -o output.out
```

The existing `use_mi50` / `use_9070xt` environment functions still select the
canonical entries. The inherited Stage7 policy accepts dedicated production
research directories outside the project (including sufficiently deep `/media`
calculation directories), and rejects project/source/install/backup, dependency
and system roots. Compilation outputs cannot use `/media` or backups. Shared
data remains readable. Default threads remain 16 OpenMP / 1 OpenBLAS; the proven
validated OpenMP override of 1–32 remains supported. Diagnostic overrides,
including UKS/full-pipeline, PBE/LSD, ELPA threshold and single-rank Cholesky,
survive both MI50 environment boundaries. No peer controls are passed.

## Manifests and failures

`stage8-attempt.json` binds the complete source file manifest/digest, local RX
recipe, runtime/wrapper hashes, dependency recipe identity, image, configure
flags, build environment, jobs and validation gate. `stage8-build.json` adds
actual ELF/libcp2k hashes, the complete installed tree, exact resolved `ldd` map,
configured cache digest and successful inspection log location. Stage7 RX
artifact hashes never approve a Stage8 library. Runtime checks recompute the
identity, cache, installed tree, artifact hashes and loader map.

Commands and exit statuses remain in unique `logs/stage8-*` directories.
Any configure/build/install/inspection error prevents a success manifest.
An existing attempt is never silently overwritten. Resume is explicit:

```sh
python3 -B /home/nowr/cp2k-hip/runtime/stage8.py build --gpu rx-validation --jobs 16 --resume
```

Resume requires an already successful, validated configure checkpoint and exact
source, recipe, runtime, wrapper, environment, jobs, gate and cache identity.
It skips configure, then repeats build/install/inspection into the same selected
root with new logs. No completed installation can be resumed. A configure
failure, changed inputs or changed cache requires coordinator archival of the
failed root and preparation of a fresh root at its exact canonical path; retain
the failure evidence. Do not hand-edit manifests to bypass a mismatch.

## Rollback

Stop jobs using the new roots. These commands require unused archive destinations
and the original backups; they preserve both installations and never overwrite:

```sh
cd /home/nowr/cp2k-hip
test -d build-mi50-stage7backup && test ! -e build-mi50-stage8failed && \
  mv -T -- build-mi50 build-mi50-stage8failed && \
  mv -T -- build-mi50-stage7backup build-mi50
test -d build-9070xt-stage7backup && test ! -e build-9070xt-stage8failed && \
  mv -T -- build-9070xt build-9070xt-stage8failed && \
  mv -T -- build-9070xt-stage7backup build-9070xt
/home/nowr/cp2k-hip/build-mi50/bin/cp2k.psmp --check
/home/nowr/cp2k-hip/build-9070xt/bin/cp2k.psmp --check
```

If a replacement root was never created, restore the corresponding backup with
`test ! -e build-mi50 && mv -T -- build-mi50-stage7backup build-mi50` or
`test ! -e build-9070xt && mv -T -- build-9070xt-stage7backup build-9070xt`.
Backup wrappers contain hardcoded canonical paths: never try running them
directly from `*-stage7backup`. The shared original runtime/configs remain
unchanged, so restoring the original directory names restores their references.
Retain validation and failed-build evidence. Removal of other-location
compilation products is a later coordinator action after evidence and final
checks; this runtime has no cleanup command.

## Local tests

```sh
python3 -B -m unittest discover -s tests -p test_stage8.py -v
git diff --check
```

Tests use workspace-local temporary trees and mocked process/POSIX boundaries;
they never invoke Docker, CMake, CP2K, GPUs, or real production paths. On Windows,
the POSIX path policy is exercised with a path probe and file operations are
confined to the workspace. Real Linux builds, loader inspection, permissions,
device selection and numerical validation remain coordinator work.
The cache regression also forces project-directory matches on Windows and
asserts the actual cache file digest. Certification tests mock Docker/process
execution while retaining real phase logs, wrapper creation and manifest checks.
Cache symlink regressions create actual temporary links, using the production
`safe()` on Linux. They cover internal soname chains and unchanged cache bytes/
digest, escaping links (including chains that leave and return), directory links
and symlink ancestors, source/recipe protection, missing outputs versus output
links, dangling/cyclic links and ancestor ownership. Windows without symlink
privilege reports these as explicit `NOT-RUN` skips. A separate narrow test mocks
only link metadata while retaining real cache bytes, directories and target file;
Windows fixtures adapt only path separators and the existing local path guard.
These mocks do not replace the coordinator's actual Linux symlink suite run.
