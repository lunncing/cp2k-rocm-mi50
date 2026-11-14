# Stage 8 validation and deployment summary

This summary describes the accepted 2026-11-14 installations and the limits of those results. Source implementation notes are in `STAGE8.md`; current operation is documented in `STAGE8-MANUAL.md`. Publication does not rebuild or replace the deployed installations.

## Source and installation identity

The complete source asset `stage8-full-source-final.tar.gz` has SHA256 `a190656cb2dc727ee8a53d203629a2bf85fc8231dfe8703f82c3e02d01ae34de`. Its per-file source-manifest digest is a different object: `81833f1cd07dd475e37f2bf1a8b8ca5aec750bf7f7877406e47aecb8fd8b8ad9`. Both architectures were built from the same frozen scientific source. All 37 scientific source files carried by the publication overlay were compared with the compiled snapshot, with checkout newline conversion accounted for; no scientific content difference was introduced during publication.

| Installation | Architecture | Installed libcp2k.so.2026.2 SHA256 |
|---|---|---|
| `/home/nowr/cp2k-hip/build-mi50` | gfx906 only | `6e81915c8331d74cf93d4fb8c58b486c1444aebbf51658c3210adc12f2f8306c` |
| `/home/nowr/cp2k-hip/build-9070xt` | gfx1201 only | `c787e3d5acbbd4f1b951376958491a810cbc1c677225126a4e5b0d00b51b9e8f` |

The formal launchers verify source, installation, executable/shared-library hashes, configured cache and resolved loader mappings. Both returned CHECK_OK again during publication preflight. This static check did not execute CP2K or a GPU workload. An external scientific-case mount had separately passed a real MI50 calculation during deployment. RX was accepted as a candidate before fresh configure/build/install at its final absolute path. Stage 7 directories remain as `*-stage7backup`; their wrappers require restoring the original canonical names before use.

## Numerical acceptance

| Case | MI50 formal build | RX formal build |
|---|---|---|
| 28-atom PBE UKS doublet, 500 Ry | PASS | PASS |
| O2 triplet, PBE UKS | PASS | PASS |
| O2 triplet, PBEsol original-path fallback | PASS | PASS |
| O2 triplet, revPBE original-path fallback | PASS | PASS |
| 84-atom closed-shell RKS | PASS | PASS |
| 512 Si, actual DIAGONALIZATION, all 512 atomic forces | PASS | PASS |

Fixed thresholds were not relaxed: absolute total-energy difference <= 1e-10 Ha and absolute difference in every atomic-force component <= 1e-9 Ha/bohr. SCF convergence and normal termination were required. UKS comparisons included printed integrated spin, absolute spin and alpha/beta electron counts. Explicit RKS mode required the RKS banner, one spin state and the actual printed total electron count; it did not invent alpha/beta values or zero spin.

References were same-device reliable implementations, not cross-card comparisons. All six RX cases used retained pre-switch Stage 7 RX outputs. MI50's main UKS and three O2 cases used same-card Stage 7; its 84-atom RKS used an already accepted same-card Stage 8 RKS output, and its 512 Si case used the original same-card Cholesky reference path. Those last two are not represented as newly obtained independent Stage 7 references. Original inputs, outputs, failed attempts and comparison JSONs remain in the local project archive and on the target host.

## Measured performance

The following are single-run CP2K internal total timers, excluding launcher verification and container startup. Each row compares the same input on the same card; the rows are not a hardware ranking.

| 28-atom UKS | Reference / s | Formal Stage 8 / s |
|---|---:|---:|
| MI50 | 441.538 | 92.816 |
| RX 9070 XT | 397.516 | 86.402 |

MI50 full SCF was 425.002 -> 69.656 s and XC was 297.460 -> 26.924 s. The earlier 71.626 s development-candidate measurement is not substituted for the formal 92.816 s result.

| 512 Si diagonalization | CP2K total / s | Full SCF / s | ELPA subphase / s | Cholesky restore / s |
|---|---|---|---|---|
| MI50 | 106.483 -> 80.938 | 98.775 -> 72.423 | 31.489 -> 32.028 | 32.291 -> 6.745 |
| RX | 129.687 -> 111.541 | 117.984 -> 102.922 | 31.604 -> 32.317 | 32.450 -> 17.240 |

RX's diagonalization reference overlapped early host compilation; those raw timings are not used to claim a controlled speedup. The MI50 improvement belongs to the Cholesky/matrix path, not to faster ELPA eigensolver execution. The AUTO threshold 12288 is a conservative policy above the measured matrix-size range, not an observed CPU/GPU crossover. OT-DIIS timings are not diagonalization evidence. Historical calculations that exited successfully but had an unavailable/failed trace collector do not establish a successful complete transfer trace.

## Peer-memory prototype

Bounded independent 16/64/256 MiB tests verified bidirectional peer copies, remote-memory kernel reads/writes, return copies and complete elementwise agreement. An initial destination-initialization ordering failure was preserved and corrected with explicit synchronization; correctness thresholds were not relaxed.

The 256 MiB copy measurements were approximately 14.17/14.11 GB/s. End-to-end remote RX cache versus pinned-host five-round totals were 0.059508147/0.045220651 s, 0.405763593/0.318688298 s and 1.65114225/1.291301021 s. These flows include verification and host waits; they are not isolated bandwidth measurements. The remote-cache flow was approximately 27–32% slower and was not enabled in CP2K.

No CP2K problem exceeding one card's capacity, cross-device VMM mapping, pooled VRAM or endurance claim has been validated. The prototype is not part of the formal runtime and has no supported user enable switch.

## Tests and remaining scope

The writer was Codex; independent read-only source review was performed with ZCode. The original reports and exact-read coverage remain local. The accepted runtime and explicit-RKS parser passed their independent reviews. Publication copied those source versions without application-code rework.

Publication preflight reran the runtime's 40 tests and the comparison tools' 17 tests on Linux, both passing. It also reran the PBE generator check, integrated ASan/UBSan suite, PW residency suite and ELPA suite from the publication tree; all four exited successfully. The accepted scientific implementation had previously completed integrated C/Fortran mock, kernel-body, dual-spin residency, ELPA capability/threshold and Cholesky failure-path suites, actual full builds and the molecular cases above. Updated development tests are shipped in the runtime-and-tests asset; the exact compiled full-source snapshot retains its older development-test files.

Multi-rank operation, every functional/mixed-XC combination, stress/cell optimization, long GEO_OPT/MD trajectories, arbitrary memory budgets, dual-job contention and Ester GUI operation are not generalized as accepted by these tests. The legacy Ester RX native entry was preserved as a hard link to the formal installed ELF and checked for version/loader compatibility; this was not a GUI test.

This public summary is a user-facing release record. Private agent traces, independent-review transcripts, local archive paths and internal retrospectives are not published.
