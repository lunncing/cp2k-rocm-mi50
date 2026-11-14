# Stage 8 offline validation tools

`parse_compare.py` parses retained CP2K outputs and compares energy, all atomic-force components, SCF convergence, normal termination and the applicable spin/electron-count evidence. It does not run CP2K.

From this directory:

```bash
python3 -B parse_compare.py --help
python3 -B -m unittest test_tools -v
```

UKS is the default molecular mode; use the explicit `--rks` option only for genuine RKS outputs. Use the established 1e-10 Ha energy and 1e-9 Ha/bohr force thresholds for the documented release comparisons. `--timing-only` cannot grant molecular acceptance. See `../../docs/STAGE8-MANUAL.md` for operational limits and the main repository tests for deployment-runtime coverage.

`stage8_runner.py` is the historical isolated-development runner retained unchanged because the accepted regression tests exercise its identity and directory guards. Its development root has been retired. Do not invoke its build/run commands or use it as the formal deployment entry. Current deployment and rebuilding use `runtime/stage8.py`, entirely below `/home/nowr/cp2k-hip`, with RX candidate acceptance before the canonical build.

The fixture logs are synthetic test inputs, not reported molecular results. The actual deployment outputs and private review records remain in the project evidence archives.
