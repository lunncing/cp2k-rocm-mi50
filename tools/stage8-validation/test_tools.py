"""Bounded local cases. All CP2K test text is synthetic; no Docker/GPU calls."""
from contextlib import redirect_stdout
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import parse_compare as parser
import stage8_runner as runner

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"


class ParserTests(unittest.TestCase):
    def parse_text(self, text, natoms=2, timing_only=False, rks=False):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "synthetic.log"
            path.write_text(text, encoding="utf-8")
            return parser.parse_log(path, natoms, timing_only, rks=rks)

    def rks_text(self):
        text = (FIXTURES / "new_d.log").read_text()
        return text.replace(
            " Spin 1\n Number of electrons: 3\n Spin 2\n Number of electrons: 2\n",
            " DFT| Spin restricted Kohn-Sham (RKS) calculation                            RKS\n"
            " DFT| Number of spin states                                                    1\n"
            " Number of electrons:                                                        256\n").replace(
                " Integrated spin density: 1.0D+00\n", "").replace(
                    " Integrated absolute spin density: 1.2D+00\n", "")

    def test_rks_valid_observed_total_and_spin_not_applicable(self):
        parsed = self.parse_text(self.rks_text(), rks=True)
        self.assertEqual(set(parsed["electron_counts"]), {"total"})
        self.assertEqual(parsed["electron_counts"]["total"]["value"], "256")
        self.assertIsNone(parsed["integrated_spin"])
        self.assertIsNone(parsed["integrated_absolute_spin"])
        self.assertEqual(parsed["spin_validation"]["status"], "not_applicable_RKS")
        self.assertIn("Spin restricted Kohn-Sham (RKS)", parsed["spin_validation"]["banner_evidence"][0]["raw"])
        self.assertTrue(parsed["spin_validation"]["spin_states_evidence"][0]["raw"].endswith("1"))
        result = parser.compare(parsed, parsed, Decimal("1e-9"), Decimal("1e-10"), Decimal(0), rks=True)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["molecular_acceptance"])
        self.assertEqual(result["spin_validation"]["status"], "not_applicable_RKS")
        self.assertNotIn("integrated_spin", result)
        self.assertNotIn("integrated_absolute_spin", result)
        self.assertEqual(result["electron_counts"]["total"]["threshold"], "0")
        self.assertEqual(len(result["force_components"]), 6)

    def test_rks_is_explicit_and_requires_both_logs(self):
        source = (FIXTURES / "new_d.log").read_text()
        for text in (self.rks_text(), source.replace("Integrated spin density: 1.0D+00", "")):
            with self.subTest(text=text[:100]), self.assertRaisesRegex(ValueError, "Missing molecular data"):
                self.parse_text(text)
        uks = self.parse_text(source)
        rks = self.parse_text(self.rks_text(), rks=True)
        for reference, candidate in ((uks, rks), (rks, uks)):
            with self.assertRaisesRegex(ValueError, "RKS evidence"):
                parser.compare(reference, candidate, Decimal("1e-9"), Decimal("1e-10"), Decimal(0), rks=True)

    def test_rks_missing_or_contradictory_evidence_rejected(self):
        source = self.rks_text()
        banner = " DFT| Spin restricted Kohn-Sham (RKS) calculation                            RKS\n"
        states = " DFT| Number of spin states                                                    1\n"
        uks = (FIXTURES / "new_d.log").read_text()
        for text in (source.replace(banner, ""), source.replace(states, ""),
                     source.replace(banner, banner.replace("RKS\n", "UKS\n")),
                     source.replace(states, states.replace("1\n", "2\n")),
                     source.replace(states, states + states.replace("1\n", "2\n")),
                     source.replace(banner, banner + " DFT| Spin-unrestricted calculation\n"),
                     uks.replace(" Spin 1", " DFT| Spin unrestricted Kohn-Sham (UKS) calculation UKS\n Spin 1"),
                     uks.replace(" Integrated spin density: 1.0D+00\n", "")):
            with self.subTest(text=text[:300]), self.assertRaisesRegex(ValueError, "RKS evidence"):
                self.parse_text(text, rks=True)

    def test_rks_total_electrons_required_finite_and_exact(self):
        source = self.rks_text()
        electrons = " Number of electrons:                                                        256\n"
        with self.assertRaisesRegex(ValueError, "total electron count"):
            self.parse_text(source.replace(electrons, ""), rks=True)
        for value in ("NaN", "Infinity", "-Inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse_text(source.replace("256", value), rks=True)
        reference = self.parse_text(source, rks=True)
        for value in ("257", "256.0000000001"):
            changed = self.parse_text(source.replace("256", value), rks=True)
            result = parser.compare(reference, changed, Decimal("1e-9"), Decimal("1e-10"), Decimal(0), rks=True)
            self.assertFalse(result["accepted"])
            self.assertFalse(result["electron_counts"]["total"]["within_threshold"])

    def test_rks_preserves_force_energy_and_completion_validation(self):
        source = self.rks_text()
        for text in (source.replace("FORCES| Sum 0.0 0.0 0.0", ""),
                     source.replace("FORCES| 2 -1.0D-03 2.0D-03 -3.0D-03 3.741657D-03", ""),
                     source.replace("FORCES| 2 ", "FORCES| 1 "),
                     source.replace("3.0D-03", "NaN"),
                     source.replace("[hartree/bohr]", "[eV/angstrom]"),
                     source.replace("[hartree] -2.6D+01", "[hartree] NaN"),
                     source.replace("converged in 2 steps", "NOT converged"),
                     source.replace("PROGRAM ENDED AT", "INTERRUPTED AT")):
            with self.subTest(text=text[-100:]), self.assertRaises(ValueError):
                self.parse_text(text, rks=True)
        reference = self.parse_text(source, rks=True)
        for old, new, field in (
                ("[hartree] -2.6D+01", "[hartree] -26.00000000011", "energy"),
                ("FORCES| 2 -1.0D-03 2.0D-03 -3.0D-03",
                 "FORCES| 2 -1.0D-03 2.0D-03 -0.0030000011", "force_components")):
            changed = self.parse_text(source.replace(old, new), rks=True)
            result = parser.compare(reference, changed, Decimal("1e-9"), Decimal("1e-10"), Decimal(0), rks=True)
            self.assertFalse(result["accepted"])
            check = result[field][-1] if field == "force_components" else result[field]
            self.assertFalse(check["within_threshold"])

    def test_rks_timing_only_never_accepts(self):
        source = self.rks_text().replace("converged in 2 steps", "NOT converged")
        parsed = self.parse_text(source, timing_only=True, rks=True)
        result = parser.compare(parsed, parsed, None, None, None, timing_only=True, rks=True)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["molecular_acceptance"])
        self.assertTrue(result["diagnostic_only"])

    def test_rks_cli_exit_codes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            reference = Path(directory) / "synthetic-reference.log"
            candidate = Path(directory) / "synthetic-candidate.log"
            source = self.rks_text()
            reference.write_text(source, encoding="utf-8")
            arguments = ["parse_compare.py", str(reference), str(candidate), "--natoms", "2",
                         "--force-atol", "1e-9", "--energy-atol", "1e-10", "--spin-atol", "0"]
            for text, rks, expected in (
                    (source, True, 0), (source, False, 2),
                    (source.replace("256", "257"), True, 1),
                    (source.replace(" Number of electrons:                                                        256\n", ""), True, 2),
                    ((FIXTURES / "new_d.log").read_text(), True, 2)):
                candidate.write_text(text, encoding="utf-8")
                output = io.StringIO()
                with self.subTest(expected=expected, rks=rks), patch(
                        "sys.argv", arguments + (["--rks"] if rks else [])), redirect_stdout(output):
                    self.assertEqual(parser.main(), expected)
                self.assertEqual(json.loads(output.getvalue())["accepted"], expected == 0)

    def test_both_formats_d_exponents_and_total_maximum(self):
        current = parser.parse_log(FIXTURES / "new_d.log", 2)
        legacy = parser.parse_log(FIXTURES / "old.log", 2)
        result = parser.compare(current, legacy, Decimal(0), Decimal(0), Decimal(0))
        self.assertTrue(result["accepted"])
        self.assertEqual(len(result["force_components"]), 6)
        self.assertEqual(current["timings"][0]["total_maximum_seconds"]["value"], "20.0")
        self.assertEqual(current["timings"][0]["self_maximum_seconds"]["value"], "0.2")
        self.assertEqual(current["total_time_seconds"]["value"], "20.0")
        self.assertEqual(current["cold_start"]["statistics"]["sum_seconds"], 12)
        self.assertEqual(current["steady_state"]["statistics"]["sum_seconds"], 2)
        self.assertEqual(len(current["scf_updates"]), 2)
        self.assertEqual(current["electron_counts"]["alpha"]["value"], "3")

    def test_appended_interrupted_complete_and_trailing_interrupted(self):
        complete = (FIXTURES / "new_d.log").read_text()
        interrupted = complete.split("*** SCF run converged")[0]
        result = self.parse_text(interrupted + complete + interrupted)
        self.assertEqual(result["selection"]["selected_invocation"], 2)
        self.assertEqual(result["selection"]["ignored_later_incomplete_invocations"], 1)
        self.assertEqual(result["energy"]["value"], "-26")
        with self.assertRaisesRegex(ValueError, "No complete"):
            self.parse_text(interrupted)

    def test_force_count_duplicate_missing_and_nonfinite(self):
        source = (FIXTURES / "new_d.log").read_text()
        for text, count in ((source, 3), (source.replace("FORCES| 2 ", "FORCES| 1 "), 2),
                            (source.replace("FORCES| 2 -1.0D-03 2.0D-03 -3.0D-03 3.741657D-03", ""), 2),
                            (source.replace("3.0D-03", "NaN"), 2),
                            (source.replace(" 2 Broy./Diag.", " 4 Broy./Diag."), 2),
                            (source.replace("converged in 2 steps", "converged in 3 steps"), 2)):
            with self.subTest(text=text[-70:], count=count):
                with self.assertRaises(ValueError):
                    self.parse_text(text, count)

    def test_unconverged_diagnostic_never_accepts(self):
        text = (FIXTURES / "new_d.log").read_text().replace(
            "*** SCF run converged in 2 steps ***", "*** SCF run NOT converged ***")
        with self.assertRaisesRegex(ValueError, "not converged"):
            self.parse_text(text)
        diagnostic = self.parse_text(text, timing_only=True)
        result = parser.compare(diagnostic, diagnostic, None, None, None, timing_only=True)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["molecular_acceptance"])
        self.assertTrue(result["diagnostic_only"])

    def test_thresholds_every_component_and_spin(self):
        source = (FIXTURES / "new_d.log").read_text()
        reference = self.parse_text(source)
        altered = self.parse_text(source.replace("FORCES| 2 -1.0D-03 2.0D-03 -3.0D-03",
                                                "FORCES| 2 -1.0D-03 2.0D-03 -3.1D-03"))
        result = parser.compare(reference, altered, Decimal("0.00001"), Decimal(0), Decimal(0))
        self.assertFalse(result["accepted"])
        self.assertFalse(result["force_components"][-1]["within_threshold"])
        spin = self.parse_text(source.replace("spin density: 1.0D+00", "spin density: 1.1D+00"))
        self.assertFalse(parser.compare(reference, spin, Decimal(0), Decimal(0), Decimal("0.01"))["accepted"])
        with self.assertRaises(ValueError):
            parser.compare(reference, reference, None, Decimal(0), Decimal(0))

    def test_missing_spin_and_incomplete_force_block(self):
        text = (FIXTURES / "new_d.log").read_text()
        for modified in (text.replace("Integrated spin density: 1.0D+00", ""),
                         text.replace("FORCES| Sum 0.0 0.0 0.0", "")):
            with self.assertRaises(ValueError):
                self.parse_text(modified)


class RunnerTests(unittest.TestCase):
    def configuration(self):
        return {"rocm": "/opt/rocm/core-10.1", "mounts": [],
                "runtime_mounts": [{"source": "/home/nowr/cp2k-hip/data", "target": "/home/nowr/cp2k-hip/data"}],
                "library_paths": ["{prefix}/install/cp2k/lib", "{prefix}/deps/lib"],
                "pkg_config_paths": [], "prefix_paths": [], "configure_args": []}

    def test_exact_data_mount_allowed_formal_install_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "config.json"
            config = self.configuration()
            path.write_text(json.dumps(config))
            with patch.object(runner, "RUNTIME_CFG", path), patch.object(runner, "exact_path", side_effect=Path):
                self.assertEqual(runner.cfg(), config)
                for forbidden in ("/home", "/home/nowr/cp2k-hip/source", "/home/nowr/cp2k-hip/build-mi50/install",
                                  "/home/nowr/cp2k-hip/build-9070xt", "/home/nowr/cp2k-hip/data/subdir"):
                    config["runtime_mounts"] = [{"source": forbidden, "target": forbidden}]
                    path.write_text(json.dumps(config))
                    with self.assertRaises(ValueError):
                        runner.cfg()

    def test_gate_and_dependency_mapping(self):
        with patch.object(runner, "cfg", return_value=self.configuration()):
            for value in ("0", "1"):
                self.assertEqual(runner.make_env(False, {"CP2K_CHOLESKY_FAST_SINGLE_RANK": value})[
                    "CP2K_CHOLESKY_FAST_SINGLE_RANK"], value)
            for value in ("yes", "2", "1\n"):
                with self.assertRaises(ValueError):
                    runner.make_env(False, {"CP2K_CHOLESKY_FAST_SINGLE_RANK": value})
            self.assertTrue(runner.make_env(False, {})["LD_LIBRARY_PATH"].startswith(str(runner.STAGE8_ROOT)))
            with self.assertRaises(ValueError):
                runner.map_path("/home/nowr/cp2k-hip/build-mi50/install/lib")
            self.assertIn(str(runner.STAGE8_ROOT), runner.map_path("{prefix}/install/cp2k/lib"))

    def test_case_protection_and_docker_no_gpu_check(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory).resolve()
            stage = root / "development"
            case = root / "cases" / "one"
            case.mkdir(parents=True)
            (stage / "source").mkdir(parents=True)
            config = self.configuration()
            with patch.object(runner, "STAGE8_ROOT", stage), patch.object(runner, "cfg", return_value=config):
                self.assertEqual(runner.validate_case(case), case)
                with self.assertRaises(ValueError):
                    runner.validate_case(stage / "source")
                with self.assertRaises(ValueError):
                    runner.validate_case(root.parent)
                with patch.object(os, "getuid", return_value=1000, create=True), patch.object(
                        os, "getgid", return_value=1000, create=True), patch.object(
                        runner, "select_mi50", side_effect=AssertionError("GPU selection during check")):
                    command = runner.docker_cmd(["ldd", "candidate"], {}, mode="check")
                self.assertNotIn("--device", command)
                self.assertIn("core=0", command)
                self.assertIn("type=bind,source=/home/nowr/cp2k-hip/data,target=/home/nowr/cp2k-hip/data,readonly", command)
                self.assertTrue(all("readonly" in command[index + 1] for index, value in enumerate(command)
                                    if value == "--mount"))

    def test_audit_rejects_production_lib_and_hash_mismatch(self):
        from subprocess import CompletedProcess
        executable = {"resolved_path": "/candidate", "sha256": "a" * 64}
        result = CompletedProcess([], 0, "libcp2k.so => /home/nowr/cp2k-hip/build-mi50/install/lib/libcp2k.so (0x1)\n", "")
        with patch.object(runner, "binary_path", return_value=Path("/candidate")), patch.object(
                runner, "elf_record", return_value=executable), patch.object(
                runner.subprocess, "run", return_value=result), patch.object(Path, "resolve", lambda self, **kw: self):
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Non-development"):
                runner.audit("/candidate")
            result.stdout = f"libcp2k.so => {runner.STAGE8_ROOT.as_posix()}/install/cp2k/lib/libcp2k.so (0x1)\n"
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                runner.audit("/candidate", "b" * 64)


def run_suite(test_case):
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(test_case))
    return 0 if result.wasSuccessful() else 1


def run_parser_tests():
    return run_suite(ParserTests)


def run_runner_tests():
    return run_suite(RunnerTests)


if __name__ == "__main__":
    unittest.main(verbosity=2)
