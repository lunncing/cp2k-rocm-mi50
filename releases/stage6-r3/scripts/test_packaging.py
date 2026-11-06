#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Disposable launcher/archive fixtures. No Docker daemon or GPU is used."""
import argparse
import io
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
import bundle
import launcher
import verify


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cp2k release fixtures ")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "package with spaces"
        (self.root / "recipe").mkdir(parents=True)
        (self.root / "manifest.json").write_text("fixture manifest\n")
        cfg = launcher.config()
        self.deps = self.base / "kept dependency"
        self.deps.mkdir()
        (self.deps / "sentinel").write_text("keep dependency bytes\n")
        cfg["mounts"] = [{"source": str(self.deps), "target": str(self.deps)}]
        (self.root / "recipe/dependencies.json").write_text(json.dumps(cfg))
        self.prefix = self.base / "isolated prefix"
        self.work = self.base / "job with spaces"
        self.work.mkdir()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(launcher, "ROOT", self.root).start()

    def installed(self):
        launcher.claim_prefix(self.prefix)
        self.binary = self.prefix / "install/cp2k/bin/cp2k.psmp"
        self.library = self.prefix / "install/cp2k/lib/libcp2k.so.2026.2"
        for p in (self.binary, self.library):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("fixture binary\n")
        (self.prefix / launcher.SUCCESS).write_bytes(bundle.json_bytes({
            "image": launcher.config()["image"], "manifest_sha256": bundle.file_hash(self.root / "manifest.json"),
            "binary_sha256": bundle.file_hash(self.binary), "library_sha256": bundle.file_hash(self.library)}))

    def test_reject_protected_relative_symlink_and_nonempty_prefix_without_writes(self):
        occupied = self.base / "user files"
        occupied.mkdir()
        sentinel = occupied / "important input"
        sentinel.write_text("do not change\n")
        link = self.base / "alias"
        link.symlink_to(occupied, target_is_directory=True)
        candidates = ["relative", "/", str(Path.home()), "/home/nowr/cp2k-mi50-production",
                      "/home/nowr/cp2k-mi50-production/sub", "/home/nowr/cp2k-mi50-20261103",
                      "/home/nowr/spack", "/home/nowr/MI50_CP2K_full_build_20261103/sub",
                      "/home/nowr/cp2k-hip/data", str(self.root), str(self.root / "build"),
                      str(self.base), str(occupied), str(link), str(link / "child"),
                      str(self.base / "a/../b"), str(self.base / "comma,path")]
        for path in candidates:
            with self.subTest(path=path), self.assertRaises(ValueError):
                launcher.check_prefix(path, fresh=True)
        self.assertEqual(sentinel.read_text(), "do not change\n")
        self.assertEqual((self.deps / "sentinel").read_text(), "keep dependency bytes\n")
        self.assertEqual(list(occupied.iterdir()), [sentinel])

    def test_owned_prefix_is_never_rebuilt_or_cleaned(self):
        self.installed()
        with self.assertRaises(ValueError):
            launcher.check_prefix(str(self.prefix), fresh=True)
        self.assertEqual(launcher.check_prefix(str(self.prefix)), self.prefix)
        self.assertEqual(self.binary.read_text(), "fixture binary\n")
        (self.prefix / "user-input").write_text("mine")
        with self.assertRaises(ValueError):
            launcher.check_prefix(str(self.prefix))
        self.assertEqual((self.prefix / "user-input").read_text(), "mine")

    def test_build_command_is_isolated_and_preserves_docker_exit(self):
        with mock.patch("verify.verify_tree"), mock.patch.object(launcher, "verify_dependencies"), \
             mock.patch.object(launcher.subprocess, "call", return_value=41) as call:
            status = launcher.main(["build", "--prefix", str(self.prefix), "--jobs", "7"])
        self.assertEqual(status, 41)
        command = call.call_args.args[0]
        for required in ("--rm", "--pull=never", "--network=none", "--read-only"):
            self.assertIn(required, command)
        for forbidden in ("--name", "--privileged", "--device"):
            self.assertNotIn(forbidden, command)
        self.assertEqual(command[-2:], ["--jobs", "7"])
        self.assertIn(f"type=bind,source={self.root},target={self.root},readonly", command)
        self.assertIn(f"type=bind,source={self.prefix},target={self.prefix}", command)
        self.assertIn(f"{os.getuid()}:{os.getgid()}", command)
        self.assertFalse((self.prefix / launcher.SUCCESS).exists())
        self.assertEqual((self.deps / "sentinel").read_text(), "keep dependency bytes\n")

    def test_missing_dependencies_leave_prefix_absent(self):
        with mock.patch("verify.verify_tree"), \
             mock.patch.object(launcher, "verify_dependencies", side_effect=ValueError("missing")):
            with self.assertRaises(ValueError):
                launcher.main(["build", "--prefix", str(self.prefix)])
        self.assertFalse(self.prefix.exists())

    def test_bad_jobs_and_run_arguments_fail(self):
        for argv in (["build", "--jobs", "0"], ["build", "--jobs", "-1"],
                     ["run"], ["run", "--"], ["run", "--check", "--", "-i", "a"]):
            with self.subTest(argv=argv), mock.patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                launcher.main(argv)

    def test_run_check_never_exposes_gpu(self):
        self.installed()
        with mock.patch.object(launcher.Path, "cwd", return_value=self.work), \
             mock.patch.object(launcher, "verify_dependencies"), \
             mock.patch.object(launcher, "select_gpu", side_effect=AssertionError("GPU accessed")), \
             mock.patch.object(launcher.subprocess, "call", return_value=0) as call:
            self.assertEqual(launcher.main(["run", "--prefix", str(self.prefix), "--check"]), 0)
        command = call.call_args.args[0]
        self.assertNotIn("--device", command)
        self.assertIn("--check", command)
        self.assertNotIn(str(self.work), command)
        self.assertEqual(command[command.index("--workdir") + 1], "/tmp")
        self.assertIn(f"type=bind,source={self.prefix},target={self.prefix},readonly", command)
        for child in ("cache", "container-home"):
            path = self.prefix / child
            self.assertIn(f"type=bind,source={path},target={path}", command)

    def test_runtime_arguments_env_and_only_selected_render_node(self):
        self.installed()
        render = self.base / "renderD130"
        render.touch()
        original_stat = Path.stat
        def fixture_stat(path, *args, **kwargs):
            return original_stat(render if str(path) == "/dev/kfd" else path, *args, **kwargs)
        host = {"CP2K_PW_GPU_RES_XC_DZ": "0", "CP2K_GPU_MEM_LIMIT_MB_RANK0": "14000",
                "CP2K_GPU_SECRET": "should-not-forward", "API_TOKEN": "should-not-forward",
                "HIP_VISIBLE_DEVICES": "1", "HSA_OVERRIDE_GFX_VERSION": "12.0.0",
                "LD_LIBRARY_PATH": "/wrong/lib", "OMP_NUM_THREADS": "3"}
        arguments = ["-i", "input with spaces.inp", "-o", "literal $(touch never).out"]
        with mock.patch.object(launcher.Path, "cwd", return_value=self.work), \
             mock.patch.object(launcher, "verify_dependencies"), \
             mock.patch.object(launcher, "select_gpu", return_value=(render, "0000:03:00.0")), \
             mock.patch.object(Path, "stat", fixture_stat), mock.patch.dict(os.environ, host, clear=True), \
             mock.patch.object(launcher.subprocess, "call", return_value=37) as call:
            self.assertEqual(launcher.main(["run", "--prefix", str(self.prefix), "--", *arguments]), 37)
        command = call.call_args.args[0]
        self.assertEqual(command[-len(arguments):], arguments)
        devices = [command[i+1] for i, value in enumerate(command) if value == "--device"]
        self.assertEqual(devices, ["/dev/kfd", str(render)])
        forwarded = [command[i+1] for i, value in enumerate(command) if value == "--env"]
        self.assertIn("OMP_NUM_THREADS=3", forwarded)
        self.assertIn("CP2K_PW_GPU_RES_XC_DZ=0", forwarded)
        self.assertIn("CP2K_GPU_MEM_LIMIT_MB=", forwarded)
        self.assertIn(f"ROCR_VISIBLE_DEVICES={launcher.UUID}", forwarded)
        self.assertFalse(any("should-not-forward" in arg or "HSA_OVERRIDE" in arg or "HIP_VISIBLE" in arg for arg in command))
        self.assertFalse((self.work / "never").exists())

    def test_runtime_clean_environment_and_no_default_memory_cap(self):
        env = launcher.environment(self.prefix, runtime=True)
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], launcher.UUID)
        self.assertEqual(env["CP2K_DATA_DIR"], str(self.root / "source/data"))
        self.assertEqual(env["HOME"], str(self.prefix / "container-home"))
        self.assertEqual(env["XDG_CACHE_HOME"], str(self.prefix / "cache"))
        self.assertTrue(env["LD_LIBRARY_PATH"].startswith(str(self.prefix / "install/cp2k/lib") + ":"))
        defaults = launcher.forwarded_environment({})
        self.assertEqual(defaults["CP2K_GPU_MEM_LIMIT_MB"], "")
        self.assertEqual(defaults["CP2K_GPU_MEM_LIMIT_MB_RANK0"], "")
        for key in ("HIP_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION", "API_TOKEN"):
            self.assertNotIn(key, env)

    def test_ldd_rejects_transitive_missing_wrong_and_symlinked_library(self):
        self.installed()
        good = f"  libcp2k.so.2026.2 => {self.library} (0x1234)\n"
        self.assertEqual(launcher.check_ldd(good, self.prefix), self.library)
        for text in (good + "libother.so => not found\n", "", good + good,
                     good.replace(str(self.prefix), "/home/nowr/cp2k-mi50-production")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                launcher.check_ldd(text, self.prefix)
        self.library.unlink()
        self.library.symlink_to(self.deps / "sentinel")
        with self.assertRaises(ValueError):
            launcher.check_ldd(good, self.prefix)

    def test_compile_failure_cannot_record_success(self):
        launcher.claim_prefix(self.prefix)
        with mock.patch.object(launcher, "logged", side_effect=subprocess.CalledProcessError(9, ["cmake"])) as logged:
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.inside_build(argparse.Namespace(prefix=str(self.prefix), jobs=2))
        self.assertEqual(logged.call_count, 1)
        command = logged.call_args.args[0]
        self.assertIn(str(self.root / "source"), command)
        self.assertIn("-DCP2K_WITH_GPU=Mi50", command)
        self.assertFalse((self.prefix / launcher.SUCCESS).exists())

    def test_reject_unsafe_working_directory(self):
        for path in (self.root, self.prefix, self.deps, self.base, Path.home(), Path("/media/job")):
            with self.subTest(path=path), mock.patch.object(launcher.Path, "cwd", return_value=path), self.assertRaises(ValueError):
                launcher.check_cwd(self.prefix)

    def test_sysfs_uuid_pci_and_device_number_are_checked(self):
        sysfs = self.base / "sys/class/drm"
        devroot = self.base / "dev"
        (devroot / "dri").mkdir(parents=True)
        (devroot / "kfd").touch()
        pci = self.base / "sys/bus/pci/devices/0000:03:00.0"
        pci.mkdir(parents=True)
        for key, value in {"unique_id": launcher.UUID[4:], "vendor": "0x1002", "device": "0x66a1"}.items():
            (pci / key).write_text(value)
        node = sysfs / "renderD130"
        node.mkdir(parents=True)
        (node / "device").symlink_to(pci, target_is_directory=True)
        (node / "dev").write_text("0:0")  # disposable regular files have st_rdev == 0
        (devroot / "dri/renderD130").touch()
        with mock.patch.object(launcher.stat, "S_ISCHR", return_value=True):
            self.assertEqual(launcher.select_gpu(sysfs, devroot), (devroot / "dri/renderD130", "0000:03:00.0"))
            (pci / "unique_id").write_text("unrelated-gpu")
            with self.assertRaises(ValueError):
                launcher.select_gpu(sysfs, devroot)
            (pci / "unique_id").write_text(launcher.UUID[4:])
            (pci / "device").write_text("0x7550")
            with self.assertRaises(ValueError):
                launcher.select_gpu(sysfs, devroot)
            (pci / "device").write_text("0x66a1")
            (node / "dev").write_text("226:131")
            with self.assertRaises(ValueError):
                launcher.select_gpu(sysfs, devroot)
        (node / "dev").write_text("0:0")
        with self.assertRaises(ValueError):
            launcher.select_gpu(sysfs, devroot)

    def test_missing_mock_compiler_is_a_failure(self):
        with mock.patch.object(verify.shutil, "which", return_value=None), self.assertRaises(ValueError):
            verify.run_mocks(self.root, "elpa")

    def test_specific_media_case_allowed_but_build_and_broad_media_rejected(self):
        case = Path("/media/fixture/volume/calculations/case with spaces")
        original_stat = Path.stat
        def fixture_stat(path, *args, **kwargs):
            return original_stat(self.work if path == case else path, *args, **kwargs)
        with mock.patch.object(launcher.Path, "cwd", return_value=case), mock.patch.object(Path, "stat", fixture_stat):
            self.assertEqual(launcher.check_cwd(self.prefix), case)
        with self.assertRaises(ValueError):
            launcher.check_prefix(str(case), fresh=True)
        for directory in (Path("/media"), Path("/media/fixture"), Path("/media/fixture/volume"), case.parent):
            with mock.patch.object(launcher.Path, "cwd", return_value=directory), self.assertRaises(ValueError):
                launcher.check_cwd(self.prefix)

    def test_generated_prefix_launcher_preserves_args_cwd_and_exit(self):
        self.prefix = self.base / "prefix with 'quotes' and $literal"
        self.installed()
        (self.root / "scripts").mkdir()
        args_file = self.base / "argv.bin"
        cwd_file = self.base / "cwd.txt"
        (self.root / "scripts/run.sh").write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\nprintf "%s\\0" "$@" > ' + shlex.quote(str(args_file)) +
            '\npwd -P > ' + shlex.quote(str(cwd_file)) + '\nexit 23\n')
        launcher.write_launcher(self.prefix)
        arguments = ["-i", "a b.inp", "-o", "literal $(touch forbidden).out", "", "single'quote"]
        result = subprocess.run([str(self.prefix / "bin/cp2k.psmp"), *arguments], cwd=self.work)
        self.assertEqual(result.returncode, 23)
        self.assertEqual(args_file.read_bytes().decode().split("\0")[:-1], ["--prefix", str(self.prefix), "--", *arguments])
        self.assertEqual(cwd_file.read_text().strip(), str(self.work))
        self.assertFalse((self.work / "forbidden").exists())
        self.assertEqual(launcher.check_prefix(str(self.prefix)), self.prefix)

    def test_runtime_cache_symlink_is_rejected(self):
        self.installed()
        cache = self.prefix / "cache"
        cache.rmdir()
        cache.symlink_to(self.deps, target_is_directory=True)
        with self.assertRaises(ValueError):
            launcher.check_prefix(str(self.prefix))
        with self.assertRaises(ValueError):
            launcher.docker_base(self.prefix)

    def test_legacy_absolute_data_mount_is_readonly_and_required(self):
        self.installed()
        path = "/home/nowr/cp2k-hip/data"
        self.assertEqual(launcher.config()["runtime_mounts"], [{"source": path, "target": path}])
        command = launcher.docker_base(self.prefix)
        self.assertIn(f"type=bind,source={path},target={path},readonly", command)
        self.assertFalse(any(arg.startswith("type=bind,source=/home/nowr/cp2k-hip,") for arg in command))
        cfg = launcher.config()
        cfg["runtime_mounts"][0]["source"] = str(self.base / "missing-data")
        with mock.patch.object(launcher, "config", return_value=cfg), \
             mock.patch.object(launcher.subprocess, "run") as docker, self.assertRaises(ValueError):
            launcher.verify_dependencies(runtime=True)
        docker.assert_not_called()

    def test_shell_entrypoints_help_and_invalid_prefix(self):
        for name in ("build.sh", "run.sh", "verify.sh"):
            proc = subprocess.run(["bash", str(bundle.ROOT / "scripts" / name), "--help"], capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
        proc = subprocess.run(["bash", str(bundle.ROOT / "scripts/build.sh"), "--prefix", "relative"],
                              text=True, capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("absolute", proc.stderr)

    def test_inside_run_np1_uses_clean_env_and_rejects_changed_install(self):
        self.installed()
        with mock.patch.object(launcher, "ldd_check", return_value=(self.binary, self.library)), \
             mock.patch.object(launcher.Path, "mkdir"), \
             mock.patch.object(launcher.os, "execvpe") as execute:
            launcher.inside_run(argparse.Namespace(prefix=str(self.prefix), check=False,
                                                   cp2k_args=["-i", "quoted input.inp"]))
            _, argv, env = execute.call_args.args
            self.assertEqual(argv, ["/usr/bin/mpirun", "-np", "1", "--nooversubscribe", "--bind-to", "none",
                                    str(self.binary), "-i", "quoted input.inp"])
            self.assertNotIn("HIP_VISIBLE_DEVICES", env)
            self.library.write_text("changed after build")
            with self.assertRaises(ValueError):
                launcher.inside_run(argparse.Namespace(prefix=str(self.prefix), check=True, cp2k_args=[]))

    def test_check_only_invokes_ldd_never_cp2k_or_mpirun(self):
        self.installed()
        output = f"libcp2k.so.2026.2 => {self.library} (0x1234)\n"
        result = subprocess.CompletedProcess(["ldd", str(self.binary)], 0, stdout=output, stderr="")
        with mock.patch.object(launcher.subprocess, "run", return_value=result) as process, \
             mock.patch.object(launcher.os, "execvpe", side_effect=AssertionError("CP2K/MPI must not start")), \
             mock.patch.object(launcher.Path, "mkdir", side_effect=AssertionError("check must not create runtime files")), \
             mock.patch("sys.stdout", new=io.StringIO()) as stdout:
            status = launcher.inside_run(argparse.Namespace(prefix=str(self.prefix), check=True, cp2k_args=[]))
        self.assertEqual(status, 0)
        self.assertIn("CHECK_OK", stdout.getvalue())
        self.assertEqual(process.call_count, 1)
        self.assertEqual(process.call_args.args[0], ["ldd", str(self.binary)])


class ArchiveTests(unittest.TestCase):
    def test_tree_and_archive_detect_corruption_extras_missing_files_and_modes(self):
        with tempfile.TemporaryDirectory(prefix="cp2k manifest fixtures ") as temporary:
            root = Path(temporary) / "extracted"
            (root / "source").mkdir(parents=True)
            payload = root / "source/data.bin"
            payload.write_bytes(b"original")
            payload.chmod(0o644)
            (root / "source").chmod(0o755)
            entries = {"source": bundle.record("source", 0o755),
                       "source/data.bin": bundle.record("source/data.bin", 0o644, b"original")}
            manifest = bundle.json_bytes({"format": 1, "package": bundle.NAME,
                "source_archive_sha256": bundle.SOURCE_SHA256, "source_overrides": bundle.SOURCE_OVERRIDES,
                "source_regular_files": 1,
                "entries": list(entries.values())})
            (root / "manifest.json").write_bytes(manifest)
            sums = f"{entries['source/data.bin']['sha256']}  source/data.bin\n{bundle.digest(manifest)}  manifest.json\n"
            (root / "SHA256SUMS").write_text(sums)
            for p in (root / "manifest.json", root / "SHA256SUMS"):
                p.chmod(0o644)
            with mock.patch.object(verify, "REQUIRED", ("source/data.bin",)):
                self.assertEqual(verify.verify_tree(root), entries)
                for data in (b"modified", b""):
                    payload.write_bytes(data)
                    with self.assertRaises(ValueError):
                        verify.verify_tree(root)
                payload.write_bytes(b"original")
                payload.chmod(0o600)
                with self.assertRaises(ValueError):
                    verify.verify_tree(root)
                payload.chmod(0o644)
                extra = root / "extra-user-file"
                extra.write_text("private")
                with self.assertRaises(ValueError):
                    verify.verify_tree(root)
                extra.unlink()
                archive = Path(temporary) / "fixture.tar.gz"
                for corrupt in (False, True):
                    with tarfile.open(archive, "w:gz") as tar:
                        for relative in ("", "source", "source/data.bin", "manifest.json", "SHA256SUMS"):
                            path = root / relative
                            item = tarfile.TarInfo(bundle.NAME + ("/" + relative if relative else ""))
                            item.mode = 0o755 if path.is_dir() else 0o644
                            if path.is_dir():
                                item.type = tarfile.DIRTYPE
                                tar.addfile(item)
                            else:
                                data = b"corrupted" if corrupt and relative == "source/data.bin" else path.read_bytes()
                                item.size = len(data)
                                tar.addfile(item, io.BytesIO(data))
                    if corrupt:
                        with self.assertRaises(ValueError):
                            verify.verify_archive(archive, root, entries)
                    else:
                        verify.verify_archive(archive, root, entries)
                payload.unlink()
                with self.assertRaises(ValueError):
                    verify.verify_tree(root)
                payload.symlink_to(root / "manifest.json")
                with self.assertRaises(ValueError):
                    verify.verify_tree(root)

    def test_tar_rejects_links_duplicate_traversal_and_unreviewed_top(self):
        cases = [(["../escape"], tarfile.REGTYPE), (["src/a", "src/a"], tarfile.REGTYPE),
                 (["src/link"], tarfile.SYMTYPE), (["private-input.inp"], tarfile.REGTYPE)]
        for names, kind in cases:
            with self.subTest(names=names):
                stream = io.BytesIO()
                with tarfile.open(fileobj=stream, mode="w") as tar:
                    for name in names:
                        item = tarfile.TarInfo(name)
                        item.type = kind
                        item.linkname = "/outside" if kind == tarfile.SYMTYPE else ""
                        tar.addfile(item)
                stream.seek(0)
                with tarfile.open(fileobj=stream, mode="r") as tar, self.assertRaises(ValueError):
                    bundle.source_members(tar)

    def test_exclusion_does_not_remove_public_coordinates_or_data(self):
        for name in ("tests/case/__pycache__/x.pyc", "tests/.build-123/x", "SPEC.md"):
            self.assertTrue(bundle.excluded(name))
        for name in ("tests/QS/water.xyz", "benchmarks/QS/input.inp", "data/BASIS_MOLOPT"):
            self.assertFalse(bundle.excluded(name))

    def test_checksums_reject_missing_duplicate_or_changed_payload(self):
        entry = bundle.record("source/file", 0o644, b"unchanged")
        manifest = b"manifest bytes"
        correct = f"{entry['sha256']}  source/file\n{bundle.digest(manifest)}  manifest.json\n".encode()
        verify.check_sums(correct, {"source/file": entry}, manifest)
        for sums in (b"", correct + correct, correct.replace(b"source/file", b"source/wrong")):
            with self.assertRaises(ValueError):
                verify.check_sums(sums, {"source/file": entry}, manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
