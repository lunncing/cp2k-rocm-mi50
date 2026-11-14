#!/usr/bin/env python3
"""Isolated Stage 8 MI50 builder/runner, derived from the supplied 8030-byte runner."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True

MI50_DEPS_ROOT = Path("/home/nowr/cp2k-hip/build-mi50")
RUNTIME_CFG = Path("/home/nowr/cp2k-hip/runtime/dependencies-mi50.json")
STAGE8_ROOT = Path("/media/nowr/Data/cp2k-stage8/build-mi50-stage8")
IMAGE = "sha256:eb883e666c1d9787ce3696c50c2256c45f6c4a5107dc6af66c2a781cda7f4fd4"
MI50_UUID = "GPU-37ac3981732c7340"
PASSTHROUGH_ENV = (
    "CP2K_GPU_HOST_FALLBACK", "CP2K_GPU_MEM_LIMIT_MB", "CP2K_GPU_MEM_LIMIT_MB_RANK0",
    "CP2K_PW_GPU_MAP_CACHE", "CP2K_PW_GPU_FUSED_XC", "CP2K_PW_GPU_RES",
    "CP2K_PW_GPU_RES_DENSITY", "CP2K_PW_GPU_RES_HARTREE", "CP2K_PW_GPU_RES_XCIN",
    "CP2K_PW_GPU_RES_POT", "CP2K_PW_GPU_RES_VERIFY", "CP2K_PW_GPU_RES_STATS",
    "CP2K_PW_GPU_RES_LAZY_DENSITY", "CP2K_PW_GPU_RES_LAZY_POT", "CP2K_PW_GPU_RES_NATURAL_G",
    "CP2K_PW_GPU_RES_NATURAL_XC", "CP2K_PW_GPU_RES_XC_PAIRS", "CP2K_PW_GPU_RES_XC_DZ",
    "CP2K_PW_GPU_RES_RESERVE_MB", "CP2K_PW_GPU_MEM_SAMPLE",
    "CP2K_PW_GPU_UKS", "CP2K_PW_GPU_XC_FULL_PIPELINE",
    "CP2K_ELPA_GPU", "CP2K_ELPA_GPU_N_MIN", "CP2K_CHOLESKY_FAST_SINGLE_RANK",
)


def beneath(path, parent):
    return path == parent or parent in path.parents


def overlap(left, right):
    return beneath(left, right) or beneath(right, left)


def exact_path(path):
    supplied = Path(path)
    if not supplied.is_absolute() or supplied.resolve() != supplied:
        raise ValueError(f"Absolute, resolved, non-symlink path required: {path}")
    return supplied


def isolated(path):
    resolved = exact_path(path)
    if not beneath(resolved, STAGE8_ROOT) or resolved == STAGE8_ROOT:
        raise ValueError(f"Outside development root: {path}")
    return resolved


def cfg():
    config = json.loads(RUNTIME_CFG.read_text(encoding="utf-8"))
    for item in config["mounts"] + config["runtime_mounts"]:
        data_mount = (item["source"] == "/home/nowr/cp2k-hip/data"
                      and item["target"] == "/home/nowr/cp2k-hip/data")
        for key in ("source", "target"):
            path = exact_path(item[key])
            if "," in str(path) or "\n" in str(path):
                raise ValueError("Unsafe Docker bind path")
            if overlap(path, STAGE8_ROOT.parent) or overlap(path, Path("/tmp")):
                raise ValueError(f"Dependency mount overlaps development/runtime workspace: {path}")
            if (overlap(path, Path("/home/nowr/cp2k-hip"))
                    and not beneath(path, MI50_DEPS_ROOT / "deps") and not data_mount):
                raise ValueError(f"Production install/source mount forbidden: {path}")
            if path == Path("/") or path == Path("/dev"):
                raise ValueError(f"Overbroad dependency mount: {path}")
    return config


def map_path(template):
    mapped = template.replace("{prefix}/install/cp2k", str(STAGE8_ROOT / "install/cp2k"))
    mapped = mapped.replace("{prefix}", str(MI50_DEPS_ROOT))
    if re.search(r"/home/nowr/cp2k-hip/build-(?:mi50|9070xt)/(?!deps(?:/|$))", mapped):
        raise ValueError(f"Production build/install path in recipe: {mapped}")
    return mapped


def recipe():
    config = cfg()
    libs = [map_path(path) for path in config["library_paths"]]
    devlibs = [str(STAGE8_ROOT / "install/cp2k/lib"), str(STAGE8_ROOT / "install/cp2k/lib64")]
    libs = list(dict.fromkeys(devlibs + libs))
    pkg = [map_path(path) for path in config["pkg_config_paths"]]
    prefixes = [map_path(path) for path in config["prefix_paths"]]
    flags = []
    for arg in config["configure_args"]:
        arg = arg.replace("{cmake_prefix}", ";".join(prefixes))
        arg = arg.replace("{cmake_library}", ";".join(libs))
        flags.append(map_path(arg))
    flags += [
        f"-DCMAKE_INSTALL_PREFIX={STAGE8_ROOT / 'install/cp2k'}",
        "-DCP2K_WITH_GPU_PROFILING=OFF",
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        "-DFETCHCONTENT_UPDATES_DISCONNECTED=ON",
    ]
    return libs, pkg, flags


def positive(value):
    if not re.fullmatch(r"[1-9][0-9]*", str(value)) or int(value) > 1024:
        raise argparse.ArgumentTypeError("Expected integer in [1, 1024]")
    return int(value)


def sha_argument(value):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise argparse.ArgumentTypeError("Expected a 64-digit SHA-256")
    return value.lower()


def select_mi50():
    matches = []
    for node in sorted(Path("/sys/class/drm").glob("renderD*")):
        device = (node / "device").resolve()
        try:
            unique = (device / "unique_id").read_text().strip().lower().removeprefix("gpu-").removeprefix("0x")
            vendor = (device / "vendor").read_text().strip().lower()
            pci_id = (device / "device").read_text().strip().lower()
        except OSError:
            continue
        if unique != MI50_UUID[4:]:
            continue
        if (vendor, pci_id) != ("0x1002", "0x66a1"):
            raise RuntimeError(f"Unexpected MI50 vendor/device: {vendor} {pci_id}")
        render = Path("/dev/dri") / node.name
        if not stat.S_ISCHR(render.stat().st_mode):
            raise RuntimeError(f"Not a render character device: {render}")
        matches.append(render)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one MI50 render node, got {matches}")
    return matches[0]


def make_env(build, host=None):
    host = os.environ if host is None else host
    rocm = cfg()["rocm"]
    libs, pkg, _ = recipe()
    env = {
        "PATH": f"{rocm}/bin:{rocm}/llvm/bin:/usr/bin:/bin",
        "LC_ALL": "C", "ROCM_PATH": rocm, "HIP_PATH": rocm, "HIP_PLATFORM": "amd",
        "LD_LIBRARY_PATH": ":".join(libs),
        "CP2K_DATA_DIR": str(STAGE8_ROOT / "source/data"),
        "HOME": str(STAGE8_ROOT / "container-home"),
        "XDG_CACHE_HOME": str(STAGE8_ROOT / "cache"),
    }
    if build:
        env.update(
            PKG_CONFIG_PATH=":".join(pkg), LIBRARY_PATH=":".join(libs),
            HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx906", HCC_AMDGPU_TARGET="gfx906",
            AMDGPU_TARGETS="gfx906", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            ROCR_VISIBLE_DEVICES="",
        )
    else:
        env.update(
            OMP_NUM_THREADS=str(positive(host.get("OMP_NUM_THREADS", "16"))),
            OPENBLAS_NUM_THREADS="1", OMP_DYNAMIC="FALSE", OMP_PROC_BIND="FALSE",
            OMP_STACKSIZE="256M", ROCR_VISIBLE_DEVICES=MI50_UUID,
        )
        for key in PASSTHROUGH_ENV:
            if key in host:
                value = host[key]
                if not value or any(char in value for char in "\n\r\0"):
                    raise ValueError(f"Invalid environment value: {key}")
                if key == "CP2K_CHOLESKY_FAST_SINGLE_RANK" and value not in ("0", "1"):
                    raise ValueError(f"{key} must be exactly 0 or 1")
                env[key] = value
        env.setdefault("CP2K_PW_GPU_RES_RESERVE_MB", "1024")
        env.setdefault("CP2K_PW_GPU_MEM_SAMPLE", "1")
    return env


def validate_case(path):
    case = exact_path(path)
    if case == STAGE8_ROOT.parent or not beneath(case, STAGE8_ROOT.parent):
        raise ValueError("Cases must be inside /media/nowr/Data/cp2k-stage8")
    protected = [STAGE8_ROOT / name for name in
                 ("source", "build", "install", "cache", "container-home", "logs")]
    protected += [Path(item[key]) for item in cfg()["mounts"] + cfg()["runtime_mounts"]
                  for key in ("source", "target")]
    if any(overlap(case, item) for item in protected):
        raise ValueError(f"Case overlaps protected source/build/dependency directory: {case}")
    if not case.is_dir() or "," in str(case) or "\n" in str(case):
        raise ValueError(f"Existing, safe case directory required: {case}")
    for root, directories, files in os.walk(case, followlinks=False):
        for name in directories + files:
            item = Path(root) / name
            info = item.lstat()
            if item.is_symlink() or (item.is_file() and info.st_nlink != 1):
                raise ValueError(f"Case symlink or hard-linked file forbidden: {item}")
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise ValueError(f"Case contains special file: {item}")
    return case


def binary_path(path):
    supplied = Path(path)
    if not supplied.is_absolute():
        raise ValueError("Executable path must be absolute")
    binary = isolated(supplied.resolve(strict=True))
    if not any(beneath(binary, STAGE8_ROOT / name) for name in ("install/cp2k", "build/cp2k")):
        raise ValueError(f"Not a development executable: {binary}")
    return binary


def docker_cmd(command, env, mode="run", case=None):
    config = cfg()
    argv = [
        "/usr/bin/docker", "--host", "unix:///var/run/docker.sock", "run", "--rm", "--pull=never",
        "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--user", f"{os.getuid()}:{os.getgid()}", "--shm-size=1g", "--pids-limit=2048",
        "--ulimit", "core=0",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=4294967296,mode=1777",
        "--tmpfs", f"/media:rw,nosuid,nodev,mode=0755,uid={os.getuid()},gid={os.getgid()}",
        "--tmpfs", f"/home/nowr/cp2k-hip:rw,nosuid,nodev,mode=0755,uid={os.getuid()},gid={os.getgid()}",
        "--entrypoint", "/usr/bin/env",
    ]

    def mount(source, readonly=True, target=None):
        if any(char in str(source) + str(target or source) for char in ",\n\r"):
            raise ValueError("Unsafe bind path")
        argv.extend(["--mount", f"type=bind,source={source},target={target or source}"
                     + (",readonly" if readonly else "")])

    mount(MI50_DEPS_ROOT / "deps")
    for item in config["mounts"] + config["runtime_mounts"]:
        mount(item["source"], target=item["target"])
    mount(isolated(STAGE8_ROOT / "source"))
    mount(isolated(STAGE8_ROOT / "install"), readonly=(mode != "build"))
    mount(isolated(STAGE8_ROOT / "build"), readonly=(mode != "build"))
    mount(isolated(STAGE8_ROOT / "cache"), readonly=(mode == "check"))
    mount(isolated(STAGE8_ROOT / "container-home"), readonly=(mode == "check"))
    mount(Path(__file__).resolve(), target="/tmp/stage8_runner.py")
    if mode == "build":
        mount(isolated(STAGE8_ROOT / "logs"), readonly=False)
    if mode == "run":
        if case is None:
            raise ValueError("Case is required for a GPU run")
        validate_case(case)
        render = select_mi50()
        argv += ["--device", "/dev/kfd", "--device", str(render)]
        for group in sorted({Path("/dev/kfd").stat().st_gid, render.stat().st_gid}):
            argv += ["--group-add", str(group)]
        mount(case, readonly=False)
    argv += ["--workdir", str(case) if case else "/tmp", IMAGE, "-i"]
    argv += [f"{key}={value}" for key, value in env.items()]
    return argv + command


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path):
    resolved = Path(path).resolve(strict=True)
    return {"path": str(path), "resolved_path": str(resolved), "bytes": resolved.stat().st_size,
            "sha256": digest(resolved)}


def elf_record(path):
    resolved = Path(path).resolve(strict=True)
    isolated(resolved)
    with resolved.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError(f"Not an ELF object: {resolved}")
    return file_record(path)


def audit(binary, expected_elf=None, expected_lib=None):
    executable = elf_record(binary_path(binary))
    started = time.monotonic()
    command = ["/usr/bin/ldd", executable["resolved_path"]]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    audit_data = {"type": "elf_audit", "elf": executable, "command": command,
                  "ldd_stdout": result.stdout, "ldd_stderr": result.stderr,
                  "ldd_exit": result.returncode, "wall_seconds": time.monotonic() - started,
                  "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
                  "gpu_invoked": False}
    print(json.dumps(audit_data), flush=True)
    if result.returncode or "not found" in result.stdout or "not found" in result.stderr:
        raise RuntimeError("ldd failed or has unresolved dependencies")
    libraries = set()
    for line in result.stdout.splitlines():
        if "libcp2k" not in line:
            continue
        match = re.search(r"(?:=>\s*)?(/[^\s]+)", line)
        if not match:
            raise RuntimeError(f"Unresolved libcp2k line: {line}")
        resolved = Path(match.group(1)).resolve(strict=True)
        if not any(beneath(resolved, STAGE8_ROOT / name) for name in ("install/cp2k", "build/cp2k")):
            raise RuntimeError(f"Non-development libcp2k forbidden: {resolved}")
        libraries.add(resolved)
    if len(libraries) != 1:
        raise RuntimeError(f"Expected exactly one development libcp2k, found {sorted(map(str, libraries))}")
    library = elf_record(libraries.pop())
    for expected, actual, label in ((expected_elf, executable, "ELF"), (expected_lib, library, "libcp2k")):
        if expected and actual["sha256"] != expected:
            raise RuntimeError(f"{label} SHA-256 mismatch")
    audit_data["libcp2k"] = library
    audit_data["status"] = "PASS"
    print(json.dumps(audit_data), flush=True)
    return audit_data


def recorded(command, directory, label):
    log = directory / f"{label}.log"
    event = {"command": command, "start_utc": datetime.now(timezone.utc).isoformat(), "log": str(log)}
    started = time.monotonic()
    exit_code = None
    try:
        with log.open("xb") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            exit_code = process.wait()
        return exit_code
    finally:
        event.update(exit=exit_code, wall_seconds=time.monotonic() - started,
                     end_utc=datetime.now(timezone.utc).isoformat())
        with (directory / "commands.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)


def run_inputs(case, arguments, extras):
    names = []
    for index, argument in enumerate(arguments):
        if argument in ("-i", "--input"):
            if index + 1 >= len(arguments):
                raise ValueError("Missing CP2K input filename")
            names.append(arguments[index + 1])
        elif argument.startswith("--input="):
            names.append(argument.split("=", 1)[1])
    if len(names) != 1:
        raise ValueError("Run requires exactly one explicit CP2K -i/--input path")
    records = []
    for role, name in [("input", names[0])] + [("restart_or_include", item) for item in extras]:
        path = (case / name).resolve(strict=True)
        if not beneath(path, case) or not path.is_file():
            raise ValueError(f"Input/restart must be a regular file in the case directory: {path}")
        records.append({"role": role, **file_record(path)})
    for path in sorted(case.rglob("*")):
        if path.is_file() and any(token in path.name.lower() for token in ("restart", ".wfn", ".kp")):
            if not any(item["resolved_path"] == str(path) for item in records):
                records.append({"role": "restart_candidate", **file_record(path)})
    return records


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    actions = result.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build", help="Build/install only in the isolated development root")
    build.add_argument("--jobs", type=positive, default=16)
    build.add_argument("--reconfigure", action="store_true")
    for name in ("check", "run"):
        action = actions.add_parser(name, help="Audit resolved ELF/libcp2k without GPU" if name == "check"
                                    else "Audit and run unchanged input on the MI50")
        action.add_argument("--binary", default=str(STAGE8_ROOT / "install/cp2k/bin/cp2k.psmp"))
        action.add_argument("--expect-elf-sha256", type=sha_argument)
        action.add_argument("--expect-libcp2k-sha256", type=sha_argument)
        if name == "run":
            action.add_argument("--np", type=positive, default=1)
            action.add_argument("--case")
            action.add_argument("--record-file", action="append", default=[],
                                help="Additional exact input/include/restart file, repeatable")
            action.add_argument("args", nargs=argparse.REMAINDER)
    actions.add_parser("self-test", help="Pure local tests; no Docker, GPU, or build")
    return result


def inner_main(arguments):
    binary, expected_elf, expected_lib, *command = arguments
    audit(binary, expected_elf or None, expected_lib or None)
    if not command:
        return 0
    print(json.dumps({"type": "guarded_command", "command": command}), flush=True)
    return subprocess.call(command)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_guard":
        return inner_main(sys.argv[2:])
    args = parser().parse_args()
    if args.action == "self-test":
        from test_tools import run_runner_tests
        return run_runner_tests()
    if sys.platform != "linux":
        raise RuntimeError("Build/check/run require Linux; self-test is portable")
    config = cfg()
    for name in ("build", "install", "logs", "cache", "container-home"):
        isolated(STAGE8_ROOT / name).mkdir(parents=True, exist_ok=True)
    directory = isolated(STAGE8_ROOT / "logs") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12])
    directory.mkdir()
    record = {"schema": 1, "action": args.action, "runner": file_record(__file__),
              "runtime_config": file_record(RUNTIME_CFG), "image": IMAGE,
              "argv": sys.argv, "recorded_files": [],
              "restart_caveat": "List includes exact named files and discovered restart candidates; "
              "use --record-file for includes/external references; not proof of which restart CP2K consumed.",
              "vram_caveat": "Sampled VRAM is a lower bound on peak use, not a continuous peak measurement."}
    exit_code = 1
    try:
        if args.action == "build":
            builddir = isolated(STAGE8_ROOT / "build/cp2k")
            builddir.mkdir(parents=True, exist_ok=True)
            isolated(STAGE8_ROOT / "install/cp2k").mkdir(parents=True, exist_ok=True)
            env = make_env(True)
            commands = []
            if args.reconfigure or not (builddir / "build.ninja").is_file():
                commands.append(["cmake", "-S", str(STAGE8_ROOT / "source"), "-B", str(builddir),
                                 "-G", "Ninja", *recipe()[2]])
            commands += [["cmake", "--build", str(builddir), "-j", str(args.jobs)],
                         ["cmake", "--install", str(builddir)]]
            for index, command in enumerate(commands):
                exit_code = recorded(docker_cmd(command, env, mode="build"), directory, f"build-{index}")
                if exit_code:
                    return exit_code
            print("STAGE8_BUILD_OK", flush=True)
            return 0
        binary = binary_path(args.binary)
        env = make_env(False)
        case = None
        command = ["/usr/bin/python3", "/tmp/stage8_runner.py", "_guard", str(binary),
                   args.expect_elf_sha256 or "", args.expect_libcp2k_sha256 or ""]
        if args.action == "run":
            case = validate_case(Path(args.case).absolute() if args.case else Path.cwd())
            cp2k_args = args.args[1:] if args.args and args.args[0] == "--" else args.args
            record["recorded_files"] = run_inputs(case, cp2k_args, args.record_file)
            command += ["/usr/bin/mpirun", "-np", str(args.np), "--nooversubscribe",
                        "--bind-to", "none", str(binary), *cp2k_args]
        else:
            env["ROCR_VISIBLE_DEVICES"] = ""
            env["HIP_VISIBLE_DEVICES"] = ""
        record["environment"] = env
        exit_code = recorded(docker_cmd(command, env, mode=args.action, case=case), directory, args.action)
        return exit_code
    except BaseException as error:
        record["error"] = str(error)
        exit_code = 1
        raise
    finally:
        record["exit"] = exit_code
        record["end_utc"] = datetime.now(timezone.utc).isoformat()
        (directory / "record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"STAGE8_RECORD={directory / 'record.json'}", flush=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"stage8_runner: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
