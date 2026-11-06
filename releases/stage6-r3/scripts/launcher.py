#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Target-specific Docker launchers. No fetching, cleanup, or GPU clock changes."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys

sys.dont_write_bytecode = True

from bundle import ROOT, NAME, file_hash, json_bytes

UUID = "GPU-37ac3981732c7340"
MARKER = ".cp2k-stage6-package.json"
SUCCESS = ".cp2k-stage6-build-ok.json"
PROTECTED = (
    "/home/nowr/cp2k-mi50-production", "/home/nowr/cp2k-mi50-20261103",
    "/home/nowr/spack/opt/spack", "/home/nowr/MI50_CP2K_full_build_20261103",
    "/home/nowr/cp2k-hip/data", "/opt/rocm", "/media",
    "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/proc", "/sys", "/usr",
)
THREAD_DEFAULTS = {"OMP_NUM_THREADS": "16", "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "FALSE",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_STACKSIZE": "256M"}
DIAGNOSTICS = (
    "CP2K_GPU_HOST_FALLBACK", "CP2K_GPU_MEM_LIMIT_MB", "CP2K_GPU_MEM_LIMIT_MB_RANK0",
    "CP2K_PW_GPU_MAP_CACHE", "CP2K_PW_GPU_FUSED_XC", "CP2K_PW_GPU_RES",
    "CP2K_PW_GPU_RES_DENSITY", "CP2K_PW_GPU_RES_HARTREE", "CP2K_PW_GPU_RES_XCIN",
    "CP2K_PW_GPU_RES_POT", "CP2K_PW_GPU_RES_VERIFY", "CP2K_PW_GPU_RES_STATS",
    "CP2K_PW_GPU_RES_LAZY_DENSITY", "CP2K_PW_GPU_RES_LAZY_POT", "CP2K_PW_GPU_RES_NATURAL_G",
    "CP2K_PW_GPU_RES_NATURAL_XC", "CP2K_PW_GPU_RES_XC_PAIRS", "CP2K_PW_GPU_RES_XC_DZ",
    "CP2K_PW_GPU_RES_RESERVE_MB",
)


def config():
    return json.loads((ROOT / "recipe/dependencies.json").read_text())


def overlaps(a, b):
    return a == b or a in b.parents or b in a.parents


def checked_path(raw):
    if not raw or not Path(raw).is_absolute() or ".." in raw.split("/"):
        raise ValueError("path must be absolute and contain no '..'")
    if any(c in raw for c in ",:;\\") or any(ord(c) < 32 for c in raw):
        raise ValueError("path contains a Docker/CMake path separator or control character")
    path = Path(raw)
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"symlinked path is not allowed: {component}")
    return path.resolve()


def protected_paths(runtime=False):
    return [ROOT.resolve(), *(Path(p) for p in PROTECTED if not (runtime and p == "/media")),
            *(Path(m["source"]).resolve() for m in config()["mounts"] + config()["runtime_mounts"])]


def check_prefix(raw, fresh=False):
    prefix = checked_path(raw)
    if prefix in (Path.home().resolve(), Path("/home"), Path("/tmp"), Path("/var"), Path("/opt"), Path("/srv")):
        raise ValueError("prefix must be a dedicated subdirectory")
    for protected in protected_paths():
        if overlaps(prefix, protected):
            raise ValueError(f"prefix overlaps protected path: {protected}")
    if not prefix.parent.is_dir():
        raise ValueError("prefix parent must already exist")
    if prefix.exists():
        if not prefix.is_dir() or prefix.stat().st_uid != os.getuid():
            raise ValueError("prefix must be a directory owned by the invoking UID")
        if fresh and any(prefix.iterdir()):
            raise ValueError("build requires a NEW or EMPTY prefix; choose a new path for every rebuild")
    elif not fresh:
        raise ValueError("prefix does not exist; build first")
    if not fresh:
        marker = prefix / MARKER
        if marker.is_symlink() or not marker.is_file():
            raise ValueError("prefix is not owned by this package (missing marker)")
        data = json.loads(marker.read_text())
        expected = {"package": NAME, "prefix": str(prefix), "uid": os.getuid(),
                    "manifest_sha256": file_hash(ROOT / "manifest.json")}
        if data != expected:
            raise ValueError("prefix ownership/package manifest mismatch")
        for child in prefix.iterdir():
            if child.is_symlink() or child.name not in (
                MARKER, SUCCESS, "build", "install", "logs", "cache", "container-home", "bin"
            ):
                raise ValueError(f"unexpected or symlinked prefix entry: {child.name}")
    return prefix


def claim_prefix(prefix):
    # No deletion, reuse of CMakeCache, or merge into a previous installation.
    if not prefix.exists():
        prefix.mkdir(mode=0o700)
    if any(prefix.iterdir()):
        raise ValueError("prefix became nonempty before claiming it")
    with (prefix / MARKER).open("x") as stream:
        stream.write(json_bytes({"package": NAME, "prefix": str(prefix), "uid": os.getuid(),
                                 "manifest_sha256": file_hash(ROOT / "manifest.json")}).decode())
    for subdir in ("build", "install", "logs", "cache", "container-home", "bin"):
        (prefix / subdir).mkdir()


def verify_dependencies(runtime=False):
    mounts = config()["mounts"] + (config()["runtime_mounts"] if runtime else [])
    for mount in mounts:
        source = checked_path(mount["source"])
        if not source.is_dir():
            raise ValueError(f"required dependency directory missing: {source}")
    subprocess.run(["docker", "image", "inspect", config()["image"]], check=True,
                   stdout=subprocess.DEVNULL)


def add_mount(args, source, target=None, readonly=True):
    source = checked_path(str(source))
    target = str(target or source)
    args += ["--mount", f"type=bind,source={source},target={target}" + (",readonly" if readonly else "")]


def docker_base(prefix, build=False):
    args = ["docker", "run", "--rm", "--pull=never", "--network=none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--user", f"{os.getuid()}:{os.getgid()}", "--shm-size=1g", "--pids-limit=2048",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=2147483648,mode=1777",
            "--entrypoint", "/bin/bash"]
    add_mount(args, ROOT)
    add_mount(args, prefix, readonly=not build)
    if not build:
        for name in ("cache", "container-home"):
            directory = prefix / name
            if directory.is_symlink() or not directory.is_dir() or directory.stat().st_uid != os.getuid():
                raise ValueError(f"runtime cache/home must be owned package directories: {directory}")
            add_mount(args, directory, readonly=False)
    for mount in config()["mounts"]:
        add_mount(args, mount["source"], mount["target"])
    if not build:
        for mount in config()["runtime_mounts"]:
            add_mount(args, mount["source"], mount["target"])
    return args


def select_gpu(sysfs=Path("/sys/class/drm"), devroot=Path("/dev")):
    """Read sysfs identity; never assume renderD128 or expose the entire dri directory."""
    matches = []
    for node in sorted(sysfs.glob("renderD*")):
        device = (node / "device").resolve()
        try:
            unique = (device / "unique_id").read_text().strip().lower().removeprefix("gpu-").removeprefix("0x")
            vendor = (device / "vendor").read_text().strip().lower()
            device_id = (device / "device").read_text().strip().lower()
        except OSError:
            continue
        if unique != UUID[4:].lower():
            continue
        if vendor != "0x1002" or device_id != "0x66a1":
            raise ValueError("MI50 UUID has unexpected PCI vendor/device ID")
        render = devroot / "dri" / node.name
        expected = tuple(int(n) for n in (node / "dev").read_text().strip().split(":"))
        st = render.stat()
        if not stat.S_ISCHR(st.st_mode) or (os.major(st.st_rdev), os.minor(st.st_rdev)) != expected:
            raise ValueError("render node does not match sysfs device number")
        matches.append((render, device.name))
    if len(matches) != 1:
        raise ValueError(f"need exactly one verified {UUID} render node; found {len(matches)}")
    if not stat.S_ISCHR((devroot / "kfd").stat().st_mode):
        raise ValueError("/dev/kfd is not a character device")
    return matches[0]


def check_cwd(prefix):
    cwd = checked_path(str(Path.cwd()))
    if cwd in (Path.home().resolve(), Path("/tmp"), Path("/home"), Path("/var"), Path("/srv")):
        raise ValueError("run from a dedicated calculation directory")
    if not cwd.is_dir() or cwd.stat().st_uid != os.getuid():
        raise ValueError("calculation directory must exist and belong to the invoking UID")
    if Path("/media") == cwd or Path("/media") in cwd.parents:
        # /media/<user>/<volume>/<work-area>/<case>; never bind the volume/work-area root.
        if len(cwd.parts) < 6:
            raise ValueError("under /media, select a dedicated case below a work-area directory")
    for protected in [prefix, *protected_paths(runtime=True)]:
        if overlaps(cwd, protected):
            raise ValueError(f"working directory overlaps protected path: {protected}")
    return cwd


def forwarded_environment(host):
    result = {key: host.get(key, value) for key, value in THREAD_DEFAULTS.items()}
    # Explicit empty values also clear any diagnostic defaults baked into the image.
    result.update({key: host.get(key, "") for key in DIAGNOSTICS})
    result["CP2K_PW_GPU_RES_RESERVE_MB"] = host.get("CP2K_PW_GPU_RES_RESERVE_MB", "1024")
    return result


def recipe_values(prefix):
    cfg = config()
    values = {"prefix": str(prefix)}
    libraries = [p.format_map(values) for p in cfg["library_paths"]]
    prefixes = [p.format_map(values) for p in cfg["prefix_paths"]]
    values.update(cmake_prefix=";".join(prefixes), cmake_library=";".join(libraries))
    return values, libraries


def environment(prefix, runtime=False):
    cfg = config()
    values, libraries = recipe_values(prefix)
    rocm = cfg["rocm"]
    # Start clean: neither host secrets nor HIP_VISIBLE_DEVICES/HSA overrides survive.
    env = {"PATH": f"{rocm}/bin:{rocm}/llvm/bin:/usr/bin:/bin", "LC_ALL": "C",
           "HOME": str(prefix / "container-home"), "XDG_CACHE_HOME": str(prefix / "cache"),
           "LD_LIBRARY_PATH": ":".join(libraries), "CP2K_DATA_DIR": str(ROOT / "source/data"),
           "ROCM_PATH": rocm, "HIP_PATH": rocm, "HIP_PLATFORM": "amd"}
    if runtime:
        env.update({k: v for k, v in forwarded_environment(os.environ).items() if v})
        env["ROCR_VISIBLE_DEVICES"] = UUID
    else:
        env.update(PKG_CONFIG_PATH=":".join(p.format_map(values) for p in cfg["pkg_config_paths"]),
                   LIBRARY_PATH=":".join(libraries), HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx906",
                   HCC_AMDGPU_TARGET="gfx906", AMDGPU_TARGETS="gfx906", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", ROCR_VISIBLE_DEVICES="")
    return env


def check_ldd(text, prefix):
    if re.search(r"\bnot found\b", text, re.I):
        raise ValueError("ldd reports missing libraries")
    matches = re.findall(r"^\s*(libcp2k\S*)\s+=>\s+(.+?)\s+\(0x[0-9a-fA-F]+\)\s*$", text, re.M)
    if len(matches) != 1:
        raise ValueError("ldd must resolve exactly one libcp2k")
    soname, resolved = matches[0]
    expected = prefix / "install/cp2k/lib" / soname
    actual = Path(resolved)
    if actual != expected or not expected.is_file() or expected.resolve().parent != expected.parent.resolve():
        raise ValueError(f"libcp2k must resolve to this exact installation: {expected}")
    # Reject an installed lib directory symlink to an old installation as well.
    if expected.parent.resolve() != expected.parent:
        raise ValueError("installed lib directory points outside the prefix")
    return expected


def ldd_check(prefix, env, save=False):
    binary = prefix / "install/cp2k/bin/cp2k.psmp"
    if not binary.is_file() or binary.is_symlink():
        raise ValueError("missing or symlinked installed cp2k.psmp")
    proc = subprocess.run(["ldd", str(binary)], env=env, text=True, capture_output=True)
    output = proc.stdout + proc.stderr
    if save:
        (prefix / "logs/ldd.log").write_text(output)
    if proc.returncode:
        raise ValueError(f"ldd exited {proc.returncode}: {output}")
    library = check_ldd(output, prefix)
    print(f"Verified libcp2k: {library}", flush=True)
    return binary, library


def logged(command, log, env):
    with log.open("w") as stream:
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout:
                stream.write(line)
                print(line, end="", flush=True)
            status = proc.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def build(args):
    prefix = check_prefix(args.prefix, fresh=True)
    from verify import verify_tree
    verify_tree(ROOT)
    verify_dependencies()
    claim_prefix(prefix)
    command = docker_base(prefix, build=True)
    command += ["--workdir", str(prefix / "build"), config()["image"],
                str(ROOT / "recipe/build_inside_container.sh"), "--prefix", str(prefix),
                "--jobs", str(args.jobs)]
    status = subprocess.call(command)
    if status == 0:
        check_success(prefix)
    return status


def inside_build(args):
    prefix = check_prefix(args.prefix)
    env = environment(prefix)
    values, _ = recipe_values(prefix)
    build_dir = prefix / "build/cp2k"
    if any((prefix / "build").iterdir()) or any((prefix / "install").iterdir()):
        raise ValueError("inside-build requires empty build/install directories")
    flags = [s.format_map(values) for s in config()["configure_args"]]
    # The image is local, and Docker networking is disabled even for CMake.
    flags += ["-DFETCHCONTENT_FULLY_DISCONNECTED=ON", "-DFETCHCONTENT_UPDATES_DISCONNECTED=ON"]
    logged(["cmake", "-S", str(ROOT / "source"), "-B", str(build_dir), "-G", "Ninja", *flags],
           prefix / "logs/configure.log", env)
    logged(["cmake", "--build", str(build_dir), "-j", str(args.jobs)], prefix / "logs/build.log", env)
    logged(["cmake", "--install", str(build_dir)], prefix / "logs/install.log", env)
    binary, library = ldd_check(prefix, env, save=True)
    write_launcher(prefix)
    (prefix / SUCCESS).write_bytes(json_bytes({"manifest_sha256": file_hash(ROOT / "manifest.json"),
        "image": config()["image"], "binary_sha256": file_hash(binary), "library_sha256": file_hash(library)}))
    print("BUILD_OK: installed and linked; no GPU calculation was launched")
    return 0


def write_launcher(prefix):
    launcher = prefix / "bin/cp2k.psmp"
    text = ("#!/usr/bin/env bash\nset -euo pipefail\nexec bash " +
            shlex.quote(str(ROOT / "scripts/run.sh")) + " --prefix " +
            shlex.quote(str(prefix)) + ' -- "$@"\n')
    with launcher.open("x") as stream:
        stream.write(text)
    launcher.chmod(0o755)
    subprocess.run(["bash", "-n", str(launcher)], check=True)


def check_success(prefix):
    path = prefix / SUCCESS
    if path.is_symlink() or not path.is_file():
        raise ValueError("prefix has no successful package build record")
    data = json.loads(path.read_text())
    if data.get("manifest_sha256") != file_hash(ROOT / "manifest.json") or data.get("image") != config()["image"]:
        raise ValueError("successful build record does not match this package/image")
    return data


def run(args):
    prefix = check_prefix(args.prefix)
    check_success(prefix)
    cwd = None if args.check else check_cwd(prefix)
    verify_dependencies(runtime=True)
    command = docker_base(prefix)
    if not args.check:
        render, pci = select_gpu()
        print(f"MI50: {UUID}, PCI {pci}, {render}", flush=True)
        command += ["--device", "/dev/kfd", "--device", str(render)]
        for group in sorted({Path("/dev/kfd").stat().st_gid, render.stat().st_gid}):
            command += ["--group-add", str(group)]
    if cwd is not None:
        add_mount(command, cwd, readonly=False)
    command += ["--workdir", str(cwd) if cwd is not None else "/tmp"]
    for key, value in forwarded_environment(os.environ).items():
        command += ["--env", f"{key}={value}"]
    command += ["--env", f"ROCR_VISIBLE_DEVICES={UUID}"]
    command += [config()["image"], "-c", 'exec /usr/bin/python3 -B "$@"', "cp2k-stage6",
                str(ROOT / "scripts/launcher.py"), "inside-run", "--prefix", str(prefix)]
    if args.check:
        command += ["--check"]
    else:
        command += ["--", *args.cp2k_args]
    return subprocess.call(command)


def inside_run(args):
    prefix = check_prefix(args.prefix)
    success = check_success(prefix)
    env = environment(prefix, runtime=True)
    binary, library = ldd_check(prefix, env)
    if file_hash(binary) != success["binary_sha256"] or file_hash(library) != success["library_sha256"]:
        raise ValueError("installed binary/libcp2k changed since BUILD_OK")
    if args.check:
        # This snapshot initializes offload even for --version; never invoke it here.
        print("CHECK_OK: ldd and installed binary/libcp2k hashes verified; CP2K was not invoked", flush=True)
        return 0
    for key in ("HOME", "XDG_CACHE_HOME"):
        Path(env[key]).mkdir(exist_ok=True)
    command = ["/usr/bin/mpirun", "-np", "1", "--nooversubscribe", "--bind-to", "none",
               str(binary), *args.cp2k_args]
    os.execvpe(command[0], command, env)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("build", "run", "inside-build", "inside-run"):
        child = sub.add_parser(action)
        child.add_argument("--prefix", default=str(Path.home() / "cp2k-mi50-stage6-release"))
        if action.endswith("build"):
            child.add_argument("--jobs", type=int, default=16)
        else:
            child.add_argument("--check", action="store_true", help="ldd and installed hashes only; no CP2K invocation or GPU")
            child.add_argument("cp2k_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action.endswith("build") and args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    if args.action.endswith("run"):
        if args.check and args.cp2k_args:
            parser.error("--check cannot be combined with CP2K arguments")
        if not args.check:
            if not args.cp2k_args or args.cp2k_args[0] != "--" or len(args.cp2k_args) == 1:
                parser.error("pass CP2K arguments after -- (or use --check)")
            args.cp2k_args = args.cp2k_args[1:]
    return {"build": build, "run": run, "inside-build": inside_build, "inside-run": inside_run}[args.action](args)


if __name__ == "__main__":
    try:
        status = main()
        raise SystemExit(status if status >= 0 else 128 - status)
    except subprocess.CalledProcessError as error:
        print(f"ERROR: command exited {error.returncode}: {error.cmd[0]}", file=sys.stderr)
        raise SystemExit(error.returncode if error.returncode > 0 else 128 - error.returncode)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"ERROR: {error}")
