#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Permanent Stage7 adapters for two fixed installations; no migration or fetching."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True
PROJECT = Path('/home/nowr/cp2k-hip')
ROOTS = {gpu: PROJECT / name for gpu, name in (('mi50', 'build-mi50'), ('rx', 'build-9070xt'))}
RUNTIME = Path(__file__).absolute().parent
SPACK = Path('/home/nowr/spack/opt/spack')
IMAGE = 'sha256:eb883e666c1d9787ce3696c50c2256c45f6c4a5107dc6af66c2a781cda7f4fd4'
UUIDS = {'mi50': 'GPU-37ac3981732c7340', 'rx': 'GPU-6284fb6bbe682160'}
RECIPES = {'mi50': '456bd37deefe8daa8367f7032c7c37c4fa7171523a3c9581452e5966d697cc57',
           'rx': 'a11ee2a38125ff2d533ef9348821383c26cbd6ad4a20a10ba077e248d4fdc7bd'}
LIB = 'libcp2k.so.2026.2'
THREADS = dict(OMP_NUM_THREADS='16', OPENBLAS_NUM_THREADS='1', OMP_DYNAMIC='FALSE',
               OMP_PROC_BIND='FALSE', OMP_STACKSIZE='256M')
DIAGNOSTICS = (
    'CP2K_GPU_HOST_FALLBACK', 'CP2K_GPU_MEM_LIMIT_MB', 'CP2K_GPU_MEM_LIMIT_MB_RANK0',
    'CP2K_PW_GPU_MAP_CACHE', 'CP2K_PW_GPU_FUSED_XC', 'CP2K_PW_GPU_RES',
    'CP2K_PW_GPU_RES_DENSITY', 'CP2K_PW_GPU_RES_HARTREE', 'CP2K_PW_GPU_RES_XCIN',
    'CP2K_PW_GPU_RES_POT', 'CP2K_PW_GPU_RES_VERIFY', 'CP2K_PW_GPU_RES_STATS',
    'CP2K_PW_GPU_RES_LAZY_DENSITY', 'CP2K_PW_GPU_RES_LAZY_POT', 'CP2K_PW_GPU_RES_NATURAL_G',
    'CP2K_PW_GPU_RES_NATURAL_XC', 'CP2K_PW_GPU_RES_XC_PAIRS', 'CP2K_PW_GPU_RES_XC_DZ',
    'CP2K_PW_GPU_RES_RESERVE_MB', 'CP2K_PW_GPU_MEM_SAMPLE',
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe(raw):
    path = Path(raw)
    require(path.is_absolute() and '..' not in path.parts and not any(ord(c) < 32 for c in str(raw))
            and not any(c in str(raw) for c in ',:;\\'), 'ambiguous absolute path')
    require(not any(p.is_symlink() for p in (path, *path.parents)), f'symlinked path: {path}')
    return path


def owned(path):
    path = safe(path)
    require(path.is_dir() and path.stat().st_uid == os.getuid(), f'owned directory required: {path}')
    return path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(safe(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with safe(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def config(gpu):
    path = safe(RUNTIME / f'dependencies-{gpu}.json')
    require(sha(path) == RECIPES[gpu], f'{gpu} dependency recipe changed')
    return read(path)


def recipe():
    cfg, root = config('mi50'), ROOTS['mi50']
    require(cfg['image'] == IMAGE, 'wrong MI50 image')
    values = dict(prefix=str(root))
    libraries = [p.format_map(values) for p in cfg['library_paths']]
    pkg = [p.format_map(values) for p in cfg['pkg_config_paths']]
    values.update(cmake_prefix=';'.join(p.format_map(values) for p in cfg['prefix_paths']),
                  cmake_library=';'.join(libraries))
    flags = [p.format_map(values) for p in cfg['configure_args']]
    flags += ['-DCP2K_WITH_GPU_PROFILING=OFF', '-DFETCHCONTENT_FULLY_DISCONNECTED=ON',
              '-DFETCHCONTENT_UPDATES_DISCONNECTED=ON']
    return libraries, pkg, flags


def environment(gpu, build=False, host=None):
    host = os.environ if host is None else host
    cfg, root = config(gpu), ROOTS[gpu]
    rocm = cfg['rocm']
    libraries = recipe()[0] if gpu == 'mi50' else cfg['library_paths']
    # A whitelist also excludes image/session loader, selector and MPI overrides.
    env = dict(PATH=f'{rocm}/bin:' + (f'{rocm}/llvm/bin:' if gpu == 'mi50' else '') + '/usr/bin:/bin',
               LC_ALL='C', ROCM_PATH=rocm, HIP_PATH=rocm, HIP_PLATFORM='amd',
               LD_LIBRARY_PATH=':'.join(libraries), CP2K_DATA_DIR=str(root / 'source/data'))
    if gpu == 'mi50':
        env.update(HOME=str(root / 'container-home'), XDG_CACHE_HOME=str(root / 'cache'))
    else:
        env.update(cfg['environment'])
    if build:
        env.update(PKG_CONFIG_PATH=':'.join(recipe()[1]), LIBRARY_PATH=':'.join(libraries),
                   HIPCC_COMPILE_FLAGS_APPEND='--offload-arch=gfx906', HCC_AMDGPU_TARGET='gfx906',
                   AMDGPU_TARGETS='gfx906', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   ROCR_VISIBLE_DEVICES='')
    else:
        threads = host.get('OMP_NUM_THREADS', '16')
        require(re.fullmatch(r'[0-9]+', threads) and 1 <= int(threads) <= 32,
                'OMP_NUM_THREADS must be an integer in 1..32')
        env.update(THREADS, OMP_NUM_THREADS=threads, ROCR_VISIBLE_DEVICES=UUIDS[gpu])
        env.update({k: host[k] for k in DIAGNOSTICS if k in host})
        env.setdefault('CP2K_PW_GPU_RES_RESERVE_MB', '1024')
        env.setdefault('CP2K_PW_GPU_MEM_SAMPLE', '1')
    return env


def identity():
    root, digest = ROOTS['mi50'], hashlib.sha256()
    source = owned(root / 'source')
    require(safe(source / 'CMakeLists.txt').is_file(), 'source/CMakeLists.txt required')
    for path in sorted(source.rglob('*')):
        safe(path)
        require(path.is_file() or path.is_dir(), f'special source entry: {path}')
        if path.is_file():
            digest.update(path.relative_to(source).as_posix().encode() + b'\0' + bytes.fromhex(sha(path)))
    return dict(schema=1, root=str(root), source_sha256=digest.hexdigest(),
                recipe_sha256=RECIPES['mi50'], image=IMAGE, flags=recipe()[2])


def case_directory(gpu, raw):
    case, root = owned(raw), ROOTS[gpu]
    validation = root / 'validation'
    if validation in case.parents:
        for path in (case, *case.parents):
            owned(path)
            if path == validation:
                return case
    require(case not in (Path.home(), *map(Path, ('/tmp', '/home', '/var', '/srv', '/mnt'))),
            'use a dedicated calculation directory')
    require('/media' not in [str(p) for p in (case, *case.parents)] or len(case.parts) >= 6,
            'under /media use volume/work-area/case, not the volume or work-area root')
    protected = (PROJECT, SPACK, Path('/opt'), *map(Path, (
        '/bin', '/boot', '/dev', '/etc', '/lib', '/lib64', '/proc', '/sys', '/usr')))
    require(not any(case == p or p in case.parents or case in p.parents for p in protected),
            'calculation directory overlaps a project/dependency/system root')
    return case


def current_case(gpu):
    cwd = Path.cwd()
    logical = Path(os.environ.get('PWD', str(cwd)))
    return case_directory(gpu, logical if logical.resolve() == cwd else cwd)


def select_gpu(sysfs=Path('/sys/class/drm'), devroot=Path('/dev')):
    matches = []
    for node in sorted(sysfs.glob('renderD*')):
        device = (node / 'device').resolve()
        try:
            unique = (device / 'unique_id').read_text().strip().lower().removeprefix('gpu-').removeprefix('0x')
            vendor, pci_id = ((device / n).read_text().strip().lower() for n in ('vendor', 'device'))
        except OSError:
            continue
        if unique != UUIDS['mi50'][4:]:
            continue
        require((vendor, pci_id) == ('0x1002', '0x66a1'), 'MI50 UUID has wrong vendor/device')
        render = devroot / 'dri' / node.name
        st = render.stat()
        expected = tuple(map(int, (node / 'dev').read_text().strip().split(':')))
        require(stat.S_ISCHR(st.st_mode) and (os.major(st.st_rdev), os.minor(st.st_rdev)) == expected,
                'render character device does not match sysfs')
        matches.append(render)
    require(len(matches) == 1, 'need exactly one verified MI50 render node')
    require(stat.S_ISCHR((devroot / 'kfd').stat().st_mode), 'kfd is not a character device')
    return matches[0]


def docker(command, env, mode='check', case=None):
    root, cfg = ROOTS['mi50'], config('mi50')
    argv = ['/usr/bin/docker', '--host', 'unix:///var/run/docker.sock', 'run', '--rm', '--pull=never',
            '--network=none', '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--user', f'{os.getuid()}:{os.getgid()}', '--shm-size=1g', '--pids-limit=2048',
            '--tmpfs', '/tmp:rw,nosuid,nodev,size=2147483648,mode=1777', '--tmpfs',
            f'{root.parent}:rw,nosuid,nodev,mode=0755,uid={os.getuid()},gid={os.getgid()}',
            '--entrypoint', '/usr/bin/env']

    def mount(path, readonly=True, target=None):
        require(safe(path).is_dir(), f'missing mount: {path}')
        argv.extend(['--mount', f'type=bind,source={path},target={safe(target or path)}'
                     + (',readonly' if readonly else '')])

    for path in (root, RUNTIME, root / 'source', root / 'deps'):
        mount(path)
    mount(root / 'install', readonly=mode != 'build')
    for name in ('cache', 'container-home'):
        mount(owned(root / name), readonly=mode == 'check')
    if mode == 'build':
        for name in ('build', 'logs'):
            mount(owned(root / name), readonly=False)
    for item in cfg['mounts'] + cfg['runtime_mounts']:
        mount(item['source'], target=item['target'])
    if mode == 'run':
        require(case is not None, 'calculation directory required')
        render = select_gpu()
        argv += ['--device', '/dev/kfd', '--device', str(render)]
        for group in sorted({Path('/dev/kfd').stat().st_gid, render.stat().st_gid}):
            argv += ['--group-add', str(group)]
        mount(case, readonly=False)
    argv += ['--workdir', str(case) if case else '/tmp', IMAGE, '-i']
    return argv + [f'{k}={v}' for k, v in env.items()] + command


def artifacts(gpu, build=False):
    root = ROOTS[gpu] / ('build/cp2k' if build else 'install/cp2k')
    paths = (root / 'bin/cp2k.psmp', root / ('src' if build else 'lib') / LIB)
    return {str(safe(p)): sha(safe(p)) for p in paths}


def parse_ldd(output, gpu):
    result, root = {}, ROOTS[gpu]
    for line in output.splitlines():
        line = re.sub(r'\s*\(0x[0-9a-fA-F]+\)\s*$', '', line).strip()
        if not line or line.startswith('linux-vdso'):
            continue
        require('not found' not in line, 'ldd reports a missing library')
        name, raw = line.split(' => ', 1) if ' => ' in line else (Path(line).name, line)
        require(raw.startswith('/') and name not in result, f'unrecognized ldd line: {line}')
        path = Path(raw).resolve(strict=True)
        require(path.is_file(), f'missing library: {path}')
        for p in (Path(raw), path):
            require(not p.is_relative_to(PROJECT) or p.is_relative_to(root / 'install/cp2k/lib')
                    or gpu == 'mi50' and p.is_relative_to(root / 'deps'), 'foreign project library')
        if name.startswith('libcp2k'):
            require(name == LIB and Path(raw) == safe(root / 'install/cp2k/lib' / LIB)
                    and path == Path(raw), 'wrong installed libcp2k')
        result[name] = str(path)
    require(LIB in result, 'ldd must resolve the installed libcp2k')
    return result


def inspect_install(gpu, env):
    hashes = artifacts(gpu)
    proc = subprocess.run(['/usr/bin/ldd', str(ROOTS[gpu] / 'install/cp2k/bin/cp2k.psmp')],
                          env=env, text=True, capture_output=True, check=True)
    return dict(artifacts=hashes, ldd=parse_ldd(proc.stdout + proc.stderr, gpu))


def installed_state(gpu):
    state = read(ROOTS[gpu] / 'stage7-build.json') if gpu == 'mi50' else config('rx')
    if gpu == 'mi50':
        require(state['identity'] == identity(), 'source/recipe/image build identity changed')
    require(artifacts(gpu) == state['artifacts'], 'installed artifact hash mismatch')
    return state


def checked(gpu, env):
    state = installed_state(gpu)
    actual = inspect_install(gpu, env)
    require(actual['artifacts'] == state['artifacts'] and actual['ldd'] == state['ldd'],
            'installed hashes or resolved ldd map changed')
    return actual


def execute(gpu, args, check=False):
    env = environment(gpu)
    checked(gpu, env)
    if check:
        print('CHECK_OK: installed hashes and resolved ldd map verified; CP2K was not invoked')
        return 0
    current_case(gpu)
    command = [str(ROOTS[gpu] / 'install/cp2k/bin/cp2k.psmp'), *args]
    if gpu == 'mi50':
        command = ['/usr/bin/mpirun', '-np', '1', '--nooversubscribe', '--bind-to', 'none', *command]
    os.execvpe(command[0], command, env)


def phase(logdir, name, command, gpu=None, build=False):
    entry = dict(command=command, start_unix_ns=time.time_ns(), exit=1)
    try:
        with (logdir / (name + '.log')).open('xb') as log:
            proc = subprocess.run(command, env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'},
                                  stdout=log, stderr=subprocess.STDOUT)
        entry['exit'] = proc.returncode
        proc.check_returncode()
        if gpu:
            entry['artifacts'] = artifacts(gpu, build)
    except OSError as exc:
        entry.update(exit=127 if isinstance(exc, FileNotFoundError) else 126, error=str(exc))
        raise subprocess.CalledProcessError(entry['exit'], command) from exc
    finally:
        entry['end_unix_ns'] = time.time_ns()
        write_new(logdir / (name + '.json'), entry)
        print(f'{name}: exit {entry["exit"]}; {logdir / (name + ".log")}', flush=True)
    return entry


def build_mi50(jobs):
    root = ROOTS['mi50']
    require(jobs > 0, 'jobs must be positive')
    for name in ('stage7-build.json', 'stage7-attempt.json'):
        require(not safe(root / name).exists(), 'prior build/attempt exists; no automatic overwrite or resume')
    for name in ('build', 'install'):
        require(not any(owned(root / name).iterdir()), 'fresh build requires empty build/install')
    expected, env = identity(), environment('mi50', build=True)
    owned(root / 'logs')
    logdir = root / 'logs' / ('stage7-' + str(time.time_ns()))
    builddir = root / 'build/cp2k'
    commands = [
        ['cmake', '-S', str(root / 'source'), '-B', str(builddir), '-G', 'Ninja', *recipe()[2]],
        ['cmake', '--build', str(builddir), '-j', str(jobs)],
        ['cmake', '--install', str(builddir)],
    ]
    commands = [docker(c, env, mode='build') for c in commands]
    write_new(root / 'stage7-attempt.json', expected)
    logdir.mkdir()
    phase(logdir, 'configure', commands[0])
    cache = {m[1]: m[2] for line in safe(builddir / 'CMakeCache.txt').read_text().splitlines()
             if (m := re.match(r'^([^#/:][^:]*):[^=]+=(.*)$', line))}
    required = dict(CMAKE_HOME_DIRECTORY=str(root / 'source'), CMAKE_CACHEFILE_DIR=str(builddir),
                    CMAKE_INSTALL_PREFIX=str(root / 'install/cp2k'), CMAKE_HIP_ARCHITECTURES='gfx906',
                    GPU_TARGETS='gfx906', GPU_BUILD_TARGETS='gfx906', AMDGPU_TARGETS='gfx906',
                    CP2K_WITH_GPU_PROFILING='OFF')
    require(all(cache.get(k) == v for k, v in required.items()), 'configured paths/architecture/profiling drift')
    phase(logdir, 'build', commands[1], 'mi50', build=True)
    phase(logdir, 'install', commands[2], 'mi50')
    command = docker(['/usr/bin/python3', '-B', str(RUNTIME / 'stage7.py'), '_inspect', '--gpu', 'mi50'],
                     environment('mi50'))
    phase(logdir, 'check', command)
    result = read(logdir / 'check.log')
    require(expected == identity() and result['artifacts'] == artifacts('mi50'), 'build identity/artifacts drift')
    write_new(root / 'stage7-build.json', dict(identity=expected, **result, logs=str(logdir)))
    print('BUILD_OK: fresh final installation and no-GPU loader check complete')
    return 0


def install_entry(gpu):
    root = ROOTS[gpu]
    safe(root / 'bin').mkdir(exist_ok=True)
    path = owned(root / 'bin') / 'cp2k.psmp'
    script = shlex.quote(str(RUNTIME / 'stage7.py'))
    content = ('#!/bin/bash\nset -euo pipefail\n' + f'export ROCR_VISIBLE_DEVICES={UUIDS[gpu]}\n'
               + 'if [[ "${1-}" == --check ]]; then\n    shift\n'
               + f'    exec /usr/bin/python3 -B {script} check --gpu {gpu} "$@"\nfi\n'
               + f'exec /usr/bin/python3 -B {script} run --gpu {gpu} -- "$@"\n')
    safe(path)
    if path.exists():
        require(path.is_file() and path.stat().st_nlink == 1 and path.read_text() == content
                and path.stat().st_uid == os.getuid() and path.stat().st_mode & 0o111,
                'existing entry differs; coordinator must archive/stage it first')
        return 0
    with path.open('x', encoding='utf-8') as stream:
        stream.write(content)
    path.chmod(0o755)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('check', 'run', 'install-entry', '_check', '_run', '_inspect', 'build-mi50'):
        p = sub.add_parser(action)
        if action == 'build-mi50':
            p.add_argument('--jobs', type=int, default=16)
        else:
            p.add_argument('--gpu', choices=ROOTS, required=True)
        if action in ('run', '_run'):
            p.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    require(os.name == 'posix', 'runtime requires Linux/POSIX')
    require(safe(RUNTIME) == PROJECT / 'runtime', 'install helper at the fixed project/runtime path')
    gpu = getattr(args, 'gpu', 'mi50')
    owned(PROJECT)
    owned(ROOTS[gpu])
    config(gpu)
    if args.action in ('run', '_run'):
        require(len(args.args) >= 2 and args.args[0] == '--', 'pass CP2K arguments after --')
        args.args = args.args[1:]
    if args.action.startswith('_'):
        require(gpu == 'mi50' and Path('/.dockerenv').is_file(), 'internal MI50 action requires Docker')
        if args.action == '_inspect':
            print(json.dumps(inspect_install(gpu, environment(gpu)), sort_keys=True))
            return 0
        return execute(gpu, getattr(args, 'args', []), args.action == '_check')
    if args.action == 'install-entry':
        return install_entry(gpu)
    if args.action == 'build-mi50':
        return build_mi50(args.jobs)
    if gpu == 'rx':
        return execute(gpu, getattr(args, 'args', []), args.action == 'check')
    installed_state(gpu)  # Reject stale artifacts before selecting or exposing any device.
    case = current_case(gpu) if args.action == 'run' else None
    command = ['/usr/bin/python3', '-B', str(RUNTIME / 'stage7.py'), '_' + args.action, '--gpu', gpu]
    if case:
        command += ['--', *args.args]
    command = docker(command, environment(gpu), mode=args.action, case=case)
    return subprocess.call(command, env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'})


if __name__ == '__main__':
    try:
        code = main()
    except subprocess.CalledProcessError as error:
        code = error.returncode
    except (OSError, ValueError, KeyError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        code = 130
    sys.exit(code if code >= 0 else 128 - code)
