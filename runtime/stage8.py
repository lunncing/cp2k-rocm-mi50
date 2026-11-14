#!/usr/bin/env python3
"""Stage8 fixed-root build/install/run adapters, copied from the frozen Stage7 runtime."""
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
ROOTS = {gpu: PROJECT / name for gpu, name in (
    ('mi50', 'build-mi50'), ('rx', 'build-9070xt'),
    ('rx-validation', 'build-9070xt-stage8-validation'))}
RUNTIME = Path(__file__).absolute().parent
SPACK = Path('/home/nowr/spack/opt/spack')
IMAGE = 'sha256:eb883e666c1d9787ce3696c50c2256c45f6c4a5107dc6af66c2a781cda7f4fd4'
UUIDS = {'mi50': 'GPU-37ac3981732c7340', 'rx': 'GPU-6284fb6bbe682160',
         'rx-validation': 'GPU-6284fb6bbe682160'}
RECIPES = {'mi50': '456bd37deefe8daa8367f7032c7c37c4fa7171523a3c9581452e5966d697cc57',
           'rx': 'a11ee2a38125ff2d533ef9348821383c26cbd6ad4a20a10ba077e248d4fdc7bd'}
RX_CACHE_SHA = 'df9e8e1a20ffc26e1091a96a6d5f8a713c3aab59097edb85defc849f3f992137'
STAGE7_SHA = 'ec1b70d8b6471e49ee67959d431082a16d78939f73e02c0e51fe3aaa0ba55157'
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
    'CP2K_PW_GPU_UKS', 'CP2K_PW_GPU_XC_FULL_PIPELINE', 'CP2K_ELPA_GPU',
    'CP2K_ELPA_GPU_N_MIN', 'CP2K_CHOLESKY_FAST_SINGLE_RANK',
    'CP2K_PW_GPU_PBE', 'CP2K_PW_GPU_PBE_LSD',
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
    gpu = 'mi50' if gpu == 'mi50' else 'rx'
    path = safe(RUNTIME / f'dependencies-{gpu}.json')
    require(sha(path) == RECIPES[gpu], f'{gpu} dependency recipe changed')
    return read(path)


def cache_values(path):
    return {match[1]: (match[2], match[3]) for line in safe(path).read_text().splitlines()
            if (match := re.match(r'^([^#/:][^:]*):([^=]+)=(.*)$', line))}


def rewrite_rx(value, root):
    original = str(ROOTS['rx'])
    return re.sub(re.escape(original) + r'/(?=recipe(?:/|;|$)|include(?:/|;|$)|install(?:/|;|$))',
                  str(root) + '/', value)


def recipe(gpu='mi50'):
    cfg, root = config(gpu), ROOTS[gpu]
    if gpu != 'mi50':
        cachefile = RUNTIME.parent / 'CMakeCache-rx.txt'
        require(sha(safe(cachefile)) == RX_CACHE_SHA, 'frozen RX cache changed')
        selected = {}
        fields = set(('BUILD_SHARED_LIBS CMAKE_BUILD_TYPE CMAKE_PREFIX_PATH CMAKE_INSTALL_PREFIX '
                      'CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_RPATH CMAKE_INSTALL_RPATH_USE_LINK_PATH '
                      'CMAKE_POSITION_INDEPENDENT_CODE CMAKE_SKIP_RPATH CMAKE_SKIP_INSTALL_RPATH '
                      'CMAKE_EXPORT_COMPILE_COMMANDS CMAKE_FIND_USE_PACKAGE_REGISTRY '
                      'CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY CMAKE_HIP_ARCHITECTURES '
                      'CMAKE_HIP_PLATFORM GPU_TARGETS GPU_BUILD_TARGETS HIP_PLATFORM '
                      'FETCHCONTENT_FULLY_DISCONNECTED Python_EXECUTABLE '
                      'COSMA_BLAS_INTERFACE COSMA_BLAS_THREADING').split())
        for key, (kind, value) in cache_values(cachefile).items():
            configurable = key in fields or key.startswith('CP2K_') or re.fullmatch(
                r'CMAKE_(C|CXX|Fortran|HIP)_(COMPILER|FLAGS(?:_[A-Z]+)?)|'
                r'CMAKE_(EXE|SHARED|MODULE|STATIC)_LINKER_FLAGS(?:_[A-Z]+)?', key)
            dependency = key.endswith('_DIR') and value.startswith(('/opt/', str(SPACK) + '/',
                                                                     str(ROOTS['rx'] / 'recipe') + '/'))
            if kind in ('BOOL', 'STRING', 'PATH') and (configurable or dependency) and 'NOTFOUND' not in value:
                selected[key] = (kind, rewrite_rx(value, root))
            if key == 'FYPP_EXECUTABLE' and value == '/home/nowr/.local/bin/fypp':
                selected[key] = ('FILEPATH', value)
        require(sum(key.startswith('CP2K_') and kind == 'BOOL'
                    for key, (kind, value) in selected.items()) == 49, 'RX feature baseline incomplete')
        selected.update(AMDGPU_TARGETS=('STRING', 'gfx1201'),
                        FETCHCONTENT_UPDATES_DISCONNECTED=('BOOL', 'ON'))
        flags = [f'-D{key}:{kind}={value}' for key, (kind, value) in sorted(selected.items())]
        libraries = [str(root / 'install/cp2k/lib'), *cfg['library_paths']]
        pkg = [str(Path(value) / 'lib/pkgconfig') for value in
               selected['CMAKE_PREFIX_PATH'][1].split(';')]
        return libraries, pkg, flags
    require(cfg['image'] == IMAGE, 'wrong MI50 image')
    values = dict(prefix=str(root))
    libraries = [path.format_map(values) for path in cfg['library_paths']]
    pkg = [path.format_map(values) for path in cfg['pkg_config_paths']]
    values.update(cmake_prefix=';'.join(path.format_map(values) for path in cfg['prefix_paths']),
                  cmake_library=';'.join(libraries))
    flags = [arg.format_map(values) for arg in cfg['configure_args']]
    flags += ['-DCP2K_WITH_GPU_PROFILING=OFF', '-DFETCHCONTENT_FULLY_DISCONNECTED=ON',
              '-DFETCHCONTENT_UPDATES_DISCONNECTED=ON']
    return libraries, pkg, flags


def environment(gpu, build=False, host=None):
    host = os.environ if host is None else host
    cfg, root = config(gpu), ROOTS[gpu]
    rocm = cfg['rocm']
    libraries, pkg, flags = recipe(gpu)
    env = dict(PATH=f'{rocm}/bin:' + (f'{rocm}/llvm/bin:' if gpu == 'mi50' else '') + '/usr/bin:/bin',
               LC_ALL='C', ROCM_PATH=rocm, HIP_PATH=rocm, HIP_PLATFORM='amd',
               LD_LIBRARY_PATH=':'.join(libraries), CP2K_DATA_DIR=str(root / 'source/data'))
    if gpu == 'mi50':
        env.update(HOME=str(root / 'container-home'), XDG_CACHE_HOME=str(root / 'cache'))
    else:
        env.update(cfg['environment'])
    if build:
        arch = 'gfx906' if gpu == 'mi50' else 'gfx1201'
        env.update(PKG_CONFIG_PATH=':'.join(pkg), LIBRARY_PATH=':'.join(libraries),
                   HIPCC_COMPILE_FLAGS_APPEND=f'--offload-arch={arch}', HCC_AMDGPU_TARGET=arch,
                   AMDGPU_TARGETS=arch, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   ROCR_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', TMPDIR=str(root / 'cache'))
    else:
        threads = host.get('OMP_NUM_THREADS', '16')
        require(re.fullmatch(r'[0-9]+', threads) and 1 <= int(threads) <= 32,
                'OMP_NUM_THREADS must be an integer in 1..32')
        env.update(THREADS, OMP_NUM_THREADS=threads, ROCR_VISIBLE_DEVICES=UUIDS[gpu])
        env.update({key: host[key] for key in DIAGNOSTICS if key in host})
        env.setdefault('CP2K_PW_GPU_RES_RESERVE_MB', '1024')
        env.setdefault('CP2K_PW_GPU_MEM_SAMPLE', '1')
    return env


def tree_manifest(directory, links=False):
    directory, result = owned(directory), {}
    for path in sorted(directory.rglob('*')):
        name = path.relative_to(directory).as_posix()
        if links and path.is_symlink():
            require(path.resolve(strict=True).is_relative_to(directory), f'escaping installed link: {path}')
            result[name] = dict(link=os.readlink(path))
        else:
            safe(path)
            require(path.is_file() or path.is_dir(), f'special tree entry: {path}')
            result[name] = dict(sha256=sha(path)) if path.is_file() else dict(directory=True)
    return result


def identity(gpu):
    root = ROOTS[gpu]
    require(safe(root / 'source/CMakeLists.txt').is_file(), 'source/CMakeLists.txt required')
    source = tree_manifest(root / 'source')
    digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    return dict(schema=8, root=str(root), source=source, source_sha256=digest,
                recipe_sha256=RECIPES['mi50' if gpu == 'mi50' else 'rx'],
                rx_cache_sha256=None if gpu == 'mi50' else RX_CACHE_SHA,
                local_recipe=None if gpu == 'mi50' else tree_manifest(root / 'recipe'),
                runtime_sha256=sha(safe(RUNTIME / 'stage8.py')),
                wrapper_sha256=sha(safe(root / 'bin/cp2k.psmp')),
                image=IMAGE if gpu == 'mi50' else None, flags=recipe(gpu)[2],
                build_environment=environment(gpu, build=True))


def validate_cache(gpu):
    root, arch = ROOTS[gpu], 'gfx906' if gpu == 'mi50' else 'gfx1201'
    cache_path = safe(root / 'build/cp2k/CMakeCache.txt')
    cache = {key: value for key, (kind, value) in cache_values(cache_path).items()}
    required = {arg[2:].split('=', 1)[0].split(':')[0]: arg.split('=', 1)[1] for arg in recipe(gpu)[2]}
    required.update(CMAKE_HOME_DIRECTORY=str(root / 'source'), CMAKE_CACHEFILE_DIR=str(root / 'build/cp2k'),
                    CMAKE_INSTALL_PREFIX=str(root / 'install/cp2k'), CMAKE_HIP_ARCHITECTURES=arch,
                    GPU_TARGETS=arch, GPU_BUILD_TARGETS=arch, AMDGPU_TARGETS=arch,
                    CP2K_WITH_GPU_PROFILING='OFF')
    require(all(cache.get(key) == value for key, value in required.items()),
            'configured flags/features/paths/architecture/profiling drift')
    for key, value in cache.items():
        require('/media/' not in value and 'stage7backup' not in value, f'forbidden cache path: {key}')
        for raw in re.findall(re.escape(str(PROJECT)) + r'/[^;\s"<>]+', value):
            scanned_path = Path(raw)
            if scanned_path.is_symlink():
                subtree = next((root / name for name in (
                    'deps', 'build/cp2k/src', 'build/cp2k/lib', 'build/cp2k/lib64',
                    'install/cp2k/lib', 'install/cp2k/lib64')
                    if scanned_path.is_relative_to(root / name)), None)
                require(subtree is not None, f'cache library link outside selected subtree: {key}')
                seen = set()
                while True:
                    require(scanned_path.is_relative_to(subtree) and '/media/' not in str(scanned_path)
                            and 'stage7backup' not in str(scanned_path), f'escaping cache library link: {key}')
                    require(re.fullmatch(r'lib[A-Za-z0-9_+.-]+\.so(?:\.[0-9]+)*', scanned_path.name),
                            f'shared library name required: {scanned_path}')
                    parent = safe(scanned_path.parent)
                    for ancestor in (parent, *parent.parents):
                        owned(ancestor)
                        if ancestor == root:
                            break
                    require(scanned_path not in seen, f'cyclic cache library link: {key}')
                    seen.add(scanned_path)
                    if not scanned_path.is_symlink():
                        require(safe(scanned_path).is_file(), f'regular cache library required: {key}')
                        break
                    scanned_path = parent / os.readlink(scanned_path)
            else:
                scanned_path = safe(raw)
            require(scanned_path.is_relative_to(root), f'foreign project cache path: {key}')
        if re.fullmatch(r'CMAKE_(?:RUNTIME|ARCHIVE|LIBRARY)_OUTPUT_DIRECTORY(?:_[A-Z]+)?', key):
            require(safe(value).is_relative_to(root / 'build/cp2k'), f'output escapes build: {key}')
    return sha(cache_path)


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
    require('/media' not in [str(path) for path in (case, *case.parents)] or len(case.parts) >= 6,
            'under /media use volume/work-area/case, not the volume or work-area root')
    protected = (PROJECT, SPACK, Path('/opt'), *map(Path, (
        '/bin', '/boot', '/dev', '/etc', '/lib', '/lib64', '/proc', '/sys', '/usr')))
    require(not any(case == path or path in case.parents or case in path.parents for path in protected),
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
            vendor, pci_id = ((device / name).read_text().strip().lower() for name in ('vendor', 'device'))
        except OSError:
            continue
        if unique != UUIDS['mi50'][4:]:
            continue
        require((vendor, pci_id) == ('0x1002', '0x66a1'), 'MI50 UUID has wrong vendor/device')
        render = devroot / 'dri' / node.name
        status = render.stat()
        expected = tuple(map(int, (node / 'dev').read_text().strip().split(':')))
        require(stat.S_ISCHR(status.st_mode) and (os.major(status.st_rdev), os.minor(status.st_rdev)) == expected,
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
    return argv + [f'{key}={value}' for key, value in env.items()] + command


def artifacts(gpu, build=False):
    root = ROOTS[gpu] / ('build/cp2k' if build else 'install/cp2k')
    paths = (root / 'bin/cp2k.psmp', root / ('src' if build else 'lib') / LIB)
    for path in paths:
        with safe(path).open('rb') as stream:
            require(stream.read(4) == b'\x7fELF', f'not an ELF artifact: {path}')
    return {str(path): sha(path) for path in paths}


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
        for candidate in (Path(raw), path):
            require(not candidate.is_relative_to(Path('/media')) and 'stage7backup' not in str(candidate),
                    'forbidden library location')
            require(not candidate.is_relative_to(PROJECT) or candidate.is_relative_to(root / 'install/cp2k/lib')
                    or gpu == 'mi50' and candidate.is_relative_to(root / 'deps'), 'foreign project library')
        if name.startswith('libcp2k'):
            require(name == LIB and Path(raw) == safe(root / 'install/cp2k/lib' / LIB)
                    and path == Path(raw), 'wrong installed libcp2k')
        result[name] = str(path)
    require(LIB in result, 'ldd must resolve the installed libcp2k')
    return result


def inspect_install(gpu, env):
    hashes = artifacts(gpu)
    proc = subprocess.run(['/usr/bin/ldd', str(ROOTS[gpu] / 'install/cp2k/bin/cp2k.psmp')],
                          env=dict(env, ROCR_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES=''),
                          text=True, capture_output=True, check=True)
    return dict(artifacts=hashes, ldd=parse_ldd(proc.stdout + proc.stderr, gpu),
                installed=tree_manifest(ROOTS[gpu] / 'install/cp2k', links=True))


def installed_state(gpu):
    state = read(ROOTS[gpu] / 'stage8-build.json')
    require(state['identity'] == identity(gpu), 'source/recipe/runtime/wrapper build identity changed')
    require(validate_cache(gpu) == state['cache_sha256'], 'configured cache changed')
    require(artifacts(gpu) == state['artifacts'], 'installed artifact hash mismatch')
    require(tree_manifest(ROOTS[gpu] / 'install/cp2k', links=True) == state['installed'],
            'installed tree changed')
    return state


def checked(gpu, env):
    state = installed_state(gpu)
    actual = inspect_install(gpu, env)
    require(all(actual[key] == state[key] for key in ('artifacts', 'ldd', 'installed')),
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


def phase(logdir, name, command, env=None, gpu=None, build=False):
    entry = dict(command=command, start_unix_ns=time.time_ns(), exit=1)
    try:
        with safe(logdir / (name + '.log')).open('xb') as log:
            proc = subprocess.run(command, env=env or {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'},
                                  cwd=logdir, stdout=log, stderr=subprocess.STDOUT)
        entry['exit'] = proc.returncode
        proc.check_returncode()
        if gpu:
            entry['artifacts'] = artifacts(gpu, build)
    except OSError as exc:
        entry.update(exit=127 if isinstance(exc, FileNotFoundError) else 126, error=str(exc))
        raise subprocess.CalledProcessError(entry['exit'], command) from exc
    except (ValueError, KeyError) as exc:
        entry.update(exit=1, error=str(exc))
        raise
    finally:
        entry['end_unix_ns'] = time.time_ns()
        write_new(logdir / (name + '.json'), entry)
        print(f'{name}: exit {entry["exit"]}; {logdir / (name + ".log")}', flush=True)
    return entry


def validation_gate(source_sha256):
    root = ROOTS['rx-validation']
    checked('rx-validation', environment('rx-validation'))
    state = read(root / 'stage8-build.json')
    approval = read(root / 'stage8-validation.json')
    require(state['identity']['source_sha256'] == source_sha256 and approval['status'] == 'PASS'
            and approval['build_manifest_sha256'] == sha(root / 'stage8-build.json'),
            'RX validation evidence is stale or not PASS')
    require(bool(approval['evidence']), 'RX validation evidence required')
    for name, digest in approval['evidence'].items():
        path = safe(root / name)
        require(path.is_relative_to(root / 'validation') and sha(path) == digest,
                'RX validation evidence mismatch')
    return dict(approval_sha256=sha(root / 'stage8-validation.json'),
                build_manifest_sha256=approval['build_manifest_sha256'])


def build_install(gpu, jobs, resume=False):
    root = ROOTS[gpu]
    require(1 <= jobs <= 1024, 'jobs must be in 1..1024')
    require(not safe(root / 'stage8-build.json').exists(), 'completed installation already exists')
    attempt_path = root / 'stage8-attempt.json'
    if not resume:
        require(not safe(attempt_path).exists(), 'prior attempt exists; exact --resume or archive required')
        for name in ('build', 'install'):
            require(not any(owned(root / name).iterdir()), 'fresh build requires empty build/install')
    for name in ('build', 'install', 'cache', 'logs', 'container-home'):
        directory = owned(root / name)
        for path in directory.rglob('*'):
            require(path.resolve(strict=True).is_relative_to(directory), f'escaping output path: {path}')
            require(path.is_file() or path.is_dir(), f'special output entry: {path}')
    install_entry(gpu)
    expected, env = identity(gpu), environment(gpu, build=True)
    gate = validation_gate(expected['source_sha256']) if gpu == 'rx' else None
    logdir = owned(root / 'logs') / ('stage8-' + str(time.time_ns()))
    if resume:
        previous = read(attempt_path)
        require(previous['identity'] == expected and previous['jobs'] == jobs
                and previous['rx_validation'] == gate, 'resume identity/jobs/validation mismatch')
        previous_logs = safe(previous['logs'])
        require(previous_logs.is_relative_to(root / 'logs'), 'resume logs outside selected root')
        checkpoint = read(previous_logs / 'configured.json')
        require(validate_cache(gpu) == checkpoint['cache_sha256'], 'resume cache drift')
    else:
        write_new(attempt_path, dict(identity=expected, jobs=jobs, logs=str(logdir), rx_validation=gate))
    logdir.mkdir()
    builddir = root / 'build/cp2k'
    commands = [
        ['cmake', '-S', str(root / 'source'), '-B', str(builddir), '-G', 'Ninja', *recipe(gpu)[2]],
        ['cmake', '--build', str(builddir), '-j', str(jobs)],
        ['cmake', '--install', str(builddir)],
    ]
    commands = [docker(command, env, mode='build') for command in commands] if gpu == 'mi50' else commands
    phase_env = None if gpu == 'mi50' else env
    if not resume:
        phase(logdir, 'configure', commands[0], phase_env)
    cache_sha = validate_cache(gpu)
    write_new(logdir / 'configured.json', dict(cache_sha256=cache_sha))
    for name, command in zip(('build', 'install'), commands[1:]):
        require(expected == identity(gpu) and cache_sha == validate_cache(gpu), 'build inputs drift')
        phase(logdir, name, command, phase_env, gpu, build=name == 'build')
        require(expected == identity(gpu) and cache_sha == validate_cache(gpu), 'build inputs drift')
    command = ['/usr/bin/python3', '-B', str(RUNTIME / 'stage8.py'), '_inspect', '--gpu', gpu]
    command = docker(command, environment(gpu)) if gpu == 'mi50' else command
    phase(logdir, 'check', command, None if gpu == 'mi50' else environment(gpu))
    result = read(logdir / 'check.log')
    require(expected == identity(gpu) and cache_sha == validate_cache(gpu)
            and result['artifacts'] == artifacts(gpu), 'build identity/artifacts drift')
    write_new(root / 'stage8-build.json', dict(identity=expected, **result, cache_sha256=cache_sha,
                                              logs=str(logdir), rx_validation=gate))
    print('BUILD_OK: fresh installation and no-GPU loader inspection complete')
    return 0


def legacy_mi50_identity():
    require(sha(safe(RUNTIME / 'stage7.py')) == STAGE7_SHA, 'original Stage7 runtime changed')
    for gpu in ('mi50', 'rx'):
        config(gpu)
    root, digest = ROOTS['mi50'], hashlib.sha256()
    source = owned(root / 'source')
    require(safe(source / 'CMakeLists.txt').is_file(), 'source/CMakeLists.txt required')
    for path in sorted(source.rglob('*')):
        safe(path)
        require(path.is_file() or path.is_dir(), f'special source entry: {path}')
        if path.is_file():
            digest.update(path.relative_to(source).as_posix().encode() + b'\0' + bytes.fromhex(sha(path)))
    return dict(schema=1, root=str(root), source_sha256=digest.hexdigest(),
                recipe_sha256=RECIPES['mi50'], image=IMAGE, flags=recipe('mi50')[2])


def certify_mi50():
    gpu, root = 'mi50', owned(ROOTS['mi50'])
    require(root == PROJECT / 'build-mi50', 'certification requires canonical MI50 root')
    for name in ('stage8-build.json', 'stage8-attempt.json'):
        require(not safe(root / name).exists(), 'prior Stage8 build/attempt exists')
    receipt_path = safe(root / 'stage7-build.json')
    provenance = dict(kind='trusted-stage7-mi50', receipt_sha256=sha(receipt_path),
                      builder_sha256=STAGE7_SHA)
    receipt = read(receipt_path)
    require(receipt['identity'] == legacy_mi50_identity(), 'legacy MI50 source/recipe/image identity changed')
    require(receipt['artifacts'] == artifacts(gpu), 'legacy MI50 artifact hash mismatch')
    cache_sha = validate_cache(gpu)
    install_entry(gpu)
    expected = identity(gpu)
    logdir = owned(root / 'logs') / ('stage8-certify-' + str(time.time_ns()))
    write_new(root / 'stage8-attempt.json', dict(identity=expected, jobs=None, logs=str(logdir),
                                               rx_validation=None, certification=provenance))
    logdir.mkdir()
    try:
        command = ['/usr/bin/python3', '-B', str(RUNTIME / 'stage8.py'), '_inspect', '--gpu', gpu]
        phase(logdir, 'check', docker(command, environment(gpu), mode='check'))
        result = read(logdir / 'check.log')
        require(receipt['identity'] == legacy_mi50_identity()
                and provenance['receipt_sha256'] == sha(receipt_path), 'legacy certification inputs drift')
        require(expected == identity(gpu) and cache_sha == validate_cache(gpu),
                'certification identity/cache drift')
        require(result['artifacts'] == receipt['artifacts'] == artifacts(gpu)
                and result['ldd'] == receipt['ldd'], 'legacy artifacts or resolved ldd map changed')
        require(result['installed'] == tree_manifest(root / 'install/cp2k', links=True),
                'certification installed tree drift')
        write_new(root / 'stage8-build.json', dict(identity=expected, **result, cache_sha256=cache_sha,
                                                  logs=str(logdir), rx_validation=None,
                                                  certification=provenance))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        write_new(logdir / 'certification-failed.json', dict(error=str(error)))
        raise
    print('CERTIFY_OK: trusted MI50 build and new no-GPU loader inspection; no numerical acceptance')
    return 0


def install_entry(gpu):
    root = ROOTS[gpu]
    safe(root / 'bin').mkdir(exist_ok=True)
    path = owned(root / 'bin') / 'cp2k.psmp'
    script = shlex.quote(str(RUNTIME / 'stage8.py'))
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
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(content)
    path.chmod(0o755)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('certify-mi50').set_defaults(gpu='mi50')
    for action in ('check', 'run', 'install-entry', '_check', '_run', '_inspect', 'build'):
        action_parser = sub.add_parser(action)
        action_parser.add_argument('--gpu', choices=ROOTS, required=True)
        if action == 'build':
            action_parser.add_argument('--jobs', type=int, default=16)
            action_parser.add_argument('--resume', action='store_true')
        if action in ('run', '_run'):
            action_parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    require(sys.platform == 'linux', 'runtime requires Linux')
    require(safe(RUNTIME) == PROJECT / 'runtime', 'install helper at the fixed project/runtime path')
    gpu = args.gpu
    owned(PROJECT)
    owned(ROOTS[gpu])
    config(gpu)
    if args.action in ('run', '_run'):
        require(len(args.args) >= 2 and args.args[0] == '--', 'pass CP2K arguments after --')
        args.args = args.args[1:]
    if args.action.startswith('_'):
        require(gpu != 'mi50' or Path('/.dockerenv').is_file(), 'internal MI50 action requires Docker')
        if args.action == '_inspect':
            print(json.dumps(inspect_install(gpu, environment(gpu)), sort_keys=True))
            return 0
        return execute(gpu, getattr(args, 'args', []), args.action == '_check')
    if args.action == 'install-entry':
        return install_entry(gpu)
    if args.action == 'build':
        return build_install(gpu, args.jobs, args.resume)
    if args.action == 'certify-mi50':
        return certify_mi50()
    if gpu != 'mi50':
        return execute(gpu, getattr(args, 'args', []), args.action == 'check')
    installed_state(gpu)
    case = current_case(gpu) if args.action == 'run' else None
    command = ['/usr/bin/python3', '-B', str(RUNTIME / 'stage8.py'), '_' + args.action, '--gpu', gpu]
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
