import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).absolute().parents[1]
SPEC = importlib.util.spec_from_file_location('stage8', WORKSPACE / 'runtime/stage8.py')
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)
ORIGINAL_SAFE = runtime.safe
POSIX_PROJECT = PurePosixPath('/home/nowr/cp2k-hip')
POSIX_ROOTS = {gpu: POSIX_PROJECT / name for gpu, name in (
    ('mi50', 'build-mi50'), ('rx', 'build-9070xt'),
    ('rx-validation', 'build-9070xt-stage8-validation'))}


class ProbePath(PurePosixPath):
    symlinks = set()
    resolutions = {}

    def is_symlink(self):
        return str(self) in self.symlinks

    def is_dir(self):
        return True

    def is_file(self):
        return True

    def resolve(self, strict=False):
        return ProbePath(self.resolutions.get(str(self), str(self)))

    def stat(self):
        return types.SimpleNamespace(st_gid=1000)


def local_path(raw):
    path = Path(raw)
    if not path.is_absolute() or '..' in path.parts or not path.resolve().is_relative_to(WORKSPACE):
        raise ValueError('test file access outside workspace')
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError('symlinked test path')
    return path


class RecipeTests(unittest.TestCase):
    def setUp(self):
        for target, value in (
            ('safe', local_path), ('PROJECT', POSIX_PROJECT), ('ROOTS', POSIX_ROOTS),
            ('SPACK', PurePosixPath('/home/nowr/spack/opt/spack'))):
            self.enterContext(patch.object(runtime, target, value))

    def test_rx_recipe_preserves_all_49_features_and_dependencies(self):
        baseline = runtime.cache_values(WORKSPACE / 'CMakeCache-rx.txt')
        for gpu in ('rx', 'rx-validation'):
            flags = runtime.recipe(gpu)[2]
            values = dict(arg[2:].split('=', 1) for arg in flags)
            features = {key: value for key, (kind, value) in baseline.items()
                        if key.startswith('CP2K_') and kind == 'BOOL'}
            self.assertEqual(len(features), 49)
            for key, value in features.items():
                self.assertEqual(values[key + ':BOOL'], value)
            self.assertEqual(values['CP2K_WITH_GPU:STRING'], 'NONE')
            self.assertEqual(values['CP2K_USE_ACCEL:STRING'], 'HIP')
            self.assertEqual(values['CMAKE_PREFIX_PATH:STRING'], baseline['CMAKE_PREFIX_PATH'][1])
            self.assertEqual(values['DBCSR_DIR:STRING'], '/opt/dbcsr-rocm101/lib/cmake/dbcsr')
            self.assertEqual(values['Libint2_DIR:PATH'],
                             str(POSIX_ROOTS[gpu] / 'recipe/local-packages/Libint2'))
            self.assertEqual(values['CMAKE_INSTALL_PREFIX:PATH'], str(POSIX_ROOTS[gpu] / 'install/cp2k'))
            for key in ('CMAKE_HIP_ARCHITECTURES', 'GPU_TARGETS', 'GPU_BUILD_TARGETS', 'AMDGPU_TARGETS'):
                self.assertEqual(values[key + ':STRING'], 'gfx1201')
            self.assertFalse(any(':INTERNAL=' in arg or 'OUTPUT_DIRECTORY' in arg for arg in flags))
            self.assertFalse(any('stage7backup' in arg or '/media/' in arg for arg in flags))
            if gpu == 'rx-validation':
                self.assertNotIn(str(POSIX_ROOTS['rx']) + '/', ';'.join(flags))

    def test_rewrite_only_selected_rx_subtrees(self):
        original = str(POSIX_ROOTS['rx'])
        value = ';'.join(original + '/' + suffix for suffix in
                         ('recipe/include', 'include/file', 'install/cp2k/lib', 'other/lib'))
        result = runtime.rewrite_rx(value, POSIX_ROOTS['rx-validation'])
        self.assertIn(str(POSIX_ROOTS['rx-validation'] / 'recipe/include'), result)
        self.assertIn(original + '/other/lib', result)
        self.assertEqual(runtime.rewrite_rx('/opt/dbcsr-rocm101/lib', POSIX_ROOTS['rx-validation']),
                         '/opt/dbcsr-rocm101/lib')

    def test_mi50_recipe_preserves_frozen_flags(self):
        flags = runtime.recipe('mi50')[2]
        self.assertIn('-DCP2K_WITH_GPU=Mi50', flags)
        self.assertIn('-DCMAKE_HIP_ARCHITECTURES=gfx906', flags)
        self.assertIn('-DCP2K_USE_DLAF=ON', flags)
        self.assertIn('-DCP2K_USE_SIRIUS=ON', flags)
        self.assertIn('-DCP2K_USE_COSMA=ON', flags)

    def test_rx_host_environment_and_validation_isolation(self):
        canonical = runtime.environment('rx', build=True, host={})
        validation = runtime.environment('rx-validation', build=True, host={})
        self.assertEqual(validation['HIPCC_COMPILE_FLAGS_APPEND'], '--offload-arch=gfx1201')
        self.assertEqual(validation['CP2K_DATA_DIR'], '/home/nowr/cp2k-hip/data')
        self.assertEqual(validation['HOME'], canonical['HOME'])
        self.assertNotIn(str(POSIX_ROOTS['rx']) + '/', ':'.join(validation.values()))
        self.assertEqual(runtime.environment('rx-validation', host={})['ROCR_VISIBLE_DEVICES'],
                         runtime.UUIDS['rx'])

    def test_diagnostics_survive_both_mi50_environment_boundaries(self):
        controls = {key: str(index) for index, key in enumerate(runtime.DIAGNOSTICS)}
        controls.update(CP2K_PEER_POOL='1', HSA_OVERRIDE_GFX_VERSION='12.0.0', OPENBLAS_NUM_THREADS='32')
        parent = runtime.environment('mi50', host=controls)
        cfg = runtime.config('mi50')
        with patch.object(runtime, 'safe', ProbePath), patch.object(runtime, 'owned', ProbePath), \
                patch.object(runtime, 'config', return_value=cfg), \
                patch.object(runtime, 'Path', ProbePath), \
                patch.object(runtime.os, 'getuid', return_value=1000, create=True), \
                patch.object(runtime.os, 'getgid', return_value=1000, create=True), \
                patch.object(runtime, 'select_gpu', return_value=ProbePath('/dev/dri/renderD128')):
            argv = runtime.docker(['python3', 'stage8.py', '_run'], parent, mode='run',
                                  case=ProbePath('/home/nowr/research/case'))
        start = argv.index(runtime.IMAGE)
        child_input = dict(item.split('=', 1) for item in argv[start + 2:-3])
        child = runtime.environment('mi50', host=child_input)
        for key in runtime.DIAGNOSTICS:
            self.assertEqual(child[key], controls[key])
        self.assertEqual((child['OMP_NUM_THREADS'], child['OPENBLAS_NUM_THREADS']), ('16', '1'))
        self.assertNotIn('CP2K_PEER_POOL', child)
        self.assertNotIn('HSA_OVERRIDE_GFX_VERSION', child)
        defaults = runtime.environment('mi50', host={})
        self.assertNotIn('CP2K_GPU_HOST_FALLBACK', defaults)
        self.assertNotIn('CP2K_GPU_MEM_LIMIT_MB', defaults)
        self.assertEqual(defaults['CP2K_PW_GPU_RES_RESERVE_MB'], '1024')
        self.assertIn('--pull=never', argv)
        self.assertIn('--network=none', argv)
        for name in ('source', 'install', 'deps'):
            self.assertIn(f'type=bind,source={POSIX_ROOTS["mi50"] / name},'
                          f'target={POSIX_ROOTS["mi50"] / name},readonly', argv)

    def test_no_devices_in_docker_check(self):
        cfg = runtime.config('mi50')
        with patch.object(runtime, 'safe', ProbePath), patch.object(runtime, 'owned', ProbePath), \
                patch.object(runtime, 'config', return_value=cfg), \
                patch.object(runtime.os, 'getuid', return_value=1000, create=True), \
                patch.object(runtime.os, 'getgid', return_value=1000, create=True), \
                patch.object(runtime, 'select_gpu', side_effect=AssertionError('GPU selection')):
            argv = runtime.docker(['python3', 'stage8.py', '_check'], {}, mode='check')
        self.assertNotIn('--device', argv)

    def test_configured_architecture_features_and_paths(self):
        for gpu in POSIX_ROOTS:
            expected = {arg[2:].split('=', 1)[0].split(':')[0]: ('STRING', arg.split('=', 1)[1])
                        for arg in runtime.recipe(gpu)[2]}
            root = POSIX_ROOTS[gpu]
            expected.update(CMAKE_HOME_DIRECTORY=('INTERNAL', str(root / 'source')),
                            CMAKE_CACHEFILE_DIR=('INTERNAL', str(root / 'build/cp2k')))
            with patch.object(runtime, 'safe', ProbePath), \
                    patch.object(runtime, 'cache_values', return_value=expected), \
                    patch.object(runtime, 'recipe', return_value=([], [], [
                        f'-D{key}={value}' for key, (kind, value) in expected.items()
                        if kind != 'INTERNAL'])), patch.object(runtime, 'sha', return_value='cache'):
                self.assertEqual(runtime.validate_cache(gpu), 'cache')
                for key, wrong in (
                    ('CMAKE_HIP_ARCHITECTURES', 'gfx1201' if gpu == 'mi50' else 'gfx906'),
                    ('CP2K_WITH_GPU_PROFILING', 'ON'),
                    ('CMAKE_CACHEFILE_DIR', str(POSIX_ROOTS['rx'] / 'build')),
                    ('CMAKE_RUNTIME_OUTPUT_DIRECTORY', '/media/nowr/old/build'),
                    ('CMAKE_LIBRARY_OUTPUT_DIRECTORY', str(root) + '-stage7backup/build'),
                ):
                    saved = expected.get(key)
                    expected[key] = ('STRING', wrong)
                    with self.subTest(gpu=gpu, key=key), self.assertRaises(ValueError):
                        runtime.validate_cache(gpu)
                    if saved is None:
                        del expected[key]
                    else:
                        expected[key] = saved
                original = expected['CP2K_USE_ELPA']
                expected['CP2K_USE_ELPA'] = ('BOOL', 'OFF')
                with self.assertRaises(ValueError):
                    runtime.validate_cache(gpu)
                expected['CP2K_USE_ELPA'] = original


class FilesystemTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory(prefix='stage8-test-', dir=WORKSPACE))
        self.project = Path(self.temp) / 'project'
        self.roots = {gpu: self.project / root.name for gpu, root in POSIX_ROOTS.items()}
        for target, value in (('safe', local_path), ('PROJECT', self.project), ('ROOTS', self.roots),
                              ('SPACK', Path(self.temp) / 'spack')):
            self.enterContext(patch.object(runtime, target, value))
        self.enterContext(patch.object(runtime.os, 'getuid', return_value=0 if os.name == 'nt' else os.getuid(),
                                       create=True))
        self.enterContext(patch.object(runtime, 'recipe', side_effect=self.fixture_recipe))
        self.enterContext(patch.object(runtime, 'environment', return_value={'PATH': '/usr/bin:/bin'}))
        for root in self.roots.values():
            for name in ('source', 'recipe', 'build', 'install', 'cache', 'container-home', 'logs', 'bin'):
                (root / name).mkdir(parents=True)
            (root / 'source/CMakeLists.txt').write_text('frozen source\n')
            (root / 'source/science.F').write_text('frozen science\n')
            (root / 'recipe/Libint2Config.cmake').write_text('frozen dependency recipe\n')
            (root / 'bin/cp2k.psmp').write_text('stage8 wrapper fixture\n')

    def fixture_recipe(self, gpu='mi50'):
        root = self.roots[gpu]
        arch = 'gfx906' if gpu == 'mi50' else 'gfx1201'
        values = dict(CMAKE_INSTALL_PREFIX=str(root / 'install/cp2k'),
                      CP2K_WITH_GPU_PROFILING='OFF', CP2K_USE_ELPA='ON')
        values.update({key: arch for key in
                       ('CMAKE_HIP_ARCHITECTURES', 'GPU_TARGETS', 'GPU_BUILD_TARGETS', 'AMDGPU_TARGETS')})
        return [], [], [f'-D{key}={value}' for key, value in values.items()]

    def configure_fixture(self, gpu):
        root = self.roots[gpu]
        builddir = root / 'build/cp2k'
        builddir.mkdir(exist_ok=True)
        entries = [arg[2:].replace('=', ':STRING=', 1) for arg in self.fixture_recipe(gpu)[2]]
        entries += [f'CMAKE_HOME_DIRECTORY:INTERNAL={root / "source"}',
                    f'CMAKE_CACHEFILE_DIR:INTERNAL={builddir}']
        (builddir / 'CMakeCache.txt').write_text('\n'.join(entries))

    def install_fixture(self, gpu):
        root = self.roots[gpu] / 'install/cp2k'
        for suffix in ('bin/cp2k.psmp', 'lib/' + runtime.LIB):
            path = root / suffix
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'\x7fELF' + suffix.encode())

    def inspection_fixture(self, gpu):
        return dict(artifacts=runtime.artifacts(gpu),
                    installed=runtime.tree_manifest(self.roots[gpu] / 'install/cp2k', links=True),
                    ldd={runtime.LIB: str(self.roots[gpu] / 'install/cp2k/lib' / runtime.LIB),
                         'libc.so.6': '/usr/lib/libc.so.6'})

    def snapshot(self, gpu):
        self.configure_fixture(gpu)
        self.install_fixture(gpu)
        state = dict(identity=runtime.identity(gpu), cache_sha256=runtime.validate_cache(gpu),
                     **self.inspection_fixture(gpu))
        runtime.write_new(self.roots[gpu] / 'stage8-build.json', state)
        return state

    def test_cache_digest_with_project_directory_values_on_all_platforms(self):
        for gpu, root in self.roots.items():
            with self.subTest(gpu=gpu):
                self.configure_fixture(gpu)
                cache_path = root / 'build/cp2k/CMakeCache.txt'
                expected = hashlib.sha256(cache_path.read_bytes()).hexdigest()
                self.assertEqual(runtime.validate_cache(gpu), expected)
                with patch.object(runtime.re, 'findall', return_value=[str(root / 'build/cp2k')]) as scan:
                    self.assertEqual(runtime.validate_cache(gpu), expected)
                    self.assertGreater(scan.call_count, 0)
                with patch.object(runtime.re, 'findall', return_value=[str(self.project / 'foreign')]), \
                        self.assertRaisesRegex(ValueError, 'foreign project cache path'):
                    runtime.validate_cache(gpu)

    def cache_entries(self, gpu, **entries):
        self.configure_fixture(gpu)
        cache_path = self.roots[gpu] / 'build/cp2k/CMakeCache.txt'
        with cache_path.open('a') as stream:
            stream.write(''.join(f'\n{key}:STRING={value}' for key, value in entries.items()))
        return cache_path

    def cache_scan(self, gpu='mi50'):
        findall = runtime.re.findall
        with patch.object(runtime, 'safe', local_path if os.name == 'nt' else ORIGINAL_SAFE), \
                patch.object(runtime.re, 'findall', side_effect=lambda pattern, value:
                             findall(pattern, value.replace(str(self.project) + '\\',
                                                            str(self.project) + '/'))):
            return runtime.validate_cache(gpu)

    def cache_link(self, path, target, directory=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.symlink_to(target, target_is_directory=directory)
        except OSError as error:
            if os.name == 'nt' and getattr(error, 'winerror', None) == 1314:
                self.skipTest('NOT-RUN: Windows lacks symlink privilege; actual links require Linux rerun')
            raise
        return path

    def test_cache_accepts_internal_soname_chains_without_changing_digest(self):
        self.cache_link(Path(self.temp) / 'link-probe', self.roots['mi50'], directory=True)
        for gpu, root in self.roots.items():
            for location in ('deps/elpa/lib', 'build/cp2k/src', 'build/cp2k/lib',
                             'build/cp2k/lib64', 'install/cp2k/lib', 'install/cp2k/lib64'):
                with self.subTest(gpu=gpu, location=location):
                    library = root / location / 'libelpa_openmp.so.19.0.0'
                    library.parent.mkdir(parents=True, exist_ok=True)
                    library.write_bytes(b'\x7fELFsoname fixture')
                    soname = self.cache_link(library.with_name('libelpa_openmp.so.19'), library.name)
                    alias = self.cache_link(library.with_name('libelpa_openmp.so'), soname)
                    cache_path = self.cache_entries(gpu, ELPA_LIBRARY=alias)
                    original = cache_path.read_bytes()
                    self.assertEqual(self.cache_scan(gpu), hashlib.sha256(original).hexdigest())
                    self.assertEqual(cache_path.read_bytes(), original)
                    self.assertTrue(alias.is_symlink())
                    self.assertEqual(library.read_bytes(), b'\x7fELFsoname fixture')
                    with self.assertRaisesRegex(ValueError, 'symlink'):
                        runtime.safe(alias)

    def test_cache_rejects_escaping_final_library_links_and_returning_chains(self):
        self.cache_link(Path(self.temp) / 'link-probe', self.roots['mi50'], directory=True)
        root = self.roots['mi50']
        alias = root / 'deps/elpa/lib/libelpa_openmp.so'
        destinations = (self.roots['rx'] / 'deps/libescape.so.1',
                        self.project / 'build-mi50-stage7backup/deps/libescape.so.1',
                        Path(self.temp) / 'media/volume/libescape.so.1',
                        Path(self.temp) / 'external/libescape.so.1',
                        root / 'source/libescape.so.1', root / 'recipe/libescape.so.1',
                        root / 'install/cp2k/lib/libescape.so.1')
        for target in destinations:
            with self.subTest(target=target):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'\x7fELFoutside allowed subtree')
                self.cache_link(alias, target)
                self.cache_entries('mi50', ELPA_LIBRARY=alias)
                with self.assertRaisesRegex(ValueError, 'escaping cache library link'):
                    self.cache_scan()
                alias.unlink()
        library = alias.with_name('libelpa_openmp.so.19')
        library.write_bytes(b'\x7fELFinternal')
        outside = self.cache_link(self.roots['rx'] / 'deps/libreturn.so', library)
        self.cache_link(alias, outside)
        with self.assertRaisesRegex(ValueError, 'escaping cache library link'):
            self.cache_scan()

    def test_cache_rejects_library_directory_links_and_symlink_ancestors(self):
        self.cache_link(Path(self.temp) / 'link-probe', self.roots['mi50'], directory=True)
        root = self.roots['mi50']
        directory = root / 'deps/elpa/lib'
        directory.mkdir(parents=True)
        library = directory / 'libelpa_openmp.so.19'
        library.write_bytes(b'\x7fELFinternal')
        linked_directory = self.cache_link(root / 'deps/libalias.so', directory, directory=True)
        final_link = self.cache_link(directory / 'libelpa_openmp.so',
                                     linked_directory / library.name)
        for path in (linked_directory, linked_directory / library.name, final_link):
            with self.subTest(path=path):
                self.cache_entries('mi50', ELPA_LIBRARY=path)
                with self.assertRaises(ValueError):
                    self.cache_scan()
        selected_link = self.cache_link(self.project / 'selected-root', root, directory=True)
        with patch.dict(runtime.ROOTS, mi50=selected_link), self.assertRaisesRegex(ValueError, 'symlink'):
            self.cache_scan()

    def test_cache_rejects_source_recipe_and_nonlibrary_aliases(self):
        self.cache_link(Path(self.temp) / 'link-probe', self.roots['mi50'], directory=True)
        root = self.roots['mi50']
        for location in ('source', 'recipe', 'build/cp2k/CMakeFiles', 'install/cp2k/bin', 'bin'):
            with self.subTest(location=location):
                target = root / location / 'libfixture.so.1'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'\x7fELFfixture')
                alias = self.cache_link(target.with_name('libfixture.so'), target.name)
                self.cache_entries('mi50', EXTRA_LIBRARY=alias)
                with self.assertRaisesRegex(ValueError, 'outside selected subtree'):
                    self.cache_scan()
                if location in ('source', 'recipe'):
                    with self.assertRaisesRegex(ValueError, 'symlink'):
                        runtime.tree_manifest(root / location)
        target = root / 'deps/libfixture.so.1'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'\x7fELFfixture')
        for name in ('config.cmake', 'libbad,name.so'):
            alias = self.cache_link(target.with_name(name), target.name)
            self.cache_entries('mi50', EXTRA_LIBRARY=alias)
            with self.assertRaisesRegex(ValueError, 'shared library name required'):
                self.cache_scan()
        alias = self.cache_link(target.with_name('libfixture.so'), target.name)
        self.cache_entries('mi50', EXTRA_LIBRARY=str(alias.parent / '..' / 'deps' / alias.name))
        with self.assertRaises(ValueError):
            self.cache_scan()

    def test_cache_allows_missing_outputs_but_rejects_output_links(self):
        self.cache_link(Path(self.temp) / 'link-probe', self.roots['mi50'], directory=True)
        root = self.roots['mi50']
        target = root / 'build/cp2k/real-output'
        target.mkdir(parents=True)
        for kind in ('RUNTIME', 'ARCHIVE', 'LIBRARY'):
            for suffix in ('', '_RELEASE'):
                key = f'CMAKE_{kind}_OUTPUT_DIRECTORY{suffix}'
                output = root / 'build/cp2k/generated' / key
                with self.subTest(key=key):
                    cache_path = self.cache_entries('mi50', **{key: output})
                    self.assertFalse(output.exists())
                    self.assertEqual(self.cache_scan(), hashlib.sha256(cache_path.read_bytes()).hexdigest())
                    self.cache_link(output, target, directory=True)
                    with self.assertRaises(ValueError):
                        self.cache_scan()
        library = root / 'build/cp2k/src/libfixture.so.1'
        library.parent.mkdir()
        library.write_bytes(b'\x7fELFfixture')
        alias = self.cache_link(library.with_name('libfixture.so'), library.name)
        self.cache_entries('mi50', CMAKE_LIBRARY_OUTPUT_DIRECTORY=alias)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.cache_scan()

    def test_cache_rejects_dangling_and_cyclic_library_links(self):
        self.cache_link(Path(self.temp) / 'link-probe', self.roots['mi50'], directory=True)
        root = self.roots['mi50']
        alias = self.cache_link(root / 'deps/libfixture.so', 'libfixture.so.1')
        self.cache_entries('mi50', EXTRA_LIBRARY=alias)
        with self.assertRaisesRegex(ValueError, 'regular cache library required'):
            self.cache_scan()
        self.cache_link(alias.with_name('libfixture.so.1'), alias.name)
        with self.assertRaisesRegex(ValueError, 'cyclic cache library link'):
            self.cache_scan()

    def test_cache_requires_owned_library_ancestors(self):
        self.cache_link(Path(self.temp) / 'link-probe', self.roots['mi50'], directory=True)
        root = self.roots['mi50']
        library = root / 'deps/elpa/lib/libfixture.so.1'
        library.parent.mkdir(parents=True)
        library.write_bytes(b'\x7fELFfixture')
        alias = self.cache_link(library.with_name('libfixture.so'), library.name)
        cache_path = self.cache_entries('mi50', EXTRA_LIBRARY=alias)
        self.assertEqual(self.cache_scan(), hashlib.sha256(cache_path.read_bytes()).hexdigest())
        wrong_uid = os.getuid() + 1
        with patch.object(runtime.os, 'getuid', return_value=wrong_uid), \
                self.assertRaisesRegex(ValueError, 'owned directory required'):
            self.cache_scan()

    def test_cache_soname_policy_with_mocked_link_metadata(self):
        root = self.roots['mi50']
        library = root / 'deps/elpa/lib/libfixture.so.1'
        library.parent.mkdir(parents=True)
        library.write_bytes(b'\x7fELFfixture')
        alias = library.with_name('libfixture.so')
        cache_path = self.cache_entries('mi50', EXTRA_LIBRARY=alias)
        original = cache_path.read_bytes()
        links = {alias: library.name}
        is_symlink, readlink = Path.is_symlink, os.readlink
        with patch.object(Path, 'is_symlink', autospec=True, side_effect=lambda path:
                          path in links or is_symlink(path)), \
                patch.object(runtime.os, 'readlink', side_effect=lambda path, **kwargs:
                             links[path] if path in links else readlink(path, **kwargs)):
            self.assertEqual(self.cache_scan(), hashlib.sha256(original).hexdigest())
            self.assertEqual(cache_path.read_bytes(), original)
            links[alias] = str(self.roots['rx'] / 'deps/libfixture.so.1')
            with self.assertRaisesRegex(ValueError, 'escaping cache library link'):
                self.cache_scan()
            links[alias] = library.name
            links[library.parent] = str(root / 'deps/elsewhere')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                self.cache_scan()

    def certification_fixture(self):
        root = self.roots['mi50']
        self.configure_fixture('mi50')
        self.install_fixture('mi50')
        (root / 'bin/cp2k.psmp').unlink()
        source_digest = hashlib.sha256(b''.join(
            name.encode() + b'\0' + hashlib.sha256((root / 'source' / name).read_bytes()).digest()
            for name in ('CMakeLists.txt', 'science.F'))).hexdigest()
        inspection = self.inspection_fixture('mi50')
        legacy_logs = root / 'logs/stage7-fixture'
        legacy_logs.mkdir()
        receipt = dict(identity=dict(schema=1, root=str(root), source_sha256=source_digest,
                                     recipe_sha256=runtime.RECIPES['mi50'], image=runtime.IMAGE,
                                     flags=self.fixture_recipe('mi50')[2]),
                       artifacts=inspection['artifacts'], ldd=inspection['ldd'], logs=str(legacy_logs))
        runtime.write_new(legacy_logs / 'check.log', {key: inspection[key] for key in ('artifacts', 'ldd')})
        runtime.write_new(root / 'stage7-build.json', receipt)
        return receipt

    def certification_process(self, result, returncode=0, mutate=None):
        def run(command, **kwargs):
            self.assertEqual(command, ['mock-docker-inspection'])
            if mutate:
                mutate()
            kwargs['stdout'].write(json.dumps(result).encode())
            return subprocess.CompletedProcess(command, returncode)
        return run

    def test_certify_mi50_inspects_without_compilation_and_supports_checked(self):
        receipt = self.certification_fixture()
        root = self.roots['mi50']
        receipt_sha = runtime.sha(root / 'stage7-build.json')
        old_log_sha = runtime.sha(root / 'logs/stage7-fixture/check.log')
        result = self.inspection_fixture('mi50')
        with patch.object(runtime, 'docker', return_value=['mock-docker-inspection']) as docker, \
                patch.object(runtime.subprocess, 'run',
                             side_effect=self.certification_process(result)) as run, \
                patch.object(runtime, 'select_gpu', side_effect=AssertionError('GPU selected')):
            self.assertEqual(runtime.certify_mi50(), 0)
        docker.assert_called_once_with(
            ['/usr/bin/python3', '-B', str(runtime.RUNTIME / 'stage8.py'), '_inspect', '--gpu', 'mi50'],
            runtime.environment('mi50'), mode='check')
        run.assert_called_once()
        state = runtime.installed_state('mi50')
        self.assertEqual(state['certification'], dict(kind='trusted-stage7-mi50',
                         receipt_sha256=receipt_sha, builder_sha256=runtime.STAGE7_SHA))
        self.assertEqual(state['artifacts'], receipt['artifacts'])
        self.assertEqual(state['ldd'], receipt['ldd'])
        self.assertEqual(state['installed'], result['installed'])
        self.assertIsNone(state['rx_validation'])
        self.assertEqual(runtime.sha(root / 'stage7-build.json'), receipt_sha)
        self.assertEqual(runtime.sha(root / 'logs/stage7-fixture/check.log'), old_log_sha)
        self.assertIn('check --gpu mi50', (root / 'bin/cp2k.psmp').read_text())
        logdir = Path(state['logs'])
        self.assertEqual({path.name for path in logdir.iterdir()}, {'check.json', 'check.log'})
        self.assertEqual(runtime.read(logdir / 'check.json')['exit'], 0)
        with patch.object(runtime, 'inspect_install', return_value=result):
            self.assertEqual(runtime.checked('mi50', {}), result)

    def test_certify_rejects_stale_source_artifact_cache_and_receipt(self):
        receipt = self.certification_fixture()
        root = self.roots['mi50']
        changes = [(root / name, (root / name).read_bytes() + b'\nchanged')
                   for name in ('source/science.F', 'install/cp2k/bin/cp2k.psmp')]
        cache = root / 'build/cp2k/CMakeCache.txt'
        changes.append((cache, cache.read_bytes().replace(b'gfx906', b'gfx1201')))
        for key, value in (('root', str(root) + '-stage7backup'), ('source_sha256', 'stale'),
                           ('flags', []), ('image', 'wrong image'), ('recipe_sha256', 'stale')):
            changed = dict(receipt, identity=dict(receipt['identity'], **{key: value}))
            changes.append((root / 'stage7-build.json', json.dumps(changed).encode()))
        with patch.object(runtime.subprocess, 'run') as run:
            for path, changed in changes:
                with self.subTest(path=path, changed=changed):
                    original = path.read_bytes()
                    try:
                        path.write_bytes(changed)
                        with self.assertRaises(ValueError):
                            runtime.certify_mi50()
                        self.assertFalse((root / 'stage8-build.json').exists())
                        self.assertFalse((root / 'stage8-attempt.json').exists())
                        self.assertFalse((root / 'bin/cp2k.psmp').exists())
                    finally:
                        path.write_bytes(original)
            run.assert_not_called()

    def test_certify_requires_frozen_original_runtime_and_both_recipes(self):
        self.certification_fixture()
        original_sha = runtime.sha
        for name in ('stage7.py', 'dependencies-mi50.json', 'dependencies-rx.json'):
            changed_path = runtime.RUNTIME / name
            with self.subTest(name=name), \
                    patch.object(runtime, 'sha', side_effect=lambda path:
                                 'stale' if Path(path) == changed_path else original_sha(path)), \
                    patch.object(runtime.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'changed'):
                    runtime.certify_mi50()
                run.assert_not_called()
                self.assertFalse((self.roots['mi50'] / 'stage8-build.json').exists())

    def failed_certification(self, result, returncode=0, mutate=None):
        root = self.roots['mi50']
        with patch.object(runtime, 'docker', return_value=['mock-docker-inspection']), \
                patch.object(runtime.subprocess, 'run',
                             side_effect=self.certification_process(result, returncode, mutate)):
            with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                runtime.certify_mi50()
        self.assertFalse((root / 'stage8-build.json').exists())
        logdir = Path(runtime.read(root / 'stage8-attempt.json')['logs'])
        self.assertEqual(runtime.read(logdir / 'check.json')['exit'], returncode)
        self.assertTrue((logdir / 'check.log').is_file())
        self.assertTrue((logdir / 'certification-failed.json').is_file())
        with self.assertRaisesRegex(ValueError, 'prior Stage8'):
            runtime.certify_mi50()

    def test_certify_rejects_resolved_ldd_drift_and_preserves_logs(self):
        self.certification_fixture()
        result = self.inspection_fixture('mi50')
        result['ldd']['libc.so.6'] = '/different/libc.so.6'
        self.failed_certification(result)

    def test_certify_rejects_inspected_artifact_drift(self):
        self.certification_fixture()
        result = self.inspection_fixture('mi50')
        result['artifacts'][str(self.roots['mi50'] / 'install/cp2k/bin/cp2k.psmp')] = 'stale'
        self.failed_certification(result)

    def test_certify_rechecks_receipt_after_inspection(self):
        self.certification_fixture()
        path = self.roots['mi50'] / 'stage7-build.json'
        original = path.read_bytes()
        self.failed_certification(self.inspection_fixture('mi50'),
                                  mutate=lambda: path.write_bytes(original + b'\n'))

    def test_certify_rechecks_source_after_inspection(self):
        self.certification_fixture()
        path = self.roots['mi50'] / 'source/science.F'
        self.failed_certification(self.inspection_fixture('mi50'),
                                  mutate=lambda: path.write_text('changed during inspection'))

    def test_certify_failed_inspection_preserves_logs(self):
        self.certification_fixture()
        self.failed_certification(self.inspection_fixture('mi50'), returncode=7)

    def test_certify_rejects_existing_stage8_manifest_or_attempt(self):
        self.certification_fixture()
        root = self.roots['mi50']
        for name in ('stage8-build.json', 'stage8-attempt.json'):
            path = root / name
            runtime.write_new(path, {'existing': True})
            with self.subTest(name=name), patch.object(runtime.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'prior Stage8'):
                    runtime.certify_mi50()
                run.assert_not_called()
                self.assertEqual(runtime.read(path), {'existing': True})
            path.unlink()

    def test_certify_cli_rejects_rx_arbitrary_roots_and_resume(self):
        for args in (['--gpu', 'rx'], ['--root', '/tmp/build'], ['--resume']):
            with self.subTest(args=args), patch.object(sys, 'stderr', io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                runtime.main(['certify-mi50', *args])
            self.assertEqual(error.exception.code, 2)

    def test_artifact_and_ldd_mismatch_rejected_on_both_gpus(self):
        for gpu in ('mi50', 'rx', 'rx-validation'):
            state = self.snapshot(gpu)
            with patch.object(runtime, 'inspect_install', return_value=self.inspection_fixture(gpu)):
                runtime.checked(gpu, {})
            changed = dict(self.inspection_fixture(gpu), ldd={runtime.LIB: '/foreign/libcp2k.so'})
            with patch.object(runtime, 'inspect_install', return_value=changed), self.assertRaises(ValueError):
                runtime.checked(gpu, {})
            binary = self.roots[gpu] / 'install/cp2k/bin/cp2k.psmp'
            binary.write_bytes(b'\x7fELFchanged')
            with self.assertRaisesRegex(ValueError, 'artifact hash'):
                runtime.installed_state(gpu)
            self.assertNotEqual(runtime.artifacts(gpu), state['artifacts'])

    def test_source_wrapper_local_recipe_cache_and_installed_tree_drift(self):
        self.snapshot('rx')
        root = self.roots['rx']
        for name in ('source/science.F', 'bin/cp2k.psmp', 'recipe/Libint2Config.cmake',
                     'build/cp2k/CMakeCache.txt'):
            path = root / name
            original = path.read_bytes()
            path.write_bytes(original + b'\nchanged\n')
            with self.subTest(name=name), self.assertRaises(ValueError):
                runtime.installed_state('rx')
            path.write_bytes(original)
        (root / 'install/cp2k/extra.txt').write_text('unrecorded')
        with self.assertRaisesRegex(ValueError, 'installed tree'):
            runtime.installed_state('rx')

    def test_rx_never_accepts_stage7_dependency_artifacts(self):
        self.configure_fixture('rx')
        self.install_fixture('rx')
        with self.assertRaises(FileNotFoundError):
            runtime.installed_state('rx')

    def test_plain_source_manifest_records_every_file_without_git(self):
        identity = runtime.identity('rx-validation')
        self.assertEqual(set(identity['source']), {'CMakeLists.txt', 'science.F'})
        self.assertEqual(identity['source']['science.F']['sha256'],
                         runtime.sha(self.roots['rx-validation'] / 'source/science.F'))
        self.assertEqual(identity['runtime_sha256'], runtime.sha(WORKSPACE / 'runtime/stage8.py'))
        self.assertIn('wrapper_sha256', identity)

    def test_external_calculations_allowed_protected_locations_rejected(self):
        case = Path(self.temp) / 'research/job'
        case.mkdir(parents=True)
        self.assertEqual(runtime.case_directory('rx', case), case)
        backup = self.project / 'build-9070xt-stage7backup/job'
        backup.mkdir(parents=True)
        for path in (self.project, self.roots['rx'] / 'source', self.roots['rx'] / 'install', backup):
            with self.subTest(path=path), self.assertRaises(ValueError):
                runtime.case_directory('rx', path)
        validation = self.roots['rx-validation'] / 'validation/case'
        validation.mkdir(parents=True)
        self.assertEqual(runtime.case_directory('rx-validation', validation), validation)
        with self.assertRaises(ValueError):
            runtime.case_directory('rx', validation)

    def test_check_never_executes_cp2k(self):
        with patch.object(runtime, 'checked', return_value={}) as checked, \
                patch.object(runtime.os, 'execvpe', side_effect=AssertionError('CP2K invoked')):
            self.assertEqual(runtime.execute('rx', [], check=True), 0)
            checked.assert_called_once()

    def test_inspect_only_calls_ldd_with_hidden_devices(self):
        self.install_fixture('rx')
        process = types.SimpleNamespace(stdout='mock ldd', stderr='')
        with patch.object(runtime.subprocess, 'run', return_value=process) as run, \
                patch.object(runtime, 'parse_ldd', return_value={runtime.LIB: '/mock'}):
            runtime.inspect_install('rx', {'ROCR_VISIBLE_DEVICES': runtime.UUIDS['rx']})
        self.assertEqual(run.call_args.args[0][0], '/usr/bin/ldd')
        self.assertEqual(run.call_args.kwargs['env']['ROCR_VISIBLE_DEVICES'], '')
        self.assertEqual(run.call_args.kwargs['env']['HIP_VISIBLE_DEVICES'], '')

    def test_non_elf_rejected_before_ldd(self):
        self.install_fixture('mi50')
        (self.roots['mi50'] / 'install/cp2k/lib' / runtime.LIB).write_text('shell script')
        with patch.object(runtime.subprocess, 'run') as run, self.assertRaisesRegex(ValueError, 'ELF'):
            runtime.inspect_install('mi50', {})
        run.assert_not_called()

    def test_differing_wrapper_is_not_overwritten(self):
        path = self.roots['rx'] / 'bin/cp2k.psmp'
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'archive'):
            runtime.install_entry('rx')
        self.assertEqual(path.read_bytes(), original)

    def test_new_wrapper_targets_only_selected_stage8_root(self):
        path = self.roots['rx-validation'] / 'bin/cp2k.psmp'
        path.unlink()
        runtime.install_entry('rx-validation')
        content = path.read_text()
        self.assertIn('stage8.py', content)
        self.assertIn('check --gpu rx-validation', content)
        self.assertIn('run --gpu rx-validation -- "$@"', content)
        self.assertIn('--check', content)

    def fake_phase(self, failure=None):
        def phase(logdir, name, command, *args, **kwargs):
            runtime.write_new(logdir / (name + '.json'), dict(command=command))
            if name == failure:
                raise subprocess.CalledProcessError(7, command)
            if name == 'configure':
                self.configure_fixture('rx-validation')
            if name == 'install':
                self.install_fixture('rx-validation')
            if name == 'check':
                runtime.write_new(logdir / 'check.log', self.inspection_fixture('rx-validation'))
        return phase

    def test_failed_build_and_exact_explicit_resume_on_host(self):
        root = self.roots['rx-validation']
        with patch.object(runtime, 'install_entry', return_value=0), \
                patch.object(runtime, 'phase', side_effect=self.fake_phase('build')):
            with self.assertRaises(subprocess.CalledProcessError):
                runtime.build_install('rx-validation', 16)
        self.assertTrue((root / 'stage8-attempt.json').is_file())
        self.assertFalse((root / 'stage8-build.json').exists())
        with self.assertRaisesRegex(ValueError, 'prior attempt'):
            runtime.build_install('rx-validation', 16)
        with patch.object(runtime, 'install_entry', return_value=0), \
                patch.object(runtime, 'phase', side_effect=self.fake_phase()) as phase, \
                patch.object(runtime, 'docker', side_effect=AssertionError('RX entered Docker')):
            with self.assertRaisesRegex(ValueError, 'resume identity/jobs'):
                runtime.build_install('rx-validation', 8, resume=True)
            self.assertEqual(runtime.build_install('rx-validation', 16, resume=True), 0)
            self.assertEqual([call.args[1] for call in phase.call_args_list], ['build', 'install', 'check'])
        runtime.installed_state('rx-validation')
        self.assertFalse((self.roots['rx'] / 'stage8-build.json').exists())

    def test_canonical_rx_requires_inspection_bound_validation_evidence(self):
        self.snapshot('rx-validation')
        root = self.roots['rx-validation']
        evidence = root / 'validation/gfx1201.log'
        evidence.parent.mkdir()
        evidence.write_text('coordinator validation fixture')
        approval = dict(status='PASS', build_manifest_sha256=runtime.sha(root / 'stage8-build.json'),
                        evidence={'validation/gfx1201.log': runtime.sha(evidence)})
        runtime.write_new(root / 'stage8-validation.json', approval)
        source_sha = runtime.identity('rx')['source_sha256']
        with patch.object(runtime, 'inspect_install', return_value=self.inspection_fixture('rx-validation')):
            self.assertIn('approval_sha256', runtime.validation_gate(source_sha))
            evidence.write_text('changed evidence')
            with self.assertRaisesRegex(ValueError, 'evidence mismatch'):
                runtime.validation_gate(source_sha)
            with self.assertRaisesRegex(ValueError, 'stale'):
                runtime.validation_gate('wrong source')


class PathAndLoaderTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(runtime, 'Path', ProbePath))
        self.enterContext(patch.object(runtime, 'PROJECT', POSIX_PROJECT))
        self.enterContext(patch.object(runtime, 'ROOTS', POSIX_ROOTS))
        ProbePath.symlinks = set()
        ProbePath.resolutions = {}

    def test_absolute_paths_and_symlink_ancestors(self):
        self.assertEqual(ORIGINAL_SAFE('/home/nowr/research/job'), ProbePath('/home/nowr/research/job'))
        for path in ('relative/job', '/home/nowr/../escape', '/home/nowr/bad,path', '/tmp/a\nb'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                ORIGINAL_SAFE(path)
        ProbePath.symlinks.add('/home/nowr/cp2k-hip/build-9070xt-stage8-validation')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            ORIGINAL_SAFE('/home/nowr/cp2k-hip/build-9070xt-stage8-validation/build')

    def test_ldd_resolution_rejects_foreign_roots_missing_and_duplicates(self):
        for gpu in POSIX_ROOTS:
            library = str(POSIX_ROOTS[gpu] / 'install/cp2k/lib' / runtime.LIB)
            line = f'{runtime.LIB} => {library} (0x1234)\n'
            self.assertEqual(runtime.parse_ldd(line, gpu)[runtime.LIB], library)
            for extra in ('libbad.so => not found\n', line,
                          'libforeign.so => /media/nowr/old/lib.so\n',
                          'libforeign.so => /home/nowr/cp2k-hip/build-mi50-stage7backup/lib.so\n'):
                with self.subTest(gpu=gpu, extra=extra), self.assertRaises(ValueError):
                    runtime.parse_ldd(line + extra, gpu)
            if gpu == 'rx-validation':
                foreign = str(POSIX_ROOTS['rx'] / 'install/cp2k/lib' / runtime.LIB)
                with self.assertRaises(ValueError):
                    runtime.parse_ldd(f'{runtime.LIB} => {foreign}', gpu)
            ProbePath.resolutions[library] = '/media/nowr/escaped/lib.so'
            with self.assertRaises(ValueError):
                runtime.parse_ldd(line, gpu)
            ProbePath.resolutions.clear()


if __name__ == '__main__':
    unittest.main(verbosity=2)
