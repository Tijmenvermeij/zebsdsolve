"""Development launcher checks; no compiler, GPU, network or external test packages."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import dev
import dev_bootstrap


class DevelopmentTests(unittest.TestCase):
    def test_debounce_waits_for_quiet_and_fires_once(self):
        changes = dev.Debouncer({'a': 1}, 0.5)
        self.assertFalse(changes.update({'a': 2}, 1.0))
        self.assertFalse(changes.update({'a': 3}, 1.4))
        self.assertFalse(changes.update({'a': 3}, 1.8))
        self.assertTrue(changes.update({'a': 3}, 2.0))
        self.assertFalse(changes.update({'a': 3}, 3.0))
        # A failed build consumes the event; the next edit triggers another build.
        self.assertFalse(changes.update({'a': 4}, 4.0))
        self.assertTrue(changes.update({'a': 4}, 4.6))

    def test_edits_during_build_are_seen(self):
        changes = dev.Debouncer({'a': 1}, 0.5)
        # Snapshot was captured before a long initial build.
        self.assertFalse(changes.update({'a': 2}, 10.0))
        self.assertTrue(changes.update({'a': 2}, 10.6))

    def test_snapshot_detects_add_delete_and_header_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            header = root / 'src/test.h'
            empty = dev.source_snapshot(root)
            header.write_text('a')
            first = dev.source_snapshot(root)
            self.assertNotEqual(empty, first)
            header.write_text('longer')
            self.assertNotEqual(first, dev.source_snapshot(root))
            header.unlink()
            self.assertEqual(empty, dev.source_snapshot(root))
            (root / 'src/ignored.txt').write_text('ignored')
            self.assertEqual(empty, dev.source_snapshot(root))

    def test_platform_paths(self):
        root, gpu = Path('generation'), Path('gpu')
        self.assertEqual(dev.library_paths(root, gpu, 'win32'),
                         (root / 'bin/zebsdsolve.dll', gpu / 'lib/wgpu_native.dll'))
        self.assertEqual(dev.python_executable(root, 'win32'), root / '.venv/Scripts/python.exe')
        self.assertEqual(dev.library_paths(root, gpu, 'darwin')[0], root / 'lib/libzebsdsolve.dylib')
        self.assertEqual(dev.library_paths(root, gpu, 'linux')[0], root / 'lib/libzebsdsolve.so')

    def test_failed_build_preserves_success_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dev_bootstrap.manifest_path(root)
            dev.publish_manifest(manifest, root / 'old.dll', root / 'gpu.dll')
            previous = manifest.read_bytes()
            with patch.object(dev, 'ROOT', root), patch.object(dev, 'MANIFEST', manifest), \
                 patch.object(dev, 'fetch_zig', return_value='zig'), \
                 patch.object(dev, 'fetch_wgpu', return_value=root / 'gpu'), \
                 patch.object(dev, 'run_process', return_value=1):
                self.assertFalse(dev.build())
            self.assertEqual(previous, manifest.read_bytes())

    def test_each_success_uses_unique_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dev_bootstrap.manifest_path(root)
            gpu = root / 'gpu'
            gpu_library = dev.library_paths(root, gpu)[1]
            gpu_library.parent.mkdir(parents=True)
            gpu_library.touch()
            prefixes = []

            def compile_fake(argv, **kwargs):
                prefix = Path(argv[argv.index('--prefix') + 1])
                prefixes.append(prefix)
                library = dev.library_paths(prefix, gpu)[0]
                library.parent.mkdir(parents=True)
                library.touch()
                return 0

            with patch.object(dev, 'ROOT', root), patch.object(dev, 'MANIFEST', manifest), \
                 patch.object(dev, 'fetch_zig', return_value='zig'), \
                 patch.object(dev, 'fetch_wgpu', return_value=gpu), \
                 patch.object(dev, 'run_process', side_effect=compile_fake):
                self.assertTrue(dev.build())
                first = json.loads(manifest.read_text())['library']
                self.assertTrue(dev.build())
                self.assertNotEqual(first, json.loads(manifest.read_text())['library'])
            self.assertNotEqual(*prefixes)
            self.assertTrue(Path(first).exists())  # Loaded generations are retained.
            self.assertFalse(list(manifest.parent.glob('*.tmp')))

    def test_fresh_process_bootstrap_outside_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'python').mkdir()
            (root / 'python/dev_test_source.py').write_text('value = 42\n')
            manifest = dev_bootstrap.manifest_path(root)
            for name in ('first.dll', 'second.dll'):
                dev.publish_manifest(manifest, root / name, root / 'gpu.dll')
                code = ('import sys; sys.path.insert(0, ' + repr(str(Path(dev.__file__).parent)) + '); '
                        'import dev_bootstrap; dev_bootstrap.configure(' + repr(str(root)) + '); '
                        'import os, dev_test_source; print(dev_test_source.value); '
                        'print(os.environ["ZEBSDSOLVE_LIB"])')
                env = os.environ.copy()
                env.pop('ZEBSDSOLVE_LIB', None)
                env.pop('ZEBSDSOLVE_WGPU', None)
                output = subprocess.check_output([sys.executable, '-S', '-c', code], cwd=directory,
                                                 env=env, text=True)
                self.assertIn('42\n', output)
                self.assertIn(name, output)

    def test_child_refreshes_inherited_automatic_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dev_bootstrap.manifest_path(root)
            old_path = sys.path[:]
            try:
                with patch.dict(os.environ, {}, clear=True):
                    dev.publish_manifest(manifest, root / 'first.dll', root / 'first_gpu.dll')
                    dev_bootstrap.configure(root)
                    inherited = os.environ.copy()
                dev.publish_manifest(manifest, root / 'second.dll', root / 'second_gpu.dll')
                code = ('import sys; sys.path.insert(0, ' + repr(str(Path(dev.__file__).parent)) + '); '
                        'import dev_bootstrap; dev_bootstrap.configure(' + repr(str(root)) + '); '
                        'import json, os; print(json.dumps([os.environ["ZEBSDSOLVE_LIB"], '
                        'os.environ["ZEBSDSOLVE_WGPU"]]))')
                # Inherited automatic values refresh independently of explicit overrides.
                for explicit in (None, 'ZEBSDSOLVE_LIB', 'ZEBSDSOLVE_WGPU'):
                    environment = inherited.copy()
                    if explicit:
                        environment[explicit] = str(root / 'explicit.dll')
                    values = json.loads(subprocess.check_output(
                        [sys.executable, '-S', '-c', code], cwd=directory,
                        env=environment, text=True))
                    self.assertEqual(values, [
                        str(root / 'explicit.dll') if explicit == 'ZEBSDSOLVE_LIB' else str((root / 'second.dll').resolve()),
                        str(root / 'explicit.dll') if explicit == 'ZEBSDSOLVE_WGPU' else str((root / 'second_gpu.dll').resolve()),
                    ])
                self.assertTrue(inherited['ZEBSDSOLVE_LIB'].endswith('first.dll'))
            finally:
                sys.path[:] = old_path

    def test_bootstrap_honors_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev.publish_manifest(dev_bootstrap.manifest_path(root), root / 'new.dll', root / 'gpu.dll')
            old_path = sys.path[:]
            try:
                with patch.dict(os.environ, {'ZEBSDSOLVE_LIB': 'explicit.dll'}):
                    dev_bootstrap.configure(root)
                    self.assertEqual(os.environ['ZEBSDSOLVE_LIB'], 'explicit.dll')
            finally:
                sys.path[:] = old_path


if __name__ == '__main__':
    unittest.main()
