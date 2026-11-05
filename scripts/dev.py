#!/usr/bin/env python3
"""Local Python development: setup once, watch engine edits, start fresh Python sessions."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import venv

from fetch_deps import fetch_wgpu, fetch_zig, host_wheel_tag
from dev_bootstrap import manifest_path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = manifest_path(ROOT)


def python_executable(root=ROOT, platform=sys.platform):
    return root / '.venv' / ('Scripts/python.exe' if platform == 'win32' else 'bin/python')


def library_paths(prefix, wgpu, platform=sys.platform):
    if platform == 'win32':
        return prefix / 'bin/zebsdsolve.dll', wgpu / 'lib/wgpu_native.dll'
    suffix = 'dylib' if platform == 'darwin' else 'so'
    return prefix / f'lib/libzebsdsolve.{suffix}', wgpu / f'lib/libwgpu_native.{suffix}'


def run_process(argv, **kwargs):
    """Keep build children in one group, and reap them before leaving on Ctrl-C."""
    options = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
               if sys.platform == 'win32' else {'start_new_session': True})
    process = subprocess.Popen([str(a) for a in argv], **options, **kwargs)
    try:
        return process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            try:
                if sys.platform == 'win32':
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt):
                if sys.platform == 'win32':
                    # taskkill includes Zig's compiler children, not just its driver.
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
        raise


def publish_manifest(path, library, wgpu):
    """Readers see either complete successful generation, never a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='dev-build-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump({'library': str(library.resolve()), 'wgpu': str(wgpu.resolve())}, stream)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build(optimize='ReleaseFast'):
    zig = os.environ.get('ZIG') or fetch_zig()
    wgpu = fetch_wgpu(host_wheel_tag())
    prefix = ROOT / '.deps' / 'dev-builds' / uuid.uuid4().hex
    print(f'Building {prefix.name} ({optimize})', flush=True)
    environment = os.environ.copy()
    environment.setdefault('ZIG_GLOBAL_CACHE_DIR', str(ROOT / '.deps/zig-global-cache'))
    code = run_process([zig, 'build', f'-Doptimize={optimize}', f'-Dwgpu={wgpu}',
                        '--prefix', prefix], cwd=ROOT, env=environment)
    if code:
        print('Build failed; keeping the last successful library.', flush=True)
        return False
    library, gpu_library = library_paths(prefix, wgpu)
    for path in (library, gpu_library):
        if not path.is_file():
            raise FileNotFoundError(f'Build did not produce required library: {path}')
    publish_manifest(MANIFEST, library, gpu_library)
    print(f'Ready: {library}\nStart a new Python session to use this build.', flush=True)
    return True


def source_snapshot(root=ROOT):
    paths = [root / 'build.zig', root / 'build.zig.zon']
    for directory in ('src', 'shaders'):
        paths.extend(p for p in (root / directory).rglob('*')
                     if p.suffix in {'.zig', '.wgsl', '.h', '.c', '.cpp', '.hpp'})
    snapshot = {}
    for path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:  # Editors may rename/delete while we scan.
            continue
        if path.is_file():
            snapshot[str(path.relative_to(root))] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


class Debouncer:
    def __init__(self, snapshot, delay):
        self.snapshot = snapshot
        self.delay = delay
        self.changed_at = None

    def update(self, snapshot, now):
        if snapshot != self.snapshot:
            self.snapshot = snapshot
            self.changed_at = now
        if self.changed_at is not None and now - self.changed_at >= self.delay:
            self.changed_at = None
            return True
        return False


def watch(optimize, interval, debounce):
    # Snapshot before the build so edits made during compilation are rebuilt too.
    changes = Debouncer(source_snapshot(), debounce)
    build(optimize)
    print('Watching src/, shaders/ and build.zig; Ctrl-C stops.', flush=True)
    while True:
        if changes.update(source_snapshot(), time.monotonic()):
            build(optimize)
        time.sleep(interval)


def setup(optimize):
    python = python_executable()
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(ROOT / '.venv')
    # Read the project's declared runtime dependencies without invoking its wheel builder.
    metadata = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    project = metadata.split('[project]', 1)[1].split('\n[', 1)[0]
    match = re.search(r'^dependencies\s*=\s*(\[[\s\S]*?\])', project, re.MULTILINE)
    if match is None:
        raise ValueError('No project.dependencies found in pyproject.toml')
    dependencies = ast.literal_eval(match.group(1))
    if dependencies and run_process([python, '-m', 'pip', 'install', *dependencies], cwd=ROOT):
        raise RuntimeError('Installing Python dependencies failed')
    site_path = subprocess.check_output(
        [str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'],
        cwd=ROOT, text=True).strip()
    # This executes only in this dedicated environment, including from another cwd.
    bootstrap = ('import sys; sys.path.insert(0, ' + repr(str(ROOT / 'scripts')) +
                 '); import dev_bootstrap; dev_bootstrap.configure(' + repr(str(ROOT)) + ')\n')
    (Path(site_path) / 'zebsdsolve-dev.pth').write_text(bootstrap, encoding='utf-8')
    if not build(optimize):
        return 1
    print(f'Python: {python}\nSource: {ROOT / "python"}\n'
          'Keep `python scripts/dev.py watch` running while editing the engine.', flush=True)
    return 0


def positive(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be greater than zero')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in [('setup', 'create .venv, install runtime dependencies, build native'),
                            ('build', 'build and publish a new native library'),
                            ('watch', 'build, then rebuild on engine edits')]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument('--optimize', choices=['Debug', 'ReleaseSafe', 'ReleaseFast', 'ReleaseSmall'],
                             default='ReleaseFast')
        if name == 'watch':
            command.add_argument('--interval', type=positive, default=0.3)
            command.add_argument('--debounce', type=positive, default=0.5)
    command = commands.add_parser('run', help='start fresh .venv Python (script, -c, -m or interactive)')
    command.add_argument('python_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.command == 'setup':
            return setup(args.optimize)
        if args.command == 'build':
            return 0 if build(args.optimize) else 1
        if args.command == 'watch':
            watch(args.optimize, args.interval, args.debounce)
        else:
            python = python_executable()
            if not python.is_file():
                parser.error('run setup first')
            python_args = args.python_args
            if python_args[:1] == ['--']:
                python_args = python_args[1:]
            return run_process([python, *python_args])
    except KeyboardInterrupt:
        print('\nStopped.', flush=True)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    sys.exit(main())
