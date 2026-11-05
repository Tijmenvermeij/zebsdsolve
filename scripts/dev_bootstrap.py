"""Startup hook installed only in the checkout's dedicated development .venv."""
import json
import os
from pathlib import Path
import sys

from fetch_deps import host_wheel_tag


def manifest_path(root):
    return Path(root) / ".deps" / f"dev-build-{host_wheel_tag()}.json"


def configure(root):
    root = Path(root)
    source = str(root / 'python')
    if source not in sys.path:
        sys.path.insert(0, source)
    manifest = manifest_path(root)
    if not manifest.is_file():
        return
    with manifest.open(encoding='utf-8') as stream:
        current = json.load(stream)
    # A child of an older IDE/kernel inherits these values. Refresh values we
    # selected automatically, while preserving each explicit caller override.
    for name, field in (('ZEBSDSOLVE_LIB', 'library'), ('ZEBSDSOLVE_WGPU', 'wgpu')):
        marker = '_ZEBSDSOLVE_DEV_' + field.upper()
        value = os.environ.get(name)
        if value is None or value == os.environ.get(marker):
            os.environ[name] = current[field]
            os.environ[marker] = current[field]
        else:
            os.environ.pop(marker, None)
