#! /usr/bin/env python3
"""Build hook for oasys-barc4ro.

All static metadata lives in pyproject.toml. This file exists ONLY to generate
barc4ro/version.py (git revision stamping) at build time, which is imperative
logic and therefore cannot be expressed declaratively in pyproject.toml.

The setuptools.build_meta backend imports and runs this file, so the generation
still happens for `pip install .`, `pip install -e .` and `python -m build`.
"""

import importlib.util
import os
import subprocess

from setuptools import setup

# Keep in sync with [project] version in pyproject.toml
VERSION = '2026.07.15.3'
ISRELEASED = False


def git_version():
    """Return the git revision as a string. Copied from numpy setup.py."""
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except (OSError, subprocess.SubprocessError):
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def _load_existing_version(path='barc4ro/version.py'):
    """Import an already-generated version.py without SourceFileLoader.load_module()."""
    spec = importlib.util.spec_from_file_location('barc4ro.version', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_version_py(filename='barc4ro/version.py'):
    # Copied from numpy setup.py
    cnt = """
# THIS FILE IS GENERATED FROM barc4ro SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
    short_version += ".dev"
"""
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists(filename):
        # must be a source distribution, use existing version file
        GIT_REVISION = _load_existing_version(filename).git_revision
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    with open(filename, 'w') as a:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})


if __name__ == '__main__' or True:
    # Runs both under `python setup.py ...` and when imported by the PEP 517 backend.
    write_version_py()
    setup()