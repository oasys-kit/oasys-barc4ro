#! /usr/bin/env python3

from importlib.machinery import SourceFileLoader
import os
import subprocess

#
# memorandum (for documentation with numpydoc style)
#
# ** install sphinx:
# pip install sphinx
# pip install sphinxcontrib-apidoc
# pip install sphinx-rtd-theme
# pip install nbsphinx

# ** some initialization
# sphinx-quickstart  # needed only once...
# ** iterate
# rm -rf docs/barc4ro*.rst docs/modules.rst ./docs/generated/
# sphinx-apidoc -o docs barc4ro
# make clean html # results are in ./_build/html/index.html

try:
    from setuptools import find_packages, setup
except AttributeError:
    from setuptools import find_packages, setup

NAME = 'oasys-barc4ro'

VERSION = '2025.09.09'
ISRELEASED = False

DESCRIPTION = 'X-ray Refractive Optics Library for Physical Optics'
README_FILE = os.path.join(os.path.dirname(__file__), 'README.rst')
LONG_DESCRIPTION = open(README_FILE).read()
AUTHOR = 'Rafael Celestre, Luca Rebuffi'
AUTHOR_EMAIL = 'rafael.celestre@esrf.fr'
URL = 'https://github.com/oasys-kit/oasys-barc4ro'
DOWNLOAD_URL = 'https://github.com/oasys-kit/oasys-barc4ro'
MAINTAINER = 'Luca Rebuffi'
MAINTAINER_EMAIL = 'lrebuffi@anl.gov'
LICENSE = 'GPLv3'

KEYWORDS = (
    'dictionary',
    'glossary',
    'synchrotron'
    'simulation',
)

CLASSIFIERS = (
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Environment :: Plugins',
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: '
    'GNU General Public License v3 or later (GPLv3+)',
    'Operating System :: POSIX',
    'Operating System :: Microsoft :: Windows',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
)

INSTALL_REQUIRES = (
    'scipy',
)

SETUP_REQUIRES = (
    'setuptools',
)


# Return the git revision as a string
def git_version():
    """Return the git revision as a string.

    Copied from numpy setup.py
    """
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
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


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
    elif os.path.exists('barc4ro/version.py'):
        # must be a source distribution, use existing version file
        version = SourceFileLoader('barc4ro.version', 'barc4ro/version.py').load_module()
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


PACKAGES = find_packages(exclude=('*.tests', '*.tests.*', 'tests.*', 'tests'))

PACKAGE_DATA = {
}


def setup_package():
    write_version_py()
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        url=URL,
        download_url=DOWNLOAD_URL,
        license=LICENSE,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        # extra setuptools args
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        setup_requires=SETUP_REQUIRES,
    )

if __name__ == '__main__':
    setup_package()
