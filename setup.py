# Copyright (C) 2021 Matthias Guggenmos
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


DESCRIPTION = (
    'Reverse engineering of Metacognition toolbox'
)

DISTNAME = 'remeta'
MAINTAINER = 'Matthias Guggenmos'
MAINTAINER_EMAIL = 'mg.corresponding@gmail.com'
VERSION = '0.99'
LICENCE = 'MIT License'

INSTALL_REQUIRES = [
    'numpy>=1.20',
    'scipy>=1.3',
    'multiprocessing_on_dill>=3.5.0a4',
    'matplotlib>=3.6.0',
    'numdifftools>=0.9.38']

PACKAGES = ['remeta']

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13'
]

try:
    from setuptools import setup

    _has_setuptools = True
except ImportError:
    from distutils.core import setup

if __name__ == '__main__':

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        license=LICENCE,
        version=VERSION,
        install_requires=INSTALL_REQUIRES,
        include_package_data=True,
        packages=PACKAGES,
        classifiers=CLASSIFIERS,
    )
