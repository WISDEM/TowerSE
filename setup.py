#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup

setup(
    name='TowerSE',
    version='0.1',
    description='Tower Systems Engineering Model',
    author='S. Andrew Ning',
    author_email='andrew.ning@nrel.gov',
    install_requires=['commonse', 'pbeam'],
    package_dir={'': 'src'},
    packages=['towerse'],
    license='Apache License, Version 2.0',
    dependency_links=['https://github.com/NREL-WISDEM/pBEAM/tarball/master#egg=pbeam',
        'https://github.nrel.gov/sning/CommonSE/tarball/master#egg=commonse'],
    zip_safe=False
)