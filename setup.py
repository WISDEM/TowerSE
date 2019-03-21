#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup

setup(
    name='TowerSE',
    version='0.1.1',
    description='Tower Systems Engineering Model',
    author='S. Andrew Ning',
    author_email='andrew.ning@nrel.gov',
    install_requires=['openmdao>=1.5', 'commonse', 'pyframe3dd'],
    package_dir={'': 'src'},
    packages=['towerse'],
    license='Apache License, Version 2.0',
    #dependency_links=['https://github.com/WISDEM/pBEAM/tarball/master#egg=pbeam',
        #'https://github.com/WISDEM/CommonSE/tarball/master#egg=commonse'],
        #Need to fix these
    zip_safe=False
)
