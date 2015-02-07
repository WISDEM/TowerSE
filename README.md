TowerSE
=======

<!-- Systems Engineering Model of Wind Turbine Towers (OpenMDAO plugin)

To install (for development purposes, the packaged version does all these steps for you):

1) activate OpenMDAO

    source openmdao/bin/activate

2) install CommonSE 
    
    git clone https://github.nrel.gov/sning/CommonSE
    cd CommonSE
    plugin install

3) install pBEAM (requires C compiler and Boost)

    plugin install -f https://github.com/NREL-WISDEM/pBEAM/tarball/master#egg=pBEAM-0.1.0 pBEAM

4) install TowerSE
    
    git clone https://github.nrel.gov/sning/TowerSE
    cd TowerSE
    plugin install
======= -->
TowerSE is a systems engineering model of cylindrical shell wind turbine towers.  The analysis uses beam finite element theory, cylinder drag data, linear wave theory, and shell/global buckling methods from wind turbine standards.  The module is developed as an OpenMDAO assembly.

Author: [S. Andrew Ning](mailto:nrel.wisdem+towerse@gmail.com)

## Version

This software is a beta version 0.1.4.

## Detailed Documentation

For detailed documentation see <http://wisdem.github.io/TowerSE/>

## Prerequisites

General: NumPy, SciPy, Swig, pyWin32, MatlPlotLib, Lxml, OpenMDAO

## Dependencies:

Wind Plant Framework: [FUSED-Wind](http://fusedwind.org) (Framework for Unified Systems Engineering and Design of Wind Plants)

Sub-Models: CommonSE, pBEAM

Supporting python packages: Pandas, Algopy, Zope.interface, Sphinx, Xlrd, PyOpt, py2exe, Pyzmq, Sphinxcontrib-bibtex, Sphinxcontrib-zopeext, Numpydoc, Ipython

## Installation

First, clone the [repository](https://github.com/WISDEM/TowerSE)
or download the releases and uncompress/unpack (TowerSE.py-|release|.tar.gz or TowerSE.py-|release|.zip) from the website link at the bottom the [WISDEM site](http://nwtc.nrel.gov/WISDEM).

Install TowerSE with the following command:

    $ python setup.py install

or if in an activated OpenMDAO environment:

    $ plugin install


## Run Unit Tests

To check if installation was successful try to import the module from within an activated OpenMDAO environment:

    $ python
    > import towerse.tower

You may also run the unit tests.

    $ python src/towerse/test/test_gradients.py

For software issues please use <https://github.com/WISDEM/TowerSE/issues>.  For functionality and theory related questions and comments please use the NWTC forum for [Systems Engineering Software Questions](https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002).
