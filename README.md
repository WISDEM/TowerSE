TowerSE is a systems engineering model of cylindrical shell wind turbine towers.  The analysis uses beam finite element theory, cylinder drag data, linear wave theory, and shell/global buckling methods from wind turbine standards.  The module is developed as an OpenMDAO assembly.

Author: [S. Andrew Ning](mailto:andrew.ning@nrel.gov)

## Prerequisites

Fortran compiler, C compiler, NumPy, SciPy

## Installation

Install TowerSE with the following command.

    $ python setup.py install

or if in an activated OpenMDAO environment

    $ plugin install


## Run Unit Tests

To check if installation was successful try to import the module

    $ python
    > import towerse.tower

You may also run the unit tests.

    $ python src/towerse/test/test_gradients.py

## Detailed Documentation

Online documentation is available at <http://wisdem.github.io/TowerSE/>