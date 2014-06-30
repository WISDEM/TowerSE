Installation
------------

.. admonition:: Prerequisites
   :class: warning

   C compiler, Fortran compiler, NumPy, SciPy

Clone the repository at `<https://github.com/WISDEM/TowerSE>`_
or download the releases and uncomopress/unpack (TowerSE.py-|release|.tar.gz or TowerSE.py-|release|.zip)

Install TowerSE with the following command.

.. code-block:: bash

   $ python setup.py install

To check if installation was successful try to import the module

.. code-block:: bash

    $ python

.. code-block:: python

    > import towerse.tower

or run the unit tests for the gradient checks

.. code-block:: bash

   $ python src/towerse/test/test_tower_gradients.py

An "OK" signifies that all the tests passed.

.. only:: latex

    An HTML version of this documentation that contains further details and links to the source code is available at `<http://wisdem.github.io/TowerSE>`_
