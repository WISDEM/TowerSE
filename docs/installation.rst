Installation
------------

.. admonition:: Prerequisites
   :class: warning

   C compiler, Fortran compiler, NumPy, SciPy

Download either TowerSE.py-|release|.tar.gz or TowerSE.py-|release|.zip and uncompress/unpack it.

Install TowerSE with the following command.

.. code-block:: bash

   $ python setup.py install

To check if installation was successful run the unit tests for the NREL 5-MW model

.. code-block:: bash

   $ python src/test/test_ccblade.py

An "OK" signifies that all the tests passed.

.. only:: latex

    To access an HTML version of this documentation that contains further details and links to the source code, open <link>_.
