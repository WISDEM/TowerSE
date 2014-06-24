.. _theory:

Theory
------

Theory for the finite element codes are available at their respective websites: `pBEAM <https://github.com/WISDEM/pBEAM>`_ and `Frame3DD <http://frame3dd.sourceforge.net/>`_.  Two different buckling approaches are implemented.  A shell buckling method from Eurocode :cite:`European-Committee-for-Standardisation1993`, and a global buckling method from Germanischer Lloyd :cite:`GL2005`.  The implementation of the Eurocode buckling is modified slightly so as to produce continuously differentiable output.  Hoop stress is estimated using the Eurocode method.  Axial and shear stress calculations are done for cylindrical shell sections and are combined with hoop stress into a von Mises stress.  Fatigue uses supplied damage equivalent moments, which are converted to stress for the given geometry.  Using the stress, and inputs for the number of cycles and slope of the S-N curve allows for a damage calculation.

Computation of drag loads is done assuming drag over a smooth circular cylinder.  The calculation of the resulting drag is separated from the actual velocity distributions, which are handled in the commonse.environment module.  The environment model provides default implementations for power-law wind profiles, logarithmic-law wind profiles, and linear wave theory.  A textbook model is used for soil stiffness properties :cite:`Arya1979`.  The rotor-nacelle-assembly mass properties are transfered to the tower top using the generalized parallel axis theorem.






.. only:: html

    :bib:`Bibliography`

.. bibliography:: references.bib
    :style: unsrt