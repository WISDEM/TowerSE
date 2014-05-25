#!/usr/bin/env python
# encoding: utf-8
"""
test_tower_gradients.py

Created by Andrew Ning on 2013-12-20.
Copyright (c) NREL. All rights reserved.
"""

import unittest
import numpy as np
from commonse.utilities import check_gradient, check_gradient_unit_test
from towerse.tower import TowerWindDrag, TowerWaveDrag, TowerDiscretization, RNAMass, RotorLoads, GeometricConstraints


class TestTowerWindDrag(unittest.TestCase):

    def test1(self):

        twd = TowerWindDrag()
        twd.U = [0., 8.80496275, 10.11424623, 10.96861453, 11.61821801, 12.14846828, 12.59962946, 12.99412772, 13.34582791, 13.66394248, 13.95492553, 14.22348635, 14.47317364, 14.70673252, 14.92633314, 15.13372281, 15.33033057, 15.51734112, 15.69574825, 15.86639432, 16.03]
        twd.z = [0., 4.38, 8.76, 13.14, 17.52, 21.9, 26.28, 30.66, 35.04, 39.42, 43.8, 48.18, 52.56, 56.94, 61.32, 65.7, 70.08, 74.46, 78.84, 83.22, 87.6]
        twd.d = [6., 5.8935, 5.787, 5.6805, 5.574, 5.4675, 5.361, 5.2545, 5.148, 5.0415, 4.935, 4.8285, 4.722, 4.6155, 4.509, 4.4025, 4.296, 4.1895, 4.083, 3.9765, 3.87]
        twd.beta = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        twd.rho = 1.225
        twd.mu = 1.7934e-05

        names, errors = check_gradient(twd)

        for name, err in zip(names, errors):

            if name == 'd_windLoads.Px[0] / d_U[0]':
                tol = 2e-5  # central difference not accurate right at Re=0
            else:
                tol = 1e-6

            try:
                self.assertLessEqual(err, tol)
            except AssertionError, e:
                print '*** error in:', name
                raise e



class TestTowerWaveDrag(unittest.TestCase):

    def test1(self):

        twd = TowerWaveDrag()
        twd.U = [0., 8.80496275, 10.11424623, 10.96861453, 11.61821801, 12.14846828, 12.59962946, 12.99412772, 13.34582791, 13.66394248, 13.95492553, 14.22348635, 14.47317364, 14.70673252, 14.92633314, 15.13372281, 15.33033057, 15.51734112, 15.69574825, 15.86639432, 16.03]
        twd.z = [0., 4.38, 8.76, 13.14, 17.52, 21.9, 26.28, 30.66, 35.04, 39.42, 43.8, 48.18, 52.56, 56.94, 61.32, 65.7, 70.08, 74.46, 78.84, 83.22, 87.6]
        twd.d = [6., 5.8935, 5.787, 5.6805, 5.574, 5.4675, 5.361, 5.2545, 5.148, 5.0415, 4.935, 4.8285, 4.722, 4.6155, 4.509, 4.4025, 4.296, 4.1895, 4.083, 3.9765, 3.87]
        twd.beta = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        twd.rho = 1.225
        twd.mu = 1.7934e-05

        twd.A = 1.1*twd.U
        twd.cm = 2.0

        names, errors = check_gradient(twd)

        for name, err in zip(names, errors):

            if name == 'd_waveLoads.Px[0] / d_U[0]':
                tol = 2e-5  # central difference not accurate right at Re=0
            else:
                tol = 1e-6

            try:
                self.assertLessEqual(err, tol)
            except AssertionError, e:
                print '*** error in:', name
                raise e



class TestTowerDiscretization(unittest.TestCase):

    def test1(self):

        td = TowerDiscretization()
        td.towerHeight = 87.6
        td.z = np.array([0.0, 0.5, 1.0])
        td.d = np.array([6.0, 4.935, 3.87])
        td.t = np.array([0.0351, 0.0299, 0.0247])
        td.n = np.array([10, 7])
        td.n_reinforced = 3

        check_gradient_unit_test(self, td)


    def test2(self):

        td = TowerDiscretization()
        td.towerHeight = np.random.rand(1)[0]
        td.z = np.array([0.0, 0.5, 1.0])
        td.d = np.random.rand(3)
        td.t = np.random.rand(3)
        td.n = np.array([10, 7])
        td.n_reinforced = 3

        check_gradient_unit_test(self, td)



class TestRNAMass(unittest.TestCase):

    def test1(self):

        rna = RNAMass()
        rna.blades_mass = 15241.323 * 3
        rna.hub_mass = 50421.4
        rna.nac_mass = 221245.8
        rna.hub_cm = [-6.3, 0., 3.15]
        rna.nac_cm = [-0.32, 0., 2.4]
        rna.blades_I = [26375976., 13187988., 13187988., 0., 0., 0.]
        rna.hub_I = [127297.8, 127297.8, 127297.8, 0., 0., 0.]
        rna.nac_I = [9908302.58, 912488.28, 1160903.54, 0., 0., 0.]

        check_gradient_unit_test(self, rna, tol=1e-5)


    def test2(self):

        rna = RNAMass()
        rna.blades_mass = np.random.rand(1)[0]
        rna.hub_mass = np.random.rand(1)[0]
        rna.nac_mass = np.random.rand(1)[0]
        rna.hub_cm = np.random.rand(3)
        rna.nac_cm = np.random.rand(3)
        rna.blades_I = np.random.rand(6)
        rna.hub_I = np.random.rand(6)
        rna.nac_I = np.random.rand(6)

        check_gradient_unit_test(self, rna)



class TestRotorLoads(unittest.TestCase):

    def test1(self):

        loads = RotorLoads()
        loads.T = 123.0
        loads.Q = 4843.0
        loads.r_hub = [2.0, -3.2, 4.5]
        loads.rna_cm = [-3.0, 1.6, -4.0]
        loads.m_RNA = 200.0
        loads.tilt = 13.2
        loads.g = 9.81

        check_gradient_unit_test(self, loads)


    def test2(self):

        loads = RotorLoads()
        loads.T = 123.0
        loads.Q = 4843.0
        loads.r_hub = [2.0, -3.2, 4.5]
        loads.rna_cm = [-3.0, 1.6, -4.0]
        loads.m_RNA = 200.0
        loads.tilt = 13.2
        loads.g = 9.81
        loads.downwind = True

        check_gradient_unit_test(self, loads)




class TestGeometricConstraints(unittest.TestCase):

    def test1(self):

        gc = GeometricConstraints()
        gc.d = [4.0, 3.0, 2.0]
        gc.t = [0.4, 0.23, 0.14]


        check_gradient_unit_test(self, gc)



if __name__ == "__main__":
    unittest.main()
