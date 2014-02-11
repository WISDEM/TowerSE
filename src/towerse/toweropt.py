#!/usr/bin/env python
# encoding: utf-8
"""
toweropt.py

Created by Andrew Ning on 2013-12-03.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from pyopt_driver.pyopt_driver import pyOptDriver
from openmdao.lib.casehandlers.api import DumpCaseRecorder
from openmdao.main.datatypes.api import Float

from towerse.tower import TowerSE



class TowerOpt(TowerSE):

    freq1p = Float(iotype='in', units='Hz', desc='1P rotor frequency')


    def configure(self):
        super(TowerOpt, self).configure()

        # set optimizer
        self.replace('driver', pyOptDriver())
        self.driver.optimizer = 'SNOPT'
        self.driver.options = {'Major feasibility tolerance': 1e-6,
                               'Minor feasibility tolerance': 1e-6,
                               'Major optimality tolerance': 1e-4,
                               'Function precision': 1e-8,
                               'Iterations limit': 500000000}
                               # "New basis file": 10}
        # if os.path.exists("fort.10"):
        #     self.driver.options["Old basis file"] = 10

        # Objective
        self.driver.add_objective('tower1.mass / 300000')

        # Design Variables
        self.z = np.zeros(3)
        self.d = np.zeros(3)
        self.driver.add_parameter('z[1]', low=0.25, high=0.75)
        self.driver.add_parameter('d[:-1]', low=3.87, high=20.0)
        self.driver.add_parameter('t', low=0.005, high=0.2)

        # outfile = open('resultso.txt', 'w')
        # self.driver.recorders = [DumpCaseRecorder(outfile)]
        self.driver.recorders = [DumpCaseRecorder()]

        # Constraints
        self.driver.add_constraint('tower1.stress <= 0.0')
        self.driver.add_constraint('tower2.stress <= 0.0')
        self.driver.add_constraint('tower1.buckling <= 0.0')
        self.driver.add_constraint('tower2.buckling <= 0.0')
        self.driver.add_constraint('tower1.damage <= 1.0')
        self.driver.add_constraint('gc.weldability <= 0.0')
        self.driver.add_constraint('gc.manufactuability <= 0.0')
        self.driver.add_constraint('tower1.f1 >= 1.1*freq1p')





if __name__ == '__main__':


    from math import pi
    from towerse.tower import TowerWithpBEAM
    from commonse.environment import PowerWind, TowerSoil


    tower = TowerOpt()


    # ---- tower ------
    tower.replace('wind1', PowerWind())
    tower.replace('wind2', PowerWind())
    # onshore (no waves)
    tower.replace('soil', TowerSoil())
    tower.replace('tower1', TowerWithpBEAM())
    tower.replace('tower2', TowerWithpBEAM())

    tower.wind1.missing_deriv_policy = 'assume_zero'  # TODO: remove these later after OpenMDAO fixes this issue
    tower.wind2.missing_deriv_policy = 'assume_zero'
    tower.soil.missing_deriv_policy = 'assume_zero'
    tower.tower1.missing_deriv_policy = 'assume_zero'
    tower.tower2.missing_deriv_policy = 'assume_zero'

    # geometry
    tower.towerHeight = 87.6
    tower.z = np.array([0.0, 0.5, 1.0])
    tower.d = [6.0, 4.935, 3.87]
    tower.t = [0.027*1.3, 0.023*1.3, 0.019*1.3]
    tower.n = [10, 10]
    tower.n_reinforced = 3
    tower.yaw = 0.0
    tower.tilt = 5.0


    # constraints
    V_max = 80.0  # tip speed
    D = 126.0
    tower.min_d_to_t = 120.0
    tower.min_taper = 0.4
    tower.freq1p = V_max / (D/2) / (2*pi)  # convert to Hz

    # safety factors
    tower.gamma_f = 1.35
    tower.gamma_m = 1.3
    tower.gamma_n = 1.0

    # max Thrust case
    tower.wind_Uref1 = 16.030
    tower.rotorT1 = 1.3295e6
    tower.rotorQ1 = 6.2829e6

    # max wind speed case
    tower.wind_Uref2 = 67.89
    tower.rotorT2 = 1.1770e6
    tower.rotorQ2 = 1.5730e6

    # blades
    tower.blades_mass = 3 * 15241.323
    bladeI = 3 * 8791992.000
    tower.blades_I = np.array([bladeI, bladeI/2.0, bladeI/2.0, 0.0, 0.0, 0.0])

    # hub
    tower.hub_mass = 50421.4
    tower.hub_cm = np.array([-6.30, 0, 3.15])
    tower.hub_I = np.array([127297.8, 127297.8, 127297.8, 0.0, 0.0, 0.0])

    # nacelle
    tower.nac_mass = 221245.8
    tower.nac_cm = np.array([-0.32, 0, 2.40])
    tower.nac_I = np.array([9908302.58, 912488.28, 1160903.54, 0.0, 0.0, 0.0])

    # wind
    towerToShaft = 2.0
    tower.wind_zref = tower.towerHeight + towerToShaft
    tower.wind_z0 = 0.0
    tower.wind1.shearExp = 0.2
    tower.wind2.shearExp = 0.2

    # soil
    tower.soil.rigid = 6*[True]


    # fatigue
    tower.z_DEL = np.array([0.000, 1.327, 3.982, 6.636, 9.291, 11.945, 14.600, 17.255, 19.909, 22.564, 25.218, 27.873, 30.527, 33.182, 35.836, 38.491, 41.145, 43.800, 46.455, 49.109, 51.764, 54.418, 57.073, 59.727, 62.382, 65.036, 67.691, 70.345, 73.000, 75.655, 78.309, 80.964, 83.618, 86.273, 87.600])
    tower.M_DEL = 1e3*np.array([8.2940E+003, 8.1518E+003, 7.8831E+003, 7.6099E+003, 7.3359E+003, 7.0577E+003, 6.7821E+003, 6.5119E+003, 6.2391E+003, 5.9707E+003, 5.7070E+003, 5.4500E+003, 5.2015E+003, 4.9588E+003, 4.7202E+003, 4.4884E+003, 4.2577E+003, 4.0246E+003, 3.7942E+003, 3.5664E+003, 3.3406E+003, 3.1184E+003, 2.8977E+003, 2.6811E+003, 2.4719E+003, 2.2663E+003, 2.0673E+003, 1.8769E+003, 1.7017E+003, 1.5479E+003, 1.4207E+003, 1.3304E+003, 1.2780E+003, 1.2673E+003, 1.2761E+003])
    tower.gamma_fatigue = 1.35*1.3*1.0
    tower.life = 20.0
    tower.m_SN = 4

    # use defaults
    # tower.wind_rho
    # tower.wind_mu
    # tower.g
    # tower.E
    # tower.G
    # tower.rho
    # tower.sigma_y

    # onshore
    # tower.wave_rho
    # tower.wave_mu
    # tower.wave_cm



    # tower.check_gradient('driver')

    # tower.driver.gradient_options.force_fd = True
    tower.driver.gradient_options.fd_step_type = 'relative'


    tower.run()

















#     # blades (optimized)
#     tower.blade_mass = 15241.323
#     nBlades = 3
#     bladeI = 8791992.000 * nBlades
#     tower.blade_I = np.array([bladeI, bladeI/2.0, bladeI/2.0, 0.0, 0.0, 0.0])
#     tower.nBlades = nBlades

#     # hub (optimized)
#     tower.hub_mass = 50421.4
#     tower.hub_cm = np.array([-6.30, 0, 3.15])
#     tower.hub_I = np.array([127297.8, 127297.8, 127297.8, 0.0, 0.0, 0.0])

#     # nacelle (optimized)
#     tower.nac_mass = 221245.8
#     tower.nac_cm = np.array([-0.32, 0, 2.40])
#     tower.nac_I = np.array([9908302.58, 912488.28, 1160903.54, 0.0, 0.0, 0.0])

#     # max Thrust case
#     tower.F1 = np.array([1.3295e6, -2.2694e4, -4.6184e6])
#     tower.M1 = np.array([6.2829e6, -1.0477e6, 3.9029e6])
#     V1 = 16.030

#     # max wind speed case
#     tower.F2 = np.array([1.1770e6, -1.3607e5, -3.5970e6])
#     tower.M2 = np.array([1.5730e6, 2.2660e6, -3.6542e6])
#     V2 = 67.89

#     # damage
#     tower.life = 20.0
#     tower.gamma_fatigue = 1.35*1.3*1.0
#     tower.z_DEL = np.array([0.000, 1.327, 3.982, 6.636, 9.291, 11.945, 14.600, 17.255, 19.909, 22.564, 25.218, 27.873, 30.527, 33.182, 35.836, 38.491, 41.145, 43.800, 46.455, 49.109, 51.764, 54.418, 57.073, 59.727, 62.382, 65.036, 67.691, 70.345, 73.000, 75.655, 78.309, 80.964, 83.618, 86.273, 87.600])
#     tower.M_DEL = 1e3*np.array([8.2940E+003, 8.1518E+003, 7.8831E+003, 7.6099E+003, 7.3359E+003, 7.0577E+003, 6.7821E+003, 6.5119E+003, 6.2391E+003, 5.9707E+003, 5.7070E+003, 5.4500E+003, 5.2015E+003, 4.9588E+003, 4.7202E+003, 4.4884E+003, 4.2577E+003, 4.0246E+003, 3.7942E+003, 3.5664E+003, 3.3406E+003, 3.1184E+003, 2.8977E+003, 2.6811E+003, 2.4719E+003, 2.2663E+003, 2.0673E+003, 1.8769E+003, 1.7017E+003, 1.5479E+003, 1.4207E+003, 1.3304E+003, 1.2780E+003, 1.2673E+003, 1.2761E+003])


#     tower.gamma_f = 1.0  # Andy already added them in
#     tower.gamma_m = 1.3
#     tower.gamma_n = 1.0


#     # wind
#     tower.wind1.Uref = V1
#     tower.wind1.zref = towerHt
#     tower.wind1.z0 = 0.0
#     tower.wind1.shearExp = 0.2

#     tower.wind2.Uref = V2
#     tower.wind2.zref = towerHt
#     tower.wind2.z0 = 0.0
#     tower.wind2.shearExp = 0.2


#     # soil
#     tower.soil1.rigid = 6*[True]
#     tower.soil2.rigid = 6*[True]


#     tower.driver.iprint = 3
#     # tower.driver.pyopt_diff = True
#     # tower.driver.accuracy = 1e-10
#     tower.run()


#     print tower.pre.z_break
#     print tower.b_d.input
#     print tower.b_t.input

# #     mass = 232111.012709
# # f1 = 0.284825529452
# # f2 = 0.288560436057
# # stress1 = [-0.53198346 -0.52326295 -0.51530033 -0.50840332 -0.50297342 -0.49953483
# #  -0.49877358 -0.50159131 -0.50917961 -0.52312325 -0.54554388 -0.55869269
# #  -0.57446226 -0.59333339 -0.61589173 -0.6428479  -0.67504787 -0.71343611
# #  -0.75882675 -0.81083449 -0.86245745]
# # stress2 = [-0.4889579  -0.48380576 -0.47979027 -0.47712482 -0.47611691 -0.47718142
# #  -0.48086616 -0.48789111 -0.49920432 -0.51606048 -0.54013009 -0.55366842
# #  -0.56928817 -0.58734819 -0.6082872  -0.63264356 -0.661075   -0.69436714
# #  -0.73338683 -0.77880311 -0.82974017]
# # buckling1 = [-0.13730564 -0.15420826 -0.03037412]
# # buckling2 = [  1.51275503e-09  -8.14012471e-02   1.98538750e-08]
# # damage = [ 0.88000362  0.92775686  0.96368779  0.98768153  0.99651437  0.99972827
# #   1.          0.98897983  0.9426402   0.86429958  0.75201071  0.69251103
# #   0.630452    0.56663153  0.50445855  0.4518828   0.42069249  0.42063254
# #   0.47710294  0.6353093   1.        ]
# # weldability = [-1.22324308 -0.75489307 -2.21564455]
# # manufactuability = -0.199552233137



    # print 'mass =', tower.mass
    # print 'f1 =', tower.f1
    # print 'f2 =', tower.f2
    # print 'top_deflection1 =', tower.top_deflection1
    # print 'top_deflection2 =', tower.top_deflection2
    # print 'stress1 =', tower.stress1
    # print 'stress2 =', tower.stress2
    # print 'z_buckling =', tower.z_buckling
    # print 'buckling1 =', tower.buckling1
    # print 'buckling2 =', tower.buckling2
    # print 'damage1 =', tower.damage1
    # print 'damage2 =', tower.damage2
    # print 'weldability =', tower.weldability
    # print 'manufactuability =', tower.manufactuability

