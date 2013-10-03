#!/usr/bin/env python
# encoding: utf-8
"""
towerstruc.py

Created by Andrew Ning on 2012-01-20.
Copyright (c) NREL. All rights reserved.
"""

import math
import numpy as np
from openmdao.main.api import Component, Assembly
from openmdao.main.datatypes.api import Array, Float, VarTree

from utilities import Vector, MassMomentInertia
from wisdeom.common import _pBEAM
from shellBuckling import shellBuckling
from toweraero import AeroLoads, Wind, Wave, PowerWind, LinearWaves, TowerWindDrag, TowerWaveDrag


class Soil(Component):

    k = Array(iotype='out')


class TowerSoil(Soil):

    G = Float(iotype='in', units='Pa', desc='shear modulus of soil')
    nu = Float(iotype='in', desc='Poisson''s ratio of soil')
    h = Float(iotype='in', units='m', desc='depth of foundation in the soil')
    rigid = Array(iotype='in', dtype=np.bool, desc='directions that should be considered infinitely rigid\
        order is x, theta_x, y, theta_y, z, theta_z')
    r0 = Float(iotype='in', units='m', desc='radius of base of tower')


    def execute(self):

        G = self.G
        nu = self.nu
        h = self.depth
        r0 = self.r0

        # vertical
        eta = 1.0 + 0.6*(1.0-nu)*h/r0
        k_z = 4*G*r0*eta/(1.0-nu)

        # horizontal
        eta = 1.0 + 0.55*(2.0-nu)*h/r0
        k_x = 32.0*(1.0-nu)*G*r0*eta/(7.0-8.0*nu)

        # rocking
        eta = 1.0 + 1.2*(1.0-nu)*h/r0 + 0.2*(2.0-nu)*(h/r0)**3
        k_thetax = 8.0*G*r0**3*eta/(3.0*(1.0-nu))

        # torsional
        k_phi = 16.0*G*r0**3/3.0

        k = np.array([k_x, k_thetax, k_x, k_thetax, k_z, k_phi])
        k[self.rigid] = float('inf')

        return k


class TowerDiscretization(Component):

    # geometry
    z = Array(iotype='in', units='m', desc='locations along tower, linear lofting between')
    d = Array(iotype='in', units='m', desc='tower diameter at corresponding locations')
    t = Array(iotype='in', units='m', desc='shell thickness at corresponding locations')
    n = Array(iotype='in', dtype=np.int, desc='number of finite elements between sections.  array length should be ``len(z)-1``')

    # geometry
    z_node = Array(iotype='out', units='m', desc='locations along tower, linear lofting between')
    d_node = Array(iotype='out', units='m', desc='tower diameter at corresponding locations')
    t_node = Array(iotype='out', units='m', desc='shell thickness at corresponding locations')

    def execute(self):

        # compute nodal locations
        self.z_node = np.array([self.z[0]])
        for i in range(len(self.n)):
            znode = np.linspace(self.z[i], self.z[i+1], self.n[i]+1)
            self.z_node = np.r_[self.z_node, znode[1:]]

        # interpolate
        self.d_node = np.interp(self.z_node, self.z, self.d)
        self.t_node = np.interp(self.z_node, self.z, self.t)




class TowerStruc(Component):

    # geometry
    z = Array(iotype='in', units='m', desc='locations along tower')
    d = Array(iotype='in', units='m', desc='tower diameter at corresponding locations')
    t = Array(iotype='in', units='m', desc='shell thickness at corresponding locations')
    # n = Array(iotype='in', dtype=np.int, desc='number of finite elements between sections.  array length should be ``len(z)-1``')
    L_reinforced = Float(iotype='in', units='m', desc='reinforcement length for buckling')

    windLoads = VarTree(AeroLoads(), iotype='in')
    waveLoads = VarTree(AeroLoads(), iotype='in')
    # zLoads = Array(iotype='in', units='m')
    # qWind = Array(iotype='in', units='N/m**2')
    # qWave = Array(iotype='in', units='N/m**2')
    # betaWind = Array(iotype='in', units='deg')
    # betaWave = Array(iotype='in', units='deg')
    yaw = Float(iotype='in', units='deg')

    top_F = VarTree(Vector, iotype='in')
    top_M = VarTree(Vector, iotype='in')
    top_m = Float(iotype='in')
    top_I = VarTree(MassMomentInertia, iotype='in')
    top_cm = Array(iotype='in')

    k_soil = Array(iotype='in', desc='stiffness properties at base of foundation')

    # material properties
    E = Float(210e9, iotype='in', units='N/m**2', desc='material modulus of elasticity')
    G = Float(80.8e9, iotype='in', units='N/m**2', desc='material shear modulus')
    rho = Float(8500.0, iotype='in', units='kg/m**3', desc='material density')
    sigma_y = Float(450.0e6, iotype='in', units='N/m**2', desc='yield stress')

    # safety factors
    gamma_f = Float(1.35, iotype='in', desc='safety factor on loads')
    gamma_m = Float(1.1, iotype='in', desc='safety factor on materials')
    gamma_n = Float(1.0, iotype='in', desc='safety factor on consequence of failure')

    # ---- in --------

    # outputs
    mass = Float(iotype='out')
    f1 = Float(iotype='out', units='Hz', desc='first natural frequency')
    f2 = Float(iotype='out', units='Hz', desc='second natural frequency')
    top_deflection = Float(iotype='out', units='m', desc='deflection of tower top in yaw-aligned +x direction')
    z_stress = Array(iotype='out', units='m', desc='z-locations along tower where stress is evaluted')
    stress = Array(iotype='out', units='N/m**2', desc='von Mises stress along tower on downwind side (yaw-aligned +x).  normalized by yield stress.  includes safety factors.')
    z_buckling = Array(iotype='out', units='m', desc='z-locations along tower where shell buckling is evaluted')
    buckling = Array(iotype='out', desc='a shell buckling constraint.  should be <= 0 for feasibility.  includes safety factors')


    def execute(self):

        z = self.z
        d = self.d
        t = self.t
        nodes = len(z)


        # aero/hydro loads
        wind = self.windLoads
        wave = self.waveLoads
        windLoads = wind.P.toDirVec().inertialToWind(wind.beta).windToYaw(self.yaw)
        waveLoads = wave.P.toDirVec().inertialToWind(wave.beta).windToYaw(self.yaw)

        Px = np.interp(z, wind.z, windLoads.x) + np.interp(z, wave.z, waveLoads.x)
        Py = np.interp(z, wind.z, windLoads.y) + np.interp(z, wave.z, waveLoads.y)
        Pz = np.interp(z, wind.z, windLoads.z) + np.interp(z, wave.z, waveLoads.z)

        # add weight loads
        Pz_weight = -self.rho*self.g*math.pi*d*t
        Pz += Pz_weight

        # pBEAM loads object
        loads = _pBEAM.Loads(nodes, Px, Py, Pz)

        # material properties
        mat = _pBEAM.Material(self.E, self.G, self.rho)

        # RNA properties
        topI = self.topI
        I = np.array([topI.xx, topI.yy, topI.zz, topI.xy, topI.xz, topI.yz])
        top = _pBEAM.TipData(self.top_m, self.top_cm, I, self.top_F, self.top_M)

        # soil
        soil = _pBEAM.BaseData(self.k_soil, float('inf'))

        # tower object
        tower = _pBEAM.Beam(nodes, z, d, t, loads, mat, top, soil)

        # mass
        self.mass = tower.mass()

        # natural frequncies
        self.f1, self.f2 = tower.naturalFrequencies(2)

        # deflections due to loading from tower top and wind/wave loads
        dx, dy, dz, dthetax, dthetay, dthetaz = self.tower.displacement()
        self.top_deflection = dx[-1]  # in yaw-aligned direction

        # axial stress (all stress evaluated on +x yaw side)
        axial_stress = self.E*tower.axialStrain(nodes, d/2.0, 0.0*d, z)

        # shear stress
        Vx, Vy, Fz, Mx, My, Tz = self.tower.shearAndBending()
        A = math.pi * d * t
        shear_stress = 2 * Vx / A

        # hoop_stress (Eurocode method)
        C_theta = 1.5
        r = d/2.0
        omega = self.L_reinforced/np.sqrt(r*t)
        k_w = 0.46*(1.0 + 0.1*np.sqrt(C_theta/omega*r/t))
        k_w = np.maximum(0.65, np.minimum(1.0, k_w))
        q_dyn = np.interp(z, wind.z, wind.q) + np.interp(z, wave.z, wave.q)
        Peq = k_w*q_dyn
        hoop_stress = -Peq*r/t

        # von mises stress
        a = ((axial_stress + hoop_stress)/2.0)**2
        b = ((axial_stress - hoop_stress)/2.0)**2
        c = shear_stress**2
        von_mises = np.sqrt(a + 3.0*(b+c))

        # safety factors
        gamma = self.gamma_f * self.gamma_m * self.gamma_n

        self.z_stress = z
        self.stress = gamma * von_mises / self.sigma_y  # downwind side (yaw-aligned +x)

        gamma_b = self.gamma_m * self.gamma_n
        zb, buckling = shellBuckling(1, axial_stress, hoop_stress, shear_stress,
                                     self.L_reinforced, self.gamma_f, gamma_b)
        self.z_buckling = zb
        self.buckling = buckling  # yaw-aligned +x side




class Tower(Assembly):

    wind_rho = Float(iotype='in', units='kg/m**3', desc='air density')
    wind_mu = Float(iotype='in', units='kg/(m*s)', desc='dynamic viscosity of air')

    wave_rho = Float(iotype='in', units='kg/m**3', desc='water density')
    wave_mu = Float(iotype='in', units='kg/(m*s)', desc='dynamic viscosity of water')
    wave_cm = Float(2.0, iotype='in', desc='mass coefficient')


    def configure(self):

        self.add('geometry', TowerDiscretization())
        self.add('wind', Wind())
        self.add('wave', Wave())
        self.add('windLoads', TowerWindDrag())
        self.add('waveLoads', TowerWaveDrag())
        self.add('soil', Soil())
        self.add('tower', TowerStruc())

        self.driver.workflow.add(['geometry', 'wind', 'wave', 'windLoads', 'waveLoads', 'soil', 'tower'])

        # wind
        self.connect('geometry.z', 'wind.z')

        # wave
        self.connect('geometry.z', 'wave.z')

        # wind loads
        self.connect('wind.U', 'windLoads.U')
        self.connect('wind.beta', 'windLoads.beta')
        self.connect('wind_rho', 'windLoads.rho')
        self.connect('wind_mu', 'windLoads.mu')
        self.connect('geometry.z', 'windLoads.z')
        self.connect('geometry.d', 'windLoads.d')

        # wave loads
        self.connect('wave.U', 'waveLoads.U')
        self.connect('wave.A', 'waveLoads.A')
        self.connect('wave.beta', 'waveLoads.beta')
        self.connect('wave_rho', 'waveLoads.rho')
        self.connect('wave_mu', 'waveLoads.mu')
        self.connect('wave_cm', 'waveLoads.cm')
        self.connect('geometry.z', 'waveLoads.z')
        self.connect('geometry.d', 'waveLoads.d')

        # tower
        self.connect('geometry.z', 'tower.z')
        self.connect('geometry.d', 'tower.d')
        self.connect('geometry.t', 'tower.t')
        self.connect('windLoads.windLoads', 'tower.windLoads')
        self.connect('waveLoads.waveLoads', 'tower.waveLoads')
        self.connect('soil.k', 'tower.k_soil')

        # passthroughs
        self.create_passthrough('L_reinforced')
        self.create_passthrough('yaw')
        self.create_passthrough('top_F')
        self.create_passthrough('top_M')
        self.create_passthrough('top_m')
        self.create_passthrough('top_I')
        self.create_passthrough('top_cm')
        self.create_passthrough('E')
        self.create_passthrough('G')
        self.create_passthrough('rho')
        self.create_passthrough('sigma_y')
        self.create_passthrough('gamma_f')
        self.create_passthrough('gamma_m')
        self.create_passthrough('gamma_n')


        self.create_passthrough('tower.mass')
        self.create_passthrough('tower.f1')
        self.create_passthrough('tower.f2')
        self.create_passthrough('tower.top_deflection')
        self.create_passthrough('tower.z_stress')
        self.create_passthrough('tower.stress')
        self.create_passthrough('tower.z_buckling')
        self.create_passthrough('tower.buckling')





