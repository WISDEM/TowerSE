#!/usr/bin/env python
# encoding: utf-8
"""
towerstruc.py

Created by Andrew Ning on 2012-01-20.
Copyright (c) NREL. All rights reserved.
"""

import math
import numpy as np
from openmdao.main.api import VariableTree, Component, Assembly
from openmdao.main.datatypes.api import Int, Float, Array, VarTree

from commonse import _akima, sind, cosd, Vector
from commonse.environment import Wind, Wave, Soil
import _pBEAM
from towerSupplement import shellBuckling, fatigue



# -----------------
#  Helper Functions
# -----------------

def cylinderDrag(Re):
    """Drag coefficient for a smooth circular cylinder.

    Parameters
    ----------
    Re : array_like
        Reynolds number

    Returns
    -------
    cd : array_like
        drag coefficient (normalized by cylinder diameter)

    """

    Re /= 1.0e6

    # "Experiments on the Flow Past a Circular Cylinder at Very High Reynolds Numbers", Roshko
    Re_pt = [0.00001, 0.0001, 0.0010, 0.0100, 0.0200, 0.1220, 0.2000, 0.3000, 0.4000,
             0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 5.0000, 10.0000]
    cd_pt = [4.0000,  2.0000, 1.1100, 1.1100, 1.2000, 1.2000, 1.1700, 0.9000, 0.5400,
             0.3100, 0.3800, 0.4600, 0.5300, 0.5700, 0.6100, 0.6400, 0.6700, 0.7000, 0.7000]

    # interpolate
    cd = np.zeros_like(Re)
    cd[Re != 0] = _akima.interpolate(np.log10(Re_pt), cd_pt, np.log10(Re[Re != 0]))

    return cd




# -----------------
#  Variable Trees
# -----------------

class AeroLoads(VariableTree):
    """wind/wave loads"""

    P = VarTree(Vector(), units='N/m', desc='distributed loads')
    q = Array(units='N/m**2', desc='dynamic pressure')
    z = Array(units='m', desc='corresponding heights')
    beta = Array(units='deg', desc='wind/wave angle relative to inertia c.s.')


# -----------------
#  Components
# -----------------


class TowerWindDrag(Component):
    """drag forces on a cylindrical tower due to wind"""

    # in
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='air density')
    mu = Float(1.7934e-5, iotype='in', units='kg/(m*s)', desc='dynamic viscosity of air')
    U = Array(iotype='in', units='m/s', desc='magnitude of wind speed')
    z = Array(iotype='in', units='m', desc='heights where wind speed was computed')
    d = Array(iotype='in', units='m', desc='corresponding diameter of cylinder section')
    beta = Array(iotype='in', units='deg', desc='corresponding wind angles relative to inertial coordinate system')

    # out
    windLoads = VarTree(AeroLoads(), iotype='out', desc='wind loads in inertial coordinate system')


    def execute(self):

        # dynamic pressure
        q = 0.5*self.rho*self.U**2

        # Reynolds number and drag
        Re = self.rho*self.U*self.d/self.mu
        cd = cylinderDrag(Re)
        Fp = q*cd*self.d

        # components of distributed loads
        Px = Fp*cosd(self.beta)
        Py = Fp*sind(self.beta)
        Pz = 0*Fp

        # pack data
        self.windLoads.P.x = Px
        self.windLoads.P.y = Py
        self.windLoads.P.z = Pz
        self.windLoads.q = q
        self.windLoads.z = self.z
        self.windLoads.beta = self.beta



class TowerWaveDrag(Component):
    """drag forces on a cylindrical tower due to waves"""

    # in
    rho = Float(1027.0, iotype='in', units='kg/m**3', desc='water density')
    mu = Float(1.3351e-3, iotype='in', units='kg/(m*s)', desc='dynamic viscosity of water')
    cm = Float(2.0, iotype='in', desc='mass coefficient')
    U = Array(iotype='in', units='m/s', desc='magnitude of wave speed')
    A = Array(iotype='in', units='m/s**2', desc='magnitude of wave acceleration')
    z = Array(iotype='in', units='m', desc='heights where wave speed was computed')
    d = Array(iotype='in', units='m', desc='corresponding diameter of cylinder section')
    beta = Array(iotype='in', units='deg', desc='corresponding wave angles relative to inertial coordinate system')

    # out
    waveLoads = VarTree(AeroLoads(), iotype='out', desc='wave loads in inertial coordinate system')


    def execute(self):

        # dynamic pressure
        q = 0.5*self.rho*self.U**2

        # Reynolds number and drag
        Re = self.rho*self.U*self.d/self.mu
        cd = cylinderDrag(Re)

        # inertial and drag forces
        Fi = self.rho*self.cm*math.pi/4.0*self.d**2*self.A  # Morrison's equation
        Fd = q*cd*self.d
        Fp = Fi + Fd

        # components of distributed loads
        Px = Fp*cosd(self.beta)
        Py = Fp*sind(self.beta)
        Pz = 0*Fp

        # pack data
        self.waveLoads.P.x = Px
        self.waveLoads.P.y = Py
        self.waveLoads.P.z = Pz
        self.waveLoads.q = q
        self.waveLoads.z = self.z
        self.waveLoads.beta = self.beta



class TowerDiscretization(Component):
    """discretize geometry into finite element nodes"""

    # in
    z = Array(iotype='in', units='m', desc='locations along tower, linear lofting between')
    d = Array(iotype='in', units='m', desc='tower diameter at corresponding locations')
    t = Array(iotype='in', units='m', desc='shell thickness at corresponding locations')
    n = Array(iotype='in', dtype=np.int, desc='number of finite elements between sections.  array length should be ``len(z)-1``')

    # out
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



class RNAMass(Component):

    # in
    blade_mass = Float(iotype='in', units='kg', desc='mass of one blade')
    hub_mass = Float(iotype='in', units='kg', desc='mass of hub')
    nac_mass = Float(iotype='in', units='kg', desc='mass of nacelle')
    nBlades = Int(iotype='in', desc='number of blades')

    hub_cm = Array(iotype='in', units='m', desc='location of hub center of mass relative to tower top in yaw-aligned c.s.')
    nac_cm = Array(iotype='in', units='m', desc='location of nacelle center of mass relative to tower top in yaw-aligned c.s.')

    # order for all moments of inertia is (xx, yy, zz, xy, xz, yz) in the yaw-aligned coorinate system
    blade_I = Array(iotype='in', units='kg*m**2', desc='mass moments of inertia of all blades about hub center')
    hub_I = Array(iotype='in', units='kg*m**2', desc='mass moments of inertia of hub about its center of mass')
    nac_I = Array(iotype='in', units='kg*m**2', desc='mass moments of inertia of nacelle about its center of mass')

    # out
    rna_mass = Float(iotype='out', units='kg', desc='total mass of RNA')
    rna_cm = Array(iotype='out', units='m', desc='location of RNA center of mass relative to tower top in yaw-aligned c.s.')
    rna_I_TT = Array(iotype='out', units='kg*m**2', desc='mass moments of inertia of RNA about tower top in yaw-aligned coordinate system')


    def _assembleI(self, Ixx, Iyy, Izz, Ixy, Ixz, Iyz):
        return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


    def _unassembleI(self, I):
        return np.array([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])


    def execute(self):

        rotor_mass = self.blade_mass*self.nBlades + self.hub_mass
        nac_mass = self.nac_mass

        # rna mass
        self.rna_mass = rotor_mass + nac_mass

        # rna cm
        self.rna_cm = (rotor_mass*self.hub_cm + nac_mass*self.nac_cm)/self.rna_mass

        # rna I
        blade_I = self._assembleI(*self.blade_I)
        hub_I = self._assembleI(*self.hub_I)
        nac_I = self._assembleI(*self.nac_I)
        rotor_I = blade_I + hub_I

        R = self.hub_cm
        rotor_I_TT = rotor_I + rotor_mass*(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        R = self.nac_cm
        nac_I_TT = nac_I + nac_mass*(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        self.rna_I_TT = self._unassembleI(rotor_I_TT + nac_I_TT)




class TowerStruc(Component):
    """structural analysis of cylindrical tower

    all forces, moments, distances should be given (and returned) in the yaw-aligned coordinate system
    """

    # geometry
    z = Array(iotype='in', units='m', desc='locations along tower')
    d = Array(iotype='in', units='m', desc='tower diameter at corresponding locations')
    t = Array(iotype='in', units='m', desc='shell thickness at corresponding locations')
    L_reinforced = Float(iotype='in', units='m', desc='reinforcement length for buckling')
    yaw = Float(0.0, iotype='in', units='deg')

    # wind/wave loads
    windLoads = VarTree(AeroLoads(), iotype='in')
    waveLoads = VarTree(AeroLoads(), iotype='in')
    g = Float(9.81, iotype='in', units='m/s')

    # top mass
    top_F = Array(iotype='in')
    top_M = Array(iotype='in')
    top_m = Float(iotype='in')
    top_I = Array(iotype='in', units='kg*m**2', desc='mass moments of inertia. order: (xx, yy, zz, xy, xz, yz)')
    top_cm = Array(iotype='in')

    # soil
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

    life = Int(20, iotype='in', desc='fatigue life of tower')

    # outputs
    mass = Float(iotype='out')
    f1 = Float(iotype='out', units='Hz', desc='first natural frequency')
    f2 = Float(iotype='out', units='Hz', desc='second natural frequency')
    top_deflection = Float(iotype='out', units='m', desc='deflection of tower top in yaw-aligned +x direction')
    stress = Array(iotype='out', units='N/m**2', desc='von Mises stress along tower on downwind side (yaw-aligned +x).  normalized by yield stress.  includes safety factors.')
    z_buckling = Array(iotype='out', units='m', desc='z-locations along tower where shell buckling is evaluted')
    buckling = Array(iotype='out', desc='a shell buckling constraint.  should be <= 0 for feasibility.  includes safety factors')
    damage = Array(iotype='out', desc='fatigue damage at each tower section')


    def execute(self):

        z = self.z
        d = self.d
        t = self.t
        nodes = len(z)


        # aero/hydro loads
        wind = self.windLoads
        wave = self.waveLoads
        hubHt = self.z[-1]  # top of tower
        betaMain = np.interp(hubHt, self.z, wind.beta)  # wind coordinate system defined relative to hub height
        windLoads = wind.P.toDirVec().inertialToWind(betaMain).windToYaw(self.yaw)
        waveLoads = wave.P.toDirVec().inertialToWind(betaMain).windToYaw(self.yaw)

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
        top = _pBEAM.TipData(self.top_m, self.top_cm, self.top_I, self.top_F, self.top_M)

        # soil
        soil = _pBEAM.BaseData(self.k_soil, float('inf'))

        # tower object
        tower = _pBEAM.Beam(nodes, z, d, t, loads, mat, top, soil)

        # mass
        self.mass = tower.mass()

        # natural frequncies
        self.f1, self.f2 = tower.naturalFrequencies(2)

        # deflections due to loading from tower top and wind/wave loads
        dx, dy, dz, dthetax, dthetay, dthetaz = tower.displacement()
        self.top_deflection = dx[-1]  # in yaw-aligned direction

        # axial stress (all stress evaluated on +x yaw side)
        axial_stress = self.E*tower.axialStrain(nodes, d/2.0, 0.0*d, z)

        # shear stress
        Vx, Vy, Fz, Mx, My, Tz = tower.shearAndBending()
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

        # stress
        self.stress = gamma * von_mises / self.sigma_y - 1  # downwind side (yaw-aligned +x)

        # buckling
        gamma_b = self.gamma_m * self.gamma_n
        zb, buckling = shellBuckling(self.z, self.d, self.t, 1, axial_stress, hoop_stress, shear_stress,
                                     self.L_reinforced, self.E, self.sigma_y, self.gamma_f, gamma_b)
        self.z_buckling = zb
        self.buckling = buckling  # yaw-aligned +x side

        # fatigue
        N_DEL = [365*24*3600*self.life]*nodes
        M_DEL = My
        m = 4  # S/N slope
        DC = 80.0  # max stress

        self.damage = fatigue(M_DEL, N_DEL, d, t, m, DC, gamma, stress_factor=1.0, weld_factor=True)



# -----------------
#  Assembly
# -----------------


class Tower(Assembly):

    wind_rho = Float(1.225, iotype='in', units='kg/m**3', desc='air density')
    wind_mu = Float(1.7934e-5, iotype='in', units='kg/(m*s)', desc='dynamic viscosity of air')

    wave_rho = Float(1027.0, iotype='in', units='kg/m**3', desc='water density')
    wave_mu = Float(1.3351e-3, iotype='in', units='kg/(m*s)', desc='dynamic viscosity of water')
    wave_cm = Float(2.0, iotype='in', desc='mass coefficient')

    def configure(self):

        self.add('geometry', TowerDiscretization())
        self.add('wind', Wind())
        self.add('wave', Wave())
        self.add('windLoads', TowerWindDrag())
        self.add('waveLoads', TowerWaveDrag())
        self.add('soil', Soil())
        self.add('rna', RNAMass())
        self.add('tower', TowerStruc())

        self.driver.workflow.add(['geometry', 'wind', 'wave', 'windLoads', 'waveLoads', 'soil', 'rna', 'tower'])

        # wind
        self.connect('geometry.z_node', 'wind.z')

        # wave
        self.connect('geometry.z_node', 'wave.z')

        # wind loads
        self.connect('wind.U', 'windLoads.U')
        self.connect('wind.beta', 'windLoads.beta')
        self.connect('wind_rho', 'windLoads.rho')
        self.connect('wind_mu', 'windLoads.mu')
        self.connect('geometry.z_node', 'windLoads.z')
        self.connect('geometry.d_node', 'windLoads.d')

        # wave loads
        self.connect('wave.U', 'waveLoads.U')
        self.connect('wave.A', 'waveLoads.A')
        self.connect('wave.beta', 'waveLoads.beta')
        self.connect('wave_rho', 'waveLoads.rho')
        self.connect('wave_mu', 'waveLoads.mu')
        self.connect('wave_cm', 'waveLoads.cm')
        self.connect('geometry.z_node', 'waveLoads.z')
        self.connect('geometry.d_node', 'waveLoads.d')


        # tower
        self.connect('geometry.z_node', 'tower.z')
        self.connect('geometry.d_node', 'tower.d')
        self.connect('geometry.t_node', 'tower.t')
        self.connect('windLoads.windLoads', 'tower.windLoads')
        self.connect('waveLoads.waveLoads', 'tower.waveLoads')
        self.connect('soil.k', 'tower.k_soil')
        self.connect('rna.rna_mass', 'tower.top_m')
        self.connect('rna.rna_cm', 'tower.top_cm')
        self.connect('rna.rna_I_TT', 'tower.top_I')

        # passthroughs
        self.create_passthrough('geometry.z')
        self.create_passthrough('geometry.d')
        self.create_passthrough('geometry.t')
        self.create_passthrough('geometry.n')

        self.create_passthrough('rna.blade_mass')
        self.create_passthrough('rna.hub_mass')
        self.create_passthrough('rna.nac_mass')
        self.create_passthrough('rna.nBlades')
        self.create_passthrough('rna.hub_cm')
        self.create_passthrough('rna.nac_cm')
        self.create_passthrough('rna.blade_I')
        self.create_passthrough('rna.hub_I')
        self.create_passthrough('rna.nac_I')


        self.create_passthrough('tower.L_reinforced')
        self.create_passthrough('tower.yaw')
        self.create_passthrough('tower.g')
        self.create_passthrough('tower.top_F')
        self.create_passthrough('tower.top_M')
        self.create_passthrough('tower.E')
        self.create_passthrough('tower.G')
        self.create_passthrough('tower.rho')
        self.create_passthrough('tower.sigma_y')
        self.create_passthrough('tower.gamma_f')
        self.create_passthrough('tower.gamma_m')
        self.create_passthrough('tower.gamma_n')
        self.create_passthrough('tower.life')


        self.create_passthrough('tower.mass')
        self.create_passthrough('tower.f1')
        self.create_passthrough('tower.f2')
        self.create_passthrough('tower.top_deflection')
        self.create_passthrough('tower.stress')
        self.create_passthrough('tower.z_buckling')
        self.create_passthrough('tower.buckling')
        self.create_passthrough('tower.damage')



# if __name__ == '__main__':

#     from commonse.environment import PowerWind, TowerSoil

#     tower = Tower()

#     # geometry
#     towerHt = 87.6
#     tower.z = towerHt*np.array([0.0, 0.5, 1.0])
#     tower.d = [6.0, 4.935, 3.87]
#     tower.t = [0.027*1.3, 0.023*1.3, 0.019*1.3]
#     tower.n = [10, 10]
#     tower.L_reinforced = towerHt/3.0
#     tower.yaw = 0.0

#     # top mass
#     tower.top_m = 359082.653015
#     tower.top_I = [2960437.0, 3253223.0, 3264220.0, 0.0, -18400.0, 0.0]
#     tower.top_cm = [-1.9, 0.0, 1.75]
#     tower.top_F = [1478579.28056464, 0., -3522600.82607833]
#     tower.top_M = [10318177.27285694, 0., 0.]

#     # wind
#     wind = PowerWind()
#     wind.Uref = 20.9
#     wind.zref = towerHt
#     wind.z0 = 0.0
#     wind.shearExp = 0.2
#     tower.replace('wind', wind)

#     # soil
#     soil = TowerSoil()
#     soil.rigid = 6*[True]
#     tower.replace('soil', soil)

#     tower.run()


#     print 'mass =', tower.mass
#     print 'f1 =', tower.f1
#     print 'f2 =', tower.f2
#     print 'top_deflection =', tower.top_deflection
#     print 'stress =', tower.stress
#     print 'z_buckling =', tower.z_buckling
#     print 'buckling =', tower.buckling
#     print 'damage =', tower.damage

