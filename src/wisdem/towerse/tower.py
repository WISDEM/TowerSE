#!/usr/bin/env python
# encoding: utf-8
"""
towerstruc.py

Created by Andrew Ning on 2012-01-20.
Copyright (c) NREL. All rights reserved.
"""

import math
import numpy as np
from scipy.optimize import brentq
from openmdao.main.api import VariableTree, Component, Assembly
from openmdao.main.datatypes.api import Int, Float, Array, VarTree

from wisdem.commonse import _akima, sind, cosd, Vector
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
#  Base Components
# -----------------


class Wind(Component):
    """base component for wind speed/direction"""

    # in
    z = Array(iotype='in', units='m', desc='heights where wind speed was computed')

    # out
    U = Array(iotype='out', units='m/s', desc='magnitude of wind speed at each z location')
    beta = Array(iotype='out', units='deg', desc='corresponding wind angles relative to inertial coordinate system')


class Wave(Component):
    """base component for wave speed/direction"""

    # in
    z = Array(iotype='in', units='m', desc='heights where wave speed was computed')

    # out
    U = Array(iotype='out', units='m/s', desc='magnitude of wave speed at each z location')
    A = Array(iotype='out', units='m/s**2', desc='magnitude of wave acceleration at each z location')
    beta = Array(iotype='out', units='deg', desc='corresponding wave angles relative to inertial coordinate system')


    def execute(self):
        """default to no waves"""
        n = len(self.z)
        self.U = np.zeros(n)
        self.A = np.zeros(n)
        self.beta = np.zeros(n)


class Soil(Component):
    """base component for soil stiffness"""

    # out
    k = Array(iotype='out', units='N/m', desc='spring stiffness. rigid directions should use \
        ``float(''inf'')``. order: (x, theta_x, y, theta_y, z, theta_z)')



# -----------------------
#  Subclassed Components
# -----------------------


class PowerWind(Wind):
    """power-law profile wind"""

    # variables
    Uref = Float(iotype='in', units='m/s', desc='reference velocity of power-law model')
    zref = Float(iotype='in', units='m', desc='corresponding reference height')
    z0 = Float(0.0, iotype='in', units='m', desc='bottom of wind profile (height of ground/sea)')

    # parameters
    shearExp = Float(0.2, iotype='in', desc='shear exponent')
    betaWind = Float(0.0, iotype='in', units='deg', desc='wind angle relative to inertial coordinate system')


    def execute(self):

        # rename
        z = self.z
        zref = self.zref
        z0 = self.z0

        # velocity
        self.U = np.zeros_like(z)
        idx = z > z0
        self.U[idx] = self.Uref*((z[idx] - z0)/(zref - z0))**self.shearExp
        self.beta = self.betaWind*np.ones_like(z)


    def calculate_first_derivatives(self):

        # rename
        z = self.z
        zref = self.zref
        z0 = self.z0

        self.dU_dUref = np.zeros_like(z)
        self.dU_dz = np.zeros_like(z)
        self.dU_dzref = np.zeros_like(z)
        self.dU_dz0 = np.zeros_like(z)

        idx = z > z0
        self.dU_dUref[idx] = ((z[idx] - z0)/(zref - z0))**self.shearExp
        self.dU_dz[idx] = self.U[idx]*self.shearExp * 1.0/(z[idx] - z0)
        self.dU_dzref[idx] = self.U[idx]*self.shearExp * -1.0/(zref - z0)
        self.dU_dz0[idx] = self.U[idx]*self.shearExp * (1.0/(zref - z0) - 1.0/(z[idx] - z0))



class LogWind(Wind):
    """logarithmic-profile wind"""

    # ---------- in -----------------
    Uref = Float(iotype='in', units='m/s', desc='reference velocity of power-law model')
    zref = Float(iotype='in', units='m', desc='corresponding reference height')
    z0 = Float(0.0, iotype='in', units='m', desc='bottom of wind profile (height of ground/sea)')
    z_roughness = Float(10.0, iotype='in', units='mm', desc='surface roughness length')
    betaWind = Float(0.0, iotype='in', units='deg', desc='wind angle relative to inertial coordinate system')


    def execute(self):

        # rename
        z = self.z
        zref = self.zref
        z0 = self.z0
        z_roughness = self.z_roughness

        # find velocity
        self.U = np.zeros_like(z)
        idx = [z > z0]
        self.U[idx] = self.Uref*(np.log((z[idx] - z0)/z_roughness) / math.log((zref - z0)/z_roughness))
        self.beta = self.betaWind*np.ones_like(z)



class LinearWaves(Wave):
    """linear (Airy) wave theory"""

    # ---------- in -------------
    hs = Float(iotype='in', units='m', desc='significant wave height (crest-to-trough)')
    T = Float(iotype='in', units='s', desc='period of waves')
    g = Float(9.81, iotype='in', units='m/s**2', desc='acceleration of gravity')
    Uc = Float(iotype='in', units='m/s', desc='mean current speed')
    betaWave = Float(0.0, iotype='in', units='deg', desc='wave angle relative to inertial coordinate system')
    z_surface = Float(iotype='in', units='m', desc='vertical location of water surface')
    z_floor = Float(0.0, iotype='in', units='m', desc='vertical location of sea floor')


    def execute(self):

        # water depth
        d = self.z_surface - self.z_floor

        # design wave height
        h = 1.1*self.hs

        # circular frequency
        omega = 2.0*math.pi/self.T

        # compute wave number from dispersion relationship
        k = brentq(lambda k: omega**2 - self.g*k*math.tanh(d*k), 0, 10*omega**2/self.g)

        # zero at surface
        z_rel = self.z - self.z_surface

        # maximum velocity
        self.U = h/2.0*omega*np.cosh(k*(z_rel + d))/math.sinh(k*d) + self.Uc

        # check heights
        self.U[np.logical_or(self.z < self.z_floor, self.z > self.z_surface)] = 0

        # acceleration
        self.A = self.U * omega

        # angles
        self.beta = self.betaWave*np.ones_like(self.z)



class TowerSoil(Soil):
    """textbook soil stiffness method"""

    # in
    G = Float(140e6, iotype='in', units='Pa', desc='shear modulus of soil')
    nu = Float(0.4, iotype='in', desc='Poisson''s ratio of soil')
    depth = Float(1.0, iotype='in', units='m', desc='depth of foundation in the soil')
    rigid = Array(iotype='in', dtype=np.bool, desc='directions that should be considered infinitely rigid\
        order is x, theta_x, y, theta_y, z, theta_z')
    r0 = Float(1.0, iotype='in', units='m', desc='radius of base of tower')


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

        self.k = np.array([k_x, k_thetax, k_x, k_thetax, k_z, k_phi])
        self.k[self.rigid] = float('inf')



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




class TowerStruc(Component):
    """structural analysis of cylindrical tower"""

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
        self.stress = gamma * von_mises / self.sigma_y  # downwind side (yaw-aligned +x)

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
        self.add('tower', TowerStruc())

        self.driver.workflow.add(['geometry', 'wind', 'wave', 'windLoads', 'waveLoads', 'soil', 'tower'])

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

        # passthroughs
        self.create_passthrough('geometry.z')
        self.create_passthrough('geometry.d')
        self.create_passthrough('geometry.t')
        self.create_passthrough('geometry.n')
        self.create_passthrough('tower.L_reinforced')
        self.create_passthrough('tower.yaw')
        self.create_passthrough('tower.top_F')
        self.create_passthrough('tower.top_M')
        self.create_passthrough('tower.top_m')
        self.create_passthrough('tower.top_I')
        self.create_passthrough('tower.top_cm')
        self.create_passthrough('tower.E')
        self.create_passthrough('tower.G')
        self.create_passthrough('tower.rho')
        self.create_passthrough('tower.sigma_y')
        self.create_passthrough('tower.gamma_f')
        self.create_passthrough('tower.gamma_m')
        self.create_passthrough('tower.gamma_n')


        self.create_passthrough('tower.mass')
        self.create_passthrough('tower.f1')
        self.create_passthrough('tower.f2')
        self.create_passthrough('tower.top_deflection')
        self.create_passthrough('tower.stress')
        self.create_passthrough('tower.z_buckling')
        self.create_passthrough('tower.buckling')
        self.create_passthrough('tower.damage')



if __name__ == '__main__':

    tower = Tower()

    # geometry
    towerHt = 87.6
    tower.z = towerHt*np.array([0.0, 0.5, 1.0])
    tower.d = [6.0, 4.935, 3.87]
    tower.t = [0.027*1.3, 0.023*1.3, 0.019*1.3]
    tower.n = [10, 10]
    tower.L_reinforced = towerHt/3.0
    tower.yaw = 0.0

    # top mass
    tower.top_m = 359082.653015
    tower.top_I = [2960437.0, 3253223.0, 3264220.0, 0.0, -18400.0, 0.0]
    tower.top_cm = [-1.9, 0.0, 1.75]
    tower.top_F = [1478579.28056464, 0., -3522600.82607833]
    tower.top_M = [10318177.27285694, 0., 0.]

    # wind
    wind = PowerWind()
    wind.Uref = 20.9
    wind.zref = towerHt
    wind.z0 = 0.0
    wind.shearExp = 0.2
    tower.replace('wind', wind)

    # soil
    soil = TowerSoil()
    soil.rigid = 6*[True]
    tower.replace('soil', soil)

    tower.run()


    print 'mass =', tower.mass
    print 'f1 =', tower.f1
    print 'f2 =', tower.f2
    print 'top_deflection =', tower.top_deflection
    print 'stress =', tower.stress
    print 'z_buckling =', tower.z_buckling
    print 'buckling =', tower.buckling
    print 'damage =', tower.damage

