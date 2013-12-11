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
    n_reinforced = Int(iotype='in', desc='must be a minimum of 1 (top and bottom)')

    # out
    z_node = Array(iotype='out', units='m', desc='locations along tower, linear lofting between')
    d_node = Array(iotype='out', units='m', desc='tower diameter at corresponding locations')
    t_node = Array(iotype='out', units='m', desc='shell thickness at corresponding locations')
    z_reinforced = Array(iotype='out')

    def execute(self):

        # compute nodal locations
        self.z_node = np.array([self.z[0]])
        for i in range(len(self.n)):
            znode = np.linspace(self.z[i], self.z[i+1], self.n[i]+1)
            self.z_node = np.r_[self.z_node, znode[1:]]

        # interpolate
        self.d_node = np.interp(self.z_node, self.z, self.d)
        self.t_node = np.interp(self.z_node, self.z, self.t)

        # reinforcement distances
        self.z_reinforced = np.linspace(self.z[0], self.z[-1], self.n_reinforced+1)



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
    z_reinforced = Array(iotype='in', units='m', desc='reinforcement positions for buckling')
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
    m_SN = Int(4, iotype='in', desc='slope of S/N curve')
    DC = Float(80.0, iotype='in', desc='standard value of stress')
    gamma_fatigue = Float(1.485, iotype='in', desc='total safety factor for fatigue')
    z_DEL = Array(iotype='in')
    M_DEL = Array(iotype='in')

    # outputs
    mass = Float(iotype='out')
    f1 = Float(iotype='out', units='Hz', desc='first natural frequency')
    f2 = Float(iotype='out', units='Hz', desc='second natural frequency')
    top_deflection = Float(iotype='out', units='m', desc='deflection of tower top in yaw-aligned +x direction')
    stress = Array(iotype='out', units='N/m**2', desc='von Mises stress along tower on downwind side (yaw-aligned +x).  normalized by yield stress.  includes safety factors.')
    z_buckling = Array(iotype='out', units='m', desc='z-locations along tower where shell buckling is evaluted')
    buckling = Array(iotype='out', desc='a shell buckling constraint.  should be <= 0 for feasibility.  includes safety factors')
    damage = Array(iotype='out', desc='fatigue damage at each tower section')


    def aerohydroLoadsAtNodes(self):
        """rotate wind loads to yaw c.s. and interpolate onto nodes"""

        # aero/hydro loads
        wind = self.windLoads
        wave = self.waveLoads
        hubHt = self.z[-1]  # top of tower
        betaMain = np.interp(hubHt, self.z, wind.beta)  # wind coordinate system defined relative to hub height
        windLoads = wind.P.toDirVec().inertialToWind(betaMain).windToYaw(self.yaw)
        waveLoads = wave.P.toDirVec().inertialToWind(betaMain).windToYaw(self.yaw)

        Px = np.interp(self.z, wind.z, windLoads.x) + np.interp(self.z, wave.z, waveLoads.x)
        Py = np.interp(self.z, wind.z, windLoads.y) + np.interp(self.z, wave.z, waveLoads.y)
        Pz = np.interp(self.z, wind.z, windLoads.z) + np.interp(self.z, wave.z, waveLoads.z)


        return Px, Py, Pz


    def hoopStressEurocode(self):
        """default method for computing hoop stress using Eurocode method"""

        wind = self.windLoads
        wave = self.waveLoads
        r = self.d/2.0
        t = self.t

        C_theta = 1.5
        omega = (self.z_reinforced[1] - self.z_reinforced[0])/np.sqrt(r*t)
        k_w = 0.46*(1.0 + 0.1*np.sqrt(C_theta/omega*r/t))
        k_w = np.maximum(0.65, np.minimum(1.0, k_w))
        q_dyn = np.interp(self.z, wind.z, wind.q) + np.interp(self.z, wave.z, wave.q)
        Peq = k_w*q_dyn
        hoop_stress = -Peq*r/t

        return hoop_stress


    def vonMisesStressMargin(self, axial_stress, hoop_stress, shear_stress):
        """combine stress for von Mises"""

        # von mises stress
        a = ((axial_stress + hoop_stress)/2.0)**2
        b = ((axial_stress - hoop_stress)/2.0)**2
        c = shear_stress**2
        von_mises = np.sqrt(a + 3.0*(b+c))

        # safety factor
        gamma = self.gamma_f * self.gamma_m * self.gamma_n

        # stress margin
        stress_margin = gamma * von_mises / self.sigma_y - 1

        return stress_margin


    def shellBucklingEurocode(self, axial_stress, hoop_stress, shear_stress):
        """default method to compute shell buckling using Eurocode method"""

        # buckling
        gamma_b = self.gamma_m * self.gamma_n
        zb, buckling = shellBuckling(self.z, self.d, self.t, 1, axial_stress, hoop_stress, shear_stress,
                                     self.z_reinforced, self.E, self.sigma_y, self.gamma_f, gamma_b)

        return zb, buckling


    def fatigue(self):
        """compute damage from provided damage equivalent moments"""

        # fatigue
        N_DEL = [365*24*3600*self.life]*len(self.z)
        M_DEL = np.interp(self.z, self.z_DEL, self.M_DEL)

        damage = fatigue(M_DEL, N_DEL, self.d, self.t, self.m_SN, self.DC, self.gamma_fatigue, stress_factor=1.0, weld_factor=True)

        return damage



class TowerWithpBEAM(TowerStruc):

    import _pBEAM


    def execute(self):

        _pBEAM = self._pBEAM
        z = self.z
        d = self.d
        t = self.t
        nodes = len(z)


        # aero/hydro loads
        Px, Py, Pz = self.aerohydroLoadsAtNodes()

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
        hoop_stress = self.hoopStressEurocode()

        # von mises stress
        self.stress = self.vonMisesStressMargin(axial_stress, hoop_stress, shear_stress)

        # buckling
        self.z_buckling, self.buckling = self.shellBucklingEurocode(axial_stress, hoop_stress, shear_stress)

        # fatigue
        self.damage = self.fatigue()



class TowerWithFrame3DD(TowerStruc):

    import frame3dd

    def execute(self):

        frame3dd = self.frame3dd

        # ------- node data ----------------
        n = len(self.z)
        node = np.arange(1, n+1)
        x = np.zeros(n)
        y = np.zeros(n)
        z = self.z
        r = np.zeros(n)

        nodes = frame3dd.NodeData(node, x, y, z, r)
        # -----------------------------------

        # ------ reaction data ------------

        # rigid base
        node = np.array([1])
        Rx = np.ones(1)
        Ry = np.ones(1)
        Rz = np.ones(1)
        Rxx = np.ones(1)
        Ryy = np.ones(1)
        Rzz = np.ones(1)

        reactions = frame3dd.ReactionData(node, Rx, Ry, Rz, Rxx, Ryy, Rzz)
        # -----------------------------------


        # ------ frame element data ------------
        d = (self.d[:-1] + self.d[1:])/2.0  # average for element with constant properties
        t = (self.t[:-1] + self.t[1:])/2.0  # average for element with constant properties
        ro = d/2.0 + t/2.0
        ri = d/2.0 - t/2.0

        element = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n+1)
        Ax = math.pi * (ro**2 - ri**2)
        Asy = Ax / (0.54414 + 2.97294*(ri/ro) - 1.51899*(ri/ro)**2)
        Asz = Ax / (0.54414 + 2.97294*(ri/ro) - 1.51899*(ri/ro)**2)
        Jx = math.pi/2.0 * (ro**4 - ri**4)
        Iy = Jx/2.0
        Iz = Jx/2.0
        E = self.E*np.ones(n-1)
        G = self.G*np.ones(n-1)
        roll = np.zeros(n-1)
        density = self.rho*np.ones(n-1)

        elements = frame3dd.ElementData(element, N1, N2, Ax, Asy, Asz, Jx, Iy, Iz, E, G, roll, density)

        # -----------------------------------


        # ------ other data ------------

        shear = 1               # 1: include shear deformation
        geom = 1                # 1: include geometric stiffness
        exagg_static = 1.0     # exaggerate mesh deformations
        dx = 10.0               # x-axis increment for internal forces

        other = frame3dd.OtherData(shear, geom, exagg_static, dx)

        # -----------------------------------

        # initialize frame3dd object
        tower = frame3dd.Frame(nodes, reactions, elements, other)


        # ------ static load case 1 ------------

        # gravity in the X, Y, Z, directions (global)
        gx = 0.0
        gy = 0.0
        gz = -self.g

        load = frame3dd.StaticLoadCase(gx, gy, gz)

        # tower top load

        nF = np.array([n])
        Fx = np.array([self.top_F[0]])
        Fy = np.array([self.top_F[1]])
        Fz = np.array([self.top_F[2]])
        Mxx = np.array([self.top_M[0]])
        Myy = np.array([self.top_M[1]])
        Mzz = np.array([self.top_M[2]])

        load.changePointLoads(nF, Fx, Fy, Fz, Mxx, Myy, Mzz)


        # aero/hydro loads
        Px, Py, Pz = self.aerohydroLoadsAtNodes()

        # switch to local c.s.
        Px, Py, Pz = Pz, Py, -Px

        # trapezoidally distributed loads
        EL = np.arange(1, n)
        xx1 = np.zeros(n-1)
        xx2 = z[1:] - z[:-1] - 1e-6
        wx1 = Px[:-1]
        wx2 = Px[1:]
        xy1 = np.zeros(n-1)
        xy2 = z[1:] - z[:-1] - 1e-6
        wy1 = Py[:-1]
        wy2 = Py[1:]
        xz1 = np.zeros(n-1)
        xz2 = z[1:] - z[:-1] - 1e-6
        wz1 = Pz[:-1]
        wz2 = Pz[1:]

        load.changeTrapezoidalLoads(EL, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)


        tower.addLoadCase(load)

        # -----------------------------------


        # ------ dyamic analysis data ------------

        nM = 3              # number of desired dynamic modes of vibration
        Mmethod = 1         # 1: subspace Jacobi     2: Stodola
        lump = 0            # 0: consistent mass ... 1: lumped mass matrix
        tol = 1e-9          # mode shape tolerance
        shift = 0.0         # shift value ... for unrestrained structures
        exagg_modal = 1.0  # exaggerate modal mesh deformations

        dynamic = frame3dd.DynamicAnalysis(nM, Mmethod, lump, tol, shift, exagg_modal)

        # extra node inertia data
        N = np.array([n])
        EMs = np.array([self.top_m])
        EMx = np.array([self.top_I[0]])
        EMy = np.array([self.top_I[1]])
        EMz = np.array([self.top_I[2]])
        EMxy = np.array([self.top_I[3]])
        EMxz = np.array([self.top_I[4]])
        EMyz = np.array([self.top_I[5]])
        rhox = np.array([self.top_cm[0]])
        rhoy = np.array([self.top_cm[1]])
        rhoz = np.array([self.top_cm[2]])

        dynamic.changeExtraInertia(N, EMs, EMx, EMy, EMz, EMxy, EMxz, EMyz, rhox, rhoy, rhoz)

        # extra frame element mass data
        # dynamic.changeExtraMass(EL, EMs)

        # set dynamic analysis
        tower.useDynamicAnalysis(dynamic)

        # ------------------------------------


        # run the analysis
        displacements, forces, reactions, internalForces, mass, modal = tower.run()
        iCase = 0

        # mass
        self.mass = mass.struct_mass

        # natural frequncies
        self.f1 = modal.freq[0]
        self.f2 = modal.freq[1]

        # deflections due to loading from tower top and wind/wave loads
        self.top_deflection = displacements.dx[iCase, n-1]  # in yaw-aligned direction


        # shear and bending (convert from local to global c.s.)
        Fz = forces.Nx[iCase, :]
        # Vy = forces.Vy[iCase, :]
        Vx = -forces.Vz[iCase, :]
        # Tzz = forces.Txx[iCase, :]
        Myy = forces.Myy[iCase, :]
        # Mxx = -forces.Mzz[iCase, :]


        # one per element (first negative b.c. need reaction)
        Fz = np.concatenate([[-Fz[0]], Fz[1::2]])
        Vx = np.concatenate([[-Vx[0]], Vx[1::2]])
        Myy = np.concatenate([[-Myy[0]], Myy[1::2]])


        # axial and shear stress (all stress evaluated on +x yaw side)
        A = math.pi * self.d * self.t
        Iyy = math.pi/8.0 * self.d**3 * self.t
        axial_stress = Fz/A - Myy/Iyy*self.d/2.0
        shear_stress = 2 * Vx / A

        # hoop_stress (Eurocode method)
        hoop_stress = self.hoopStressEurocode()

        # von mises stress
        self.stress = self.vonMisesStressMargin(axial_stress, hoop_stress, shear_stress)

        # buckling
        self.z_buckling, self.buckling = self.shellBucklingEurocode(axial_stress, hoop_stress, shear_stress)

        # fatigue
        self.damage = self.fatigue()





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
        self.connect('geometry.z_reinforced', 'tower.z_reinforced')
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
        self.create_passthrough('geometry.n_reinforced')

        self.create_passthrough('rna.nBlades')
        self.create_passthrough('rna.blade_mass')
        self.create_passthrough('rna.blade_I')
        self.create_passthrough('rna.hub_mass')
        self.create_passthrough('rna.hub_cm')
        self.create_passthrough('rna.hub_I')
        self.create_passthrough('rna.nac_mass')
        self.create_passthrough('rna.nac_cm')
        self.create_passthrough('rna.nac_I')


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
        self.create_passthrough('tower.z_DEL')
        self.create_passthrough('tower.M_DEL')


        self.create_passthrough('tower.mass')
        self.create_passthrough('tower.f1')
        self.create_passthrough('tower.f2')
        self.create_passthrough('tower.top_deflection')
        self.create_passthrough('tower.stress')
        self.create_passthrough('tower.z_buckling')
        self.create_passthrough('tower.buckling')
        self.create_passthrough('tower.damage')



if __name__ == '__main__':

# mass = 349486.79362
# f1 = 0.331531844442
# f2 = 0.334804545504
# f1 = 0.896151401523  # no top mass
# f2 = 0.896151401531
# top_deflection = 0.717842009951
# stress = [-0.57025437 -0.57071299 -0.57190392 -0.5739287  -0.5769025  -0.58095692
#  -0.5862422  -0.59292993 -0.60121621 -0.61132526 -0.62351375 -0.63807553
#  -0.65534699 -0.67571265 -0.69961022 -0.72753288 -0.76002246 -0.79763238
#  -0.8407783  -0.88903907 -0.93620526]
# z_buckling = [  0.   29.2  58.4]
# buckling = [-0.53537088 -0.57748296 -0.73160984]
# damage = [ 0.25643524  0.2473591   0.23628353  0.22239239  0.20746037  0.19081596
#   0.17398252  0.15775199  0.14202496  0.12645153  0.10980775  0.09283941
#   0.07611107  0.06010802  0.04563495  0.03308762  0.02282333  0.01566192
#   0.01157356  0.0107948   0.01400229]


    from commonse.environment import PowerWind, TowerSoil

    tower = Tower()
    tower.replace('wind', PowerWind())
    tower.replace('soil', TowerSoil())
    # tower.replace('tower', TowerWithpBEAM())
    tower.replace('tower', TowerWithFrame3DD())

    # geometry
    towerHt = 87.6
    tower.z = towerHt*np.array([0.0, 0.5, 1.0])
    tower.d = [6.0, 4.935, 3.87]
    tower.t = [0.027*1.3, 0.023*1.3, 0.019*1.3]
    tower.n = [10, 10]
    tower.n_reinforced = 3
    tower.yaw = 0.0

    # top mass
    # tower.top_m = 359082.653015
    # tower.top_I = [2960437.0, 3253223.0, 3264220.0, 0.0, -18400.0, 0.0]
    # tower.top_cm = [-1.9, 0.0, 1.75]
    # tower.top_F = [1478579.28056464, 0., -3522600.82607833]
    # tower.top_M = [10318177.27285694, 0., 0.]

    # blades (optimized)
    nBlades = 3
    tower.nBlades = nBlades
    tower.blade_mass = 15241.323
    bladeI = 8791992.000 * nBlades
    tower.blade_I = np.array([bladeI, bladeI/2.0, bladeI/2.0, 0.0, 0.0, 0.0])

    # hub (optimized)
    tower.hub_mass = 50421.4
    tower.hub_cm = np.array([-6.30, 0, 3.15])
    tower.hub_I = np.array([127297.8, 127297.8, 127297.8, 0.0, 0.0, 0.0])

    # nacelle (optimized)
    tower.nac_mass = 221245.8
    tower.nac_cm = np.array([-0.32, 0, 2.40])
    tower.nac_I = np.array([9908302.58, 912488.28, 1160903.54, 0.0, 0.0, 0.0])

    # max Thrust case
    F1 = np.array([1.3295e6, -2.2694e4, -4.6184e6])
    M1 = np.array([6.2829e6, -1.0477e6, 3.9029e6])
    V1 = 16.030

    tower.top_F = F1
    tower.top_M = M1


    # damage
    tower.z_DEL = np.array([0.000, 1.327, 3.982, 6.636, 9.291, 11.945, 14.600, 17.255, 19.909, 22.564, 25.218, 27.873, 30.527, 33.182, 35.836, 38.491, 41.145, 43.800, 46.455, 49.109, 51.764, 54.418, 57.073, 59.727, 62.382, 65.036, 67.691, 70.345, 73.000, 75.655, 78.309, 80.964, 83.618, 86.273, 87.600])
    tower.M_DEL = 1e3*np.array([8.2940E+003, 8.1518E+003, 7.8831E+003, 7.6099E+003, 7.3359E+003, 7.0577E+003, 6.7821E+003, 6.5119E+003, 6.2391E+003, 5.9707E+003, 5.7070E+003, 5.4500E+003, 5.2015E+003, 4.9588E+003, 4.7202E+003, 4.4884E+003, 4.2577E+003, 4.0246E+003, 3.7942E+003, 3.5664E+003, 3.3406E+003, 3.1184E+003, 2.8977E+003, 2.6811E+003, 2.4719E+003, 2.2663E+003, 2.0673E+003, 1.8769E+003, 1.7017E+003, 1.5479E+003, 1.4207E+003, 1.3304E+003, 1.2780E+003, 1.2673E+003, 1.2761E+003])


    # wind
    tower.wind.Uref = V1
    tower.wind.zref = towerHt
    tower.wind.z0 = 0.0
    tower.wind.shearExp = 0.2


    # soil
    tower.soil.rigid = 6*[True]


    tower.run()


    print 'mass =', tower.mass
    print 'f1 =', tower.f1
    print 'f2 =', tower.f2
    print 'top_deflection =', tower.top_deflection
    print 'stress =', tower.stress
    print 'z_buckling =', tower.z_buckling
    print 'buckling =', tower.buckling
    print 'damage =', tower.damage

