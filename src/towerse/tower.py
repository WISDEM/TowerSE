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
from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Slot

from commonse.utilities import sind, cosd, linspace_with_deriv, interp_with_deriv
from commonse.csystem import DirectionVector
from commonse.environment import WindBase, WaveBase, SoilBase
from towerSupplement import shellBuckling, fatigue
from akima import Akima


# "Experiments on the Flow Past a Circular Cylinder at Very High Reynolds Numbers", Roshko
Re_pt = [0.00001, 0.0001, 0.0010, 0.0100, 0.0200, 0.1220, 0.2000, 0.3000, 0.4000,
         0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 5.0000, 10.0000]
cd_pt = [4.0000,  2.0000, 1.1100, 1.1100, 1.2000, 1.2000, 1.1700, 0.9000, 0.5400,
         0.3100, 0.3800, 0.4600, 0.5300, 0.5700, 0.6100, 0.6400, 0.6700, 0.7000, 0.7000]

drag_spline = Akima(np.log10(Re_pt), cd_pt, delta_x=0.0)


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

    ReN = Re / 1.0e6

    cd = np.zeros_like(Re)
    dcd_dRe = np.zeros_like(Re)
    idx = ReN > 0
    cd[idx], dcd_dRe[idx] = drag_spline.interp(np.log10(ReN[idx]))
    dcd_dRe[idx] /= (Re[idx]*math.log(10))  # chain rule

    return cd, dcd_dRe




# -----------------
#  Variable Trees
# -----------------

class AeroLoads(VariableTree):
    """wind/wave loads"""

    Px = Array(units='N/m', desc='distributed loads')
    Py = Array(units='N/m', desc='distributed loads')
    Pz = Array(units='N/m', desc='distributed loads')
    q = Array(units='N/m**2', desc='dynamic pressure')
    z = Array(units='m', desc='corresponding heights')
    beta = Array(units='deg', desc='wind/wave angle relative to inertia c.s.')


# -----------------
#  Components
# -----------------


class TowerWindDrag(Component):
    """drag forces on a cylindrical tower due to wind"""

    # TODO: add required=True back into these Arrays. openmdao bug.  Also in wave
    # variables
    U = Array(iotype='in', units='m/s', desc='magnitude of wind speed')
    z = Array(iotype='in', units='m', desc='heights where wind speed was computed')
    d = Array(iotype='in', units='m', desc='corresponding diameter of cylinder section')

    # parameters
    beta = Array(iotype='in', units='deg', desc='corresponding wind angles relative to inertial coordinate system')
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='air density')
    mu = Float(1.7934e-5, iotype='in', units='kg/(m*s)', desc='dynamic viscosity of air')

    # out
    windLoads = VarTree(AeroLoads(), iotype='out', desc='wind loads in inertial coordinate system')


    def execute(self):

        rho = self.rho
        U = self.U
        d = self.d
        mu = self.mu
        beta = self.beta

        # dynamic pressure
        q = 0.5*rho*U**2

        # Reynolds number and drag
        Re = rho*U*d/mu
        cd, dcd_dRe = cylinderDrag(Re)
        Fp = q*cd*d

        # components of distributed loads
        Px = Fp*cosd(beta)
        Py = Fp*sind(beta)
        Pz = 0*Fp

        # pack data
        self.windLoads.Px = Px
        self.windLoads.Py = Py
        self.windLoads.Pz = Pz
        self.windLoads.q = q
        self.windLoads.z = self.z
        self.windLoads.beta = beta

        # derivatives
        self.dq_dU = rho*U
        const = (self.dq_dU*cd + q*dcd_dRe*rho*d/mu)*d
        self.dPx_dU = const*cosd(beta)
        self.dPy_dU = const*sind(beta)

        const = (cd + dcd_dRe*Re)*q
        self.dPx_dd = const*cosd(beta)
        self.dPy_dd = const*sind(beta)


    def linearize(self):

        n = len(self.z)

        zeron = np.zeros((n, n))

        dPx = np.hstack([np.diag(self.dPx_dU), zeron, np.diag(self.dPx_dd)])
        dPy = np.hstack([np.diag(self.dPy_dU), zeron, np.diag(self.dPy_dd)])
        dPz = np.zeros((n, 3*n))
        dq = np.hstack([np.diag(self.dq_dU), np.zeros((n, 2*n))])
        dz = np.hstack([zeron, np.eye(n), zeron])

        self.J = np.vstack([dPx, dPy, dPz, dq, dz])


    def provideJ(self):

        inputs = ('U', 'z', 'd')
        outputs = ('windLoads.Px', 'windLoads.Py', 'windLoads.Pz', 'windLoads.q', 'windLoads.z')

        return inputs, outputs, self.J



class TowerWaveDrag(Component):
    """drag forces on a cylindrical tower due to waves"""

    # variables
    U = Array(iotype='in', units='m/s', desc='magnitude of wave speed')
    A = Array(iotype='in', units='m/s**2', desc='magnitude of wave acceleration')
    z = Array(iotype='in', units='m', desc='heights where wave speed was computed')
    d = Array(iotype='in', units='m', desc='corresponding diameter of cylinder section')

    # parameters
    beta = Array(iotype='in', units='deg', desc='corresponding wave angles relative to inertial coordinate system')
    rho = Float(1027.0, iotype='in', units='kg/m**3', desc='water density')
    mu = Float(1.3351e-3, iotype='in', units='kg/(m*s)', desc='dynamic viscosity of water')
    cm = Float(2.0, iotype='in', desc='mass coefficient')

    # out
    waveLoads = VarTree(AeroLoads(), iotype='out', desc='wave loads in inertial coordinate system')


    def execute(self):

        rho = self.rho
        U = self.U
        d = self.d
        mu = self.mu
        beta = self.beta

        # dynamic pressure
        q = 0.5*rho*U**2

        # Reynolds number and drag
        Re = rho*U*d/mu
        cd, dcd_dRe = cylinderDrag(Re)

        # inertial and drag forces
        Fi = rho*self.cm*math.pi/4.0*d**2*self.A  # Morrison's equation
        Fd = q*cd*d
        Fp = Fi + Fd

        # components of distributed loads
        Px = Fp*cosd(beta)
        Py = Fp*sind(beta)
        Pz = 0*Fp

        # pack data
        self.waveLoads.Px = Px
        self.waveLoads.Py = Py
        self.waveLoads.Pz = Pz
        self.waveLoads.q = q
        self.waveLoads.z = self.z
        self.waveLoads.beta = beta

        # derivatives
        self.dq_dU = rho*U
        const = (self.dq_dU*cd + q*dcd_dRe*rho*d/mu)*d
        self.dPx_dU = const*cosd(beta)
        self.dPy_dU = const*sind(beta)

        const = (cd + dcd_dRe*Re)*q + rho*self.cm*math.pi/4.0*2*d*self.A
        self.dPx_dd = const*cosd(beta)
        self.dPy_dd = const*sind(beta)

        const = rho*self.cm*math.pi/4.0*d**2
        self.dPx_dA = const*cosd(beta)
        self.dPy_dA = const*sind(beta)


    def linearize(self):

        n = len(self.z)

        zeron = np.zeros((n, n))

        dPx = np.hstack([np.diag(self.dPx_dU), np.diag(self.dPx_dA), zeron, np.diag(self.dPx_dd)])
        dPy = np.hstack([np.diag(self.dPy_dU), np.diag(self.dPy_dA), zeron, np.diag(self.dPy_dd)])
        dPz = np.zeros((n, 4*n))
        dq = np.hstack([np.diag(self.dq_dU), np.zeros((n, 3*n))])
        dz = np.hstack([zeron, zeron, np.eye(n), zeron])

        self.J = np.vstack([dPx, dPy, dPz, dq, dz])


    def provideJ(self):

        inputs = ('U', 'A', 'z', 'd')
        outputs = ('waveLoads.Px', 'waveLoads.Py', 'waveLoads.Pz', 'waveLoads.q', 'waveLoads.z')

        return inputs, outputs, self.J





class TowerDiscretization(Component):
    """discretize geometry into finite element nodes"""

    # in
    towerHeight = Float(iotype='in', units='m')
    z = Array(iotype='in', desc='locations along unit tower, linear lofting between')
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

        self.z *= self.towerHeight  # TODO: fix gradients

        n1 = sum(self.n) + 1
        n2 = len(self.z)
        self.dznode_dz = np.zeros((n1, n2))

        # compute nodal locations (and gradients)
        self.z_node = np.array([self.z[0]])

        nlast = 0
        for i in range(len(self.n)):
            znode, dznode_dzi, dznode_dzip = linspace_with_deriv(self.z[i], self.z[i+1], self.n[i]+1)
            self.z_node = np.r_[self.z_node, znode[1:]]

            # gradients
            self.dznode_dz[nlast:nlast+self.n[i]+1, i] = dznode_dzi
            self.dznode_dz[nlast:nlast+self.n[i]+1, i+1] = dznode_dzip
            nlast += self.n[i]


        # interpolate (and gradients)
        self.d_node = np.interp(self.z_node, self.z, self.d)
        self.t_node = np.interp(self.z_node, self.z, self.t)

        self.d_node, ddnode_dznode, ddnode_dz, self.ddnode_dd = interp_with_deriv(self.z_node, self.z, self.d)
        self.t_node, dtnode_dznode, dtnode_dz, self.dtnode_dt = interp_with_deriv(self.z_node, self.z, self.t)

        # chain rule
        self.ddnode_dz = ddnode_dz + np.dot(ddnode_dznode, self.dznode_dz)
        self.dtnode_dz = dtnode_dz + np.dot(dtnode_dznode, self.dznode_dz)


        # reinforcement distances
        self.z_reinforced, dzr_dz0, dzr_dzend = linspace_with_deriv(self.z[0], self.z[-1], self.n_reinforced+1)
        self.dzr_dz = np.zeros((len(self.z_reinforced), n2))
        self.dzr_dz[:, 0] = dzr_dz0
        self.dzr_dz[:, -1] = dzr_dzend



    def linearize(self):

        n = len(self.z_node)
        m = len(self.z)
        n2 = len(self.z_reinforced)

        dzn = np.hstack([self.dznode_dz, np.zeros((n, 2*m))])
        ddn = np.hstack([self.ddnode_dz, self.ddnode_dd, np.zeros((n, m))])
        dtn = np.hstack([self.dtnode_dz, np.zeros((n, m)), self.dtnode_dt])
        dzr = np.hstack([self.dzr_dz, np.zeros((n2, 2*m))])

        self.J = np.vstack([dzn, ddn, dtn, dzr])


    def provideJ(self):

        inputs = ('z', 'd', 't')
        outputs = ('z_node', 'd_node', 't_node', 'z_reinforced')

        return inputs, outputs, self.J


class RNAMass(Component):

    # variables
    blades_mass = Float(iotype='in', units='kg', desc='mass of all blade')
    hub_mass = Float(iotype='in', units='kg', desc='mass of hub')
    nac_mass = Float(iotype='in', units='kg', desc='mass of nacelle')

    hub_cm = Array(iotype='in', units='m', desc='location of hub center of mass relative to tower top in yaw-aligned c.s.')
    nac_cm = Array(iotype='in', units='m', desc='location of nacelle center of mass relative to tower top in yaw-aligned c.s.')

    # TODO: check on this???
    # order for all moments of inertia is (xx, yy, zz, xy, xz, yz) in the yaw-aligned coorinate system
    blades_I = Array(iotype='in', units='kg*m**2', desc='mass moments of inertia of all blades about hub center')
    hub_I = Array(iotype='in', units='kg*m**2', desc='mass moments of inertia of hub about its center of mass')
    nac_I = Array(iotype='in', units='kg*m**2', desc='mass moments of inertia of nacelle about its center of mass')

    # TODO: fix gradients

    # outputs
    rna_mass = Float(iotype='out', units='kg', desc='total mass of RNA')
    rna_cm = Array(iotype='out', units='m', desc='location of RNA center of mass relative to tower top in yaw-aligned c.s.')
    rna_I_TT = Array(iotype='out', units='kg*m**2', desc='mass moments of inertia of RNA about tower top in yaw-aligned coordinate system')


    def _assembleI(self, Ixx, Iyy, Izz, Ixy, Ixz, Iyz):
        return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


    def _unassembleI(self, I):
        return np.array([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])


    def execute(self):

        self.rotor_mass = self.blades_mass + self.hub_mass
        self.nac_mass = self.nac_mass

        # rna mass
        self.rna_mass = self.rotor_mass + self.nac_mass

        # rna cm
        self.rna_cm = (self.rotor_mass*self.hub_cm + self.nac_mass*self.nac_cm)/self.rna_mass

        # rna I
        blades_I = self._assembleI(*self.blades_I)
        hub_I = self._assembleI(*self.hub_I)
        nac_I = self._assembleI(*self.nac_I)
        rotor_I = blades_I + hub_I

        R = self.hub_cm
        rotor_I_TT = rotor_I + self.rotor_mass*(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        R = self.nac_cm
        nac_I_TT = nac_I + self.nac_mass*(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        self.rna_I_TT = self._unassembleI(rotor_I_TT + nac_I_TT)


    def linearize(self):

        # mass
        dmass = np.hstack([np.array([self.nBlades, 1.0, 1.0]), np.zeros(2*3+3*6)])

        # cm
        top = (self.rotor_mass*self.hub_cm + self.nac_mass*self.nac_cm)
        dcm_dblademass = (self.rna_mass*self.nBlades*self.hub_cm - top*self.nBlades)/self.rna_mass**2
        dcm_dhubmass = (self.rna_mass*self.hub_cm - top)/self.rna_mass**2
        dcm_dnacmass = (self.rna_mass*self.nac_cm - top)/self.rna_mass**2
        dcm_dhubcm = self.rotor_mass/self.rna_mass*np.eye(3)
        dcm_dnaccm = self.nac_mass/self.rna_mass*np.eye(3)

        dcm = np.hstack([dcm_dblademass[:, np.newaxis], dcm_dhubmass[:, np.newaxis],
            dcm_dnacmass[:, np.newaxis], dcm_dhubcm, dcm_dnaccm, np.zeros((3, 3*6))])

        # I
        R = self.hub_cm
        const = self._unassembleI(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        const = const[:, np.newaxis]
        dI_dblademass = self.nBlades*const
        dI_dhubmass = const
        dI_drx = self.rotor_mass*self._unassembleI(2*R[0]*np.eye(3) - np.array([[2*R[0], R[1], R[2]], [R[1], 0.0, 0.0], [R[2], 0.0, 0.0]]))
        dI_dry = self.rotor_mass*self._unassembleI(2*R[1]*np.eye(3) - np.array([[0.0, R[0], 0.0], [R[0], 2*R[1], R[2]], [0.0, R[2], 0.0]]))
        dI_drz = self.rotor_mass*self._unassembleI(2*R[2]*np.eye(3) - np.array([[0.0, 0.0, R[0]], [0.0, 0.0, R[1]], [R[0], R[1], 2*R[2]]]))
        dI_dhubcm = np.vstack([dI_drx, dI_dry, dI_drz]).T

        R = self.nac_cm
        const = self._unassembleI(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        const = const[:, np.newaxis]
        dI_dnacmass = const
        dI_drx = self.nac_mass*self._unassembleI(2*R[0]*np.eye(3) - np.array([[2*R[0], R[1], R[2]], [R[1], 0.0, 0.0], [R[2], 0.0, 0.0]]))
        dI_dry = self.nac_mass*self._unassembleI(2*R[1]*np.eye(3) - np.array([[0.0, R[0], 0.0], [R[0], 2*R[1], R[2]], [0.0, R[2], 0.0]]))
        dI_drz = self.nac_mass*self._unassembleI(2*R[2]*np.eye(3) - np.array([[0.0, 0.0, R[0]], [0.0, 0.0, R[1]], [R[0], R[1], 2*R[2]]]))
        dI_dnaccm = np.vstack([dI_drx, dI_dry, dI_drz]).T

        dI_dbladeI = np.eye(6)
        dI_dhubI = np.eye(6)
        dI_dnacI = np.eye(6)

        dI = np.hstack([dI_dblademass, dI_dhubmass, dI_dnacmass, dI_dhubcm, dI_dnaccm,
            dI_dbladeI, dI_dhubI, dI_dnacI])

        self.J = np.vstack([dmass, dcm, dI])



    def provideJ(self):

        inputs = ('blade_mass', 'hub_mass', 'nac_mass', 'hub_cm', 'nac_cm', 'blade_I', 'hub_I', 'nac_I')
        outputs = ('rna_mass', 'rna_cm', 'rna_I_TT')

        return inputs, outputs, self.J



class RotorLoads(Component):

    T = Float(iotype='in', desc='thrust in hub-aligned coordinate system')
    Q = Float(iotype='in', desc='torque in hub-aligned coordinate system')
    r_hub = Array(iotype='in', desc='position of rotor hub relative to tower top in yaw-aligned c.s.')
    tilt = Float(iotype='in', units='deg')
    g = Float(9.81, iotype='in', units='m/s**2')
    m_RNA = Float(iotype='in', units='kg')

    top_F = Array(iotype='out')  # in yaw-aligned
    top_M = Array(iotype='out')

    def execute(self):

        F = DirectionVector(self.T, 0.0, 0.0).hubToYaw(self.tilt)
        M = DirectionVector(self.Q, 0.0, 0.0).hubToYaw(self.tilt)

        F.z -= self.m_RNA*self.g

        r = DirectionVector(self.r_hub[0], self.r_hub[1], self.r_hub[2])
        M = M - r.cross(F)

        self.top_F = np.array([F.x, F.y, F.z])
        self.top_M = np.array([M.x, M.y, M.z])




class TowerBase(Component):
    """structural analysis of cylindrical tower

    all forces, moments, distances should be given (and returned) in the yaw-aligned coordinate system
    """

    # geometry
    towerHeight = Float(iotype='in', units='m')
    z = Array(iotype='in', desc='locations along unit tower')
    d = Array(iotype='in', units='m', desc='tower diameter at corresponding locations')
    t = Array(iotype='in', units='m', desc='shell thickness at corresponding locations')
    z_reinforced = Array(iotype='in', units='m', desc='reinforcement positions for buckling')
    yaw = Float(0.0, iotype='in', units='deg')

    # wind/wave loads
    windLoads = VarTree(AeroLoads(), iotype='in')
    waveLoads = VarTree(AeroLoads(), iotype='in')
    g = Float(9.81, iotype='in', units='m/s**2')

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

    life = Float(20.0, iotype='in', desc='fatigue life of tower')
    m_SN = Int(4, iotype='in', desc='slope of S/N curve')
    DC = Float(80.0, iotype='in', desc='standard value of stress')
    gamma_fatigue = Float(1.755, iotype='in', desc='total safety factor for fatigue')
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
        windLoads = DirectionVector(wind.Px, wind.Py, wind.Pz).inertialToWind(betaMain).windToYaw(self.yaw)
        waveLoads = DirectionVector(wave.Px, wave.Py, wave.Pz).inertialToWind(betaMain).windToYaw(self.yaw)

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



class TowerWithpBEAM(TowerBase):


    def __init__(self):
        import _pBEAM  # import only if instantiated so that Frame3DD is not required for all users
        self._pBEAM = _pBEAM
        super(TowerWithpBEAM, self).__init__()


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



class TowerWithFrame3DD(TowerBase):


    def __init__(self):
        import frame3dd  # import only if instantiated so that Frame3DD is not required for all users
        self.frame3dd = frame3dd
        super(TowerWithFrame3DD, self).__init__()


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
        Kx = np.array(self.k_soil[0])
        Ky = np.array(self.k_soil[2])
        Kz = np.array(self.k_soil[4])
        Ktx = np.array(self.k_soil[1])
        Kty = np.array(self.k_soil[3])
        Ktz = np.array(self.k_soil[5])
        rigid = float('inf')

        reactions = frame3dd.ReactionData(node, Kx, Ky, Kz, Ktx, Kty, Ktz, rigid)
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


        # ------ options ------------

        shear = True        # 1: include shear deformation
        geom = False        # 1: include geometric stiffness
        dx = 10.0           # x-axis increment for internal forces
        options = frame3dd.Options(shear, geom, dx)

        # -----------------------------------

        # initialize frame3dd object
        tower = frame3dd.Frame(nodes, reactions, elements, options)


        # ------ add extra mass ------------

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
        addGravityLoad = False

        tower.changeExtraNodeMass(N, EMs, EMx, EMy, EMz, EMxy, EMxz, EMyz, rhox, rhoy, rhoz, addGravityLoad)
        # ------------------------------------

        # ------- enable dynamic analysis ----------

        nM = 2              # number of desired dynamic modes of vibration (below only necessary if nM > 0)
        Mmethod = 1         # 1: subspace Jacobi     2: Stodola
        lump = 0            # 0: consistent mass ... 1: lumped mass matrix
        tol = 1e-9          # mode shape tolerance
        shift = 0.0         # shift value ... for unrestrained structures
        tower.enableDynamics(nM, Mmethod, lump, tol, shift)

        # ----------------------------


        # ------ static load case 1 ------------

        # gravity in the X, Y, Z, directions (global)
        gx = 0.0
        gy = 0.0
        gz = -self.g

        load = frame3dd.StaticLoadCase(gx, gy, gz)

        # tower top load  (TODO: remove RNA weight from these forces since Frame3DD accounts for it)

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


class TowerSE(Assembly):

    # geometry
    towerHeight = Float(iotype='in', units='m')
    z = Array(iotype='in', desc='locations along unit tower, linear lofting between')
    d = Array(iotype='in', units='m', desc='tower diameter at corresponding locations')
    t = Array(iotype='in', units='m', desc='shell thickness at corresponding locations')
    n = Array(iotype='in', dtype=np.int, desc='number of finite elements between sections.  array length should be ``len(z)-1``')
    n_reinforced = Int(iotype='in', desc='must be a minimum of 1 (top and bottom)')
    yaw = Float(0.0, iotype='in', units='deg')
    tilt = Float(0.0, iotype='in', units='deg')

    # environment
    wind_rho = Float(1.225, iotype='in', units='kg/m**3', desc='air density')
    wind_mu = Float(1.7934e-5, iotype='in', units='kg/(m*s)', desc='dynamic viscosity of air')

    wind_Uref1 = Float(iotype='in', units='m/s', desc='reference wind speed (usually at hub height)')
    wind_Uref2 = Float(iotype='in', units='m/s', desc='reference wind speed (usually at hub height)')
    wind_zref = Float(iotype='in', units='m', desc='corresponding reference height')
    wind_z0 = Float(0.0, iotype='in', units='m', desc='bottom of wind profile (height of ground/sea)')


    wave_rho = Float(1027.0, iotype='in', units='kg/m**3', desc='water density')
    wave_mu = Float(1.3351e-3, iotype='in', units='kg/(m*s)', desc='dynamic viscosity of water')
    wave_cm = Float(2.0, iotype='in', desc='mass coefficient')

    g = Float(9.81, iotype='in', units='m/s**2')

    # rotor loads
    rotorT1 = Float(iotype='in', desc='thrust in hub-aligned coordinate system')
    rotorQ1 = Float(iotype='in', desc='torque in hub-aligned coordinate system')
    rotorT2 = Float(iotype='in', desc='thrust in hub-aligned coordinate system')
    rotorQ2 = Float(iotype='in', desc='torque in hub-aligned coordinate system')

    # RNA mass properties
    blades_mass = Float(iotype='in', units='kg', desc='mass of all blade')
    hub_mass = Float(iotype='in', units='kg', desc='mass of hub')
    nac_mass = Float(iotype='in', units='kg', desc='mass of nacelle')

    hub_cm = Array(iotype='in', units='m', desc='location of hub center of mass relative to tower top in yaw-aligned c.s.')
    nac_cm = Array(iotype='in', units='m', desc='location of nacelle center of mass relative to tower top in yaw-aligned c.s.')

    blades_I = Array(iotype='in', units='kg*m**2', desc='mass moments of inertia of all blades about hub center')
    hub_I = Array(iotype='in', units='kg*m**2', desc='mass moments of inertia of hub about its center of mass')
    nac_I = Array(iotype='in', units='kg*m**2', desc='mass moments of inertia of nacelle about its center of mass')

    # material properties
    E = Float(210e9, iotype='in', units='N/m**2', desc='material modulus of elasticity')
    G = Float(80.8e9, iotype='in', units='N/m**2', desc='material shear modulus')
    rho = Float(8500.0, iotype='in', units='kg/m**3', desc='material density')
    sigma_y = Float(450.0e6, iotype='in', units='N/m**2', desc='yield stress')

    # safety factors
    gamma_f = Float(1.35, iotype='in', desc='safety factor on loads')
    gamma_m = Float(1.1, iotype='in', desc='safety factor on materials')
    gamma_n = Float(1.0, iotype='in', desc='safety factor on consequence of failure')

    # fatigue parameters
    life = Float(20.0, iotype='in', desc='fatigue life of tower')
    m_SN = Int(4, iotype='in', desc='slope of S/N curve')
    DC = Float(80.0, iotype='in', desc='standard value of stress')
    gamma_fatigue = Float(1.755, iotype='in', desc='total safety factor for fatigue')
    z_DEL = Array(iotype='in')
    M_DEL = Array(iotype='in')

    # replace
    wind1 = Slot(WindBase)
    wind2 = Slot(WindBase)
    wave1 = Slot(WaveBase)
    wave2 = Slot(WaveBase)
    soil = Slot(SoilBase)
    tower1 = Slot(TowerBase)
    tower2 = Slot(TowerBase)

    # outputs
    mass = Float(iotype='out', units='kg')
    f1 = Float(iotype='out', units='Hz', desc='first natural frequency')
    f2 = Float(iotype='out', units='Hz', desc='second natural frequency')
    top_deflection1 = Float(iotype='out', units='m', desc='deflection of tower top in yaw-aligned +x direction')
    top_deflection2 = Float(iotype='out', units='m', desc='deflection of tower top in yaw-aligned +x direction')
    z_nodes = Array(iotype='out', units='m')
    stress1 = Array(iotype='out', units='N/m**2', desc='von Mises stress along tower on downwind side (yaw-aligned +x).  normalized by yield stress.  includes safety factors.')
    stress2 = Array(iotype='out', units='N/m**2', desc='von Mises stress along tower on downwind side (yaw-aligned +x).  normalized by yield stress.  includes safety factors.')
    z_buckling = Array(iotype='out', units='m', desc='z-locations along tower where shell buckling is evaluted')
    buckling1 = Array(iotype='out', desc='a shell buckling constraint.  should be <= 0 for feasibility.  includes safety factors')
    buckling2 = Array(iotype='out', desc='a shell buckling constraint.  should be <= 0 for feasibility.  includes safety factors')
    damage = Array(iotype='out', desc='fatigue damage at each tower section')

    def configure(self):

        self.add('geometry', TowerDiscretization())
        self.add('wind1', WindBase())
        self.add('wind2', WindBase())
        self.add('wave1', WaveBase())
        self.add('wave2', WaveBase())
        self.add('windLoads1', TowerWindDrag())
        self.add('windLoads2', TowerWindDrag())
        self.add('waveLoads1', TowerWaveDrag())
        self.add('waveLoads2', TowerWaveDrag())
        self.add('soil', SoilBase())
        self.add('rna', RNAMass())
        self.add('rotorloads1', RotorLoads())
        self.add('rotorloads2', RotorLoads())
        self.add('tower1', TowerBase())
        self.add('tower2', TowerBase())

        # self.driver.workflow.add(['geometry', 'wind', 'wave', 'windLoads', 'waveLoads', 'soil', 'rna', 'rotorloads', 'tower'])
        self.driver.workflow.add(['geometry', 'wind1', 'wind2', 'wave1', 'wave2',
            'windLoads1', 'windLoads2', 'waveLoads1', 'waveLoads2', 'soil', 'rna',
            'rotorloads1', 'rotorloads2', 'tower1', 'tower2'])
        # TODO: probably better to do this with a driver or something rather than manually setting 2 cases

        # connections to geometry
        self.connect('towerHeight', 'geometry.towerHeight')
        self.connect('z', 'geometry.z')
        self.connect('d', 'geometry.d')
        self.connect('t', 'geometry.t')
        self.connect('n', 'geometry.n')
        self.connect('n_reinforced', 'geometry.n_reinforced')


        # connections to wind1
        self.connect('geometry.z_node', 'wind1.z')
        self.connect('wind_Uref1', 'wind1.Uref')
        self.connect('wind_zref', 'wind1.zref')
        self.connect('wind_z0', 'wind1.z0')

        # connections to wind2
        self.connect('geometry.z_node', 'wind2.z')
        self.connect('wind_Uref2', 'wind2.Uref')
        self.connect('wind_zref', 'wind2.zref')
        self.connect('wind_z0', 'wind2.z0')

        # connections to wave1
        self.connect('geometry.z_node', 'wave1.z')

        # connections to wave2
        self.connect('geometry.z_node', 'wave2.z')

        # connections to windLoads1
        self.connect('wind1.U', 'windLoads1.U')
        self.connect('wind1.beta', 'windLoads1.beta')
        self.connect('wind_rho', 'windLoads1.rho')
        self.connect('wind_mu', 'windLoads1.mu')
        self.connect('geometry.z_node', 'windLoads1.z')
        self.connect('geometry.d_node', 'windLoads1.d')

        # connections to windLoads2
        self.connect('wind2.U', 'windLoads2.U')
        self.connect('wind2.beta', 'windLoads2.beta')
        self.connect('wind_rho', 'windLoads2.rho')
        self.connect('wind_mu', 'windLoads2.mu')
        self.connect('geometry.z_node', 'windLoads2.z')
        self.connect('geometry.d_node', 'windLoads2.d')

        # connections to waveLoads1
        self.connect('wave1.U', 'waveLoads1.U')
        self.connect('wave1.A', 'waveLoads1.A')
        self.connect('wave1.beta', 'waveLoads1.beta')
        self.connect('wave_rho', 'waveLoads1.rho')
        self.connect('wave_mu', 'waveLoads1.mu')
        self.connect('wave_cm', 'waveLoads1.cm')
        self.connect('geometry.z_node', 'waveLoads1.z')
        self.connect('geometry.d_node', 'waveLoads1.d')

        # connections to waveLoads2
        self.connect('wave2.U', 'waveLoads2.U')
        self.connect('wave2.A', 'waveLoads2.A')
        self.connect('wave2.beta', 'waveLoads2.beta')
        self.connect('wave_rho', 'waveLoads2.rho')
        self.connect('wave_mu', 'waveLoads2.mu')
        self.connect('wave_cm', 'waveLoads2.cm')
        self.connect('geometry.z_node', 'waveLoads2.z')
        self.connect('geometry.d_node', 'waveLoads2.d')

        # connections to rna
        self.connect('blades_mass', 'rna.blades_mass')
        self.connect('blades_I', 'rna.blades_I')
        self.connect('hub_mass', 'rna.hub_mass')
        self.connect('hub_cm', 'rna.hub_cm')
        self.connect('hub_I', 'rna.hub_I')
        self.connect('nac_mass', 'rna.nac_mass')
        self.connect('nac_cm', 'rna.nac_cm')
        self.connect('nac_I', 'rna.nac_I')

        # connections to rotorloads1
        self.connect('rotorT1', 'rotorloads1.T')
        self.connect('rotorQ1', 'rotorloads1.Q')
        self.connect('hub_cm', 'rotorloads1.r_hub')
        self.connect('tilt', 'rotorloads1.tilt')
        self.connect('g', 'rotorloads1.g')
        self.connect('rna.rna_mass', 'rotorloads1.m_RNA')

        # connections to rotorloads2
        self.connect('rotorT2', 'rotorloads2.T')
        self.connect('rotorQ2', 'rotorloads2.Q')
        self.connect('hub_cm', 'rotorloads2.r_hub')
        self.connect('tilt', 'rotorloads2.tilt')
        self.connect('g', 'rotorloads2.g')
        self.connect('rna.rna_mass', 'rotorloads2.m_RNA')

        # connections to tower
        self.connect('geometry.z_node', 'tower1.z')
        self.connect('geometry.d_node', 'tower1.d')
        self.connect('geometry.t_node', 'tower1.t')
        self.connect('geometry.z_reinforced', 'tower1.z_reinforced')
        self.connect('windLoads1.windLoads', 'tower1.windLoads')
        self.connect('waveLoads1.waveLoads', 'tower1.waveLoads')
        self.connect('soil.k', 'tower1.k_soil')
        self.connect('rna.rna_mass', 'tower1.top_m')
        self.connect('rna.rna_cm', 'tower1.top_cm')
        self.connect('rna.rna_I_TT', 'tower1.top_I')
        self.connect('rotorloads1.top_F', 'tower1.top_F')
        self.connect('rotorloads1.top_M', 'tower1.top_M')
        self.connect('yaw', 'tower1.yaw')
        self.connect('g', 'tower1.g')
        self.connect('E', 'tower1.E')
        self.connect('G', 'tower1.G')
        self.connect('rho', 'tower1.rho')
        self.connect('sigma_y', 'tower1.sigma_y')
        self.connect('gamma_f', 'tower1.gamma_f')
        self.connect('gamma_m', 'tower1.gamma_m')
        self.connect('gamma_n', 'tower1.gamma_n')
        self.connect('life', 'tower1.life')
        self.connect('m_SN', 'tower1.m_SN')
        self.connect('DC', 'tower1.DC')
        self.connect('gamma_fatigue', 'tower1.gamma_fatigue')
        self.connect('z_DEL', 'tower1.z_DEL')
        self.connect('M_DEL', 'tower1.M_DEL')

        # connections to tower
        self.connect('geometry.z_node', 'tower2.z')
        self.connect('geometry.d_node', 'tower2.d')
        self.connect('geometry.t_node', 'tower2.t')
        self.connect('geometry.z_reinforced', 'tower2.z_reinforced')
        self.connect('windLoads2.windLoads', 'tower2.windLoads')
        self.connect('waveLoads2.waveLoads', 'tower2.waveLoads')
        self.connect('soil.k', 'tower2.k_soil')
        self.connect('rna.rna_mass', 'tower2.top_m')
        self.connect('rna.rna_cm', 'tower2.top_cm')
        self.connect('rna.rna_I_TT', 'tower2.top_I')
        self.connect('rotorloads2.top_F', 'tower2.top_F')
        self.connect('rotorloads2.top_M', 'tower2.top_M')
        self.connect('yaw', 'tower2.yaw')
        self.connect('g', 'tower2.g')
        self.connect('E', 'tower2.E')
        self.connect('G', 'tower2.G')
        self.connect('rho', 'tower2.rho')
        self.connect('sigma_y', 'tower2.sigma_y')
        self.connect('gamma_f', 'tower2.gamma_f')
        self.connect('gamma_m', 'tower2.gamma_m')
        self.connect('gamma_n', 'tower2.gamma_n')
        self.connect('life', 'tower2.life')
        self.connect('m_SN', 'tower2.m_SN')
        self.connect('DC', 'tower2.DC')
        self.connect('gamma_fatigue', 'tower2.gamma_fatigue')
        self.connect('z_DEL', 'tower2.z_DEL')
        self.connect('M_DEL', 'tower2.M_DEL')


        # connections to outputs
        self.connect('tower1.mass', 'mass')
        self.connect('tower1.f1', 'f1')
        self.connect('tower1.f2', 'f2')
        self.connect('tower1.top_deflection', 'top_deflection1')
        self.connect('tower2.top_deflection', 'top_deflection2')
        self.connect('tower1.z', 'z_nodes')
        self.connect('tower1.stress', 'stress1')
        self.connect('tower2.stress', 'stress2')
        self.connect('tower1.z_buckling', 'z_buckling')
        self.connect('tower1.buckling', 'buckling1')
        self.connect('tower2.buckling', 'buckling2')
        self.connect('tower1.damage', 'damage')



if __name__ == '__main__':

    # # start = 4.2
    # # stop = 6.8
    # # num = 8

    # # y, dy_dstart, dy_dstop = linspace_with_deriv(start, stop, num)


    # # yp, blah, blah = linspace_with_deriv(start+1e-6, stop, num)
    # # fd1 = (yp - y)/1e-6

    # # yp, blah, blah = linspace_with_deriv(start, stop+1e-6, num)
    # # fd2 = (yp - y)/1e-6

    # # print dy_dstart
    # # print fd1
    # # print dy_dstart - fd1

    # # print dy_dstop
    # # print fd2
    # # print dy_dstop - fd2

    # # exit()

    # from commonse.environment import check_gradient

    # # # twd = TowerWindDrag()
    # # twd = TowerWaveDrag()
    # # twd.U = [0., 8.80496275, 10.11424623, 10.96861453, 11.61821801, 12.14846828, 12.59962946, 12.99412772, 13.34582791, 13.66394248, 13.95492553, 14.22348635, 14.47317364, 14.70673252, 14.92633314, 15.13372281, 15.33033057, 15.51734112, 15.69574825, 15.86639432, 16.03]
    # # twd.z = [0., 4.38, 8.76, 13.14, 17.52, 21.9, 26.28, 30.66, 35.04, 39.42, 43.8, 48.18, 52.56, 56.94, 61.32, 65.7, 70.08, 74.46, 78.84, 83.22, 87.6]
    # # twd.d = [6., 5.8935, 5.787, 5.6805, 5.574, 5.4675, 5.361, 5.2545, 5.148, 5.0415, 4.935, 4.8285, 4.722, 4.6155, 4.509, 4.4025, 4.296, 4.1895, 4.083, 3.9765, 3.87]
    # # twd.beta = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    # # twd.rho = 1.225
    # # twd.mu = 1.7934e-05

    # # twd.A = 1.1*twd.U
    # # twd.cm = 2.0

    # # check_gradient(twd)

    # # td = TowerDiscretization()
    # # td.z = np.array([0.0, 43.8, 87.6])
    # # td.d = np.array([6.0, 4.935, 3.87])
    # # td.t = np.array([0.0351, 0.0299, 0.0247])
    # # td.n = np.array([10, 7])
    # # td.n_reinforced = 3

    # # check_gradient(td)

    # rna = RNAMass()
    # rna.blade_mass = 15241.323
    # rna.hub_mass = 50421.4
    # rna.nac_mass = 221245.8
    # rna.hub_cm = [-6.3, 0.,  3.15]
    # rna.nac_cm = [-0.32, 0.,   2.4 ]
    # rna.blade_I = [ 26375976., 13187988., 13187988., 0., 0., 0.]
    # rna.hub_I = [ 127297.8, 127297.8, 127297.8, 0., 0., 0. ]
    # rna.nac_I = [ 9908302.58, 912488.28, 1160903.54, 0., 0., 0.  ]
    # rna.nBlades = 3

    # # rna.run()

    # # r1 = rna.rna_I_TT[0]

    # # rna.hub_cm[1] -= 1e-6
    # # rna.run()
    # # r2 = rna.rna_I_TT[0]
    # # print r1
    # # print r2
    # # print r2 - r1
    # # print (r2 - r1)/1e-6

    # # exit()

    # check_gradient(rna)

    # exit()


    from commonse.environment import PowerWind, TowerSoil

    tower = Tower()
    tower.replace('wind', PowerWind())
    tower.replace('soil', TowerSoil())
    tower.replace('tower', TowerWithpBEAM())
    # tower.replace('tower', TowerWithFrame3DD())

    # geometry
    tower.towerHeight = 87.6
    tower.z = np.array([0.0, 0.5, 1.0])
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

    # tower.soil.rigid = 6*[False]



    tower.run()


    print 'mass =', tower.mass
    print 'f1 =', tower.f1
    print 'f2 =', tower.f2
    print 'top_deflection =', tower.top_deflection
    print 'stress =', tower.stress
    print 'z_buckling =', tower.z_buckling
    print 'buckling =', tower.buckling
    print 'damage =', tower.damage

