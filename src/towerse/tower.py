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
from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Slot, Bool

from commonse.utilities import sind, cosd, linspace_with_deriv, interp_with_deriv, hstack, vstack
from commonse.csystem import DirectionVector
from commonse.environment import WindBase, WaveBase, SoilBase
from towerSupplement import fatigue, hoopStressEurocode, shellBucklingEurocode, \
    bucklingGL, vonMisesStressUtilization
from akima import Akima


# "Experiments on the Flow Past a Circular Cylinder at Very High Reynolds Numbers", Roshko
Re_pt = [0.00001, 0.0001, 0.0010, 0.0100, 0.0200, 0.1220, 0.2000, 0.3000, 0.4000,
         0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 5.0000, 10.0000]
cd_pt = [4.0000,  2.0000, 1.1100, 1.1100, 1.2000, 1.2000, 1.1700, 0.9000, 0.5400,
         0.3100, 0.3800, 0.4600, 0.5300, 0.5700, 0.6100, 0.6400, 0.6700, 0.7000, 0.7000]

drag_spline = Akima(np.log10(Re_pt), cd_pt, delta_x=0.0)  # exact akima because control points do not change


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

    # variables
    U = Array(iotype='in', units='m/s', desc='magnitude of wind speed')
    z = Array(iotype='in', units='m', desc='heights where wind speed was computed')
    d = Array(iotype='in', units='m', desc='corresponding diameter of cylinder section')

    # parameters
    beta = Array(iotype='in', units='deg', desc='corresponding wind angles relative to inertial coordinate system')
    rho = Float(1.225, iotype='in', units='kg/m**3', desc='air density')
    mu = Float(1.7934e-5, iotype='in', units='kg/(m*s)', desc='dynamic viscosity of air')
    cd_usr = Float(iotype='in', desc='User input drag coefficient to override Reynolds number based one')

    # out
    windLoads = VarTree(AeroLoads(), iotype='out', desc='wind loads in inertial coordinate system')

    missing_deriv_policy = 'assume_zero'


    def execute(self):

        rho = self.rho
        U = self.U
        d = self.d
        mu = self.mu
        beta = self.beta

        # dynamic pressure
        q = 0.5*rho*U**2

        # Reynolds number and drag
        if self.cd_usr:
            cd = self.cd_usr
            Re = 1.0
            dcd_dRe = 0.0
        else:
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


    def list_deriv_vars(self):

        inputs = ('U', 'z', 'd')
        outputs = ('windLoads.Px', 'windLoads.Py', 'windLoads.Pz', 'windLoads.q', 'windLoads.z')

        return inputs, outputs


    def provideJ(self):

        n = len(self.z)

        zeron = np.zeros((n, n))

        dPx = np.hstack([np.diag(self.dPx_dU), zeron, np.diag(self.dPx_dd)])
        dPy = np.hstack([np.diag(self.dPy_dU), zeron, np.diag(self.dPy_dd)])
        dPz = np.zeros((n, 3*n))
        dq = np.hstack([np.diag(self.dq_dU), np.zeros((n, 2*n))])
        dz = np.hstack([zeron, np.eye(n), zeron])

        J = np.vstack([dPx, dPy, dPz, dq, dz])

        return J





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
    cd_usr = Float(iotype='in', desc='User input drag coefficient to override Reynolds number based one')

    # out
    waveLoads = VarTree(AeroLoads(), iotype='out', desc='wave loads in inertial coordinate system')


    missing_deriv_policy = 'assume_zero'


    def execute(self):

        rho = self.rho
        U = self.U
        d = self.d
        mu = self.mu
        beta = self.beta

        # dynamic pressure
        q = 0.5*rho*U**2

        # Reynolds number and drag
        if self.cd_usr:
            cd = self.cd_usr
            dcd_dRe = 0.0
        else:
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


    def list_deriv_vars(self):

        inputs = ('U', 'A', 'z', 'd')
        outputs = ('waveLoads.Px', 'waveLoads.Py', 'waveLoads.Pz', 'waveLoads.q', 'waveLoads.z', 'waveLoads.beta')

        return inputs, outputs


    def provideJ(self):

        n = len(self.z)

        zeron = np.zeros((n, n))

        dPx = np.hstack([np.diag(self.dPx_dU), np.diag(self.dPx_dA), zeron, np.diag(self.dPx_dd)])
        dPy = np.hstack([np.diag(self.dPy_dU), np.diag(self.dPy_dA), zeron, np.diag(self.dPy_dd)])
        dPz = np.zeros((n, 4*n))
        dq = np.hstack([np.diag(self.dq_dU), np.zeros((n, 3*n))])
        dz = np.hstack([zeron, zeron, np.eye(n), zeron])

        J = np.vstack([dPx, dPy, dPz, dq, dz, np.zeros((n, 4*n))])  # TODO: remove these zeros after OpenMDAO bug fix (don't need waveLoads.beta)

        return J







class TowerDiscretization(Component):
    """discretize geometry into finite element nodes"""

    # variables
    towerHeight = Float(iotype='in', units='m')
    monopileHeight = Float(0.0, iotype='in', units='m')
    z = Array(iotype='in', desc='locations along unit tower, linear lofting between')
    d = Array(iotype='in', units='m', desc='tower diameter at corresponding locations')
    t = Array(iotype='in', units='m', desc='shell thickness at corresponding locations')
    d_monopile = Float(iotype='in', units='m')
    t_monopile = Float(iotype='in', units='m')

    # parameters
    n = Array(iotype='in', dtype=np.int, desc='number of finite elements between sections.  array length should be ``len(z)-1``')
    n_monopile = Int(iotype='in', desc='must be a minimum of 1 (top and bottom)')
    L_reinforced = Float(iotype='in', units='m')

    # out
    z_node = Array(iotype='out', units='m', desc='locations along tower, linear lofting between')
    d_node = Array(iotype='out', units='m', desc='tower diameter at corresponding locations')
    t_node = Array(iotype='out', units='m', desc='shell thickness at corresponding locations')
    L_reinforced_node = Array(iotype='out', units='m')


    missing_deriv_policy = 'assume_zero'


    def execute(self):

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

        # TODO: redo gradients for L_reinforced, although it probably won't ever change

        # reinforcement distances
        self.L_reinforced_node = self.L_reinforced*np.ones_like(self.z_node)

        # self.z_reinforced, dzr_dz0, dzr_dzend = linspace_with_deriv(self.z[0], self.z[-1], self.n_reinforced+1)
        # self.dzr_dz = np.zeros((len(self.z_reinforced), n2))
        # self.dzr_dz[:, 0] = dzr_dz0
        # self.dzr_dz[:, -1] = dzr_dzend

        # make dimensional
        towerHt = self.towerHeight
        self.z_node *= towerHt
        self.dznode_dz *= towerHt
        # self.z_reinforced *= towerHt
        # self.dzr_dz *= towerHt

        # TODO: redo gradients for monopile
        if self.monopileHeight > 0:
            z_monopile = np.linspace(self.z[0] - self.monopileHeight, self.z[0], self.n_monopile)
            d_monopile = self.d_monopile * np.ones_like(z_monopile)
            t_monopile = self.t_monopile * np.ones_like(z_monopile)
            L_monopile = self.L_reinforced * np.ones_like(z_monopile)

            self.z_node = np.concatenate([z_monopile[:-1], self.z_node])
            self.d_node = np.concatenate([d_monopile[:-1], self.d_node])
            self.t_node = np.concatenate([t_monopile[:-1], self.t_node])
            self.L_reinforced_node = np.concatenate([L_monopile[:-1], self.L_reinforced_node])
            # self.z_reinforced = np.concatenate([[z_monopile[0]], self.z_reinforced])


    def list_deriv_vars(self):

        inputs = ('towerHeight', 'z', 'd', 't')
        outputs = ('z_node', 'd_node', 't_node')  # , 'z_reinforced')

        return inputs, outputs


    def provideJ(self):

        n = len(self.z_node)
        m = len(self.z)
        # n2 = len(self.z_reinforced)

        dzn = hstack([self.z_node/self.towerHeight, self.dznode_dz, np.zeros((n, 2*m))])
        ddn = hstack([np.zeros(n), self.ddnode_dz, self.ddnode_dd, np.zeros((n, m))])
        dtn = hstack([np.zeros(n), self.dtnode_dz, np.zeros((n, m)), self.dtnode_dt])
        # dzr = hstack([self.z_reinforced/self.towerHeight, self.dzr_dz, np.zeros((n2, 2*m))])

        J = np.vstack([dzn, ddn, dtn])

        return J




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


    def list_deriv_vars(self):

        inputs = ('blades_mass', 'hub_mass', 'nac_mass', 'hub_cm', 'nac_cm', 'blades_I', 'hub_I', 'nac_I')
        outputs = ('rna_mass', 'rna_cm', 'rna_I_TT')

        return inputs, outputs


    def provideJ(self):

        # mass
        dmass = np.hstack([np.array([1.0, 1.0, 1.0]), np.zeros(2*3+3*6)])

        # cm
        top = (self.rotor_mass*self.hub_cm + self.nac_mass*self.nac_cm)
        dcm_dblademass = (self.rna_mass*self.hub_cm - top)/self.rna_mass**2
        dcm_dhubmass = (self.rna_mass*self.hub_cm - top)/self.rna_mass**2
        dcm_dnacmass = (self.rna_mass*self.nac_cm - top)/self.rna_mass**2
        dcm_dhubcm = self.rotor_mass/self.rna_mass*np.eye(3)
        dcm_dnaccm = self.nac_mass/self.rna_mass*np.eye(3)

        dcm = hstack([dcm_dblademass, dcm_dhubmass, dcm_dnacmass, dcm_dhubcm,
            dcm_dnaccm, np.zeros((3, 3*6))])

        # I
        R = self.hub_cm
        const = self._unassembleI(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        dI_dblademass = const
        dI_dhubmass = const
        dI_drx = self.rotor_mass*self._unassembleI(2*R[0]*np.eye(3) - np.array([[2*R[0], R[1], R[2]], [R[1], 0.0, 0.0], [R[2], 0.0, 0.0]]))
        dI_dry = self.rotor_mass*self._unassembleI(2*R[1]*np.eye(3) - np.array([[0.0, R[0], 0.0], [R[0], 2*R[1], R[2]], [0.0, R[2], 0.0]]))
        dI_drz = self.rotor_mass*self._unassembleI(2*R[2]*np.eye(3) - np.array([[0.0, 0.0, R[0]], [0.0, 0.0, R[1]], [R[0], R[1], 2*R[2]]]))
        dI_dhubcm = np.vstack([dI_drx, dI_dry, dI_drz]).T

        R = self.nac_cm
        const = self._unassembleI(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        dI_dnacmass = const
        dI_drx = self.nac_mass*self._unassembleI(2*R[0]*np.eye(3) - np.array([[2*R[0], R[1], R[2]], [R[1], 0.0, 0.0], [R[2], 0.0, 0.0]]))
        dI_dry = self.nac_mass*self._unassembleI(2*R[1]*np.eye(3) - np.array([[0.0, R[0], 0.0], [R[0], 2*R[1], R[2]], [0.0, R[2], 0.0]]))
        dI_drz = self.nac_mass*self._unassembleI(2*R[2]*np.eye(3) - np.array([[0.0, 0.0, R[0]], [0.0, 0.0, R[1]], [R[0], R[1], 2*R[2]]]))
        dI_dnaccm = np.vstack([dI_drx, dI_dry, dI_drz]).T

        dI_dbladeI = np.eye(6)
        dI_dhubI = np.eye(6)
        dI_dnacI = np.eye(6)

        dI = hstack([dI_dblademass, dI_dhubmass, dI_dnacmass, dI_dhubcm, dI_dnaccm,
            dI_dbladeI, dI_dhubI, dI_dnacI])

        J = np.vstack([dmass, dcm, dI])

        return J






class RotorLoads(Component):

    # variables
    F = Array(iotype='in', desc='forces in hub-aligned coordinate system')
    M = Array(iotype='in', desc='moments in hub-aligned coordinate system')
    r_hub = Array(iotype='in', desc='position of rotor hub relative to tower top in yaw-aligned c.s.')
    m_RNA = Float(iotype='in', units='kg', desc='mass of rotor nacelle assembly')
    rna_cm = Array(iotype='in', units='m', desc='location of RNA center of mass relative to tower top in yaw-aligned c.s.')

    # parameters
    downwind = Bool(False, iotype='in')
    tilt = Float(iotype='in', units='deg')
    g = Float(9.81, iotype='in', units='m/s**2')

    # out
    top_F = Array(iotype='out')  # in yaw-aligned
    top_M = Array(iotype='out')

    missing_deriv_policy = 'assume_zero'


    def execute(self):

        F = self.F
        M = self.M

        F = DirectionVector.fromArray(F).hubToYaw(self.tilt)
        M = DirectionVector.fromArray(M).hubToYaw(self.tilt)

        # change x-direction if downwind
        r_hub = np.copy(self.r_hub)
        rna_cm = np.copy(self.rna_cm)
        if self.downwind:
            r_hub[0] *= -1
            rna_cm[0] *= -1
        r_hub = DirectionVector.fromArray(r_hub)
        rna_cm = DirectionVector.fromArray(rna_cm)
        self.save_rhub = r_hub
        self.save_rcm = rna_cm

        # aerodynamic moments
        M = M + r_hub.cross(F)
        self.saveF = F

        # add weight loads
        F_w = DirectionVector(0.0, 0.0, -self.m_RNA*self.g)
        M_w = rna_cm.cross(F_w)
        self.saveF_w = F_w

        F = F + F_w
        M = M + M_w

        # put back in array
        self.top_F = np.array([F.x, F.y, F.z])
        self.top_M = np.array([M.x, M.y, M.z])


    def list_deriv_vars(self):

        inputs = ('T', 'Q', 'r_hub', 'm_RNA', 'rna_cm')
        outputs = ('top_F', 'top_M')

        return inputs, outputs

    def provideJ(self):

        dF = DirectionVector(self.T, 0.0, 0.0).hubToYaw(self.tilt)
        dFx, dFy, dFz = dF.dx, dF.dy, dF.dz

        dtopF_dT = np.array([dFx['dx'], dFy['dx'], dFz['dx']])
        dtopF_w_dm = np.array([0.0, 0.0, -self.g])

        dtopF = hstack([dtopF_dT, np.zeros((3, 4)), dtopF_w_dm, np.zeros((3, 3))])


        dM = DirectionVector(self.Q, 0.0, 0.0).hubToYaw(self.tilt)
        dMx, dMy, dMz = dM.dx, dM.dy, dM.dz
        dMxcross, dMycross, dMzcross = self.save_rhub.cross_deriv(self.saveF, 'dr', 'dF')

        dtopM_dQ = np.array([dMx['dx'], dMy['dx'], dMz['dx']])
        dM_dF = np.array([dMxcross['dF'], dMycross['dF'], dMzcross['dF']])

        dtopM_dT = np.dot(dM_dF, dtopF_dT)
        dtopM_dr = np.array([dMxcross['dr'], dMycross['dr'], dMzcross['dr']])

        dMx_w_cross, dMy_w_cross, dMz_w_cross = self.save_rcm.cross_deriv(self.saveF_w, 'dr', 'dF')

        dtopM_drnacm = np.array([dMx_w_cross['dr'], dMy_w_cross['dr'], dMz_w_cross['dr']])
        dtopM_dF_w = np.array([dMx_w_cross['dF'], dMy_w_cross['dF'], dMz_w_cross['dF']])
        dtopM_dm = np.dot(dtopM_dF_w, dtopF_w_dm)

        if self.downwind:
            dtopM_dr[:, 0] *= -1
            dtopM_drnacm[:, 0] *= -1

        dtopM = hstack([dtopM_dT, dtopM_dQ, dtopM_dr, dtopM_dm, dtopM_drnacm])

        J = vstack([dtopF, dtopM])

        return J



class GeometricConstraints(Component):
    """docstring for OtherConstraints"""

    d = Array(iotype='in')
    t = Array(iotype='in')
    min_d_to_t = Float(120.0, iotype='in')
    min_taper = Float(0.4, iotype='in')

    weldability = Array(iotype='out')
    manufactuability = Float(iotype='out')


    def execute(self):

        self.weldability = (self.min_d_to_t - self.d/self.t) / self.min_d_to_t
        self.manufactuability = self.min_taper - self.d[-1] / self.d[0]  # taper ratio


    def list_deriv_vars(self):

        inputs = ('d', 't')
        outputs = ('weldability', 'manufactuability')

        return inputs, outputs


    def provideJ(self):

        dw_dd = np.diag(-1.0/self.t/self.min_d_to_t)
        dw_dt = np.diag(self.d/self.t**2/self.min_d_to_t)

        dw = np.hstack([dw_dd, dw_dt])


        dm_dd = np.zeros_like(self.d)
        dm_dd[0] = self.d[-1]/self.d[0]**2
        dm_dd[-1] = -1.0/self.d[0]

        dm = np.hstack([dm_dd, np.zeros(len(self.t))])

        J = np.vstack([dw, dm])

        return J



class TowerBase(Component):
    """structural analysis of cylindrical tower

    all forces, moments, distances should be given (and returned) in the yaw-aligned coordinate system
    """

    # geometry
    towerHeight = Float(iotype='in', units='m')
    z = Array(iotype='in', desc='locations along unit tower')
    d = Array(iotype='in', units='m', desc='tower diameter at corresponding locations')
    t = Array(iotype='in', units='m', desc='shell thickness at corresponding locations')
    L_reinforced = Array(iotype='in', units='m', desc='reinforcement positions for buckling')
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
    gamma_b = Float(1.1, iotype='in', desc='buckling safety factor')

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
    # z_buckling = Array(iotype='out', units='m', desc='z-locations along tower where shell buckling is evaluted')
    shellBuckling = Array(iotype='out', desc='a shell buckling constraint.  should be <= 0 for feasibility.  includes safety factors')
    buckling = Array(iotype='out', desc='a global buckling constraint.  should be <= 0 for feasibility.  includes safety factors')
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




    def fatigue(self):
        """compute damage from provided damage equivalent moments"""

        # TODO: temporary hack to extract tower portion
        idx = self.z >= 0
        z = self.z[idx]
        d = self.d[idx]
        t = self.t[idx]

        # fatigue
        N_DEL = [365*24*3600*self.life]*len(z)
        M_DEL = np.interp(z, self.towerHeight*self.z_DEL, self.M_DEL)

        damage = fatigue(M_DEL, N_DEL, d, t, self.m_SN, self.DC, self.gamma_fatigue, stress_factor=1.0, weld_factor=True)

        # TODO: more hack
        damage = np.concatenate([np.zeros(len(self.z)-len(z)), damage])

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

        # I = math.pi/8 * d**3 * t
        # axial_stress2 = -My*d/2.0/I + Fz/A

        # hoop_stress (Eurocode method)
        hoop_stress = hoopStressEurocode(self.windLoads, self.waveLoads, z, d, t, self.L_reinforced)

        # von mises stress
        self.stress = vonMisesStressUtilization(axial_stress, hoop_stress, shear_stress,
            self.gamma_f*self.gamma_m*self.gamma_n, self.sigma_y)

        # buckling
        self.shellBuckling = shellBucklingEurocode(d, t, axial_stress, hoop_stress, shear_stress,
            self.L_reinforced, self.E*np.ones(nodes), self.sigma_y*np.ones(nodes),
            self.gamma_f, self.gamma_b)

        tower_height = self.z[-1] - self.z[0]
        self.buckling = bucklingGL(d, t, Fz, My, tower_height, self.E*np.ones(nodes),
            self.sigma_y*np.ones(nodes), self.gamma_f, self.gamma_b)

        # fatigue
        self.damage = self.fatigue()


    # def list_deriv_vars(self):

    #     inputs = ('z', 'd', 't')
    #     outputs = ('mass',)

    #     return inputs, outputs


    # def provideJ(self):

    #     # m = 0.0
    #     # for i in range(nodes - 1):
    #     #     m += (z[i+1]-z[i]) * (1.0/3*(d[i]*t[i] + d[i+1]*t[i+1]) + 1.0/6*(d[i]*t[i+1] + d[i+1]*t[i]))
    #     # m *= self.rho * pi

    #     z = self.z
    #     d = self.d
    #     t = self.t

    #     n = len(z)
    #     dmass_dz = np.zeros(n)
    #     dmass_dd = np.zeros(n)
    #     dmass_dt = np.zeros(n)

    #     for i in range(n - 1):
    #         factor = 1.0/3*(d[i]*t[i] + d[i+1]*t[i+1]) + 1.0/6*(d[i]*t[i+1] + d[i+1]*t[i])
    #         dmass_dz[i] = -factor
    #         dmass_dz[i+1] = factor
    #         dmass_dd[i] = (z[i+1]-z[i]) * (1.0/3*t[i] + 1.0/6*t[i+1])
    #         dmass_dd[i+1] = (z[i+1]-z[i]) * (1.0/3*t[i+1] + 1.0/6*t[i])
    #         dmass_dt[i] = (z[i+1]-z[i]) * (1.0/3*d[i] + 1.0/6*d[i+1])
    #         dmass_dt[i+1] = (z[i+1]-z[i]) * (1.0/3*d[i+1] + 1.0/6*d[i])

    #     dmass = self.rho * pi * hstack([dmass_dz, dmass_dd, dmass_dt])


    #     # Vx = np.zeros(nodes)
    #     # Vy = np.zeros(nodes)
    #     # Fz = np.zeros(nodes)
    #     # Mx = np.zeros(nodes)
    #     # My = np.zeros(nodes)
    #     # Tz = np.zeros(nodes)
    #     # Vx[-1] = self.top_F[0]
    #     # Vy[-1] = self.top_F[1]
    #     # Fz[-1] = self.top_F[2]
    #     # Mx[-1] = self.top_M[0]
    #     # My[-1] = self.top_M[1]
    #     # Tz[-1] = self.top_M[2]
    #     # for i in reversed(range(nodes-1)):
    #     #     Vx[i] = Vx[i+1] + 0.5*(Px[i] + Px[i+1])*(z[i+1]-z[i])
    #     #     Vy[i] = Vy[i+1] + 0.5*(Py[i] + Py[i+1])*(z[i+1]-z[i])
    #     #     Fz[i] = Fz[i+1] + 0.5*(Pz[i] + Pz[i+1])*(z[i+1]-z[i])

    #     #     Mx[i] = Mx[i+1] + Vy[i+1]*(z[i+1]-z[i]) + (Py[i]/6.0 + Py[i+1]/3.0)*(z[i+1]-z[i])**2
    #     #     My[i] = My[i+1] + Vx[i+1]*(z[i+1]-z[i]) + (Px[i]/6.0 + Px[i+1]/3.0)*(z[i+1]-z[i])**2
    #     #     Tz[i] = Tz[i+1]

    #     J = vstack([dmass])

    #     return J


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
        hoop_stress = hoopStressEurocode(self.windLoads, self.waveLoads,
            self.z, self.d, self.t, self.z_reinforced)

        # von mises stress
        self.stress = vonMisesStressUtilization(axial_stress, hoop_stress, shear_stress,
            self.gamma_f*self.gamma_m*self.gamma_n, self.sigma_y)

        # buckling
        self.buckling = shellBucklingEurocode(d, t, axial_stress, hoop_stress, shear_stress,
            self.L_reinforced, self.E*np.ones(nodes), self.sigma_y*np.ones(nodes), self.gamma_f, self.gamma_b)

        # fatigue
        self.damage = self.fatigue()





# -----------------
#  Assembly
# -----------------


class TowerSE(Assembly):

    # geometry
    towerHeight = Float(iotype='in', units='m')
    monopileHeight = Float(0.0, iotype='in', units='m')
    d_monopile = Float(iotype='in', units='m')
    t_monopile = Float(iotype='in', units='m')
    z = Array(iotype='in', desc='locations along unit tower, linear lofting between')
    d = Array(iotype='in', units='m', desc='tower diameter at corresponding locations')
    t = Array(iotype='in', units='m', desc='shell thickness at corresponding locations')
    n = Array(iotype='in', dtype=np.int, desc='number of finite elements between sections.  array length should be ``len(z)-1``')
    L_reinforced = Float(iotype='in', units='m')
    n_monopile = Int(iotype='in', desc='must be a minimum of 1 (top and bottom)')
    yaw = Float(0.0, iotype='in', units='deg')
    tilt = Float(0.0, iotype='in', units='deg')
    downwind = Bool(False, iotype='in')

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

    # constraint parameters
    min_d_to_t = Float(120.0, iotype='in')
    min_taper = Float(0.4, iotype='in')


    # rotor loads
    rotorF1 = Array(iotype='in', desc='forces in hub-aligned coordinate system at rotor hub')
    rotorM1 = Array(iotype='in', desc='moments in hub-aligned coordinate system at rotor hub')
    rotorF2 = Array(iotype='in', desc='forces in hub-aligned coordinate system at rotor hub')
    rotorM2 = Array(iotype='in', desc='moments in hub-aligned coordinate system at rotor hub')

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
    gamma_b = Float(1.1, iotype='in', desc='buckling safety factor')

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
    shellBuckling1 = Array(iotype='out', desc='a shell buckling constraint.  should be <= 0 for feasibility.  includes safety factors')
    shellBuckling2 = Array(iotype='out', desc='a shell buckling constraint.  should be <= 0 for feasibility.  includes safety factors')
    buckling1 = Array(iotype='out', desc='a global buckling constraint.  should be <= 0 for feasibility.  includes safety factors')
    buckling2 = Array(iotype='out', desc='a global buckling constraint.  should be <= 0 for feasibility.  includes safety factors')
    damage = Array(iotype='out', desc='fatigue damage at each tower section')
    weldability = Array(iotype='out')
    manufactuability = Float(iotype='out')


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
        self.add('gc', GeometricConstraints())

        # self.driver.workflow.add(['geometry', 'wind', 'wave', 'windLoads', 'waveLoads', 'soil', 'rna', 'rotorloads', 'tower'])
        self.driver.workflow.add(['geometry', 'wind1', 'wind2', 'wave1', 'wave2',
            'windLoads1', 'windLoads2', 'waveLoads1', 'waveLoads2', 'soil', 'rna',
            'rotorloads1', 'rotorloads2', 'tower1', 'tower2', 'gc'])
        # TODO: probably better to do this with a driver or something rather than manually setting 2 cases

        # connections to geometry
        self.connect('towerHeight', 'geometry.towerHeight')
        self.connect('monopileHeight', 'geometry.monopileHeight')
        self.connect('d_monopile', 'geometry.d_monopile')
        self.connect('t_monopile', 'geometry.t_monopile')
        self.connect('z', 'geometry.z')
        self.connect('d', 'geometry.d')
        self.connect('t', 'geometry.t')
        self.connect('n', 'geometry.n')
        self.connect('n_monopile', 'geometry.n_monopile')
        self.connect('L_reinforced', 'geometry.L_reinforced')


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
        self.connect('downwind', 'rotorloads1.downwind')
        self.connect('rotorF1', 'rotorloads1.F')
        self.connect('rotorM1', 'rotorloads1.M')
        self.connect('hub_cm', 'rotorloads1.r_hub')
        self.connect('rna.rna_cm', 'rotorloads1.rna_cm')
        self.connect('tilt', 'rotorloads1.tilt')
        self.connect('g', 'rotorloads1.g')
        self.connect('rna.rna_mass', 'rotorloads1.m_RNA')

        # connections to rotorloads2
        self.connect('downwind', 'rotorloads2.downwind')
        self.connect('rotorF2', 'rotorloads2.F')
        self.connect('rotorM2', 'rotorloads2.M')
        self.connect('hub_cm', 'rotorloads2.r_hub')
        self.connect('rna.rna_cm', 'rotorloads2.rna_cm')
        self.connect('tilt', 'rotorloads2.tilt')
        self.connect('g', 'rotorloads2.g')
        self.connect('rna.rna_mass', 'rotorloads2.m_RNA')

        # connections to tower
        self.connect('towerHeight', 'tower1.towerHeight')
        self.connect('geometry.z_node', 'tower1.z')
        self.connect('geometry.d_node', 'tower1.d')
        self.connect('geometry.t_node', 'tower1.t')
        self.connect('geometry.L_reinforced_node', 'tower1.L_reinforced')
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
        self.connect('gamma_b', 'tower1.gamma_b')
        self.connect('life', 'tower1.life')
        self.connect('m_SN', 'tower1.m_SN')
        self.connect('DC', 'tower1.DC')
        self.connect('gamma_fatigue', 'tower1.gamma_fatigue')
        self.connect('z_DEL', 'tower1.z_DEL')
        self.connect('M_DEL', 'tower1.M_DEL')

        # connections to tower
        self.connect('towerHeight', 'tower2.towerHeight')
        self.connect('geometry.z_node', 'tower2.z')
        self.connect('geometry.d_node', 'tower2.d')
        self.connect('geometry.t_node', 'tower2.t')
        self.connect('geometry.L_reinforced_node', 'tower2.L_reinforced')
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
        self.connect('gamma_b', 'tower2.gamma_b')
        self.connect('life', 'tower2.life')
        self.connect('m_SN', 'tower2.m_SN')
        self.connect('DC', 'tower2.DC')
        self.connect('gamma_fatigue', 'tower2.gamma_fatigue')
        self.connect('z_DEL', 'tower2.z_DEL')
        self.connect('M_DEL', 'tower2.M_DEL')

        # connections to gc
        self.connect('d', 'gc.d')
        self.connect('t', 'gc.t')
        self.connect('min_d_to_t', 'gc.min_d_to_t')
        self.connect('min_taper', 'gc.min_taper')



        # connections to outputs
        self.connect('tower1.mass', 'mass')
        self.connect('tower1.f1', 'f1')
        self.connect('tower1.f2', 'f2')
        self.connect('tower1.top_deflection', 'top_deflection1')
        self.connect('tower2.top_deflection', 'top_deflection2')
        self.connect('tower1.z', 'z_nodes')
        self.connect('tower1.stress', 'stress1')
        self.connect('tower2.stress', 'stress2')
        self.connect('tower1.buckling', 'buckling1')
        self.connect('tower2.buckling', 'buckling2')
        self.connect('tower1.shellBuckling', 'shellBuckling1')
        self.connect('tower2.shellBuckling', 'shellBuckling2')
        self.connect('tower1.damage', 'damage')
        self.connect('gc.weldability', 'weldability')
        self.connect('gc.manufactuability', 'manufactuability')



if __name__ == '__main__':




    # --- tower setup ------
    from commonse.environment import PowerWind, TowerSoil
    # from towerse.tower import TowerWithpBEAM

    tower = TowerSE()

    tower.replace('wind1', PowerWind())
    tower.replace('wind2', PowerWind())
    # onshore (no waves)
    tower.replace('soil', TowerSoil())
    tower.replace('tower1', TowerWithpBEAM())
    tower.replace('tower2', TowerWithpBEAM())
    # ---------------

    # --- geometry ---
    tower.towerHeight = 87.6
    tower.z = [0.0, 0.5, 1.0]
    tower.d = [6.0, 4.935, 3.87]
    tower.t = [0.027*1.3, 0.023*1.3, 0.019*1.3]
    tower.n = [10, 10]
    tower.L_reinforced = 30.0
    tower.yaw = 0.0
    tower.tilt = 5.0
    # ---------------

    # --- blades ---
    tower.blades_mass = 3 * 15241.323
    bladeI = 3 * 8791992.000
    tower.blades_I = np.array([bladeI, bladeI/2.0, bladeI/2.0, 0.0, 0.0, 0.0])
    # ---------------

    # --- hub ---
    tower.hub_mass = 50421.4
    tower.hub_cm = np.array([-6.30, 0, 3.15])
    tower.hub_I = np.array([127297.8, 127297.8, 127297.8, 0.0, 0.0, 0.0])
    # ---------------

    # --- nacelle ---
    tower.nac_mass = 221245.8
    tower.nac_cm = np.array([-0.32, 0, 2.40])
    tower.nac_I = np.array([9908302.58, 912488.28, 1160903.54, 0.0, 0.0, 0.0])
    # ---------------

    # --- wind ---
    towerToShaft = 2.0
    tower.wind_zref = tower.towerHeight + towerToShaft
    tower.wind_z0 = 0.0
    tower.wind1.shearExp = 0.2
    tower.wind2.shearExp = 0.2
    # ---------------

    # --- soil ---
    tower.soil.rigid = 6*[True]
    # ---------------

    # --- loading case 1: max Thrust ---
    tower.wind_Uref1 = 16.030
    tower.rotorF1 = [1.3295e6, 0.0, 0.0]
    tower.rotorM1 = [6.2829e6, 0.0, 0.0]
    # ---------------

    # --- loading case 2: max wind speed ---
    tower.wind_Uref2 = 67.89
    tower.rotorF2 = [1.1770e6, 0.0, 0.0]
    tower.rotorM2 = [1.5730e6, 0.0, 0.0]
    # ---------------

    # --- safety factors ---
    tower.gamma_f = 1.35
    tower.gamma_m = 1.3
    tower.gamma_n = 1.0
    tower.gamma_b = 1.1
    # ---------------

    # --- fatigue ---
    tower.z_DEL = 1.0/87.6*np.array([0.000, 1.327, 3.982, 6.636, 9.291, 11.945, 14.600, 17.255, 19.909, 22.564, 25.218, 27.873, 30.527, 33.182, 35.836, 38.491, 41.145, 43.800, 46.455, 49.109, 51.764, 54.418, 57.073, 59.727, 62.382, 65.036, 67.691, 70.345, 73.000, 75.655, 78.309, 80.964, 83.618, 86.273, 87.600])
    tower.M_DEL = 1e3*np.array([8.2940E+003, 8.1518E+003, 7.8831E+003, 7.6099E+003, 7.3359E+003, 7.0577E+003, 6.7821E+003, 6.5119E+003, 6.2391E+003, 5.9707E+003, 5.7070E+003, 5.4500E+003, 5.2015E+003, 4.9588E+003, 4.7202E+003, 4.4884E+003, 4.2577E+003, 4.0246E+003, 3.7942E+003, 3.5664E+003, 3.3406E+003, 3.1184E+003, 2.8977E+003, 2.6811E+003, 2.4719E+003, 2.2663E+003, 2.0673E+003, 1.8769E+003, 1.7017E+003, 1.5479E+003, 1.4207E+003, 1.3304E+003, 1.2780E+003, 1.2673E+003, 1.2761E+003])
    tower.gamma_fatigue = 1.35*1.3*1.0
    tower.life = 20.0
    tower.m_SN = 4
    # ---------------

    # --- constraints ---
    tower.min_d_to_t = 120.0
    tower.min_taper = 0.4
    # ---------------

    # V_max = 80.0  # tip speed
    # D = 126.0
    # tower.freq1p = V_max / (D/2) / (2*pi)  # convert to Hz


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

    # --- run ---
    tower.run()

    print 'mass (kg) =', tower.mass
    print 'f1 (Hz) =', tower.f1
    print 'f2 (Hz) =', tower.f2
    print 'top_deflection1 (m) =', tower.top_deflection1
    print 'top_deflection2 (m) =', tower.top_deflection2
    print 'weldability =', tower.weldability
    print 'manufactuability =', tower.manufactuability


    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.0, 3.5))
    plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
    plt.plot(tower.stress1, tower.z_nodes, label='stress1')
    plt.plot(tower.stress2, tower.z_nodes, label='stress1')
    plt.plot(tower.shellBuckling1, tower.z_nodes, label='shell buckling 1')
    plt.plot(tower.shellBuckling2, tower.z_nodes, label='shell buckling 2')
    plt.plot(tower.buckling1, tower.z_nodes, label='global buckling 1')
    plt.plot(tower.buckling2, tower.z_nodes, label='global buckling 2')
    plt.plot(tower.damage, tower.z_nodes, label='damage')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2)
    plt.show()
    # ------------


