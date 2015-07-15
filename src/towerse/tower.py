#!/usr/bin/env python
# encoding: utf-8
"""
towerstruc.py

Created by Andrew Ning on 2012-01-20.
Copyright (c) NREL. All rights reserved.

HISTORY:  2012 created
          -7/2014:  R.D. Bugs found in the call to shellBucklingEurocode from towerwithFrame3DD. Fixed.
                    Also set_as_top added.
          -10/2014: R.D. Merged back with some changes Andrew did on his end.
          -12/2014: A.N. fixed some errors from the merge (redundant drag calc).  pep8 compliance.  removed several unneccesary variables and imports (including set_as_top)
          - 6/2015: A.N. major rewrite.  removed pBEAM.  can add spring stiffness anywhere.  can add mass anywhere.
            can use different material props throughout.
          - 7/2015 : R.D. modified to use commonse modules.
 """

import math
import sys
import numpy as np
from openmdao.main.api import VariableTree, Component, Assembly
from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Bool, Slot

#from commonse.utilities import linspace_with_deriv, interp_with_deriv, hstack, vstack

from commonse.WindWaveDrag import FluidLoads, AeroHydroLoads, TowerWindDrag, TowerWaveDrag

from commonse.environment import WindBase, WaveBase  # , SoilBase

from commonse import Tube

from fusedwind.turbine.tower import TowerFromCSProps
from fusedwind.interface import implement_base

from commonse.UtilizationSupplement import fatigue, hoopStressEurocode, shellBucklingEurocode, \
    bucklingGL, vonMisesStressUtilization

import frame3dd



# -----------------
#  Helper Functions
# -----------------


# -----------------
#  Components
# -----------------

class TowerDiscretization(Component):
    """discretize geometry into finite element nodes"""

    #inputs

    # variables
    z_param = Array(iotype='in', units='m', desc='parameterized locations along tower, linear lofting between')
    d_param = Array(iotype='in', units='m', desc='tower diameter at corresponding locations')
    t_param = Array(iotype='in', units='m', desc='shell thickness at corresponding locations')
    z_full = Array(iotype='in', units='m', desc='locations along tower')

    # out
    d_full = Array(iotype='out', units='m', desc='tower diameter at corresponding locations')
    t_full = Array(iotype='out', units='m', desc='shell thickness at corresponding locations')

    def execute(self):

        self.d_full = np.interp(self.z_full, self.z_param, self.d_param)
        self.t_full = np.interp(self.z_full, self.z_param, self.t_param)


class GeometricConstraints(Component):
    """docstring for OtherConstraints"""

    d = Array(iotype='in')
    t = Array(iotype='in')
    min_d_to_t = Float(120.0, iotype='in')
    min_taper = Float(0.4, iotype='in')

    weldability = Array(iotype='out')
    manufacturability = Array(iotype='out')

    def execute(self):

        self.weldability = (self.min_d_to_t - self.d/self.t) / self.min_d_to_t
        manufacturability = self.min_taper - self.d[1:]/self.d[:-1]  # taper ratio)
        self.manufacturability = np.hstack((manufacturability, manufacturability[-1]))

    # def list_deriv_vars(self):

    #     inputs = ('d', 't')
    #     outputs = ('weldability', 'manufacturability')
    #     return inputs, outputs

    # def provideJ(self):

    #     dw_dd = np.diag(-1.0/self.t/self.min_d_to_t)
    #     dw_dt = np.diag(self.d/self.t**2/self.min_d_to_t)

    #     dw = np.hstack([dw_dd, dw_dt])



    #     dm_dd = np.zeros_like(self.d)
    #     dm_dd[0] = self.d[-1]/self.d[0]**2
    #     dm_dd[-1] = -1.0/self.d[0]

    #     dm = np.hstack([dm_dd, np.zeros(len(self.t))])


class CylindricalShellProperties(Component):
    d = Array(iotype='in', units='m', desc='tower diameter at corresponding locations')
    t = Array(iotype='in', units='m', desc='shell thickness at corresponding locations')

    Az = Array(iotype='out', units='m**2', desc='cross-sectional area')
    Asx = Array(iotype='out', units='m**2', desc='x shear area')
    Asy = Array(iotype='out', units='m**2', desc='y shear area')
    Jz = Array(iotype='out', units='m**4', desc='polar moment of inertia')
    Ixx = Array(iotype='out', units='m**4', desc='area moment of inertia about x-axis')
    Iyy = Array(iotype='out', units='m**4', desc='area moment of inertia about y-axis')

    def execute(self):

        tube=Tube(d,t)
        self.Az = tube.Area
        self.Asx = tube.Asx
        self.Asy = tube.Asy
        self.Jz = tube.J0
        self.Ixx = tube.Jxx
        self.Iyy = tube.Jyy

##        ro = self.d/2.0 + self.t/2.0
##        ri = self.d/2.0 - self.t/2.0
##        self.Az = math.pi * (ro**2 - ri**2)
##        self.Asx = self.Az / (0.54414 + 2.97294*(ri/ro) - 1.51899*(ri/ro)**2)
##        self.Asy = self.Az / (0.54414 + 2.97294*(ri/ro) - 1.51899*(ri/ro)**2)
##        self.Jz = math.pi/2.0 * (ro**4 - ri**4)
##        self.Ixx = self.Jz/2.0
##        self.Iyy = self.Jz/2.0



@implement_base(TowerFromCSProps)
class TowerFrame3DD(Component):

    # cross-sectional data along tower.
    z = Array(iotype='in', units='m', desc='location along tower.  start at bottom at go to top.')
    Az = Array(iotype='in', units='m**2', desc='cross-sectional area')
    Asx = Array(iotype='in', units='m**2', desc='x shear area')
    Asy = Array(iotype='in', units='m**2', desc='y shear area')
    Jz = Array(iotype='in', units='m**4', desc='polar moment of inertia')
    Ixx = Array(iotype='in', units='m**4', desc='area moment of inertia about x-axis')
    Iyy = Array(iotype='in', units='m**4', desc='area moment of inertia about y-axis')

    E = Array(iotype='in', units='N/m**2', desc='modulus of elasticity')
    G = Array(iotype='in', units='N/m**2', desc='shear modulus')
    rho = Array(iotype='in', units='kg/m**3', desc='material density')
    sigma_y = Array(iotype='in', units='N/m**2', desc='yield stress')

    # effective geometry -- used for handbook methods to estimate hoop stress, buckling, fatigue
    # length should be one less than z
    d = Array(iotype='in', units='m', desc='effective tower diameter for section')
    t = Array(iotype='in', units='m', desc='effective shell thickness for section')
    L_reinforced = Array(iotype='in', units='m')

    # locations where stress should be evaluated
    theta_stress = Array(iotype='in', units='deg', desc='location along azimuth where stress should be evaluated.  0 corresponds to +x axis.  follows unit circle direction and c.s.')

    # spring reaction data.  Use float('inf') for rigid constraints.
    kidx = Array(iotype='in', desc='indices of z where external stiffness reactions should be applied.')
    kx = Array(iotype='in', units='m', desc='spring stiffness in x-direction')
    ky = Array(iotype='in', units='m', desc='spring stiffness in y-direction')
    kz = Array(iotype='in', units='m', desc='spring stiffness in z-direction')
    ktx = Array(iotype='in', units='m', desc='spring stiffness in theta_x-rotation')
    kty = Array(iotype='in', units='m', desc='spring stiffness in theta_y-rotation')
    ktz = Array(iotype='in', units='m', desc='spring stiffness in theta_z-rotation')

    # extra mass
    midx = Array(iotype='in', desc='indices where added mass should be applied.')
    m = Array(iotype='in', units='kg', desc='added mass')
    mIxx = Array(iotype='in', units='kg*m**2', desc='x mass moment of inertia about some point p')
    mIyy = Array(iotype='in', units='kg*m**2', desc='y mass moment of inertia about some point p')
    mIzz = Array(iotype='in', units='kg*m**2', desc='z mass moment of inertia about some point p')
    mIxy = Array(iotype='in', units='kg*m**2', desc='xy mass moment of inertia about some point p')
    mIxz = Array(iotype='in', units='kg*m**2', desc='xz mass moment of inertia about some point p')
    mIyz = Array(iotype='in', units='kg*m**2', desc='yz mass moment of inertia about some point p')
    mrhox = Array(iotype='in', units='m', desc='x-location of p relative to node')
    mrhoy = Array(iotype='in', units='m', desc='y-location of p relative to node')
    mrhoz = Array(iotype='in', units='m', desc='z-location of p relative to node')
    addGravityLoadForExtraMass = Bool(True, iotype='in', desc='add gravitational load')

    # gravitational load
    g = Float(9.81, iotype='in', units='m/s**2', desc='acceleration of gravity (magnitude)')

    # point loads (if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also)
    plidx = Array(iotype='in', desc='indices where point loads should be applied.')
    Fx = Array(iotype='in', units='N', desc='point force in x-direction')
    Fy = Array(iotype='in', units='N', desc='point force in y-direction')
    Fz = Array(iotype='in', units='N', desc='point force in z-direction')
    Mxx = Array(iotype='in', units='N*m', desc='point moment about x-axis')
    Myy = Array(iotype='in', units='N*m', desc='point moment about y-axis')
    Mzz = Array(iotype='in', units='N*m', desc='point moment about z-axis')

    # combined wind-water distributed loads
    WWloads = VarTree(FluidLoads(), iotype='in', desc='combined wind and wave loads')
    ##Px   = Array(iotype='in', units='N/m', desc='force per unit length in x-direction')
    ##Py   = Array(iotype='in', units='N/m', desc='force per unit length in y-direction')
    ##Pz   = Array(iotype='in', units='N/m', desc='force per unit length in z-direction')
    ##qdyn = Array(iotype='in', units='N/m**2', desc='dynamic pressure')

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

    # options
    shear = Bool(True, iotype='in', desc='include shear deformation')
    geom = Bool(False, iotype='in', desc='include geometric stiffness')
    dx = Float(5.0, iotype='in', desc='z-axis increment for internal forces')
    nM = Int(2, iotype='in', desc='number of desired dynamic modes of vibration (below only necessary if nM > 0)')
    Mmethod = Int(1, iotype='in', desc='1: subspace Jacobi, 2: Stodola')
    lump = Int(0, iotype='in', desc='0: consistent mass, 1: lumped mass matrix')
    tol = Float(1e-9, iotype='in', desc='mode shape tolerance')
    shift = Float(0.0, iotype='in', desc='shift value ... for unrestrained structures')

    # outputs
    mass = Float(iotype='out')
    f1 = Float(iotype='out', units='Hz', desc='First natural frequency')
    f2 = Float(iotype='out', units='Hz', desc='Second natural frequency')
    top_deflection = Float(iotype='out', units='m', desc='Deflection of tower top in yaw-aligned +x direction')
    stress = Array(iotype='out', units='N/m**2', desc='Von Mises stress utilization along tower at specified locations.  incudes safety factor.')
    shell_buckling = Array(iotype='out', desc='Shell buckling constraint.  Should be < 1 for feasibility.  Includes safety factors')
    global_buckling = Array(iotype='out', desc='Global buckling constraint.  Should be < 1 for feasibility.  Includes safety factors')
    damage = Array(iotype='out', desc='Fatigue damage at each tower section')
    weldability = Array(iotype='out')
    manufacturability = Array(iotype='out')


    def execute(self):

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
        node = self.kidx + 1  # add one because 0-based index but 1-based node numbering
        rigid = float('inf')

        reactions = frame3dd.ReactionData(node, self.kx, self.ky, self.kz, self.ktx, self.kty, self.ktz, rigid)
        # -----------------------------------

        # ------ frame element data ------------
        element = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n+1)

        roll = np.zeros(n-1)

        # average across element b.c. frame3dd uses constant section elements
        Az = 0.5*(self.Az[:-1] + self.Az[1:])
        Asx = 0.5*(self.Asx[:-1] + self.Asx[1:])
        Asy = 0.5*(self.Asy[:-1] + self.Asy[1:])
        Jz = 0.5*(self.Jz[:-1] + self.Jz[1:])
        Ixx = 0.5*(self.Ixx[:-1] + self.Ixx[1:])
        Iyy = 0.5*(self.Iyy[:-1] + self.Iyy[1:])
        E = 0.5*(self.E[:-1] + self.E[1:])
        G = 0.5*(self.G[:-1] + self.G[1:])
        rho = 0.5*(self.rho[:-1] + self.rho[1:])

        elements = frame3dd.ElementData(element, N1, N2, Az, Asx, Asy, Jz,
            Ixx, Iyy, E, G, roll, rho)
        # -----------------------------------


        # ------ options ------------
        options = frame3dd.Options(self.shear, self.geom, self.dx)
        # -----------------------------------

        # initialize frame3dd object
        tower = frame3dd.Frame(nodes, reactions, elements, options)


        # ------ add extra mass ------------

        # extra node inertia data
        N = self.midx + 1

        tower.changeExtraNodeMass(N, self.m, self.mIxx, self.mIyy, self.mIzz, self.mIxy, self.mIxz, self.mIyz,
            self.mrhox, self.mrhoy, self.mrhoz, self.addGravityLoadForExtraMass)

        # ------------------------------------

        # ------- enable dynamic analysis ----------
        tower.enableDynamics(self.nM, self.Mmethod, self.lump, self.tol, self.shift)
        # ----------------------------

        # ------ static load case 1 ------------

        # gravity in the X, Y, Z, directions (global)
        gx = 0.0
        gy = 0.0
        gz = -self.g

        load = frame3dd.StaticLoadCase(gx, gy, gz)


        # point loads
        nF = self.plidx + 1
        load.changePointLoads(nF, self.Fx, self.Fy, self.Fz, self.Mxx, self.Myy, self.Mzz)


        # distributed loads
        Px, Py, Pz = self.WWloads.Pz, self.WWloads.Py, -self.WWloads.Px  # switch to local c.s.
        z = self.z

        # trapezoidally distributed loads
        EL = np.arange(1, n)
        xx1 = np.zeros(n-1)
        xx2 = z[1:] - z[:-1] - 1e-6  # subtract small number b.c. of precision
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

        # deflections due to loading (from tower top and wind/wave loads)
        self.top_deflection = displacements.dx[iCase, n-1]  # in yaw-aligned direction


        # shear and bending (convert from local to global c.s.)
        Fz = forces.Nx[iCase, :]
        Vy = forces.Vy[iCase, :]
        Vx = -forces.Vz[iCase, :]

        Mzz = forces.Txx[iCase, :]
        Myy = forces.Myy[iCase, :]
        Mxx = -forces.Mzz[iCase, :]

        # one per element (first negative b.c. need reaction)
        Fz = np.concatenate([[-Fz[0]], Fz[1::2]])
        Vx = np.concatenate([[-Vx[0]], Vx[1::2]])
        Vy = np.concatenate([[-Vy[0]], Vy[1::2]])

        Mzz = np.concatenate([[-Mzz[0]], Mzz[1::2]])
        Myy = np.concatenate([[-Myy[0]], Myy[1::2]])
        Mxx = np.concatenate([[-Mxx[0]], Mxx[1::2]])

        # axial and shear stress
        ##R = self.d/2.0
        ##x_stress = R*np.cos(self.theta_stress)
        ##y_stress = R*np.sin(self.theta_stress)
        ##axial_stress = Fz/self.Az + Mxx/self.Ixx*y_stress - Myy/self.Iyy*x_stress
#        V = Vy*x_stress/R - Vx*y_stress/R  # shear stress orthogonal to direction x,y
#        shear_stress = 2. * V / self.Az  # coefficient of 2 for a hollow circular section, but should be conservative for other shapes
        axial_stress = Fz/A - np.sqrt(Mxx**2+Myy**2)/Iyy*self.d/2.0  #More conservative, just use the tilted bending and add total max shear as well at the same point
        shear_stress = 2. * np.sqrt(Vx**2+Vy**2) / A # coefficient of 2 for a hollow circular section, but should be conservative for other shapes

        # hoop_stress (Eurocode method)
        hoop_stress = hoopStressEurocode(self.z, self.d, self.t, self.L_reinforced, self.WWloads.qdyn)

        # von mises stress
        self.stress = vonMisesStressUtilization(axial_stress, hoop_stress, shear_stress,
                      self.gamma_f*self.gamma_m*self.gamma_n, self.sigma_y)

        # shell buckling
        self.shell_buckling = shellBucklingEurocode(self.d, self.t, axial_stress, hoop_stress,
                      shear_stress, self.L_reinforced, self.E, self.sigma_y, self.gamma_f, self.gamma_b)

        # global buckling
        tower_height = self.z[-1] - self.z[0]
        M = np.sqrt(Mxx**2 + Myy**2)
        self.global_buckling = bucklingGL(self.d, self.t, Fz, M, tower_height, self.E,
            self.sigma_y, self.gamma_f, self.gamma_b)

        # fatigue
        N_DEL = [365*24*3600*self.life]*len(z)
        damage=np.zeros(z.size)
        if any(self.M_DEL):
            M_DEL = np.interp(z, self.z_DEL, self.M_DEL)

            damage = fatigue(M_DEL, N_DEL, d, t, self.m_SN, self.DC, self.gamma_fatigue, stress_factor=1.0, weld_factor=True)

        # TODO: more hack NOT SURE WHAT THIS IS, but it was there originally, commented out for now
#        damage = np.concatenate([np.zeros(len(self.z)-len(z)), damage])


# -----------------
#  Assembly
# -----------------



class TowerSE(Assembly):

    # geometry parameters
    z_param = Array(iotype='in', units='m', desc='parameterized locations along tower, linear lofting between')
    d_param = Array(iotype='in', units='m', desc='tower diameter at corresponding locations')
    t_param = Array(iotype='in', units='m', desc='shell thickness at corresponding locations')


    # geometry
    z_full = Array(iotype='in', units='m', desc='locations along tower')
    L_reinforced = Array(iotype='in', units='m')
    theta_stress = Array(iotype='in', units='deg', desc='location along azimuth where stress should be evaluated.  0 corresponds to +x axis.  follows unit circle direction and c.s.')

    # wind/wave
    wind_rho = Float(1.225, iotype='in', units='kg/m**3', desc='air density')
    wind_mu = Float(1.7934e-5, iotype='in', units='kg/(m*s)', desc='dynamic viscosity of air')
    wind_Uref1 = Float(iotype='in', units='m/s', desc='reference wind speed (usually at hub height)')
    wind_Uref2 = Float(iotype='in', units='m/s', desc='reference wind speed (usually at hub height)')
    wind_zref = Float(iotype='in', units='m', desc='corresponding reference height')
    wind_z0 = Float(0.0, iotype='in', units='m', desc='bottom of wind profile (height of ground/sea)')
    wind_cd = Array(iotype='in', desc='Cd coefficient, if left blank it will be calculated based on cylinder Re')

    wave_rho = Float(1027.0, iotype='in', units='kg/m**3', desc='water density')
    wave_mu = Float(1.3351e-3, iotype='in', units='kg/(m*s)', desc='dynamic viscosity of water')
    wave_cm = Float(2.0, iotype='in', desc='mass coefficient')
    wave_cd = Array(iotype='in', desc='Cd coefficient, if left blank it will be calculated based on cylinder Re')

    yaw = Float(0.0, iotype='in', units='deg', desc='yaw angle')

    # material props
    E = Array(iotype='in', units='N/m**2', desc='modulus of elasticity')
    G = Array(iotype='in', units='N/m**2', desc='shear modulus')
    rho = Array(iotype='in', units='kg/m**3', desc='material density')
    sigma_y = Array(iotype='in', units='N/m**2', desc='yield stress')

    # spring reaction data.  Use float('inf') for rigid constraints.
    kidx = Array(iotype='in', desc='indices of z where external stiffness reactions should be applied.')
    kx = Array(iotype='in', units='m', desc='spring stiffness in x-direction')
    ky = Array(iotype='in', units='m', desc='spring stiffness in y-direction')
    kz = Array(iotype='in', units='m', desc='spring stiffness in z-direction')
    ktx = Array(iotype='in', units='m', desc='spring stiffness in theta_x-rotation')
    kty = Array(iotype='in', units='m', desc='spring stiffness in theta_y-rotation')
    ktz = Array(iotype='in', units='m', desc='spring stiffness in theta_z-rotation')

    # extra mass
    midx = Array(iotype='in', desc='indices where added mass should be applied.')
    m = Array(iotype='in', units='kg', desc='added mass')
    mIxx = Array(iotype='in', units='kg*m**2', desc='x mass moment of inertia about some point p')
    mIyy = Array(iotype='in', units='kg*m**2', desc='y mass moment of inertia about some point p')
    mIzz = Array(iotype='in', units='kg*m**2', desc='z mass moment of inertia about some point p')
    mIxy = Array(iotype='in', units='kg*m**2', desc='xy mass moment of inertia about some point p')
    mIxz = Array(iotype='in', units='kg*m**2', desc='xz mass moment of inertia about some point p')
    mIyz = Array(iotype='in', units='kg*m**2', desc='yz mass moment of inertia about some point p')
    mrhox = Array(iotype='in', units='m', desc='x-location of p relative to node')
    mrhoy = Array(iotype='in', units='m', desc='y-location of p relative to node')
    mrhoz = Array(iotype='in', units='m', desc='z-location of p relative to node')
    addGravityLoadForExtraMass = Bool(True, iotype='in', desc='add gravitational load')

    # gravitational load
    g = Float(9.81, iotype='in', units='m/s**2', desc='acceleration of gravity (magnitude)')

    # safety factors
    gamma_f = Float(1.35, iotype='in', desc='safety factor on loads')
    gamma_m = Float(1.1, iotype='in', desc='safety factor on materials')
    gamma_n = Float(1.0, iotype='in', desc='safety factor on consequence of failure')
    gamma_b = Float(1.1, iotype='in', desc='buckling safety factor')
    gamma_fatigue = Float(1.755, iotype='in', desc='total safety factor for fatigue')

    # replace
    wind1 = Slot(WindBase)
    wind2 = Slot(WindBase)
    wave1 = Slot(WaveBase)
    wave2 = Slot(WaveBase)

    # point loads (if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also)
    plidx1 = Array(iotype='in', desc='indices where point loads should be applied.')
    Fx1 = Array(iotype='in', units='N', desc='point force in x-direction')
    Fy1 = Array(iotype='in', units='N', desc='point force in y-direction')
    Fz1 = Array(iotype='in', units='N', desc='point force in z-direction')
    Mxx1 = Array(iotype='in', units='N*m', desc='point moment about x-axis')
    Myy1 = Array(iotype='in', units='N*m', desc='point moment about y-axis')
    Mzz1 = Array(iotype='in', units='N*m', desc='point moment about z-axis')

    plidx2 = Array(iotype='in', desc='indices where point loads should be applied.')
    Fx2 = Array(iotype='in', units='N', desc='point force in x-direction')
    Fy2 = Array(iotype='in', units='N', desc='point force in y-direction')
    Fz2 = Array(iotype='in', units='N', desc='point force in z-direction')
    Mxx2 = Array(iotype='in', units='N*m', desc='point moment about x-axis')
    Myy2 = Array(iotype='in', units='N*m', desc='point moment about y-axis')
    Mzz2 = Array(iotype='in', units='N*m', desc='point moment about z-axis')

    # constraint parameters
    min_d_to_t = Float(120.0, iotype='in')
    min_taper = Float(0.4, iotype='in')

    # fatigue parameters
    life = Float(20.0, iotype='in', desc='fatigue life of tower')
    m_SN = Int(4, iotype='in', desc='slope of S/N curve')
    DC = Float(80.0, iotype='in', desc='standard value of stress')
    z_DEL = Array(iotype='in')
    M_DEL = Array(iotype='in')

    # frame3ddd options
    shear = Bool(True, iotype='in', desc='include shear deformation')
    geom = Bool(False, iotype='in', desc='include geometric stiffness')
    dx = Float(5.0, iotype='in', desc='z-axis increment for internal forces')
    nM = Int(2, iotype='in', desc='number of desired dynamic modes of vibration (below only necessary if nM > 0)')
    Mmethod = Int(1, iotype='in', desc='1: subspace Jacobi, 2: Stodola')
    lump = Int(0, iotype='in', desc='0: consistent mass, 1: lumped mass matrix')
    tol = Float(1e-9, iotype='in', desc='mode shape tolerance')
    shift = Float(0.0, iotype='in', desc='shift value ... for unrestrained structures')

    # outputs
    mass = Float(iotype='out', units='kg')
    f1 = Float(iotype='out', units='Hz', desc='First natural frequency')
    f2 = Float(iotype='out', units='Hz', desc='Second natural frequency')
    top_deflection1 = Float(iotype='out', units='m', desc='Deflection of tower top in yaw-aligned +x direction')
    top_deflection2 = Float(iotype='out', units='m', desc='Deflection of tower top in yaw-aligned +x direction')
    stress1 = Array(iotype='out', units='N/m**2', desc='Von Mises stress along tower on downwind side (yaw-aligned +x).  Normalized by yield stress.  Includes safety factors.')
    stress2 = Array(iotype='out', units='N/m**2', desc='Von Mises stress along tower on downwind side (yaw-aligned +x).  Normalized by yield stress.  Includes safety factors.')
    shell_buckling1 = Array(iotype='out', desc='Shell buckling constraint load case #1.  Should be < 1 for feasibility.  Includes safety factors')
    shell_buckling2 = Array(iotype='out', desc='Shell buckling constraint load case #2.  Should be < 1 for feasibility.  Includes safety factors')
    global_buckling1 = Array(iotype='out', desc='Global buckling constraint load case #1.  Should be < 1 for feasibility.  Includes safety factors')
    global_buckling2 = Array(iotype='out', desc='Global buckling constraint load case #2.  Should be < 1 for feasibility.  Includes safety factors')
    damage = Array(iotype='out', desc='Fatigue damage at each tower section')
    weldability = Array(iotype='out')
    manufacturability = Array(iotype='out')


    def configure(self):

        self.add('geometry', TowerDiscretization())
        # two load cases.  TODO: use a case iterator
        self.add('wind1', WindBase())
        self.add('wind2', WindBase())
        self.add('wave1', WaveBase())
        self.add('wave2', WaveBase())
        self.add('windLoads1', TowerWindDrag())
        self.add('windLoads2', TowerWindDrag())
        self.add('waveLoads1', TowerWaveDrag())
        self.add('waveLoads2', TowerWaveDrag())
        self.add('distLoads1', AeroHydroLoads())
        self.add('distLoads2', AeroHydroLoads())
        self.add('props', CylindricalShellProperties())
        self.add('tower1', TowerFrame3DD())
        self.add('tower2', TowerFrame3DD())
        self.add('gc', GeometricConstraints())

        self.driver.workflow.add(['geometry', 'wind1', 'wind2', 'wave1', 'wave2',
                'windLoads1', 'windLoads2', 'waveLoads1', 'waveLoads2', 'distLoads1', 'distLoads2',
                'geometry', 'props', 'tower1', 'tower2', 'gc'])


        # connections to geometry
        self.connect('z_param', 'geometry.z_param')
        self.connect('d_param', 'geometry.d_param')
        self.connect('t_param', 'geometry.t_param')
        self.connect('z_full', 'geometry.z_full')

        # connections to wind1
        self.connect('geometry.z_full', 'wind1.z')
        self.connect('wind_Uref1', 'wind1.Uref')
        self.connect('wind_zref', 'wind1.zref')
        self.connect('wind_z0', 'wind1.z0')

        # connections to wind2
        self.connect('geometry.z_full', 'wind2.z')
        self.connect('wind_Uref2', 'wind2.Uref')
        self.connect('wind_zref', 'wind2.zref')
        self.connect('wind_z0', 'wind2.z0')

        # connections to wave1 and wave2
        self.connect('geometry.z_full', 'wave1.z')
        self.connect('geometry.z_full', 'wave2.z')

        # connections to windLoads1
        self.connect('wind1.U', 'windLoads1.U')
        self.connect('geometry.z_full', 'windLoads1.z')
        self.connect('geometry.d_full', 'windLoads1.d')
        self.connect('wind1.beta', 'windLoads1.beta')
        self.connect('wind_rho', 'windLoads1.rho')
        self.connect('wind_mu', 'windLoads1.mu')
        self.connect('wind_cd', 'windLoads1.cd_usr')

        # connections to windLoads2
        self.connect('wind2.U', 'windLoads2.U')
        self.connect('geometry.z_full', 'windLoads2.z')
        self.connect('geometry.d_full', 'windLoads2.d')
        self.connect('wind2.beta', 'windLoads2.beta')
        self.connect('wind_rho', 'windLoads2.rho')
        self.connect('wind_mu', 'windLoads2.mu')
        self.connect('wind_cd', 'windLoads2.cd_usr')

        # connections to waveLoads1
        self.connect('wave1.U', 'waveLoads1.U')
        self.connect('wave1.A', 'waveLoads1.A')
        self.connect('geometry.z_full', 'waveLoads1.z')
        self.connect('geometry.d_full', 'waveLoads1.d')
        self.connect('wave1.beta', 'waveLoads1.beta')
        self.connect('wave_rho', 'waveLoads1.rho')
        self.connect('wave_mu', 'waveLoads1.mu')
        self.connect('wave_cm', 'waveLoads1.cm')
        self.connect('wave_cd', 'waveLoads1.cd_usr')

        # connections to waveLoads2
        self.connect('wave2.U', 'waveLoads2.U')
        self.connect('wave2.A', 'waveLoads2.A')
        self.connect('geometry.z_full', 'waveLoads2.z')
        self.connect('geometry.d_full', 'waveLoads2.d')
        self.connect('wave2.beta', 'waveLoads2.beta')
        self.connect('wave_rho', 'waveLoads2.rho')
        self.connect('wave_mu', 'waveLoads2.mu')
        self.connect('wave_cm', 'waveLoads2.cm')

        self.connect('wave_cd', 'waveLoads2.cd_usr')

        # connections to distLoads1
        self.connect('windLoads1.windLoads', 'distLoads1.windLoads')
        self.connect('waveLoads1.waveLoads', 'distLoads1.waveLoads')
        self.connect('geometry.z_full', 'distLoads1.z')
        self.connect('yaw', 'distLoads1.yaw')

        # connections to distLoads2
        self.connect('windLoads2.windLoads', 'distLoads2.windLoads')
        self.connect('waveLoads2.waveLoads', 'distLoads2.waveLoads')
        self.connect('geometry.z_full', 'distLoads2.z')
        self.connect('yaw', 'distLoads2.yaw')

        # connections to props
        self.connect('geometry.d_full', 'props.d')
        self.connect('geometry.t_full', 'props.t')

        # connect to tower1
        self.connect('z_full', 'tower1.z')
        self.connect('props.Az', 'tower1.Az')
        self.connect('props.Asx', 'tower1.Asx')
        self.connect('props.Asy', 'tower1.Asy')
        self.connect('props.Jz', 'tower1.Jz')
        self.connect('props.Ixx', 'tower1.Ixx')
        self.connect('props.Iyy', 'tower1.Iyy')
        self.connect('E', 'tower1.E')
        self.connect('G', 'tower1.G')
        self.connect('rho', 'tower1.rho')
        self.connect('sigma_y', 'tower1.sigma_y')
        self.connect('geometry.d_full', 'tower1.d')
        self.connect('geometry.t_full', 'tower1.t')
        self.connect('L_reinforced', 'tower1.L_reinforced')
        self.connect('theta_stress', 'tower1.theta_stress')
        self.connect('kidx', 'tower1.kidx')
        self.connect('kx', 'tower1.kx')
        self.connect('ky', 'tower1.ky')
        self.connect('kz', 'tower1.kz')
        self.connect('ktx', 'tower1.ktx')
        self.connect('kty', 'tower1.kty')
        self.connect('ktz', 'tower1.ktz')
        self.connect('midx', 'tower1.midx')
        self.connect('m', 'tower1.m')
        self.connect('mIxx', 'tower1.mIxx')
        self.connect('mIyy', 'tower1.mIyy')
        self.connect('mIzz', 'tower1.mIzz')
        self.connect('mIxy', 'tower1.mIxy')
        self.connect('mIxz', 'tower1.mIxz')
        self.connect('mIyz', 'tower1.mIyz')
        self.connect('mrhox', 'tower1.mrhox')
        self.connect('mrhoy', 'tower1.mrhoy')
        self.connect('mrhoz', 'tower1.mrhoz')
        self.connect('addGravityLoadForExtraMass', 'tower1.addGravityLoadForExtraMass')
        self.connect('g', 'tower1.g')
        self.connect('plidx1', 'tower1.plidx')
        self.connect('Fx1', 'tower1.Fx')
        self.connect('Fy1', 'tower1.Fy')
        self.connect('Fz1', 'tower1.Fz')
        self.connect('Mxx1', 'tower1.Mxx')
        self.connect('Myy1', 'tower1.Myy')
        self.connect('Mzz1', 'tower1.Mzz')
        ##self.connect('distLoads1.Px',   'tower1.Px')
        ##self.connect('distLoads1.Py',   'tower1.Py')
        ##self.connect('distLoads1.Pz',   'tower1.Pz')
        ##self.connect('distLoads1.qdyn', 'tower1.qdyn')
        self.connect('distLoads1.outloads', 'tower1.WWloads')

        self.connect('gamma_f', 'tower1.gamma_f')
        self.connect('gamma_m', 'tower1.gamma_m')
        self.connect('gamma_n', 'tower1.gamma_n')
        self.connect('gamma_b', 'tower1.gamma_b')
        self.connect('life', 'tower1.life')
        self.connect('m_SN', 'tower1.m_SN')
        self.connect('DC', 'tower1.DC')
        self.connect('z_DEL', 'tower1.z_DEL')
        self.connect('M_DEL', 'tower1.M_DEL')
        self.connect('gamma_fatigue', 'tower1.gamma_fatigue')
        self.connect('shear', 'tower1.shear')
        self.connect('geom', 'tower1.geom')
        self.connect('dx', 'tower1.dx')
        self.connect('nM', 'tower1.nM')
        self.connect('Mmethod', 'tower1.Mmethod')
        self.connect('lump', 'tower1.lump')
        self.connect('tol', 'tower1.tol')
        self.connect('shift', 'tower1.shift')

        # connect to tower2
        self.connect('z_full', 'tower2.z')
        self.connect('props.Az', 'tower2.Az')
        self.connect('props.Asx', 'tower2.Asx')
        self.connect('props.Asy', 'tower2.Asy')
        self.connect('props.Jz', 'tower2.Jz')
        self.connect('props.Ixx', 'tower2.Ixx')
        self.connect('props.Iyy', 'tower2.Iyy')
        self.connect('E', 'tower2.E')
        self.connect('G', 'tower2.G')
        self.connect('rho', 'tower2.rho')
        self.connect('sigma_y', 'tower2.sigma_y')
        self.connect('geometry.d_full', 'tower2.d')
        self.connect('geometry.t_full', 'tower2.t')
        self.connect('L_reinforced', 'tower2.L_reinforced')
        self.connect('theta_stress', 'tower2.theta_stress')
        self.connect('kidx', 'tower2.kidx')
        self.connect('kx', 'tower2.kx')
        self.connect('ky', 'tower2.ky')
        self.connect('kz', 'tower2.kz')
        self.connect('ktx', 'tower2.ktx')
        self.connect('kty', 'tower2.kty')
        self.connect('ktz', 'tower2.ktz')
        self.connect('midx', 'tower2.midx')
        self.connect('m', 'tower2.m')
        self.connect('mIxx', 'tower2.mIxx')
        self.connect('mIyy', 'tower2.mIyy')
        self.connect('mIzz', 'tower2.mIzz')
        self.connect('mIxy', 'tower2.mIxy')
        self.connect('mIxz', 'tower2.mIxz')
        self.connect('mIyz', 'tower2.mIyz')
        self.connect('mrhox', 'tower2.mrhox')
        self.connect('mrhoy', 'tower2.mrhoy')
        self.connect('mrhoz', 'tower2.mrhoz')
        self.connect('addGravityLoadForExtraMass', 'tower2.addGravityLoadForExtraMass')
        self.connect('g', 'tower2.g')
        self.connect('plidx2', 'tower2.plidx')
        self.connect('Fx2', 'tower2.Fx')
        self.connect('Fy2', 'tower2.Fy')
        self.connect('Fz2', 'tower2.Fz')
        self.connect('Mxx2', 'tower2.Mxx')
        self.connect('Myy2', 'tower2.Myy')
        self.connect('Mzz2', 'tower2.Mzz')
        ##self.connect('distLoads2.Px', 'tower2.Px')
        ##self.connect('distLoads2.Py', 'tower2.Py')
        ##self.connect('distLoads2.Pz', 'tower2.Pz')
        ##self.connect('distLoads2.qdyn', 'tower2.qdyn')
        self.connect('distLoads2.outloads', 'tower2.WWloads')

        self.connect('gamma_f', 'tower2.gamma_f')
        self.connect('gamma_m', 'tower2.gamma_m')
        self.connect('gamma_n', 'tower2.gamma_n')
        self.connect('gamma_b', 'tower2.gamma_b')
        self.connect('life', 'tower2.life')
        self.connect('m_SN', 'tower2.m_SN')
        self.connect('DC', 'tower2.DC')
        self.connect('z_DEL', 'tower2.z_DEL')
        self.connect('M_DEL', 'tower2.M_DEL')
        self.connect('gamma_fatigue', 'tower2.gamma_fatigue')
        self.connect('shear', 'tower2.shear')
        self.connect('geom', 'tower2.geom')
        self.connect('dx', 'tower2.dx')
        self.connect('nM', 'tower2.nM')
        self.connect('Mmethod', 'tower2.Mmethod')
        self.connect('lump', 'tower2.lump')
        self.connect('tol', 'tower2.tol')
        self.connect('shift', 'tower2.shift')

        # connections to gc
        self.connect('d_param', 'gc.d')
        self.connect('t_param', 'gc.t')
        self.connect('min_d_to_t', 'gc.min_d_to_t')
        self.connect('min_taper', 'gc.min_taper')


        # outputs
        self.connect('tower1.mass', 'mass')
        self.connect('tower1.f1', 'f1')
        self.connect('tower1.f2', 'f2')
        self.connect('tower1.top_deflection', 'top_deflection1')
        self.connect('tower2.top_deflection', 'top_deflection2')
        self.connect('tower1.stress', 'stress1')
        self.connect('tower2.stress', 'stress2')
        self.connect('tower1.global_buckling', 'global_buckling1')
        self.connect('tower2.global_buckling', 'global_buckling2')
        self.connect('tower1.shell_buckling', 'shell_buckling1')
        self.connect('tower2.shell_buckling', 'shell_buckling2')
        self.connect('tower1.damage', 'damage')
        self.connect('gc.weldability', 'weldability')
        self.connect('gc.manufacturability', 'manufacturability')


if __name__ == '__main__':


    optimize = False

    # --- tower setup ------
    from commonse.environment import PowerWind

    tower = TowerSE()


    # ---- tower ------
    tower.replace('wind1', PowerWind())
    tower.replace('wind2', PowerWind())
    # onshore (no waves)

    # --- geometry ----
    tower.z_param = [0.0, 43.8, 87.6]
    tower.d_param = [6.0, 4.935, 3.87]
    tower.t_param = [0.027*1.3, 0.023*1.3, 0.019*1.3]
    n = 15
    tower.z_full = np.linspace(0.0, 87.6, n)
    tower.L_reinforced = 30.0*np.ones(n)  # [m] buckling length
    tower.theta_stress = 0.0*np.ones(n)
    tower.yaw = 0.0

    # --- material props ---
    tower.E = 210e9*np.ones(n)
    tower.G = 80.8e9*np.ones(n)
    tower.rho = 8500.0*np.ones(n)
    tower.sigma_y = 450.0e6*np.ones(n)

    # --- spring reaction data.  Use float('inf') for rigid constraints. ---
    tower.kidx = [0]  # applied at base
    tower.kx = [float('inf')]
    tower.ky = [float('inf')]
    tower.kz = [float('inf')]
    tower.ktx = [float('inf')]
    tower.kty = [float('inf')]
    tower.ktz = [float('inf')]

    # --- extra mass ----
    tower.midx = [n-1]  # RNA mass at top
    tower.m = [285598.8]
    tower.mIxx = [1.14930678e+08]
    tower.mIyy = [2.20354030e+07]
    tower.mIzz = [1.87597425e+07]
    tower.mIxy = [0.00000000e+00]
    tower.mIxz = [5.03710467e+05]
    tower.mIyz = [0.00000000e+00]
    tower.mrhox = [-1.13197635]
    tower.mrhoy = [0.]
    tower.mrhoz = [0.50875268]
    tower.addGravityLoadForExtraMass = True
    # -----------

    # --- wind ---
    tower.wind_zref = 90.0
    tower.wind_z0 = 0.0
    tower.wind1.shearExp = 0.2
    tower.wind2.shearExp = 0.2
    # ---------------

    # if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also
    # # --- loading case 1: max Thrust ---
    tower.wind_Uref1 = 11.73732
    tower.plidx1 = [n-1]  # at tower top
    tower.Fx1 = [1284744.19620519]
    tower.Fy1 = [0.]
    tower.Fz1 = [-2914124.84400512]
    tower.Mxx1 = [3963732.76208099]
    tower.Myy1 = [-2275104.79420872]
    tower.Mzz1 = [-346781.68192839]
    # # ---------------

    # # --- loading case 2: max wind speed ---
    tower.wind_Uref2 = 70.0
    tower.plidx1 = [n-1]  # at tower top
    tower.Fx1 = [930198.60063279]
    tower.Fy1 = [0.]
    tower.Fz1 = [-2883106.12368949]
    tower.Mxx1 = [-1683669.22411597]
    tower.Myy1 = [-2522475.34625363]
    tower.Mzz1 = [147301.97023764]
    # # ---------------

    # --- safety factors ---
    tower.gamma_f = 1.35
    tower.gamma_m = 1.3
    tower.gamma_n = 1.0
    tower.gamma_b = 1.1
    # ---------------

    # --- fatigue ---
    tower.z_DEL = np.array([0.000, 1.327, 3.982, 6.636, 9.291, 11.945, 14.600, 17.255, 19.909, 22.564, 25.218, 27.873, 30.527, 33.182, 35.836, 38.491, 41.145, 43.800, 46.455, 49.109, 51.764, 54.418, 57.073, 59.727, 62.382, 65.036, 67.691, 70.345, 73.000, 75.655, 78.309, 80.964, 83.618, 86.273, 87.600])
    tower.M_DEL = 1e3*np.array([8.2940E+003, 8.1518E+003, 7.8831E+003, 7.6099E+003, 7.3359E+003, 7.0577E+003, 6.7821E+003, 6.5119E+003, 6.2391E+003, 5.9707E+003, 5.7070E+003, 5.4500E+003, 5.2015E+003, 4.9588E+003, 4.7202E+003, 4.4884E+003, 4.2577E+003, 4.0246E+003, 3.7942E+003, 3.5664E+003, 3.3406E+003, 3.1184E+003, 2.8977E+003, 2.6811E+003, 2.4719E+003, 2.2663E+003, 2.0673E+003, 1.8769E+003, 1.7017E+003, 1.5479E+003, 1.4207E+003, 1.3304E+003, 1.2780E+003, 1.2673E+003, 1.2761E+003])
    tower.gamma_fatigue = 1.35*1.3*1.0
    tower.life = 20.0
    tower.m_SN = 4
    # ---------------

    # --- constraints ---
    tower.min_d_to_t = 120.0
    tower.min_taper = 0.4
    # ---------------

    # # V_max = 80.0  # tip speed
    # # D = 126.0
    # # tower.freq1p = V_max / (D/2) / (2*pi)  # convert to Hz


    # # --- run ---
    tower.run()

    print 'mass (kg) =', tower.mass
    print 'f1 (Hz) =', tower.f1
    print 'f2 (Hz) =', tower.f2
    print 'top_deflection1 (m) =', tower.top_deflection1
    print 'top_deflection2 (m) =', tower.top_deflection2
    print 'weldability =', tower.weldability
    print 'manufacturability =', tower.manufacturability
    print 'stress1 =', tower.stress1
    print 'stress1 =', tower.stress2
    print 'zs=', tower.tower1.z
    print 'ds=', tower.tower1.d
    print 'ts=', tower.tower1.t
    print 'GL buckling =', tower.global_buckling1
    print 'GL buckling =', tower.global_buckling2
    print 'Shell buckling =', tower.shell_buckling1
    print 'Shell buckling =', tower.shell_buckling2
    print 'damage =', tower.damage

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.0, 3.5))
    plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
    plt.plot(tower.stress1, tower.z_full, label='stress1')
    plt.plot(tower.stress2, tower.z_full, label='stress2')
    plt.plot(tower.shell_buckling1, tower.z_full, label='shell buckling 1')
    plt.plot(tower.shell_buckling2, tower.z_full, label='shell buckling 2')
    plt.plot(tower.global_buckling1, tower.z_full, label='global buckling 1')
    plt.plot(tower.global_buckling2, tower.z_full, label='global buckling 2')
    plt.plot(tower.damage, tower.z_full, label='damage')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2)
    plt.xlabel('utilization')
    plt.ylabel('height along tower (m)')
    plt.show()
    # ------------


    if optimize:

        # --- optimizer imports ---
        from pyopt_driver.pyopt_driver import pyOptDriver
        from openmdao.lib.casehandlers.api import DumpCaseRecorder
        # ----------------------

        # --- Setup Pptimizer ---
        tower.replace('driver', pyOptDriver())
        tower.driver.optimizer = 'SNOPT'
        tower.driver.options = {'Major feasibility tolerance': 1e-6,
                               'Minor feasibility tolerance': 1e-6,
                               'Major optimality tolerance': 1e-5,
                               'Function precision': 1e-8}
        # ----------------------

        # --- Objective ---
        tower.driver.add_objective('tower1.mass / 300000')
        # ----------------------

        # --- Design Variables ---
        tower.driver.add_parameter('z_param[1]', low=0.0, high=87.0)
        tower.driver.add_parameter('d_param[:-1]', low=3.87, high=20.0)
        tower.driver.add_parameter('t_param', low=0.005, high=0.2)
        # ----------------------

        # --- recorder ---
        tower.recorders = [DumpCaseRecorder()]
        # ----------------------

        # --- Constraints ---
        tower.driver.add_constraint('tower1.stress <= 1.0')
        tower.driver.add_constraint('tower2.stress <= 1.0')
        tower.driver.add_constraint('tower1.global_buckling <= 1.0')
        tower.driver.add_constraint('tower2.global_buckling <= 1.0')
        tower.driver.add_constraint('tower1.shell_buckling <= 1.0')
        tower.driver.add_constraint('tower2.shell_buckling <= 1.0')
        tower.driver.add_constraint('tower1.damage <= 1.0')
        tower.driver.add_constraint('gc.weldability <= 0.0')
        tower.driver.add_constraint('gc.manufacturability <= 0.0')
        freq1p = 0.2  # 1P freq in Hz
        tower.driver.add_constraint('tower1.f1 >= 1.1*%f' % freq1p)
        # ----------------------

        # --- run opt ---
        tower.run()
        # ---------------

