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
          - 1/2018 : G.B. modified for easier use with other modules, reducing user input burden, and shifting more to commonse
          - 7/2018 : E.G. converted to pure python implimentation for new fusedwind wrapper
 """

import numpy as np
from openmdao.api import Component, Group, Problem, IndepVarComp

from commonse.pyWindWaveDrag import AeroHydroLoads, CylinderWindDrag, CylinderWaveDrag

from commonse.pyenvironment import WindBase, WaveBase, SoilBase, PowerWind, LogWind
from commonse.pytube import CylindricalShellProperties
from commonse.utilities import assembleI, unassembleI
from commonse import gravity, eps

from commonse.pyvertical_cylinder import CylinderDiscretization, CylinderMass, CylinderFrame3DD # TODO: add python object version
#from fusedwind.turbine.tower import TowerFromCSProps
#from fusedwind.interface import implement_base

import commonse.pyUtilizationSupplement as Util


# -----------------
#  Components
# -----------------

class TowerDiscretization(object):
    def __init__(self):
        super(TowerDiscretization, self).__init__()
        # Parameters
        # self.add_param('hub_height', val=0.0, units='m', desc='diameter at tower base')
        # self.add_param('z_end', val=0.0, units='m', desc='Last node point on tower')
        
        # Output
        self.height_constraint = 0.0 # self.add_output('height_constraint', val=0.0, units='m', desc='mismatch between tower height and desired hub_height')

    def compute(self, hub_height, z_end):
        self.height_constraint = hub_height - z_end

        # derivatives
        self.d_ConH_d_hubHeight = 1
        self.d_ConH_d_zEnd = -1

    def provideJ(self):
        self.J = np.array([[self.d_ConH_d_hubHeight],[self.d_ConH_d_zEnd]])

        return self.J
        
class TowerMass(object):

    def __init__(self, nPoints):
        super(TowerMass, self).__init__()
        # Parameters
        # self.add_param('cylinder_mass', val=np.zeros(nPoints-1), units='kg', desc='Total cylinder mass')
        # self.add_param('cylinder_center_of_mass', val=0.0, units='m', desc='z-position of center of mass of cylinder')
        # self.add_param('cylinder_section_center_of_mass', val=np.zeros(nPoints-1), units='m', desc='z position of center of mass of each can in the cylinder')
        # self.add_param('cylinder_I_base', np.zeros((6,)), units='kg*m**2', desc='mass moment of inertia of cylinder about base [xx yy zz xy xz yz]')
        
        # Output
        self.tower_mass = 0.0 # self.add_output('tower_mass', val=0.0, units='kg', desc='Total tower mass')
        self.tower_center_of_mass = 0.0 # self.add_output('tower_center_of_mass', val=0.0, units='m', desc='z-position of center of mass of tower')
        self.tower_section_center_of_mass = np.zeros(nPoints-1) # self.add_output('tower_section_center_of_mass', val=np.zeros(nPoints-1), units='m', desc='z position of center of mass of each can in the tower')
        self.tower_I_base = np.zeros((6,)) # self.add_output('tower_I_base', np.zeros((6,)), units='kg*m**2', desc='mass moment of inertia of tower about base [xx yy zz xy xz yz]')
        
    def compute(self, cylinder_mass, cylinder_center_of_mass, cylinder_section_center_of_mass, cylinder_I_base):

        self.tower_mass = cylinder_mass.sum()
        self.tower_center_of_mass = cylinder_center_of_mass
        self.tower_section_center_of_mass = cylinder_section_center_of_mass
        self.tower_I_base = cylinder_I_base

        # derivatives
        npts = len(cylinder_section_center_of_mass)
        zeroPts = np.zeros(npts)
        zero6 = np.zeros(6)

        self.d_towerM_d_cylM = np.ones(len(cylinder_mass))
        self.d_towerM_d_cylCM = 0.0
        self.d_towerM_d_cylSecCtM = zeroPts 
        self.d_towerM_d_cylIbase = zero6

        self.d_towerCM_d_cylM = 0.0
        self.d_towerCM_d_cylCM = 1.0
        self.d_towerCM_d_cylSecCM = zeroPts
        self.d_towerCM_d_cylIbase = zero6

        self.d_towerSecCM_d_cylM = 0.0
        self.d_towerSecCM_d_cylCM = 0.0
        self.d_towerSecCM_d_cylSecCM = np.eye(npts)
        self.d_towerSecCM_d_cylIbase = np.zeros((npts,6))

        self.d_towerIbase_d_cylM = 1.0
        self.d_towerIbase_d_cylCM = 0.0
        self.d_towerIbase_d_cylSecCM = np.zeros((6, npts))
        self.d_towerIbase_d_cylIbase = np.eye(len(cylinder_I_base))

    def provideJ(self):

        self.J = np.array([[self.d_towerM_d_cylM, self.d_towerM_d_cylCM, self.d_towerM_d_cylSecCtM, self.d_towerM_d_cylIbase],\
                           [self.d_towerCM_d_cylM, self.d_towerCM_d_cylCM, self.d_towerCM_d_cylSecCtM, self.d_towerCM_d_cylIbase],\
                           [self.d_towerSecCM_d_cylM, self.d_towerSecCM_d_cylCM, self.d_towerSecCM_d_cylSecCtM, self.d_towerSecCM_d_cylIbase],\
                           [self.d_towerIbase_d_cylM, self.d_towerIbase_d_cylCM, self.d_towerIbase_d_cylSecCtM, self.d_towerIbase_d_cylIbase]])

        return self.J
        
        
class TurbineMass(object):

    def __init__(self):
        super(TurbineMass, self).__init__()
        
        # Parameters
        # self.add_param('hubH', val=0.0, units='m', desc='Hub-height')
        # self.add_param('rna_mass', val=0.0, units='kg', desc='Total tower mass')
        # self.add_param('rna_I', np.zeros((6,)), units='kg*m**2', desc='mass moment of inertia of rna about tower top [xx yy zz xy xz yz]')
        # self.add_param('rna_cg', np.zeros((3,)), units='m', desc='xyz-location of rna cg relative to tower top')
        
        # self.add_param('tower_mass', val=0.0, units='kg', desc='Total tower mass')
        # self.add_param('tower_center_of_mass', val=0.0, units='m', desc='z-position of center of mass of tower')
        # self.add_param('tower_I_base', np.zeros((6,)), units='kg*m**2', desc='mass moment of inertia of tower about base [xx yy zz xy xz yz]')

        # Output
        self.turbine_mass = 0.0 # self.add_output('turbine_mass', val=0.0, units='kg', desc='Total mass of tower+rna')
        self.turbine_center_of_mass = np.zeros((3,)) # self.add_output('turbine_center_of_mass', val=np.zeros((3,)), units='m', desc='xyz-position of tower+rna center of mass')
        self.turbine_I_base = np.zeros((6,)) # self.add_output('turbine_I_base', np.zeros((6,)), units='kg*m**2', desc='mass moment of inertia of tower about base [xx yy zz xy xz yz]')
       
        # Derivatives
        self.deriv_options = {}
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5
        
    def compute(self, hubH, rna_mass, rna_I, rna_cg, tower_mass, tower_center_of_mass, tower_I_base):
        self.turbine_mass = rna_mass + tower_mass
        
        cg_rna   = rna_cg + np.array([0.0, 0.0, hubH])
        cg_tower = np.array([0.0, 0.0, tower_center_of_mass])
        self.turbine_center_of_mass = (rna_mass*cg_rna + tower_mass*cg_tower) / self.turbine_mass

        R = cg_rna
        I_tower = assembleI(tower_I_base)
        I_rna   = assembleI(rna_I) + rna_mass*(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        self.turbine_I_base = unassembleI(I_tower + I_rna)
        
    

class TowerPreFrame(object):
    def __init__(self, nFull):
        super(TowerPreFrame, self).__init__()

        # self.add_param('z', np.zeros(nFull), units='m', desc='location along tower. start at bottom and go to top')

        # # extra mass
        # self.add_param('mass', 0.0, units='kg', desc='added mass')
        # self.add_param('mI', np.zeros((6,)), units='kg*m**2', desc='mass moment of inertia about some point p [xx yy zz xy xz yz]')
        # self.add_param('mrho', np.zeros((3,)), units='m', desc='xyz-location of p relative to node')

        # # point loads
        # self.add_param('rna_F', np.zeros((3,)), units='N', desc='rna force')
        # self.add_param('rna_M', np.zeros((3,)), units='N*m', desc='rna moment')
        
        # spring reaction data.  Use float('inf') for rigid constraints.
        nK = 1
        self.kidx = np.zeros(nK, dtype=np.int_) # self.add_output('kidx', np.zeros(nK, dtype=np.int_), desc='indices of z where external stiffness reactions should be applied.', pass_by_obj=True)
        self.kx = np.zeros(nK) # self.add_output('kx', np.zeros(nK), units='m', desc='spring stiffness in x-direction', pass_by_obj=True)
        self.ky = np.zeros(nK) # self.add_output('ky', np.zeros(nK), units='m', desc='spring stiffness in y-direction', pass_by_obj=True)
        self.kz = np.zeros(nK) # self.add_output('kz', np.zeros(nK), units='m', desc='spring stiffness in z-direction', pass_by_obj=True)
        self.ktx = np.zeros(nK) # self.add_output('ktx', np.zeros(nK), units='m', desc='spring stiffness in theta_x-rotation', pass_by_obj=True)
        self.kty = np.zeros(nK) # self.add_output('kty', np.zeros(nK), units='m', desc='spring stiffness in theta_y-rotation', pass_by_obj=True)
        self.ktz = np.zeros(nK) # self.add_output('ktz', np.zeros(nK), units='m', desc='spring stiffness in theta_z-rotation', pass_by_obj=True)

        # extra mass
        nMass = 1
        self.midx = np.zeros(nMass, dtype=np.int_) # self.add_output('midx', np.zeros(nMass, dtype=np.int_), desc='indices where added mass should be applied.', pass_by_obj=True)
        self.m = np.zeros(nMass) # self.add_output('m', np.zeros(nMass), units='kg', desc='added mass')
        self.mIxx = np.zeros(nMass) # self.add_output('mIxx', np.zeros(nMass), units='kg*m**2', desc='x mass moment of inertia about some point p')
        self.mIyy = np.zeros(nMass) # self.add_output('mIyy', np.zeros(nMass), units='kg*m**2', desc='y mass moment of inertia about some point p')
        self.mIzz = np.zeros(nMass) # self.add_output('mIzz', np.zeros(nMass), units='kg*m**2', desc='z mass moment of inertia about some point p')
        self.mIxy = np.zeros(nMass) # self.add_output('mIxy', np.zeros(nMass), units='kg*m**2', desc='xy mass moment of inertia about some point p')
        self.mIxz = np.zeros(nMass) # self.add_output('mIxz', np.zeros(nMass), units='kg*m**2', desc='xz mass moment of inertia about some point p')
        self.mIyz = np.zeros(nMass) # self.add_output('mIyz', np.zeros(nMass), units='kg*m**2', desc='yz mass moment of inertia about some point p')
        self.mrhox = np.zeros(nMass) # self.add_output('mrhox', np.zeros(nMass), units='m', desc='x-location of p relative to node')
        self.mrhoy = np.zeros(nMass) # self.add_output('mrhoy', np.zeros(nMass), units='m', desc='y-location of p relative to node')
        self.mrhoz = np.zeros(nMass) # self.add_output('mrhoz', np.zeros(nMass), units='m', desc='z-location of p relative to node')

        # point loads (if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also)
        nPL = 1
        self.plidx = np.zeros(nPL, dtype=np.int_) # self.add_output('plidx', np.zeros(nPL, dtype=np.int_), desc='indices where point loads should be applied.', pass_by_obj=True)
        self.Fx = np.zeros(nPL) # self.add_output('Fx', np.zeros(nPL), units='N', desc='point force in x-direction')
        self.Fy = np.zeros(nPL) # self.add_output('Fy', np.zeros(nPL), units='N', desc='point force in y-direction')
        self.Fz = np.zeros(nPL) # self.add_output('Fz', np.zeros(nPL), units='N', desc='point force in z-direction')
        self.Mxx = np.zeros(nPL) # self.add_output('Mxx', np.zeros(nPL), units='N*m', desc='point moment about x-axis')
        self.Myy = np.zeros(nPL) # self.add_output('Myy', np.zeros(nPL), units='N*m', desc='point moment about y-axis')
        self.Mzz = np.zeros(nPL) # self.add_output('Mzz', np.zeros(nPL), units='N*m', desc='point moment about z-axis')

        
    def compute(self, z, mass, mI, mrho, rna_F, rna_M):
        # Prepare for reactions: rigid at tower base
        self.kidx = np.array([ 0 ], dtype=np.int_)
        self.kx   = np.array([ np.inf ])
        self.ky   = np.array([ np.inf ])
        self.kz   = np.array([ np.inf ])
        self.ktx  = np.array([ np.inf ])
        self.kty  = np.array([ np.inf ])
        self.ktz  = np.array([ np.inf ])

        # Prepare RNA for "extra node mass"
        self.midx  = np.array([ len(z)-1 ], dtype=np.int_)
        self.m     = np.array([ mass ])
        self.mIxx  = np.array([ mI[0] ])
        self.mIyy  = np.array([ mI[1] ])
        self.mIzz  = np.array([ mI[2] ])
        self.mIxy  = np.array([ mI[3] ])
        self.mIxz  = np.array([ mI[4] ])
        self.mIyz  = np.array([ mI[5] ])
        self.mrhox = np.array([ mrho[0] ])
        self.mrhoy = np.array([ mrho[1] ])
        self.mrhoz = np.array([ mrho[2] ])

        # Prepare point forces at RNA node
        self.plidx = np.array([ len(z)-1 ], dtype=np.int_)
        self.Fx    = np.array([ rna_F[0] ])
        self.Fy    = np.array([ rna_F[1] ])
        self.Fz    = np.array([ rna_F[2] ])
        self.Mxx   = np.array([ rna_M[0] ])
        self.Myy   = np.array([ rna_M[1] ])
        self.Mzz   = np.array([ rna_M[2] ])

    def list_deriv_vars(self):
        inputs = ('mass', 'mI', 'mrho', 'rna_F', 'rna_M')
        outputs = ('m', 'mIxx', 'mIyy', 'mIzz', 'mIxy', 'mIxz', 'mIyz', 'Fx', 'Fy', 'Fz', 'Mxx', 'Myy', 'Mzz')
        return inputs, outputs
        
    def linearize(self, params, unknowns, resids):
        inp,out = self.list_deriv_vars()
        n_inp = len(inp)
        n_out = len(out)

        self.J = np.zeros((n_out,n_inp))
        for i, out_i in enumerate(out):
            for j, inp_j in enumerate(inp):
                    self.J[i,j] = np.zeros((n_out, n_inp))

        self.J[inp.index('m'), out.index('mass')]    = 1.0
        self.J[inp.index('mIxx'), out.index('mI')]   = np.eye(6)[0,:]
        self.J[inp.index('mIyy'), out.index('mI')]   = np.eye(6)[1,:]
        self.J[inp.index('mIzz'), out.index('mI')]   = np.eye(6)[2,:]
        self.J[inp.index('mIxy'), out.index('mI')]   = np.eye(6)[3,:]
        self.J[inp.index('mIxz'), out.index('mI')]   = np.eye(6)[4,:]
        self.J[inp.index('mIyz'), out.index('mI')]   = np.eye(6)[5,:]
        self.J[inp.index('Fx'), out.index('rna_F')]  = np.eye(3)[0,:]
        self.J[inp.index('Fy'), out.index('rna_F')]  = np.eye(3)[2,:]
        self.J[inp.index('Fz'), out.index('rna_F')]  = np.eye(3)[2,:]
        self.J[inp.index('Mxx'), out.index('rna_M')] = np.eye(3)[0,:]
        self.J[inp.index('Myy'), out.index('rna_M')] = np.eye(3)[2,:]
        self.J[inp.index('Mzz'), out.index('rna_M')] = np.eye(3)[2,:]

        
class TowerPostFrame(object):
    def __init__(self, nFull, nDEL):
        super(TowerPostFrame, self).__init__()

        # # effective geometry -- used for handbook methods to estimate hoop stress, buckling, fatigue
        # self.add_param('z', np.zeros(nFull), units='m', desc='location along tower. start at bottom and go to top')
        # self.add_param('d', np.zeros(nFull), units='m', desc='effective tower diameter for section')
        # self.add_param('t', np.zeros(nFull), units='m', desc='effective shell thickness for section')
        # self.add_param('L_reinforced', 0.0, units='m', desc='buckling length')

        # # Material properties
        # self.add_param('E', 0.0, units='N/m**2', desc='modulus of elasticity')

        # # Processed Frame3DD outputs
        # self.add_param('Fz', np.zeros(nFull), units='N', desc='Axial foce in vertical z-direction in cylinder structure.')
        # self.add_param('Mxx', np.zeros(nFull), units='N*m', desc='Moment about x-axis in cylinder structure.')
        # self.add_param('Myy', np.zeros(nFull), units='N*m', desc='Moment about y-axis in cylinder structure.')
        # self.add_param('axial_stress', val=np.zeros(nFull), units='N/m**2', desc='axial stress in tower elements')
        # self.add_param('shear_stress', val=np.zeros(nFull), units='N/m**2', desc='shear stress in tower elements')
        # self.add_param('hoop_stress' , val=np.zeros(nFull), units='N/m**2', desc='hoop stress in tower elements')

        # # safety factors
        # self.add_param('gamma_f', 1.35, desc='safety factor on loads')
        # self.add_param('gamma_m', 1.1, desc='safety factor on materials')
        # self.add_param('gamma_n', 1.0, desc='safety factor on consequence of failure')
        # self.add_param('gamma_b', 1.1, desc='buckling safety factor')
        # self.add_param('sigma_y', 0.0, units='N/m**2', desc='yield stress')
        # self.add_param('gamma_fatigue', 1.755, desc='total safety factor for fatigue')

        # # fatigue parameters
        # self.add_param('life', 20.0, desc='fatigue life of tower')
        # self.add_param('m_SN', 4, desc='slope of S/N curve', pass_by_obj=True)
        # self.add_param('DC', 80.0, desc='standard value of stress')
        # self.add_param('z_DEL', np.zeros(nDEL), desc='absolute z coordinates of corresponding fatigue parameters', pass_by_obj=True)
        # self.add_param('M_DEL', np.zeros(nDEL), desc='fatigue parameters at corresponding z coordinates', pass_by_obj=True)

        # outputs
        self.f1 = 0.0 # self.add_output('f1', 0.0, units='Hz', desc='First natural frequency')
        self.f2 = 0.0 # self.add_output('f2', 0.0, units='Hz', desc='Second natural frequency')
        self.top_deflection = 0.0 # self.add_output('top_deflection', 0.0, units='m', desc='Deflection of tower top in yaw-aligned +x direction')
        self.stress = np.zeros(nFull) # self.add_output('stress', np.zeros(nFull), desc='Von Mises stress utilization along tower at specified locations.  incudes safety factor.')
        self.shell_buckling = np.zeros(nFull) # self.add_output('shell_buckling', np.zeros(nFull), desc='Shell buckling constraint.  Should be < 1 for feasibility.  Includes safety factors')
        self.global_buckling = np.zeros(nFull) # self.add_output('global_buckling', np.zeros(nFull), desc='Global buckling constraint.  Should be < 1 for feasibility.  Includes safety factors')
        self.damage = np.zeros(nFull) # self.add_output('damage', np.zeros(nFull), desc='Fatigue damage at each tower section')
        self.turbine_F = val=np.zeros(3) # self.add_output('turbine_F', val=np.zeros(3), units='N', desc='Total force on tower+rna')
        self.turbine_M = val=np.zeros(3) # self.add_output('turbine_M', val=np.zeros(3), units='N*m', desc='Total x-moment on tower+rna measured at base')
        
        # Derivatives
        self.deriv_options = {}
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5

        
    def compute(self, z, d, t, L_reinforced, E, Fz, Mxx, Myy, axial_stress, shear_stress, hoop_stress, gamma_f, gamma_m, gamma_n, \
                gamma_b, sigma_y, gamma_fatigue, life, m_SN, DC, z_DEL, M_DEL):
        # Unpack some variables
        sigma_y      = sigma_y * np.ones(axial_stress.shape)
        E            = E * np.ones(axial_stress.shape)
        L_reinforced = L_reinforced * np.ones(axial_stress.shape)
        
        # von mises stress
        self.stress = Util.vonMisesStressUtilization(axial_stress, hoop_stress, shear_stress, gamma_f*gamma_m*gamma_n, sigma_y)

        # shell buckling
        self.shell_buckling = Util.shellBucklingEurocode(d, t, axial_stress, hoop_stress, shear_stress, L_reinforced, E, sigma_y, gamma_f, gamma_b)

        # global buckling
        tower_height = z[-1] - z[0]
        M = np.sqrt(Mxx**2 + Myy**2)
        self.global_buckling = Util.bucklingGL(d, t, Fz, M, tower_height, E, sigma_y, gamma_f, gamma_b)

        # fatigue
        N_DEL = 365.0*24.0*3600.0*life * np.ones(len(z))
        self.damage = np.zeros(N_DEL.shape)
        if any(M_DEL):
            M_DEL = np.interp(z, z_DEL, M_DEL)

            self.damage = Util.fatigue(M_DEL, N_DEL, d, t, m_SN, DC, gamma_fatigue, stress_factor=1.0, weld_factor=True)

# -----------------
#  Assembly
# -----------------

class TowerLeanSE(Group):

    def __init__(self, nPoints, nFull):
        super(TowerLeanSE, self).__init__()
        nRefine = (nFull-1)/(nPoints-1)

        self.geometry = CylinderDiscretization(nPoints, nRefine) #TODO: make a python object version
        self.tgeometry = TowerDiscretization()
        self.cylm = CylinderMass(nFull) #TODO: make a python object version
        self.tm = TowerMass(nFull)
        self.gc = Util.GeometricConstraints(nPoints)
        self.turb = TurbineMass()

    def compute(self, foundation_height, tower_section_height, tower_outer_diameter, tower_wall_thickness, tower_outfitting_factor, tower_buckling_length):
        
        self.geometry.compute(foundation_height, tower_section_height, tower_outer_diameter, tower_wall_thickness)
        # promotes
        self.z_param = self.geometry.z_param
        self.z_full  = self.geometry.z_full
        self.d_full  = self.geometry.d_full
        self.t_full  = self.geometry.t_full

        self.tgeometry.compute(self.hub_height, self.z_param)
        self.cylm.compute(self.d_full, self.t_full, self.z_full, self.material_density, tower_outfitting_factor)
        self.tm.compute(self.cylm.mass, self.cylm.center_of_mass, self.cylm.section_center_of_mass, self.cylm.I_base)
        self.gc.compute(tower_outer_diameter, tower_wall_thickness, self.min_d_to_t, self.max_taper)
        self.turb.compute(self.hub_height, self.rna_mass, self.rna_I, self.rna_cg, self.tm.tower_mass, self.tm.tower_center_of_mass, self.tm.tower_I_base)

        ## ?? from connections
        # self.section_height = tower_section_height
        # self.diameter = tower_outer_diameter
        # self.wall_thickness = tower_wall_thickness

        
class TowerSE(Group):

    def __init__(self, nLC, nPoints, nFull, nDEL, wind=''):

        super(TowerSE, self).__init__()

        self.nLC = nLC
        self.nPoints = nPoints
        self.nFull = nFull
        self.nDEL = nDEL
        self.wind = wind

        self.geom = TowerLeanSE(nPoints, nFull)
        self.props = CylindricalShellProperties(nFull)


        self.wind = [None]*nLC
        self.wave = [None]*nLC
        self.windLoads = [None]*nLC
        self.waveLoads = [None]*nLC
        self.distLoads = [None]*nLC
        self.pre = [None]*nLC
        self.tower = [None]*nLC
        self.post = [None]*nLC

        for lc in range(nLC):
            if wind.lower() == 'powerwind':
                self.wind[lc] = PowerWind(nFull)
            elif wind.lower() == 'logwind':
                self.wind[lc] = LogWind(nFull)
            else:
                raise ValueError('Unknown wind type, '+wind)

            self.wave[lc] = WaveBase(nFull)
            self.windLoads[lc] = CylinderWindDrag(nFull)
            self.waveLoads[lc] = CylinderWaveDrag(nFull)
            self.distLoads[lc] = AeroHydroLoads(nFull)
            self.pre[lc] = TowerPreFrame(nFull)
            self.tower[lc] = CylinderFrame3DD(nFull, 1, 1, 1)
            self.post[lc] = TowerPostFrame(nFull, nDEL)


    def compute(self, tower_M_DEL, tower_z_DEL, tower_force_discretization):
        
        self.geom.hub_height       = self.hub_height
        self.geom.material_density = self.material_density
        self.geom.min_d_to_t       = self.min_d_to_t
        self.geom.max_taper        = self.max_taper
        self.geom.rna_mass         = self.rna_mass
        self.geom.rna_I            = self.rna_I
        self.geom.rna_cg           = self.rna_cg

        self.geom.compute(self.foundation_height, self.tower_section_height, self.tower_outer_diameter, self.tower_wall_thickness, \
                          self.tower_outfitting_factor, self.tower_buckling_length)
        z_param = self.geom.z_param # promotes
        z_full  = self.geom.z_full  # promotes
        d_full  = self.geom.d_full  # promotes
        t_full  = self.geom.t_full  # promotes

        self.props.compute(d_full, t_full)

        # Add in all Components that drive load cases
        # Note multiple load cases have to be handled by replicating components and not groups/assemblies.
        # Replicating Groups replicates the IndepVarComps which doesn't play nicely in OpenMDAO


        for lc in range(nLC):
            
            # wind
            self.wind[lc].zref = self.zref[lc]
            self.wind[lc].z    = z_full
            self.wind[lc].z0   = self.z0[lc]
            self.wind[lc].Uref = self.Uref[lc]
            if wind.lower() == 'powerwind':
                self.wind[lc].compute(self.shearExp[lc])
            elif wind.lower() == 'logwind':
                self.wind[lc].compute(self.z_roughness[lc])
            else:
                raise ValueError('Unknown wind type, '+wind)

            # wave
            self.wave[lc].compute(self.rho_wave[lc], z_full, self.z0[lc], self.z_floor[lc]) 

            # windLoads
            self.windLoads[lc].compute(self.wind[lc].U, z_full, d_full, self.beta_wind[lc], self.rho_wind[lc], self.mu_wind[lc], self.cd_usr[lc]) 
            wd = self.windLoads[lc]

            # waveLoads
            self.waveLoads[lc].compute(self.wave[lc].U, self.wave[lc].A, self.wave[lc].p, z_full, d_full, self.beta_wave[lc], self.rho_wave[lc], \
                                       self.mu_wave[lc], self.cm[lc], self.cd_usr[lc])
            wv = self.waveLoads[lc]

            # distLoads
            self.distLoads[lc].compute(wd.windLoads_Px, wd.windLoads_Py, wd.windLoads_Pz, wd.windLoads_qdyn, wd.windLoads_z, wd.windLoads_d, \
                                       wd.windLoads_beta, wv.waveLoads_Px, wv.waveLoads_Py, wv.waveLoads_Pz, wv.waveLoads_qdyn, wv.waveLoads_z, \
                                       wv.waveLoads_d, wv.waveLoads_beta, z_full, self.yaw)

            # pre
            self.pre[lc].compute(z_full, self.rna_mass, self.rna_I, self.rna_cg, self.rna_F[lc], self.rna_M[lc]) 
            pre = self.pre[lc]

            # tower
            mrhox = [0.] # NOTE! these were not connected in tower.py, overwriting their values
            mrhoy = [0.]
            mrhoz = [0.]
            self.tower[lc].compute(z_full, self.props.Az, self.props.Asx, self.props.Asy, self.props.Jz, self.props.Ixx, self.props.Iyy, self.E, self.G, \
                                   self.material_density, self.sigma_y, self.tower_buckling_length, d_full, t_full, pre.kidx, pre.kx, pre.ky, pre.kz, \
                                   pre.ktx, pre.kty, pre.ktz, pre.midx, pre.m, pre.mIxx, pre.mIyy, pre.mIzz, pre.mIxy, pre.mIxz, pre.mIyz, mrhox, mrhoy, \
                                   mrhoz, self.addGravityLoadForExtraMass, pre.plidx, pre.Fx, pre.Fy, pre.Fz, pre.Mxx, pre.Myy, pre.Mzz, self.distLoads[lc].Px, 
                                   self.distLoads[lc].Py, self.distLoads[lc].Pz, self.distLoads[lc].qdyn, self.shear, self.geom_stiff, tower_force_discretization, \
                                   self.nM, self.Mmethod, self.lump, self.tol, self.shift)
            # self.tower[lc].compute(z_full, self.props.Az, self.props.Asx, self.props.Asy, self.props.Jz, self.props.Ixx, self.props.Iyy, self.E, self.G, self.material_density, self.sigma_y, self.tower_buckling_length, d_full, t_full, pre.kidx, pre.kx, pre.ky, pre.kz, pre.ktx, pre.kty, pre.ktz, pre.midx, pre.m, pre.mIxx, pre.mIyy, pre.mIzz, pre.mIxy, pre.mIxz, pre.mIyz, pre.mrhox, pre.mrhoy, pre.mrhoz, self.addGravityLoadForExtraMass, pre.plidx, pre.Fx, pre.Fy, pre.Fz, pre.Mxx, pre.Myy, pre.Mzz, self.distLoads[lc].Px, self.distLoads[lc].Py, self.distLoads[lc].Pz, self.distLoads[lc].qdyn, self.shear, self.geom_stiff, tower_force_discretization, self.nM, self.Mmethod, self.lump, self.tol, self.shift) # TODO: params
            tower = self.tower[lc]

            # post
            self.post[lc].compute(z_full, d_full, t_full, self.tower_buckling_length, self.E, tower.Fz_out, tower.Mxx_out, tower.Myy_out, tower.axial_stress, \
                                  tower.shear_stress, tower.hoop_stress_euro, self.gamma_f, self.gamma_m, self.gamma_n, self.gamma_b, self.sigma_y, \
                                  self.gamma_fatigue, self.life, self.m_SN, self.DC, self.tower_z_DEL, self.tower_M_DEL)

        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5

        
if __name__ == '__main__':
    #TODO: use a case iterator
    
    # --- fatigue ---  # here because length of z_DEL is needed
    tower_z_DEL = np.array([0.000, 1.327, 3.982, 6.636, 9.291, 11.945, 14.600, 17.255, 19.909, 22.564, 25.218, 27.873, 30.527, 33.182, 35.836, 38.491, 41.145, 43.800, 46.455, 49.109, 51.764, 54.418, 57.073, 59.727, 62.382, 65.036, 67.691, 70.345, 73.000, 75.655, 78.309, 80.964, 83.618, 86.273, 87.600])
    tower_M_DEL = 1e3*np.array([8.2940E+003, 8.1518E+003, 7.8831E+003, 7.6099E+003, 7.3359E+003, 7.0577E+003, 6.7821E+003, 6.5119E+003, 6.2391E+003, 5.9707E+003, 5.7070E+003, 5.4500E+003, 5.2015E+003, 4.9588E+003, 4.7202E+003, 4.4884E+003, 4.2577E+003, 4.0246E+003, 3.7942E+003, 3.5664E+003, 3.3406E+003, 3.1184E+003, 2.8977E+003, 2.6811E+003, 2.4719E+003, 2.2663E+003, 2.0673E+003, 1.8769E+003, 1.7017E+003, 1.5479E+003, 1.4207E+003, 1.3304E+003, 1.2780E+003, 1.2673E+003, 1.2761E+003])
    nDEL = len(tower_z_DEL)
    life = 20.0
    m_SN = 4
    # ---------------

    ###### Declare control parameters and initialize ######

    foundation_height = 0.
    tower_section_height = np.diff(np.array([0.0, 43.8, 87.6]))
    tower_outer_diameter = np.array([6.0, 4.935, 3.87])
    tower_wall_thickness = 1.3*np.array([0.027, 0.023, 0.019])
    tower_buckling_length = 30.0  # [m] buckling length
    tower_outfitting_factor = 1.07

    nPoints = len(tower_outer_diameter)
    nFull   = 5*(nPoints-1) + 1
    wind = 'PowerWind'
    nLC = 2

    tower = TowerSE(nLC, nPoints, nFull, nDEL, wind=wind)

    ###### assign values to params ######
    # --- geometry ----
    tower.hub_height = tower_section_height.sum()
    tower.yaw = .0
    # theta_stress = 0.0

    # --- material props ---
    tower.E = 210e9
    tower.G = 80.8e9
    tower.material_density = 8500.0
    tower.sigma_y = 450.0e6

    # --- extra mass ----
    mIxx = 1.14930678e+08
    mIyy = 2.20354030e+07
    mIzz = 1.87597425e+07
    mIxy = 0.0
    mIxz = 5.03710467e+05
    mIyz = 0.0
    mI = np.array([mIxx, mIyy, mIzz, mIxy, mIxz, mIyz])

    tower.rna_mass = np.array([285598.8])
    tower.rna_I = mI
    tower.rna_cg = np.array([-1.13197635, 0.0, 0.50875268])
    # -----------

    # --- wind & wave ---
    nLC = 2
    tower.rho_wind    = nLC*[1.225]
    tower.mu_wind     = nLC*[1.7934e-5]
    tower.zref        = nLC*[90.0]
    tower.z0          = nLC*[0.0]
    tower.z_floor     = nLC*[0.0]
    tower.z_roughness = nLC*[0.0]
    tower.shearExp    = nLC*[0.2]
    tower.cd_usr      = nLC*[None]
    tower.beta_wind   = nLC*[0.0]

    tower.cm          = nLC*[0.0]
    tower.rho_wave    = nLC*[1025.0]
    tower.mu_wave     = nLC*[1.3351e-3]
    tower.beta_wave   = nLC*[0.0]
    # ---------------

    # --- loading ---
    # cases: [max thrust, max wind speed]
    tower.Uref = [11.73732, 70.0]
    tower.Fx   = [1284744.19620519, 930198.60063279]
    tower.Fy   = [0., 0.]
    tower.Fz   = [-2914124.84400512, -2883106.12368949]
    tower.Mxx  = [3963732.76208099, -1683669.22411597]
    tower.Myy  = [-2275104.79420872, -2522475.34625363]
    tower.Mzz  = [-346781.68192839, 147301.97023764]

    tower.rna_F = nLC*[None]
    tower.rna_M = nLC*[None]
    for i, (Fx, Fy, Fz, Mxx, Myy, Mzz) in enumerate(zip(tower.Fx, tower.Fy, tower.Fz, tower.Mxx, tower.Myy, tower.Mzz)):
        tower.rna_F[i] = [Fx, Fy, Fz]
        tower.rna_M[i] = [Mxx, Myy, Mzz]
    # ---------------

    # --- safety factors ---
    tower.gamma_f = 1.35
    tower.gamma_m = 1.3
    tower.gamma_n = 1.0
    tower.gamma_b = 1.1
    tower.gamma_fatigue = 1.35*1.3*1.0
    # ---------------

    # --- settings ---
    tower.DC = 80.0
    tower.shear = True
    tower.geom_stiff = False
    tower.tower_force_discretization = 5.0
    tower.nM = 2
    tower.Mmethod = 1
    tower.lump = 0
    tower.tol = 1e-9
    tower.shift = 0.0
    tower.addGravityLoadForExtraMass = True
    # ---------------
    
    # --- fatigue ---
    tower.tower_z_DEL = tower_z_DEL
    tower.tower_M_DEL = tower_M_DEL
    tower.nDEL = nDEL
    tower.life = life
    tower.m_SN = m_SN
    # ---------------

    # --- constraints ---
    tower.min_d_to_t = 120.0
    tower.max_taper = 0.2
    # ---------------

    # # --- run ---
    tower.foundation_height = foundation_height
    tower.tower_section_height = tower_section_height
    tower.tower_outer_diameter = tower_outer_diameter
    tower.tower_wall_thickness = tower_wall_thickness
    tower.tower_outfitting_factor = tower_outfitting_factor
    tower.tower_buckling_length = tower_buckling_length

    tower.compute(tower.tower_M_DEL, tower.tower_z_DEL, tower.tower_force_discretization)

    z = tower.geom.z_full

    print 'zs=', tower.geom.z_full
    print 'ds=', tower.geom.d_full
    print 'ts=', tower.geom.t_full
    print 'mass (kg) =', tower.geom.tm.tower_mass
    print 'cg (m) =', tower.geom.tm.tower_center_of_mass
    print 'weldability =', tower.geom.gc.weldability
    print 'manufacturability =', tower.geom.gc.manufacturability

    for lc in range(nLC):
        print '\nwind: ', tower.wind[lc].Uref
        print 'f1 (Hz) =', tower.tower[lc].f1
        print 'top_deflection1 (m) =', tower.post[lc].top_deflection
        print 'stress1 =', tower.post[lc].stress
        print 'GL buckling =', tower.post[lc].global_buckling
        print 'Shell buckling =', tower.post[lc].shell_buckling
        print 'damage =', tower.post[lc].damage

    stress1 = np.copy( tower.post[0].stress )
    shellBuckle1 = np.copy( tower.post[0].shell_buckling )
    globalBuckle1 = np.copy( tower.post[0].global_buckling )
    damage1 = np.copy( tower.post[0].damage )

    stress2 = tower.post[1].stress
    shellBuckle2 = tower.post[1].shell_buckling
    globalBuckle2 = tower.post[1].global_buckling
    damage2 = tower.post[1].damage

    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.0, 3.5))
    plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
    plt.plot(stress1, z, label='stress 1')
    plt.plot(stress2, z, label='stress 2')
    plt.plot(shellBuckle1, z, label='shell buckling 1')
    plt.plot(shellBuckle2, z, label='shell buckling 2')
    plt.plot(globalBuckle1, z, label='global buckling 1')
    plt.plot(globalBuckle2, z, label='global buckling 2')
    plt.plot(damage1, z, label='damage 1')
    plt.plot(damage2, z, label='damage 2')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2)
    plt.xlabel('utilization')
    plt.ylabel('height along tower (m)')

    #plt.figure(2)
    #plt.plot(tower.d_full/2.+max(tower.d_full), z, 'ok')
    #plt.plot(tower.d_full/-2.+max(tower.d_full), z, 'ok')

    #fig = plt.figure(3)
    #ax1 = fig.add_subplot(121)
    #ax2 = fig.add_subplot(122)

    #ax1.plot(tower.wind1.U, z)
    #ax2.plot(tower.wind2.U, z)
    #plt.tight_layout()
    plt.show()

    print tower.tower[0].base_F
    print tower.tower[0].base_M
    print tower.tower[1].base_F
    print tower.tower[1].base_M
    # # ------------

    # """
    # if optimize:

    #     # --- optimizer imports ---
    #     from pyopt_driver.pyopt_driver import pyOptDriver

    #     # ----------------------

    #     # --- Setup Pptimizer ---
    #     tower.replace('driver', pyOptDriver())
    #     tower.driver.optimizer = 'SNOPT'
    #     tower.driver.options = {'Major feasibility tolerance': 1e-6,
    #                            'Minor feasibility tolerance': 1e-6,
    #                            'Major optimality tolerance': 1e-5,
    #                            'Function precision': 1e-8}
    #     # ----------------------

    #     # --- Objective ---
    #     tower.driver.add_objective('tower1.mass / 300000')
    #     # ----------------------

    #     # --- Design Variables ---
    #     tower.driver.add_parameter('z_param[1]', low=0.0, high=87.0)
    #     tower.driver.add_parameter('d_param[:-1]', low=3.87, high=20.0)
    #     tower.driver.add_parameter('t_param', low=0.005, high=0.2)
    #     # ----------------------

    #     # --- recorder ---
    #     tower.recorders = [DumpCaseRecorder()]
    #     # ----------------------

    #     # --- Constraints ---
    #     tower.driver.add_constraint('tower.stress <= 1.0')
    #     tower.driver.add_constraint('tower.global_buckling <= 1.0')
    #     tower.driver.add_constraint('tower.shell_buckling <= 1.0')
    #     tower.driver.add_constraint('tower.damage <= 1.0')
    #     tower.driver.add_constraint('gc.weldability <= 0.0')
    #     tower.driver.add_constraint('gc.manufacturability <= 0.0')
    #     freq1p = 0.2  # 1P freq in Hz
    #     tower.driver.add_constraint('tower.f1 >= 1.1*%f' % freq1p)
    #     # ----------------------

    #     # --- run opt ---
    #     tower.run()
    #     # ---------------
    # """
