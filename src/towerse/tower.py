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
 """

import numpy as np
from openmdao.api import Component, Group, Problem, IndepVarComp

from commonse.WindWaveDrag import AeroHydroLoads, CylinderWindDrag, CylinderWaveDrag

from commonse.environment import WindBase, WaveBase, SoilBase, PowerWind, LogWind
from commonse.tube import CylindricalShellProperties
from commonse.utilities import assembleI, unassembleI
from commonse import gravity, eps

from commonse.vertical_cylinder import CylinderDiscretization, CylinderMass, CylinderFrame3DD
#from fusedwind.turbine.tower import TowerFromCSProps
#from fusedwind.interface import implement_base

import commonse.UtilizationSupplement as Util


# -----------------
#  Components
# -----------------

class TowerDiscretization(Component):
    def __init__(self):
        super(TowerDiscretization, self).__init__()
        self.add_param('hub_height', val=0.0, units='m', desc='diameter at tower base')
        self.add_param('z_end', val=0.0, units='m', desc='Last node point on tower')
        self.add_output('height_constraint', val=0.0, units='m', desc='mismatch between tower height and desired hub_height')

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['height_constraint'] = params['hub_height'] - params['z_end']

    def linearize(self, params, unknowns, resids):
        J = {}
        J['height_constraint','hub_height'] = 1
        J['height_constraint','z_end'] = -1
        return J
        
class TowerMass(Component):

    def __init__(self, nPoints):
        super(TowerMass, self).__init__()
        
        self.add_param('cylinder_mass', val=np.zeros(nPoints-1), units='kg', desc='Total cylinder mass')
        self.add_param('cylinder_center_of_mass', val=0.0, units='m', desc='z-position of center of mass of cylinder')
        self.add_param('cylinder_section_center_of_mass', val=np.zeros(nPoints-1), units='m', desc='z position of center of mass of each can in the cylinder')
        self.add_param('cylinder_I_base', np.zeros((6,)), units='kg*m**2', desc='mass moment of inertia of cylinder about base [xx yy zz xy xz yz]')
        
        self.add_output('tower_mass', val=0.0, units='kg', desc='Total tower mass')
        self.add_output('tower_center_of_mass', val=0.0, units='m', desc='z-position of center of mass of tower')
        self.add_output('tower_section_center_of_mass', val=np.zeros(nPoints-1), units='m', desc='z position of center of mass of each can in the tower')
        self.add_output('tower_I_base', np.zeros((6,)), units='kg*m**2', desc='mass moment of inertia of tower about base [xx yy zz xy xz yz]')
        
    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['tower_mass']           = params['cylinder_mass'].sum()
        unknowns['tower_center_of_mass'] = params['cylinder_center_of_mass']
        unknowns['tower_section_center_of_mass'] = params['cylinder_section_center_of_mass']
        unknowns['tower_I_base']         = params['cylinder_I_base']

    def linearize(self, params, unknowns, resids):
        npts = len(params['cylinder_section_center_of_mass'])
        zeroPts = np.zeros(npts)
        zero6 = np.zeros(6)

        J = {}
        J['tower_mass','cylinder_mass'] = np.ones(len(unknowns['cylinder_mass']))
        J['tower_mass','cylinder_center_of_mass'] = 0.0
        J['tower_mass','cylinder_section_center_of_mass'] = zeroPts
        J['tower_mass','cylinder_I_base'] = zero6

        J['tower_center_of_mass','cylinder_mass'] = 0.0
        J['tower_center_of_mass','cylinder_center_of_mass'] = 1.0
        J['tower_center_of_mass','cylinder_section_center_of_mass'] = zeroPts
        J['tower_center_of_mass','cylinder_I_base'] = zero6

        J['tower_section_center_of_mass','cylinder_mass'] = 0.0
        J['tower_section_center_of_mass','cylinder_center_of_mass'] = 0.0
        J['tower_section_center_of_mass','cylinder_section_center_of_mass'] = np.eye(npts)
        J['tower_section_center_of_mass','cylinder_I_base'] = np.zeros((npts,6))

        J['tower_I_base','cylinder_mass'] = 1.0
        J['tower_I_base','cylinder_center_of_mass'] = 0.0
        J['tower_I_base','cylinder_section_center_of_mass'] = np.zeros((6,npts))
        J['tower_I_base','cylinder_I_base'] = np.eye(len(params['cylinder_I_base']))
        return J
        
        
class TurbineMass(Component):

    def __init__(self):
        super(TurbineMass, self).__init__()
        
        self.add_param('hubH', val=0.0, units='m', desc='Hub-height')
        self.add_param('rna_mass', val=0.0, units='kg', desc='Total tower mass')
        self.add_param('rna_I', np.zeros((6,)), units='kg*m**2', desc='mass moment of inertia of rna about tower top [xx yy zz xy xz yz]')
        self.add_param('rna_cg', np.zeros((3,)), units='m', desc='xyz-location of rna cg relative to tower top')
        
        self.add_param('tower_mass', val=0.0, units='kg', desc='Total tower mass')
        self.add_param('tower_center_of_mass', val=0.0, units='m', desc='z-position of center of mass of tower')
        self.add_param('tower_I_base', np.zeros((6,)), units='kg*m**2', desc='mass moment of inertia of tower about base [xx yy zz xy xz yz]')

        self.add_output('turbine_mass', val=0.0, units='kg', desc='Total mass of tower+rna')
        self.add_output('turbine_center_of_mass', val=np.zeros((3,)), units='m', desc='xyz-position of tower+rna center of mass')
        self.add_output('turbine_I_base', np.zeros((6,)), units='kg*m**2', desc='mass moment of inertia of tower about base [xx yy zz xy xz yz]')
       
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5
        
    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['turbine_mass'] = params['rna_mass'] + params['tower_mass']
        
        cg_rna   = params['rna_cg'] + np.array([0.0, 0.0, params['hubH']])
        cg_tower = np.array([0.0, 0.0, params['tower_center_of_mass']])
        unknowns['turbine_center_of_mass'] = (params['rna_mass']*cg_rna + params['tower_mass']*cg_tower) / unknowns['turbine_mass']

        R = cg_rna
        I_tower = assembleI(params['tower_I_base'])
        I_rna   = assembleI(params['rna_I']) + params['rna_mass']*(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        unknowns['turbine_I_base'] = unassembleI(I_tower + I_rna)
        
    

class TowerPreFrame(Component):
    def __init__(self, nFull):
        super(TowerPreFrame, self).__init__()

        self.add_param('z', np.zeros(nFull), units='m', desc='location along tower. start at bottom and go to top')

        # extra mass
        self.add_param('mass', 0.0, units='kg', desc='added mass')
        self.add_param('mI', np.zeros((6,)), units='kg*m**2', desc='mass moment of inertia about some point p [xx yy zz xy xz yz]')
        self.add_param('mrho', np.zeros((3,)), units='m', desc='xyz-location of p relative to node')

        # point loads
        self.add_param('rna_F', np.zeros((3,)), units='N', desc='rna force')
        self.add_param('rna_M', np.zeros((3,)), units='N*m', desc='rna moment')
        
        # spring reaction data.  Use float('inf') for rigid constraints.
        nK = 1
        self.add_output('kidx', np.zeros(nK, dtype=np.int_), desc='indices of z where external stiffness reactions should be applied.', pass_by_obj=True)
        self.add_output('kx', np.zeros(nK), units='m', desc='spring stiffness in x-direction', pass_by_obj=True)
        self.add_output('ky', np.zeros(nK), units='m', desc='spring stiffness in y-direction', pass_by_obj=True)
        self.add_output('kz', np.zeros(nK), units='m', desc='spring stiffness in z-direction', pass_by_obj=True)
        self.add_output('ktx', np.zeros(nK), units='m', desc='spring stiffness in theta_x-rotation', pass_by_obj=True)
        self.add_output('kty', np.zeros(nK), units='m', desc='spring stiffness in theta_y-rotation', pass_by_obj=True)
        self.add_output('ktz', np.zeros(nK), units='m', desc='spring stiffness in theta_z-rotation', pass_by_obj=True)

        # extra mass
        nMass = 1
        self.add_output('midx', np.zeros(nMass, dtype=np.int_), desc='indices where added mass should be applied.', pass_by_obj=True)
        self.add_output('m', np.zeros(nMass), units='kg', desc='added mass')
        self.add_output('mIxx', np.zeros(nMass), units='kg*m**2', desc='x mass moment of inertia about some point p')
        self.add_output('mIyy', np.zeros(nMass), units='kg*m**2', desc='y mass moment of inertia about some point p')
        self.add_output('mIzz', np.zeros(nMass), units='kg*m**2', desc='z mass moment of inertia about some point p')
        self.add_output('mIxy', np.zeros(nMass), units='kg*m**2', desc='xy mass moment of inertia about some point p')
        self.add_output('mIxz', np.zeros(nMass), units='kg*m**2', desc='xz mass moment of inertia about some point p')
        self.add_output('mIyz', np.zeros(nMass), units='kg*m**2', desc='yz mass moment of inertia about some point p')
        self.add_output('mrhox', np.zeros(nMass), units='m', desc='x-location of p relative to node')
        self.add_output('mrhoy', np.zeros(nMass), units='m', desc='y-location of p relative to node')
        self.add_output('mrhoz', np.zeros(nMass), units='m', desc='z-location of p relative to node')

        # point loads (if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also)
        nPL = 1
        self.add_output('plidx', np.zeros(nPL, dtype=np.int_), desc='indices where point loads should be applied.', pass_by_obj=True)
        self.add_output('Fx', np.zeros(nPL), units='N', desc='point force in x-direction')
        self.add_output('Fy', np.zeros(nPL), units='N', desc='point force in y-direction')
        self.add_output('Fz', np.zeros(nPL), units='N', desc='point force in z-direction')
        self.add_output('Mxx', np.zeros(nPL), units='N*m', desc='point moment about x-axis')
        self.add_output('Myy', np.zeros(nPL), units='N*m', desc='point moment about y-axis')
        self.add_output('Mzz', np.zeros(nPL), units='N*m', desc='point moment about z-axis')

        
    def solve_nonlinear(self, params, unknowns, resids):
        # Prepare for reactions: rigid at tower base
        self.unknowns['kidx'] = np.array([ 0 ], dtype=np.int_)
        self.unknowns['kx']   = np.array([ np.inf ])
        self.unknowns['ky']   = np.array([ np.inf ])
        self.unknowns['kz']   = np.array([ np.inf ])
        self.unknowns['ktx']  = np.array([ np.inf ])
        self.unknowns['kty']  = np.array([ np.inf ])
        self.unknowns['ktz']  = np.array([ np.inf ])

        # Prepare RNA for "extra node mass"
        self.unknowns['midx']  = np.array([ len(params['z'])-1 ], dtype=np.int_)
        self.unknowns['m']     = np.array([ params['mass'] ])
        self.unknowns['mIxx']  = np.array([ params['mI'][0] ])
        self.unknowns['mIyy']  = np.array([ params['mI'][1] ])
        self.unknowns['mIzz']  = np.array([ params['mI'][2] ])
        self.unknowns['mIxy']  = np.array([ params['mI'][3] ])
        self.unknowns['mIxz']  = np.array([ params['mI'][4] ])
        self.unknowns['mIyz']  = np.array([ params['mI'][5] ])
        self.unknowns['mrhox'] = np.array([ params['mrho'][0] ])
        self.unknowns['mrhoy'] = np.array([ params['mrho'][1] ])
        self.unknowns['mrhoz'] = np.array([ params['mrho'][2] ])

        # Prepare point forces at RNA node
        self.unknowns['plidx'] = np.array([ len(params['z'])-1 ], dtype=np.int_)
        self.unknowns['Fx']    = np.array([ params['rna_F'][0] ])
        self.unknowns['Fy']    = np.array([ params['rna_F'][1] ])
        self.unknowns['Fz']    = np.array([ params['rna_F'][2] ])
        self.unknowns['Mxx']   = np.array([ params['rna_M'][0] ])
        self.unknowns['Myy']   = np.array([ params['rna_M'][1] ])
        self.unknowns['Mzz']   = np.array([ params['rna_M'][2] ])

    def list_deriv_vars(self):
        inputs = ('mass', 'mI', 'mrho', 'rna_F', 'rna_M')
        outputs = ('m', 'mIxx', 'mIyy', 'mIzz', 'mIxy', 'mIxz', 'mIyz', 'Fx', 'Fy', 'Fz', 'Mxx', 'Myy', 'Mzz')
        return inputs, outputs
        
    def linearize(self, params, unknowns, resids):
        J = {}
        inp,out = self.list_deriv_vars()
        for o in out:
            for i in inp:
                J[o,i] = np.zeros( (len(unknowns[o]), len(params[i])) )
                
        J['m','mass']    = 1.0
        J['mIxx','mI']   = np.eye(6)[0,:]
        J['mIyy','mI']   = np.eye(6)[1,:]
        J['mIzz','mI']   = np.eye(6)[2,:]
        J['mIxy','mI']   = np.eye(6)[3,:]
        J['mIxz','mI']   = np.eye(6)[4,:]
        J['mIyz','mI']   = np.eye(6)[5,:]
        J['Fx','rna_F']  = np.eye(3)[0,:]
        J['Fy','rna_F']  = np.eye(3)[2,:]
        J['Fz','rna_F']  = np.eye(3)[2,:]
        J['Mxx','rna_M'] = np.eye(3)[0,:]
        J['Myy','rna_M'] = np.eye(3)[2,:]
        J['Mzz','rna_M'] = np.eye(3)[2,:]

        
class TowerPostFrame(Component):
    def __init__(self, nFull, nDEL):
        super(TowerPostFrame, self).__init__()

        # effective geometry -- used for handbook methods to estimate hoop stress, buckling, fatigue
        self.add_param('z', np.zeros(nFull), units='m', desc='location along tower. start at bottom and go to top')
        self.add_param('d', np.zeros(nFull), units='m', desc='effective tower diameter for section')
        self.add_param('t', np.zeros(nFull), units='m', desc='effective shell thickness for section')
        self.add_param('L_reinforced', 0.0, units='m', desc='buckling length')

        # Material properties
        self.add_param('E', 0.0, units='N/m**2', desc='modulus of elasticity')

        # Processed Frame3DD outputs
        self.add_param('Fz', np.zeros(nFull), units='N', desc='Axial foce in vertical z-direction in cylinder structure.')
        self.add_param('Mxx', np.zeros(nFull), units='N*m', desc='Moment about x-axis in cylinder structure.')
        self.add_param('Myy', np.zeros(nFull), units='N*m', desc='Moment about y-axis in cylinder structure.')
        self.add_param('axial_stress', val=np.zeros(nFull), units='N/m**2', desc='axial stress in tower elements')
        self.add_param('shear_stress', val=np.zeros(nFull), units='N/m**2', desc='shear stress in tower elements')
        self.add_param('hoop_stress' , val=np.zeros(nFull), units='N/m**2', desc='hoop stress in tower elements')

        # safety factors
        self.add_param('gamma_f', 1.35, desc='safety factor on loads')
        self.add_param('gamma_m', 1.1, desc='safety factor on materials')
        self.add_param('gamma_n', 1.0, desc='safety factor on consequence of failure')
        self.add_param('gamma_b', 1.1, desc='buckling safety factor')
        self.add_param('sigma_y', 0.0, units='N/m**2', desc='yield stress')
        self.add_param('gamma_fatigue', 1.755, desc='total safety factor for fatigue')

        # fatigue parameters
        self.add_param('life', 20.0, desc='fatigue life of tower')
        self.add_param('m_SN', 4, desc='slope of S/N curve', pass_by_obj=True)
        self.add_param('DC', 80.0, desc='standard value of stress')
        self.add_param('z_DEL', np.zeros(nDEL), desc='absolute z coordinates of corresponding fatigue parameters', pass_by_obj=True)
        self.add_param('M_DEL', np.zeros(nDEL), desc='fatigue parameters at corresponding z coordinates', pass_by_obj=True)

        # outputs
        self.add_output('f1', 0.0, units='Hz', desc='First natural frequency')
        self.add_output('f2', 0.0, units='Hz', desc='Second natural frequency')
        self.add_output('top_deflection', 0.0, units='m', desc='Deflection of tower top in yaw-aligned +x direction')
        self.add_output('stress', np.zeros(nFull), desc='Von Mises stress utilization along tower at specified locations.  incudes safety factor.')
        self.add_output('shell_buckling', np.zeros(nFull), desc='Shell buckling constraint.  Should be < 1 for feasibility.  Includes safety factors')
        self.add_output('global_buckling', np.zeros(nFull), desc='Global buckling constraint.  Should be < 1 for feasibility.  Includes safety factors')
        self.add_output('damage', np.zeros(nFull), desc='Fatigue damage at each tower section')
        self.add_output('turbine_F', val=np.zeros(3), units='N', desc='Total force on tower+rna')
        self.add_output('turbine_M', val=np.zeros(3), units='N*m', desc='Total x-moment on tower+rna measured at base')
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5

        
    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack some variables
        axial_stress = params['axial_stress']
        shear_stress = params['shear_stress']
        hoop_stress  = params['hoop_stress']
        sigma_y      = params['sigma_y'] * np.ones(axial_stress.shape)
        E            = params['E'] * np.ones(axial_stress.shape)
        L_reinforced = params['L_reinforced'] * np.ones(axial_stress.shape)
        
        # von mises stress
        unknowns['stress'] = Util.vonMisesStressUtilization(axial_stress, hoop_stress, shear_stress,
                      params['gamma_f']*params['gamma_m']*params['gamma_n'], sigma_y)

        # shell buckling
        unknowns['shell_buckling'] = Util.shellBucklingEurocode(params['d'], params['t'], axial_stress, hoop_stress,
                                                                shear_stress, L_reinforced, E, sigma_y, params['gamma_f'], params['gamma_b'])

        # global buckling
        tower_height = params['z'][-1] - params['z'][0]
        M = np.sqrt(params['Mxx']**2 + params['Myy']**2)
        unknowns['global_buckling'] = Util.bucklingGL(params['d'], params['t'], params['Fz'], M, tower_height, E,
                                                      sigma_y, params['gamma_f'], params['gamma_b'])

        # fatigue
        N_DEL = 365.0*24.0*3600.0*params['life'] * np.ones(len(params['z']))
        unknowns['damage'] = np.zeros(N_DEL.shape)
        if any(params['M_DEL']):
            M_DEL = np.interp(params['z'], params['z_DEL'], params['M_DEL'])

            unknowns['damage'] = Util.fatigue(M_DEL, N_DEL, params['d'], params['t'], params['m_SN'],
                                              params['DC'], params['gamma_fatigue'], stress_factor=1.0, weld_factor=True)

# -----------------
#  Assembly
# -----------------

class TowerLeanSE(Group):

    def __init__(self, nPoints, nFull):
        super(TowerLeanSE, self).__init__()

        nRefine = (nFull-1)/(nPoints-1)
        
        # Independent variables that are unique to TowerSE
        self.add('tower_section_height', IndepVarComp('tower_section_height', np.zeros(nPoints-1)), promotes=['*'])
        self.add('tower_outer_diameter', IndepVarComp('tower_outer_diameter', np.zeros(nPoints)), promotes=['*'])
        self.add('tower_wall_thickness', IndepVarComp('tower_wall_thickness', np.zeros(nPoints)), promotes=['*'])
        self.add('tower_outfitting_factor', IndepVarComp('tower_outfitting_factor', 0.0), promotes=['*'])
        self.add('tower_buckling_length', IndepVarComp('tower_buckling_length', 0.0), promotes=['*'])

        # All the static components
        self.add('geometry', CylinderDiscretization(nPoints, nRefine), promotes=['*'])
        self.add('tgeometry', TowerDiscretization(), promotes=['hub_height','height_constraint'])
        
        self.add('cm', CylinderMass(nFull), promotes=['material_density','z_full','d_full','t_full'])
        self.add('tm', TowerMass(nFull), promotes=['tower_mass','tower_center_of_mass','tower_I_base'])
        self.add('gc', Util.GeometricConstraints(nPoints), promotes=['min_d_to_t','min_taper','manufacturability','weldability'])
        self.add('turb', TurbineMass(), promotes=['turbine_mass','rna_mass', 'rna_cg', 'rna_I'])

        # Connections for geometry and mass
        self.connect('tower_section_height', 'section_height')
        self.connect('tower_outer_diameter', ['diameter', 'gc.d'])
        self.connect('tower_wall_thickness', ['wall_thickness', 'gc.t'])
        self.connect('tower_outfitting_factor', 'cm.outfitting_factor')
        self.connect('z_param', 'tgeometry.z_end', src_indices=[nPoints-1])
        self.connect('hub_height', 'turb.hubH')

        self.connect('cm.mass', 'tm.cylinder_mass')
        self.connect('cm.center_of_mass', 'tm.cylinder_center_of_mass')
        self.connect('cm.section_center_of_mass','tm.cylinder_section_center_of_mass')
        self.connect('cm.I_base','tm.cylinder_I_base')
        self.connect('tower_mass', 'turb.tower_mass')
        self.connect('tower_center_of_mass', 'turb.tower_center_of_mass')
        self.connect('tower_I_base', 'turb.tower_I_base')

        
class TowerSE(Group):

    def __init__(self, nLC, nPoints, nFull, nDEL, wind=''):

        super(TowerSE, self).__init__()
        
        # Independent variables that are unique to TowerSE
        self.add('tower_M_DEL', IndepVarComp('tower_M_DEL', np.zeros(nDEL), pass_by_obj=True), promotes=['*'])
        self.add('tower_z_DEL', IndepVarComp('tower_z_DEL', np.zeros(nDEL), pass_by_obj=True), promotes=['*'])
        self.add('tower_force_discretization', IndepVarComp('tower_force_discretization', 5.0), promotes=['*'])

        self.add('geom', TowerLeanSE(nPoints, nFull), promotes=['*'])
        self.add('props', CylindricalShellProperties(nFull))

        # Connections for geometry and mass
        self.connect('d_full', 'props.d')
        self.connect('t_full', 'props.t')
        
        # Add in all Components that drive load cases
        # Note multiple load cases have to be handled by replicating components and not groups/assemblies.
        # Replicating Groups replicates the IndepVarComps which doesn't play nicely in OpenMDAO
        for iLC in xrange(nLC):
            lc = '' if nLC==1 else str(iLC+1)
            
            if wind.lower() == 'powerwind':
                self.add('wind'+lc, PowerWind(nFull), promotes=['z0'])
            elif wind.lower() == 'logwind':
                self.add('wind'+lc, LogWind(nFull), promotes=['z0'])
            else:
                raise ValueError('Unknown wind type, '+wind)

            self.add('wave'+lc, WaveBase(nFull), promotes=['z_floor'])
            self.add('windLoads'+lc, CylinderWindDrag(nFull), promotes=['cd_usr'])
            self.add('waveLoads'+lc, CylinderWaveDrag(nFull), promotes=['cm','cd_usr'])
            self.add('distLoads'+lc, AeroHydroLoads(nFull))#, promotes=['yaw'])

            self.add('pre'+lc, TowerPreFrame(nFull))
            self.add('tower'+lc, CylinderFrame3DD(nFull, 1, 1, 1), promotes=['E','G','tol','Mmethod','geom','lump','shear',
                                                                             'nM','shift'])
            self.add('post'+lc, TowerPostFrame(nFull, nDEL), promotes=['E','sigma_y','DC','life','m_SN',
                                                                      'gamma_b','gamma_f','gamma_fatigue','gamma_m','gamma_n'])
            
            self.connect('z_full', ['wind'+lc+'.z', 'wave'+lc+'.z', 'windLoads'+lc+'.z', 'waveLoads'+lc+'.z', 'distLoads'+lc+'.z', 'pre'+lc+'.z', 'tower'+lc+'.z', 'post'+lc+'.z'])
            self.connect('d_full', ['windLoads'+lc+'.d', 'waveLoads'+lc+'.d', 'tower'+lc+'.d', 'post'+lc+'.d'])

            self.connect('rna_mass', 'pre'+lc+'.mass')
            self.connect('rna_cg', 'pre'+lc+'.mrho')
            self.connect('rna_I', 'pre'+lc+'.mI')
        
            self.connect('material_density', 'tower'+lc+'.rho')

            self.connect('pre'+lc+'.kidx', 'tower'+lc+'.kidx')
            self.connect('pre'+lc+'.kx', 'tower'+lc+'.kx')
            self.connect('pre'+lc+'.ky', 'tower'+lc+'.ky')
            self.connect('pre'+lc+'.kz', 'tower'+lc+'.kz')
            self.connect('pre'+lc+'.ktx', 'tower'+lc+'.ktx')
            self.connect('pre'+lc+'.kty', 'tower'+lc+'.kty')
            self.connect('pre'+lc+'.ktz', 'tower'+lc+'.ktz')
            self.connect('pre'+lc+'.midx', 'tower'+lc+'.midx')
            self.connect('pre'+lc+'.m', 'tower'+lc+'.m')
            self.connect('pre'+lc+'.mIxx', 'tower'+lc+'.mIxx')
            self.connect('pre'+lc+'.mIyy', 'tower'+lc+'.mIyy')
            self.connect('pre'+lc+'.mIzz', 'tower'+lc+'.mIzz')
            self.connect('pre'+lc+'.mIxy', 'tower'+lc+'.mIxy')
            self.connect('pre'+lc+'.mIxz', 'tower'+lc+'.mIxz')
            self.connect('pre'+lc+'.mIyz', 'tower'+lc+'.mIyz')

            self.connect('pre'+lc+'.plidx', 'tower'+lc+'.plidx')
            self.connect('pre'+lc+'.Fx', 'tower'+lc+'.Fx')
            self.connect('pre'+lc+'.Fy', 'tower'+lc+'.Fy')
            self.connect('pre'+lc+'.Fz', 'tower'+lc+'.Fz')
            self.connect('pre'+lc+'.Mxx', 'tower'+lc+'.Mxx')
            self.connect('pre'+lc+'.Myy', 'tower'+lc+'.Myy')
            self.connect('pre'+lc+'.Mzz', 'tower'+lc+'.Mzz')
            self.connect('tower_force_discretization', 'tower'+lc+'.dx')
            self.connect('t_full', ['tower'+lc+'.t','post'+lc+'.t'])

            self.connect('tower'+lc+'.Fz_out', 'post'+lc+'.Fz')
            self.connect('tower'+lc+'.Mxx_out', 'post'+lc+'.Mxx')
            self.connect('tower'+lc+'.Myy_out', 'post'+lc+'.Myy')
            self.connect('tower'+lc+'.axial_stress', 'post'+lc+'.axial_stress')
            self.connect('tower'+lc+'.shear_stress', 'post'+lc+'.shear_stress')
            self.connect('tower'+lc+'.hoop_stress_euro', 'post'+lc+'.hoop_stress')
        
            # connections to wind1
            self.connect('z0', 'wave'+lc+'.z_surface')
            #self.connect('z_floor', 'waveLoads'+lc+'.wlevel')

            # connections to windLoads1
            self.connect('wind'+lc+'.U', 'windLoads'+lc+'.U')
            #self.connect('wind'+lc+'.beta', 'windLoads'+lc+'.beta')

            # connections to waveLoads1
            self.connect('wave'+lc+'.U', 'waveLoads'+lc+'.U')
            self.connect('wave'+lc+'.A', 'waveLoads'+lc+'.A')
            #self.connect('wave'+lc+'.beta', 'waveLoads'+lc+'.beta')

            # connections to distLoads1
            self.connect('windLoads'+lc+'.windLoads_Px', 'distLoads'+lc+'.windLoads_Px')
            self.connect('windLoads'+lc+'.windLoads_Py', 'distLoads'+lc+'.windLoads_Py')
            self.connect('windLoads'+lc+'.windLoads_Pz', 'distLoads'+lc+'.windLoads_Pz')
            self.connect('windLoads'+lc+'.windLoads_qdyn', 'distLoads'+lc+'.windLoads_qdyn')
            self.connect('windLoads'+lc+'.windLoads_beta', 'distLoads'+lc+'.windLoads_beta')
            #self.connect('windLoads'+lc+'.windLoads_Px0', 'distLoads'+lc+'.windLoads_Px0')
            #self.connect('windLoads'+lc+'.windLoads_Py0', 'distLoads'+lc+'.windLoads_Py0')
            #self.connect('windLoads'+lc+'.windLoads_Pz0', 'distLoads'+lc+'.windLoads_Pz0')
            #self.connect('windLoads'+lc+'.windLoads_qdyn0', 'distLoads'+lc+'.windLoads_qdyn0')
            #self.connect('windLoads'+lc+'.windLoads_beta0', 'distLoads'+lc+'.windLoads_beta0')
            self.connect('windLoads'+lc+'.windLoads_z', 'distLoads'+lc+'.windLoads_z')
            self.connect('windLoads'+lc+'.windLoads_d', 'distLoads'+lc+'.windLoads_d')

            self.connect('waveLoads'+lc+'.waveLoads:Px', 'distLoads'+lc+'.waveLoads:Px')
            self.connect('waveLoads'+lc+'.waveLoads:Py', 'distLoads'+lc+'.waveLoads:Py')
            self.connect('waveLoads'+lc+'.waveLoads:Pz', 'distLoads'+lc+'.waveLoads:Pz')
            self.connect('waveLoads'+lc+'.waveLoads:qdyn', 'distLoads'+lc+'.waveLoads:qdyn')
            self.connect('waveLoads'+lc+'.waveLoads:beta', 'distLoads'+lc+'.waveLoads:beta')
            #self.connect('waveLoads'+lc+'.waveLoads:Px0', 'distLoads'+lc+'.waveLoads:Px0')
            #self.connect('waveLoads'+lc+'.waveLoads:Py0', 'distLoads'+lc+'.waveLoads:Py0')
            #self.connect('waveLoads'+lc+'.waveLoads:Pz0', 'distLoads'+lc+'.waveLoads:Pz0')
            #self.connect('waveLoads'+lc+'.waveLoads:qdyn0', 'distLoads'+lc+'.waveLoads:qdyn0')
            #self.connect('waveLoads'+lc+'.waveLoads:beta0', 'distLoads'+lc+'.waveLoads:beta0')
            self.connect('waveLoads'+lc+'.waveLoads:z', 'distLoads'+lc+'.waveLoads:z')
            self.connect('waveLoads'+lc+'.waveLoads:d', 'distLoads'+lc+'.waveLoads:d')

            # Tower connections
            self.connect('tower_buckling_length', ['tower'+lc+'.L_reinforced', 'post'+lc+'.L_reinforced'])
            self.connect('tower_M_DEL', 'post'+lc+'.M_DEL')
            self.connect('tower_z_DEL', 'post'+lc+'.z_DEL')

            self.connect('props.Az', 'tower'+lc+'.Az')
            self.connect('props.Asx', 'tower'+lc+'.Asx')
            self.connect('props.Asy', 'tower'+lc+'.Asy')
            self.connect('props.Jz', 'tower'+lc+'.Jz')
            self.connect('props.Ixx', 'tower'+lc+'.Ixx')
            self.connect('props.Iyy', 'tower'+lc+'.Iyy')

            self.connect('distLoads'+lc+'.Px',   'tower'+lc+'.Px')
            self.connect('distLoads'+lc+'.Py',   'tower'+lc+'.Py')
            self.connect('distLoads'+lc+'.Pz',   'tower'+lc+'.Pz')
            self.connect('distLoads'+lc+'.qdyn', 'tower'+lc+'.qdyn')

        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5

        
if __name__ == '__main__':
    # --- tower setup ------
    from commonse.environment import PowerWind
    from commonse.environment import LogWind

    # --- geometry ----
    h_param = np.diff(np.array([0.0, 43.8, 87.6]))
    d_param = np.array([6.0, 4.935, 3.87])
    t_param = 1.3*np.array([0.027, 0.023, 0.019])
    L_reinforced = 30.0  # [m] buckling length
    theta_stress = 0.0
    yaw = 0.0
    Koutfitting = 1.07

    # --- material props ---
    E = 210e9
    G = 80.8e9
    rho = 8500.0
    sigma_y = 450.0e6

    # --- extra mass ----
    m = np.array([285598.8])
    mIxx = 1.14930678e+08
    mIyy = 2.20354030e+07
    mIzz = 1.87597425e+07
    mIxy = 0.0
    mIxz = 5.03710467e+05
    mIyz = 0.0
    mI = np.array([mIxx, mIyy, mIzz, mIxy, mIxz, mIyz])
    mrho = np.array([-1.13197635, 0.0, 0.50875268])
    # -----------

    # --- wind ---
    wind_zref = 90.0
    wind_z0 = 0.0
    shearExp = 0.2
    cd_usr = None
    # ---------------

    # two load cases.  TODO: use a case iterator
    
    # # --- loading case 1: max Thrust ---
    wind_Uref1 = 11.73732
    Fx1 = 1284744.19620519
    Fy1 = 0.
    Fz1 = -2914124.84400512
    Mxx1 = 3963732.76208099
    Myy1 = -2275104.79420872
    Mzz1 = -346781.68192839
    # # ---------------

    # # --- loading case 2: max wind speed ---
    wind_Uref2 = 70.0
    Fx2 = 930198.60063279
    Fy2 = 0.
    Fz2 = -2883106.12368949
    Mxx2 = -1683669.22411597
    Myy2 = -2522475.34625363
    Mzz2 = 147301.97023764
    # # ---------------

    # --- safety factors ---
    gamma_f = 1.35
    gamma_m = 1.3
    gamma_n = 1.0
    gamma_b = 1.1
    # ---------------

    # --- fatigue ---
    z_DEL = np.array([0.000, 1.327, 3.982, 6.636, 9.291, 11.945, 14.600, 17.255, 19.909, 22.564, 25.218, 27.873, 30.527, 33.182, 35.836, 38.491, 41.145, 43.800, 46.455, 49.109, 51.764, 54.418, 57.073, 59.727, 62.382, 65.036, 67.691, 70.345, 73.000, 75.655, 78.309, 80.964, 83.618, 86.273, 87.600])
    M_DEL = 1e3*np.array([8.2940E+003, 8.1518E+003, 7.8831E+003, 7.6099E+003, 7.3359E+003, 7.0577E+003, 6.7821E+003, 6.5119E+003, 6.2391E+003, 5.9707E+003, 5.7070E+003, 5.4500E+003, 5.2015E+003, 4.9588E+003, 4.7202E+003, 4.4884E+003, 4.2577E+003, 4.0246E+003, 3.7942E+003, 3.5664E+003, 3.3406E+003, 3.1184E+003, 2.8977E+003, 2.6811E+003, 2.4719E+003, 2.2663E+003, 2.0673E+003, 1.8769E+003, 1.7017E+003, 1.5479E+003, 1.4207E+003, 1.3304E+003, 1.2780E+003, 1.2673E+003, 1.2761E+003])
    nDEL = len(z_DEL)
    gamma_fatigue = 1.35*1.3*1.0
    life = 20.0
    m_SN = 4
    # ---------------


    # --- constraints ---
    min_d_to_t = 120.0
    min_taper = 0.4
    # ---------------

    # # V_max = 80.0  # tip speed
    # # D = 126.0
    # # .freq1p = V_max / (D/2) / (2*pi)  # convert to Hz

    nPoints = len(d_param)
    nFull   = 5*(nPoints-1) + 1
    wind = 'PowerWind'
    nLC = 2
    
    prob = Problem(root=TowerSE(nLC, nPoints, nFull, nDEL, wind=wind))
    prob.setup()

    if wind=='PowerWind':
        prob['wind1.shearExp'] = prob['wind2.shearExp'] = shearExp

    # assign values to params

    # --- geometry ----
    prob['hub_height'] = h_param.sum()
    prob['tower_section_height'] = h_param
    prob['tower_outer_diameter'] = d_param
    prob['tower_wall_thickness'] = t_param
    prob['tower_buckling_length'] = L_reinforced
    prob['tower_outfitting_factor'] = Koutfitting
    prob['distLoads1.yaw'] = prob['distLoads2.yaw'] = yaw
    
    # --- material props ---
    prob['E'] = E
    prob['G'] = G
    prob['material_density'] = rho
    prob['sigma_y'] = sigma_y

    # --- extra mass ----
    prob['rna_mass'] = m
    prob['rna_I'] = mI
    prob['rna_cg'] = mrho
    # -----------

    # --- wind & wave ---
    prob['wind1.zref'] = prob['wind2.zref'] = wind_zref
    prob['z0'] = wind_z0
    prob['cd_usr'] = cd_usr
    prob['windLoads1.rho'] = prob['windLoads2.rho'] = 1.225
    prob['windLoads1.mu'] = prob['windLoads2.mu'] = 1.7934e-5
    prob['wave1.rho'] = prob['wave2.rho'] = prob['waveLoads1.rho'] = prob['waveLoads2.rho'] = 1025.0
    prob['waveLoads1.mu'] = prob['waveLoads2.mu'] = 1.3351e-3
    prob['windLoads1.beta'] = prob['windLoads2.beta'] = prob['waveLoads1.beta'] = prob['waveLoads2.beta'] = 0.0
    #prob['waveLoads1.U0'] = prob['waveLoads1.A0'] = prob['waveLoads1.beta0'] = prob['waveLoads2.U0'] = prob['waveLoads2.A0'] = prob['waveLoads2.beta0'] = 0.0
    # ---------------

    # --- safety factors ---
    prob['gamma_f'] = gamma_f
    prob['gamma_m'] = gamma_m
    prob['gamma_n'] = gamma_n
    prob['gamma_b'] = gamma_b
    prob['gamma_fatigue'] = gamma_fatigue
    # ---------------

    prob['DC'] = 80.0
    prob['shear'] = True
    prob['geom'] = False
    prob['tower_force_discretization'] = 5.0
    prob['nM'] = 2
    prob['Mmethod'] = 1
    prob['lump'] = 0
    prob['tol'] = 1e-9
    prob['shift'] = 0.0

    
    # --- fatigue ---
    prob['tower_z_DEL'] = z_DEL
    prob['tower_M_DEL'] = M_DEL
    prob['life'] = life
    prob['m_SN'] = m_SN
    # ---------------

    # --- constraints ---
    prob['min_d_to_t'] = min_d_to_t
    prob['min_taper'] = min_taper
    # ---------------


    # # --- loading case 1: max Thrust ---
    prob['wind1.Uref'] = wind_Uref1

    prob['pre1.rna_F'] = np.array([Fx1, Fy1, Fz1])
    prob['pre1.rna_M'] = np.array([Mxx1, Myy1, Mzz1])
    # # ---------------


    # # --- loading case 2: max Wind Speed ---
    prob['wind2.Uref'] = wind_Uref2

    prob['pre2.rna_F'] = np.array([Fx2, Fy2, Fz2])
    prob['pre2.rna_M' ] = np.array([Mxx2, Myy2, Mzz2])

    # # --- run ---
    prob.run()

    z = prob['z_full']

    print 'zs=', z
    print 'ds=', prob['d_full']
    print 'ts=', prob['t_full']
    print 'mass (kg) =', prob['tower_mass']
    print 'cg (m) =', prob['tower_center_of_mass']
    print 'weldability =', prob['weldability']
    print 'manufacturability =', prob['manufacturability']
    print '\nwind: ', prob['wind1.Uref']
    print 'f1 (Hz) =', prob['tower1.f1']
    print 'top_deflection1 (m) =', prob['post1.top_deflection']
    print 'stress1 =', prob['post1.stress']
    print 'GL buckling =', prob['post1.global_buckling']
    print 'Shell buckling =', prob['post1.shell_buckling']
    print 'damage =', prob['post1.damage']
    print '\nwind: ', prob['wind2.Uref']
    print 'f1 (Hz) =', prob['tower2.f1']
    print 'top_deflection2 (m) =', prob['post2.top_deflection']
    print 'stress2 =', prob['post2.stress']
    print 'GL buckling =', prob['post2.global_buckling']
    print 'Shell buckling =', prob['post2.shell_buckling']
    print 'damage =', prob['post2.damage']


    stress1 = np.copy( prob['post1.stress'] )
    shellBuckle1 = np.copy( prob['post1.shell_buckling'] )
    globalBuckle1 = np.copy( prob['post1.global_buckling'] )
    damage1 = np.copy( prob['post1.damage'] )

    stress2 = prob['post2.stress']
    shellBuckle2 = prob['post2.shell_buckling']
    globalBuckle2 = prob['post2.global_buckling']
    damage2 = prob['post2.damage']

    
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
    #plt.plot(prob['d_full']/2.+max(prob['d_full']), z, 'ok')
    #plt.plot(prob['d_full']/-2.+max(prob['d_full']), z, 'ok')

    #fig = plt.figure(3)
    #ax1 = fig.add_subplot(121)
    #ax2 = fig.add_subplot(122)

    #ax1.plot(prob['wind1.U'], z)
    #ax2.plot(prob['wind2.U'], z)
    #plt.tight_layout()
    plt.show()

    print prob['tower1.base_F']
    print prob['tower1.base_M']
    print prob['tower2.base_F']
    print prob['tower2.base_M']
    # ------------

    """
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
        tower.driver.add_constraint('tower.stress <= 1.0')
        tower.driver.add_constraint('tower.global_buckling <= 1.0')
        tower.driver.add_constraint('tower.shell_buckling <= 1.0')
        tower.driver.add_constraint('tower.damage <= 1.0')
        tower.driver.add_constraint('gc.weldability <= 0.0')
        tower.driver.add_constraint('gc.manufacturability <= 0.0')
        freq1p = 0.2  # 1P freq in Hz
        tower.driver.add_constraint('tower.f1 >= 1.1*%f' % freq1p)
        # ----------------------

        # --- run opt ---
        tower.run()
        # ---------------
    """



    
    
