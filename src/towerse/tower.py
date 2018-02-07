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

import math
import numpy as np
from openmdao.api import Component, Group, Problem, IndepVarComp

from commonse.WindWaveDrag import AeroHydroLoads, TowerWindDrag, TowerWaveDrag

from commonse.environment import WindBase, WaveBase, SoilBase, PowerWind, LogWind
from commonse.Tube import CylindricalShellProperties

from commonse import gravity, eps
import commonse.Frustum as Frustum

#from fusedwind.turbine.tower import TowerFromCSProps
#from fusedwind.interface import implement_base

import commonse.UtilizationSupplement as Util

import pyframe3dd.frame3dd as frame3dd



# -----------------
#  Components
# -----------------

#TODO need to check the length of each array
class TowerDiscretization(Component):
    """discretize geometry into finite element nodes"""

    #inputs

    def __init__(self, nPoints, nFull):
        
        super(TowerDiscretization, self).__init__()

        self.nFull = nFull
        
         # variables
        self.add_param('hub_height', val=0.0, units='m', desc='diameter at tower base')
        self.add_param('tower_section_height', np.zeros(nPoints-1), units='m', desc='parameterized section heights along tower')
        self.add_param('tower_diameter', np.zeros(nPoints), units='m', desc='tower diameter at corresponding locations')
        self.add_param('tower_wall_thickness', np.zeros(nPoints), units='m', desc='shell thickness at corresponding locations')

        #out
        self.add_output('z_param', np.zeros(nPoints), units='m', desc='parameterized locations along tower, linear lofting between')
        self.add_output('z_full', np.zeros(nFull), units='m', desc='locations along tower')
        self.add_output('d_full', np.zeros(nFull), units='m', desc='tower diameter at corresponding locations')
        self.add_output('t_full', np.zeros(nFull), units='m', desc='shell thickness at corresponding locations')
        self.add_output('height_constraint', val=0.0, units='m', desc='diameter at tower base')
        # Convenience outputs for export to other modules
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['z_param'] = np.r_[0.0, np.cumsum(params['tower_section_height'])]
        unknowns['z_full']  = np.linspace(unknowns['z_param'][0], unknowns['z_param'][-1], self.nFull) 
        unknowns['d_full']  = np.interp(unknowns['z_full'], unknowns['z_param'], params['tower_diameter'])
        unknowns['t_full']  = np.interp(unknowns['z_full'], unknowns['z_param'], params['tower_wall_thickness'])
        unknowns['height_constraint'] = params['hub_height'] - unknowns['z_param'][-1]
        
        
class TowerMass(Component):

    def __init__(self, nPoints):
        super(TowerMass, self).__init__()
        
        self.add_param('d_param', val=np.zeros(nPoints), units='m', desc='tower diameter at corresponding locations')
        self.add_param('t_param', val=np.zeros(nPoints), units='m', desc='shell thickness at corresponding locations')
        self.add_param('z_param', val=np.zeros(nPoints), units='m', desc='parameterized locations along tower, linear lofting between')
        self.add_param('rho', 0.0, units='kg/m**3', desc='material density')
        self.add_param('outfitting_factor', val=1.0, desc='Multiplier that accounts for secondary structure mass inside of tower')
        
        self.add_output('tower_mass', val=0.0, units='kg', desc='Total tower mass')
        self.add_output('tower_center_of_mass', val=0.0, units='m', desc='z-position of center of mass of tower')
        self.add_output('tower_section_center_of_mass', val=np.zeros(nPoints-1), units='m', desc='z position of center of mass of each can in the tower')
        
    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables for thickness and average radius at each can interface
        Tb = params['t_param'][:-1]
        Tt = params['t_param'][1:]
        Rb = 0.5*params['d_param'][:-1] - 0.5*Tb
        Rt = 0.5*params['d_param'][1:] - 0.5*Tt
        zz = params['z_param']
        H  = np.diff(zz)

        # Total mass of tower
        V_shell = Frustum.frustumShellVolume(Rb, Rt, Tb, Tt, H)
        unknowns['tower_mass'] = params['outfitting_factor'] * params['rho'] * V_shell.sum()
        
        # Center of mass of each can/section
        cm_section = Frustum.frustumShellCG(Rb, Rt, H)
        unknowns['tower_section_center_of_mass'] = zz[:-1] + cm_section

        # Center of mass of tower
        V_shell += eps
        unknowns['tower_center_of_mass'] = np.dot(V_shell, unknowns['tower_section_center_of_mass']) / V_shell.sum()

        
class TurbineMass(Component):

    def __init__(self):
        super(TurbineMass, self).__init__()
        
        self.add_param('hubH', val=0.0, units='m', desc='Hub-height')
        self.add_param('rna_mass', val=0.0, units='kg', desc='Total tower mass')
        self.add_param('rna_offset', val=np.zeros((3,)), units='m', desc='z-position of center of mass of tower')
        self.add_param('tower_mass', val=0.0, units='kg', desc='Total tower mass')
        self.add_param('tower_center_of_mass', val=0.0, units='m', desc='z-position of center of mass of tower')

        self.add_output('turbine_mass', val=0.0, units='kg', desc='Total mass of tower+rna')
        self.add_output('turbine_center_of_mass', val=np.zeros((3,)), units='m', desc='xyz-position of tower+rna center of mass')
        
    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['turbine_mass'] = params['rna_mass'] + params['tower_mass']
        
        cg_rna   = params['rna_offset'] + np.array([0.0, 0.0, params['hubH']])
        cg_tower = np.array([0.0, 0.0, params['tower_center_of_mass']])
        unknowns['turbine_center_of_mass'] = (params['rna_mass']*cg_rna + params['tower_mass']*cg_tower) / unknowns['turbine_mass']

        
    
#@implement_base(TowerFromCSProps)
class TowerFrame3DD(Component):

    def __init__(self, nFull, nDEL):

        super(TowerFrame3DD, self).__init__()

        # cross-sectional data along tower.
        self.add_param('z', np.zeros(nFull), units='m', desc='location along tower. start at bottom and go to top')
        self.add_param('Az', np.zeros(nFull), units='m**2', desc='cross-sectional area')
        self.add_param('Asx', np.zeros(nFull), units='m**2', desc='x shear area')
        self.add_param('Asy', np.zeros(nFull), units='m**2', desc='y shear area')
        self.add_param('Jz', np.zeros(nFull), units='m**4', desc='polar moment of inertia')
        self.add_param('Ixx', np.zeros(nFull), units='m**4', desc='area moment of inertia about x-axis')
        self.add_param('Iyy', np.zeros(nFull), units='m**4', desc='area moment of inertia about y-axis')

        self.add_param('E', 0.0, units='N/m**2', desc='modulus of elasticity')
        self.add_param('G', 0.0, units='N/m**2', desc='shear modulus')
        self.add_param('rho', 0.0, units='kg/m**3', desc='material density')
        self.add_param('sigma_y', 0.0, units='N/m**2', desc='yield stress')

        # effective geometry -- used for handbook methods to estimate hoop stress, buckling, fatigue
        self.add_param('d', np.zeros(nFull), units='m', desc='effective tower diameter for section')
        self.add_param('t', np.zeros(nFull), units='m', desc='effective shell thickness for section')
        self.add_param('L_reinforced', 0.0, units='m')

        # extra mass
        self.add_param('m', 0.0, units='kg', desc='added mass')
        self.add_param('mI', np.zeros((6,)), units='kg*m**2', desc='mass moment of inertia about some point p [xx yy zz xy xz yz]')
        self.add_param('mrho', np.zeros((3,)), units='m', desc='xyz-location of p relative to node')

        # point loads
        self.add_param('rna_F', np.zeros((3,)), units='N', desc='rna force')
        self.add_param('rna_M', np.zeros((3,)), units='N*m', desc='rna moment')

        # combined wind-water distributed loads
        #WWloads = VarTree(FluidLoads(), iotype='in', desc='combined wind and wave loads')
        self.add_param('Px', np.zeros(nFull), units='N/m', desc='force per unit length in x-direction')
        self.add_param('Py', np.zeros(nFull), units='N/m', desc='force per unit length in y-direction')
        self.add_param('Pz', np.zeros(nFull), units='N/m', desc='force per unit length in z-direction')
        self.add_param('qdyn', np.zeros(nFull), units='N/m**2', desc='dynamic pressure')

        # safety factors
        self.add_param('gamma_f', 1.35, desc='safety factor on loads')
        self.add_param('gamma_m', 1.1, desc='safety factor on materials')
        self.add_param('gamma_n', 1.0, desc='safety factor on consequence of failure')
        self.add_param('gamma_b', 1.1, desc='buckling safety factor')

        # fatigue parameters
        self.add_param('life', 20.0, desc='fatigue life of tower')
        self.add_param('m_SN', 4, desc='slope of S/N curve', pass_by_obj=True)
        self.add_param('DC', 80.0, desc='standard value of stress')
        self.add_param('gamma_fatigue', 1.755, desc='total safety factor for fatigue')
        self.add_param('z_DEL', np.zeros(nDEL), desc='absolute z coordinates of corresponding fatigue parameters', pass_by_obj=True)
        self.add_param('M_DEL', np.zeros(nDEL), desc='fatigue parameters at corresponding z coordinates', pass_by_obj=True)
        #TODO should make z relative to the height of the turbine

        # options
        self.add_param('shear', True, desc='include shear deformation', pass_by_obj=True)
        self.add_param('geom', False, desc='include geometric stiffness', pass_by_obj=True)
        self.add_param('dx', 5.0, desc='z-axis increment for internal forces')
        self.add_param('nM', 2, desc='number of desired dynamic modes of vibration (below only necessary if nM > 0)', pass_by_obj=True)
        self.add_param('Mmethod', 1, desc='1: subspace Jacobi, 2: Stodola', pass_by_obj=True)
        self.add_param('lump', 0, desc='0: consistent mass, 1: lumped mass matrix', pass_by_obj=True)
        self.add_param('tol', 1e-9, desc='mode shape tolerance')
        self.add_param('shift', 0.0, desc='shift value ... for unrestrained structures')


        # outputs
        self.add_output('f1', 0.0, units='Hz', desc='First natural frequency')
        self.add_output('f2', 0.0, units='Hz', desc='Second natural frequency')
        self.add_output('top_deflection', 0.0, units='m', desc='Deflection of tower top in yaw-aligned +x direction')
        self.add_output('stress', np.zeros(nFull), units='N/m**2', desc='Von Mises stress utilization along tower at specified locations.  incudes safety factor.')
        self.add_output('shell_buckling', np.zeros(nFull), desc='Shell buckling constraint.  Should be < 1 for feasibility.  Includes safety factors')
        self.add_output('global_buckling', np.zeros(nFull), desc='Global buckling constraint.  Should be < 1 for feasibility.  Includes safety factors')
        self.add_output('damage', np.zeros(nFull), desc='Fatigue damage at each tower section')
        self.add_output('turbine_Fx', val=0.0, units='N', desc='Total x-force on tower+rna')
        self.add_output('turbine_Fy', val=0.0, units='N', desc='Total y-force on tower+rna')
        self.add_output('turbine_Fz', val=0.0, units='N', desc='Total z-force on tower+rna')
        self.add_output('turbine_Mx', val=0.0, units='N*m', desc='Total x-moment on tower+rna measured at base')
        self.add_output('turbine_My', val=0.0, units='N*m', desc='Total y-moment on tower+rna measured at base')
        self.add_output('turbine_Mz', val=0.0, units='N*m', desc='Total z-moment on tower+rna measured at base')
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5


    def solve_nonlinear(self, params, unknowns, resids):

        # ------- node data ----------------
        z = params['z']
        n = len(z)
        node = np.arange(1, n+1, dtype=np.int64)
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)

        myones = np.ones(node.shape)
        E   = params['E'] * myones
        G   = params['G'] * myones
        rho = params['rho'] * myones
        sigma_y = params['sigma_y'] * myones

        nodes = frame3dd.NodeData(node, x, y, z, r)
        # -----------------------------------

        # ------ reaction data ------------

        # rigid base
        rnode = np.array([ node[0] ])  # add one because 0-based index but 1-based node numbering
        rigid = np.inf
        kx = ky = kz = ktx = kty = ktz = rigid * np.ones(rnode.shape)

        reactions = frame3dd.ReactionData(rnode, kx, ky, kz, ktx, kty, ktz, rigid)
        # -----------------------------------

        # ------ frame element data ------------
        element = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n+1)

        roll = np.zeros(n-1)

        # average across element b.c. frame3dd uses constant section elements
        Az  = 0.5*(params['Az'][:-1] + params['Az'][1:])
        Asx = 0.5*(params['Asx'][:-1] + params['Asx'][1:])
        Asy = 0.5*(params['Asy'][:-1] + params['Asy'][1:])
        Jz  = 0.5*(params['Jz'][:-1] + params['Jz'][1:])
        Ixx = 0.5*(params['Ixx'][:-1] + params['Ixx'][1:])
        Iyy = 0.5*(params['Iyy'][:-1] + params['Iyy'][1:])
        Ee = 0.5*(E[:-1] + E[1:])
        Ge = 0.5*(G[:-1] + G[1:])
        rhoe = 0.5*(rho[:-1] + rho[1:])
        
        elements = frame3dd.ElementData(element, N1, N2, Az, Asx, Asy, Jz,
                                        Ixx, Iyy, Ee, Ge, roll, rhoe)
        # -----------------------------------


        # ------ options ------------
        options = frame3dd.Options(params['shear'], params['geom'], params['dx'])
        # -----------------------------------

        # initialize frame3dd object
        tower = frame3dd.Frame(nodes, reactions, elements, options)


        # ------ add extra mass ------------

        # extra node inertia data
        N = np.array([ node[-1] ])
        m = np.array([ params['m'] ])
        mIxx = np.array([ params['mI'][0] ])
        mIyy = np.array([ params['mI'][1] ])
        mIzz = np.array([ params['mI'][2] ])
        mIxy = np.array([ params['mI'][3] ])
        mIxz = np.array([ params['mI'][4] ])
        mIyz = np.array([ params['mI'][5] ])
        mrhox = np.array([ params['mrho'][0] ])
        mrhoy = np.array([ params['mrho'][1] ])
        mrhoz = np.array([ params['mrho'][2] ])

        tower.changeExtraNodeMass(N, m, mIxx, mIyy, mIzz, mIxy, mIxz, mIyz, mrhox, mrhoy, mrhoz, True)

        # ------------------------------------

        # ------- enable dynamic analysis ----------
        tower.enableDynamics(params['nM'], params['Mmethod'], params['lump'], params['tol'], params['shift'])
        # ----------------------------

        # ------ static load case 1 ------------

        # gravity in the X, Y, Z, directions (global)
        gx = 0.0
        gy = 0.0
        gz = -gravity

        load = frame3dd.StaticLoadCase(gx, gy, gz)

        # point loads for rna
        nF = np.array([ node[-1] ])
        Fx = np.array([ params['rna_F'][0] ])
        Fy = np.array([ params['rna_F'][1] ])
        Fz = np.array([ params['rna_F'][2] ])
        Mx = np.array([ params['rna_M'][0] ])
        My = np.array([ params['rna_M'][1] ])
        Mz = np.array([ params['rna_M'][2] ])
        load.changePointLoads(nF, Fx, Fy, Fz, Mx, My, Mz)

        # distributed loads
        Px, Py, Pz = params['Pz'], params['Py'], -params['Px']  # switch to local c.s.
        z = params['z']

        # trapezoidally distributed loads
        EL = np.arange(1, n)
        xx1 = np.zeros(n-1)
        xx2 = z[1:] - z[:-1] - np.ones(n-1)*1e-6  # subtract small number b.c. of precision
        wx1 = Px[:-1]
        wx2 = Px[1:]
        xy1 = np.zeros(n-1)
        xy2 = z[1:] - z[:-1] - np.ones(n-1)*1e-6
        wy1 = Py[:-1]
        wy2 = Py[1:]
        xz1 = np.zeros(n-1)
        xz2 = z[1:] - z[:-1] - np.ones(n-1)*1e-6
        wz1 = Pz[:-1]
        wz2 = Pz[1:]

        load.changeTrapezoidalLoads(EL, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)

        tower.addLoadCase(load)

        # -----------------------------------
        # run the analysis
        displacements, forces, reactions, internalForces, mass, modal = tower.run()
        iCase = 0

        # natural frequncies
        unknowns['f1'] = modal.freq[0]
        unknowns['f2'] = modal.freq[1]

        # deflections due to loading (from tower top and wind/wave loads)
        unknowns['top_deflection'] = displacements.dx[iCase, n-1]  # in yaw-aligned direction


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

        # Record total forces and moments
        unknowns['turbine_Fx'] = Vx[0]
        unknowns['turbine_Fy'] = Vy[0]
        unknowns['turbine_Fz'] = Fz[0]
        unknowns['turbine_Mx'] = Mxx[0]
        unknowns['turbine_My'] = Myy[0]
        unknowns['turbine_Mz'] = Mzz[0]
        
        # axial and shear stress
        ##R = self.d/2.0
        ##x_stress = R*np.cos(self.theta_stress)
        ##y_stress = R*np.sin(self.theta_stress)
        ##axial_stress = Fz/self.Az + Mxx/self.Ixx*y_stress - Myy/self.Iyy*x_stress
#        V = Vy*x_stress/R - Vx*y_stress/R  # shear stress orthogonal to direction x,y
#        shear_stress = 2. * V / self.Az  # coefficient of 2 for a hollow circular section, but should be conservative for other shapes
        axial_stress = Fz/params['Az'] - np.sqrt(Mxx**2+Myy**2)/params['Iyy']*params['d']/2.0  #More conservative, just use the tilted bending and add total max shear as well at the same point, if you do not like it go back to the previous lines

        shear_stress = 2. * np.sqrt(Vx**2+Vy**2) / params['Az'] # coefficient of 2 for a hollow circular section, but should be conservative for other shapes

        # hoop_stress (Eurocode method)
        L_reinforced = params['L_reinforced'] * np.ones( params['z'].shape )
        hoop_stress = Util.hoopStressEurocode(params['z'], params['d'], params['t'], L_reinforced, params['qdyn'])

        # von mises stress
        unknowns['stress'] = Util.vonMisesStressUtilization(axial_stress, hoop_stress, shear_stress,
                      params['gamma_f']*params['gamma_m']*params['gamma_n'], sigma_y)

        # shell buckling
        unknowns['shell_buckling'] = Util.shellBucklingEurocode(params['d'], params['t'], axial_stress, hoop_stress,
                      shear_stress, L_reinforced, E, sigma_y, params['gamma_f'], params['gamma_b'])

        # global buckling
        tower_height = params['z'][-1] - params['z'][0]
        M = np.sqrt(Mxx**2 + Myy**2)
        unknowns['global_buckling'] = Util.bucklingGL(params['d'], params['t'], Fz, M, tower_height, E,
            sigma_y, params['gamma_f'], params['gamma_b'])

        # fatigue
        N_DEL = [365*24*3600*params['life']]*len(z)
        unknowns['damage']=np.zeros(z.size)
        if any(params['M_DEL']):
            M_DEL = np.interp(z, params['z_DEL'], params['M_DEL'])

            unknowns['damage'] = Util.fatigue(M_DEL, N_DEL, params['d'], params['t'], params['m_SN'], params['DC'], params['gamma_fatigue'], stress_factor=1.0, weld_factor=True)

        # TODO: more hack NOT SURE WHAT THIS IS, but it was there originally, commented out for now
#        damage = np.concatenate([np.zeros(len(self.z)-len(z)), damage])


# -----------------
#  Assembly
# -----------------

class TowerSE(Group):

    def __init__(self, nLC, nPoints, nFull, nDEL, wind=''):

        super(TowerSE, self).__init__()

        # Independent variables that are unique to TowerSE
        self.add('tower_section_height', IndepVarComp('tower_section_height', np.zeros(nPoints-1)), promotes=['*'])
        self.add('tower_diameter', IndepVarComp('tower_diameter', np.zeros(nPoints)), promotes=['*'])
        self.add('tower_wall_thickness', IndepVarComp('tower_wall_thickness', np.zeros(nPoints)), promotes=['*'])
        self.add('tower_outfitting_factor', IndepVarComp('tower_outfitting_factor', 0.0), promotes=['*'])
        self.add('tower_buckling_length', IndepVarComp('tower_buckling_length', 0.0), promotes=['*'])
        self.add('tower_M_DEL', IndepVarComp('tower_M_DEL', np.zeros(nDEL), pass_by_obj=True), promotes=['*'])
        self.add('tower_z_DEL', IndepVarComp('tower_z_DEL', np.zeros(nDEL), pass_by_obj=True), promotes=['*'])
        self.add('tower_force_discretization', IndepVarComp('tower_force_discretization', 5.0), promotes=['*'])
        

        # All the static components
        self.add('geometry', TowerDiscretization(nPoints, nFull), promotes=['hub_height'])
        self.add('tm', TowerMass(nPoints), promotes=['rho','tower_mass','tower_center_of_mass'])
        self.add('props', CylindricalShellProperties(nFull))
        self.add('gc', Util.GeometricConstraints(nPoints), promotes=['min_d_to_t','min_taper','manufacturability','weldability'])
        self.add('turb', TurbineMass(), promotes=['rna_mass', 'rna_offset'])

        # Connections for geometry and mass
        self.connect('tower_section_height', 'geometry.tower_section_height')
        self.connect('tower_diameter', ['geometry.tower_diameter', 'tm.d_param', 'gc.d'])
        self.connect('tower_wall_thickness', ['geometry.tower_wall_thickness', 'tm.t_param', 'gc.t'])
        self.connect('tower_outfitting_factor', 'tm.outfitting_factor')
        self.connect('geometry.d_full', 'props.d')
        self.connect('geometry.t_full', 'props.t')
        self.connect('geometry.z_param', 'tm.z_param')
        self.connect('hub_height', 'turb.hubH')

        self.connect('tower_mass', 'turb.tower_mass')
        self.connect('tower_center_of_mass', 'turb.tower_center_of_mass')

        
        # Add in all Components that drive load cases
        # Note multiple load cases have to be handled by replicating components and not groups/assemblies.
        # Replicating Groups replicates the IndepVarComps which doesn't play nicely in OpenMDAO
        for iLC in xrange(nLC):
            lc = '' if nLC==1 else str(iLC+1)
            
            if wind.lower() == 'powerwind':
                self.add('wind'+lc, PowerWind(nFull), promotes=['z0'])#, promotes=['Uref','zref','shearExp','z0','betaWind'])
            elif wind.lower() == 'logwind':
                self.add('wind'+lc, LogWind(nFull), promotes=['z0'])#, promotes=['Uref','zref','z_roughness','z0','betaWind'])
            else:
                raise ValueError('Unknown wind type, '+wind)

            self.add('wave'+lc, WaveBase(nFull), promotes=['z_floor'])
            self.add('windLoads'+lc, TowerWindDrag(nFull), promotes=['cd_usr'])
            self.add('waveLoads'+lc, TowerWaveDrag(nFull), promotes=['cm'])#, promotes=['U0','A0','beta0','cm'])
            self.add('distLoads'+lc, AeroHydroLoads(nFull))#, promotes=['yaw'])
            self.add('tower'+lc, TowerFrame3DD(nFull, nDEL), promotes=['DC','E','G','sigma_y','mI',
                                                                    'tol','Mmethod','geom','lump','shear','m_SN','nM','shift','life',
                                                                    'gamma_b','gamma_f','gamma_fatigue','gamma_m','gamma_n'])
            #'shell_buckling','global_buckling','stress','damage','top_deflection','f1','f2',
            #'turbine_Fx','turbine_Fy','turbine_Fz','turbine_Mx','turbine_My','turbine_Mz'])
            
            self.connect('geometry.z_full', ['wind'+lc+'.z', 'wave'+lc+'.z', 'windLoads'+lc+'.z', 'waveLoads'+lc+'.z', 'distLoads'+lc+'.z', 'tower'+lc+'.z'])
            self.connect('geometry.d_full', ['windLoads'+lc+'.d', 'waveLoads'+lc+'.d', 'tower'+lc+'.d'])
            self.connect('tower_force_discretization', 'tower'+lc+'.dx')

            self.connect('rho','tower'+lc+'.rho')
            self.connect('rna_mass', 'tower'+lc+'.m')
            self.connect('rna_offset', 'tower'+lc+'.mrho')
        
            self.connect('geometry.t_full', 'tower'+lc+'.t')
        
            # connections to wind1
            self.connect('z0', 'wave'+lc+'.z_surface')
            self.connect('z_floor', 'waveLoads'+lc+'.wlevel')

            # connections to windLoads1
            self.connect('wind'+lc+'.U', 'windLoads'+lc+'.U')
            self.connect('wind'+lc+'.beta', 'windLoads'+lc+'.beta')
            self.connect('cd_usr', 'waveLoads'+lc+'.cd_usr')

            # connections to waveLoads1
            self.connect('wave'+lc+'.U', 'waveLoads'+lc+'.U')
            self.connect('wave'+lc+'.A', 'waveLoads'+lc+'.A')
            self.connect('wave'+lc+'.beta', 'waveLoads'+lc+'.beta')

            # connections to distLoads1
            self.connect('windLoads'+lc+'.windLoads:Px', 'distLoads'+lc+'.windLoads:Px')
            self.connect('windLoads'+lc+'.windLoads:Py', 'distLoads'+lc+'.windLoads:Py')
            self.connect('windLoads'+lc+'.windLoads:Pz', 'distLoads'+lc+'.windLoads:Pz')
            self.connect('windLoads'+lc+'.windLoads:qdyn', 'distLoads'+lc+'.windLoads:qdyn')
            self.connect('windLoads'+lc+'.windLoads:beta', 'distLoads'+lc+'.windLoads:beta')
            self.connect('windLoads'+lc+'.windLoads:Px0', 'distLoads'+lc+'.windLoads:Px0')
            self.connect('windLoads'+lc+'.windLoads:Py0', 'distLoads'+lc+'.windLoads:Py0')
            self.connect('windLoads'+lc+'.windLoads:Pz0', 'distLoads'+lc+'.windLoads:Pz0')
            self.connect('windLoads'+lc+'.windLoads:qdyn0', 'distLoads'+lc+'.windLoads:qdyn0')
            self.connect('windLoads'+lc+'.windLoads:beta0', 'distLoads'+lc+'.windLoads:beta0')
            self.connect('windLoads'+lc+'.windLoads:z', 'distLoads'+lc+'.windLoads:z')
            self.connect('windLoads'+lc+'.windLoads:d', 'distLoads'+lc+'.windLoads:d')

            self.connect('waveLoads'+lc+'.waveLoads:Px', 'distLoads'+lc+'.waveLoads:Px')
            self.connect('waveLoads'+lc+'.waveLoads:Py', 'distLoads'+lc+'.waveLoads:Py')
            self.connect('waveLoads'+lc+'.waveLoads:Pz', 'distLoads'+lc+'.waveLoads:Pz')
            self.connect('waveLoads'+lc+'.waveLoads:qdyn', 'distLoads'+lc+'.waveLoads:qdyn')
            self.connect('waveLoads'+lc+'.waveLoads:beta', 'distLoads'+lc+'.waveLoads:beta')
            self.connect('waveLoads'+lc+'.waveLoads:Px0', 'distLoads'+lc+'.waveLoads:Px0')
            self.connect('waveLoads'+lc+'.waveLoads:Py0', 'distLoads'+lc+'.waveLoads:Py0')
            self.connect('waveLoads'+lc+'.waveLoads:Pz0', 'distLoads'+lc+'.waveLoads:Pz0')
            self.connect('waveLoads'+lc+'.waveLoads:qdyn0', 'distLoads'+lc+'.waveLoads:qdyn0')
            self.connect('waveLoads'+lc+'.waveLoads:beta0', 'distLoads'+lc+'.waveLoads:beta0')
            self.connect('waveLoads'+lc+'.waveLoads:z', 'distLoads'+lc+'.waveLoads:z')
            self.connect('waveLoads'+lc+'.waveLoads:d', 'distLoads'+lc+'.waveLoads:d')

            # Tower connections
            self.connect('tower_buckling_length', 'tower'+lc+'.L_reinforced')
            self.connect('tower_M_DEL', 'tower'+lc+'.M_DEL')
            self.connect('tower_z_DEL', 'tower'+lc+'.z_DEL')

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
    n = 15
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
    nFull = n
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
    prob['tower_diameter'] = d_param
    prob['tower_wall_thickness'] = t_param
    prob['tower_buckling_length'] = L_reinforced
    prob['tower_outfitting_factor'] = Koutfitting
    prob['distLoads1.yaw'] = prob['distLoads2.yaw'] = yaw
    
    # --- material props ---
    prob['E'] = E
    prob['G'] = G
    prob['rho'] = rho
    prob['sigma_y'] = sigma_y

    # --- extra mass ----
    prob['rna_mass'] = m
    prob['mI'] = mI
    prob['rna_offset'] = mrho
    # -----------

    # --- wind & wave ---
    prob['wind1.zref'] = prob['wind2.zref'] = wind_zref
    prob['z0'] = wind_z0
    prob['windLoads1.rho'] = prob['windLoads2.rho'] = 1.225
    prob['windLoads1.mu'] = prob['windLoads2.mu'] = 1.7934e-5
    prob['waveLoads1.rho'] = prob['waveLoads2.rho'] = 1025.0
    prob['waveLoads1.mu'] = prob['waveLoads2.mu'] = 1.3351e-3
    prob['waveLoads1.U0'] = prob['waveLoads1.A0'] = prob['waveLoads1.beta0'] = prob['waveLoads2.U0'] = prob['waveLoads2.A0'] = prob['waveLoads2.beta0'] = 0.0
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

    prob['tower1.rna_F'] = np.array([Fx1, Fy1, Fz1])
    prob['tower1.rna_M'] = np.array([Mxx1, Myy1, Mzz1])
    # # ---------------


    # # --- loading case 2: max Wind Speed ---
    prob['wind2.Uref'] = wind_Uref2

    prob['tower2.rna_F'] = np.array([Fx2, Fy2, Fz2])
    prob['tower2.rna_M' ] = np.array([Mxx2, Myy2, Mzz2])

    # # --- run ---
    prob.run()

    z = prob['geometry.z_full']

    print 'zs=', z
    print 'ds=', prob['geometry.d_full']
    print 'ts=', prob['geometry.t_full']
    print 'mass (kg) =', prob['tower_mass']
    print 'cg (m) =', prob['tower_center_of_mass']
    print 'weldability =', prob['weldability']
    print 'manufacturability =', prob['manufacturability']
    print '\nwind: ', prob['wind1.Uref']
    print 'f1 (Hz) =', prob['tower1.f1']
    print 'top_deflection1 (m) =', prob['tower1.top_deflection']
    print 'stress1 =', prob['tower1.stress']
    print 'GL buckling =', prob['tower1.global_buckling']
    print 'Shell buckling =', prob['tower1.shell_buckling']
    print 'damage =', prob['tower1.damage']
    print '\nwind: ', prob['wind2.Uref']
    print 'f1 (Hz) =', prob['tower2.f1']
    print 'top_deflection2 (m) =', prob['tower2.top_deflection']
    print 'stress2 =', prob['tower2.stress']
    print 'GL buckling =', prob['tower2.global_buckling']
    print 'Shell buckling =', prob['tower2.shell_buckling']
    print 'damage =', prob['tower2.damage']


    stress1 = np.copy( prob['tower1.stress'] )
    shellBuckle1 = np.copy( prob['tower1.shell_buckling'] )
    globalBuckle1 = np.copy( prob['tower1.global_buckling'] )
    damage1 = np.copy( prob['tower1.damage'] )

    stress2 = prob['tower2.stress']
    shellBuckle2 = prob['tower2.shell_buckling']
    globalBuckle2 = prob['tower2.global_buckling']
    damage2 = prob['tower2.damage']

    
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



    
    
