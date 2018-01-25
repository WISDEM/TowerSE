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
          - 1/2018 : G.B. modified for easier use with other modules
 """

import math
import numpy as np
from openmdao.api import Component, Group, Problem

from commonse.WindWaveDrag import AeroHydroLoads, TowerWindDrag, TowerWaveDrag

from commonse.environment import WindBase, WaveBase, SoilBase, PowerWind, LogWind

from commonse import Tube, gravity

#from fusedwind.turbine.tower import TowerFromCSProps
#from fusedwind.interface import implement_base

import commonse.UtilizationSupplement as Util

import pyframe3dd.frame3dd as frame3dd



# -----------------
#  Helper Functions
# -----------------


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
        self.add_param('h_param', np.zeros(nPoints-1), units='m', desc='parameterized section heights along tower')
        self.add_param('d_param', np.zeros(nPoints), units='m', desc='tower diameter at corresponding locations')
        self.add_param('t_param', np.zeros(nPoints), units='m', desc='shell thickness at corresponding locations')

        #out
        self.add_output('z_param', np.zeros(nPoints), units='m', desc='parameterized locations along tower, linear lofting between')
        self.add_output('z_full', np.zeros(nFull), units='m', desc='locations along tower')
        self.add_output('d_full', np.zeros(nFull), units='m', desc='tower diameter at corresponding locations')
        self.add_output('t_full', np.zeros(nFull), units='m', desc='shell thickness at corresponding locations')

        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['z_param'] = np.r_[0.0, np.cumsum(params['h_param'])]
        unknowns['z_full']  = np.linspace(unknowns['z_param'][0], unknowns['z_param'][-1], self.nFull) 
        unknowns['d_full']  = np.interp(unknowns['z_full'], unknowns['z_param'], params['d_param'])
        unknowns['t_full']  = np.interp(unknowns['z_full'], unknowns['z_param'], params['t_param'])


class CylindricalShellProperties(Component):

    def __init__(self, nFull):

        super(CylindricalShellProperties, self).__init__()

        self.add_param('d', np.zeros(nFull), units='m', desc='tower diameter at corresponding locations')
        self.add_param('t', np.zeros(nFull), units='m', desc='shell thickness at corresponding locations')

        self.add_output('Az', np.zeros(nFull), units='m**2', desc='cross-sectional area')
        self.add_output('Asx', np.zeros(nFull), units='m**2', desc='x shear area')
        self.add_output('Asy', np.zeros(nFull), units='m**2', desc='y shear area')
        self.add_output('Jz', np.zeros(nFull), units='m**4', desc='polar moment of inertia')
        self.add_output('Ixx', np.zeros(nFull), units='m**4', desc='area moment of inertia about x-axis')
        self.add_output('Iyy', np.zeros(nFull), units='m**4', desc='area moment of inertia about y-axis')

        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'


    def solve_nonlinear(self, params, unknowns, resids):

        tube = Tube(params['d'],params['t'])

        unknowns['Az'] = tube.Area
        unknowns['Asx'] = tube.Asx
        unknowns['Asy'] = tube.Asy
        unknowns['Jz'] = tube.J0
        unknowns['Ixx'] = tube.Jxx
        unknowns['Iyy'] = tube.Jyy

##        ro = self.d/2.0 + self.t/2.0
##        ri = self.d/2.0 - self.t/2.0
##        self.Az = math.pi * (ro**2 - ri**2)
##        self.Asx = self.Az / (0.54414 + 2.97294*(ri/ro) - 1.51899*(ri/ro)**2)
##        self.Asy = self.Az / (0.54414 + 2.97294*(ri/ro) - 1.51899*(ri/ro)**2)
##        self.Jz = math.pi/2.0 * (ro**4 - ri**4)
##        self.Ixx = self.Jz/2.0
##        self.Iyy = self.Jz/2.0



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

        # spring reaction data
        self.add_param('kidx', [], desc='indices of z where external stiffness reactions should be applied.', pass_by_obj=True)

        # extra mass
        self.add_param('m', 0.0, units='kg', desc='added mass')
        self.add_param('mIxx', 0.0, units='kg*m**2', desc='x mass moment of inertia about some point p')
        self.add_param('mIyy', 0.0, units='kg*m**2', desc='y mass moment of inertia about some point p')
        self.add_param('mIzz', 0.0, units='kg*m**2', desc='z mass moment of inertia about some point p')
        self.add_param('mIxy', 0.0, units='kg*m**2', desc='xy mass moment of inertia about some point p')
        self.add_param('mIxz', 0.0, units='kg*m**2', desc='xz mass moment of inertia about some point p')
        self.add_param('mIyz', 0.0, units='kg*m**2', desc='yz mass moment of inertia about some point p')
        self.add_param('mrho', np.zeros((3,)), units='m', desc='xyz-location of p relative to node')

        # point loads
        self.add_param('Fx', 0.0, units='N', desc='rna force in x-direction')
        self.add_param('Fy', 0.0, units='N', desc='rna force in y-direction')
        self.add_param('Fz', 0.0, units='N', desc='rna force in z-direction')
        self.add_param('Mxx', 0.0, units='N*m', desc='rna moment about x-axis')
        self.add_param('Myy', 0.0, units='N*m', desc='rna moment about y-axis')
        self.add_param('Mzz', 0.0, units='N*m', desc='rna moment about z-axis')

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
        self.add_param('m_SN', 4, desc='slope of S/N curve')
        self.add_param('DC', 80.0, desc='standard value of stress')
        self.add_param('gamma_fatigue', 1.755, desc='total safety factor for fatigue')
        self.add_param('z_DEL', np.zeros(nDEL), desc='absolute z coordinates of corresponding fatigue parameters')
        self.add_param('M_DEL', np.zeros(nDEL), desc='fatigue parameters at corresponding z coordinates')
        #TODO should make z relative to the height of the turbine

        # options
        self.add_param('shear', True, desc='include shear deformation')
        self.add_param('geom', False, desc='include geometric stiffness')
        self.add_param('dx', 5.0, desc='z-axis increment for internal forces')
        self.add_param('nM', 2, desc='number of desired dynamic modes of vibration (below only necessary if nM > 0)')
        self.add_param('Mmethod', 1, desc='1: subspace Jacobi, 2: Stodola')
        self.add_param('lump', 0, desc='0: consistent mass, 1: lumped mass matrix')
        self.add_param('tol', 1e-9, desc='mode shape tolerance')
        self.add_param('shift', 0.0, desc='shift value ... for unrestrained structures')


        # outputs
        self.add_output('mass', 0.0)
        self.add_output('f1', 0.0, units='Hz', desc='First natural frequency')
        self.add_output('f2', 0.0, units='Hz', desc='Second natural frequency')
        self.add_output('top_deflection', 0.0, units='m', desc='Deflection of tower top in yaw-aligned +x direction')
        self.add_output('stress', np.zeros(nFull), units='N/m**2', desc='Von Mises stress utilization along tower at specified locations.  incudes safety factor.')
        self.add_output('shell_buckling', np.zeros(nFull), desc='Shell buckling constraint.  Should be < 1 for feasibility.  Includes safety factors')
        self.add_output('global_buckling', np.zeros(nFull), desc='Global buckling constraint.  Should be < 1 for feasibility.  Includes safety factors')
        self.add_output('damage', np.zeros(nFull), desc='Fatigue damage at each tower section')
        self.add_output('weldability', np.zeros(nFull))
        self.add_output('manufacturability', np.zeros(nFull))

        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'


    def solve_nonlinear(self, params, unknowns, resids):

        # ------- node data ----------------
        z = params['z']
        n = len(z)
        node = np.arange(1, n+1)
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
        tempArray = np.array(params['kidx'])
        myones    = np.ones(tempArray.shape)
        rnode      = tempArray + np.int64(myones)  # add one because 0-based index but 1-based node numbering
        rigid     = np.inf
        kx = ky = kz = ktx = kty = ktz = rigid * myones

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
        Ixx = np.array([ params['mIxx'] ])
        Iyy = np.array([ params['mIyy'] ])
        Izz = np.array([ params['mIzz'] ])
        Ixy = np.array([ params['mIxy'] ])
        Iyz = np.array([ params['mIyz'] ])
        Ixz = np.array([ params['mIxz'] ])
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
        Fx = np.array([ params['Fx'] ])
        Fy = np.array([ params['Fy'] ])
        Fz = np.array([ params['Fz'] ])
        Mx = np.array([ params['Mxx'] ])
        My = np.array([ params['Myy'] ])
        Mz = np.array([ params['Mzz'] ])
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

        # mass
        unknowns['mass'] = mass.struct_mass

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

        #TODO weldability and manufacturability??
# -----------------
#  Assembly
# -----------------

class TowerSE(Group):

    def __init__(self, nPoints, nFull, nDEL, wind=''):

        super(TowerSE, self).__init__()

        self.add('geometry', TowerDiscretization(nPoints, nFull), promotes=['*'])
        if wind.lower() == 'powerwind':
            self.add('wind', PowerWind(nFull), promotes=['zref','z0'])
        elif wind.lower() == 'logwind':
            self.add('wind', LogWind(nFull), promotes=['zref','z0'])
        else:
            raise ValueError('Unknown wind type, '+wind)
        self.add('wave', WaveBase(nFull))
        self.add('windLoads', TowerWindDrag(nFull))
        self.add('waveLoads', TowerWaveDrag(nFull))
        self.add('distLoads', AeroHydroLoads(nFull), promotes=['yaw'])
        self.add('props', CylindricalShellProperties(nFull), promotes=['Az','Asx','Asy','Jz','Ixx','Iyy'])
        self.add('tower', TowerFrame3DD(nFull, nDEL), promotes=['E','G', 'rho','sigma_y','L_reinforced',
                                                                'kidx','m','mIxx','mIyy','mIzz','mIxy','mIxz','mIyz',
                                                                'mrho','gamma_f','gamma_m','gamma_n','gamma_b',
                                                                'life','m_SN','DC','z_DEL','M_DEL','gamma_fatigue',
                                                                'shear','geom','dx','nM','Mmethod','lump','tol',
                                                                'shift','Az','Asx','Asy','Jz','Ixx','Iyy'])
        self.add('gc', Util.GeometricConstraints(nPoints))

        # connections to gc
        self.connect('d_param', 'gc.d')
        self.connect('t_param', 'gc.t')

        # connections to wind1
        self.connect('z_full', 'wind.z')

        # connections to wave1 and wave2
        self.connect('z_full', 'wave.z')

        # connections to windLoads1
        self.connect('wind.U', 'windLoads.U')
        self.connect('z_full', 'windLoads.z')
        self.connect('d_full', 'windLoads.d')
        self.connect('wind.beta', 'windLoads.beta')

        # connections to waveLoads1
        self.connect('wave.U', 'waveLoads.U')
        self.connect('wave.A', 'waveLoads.A')
        self.connect('z_full', 'waveLoads.z')
        self.connect('d_full', 'waveLoads.d')
        self.connect('wave.beta', 'waveLoads.beta')

        # connections to distLoads1
        self.connect('z_full', 'distLoads.z')

        # connections to props
        self.connect('d_full', 'props.d')
        self.connect('t_full', 'props.t')

        # connect to tower1
        self.connect('z_full', 'tower.z')

        self.connect('d_full', 'tower.d')
        self.connect('t_full', 'tower.t')

        self.connect('distLoads.Px',   'tower.Px')
        self.connect('distLoads.Py',   'tower.Py')
        self.connect('distLoads.Pz',   'tower.Pz')
        self.connect('distLoads.qdyn', 'tower.qdyn')
        #self.connect('distLoads.outloads', 'tower.WWloads')

        self.connect('windLoads.windLoads:Px', 'distLoads.windLoads:Px')
        self.connect('windLoads.windLoads:Py', 'distLoads.windLoads:Py')
        self.connect('windLoads.windLoads:Pz', 'distLoads.windLoads:Pz')
        self.connect('windLoads.windLoads:qdyn', 'distLoads.windLoads:qdyn')
        self.connect('windLoads.windLoads:beta', 'distLoads.windLoads:beta')
        self.connect('windLoads.windLoads:Px0', 'distLoads.windLoads:Px0')
        self.connect('windLoads.windLoads:Py0', 'distLoads.windLoads:Py0')
        self.connect('windLoads.windLoads:Pz0', 'distLoads.windLoads:Pz0')
        self.connect('windLoads.windLoads:qdyn0', 'distLoads.windLoads:qdyn0')
        self.connect('windLoads.windLoads:beta0', 'distLoads.windLoads:beta0')
        self.connect('windLoads.windLoads:z', 'distLoads.windLoads:z')
        self.connect('windLoads.windLoads:d', 'distLoads.windLoads:d')

        self.connect('waveLoads.waveLoads:Px', 'distLoads.waveLoads:Px')
        self.connect('waveLoads.waveLoads:Py', 'distLoads.waveLoads:Py')
        self.connect('waveLoads.waveLoads:Pz', 'distLoads.waveLoads:Pz')
        self.connect('waveLoads.waveLoads:qdyn', 'distLoads.waveLoads:qdyn')
        self.connect('waveLoads.waveLoads:beta', 'distLoads.waveLoads:beta')
        self.connect('waveLoads.waveLoads:Px0', 'distLoads.waveLoads:Px0')
        self.connect('waveLoads.waveLoads:Py0', 'distLoads.waveLoads:Py0')
        self.connect('waveLoads.waveLoads:Pz0', 'distLoads.waveLoads:Pz0')
        self.connect('waveLoads.waveLoads:qdyn0', 'distLoads.waveLoads:qdyn0')
        self.connect('waveLoads.waveLoads:beta0', 'distLoads.waveLoads:beta0')
        self.connect('waveLoads.waveLoads:z', 'distLoads.waveLoads:z')
        self.connect('waveLoads.waveLoads:d', 'distLoads.waveLoads:d')


        # outputs TODO
        """
        self.connect('tower.mass', 'mass')
        self.connect('tower.f1', 'f1')
        self.connect('tower.top_deflection', 'top_deflection1')
        self.connect('tower.stress', 'stress1')
        self.connect('tower.global_buckling', 'global_buckling1')
        self.connect('tower.shell_buckling', 'shell_buckling1')
        self.connect('tower.damage', 'damage')
        self.connect('gc.weldability', 'weldability')
        self.connect('gc.manufacturability', 'manufacturability')
        """

        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'

        
if __name__ == '__main__':
    # --- tower setup ------
    from commonse.environment import PowerWind
    from commonse.environment import LogWind

    # --- geometry ----
    h_param = np.diff(np.array([0.0, 43.8, 87.6]))
    d_param = np.array([6.0, 4.935, 3.87])
    t_param = [0.027*1.3, 0.023*1.3, 0.019*1.3]
    n = 15
    L_reinforced = 30.0  # [m] buckling length
    theta_stress = 0.0
    yaw = 0.0

    # --- material props ---
    E = 210e9
    G = 80.8e9
    rho = 8500.0
    sigma_y = 450.0e6

    # --- spring reaction data.  Use float('inf') for rigid constraints. ---
    kidx = [0] # applied at base

    # --- extra mass ----
    m = np.array([285598.8])
    mIxx = np.array([1.14930678e+08])
    mIyy = np.array([2.20354030e+07])
    mIzz = np.array([1.87597425e+07])
    mIxy = np.array([0.00000000e+00])
    mIxz = np.array([5.03710467e+05])
    mIyz = np.array([0.00000000e+00])
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

    prob = Problem(root=TowerSE(nPoints, nFull, nDEL, wind=wind))
    prob.setup()

    if wind=='PowerWind':
        prob['wind.shearExp'] = shearExp

    # assign values to params

    # --- geometry ----
    prob['h_param'] = h_param
    prob['d_param'] = d_param
    prob['t_param'] = t_param
    prob['L_reinforced'] = L_reinforced
    prob['yaw'] = yaw

    # --- material props ---
    prob['E'] = E
    prob['G'] = G
    prob['rho'] = rho
    prob['sigma_y'] = sigma_y

    # --- spring reaction data.  Use float('inf') for rigid constraints. ---
    prob['kidx'] = kidx

    # --- extra mass ----
    prob['m'] = m
    prob['mIxx'] = mIxx
    prob['mIyy'] = mIyy
    prob['mIzz'] = mIzz
    prob['mIxy'] = mIxy
    prob['mIxz'] = mIxz
    prob['mIyz'] = mIyz
    prob['mrho'] = mrho
    # -----------

    # --- wind ---
    prob['zref'] = wind_zref
    prob['z0'] = wind_z0
    # ---------------

    # # --- loading case 1: max Thrust ---
    prob['wind.Uref'] = wind_Uref1

    prob['tower.Fx'] = Fx1
    prob['tower.Fy'] = Fy1
    prob['tower.Fz'] = Fz1
    prob['tower.Mxx'] = Mxx1
    prob['tower.Myy'] = Myy1
    prob['tower.Mzz'] = Mzz1
    # # ---------------

    # --- safety factors ---
    prob['gamma_f'] = gamma_f
    prob['gamma_m'] = gamma_m
    prob['gamma_n'] = gamma_n
    prob['gamma_b'] = gamma_b
    prob['gamma_fatigue'] = gamma_fatigue
    # ---------------

    # --- fatigue ---
    prob['z_DEL'] = z_DEL
    prob['M_DEL'] = M_DEL
    prob['life'] = life
    prob['m_SN'] = m_SN
    # ---------------

    # --- constraints ---
    prob['gc.min_d_to_t'] = min_d_to_t
    prob['gc.min_taper'] = min_taper
    # ---------------


    # # --- run ---
    prob.run()

    z = prob['z_full']

    print 'mass (kg) =', prob['tower.mass']
    print 'f1 (Hz) =', prob['tower.f1']
    print 'top_deflection1 (m) =', prob['tower.top_deflection']
    print 'weldability =', prob['gc.weldability']
    print 'manufacturability =', prob['gc.manufacturability']
    print 'stress1 =', prob['tower.stress']
    print 'zs=', z
    print 'ds=', prob['d_full']
    print 'ts=', prob['t_full']
    print 'GL buckling =', prob['tower.global_buckling']
    print 'Shell buckling =', prob['tower.shell_buckling']
    print 'damage =', prob['tower.damage']

    print 'wind: ', prob['wind.Uref']

    stress1 = np.copy( prob['tower.stress'] )
    shellBuckle1 = np.copy( prob['tower.shell_buckling'] )
    globalBuckle1 = np.copy( prob['tower.global_buckling'] )
    damage1 = np.copy( prob['tower.damage'] )



    # # --- loading case 2: max Wind Speed ---
    prob['wind.Uref'] = wind_Uref2

    prob['tower.Fx'] = Fx2
    prob['tower.Fy'] = Fy2
    prob['tower.Fz'] = Fz2
    prob['tower.Mxx'] = Mxx2
    prob['tower.Myy'] = Myy2
    prob['tower.Mzz'] = Mzz2

    prob.run()

    print 'mass (kg) =', prob['tower.mass']
    print 'f2 (Hz) =', prob['tower.f2']
    print 'top_deflection1 (m) =', prob['tower.top_deflection']
    print 'weldability =', prob['gc.weldability']
    print 'manufacturability =', prob['gc.manufacturability']
    print 'stress =', prob['tower.stress']
    print 'zs=', z
    print 'ds=', prob['d_full']
    print 'ts=', prob['t_full']
    print 'GL buckling =', prob['tower.global_buckling']
    print 'Shell buckling =', prob['tower.shell_buckling']
    print 'damage =', prob['tower.damage']

    print 'wind: ', prob['wind.Uref']

    stress2 = prob['tower.stress']
    shellBuckle2 = prob['tower.shell_buckling']
    globalBuckle2 = prob['tower.global_buckling']
    damage2 = prob['tower.damage']

    
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



    
    
