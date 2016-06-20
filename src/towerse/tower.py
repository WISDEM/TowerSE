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
from openmdao.api import Component, Group, Problem

#from commonse.utilities import linspace_with_deriv, interp_with_deriv, hstack, vstack

#from commonse.WindWaveDrag import FluidLoads, AeroHydroLoads, TowerWindDrag, TowerWaveDrag
from commonse.WindWaveDrag import AeroHydroLoads, TowerWindDrag, TowerWaveDrag #TODO FluidLoads isn't used anywhere (commented out in the code)

from commonse.environment import WindBase, WaveBase  # , SoilBase

from commonse import Tube

#from fusedwind.turbine.tower import TowerFromCSProps
#from fusedwind.interface import implement_base

from commonse.UtilizationSupplement import fatigue, hoopStressEurocode, shellBucklingEurocode, \
    bucklingGL, vonMisesStressUtilization

import frame3dd



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

         # variables
        self.add_param('z_param', np.zeros(nPoints), units='m', desc='parameterized locations along tower, linear lofting between')
        self.add_param('d_param', np.zeros(nPoints), units='m', desc='tower diameter at corresponding locations')
        self.add_param('t_param', np.zeros(nPoints), units='m', desc='shell thickness at corresponding locations')
        self.add_param('z_full', np.zeros(nFull), units='m', desc='locations along tower')

        #out
        self.add_output('d_full', np.zeros(nFull), units='m', desc='tower diameter at corresponding locations')
        self.add_output('t_full', np.zeros(nFull), units='m', desc='shell thickness at corresponding locations')
        

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['d_full'] = np.interp(params['z_full'], params['z_param'], params['d_param'])
        unknowns['t_full'] = np.interp(params['z_full'], params['z_param'], params['t_param'])


class GeometricConstraints(Component):
    """docstring for OtherConstraints"""

    def __init__(self, nPoints):

        super(GeometricConstraints, self).__init__()
        self.add_param('d', np.zeros(nPoints))
        self.add_param('t', np.zeros(nPoints))
        self.add_param('min_d_to_t', 120.0)
        self.add_param('min_taper', 0.4)

        self.add_output('weldability', np.zeros(nPoints))
        self.add_output('manufacturability', np.zeros(nPoints))


    def solve_nonlinear(self, params, unknowns, resids):

        d = params['d']
        t = params['t']
        min_d_to_t = params['min_d_to_t']
        min_taper = params['min_taper']

        unknowns['weldability'] = (min_d_to_t-d/t)/min_d_to_t
        manufacturability = min_taper-d[1:]/d[:-1] #taper ration
        unknowns['manufacturabilty'] = np.hstack((manufacturability, manufacturability[-1]))

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



#@implement_base(TowerFromCSProps) #TODO What is this??
class TowerFrame3DD(Component):
    
    def __init__(self, nFull):
    
        super(TowerFrame3DD, self).__init__()
        # cross-sectional data along tower.
        self.add_param('z', np.zeros(nFull), units='m', desc='location along tower. start at bottom and go to top')
        self.add_param('Az', np.zeros(nFull), units='m**2', desc='cross-sectional area')
        self.add_param('Asx', np.zeros(nFull), units='m**2', desc='x shear area')
        self.add_param('Asy', np.zeros(nFull), units='m**2', desc='y shear area')
        self.add_param('Jz', np.zeros(nFull), units='m**4', desc='polar moment of inertia')
        self.add_param('Ixx', np.zeros(nFull), units='m**4', desc='area moment of inertia about x-axis')
        self.add_param('Iyy', np.zeros(nFull), units='m**4', desc='area moment of inertia about y-axis')

        self.add_param('E', np.zeros(nFull), units='N/m**2', desc='modulus of elasticity')
        self.add_param('G', np.zeros(nFull), units='N/m**2', desc='shear modulus')
        self.add_param('rho', np.zeros(nFull), units='kg/m**3', desc='material density')
        self.add_param('sigma_y', np.zeros(nFull), units='N/m**2', desc='yield stress')

        # effective geometry -- used for handbook methods to estimate hoop stress, buckling, fatigue
        # length should be one less than z
        self.add_param('d', np.zeros(nFull), units='m', desc='effective tower diameter for section')
        self.add_param('t', np.zeros(nFull), units='m', desc='effective shell thickness for section')
        self.add_param('L_reinforced', np.zeros(nFull), units='m') #TODO should these three be of length (nFull-1)?

    # locations where stress should be evaluated
    #TODO LOOKS LIKE THIS ISN'T USED IN ANY CALCULATION? 
    #theta_stress = Array(iotype='in', units='deg', desc='location along azimuth where stress should be evaluated.  0 corresponds to +x axis.  follows unit circle direction and c.s.')

        # spring reaction data.  Use float('inf') for rigid constraints.
        self.add_param('kidx', np.zeros(nFull), desc='indices of z where external stiffness reactions should be applied.')
        self.add_param('kx', np.zeros(nFull), units='m', desc='spring stiffness in x-direction')
        self.add_param('ky', np.zeros(nFull), units='m', desc='spring stiffness in y-direction')
        self.add_param('kz', np.zeros(nFull), units='m', desc='spring stiffness in z-direction')
        self.add_param('ktx', np.zeros(nFull), units='m', desc='spring stiffness in theta_x-rotation')
        self.add_param('kty', np.zeros(nFull), units='m', desc='spring stiffness in theta_y-rotation')
        self.add_param('ktz', np.zeros(nFull), units='m', desc='spring stiffness in theta_z-rotation')
 
        # extra mass
        self.add_param('midx', np.zeros(nFull), desc='indices where added mass should be applied.')
        self.add_param('m', np.zeros(nFull), units='kg', desc='added mass')
        self.add_param('mIxx', np.zeros(nFull), units='kg*m**2', desc='x mass moment of inertia about some point p')
        self.add_param('mIyy', np.zeros(nFull), units='kg*m**2', desc='y mass moment of inertia about some point p')
        self.add_param('mIzz', np.zeros(nFull), units='kg*m**2', desc='z mass moment of inertia about some point p')
        self.add_param('mIxy', np.zeros(nFull), units='kg*m**2', desc='xy mass moment of inertia about some point p')
        self.add_param('mIxz', np.zeros(nFull), units='kg*m**2', desc='xz mass moment of inertia about some point p')
        self.add_param('mIyz', np.zeros(nFull), units='kg*m**2', desc='yz mass moment of inertia about some point p')
        self.add_param('mrhox', np.zeros(nFull), units='m', desc='x-location of p relative to node')
        self.add_param('mrhoy', np.zeros(nFull), units='m', desc='y-location of p relative to node')
        self.add_param('mrhoz', np.zeros(nFull), units='m', desc='z-location of p relative to node')
        self.add_param('addGravityLoadForExtraMass', True, desc='add gravitational load')


        # gravitational load
        self.add_param('g', 9.81, units='m/s**2', desc='acceleration of gravity (magnitude)')

        # point loads (if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also)
        self.add_param('plidx', np.zeros(nFull), desc='indices where point loads should be applied.')
        self.add_param('Fx', np.zeros(nFull), units='N', desc='point force in x-direction')
        self.add_param('Fy', np.zeros(nFull), units='N', desc='point force in y-direction')
        self.add_param('Fz', np.zeros(nFull), units='N', desc='point force in z-direction')
        self.add_param('Mxx', np.zeros(nFull), units='N*m', desc='point moment about x-axis')
        self.add_param('Myy', np.zeros(nFull), units='N*m', desc='point moment about y-axis')
        self.add_param('Mzz', np.zeros(nFull), units='N*m', desc='point moment about z-axis')

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
        self.add_param('z_DEL', np.zeros(nFull))
        self.add_param('M_DEL', np.zeros(nFull))

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

    def solve_nonlinear(self, params, unknowns, resids):

        # ------- node data ----------------
        z = params['z']
        n = len(z)
        node = np.arange(1, n+1)
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)

        nodes = frame3dd.NodeData(node, x, y, z, r)
        # -----------------------------------

        # ------ reaction data ------------

        # rigid base
        node = self.kidx + 1  # add one because 0-based index but 1-based node numbering
        rigid = float('inf')

        reactions = frame3dd.ReactionData(node, params['kx'], params['ky'], params['kz'], params['ktx'], params['kty'], params['ktz'], rigid)
        # -----------------------------------

        # ------ frame element data ------------
        element = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n+1)

        roll = np.zeros(n-1)

        # average across element b.c. frame3dd uses constant section elements
        Az = 0.5*(params['Az'][:-1] + params['Az'][1:])
        Asx = 0.5*(params['Asx'][:-1] + params['Asx'][1:])
        Asy = 0.5*(params['Asy'][:-1] + params['Asy'][1:])
        Jz = 0.5*(params['Jz'][:-1] + params['Jz'][1:])
        Ixx = 0.5*(params['Ixx'][:-1] + params['Ixx'][1:])
        Iyy = 0.5*(params['Iyy'][:-1] + params['Iyy'][1:])
        E = 0.5*(params['E'][:-1] + params['E'][1:])
        G = 0.5*(params['G'][:-1] + params['G'][1:])
        rho = 0.5*(params['rho'][:-1] + params['rho'][1:])

        elements = frame3dd.ElementData(element, N1, N2, Az, Asx, Asy, Jz,
            Ixx, Iyy, E, G, roll, rho)
        # -----------------------------------


        # ------ options ------------
        options = frame3dd.Options(params['shear'], params['geom'], params['dx'])
        # -----------------------------------

        # initialize frame3dd object
        tower = frame3dd.Frame(nodes, reactions, elements, options)


        # ------ add extra mass ------------

        # extra node inertia data
        N = params['midx'] + 1

        tower.changeExtraNodeMass(N, params['m'], params['mIxx'], params['mIyy'], params['mIzz'], params['mIxy'], params['mIxz'], params['mIyz'],
            params['mrhox'], params['mrhoy'], params['mrhoz'], params['addGravityLoadForExtraMass'])

        # ------------------------------------

        # ------- enable dynamic analysis ----------
        tower.enableDynamics(params['nM'], params['Mmethod'], params['lump'], params['tol'], params['shift'])
        # ----------------------------

        # ------ static load case 1 ------------

        # gravity in the X, Y, Z, directions (global)
        gx = 0.0
        gy = 0.0
        gz = -params['g']

        load = frame3dd.StaticLoadCase(gx, gy, gz)


        # point loads
        nF = params['plidx'] + 1
        load.changePointLoads(nF, params['Fx'], params['Fy'], params['Fz'], params['Mxx'], params['Myy'], params['Mzz'])


        # distributed loads
        Px, Py, Pz = params['Pz'], params['Py'], -params['Px']  # switch to local c.s.
        z = params['z'] #TODO wasn't this already defined above?

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
        shear_stress = 2. * np.sqrt(Vx**2+Vy**2) / sparams['Az'] # coefficient of 2 for a hollow circular section, but should be conservative for other shapes

        # hoop_stress (Eurocode method)
        hoop_stress = hoopStressEurocode(params['z'], params['d'], params['t'], params['L_reinforced'], params['qdyn'])

        # von mises stress
        unknowns['stress'] = vonMisesStressUtilization(axial_stress, hoop_stress, shear_stress,
                      params['gamma_f']*params['gamma_m']*params['gamma_n'], params['sigma_y'])

        # shell buckling
        unknowns['shell_buckling'] = shellBucklingEurocode(params['d'], params['t'], axial_stress, hoop_stress,
                      shear_stress, params['L_reinforced'], params['E'], params['sigma_y'], params['gamma_f'], params['gamma_b'])

        # global buckling
        tower_height = params['z'][-1] - params['z'][0]
        M = np.sqrt(Mxx**2 + Myy**2)
        unknowns['global_buckling'] = bucklingGL(params['d'], params['t'], Fz, M, tower_height, params['E'],
            params['sigma_y'], params['gamma_f'], params['gamma_b'])

        # fatigue
        N_DEL = [365*24*3600*params['life']]*len(z)
        unknowns['damage']=np.zeros(z.size)
        if any(params['M_DEL']):
            M_DEL = np.interp(z, params['z_DEL'], params['M_DEL'])

            unknowns['damage'] = fatigue(M_DEL, N_DEL, params['d'], params['t'], params['m_SN'], params['DC'], params['gamma_fatigue'], stress_factor=1.0, weld_factor=True)

        # TODO: more hack NOT SURE WHAT THIS IS, but it was there originally, commented out for now
#        damage = np.concatenate([np.zeros(len(self.z)-len(z)), damage])

        #TODO weldability and manufacturability??
# -----------------
#  Assembly
# -----------------

class TowerSE(Group):

    def __init__(self, nPoints, nFull):

        super(TowerSE, self).__init__()
        
        self.fd_options['force_fd'] = True        

        """
        # geometry parameters
        self.add_param('z_param', np.zeros(nPoints), units='m', desc='parameterized locations along tower, linear lofting between')
        self.add_param('d_param', np.zeros(nPoints), units='m', desc='tower diameter at corresponding locations')
        self.add_param('t_param', np.zeros(nPoints), units='m', desc='shell thickness at corresponding locations')

        # geometry
        self.add_param('z_full', np.zeros(nFull), units='m', desc='locations along tower')
        self.add_param('L_reinforced', np.zeros(nPoints-1), units='m')
       

        # wind/wave
        self.add_param('wind_rho', 1.225, units='kg/m**3', desc='air density')
        self.add_param('wind_mu', 1.7934e-5, units='kg/(m*s)', desc='dynamic viscosity of air')
        self.add_param('wind_Uref1', 0.0, units='m/s', desc='reference wind speed (usually at hub height)')
        self.add_param('wind_Uref2', 0.0, units='m/s', desc='reference wind speed (usually at hub height)')
        self.add_param('wind_zref', 80, units='m', desc='corresponding reference height')
        self.add_param('wind_z0', 0.0, units='m', desc='bottom of wind profile (height of ground/sea)')
        #TODO what to do here?
        self.add_param('wind_cd', 0.0, desc='Cd coefficient (it will be applied at all stations), if left blank it will be calculated based on cylinder Re')

        self.add_param('wave_rho', 1027.0, units='kg/m**3', desc='water density')
        self.add_param('wave_mu', 1.3351e-3, units='kg/(m*s)', desc='dynamic viscosity of water')
        self.add_param('wave_cm', 2.0, desc='mass coefficient')
        #TODO what to do here?
        self.add_param('wave_cd', 0.0, desc='Cd coefficient (it will be applied at all stations), if left blank it will be calculated based on cylinder Re')

        self.add_param('yaw',0.0, units='deg', desc='yaw angle')

        # material props
        self.add_param('E', np.zeros(nPoints), units='N/m**2', desc='modulus of elasticity')
        self.add_param('G', np.zeros(nPoints), units='N/m**2', desc='shear modulus')
        self.add_param('rho', np.zeros(nPoints), units='kg/m**3', desc='material density')
        self.add_param('sigma_y', np.zeros(nPoints), units='N/m**2', desc='yield stress')

        # spring reaction data.  Use float('inf') for rigid constraints.
        self.add_param('kidx', np.zeros(nPoints), desc='indices of z where external stiffness reactions should be applied.')
        self.add_param('kx', np.zeros(nPoints), units='m', desc='spring stiffness in x-direction')
        self.add_param('ky', np.zeros(nPoints), units='m', desc='spring stiffness in y-direction')
        self.add_param('kz', np.zeros(nPoints), units='m', desc='spring stiffness in z-direction')
        self.add_param('ktx', np.zeros(nPoints), units='m', desc='spring stiffness in theta_x-rotation')
        self.add_param('kty', np.zeros(nPoints), units='m', desc='spring stiffness in theta_y-rotation')
        self.add_param('ktz', np.zeros(nPoints), units='m', desc='spring stiffness in theta_z-rotation')

        # extra mass
        self.add_param('midx', np.zeros(nPoints), desc='indices where added mass should be applied.')
        self.add_param('m', np.zeros(nPoints), units='kg', desc='added mass')
        self.add_param('mIxx', units='kg*m**2', desc='x mass moment of inertia about some point p')
        self.add_param('mIyy', units='kg*m**2', desc='y mass moment of inertia about some point p')
        self.add_param('mIzz', units='kg*m**2', desc='z mass moment of inertia about some point p')
        self.add_param('mIxy', units='kg*m**2', desc='xy mass moment of inertia about some point p')
        self.add_param('mIxz', units='kg*m**2', desc='xz mass moment of inertia about some point p')
        self.add_param('mIyz', units='kg*m**2', desc='yz mass moment of inertia about some point p')
        self.add_param('mrhox', units='m', desc='x-location of p relative to node')
        self.add_param('mrhoy', units='m', desc='y-location of p relative to node')
        self.add_param('mrhoz', units='m', desc='z-location of p relative to node')
        self.add_param('addGravityLoadForExtraMass', True, desc='add gravitational load')

        # gravitational load
        self.add_param('g', 9.81, units='m/s**2', desc='acceleration of gravity (magnitude)')

        # safety factors
        self.add_param('gamma_f', 1.35, desc='safety factor on loads')
        self.add_param('gamma_m', 1.1, desc='safety factor on materials')
        self.add_param('gamma_n', 1.0, desc='safety factor on consequence of failure')
        self.add_param('gamma_b', 1.1, desc='buckling safety factor')
        self.add_param('gamma_fatigue', 1.755, desc='total safety factor for fatigue')

        # replace
        wind1 = Slot(WindBase)
        wind2 = Slot(WindBase)
        wave1 = Slot(WaveBase)
        wave2 = Slot(WaveBase)

        # point loads (if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also)
        self.add_param('plidx1', np.zeros(nPoints), desc='indices where point loads should be applied.')
        self.add_param('Fx1', np.zeros(nPoints), units='N', desc='point force in x-direction')
        self.add_param('Fy1', np.zeros(nPoints), units='N', desc='point force in y-direction')
        self.add_param('Fz1', np.zeros(nPoints), units='N', desc='point force in z-direction')
        self.add_param('Mxx1', np.zeros(nPoints), units='N*m', desc='point moment about x-axis')
        self.add_param('Myy1', np.zeros(nPoints), units='N*m', desc='point moment about y-axis')
        self.add_param('Mzz1', np.zeros(nPoints), units='N*m', desc='point moment about z-axis')

        self.add_param('plidx2', np.zeros(nPoints), desc='indices where point loads should be applied.')
        self.add_param('Fx2', np.zeros(nPoints), units='N', desc='point force in x-direction')
        self.add_param('Fy2', np.zeros(nPoints), units='N', desc='point force in y-direction')
        self.add_param('Fz2', np.zeros(nPoints), units='N', desc='point force in z-direction')
        self.add_param('Mxx2', np.zeros(nPoints), units='N*m', desc='point moment about x-axis')
        self.add_param('Myy2', np.zeros(nPoints), units='N*m', desc='point moment about y-axis')
        self.add_param('Mzz2', np.zeros(nPoints), units='N*m', desc='point moment about z-axis')

        # constraint parameters
        self.add_param('min_d_to_t', 120.0)
        self.add_param('min_taper', 0.4)

        # fatigue parameters
        self.add_param('life', 20.0, desc='fatigue life of tower')
        self.add_param('m_SN', 4, desc='slope of S/N curve')
        self.add_param('DC', 80.0, desc='standard value of stress')
        self.add_param('z_DEL', np.zeros(nPoints))
        self.add_param('M_DEL', np.zeros(nPoints))

        # frame3ddd options
        self.add_param('shear', True, desc='include shear deformation')
        self.add_param('geom', False, desc='include geometric stiffness')
        self.add_param('dx', 5.0, desc='z-axis increment for internal forces')
        self.add_param('nM', 2, desc='number of desired dynamic modes of vibration (below only necessary if nM > 0)')
        self.add_param('Mmethod', 1, desc='1: subspace Jacobi, 2: Stodola')
        self.add_param('lump', 0, desc='0: consistent mass, 1: lumped mass matrix')
        self.add_param('tol', 1e-9, desc='mode shape tolerance')
        self.add_param('shift', 0.0, desc='shift value ... for unrestrained structures')

        # outputs
        self.add_output('mass', 0.0, units='kg')
        self.add_output('f1', 0.0, units='Hz', desc='First natural frequency')
        self.add_output('f2', 0.0, units='Hz', desc='Second natural frequency')
        self.add_output('top_deflection1', 0.0, units='m', desc='Deflection of tower top in yaw-aligned +x direction')
        self.add_output('top_deflection2', 0.0, units='m', desc='Deflection of tower top in yaw-aligned +x direction')
        self.add_output('stress1', np.zeros(nPoints), units='N/m**2', desc='Von Mises stress along tower on downwind side (yaw-aligned +x).  Normalized by yield stress.  Includes safety factors.')
        self.add_output('stress2', np.zeros(nPoints), units='N/m**2', desc='Von Mises stress along tower on downwind side (yaw-aligned +x).  Normalized by yield stress.  Includes safety factors.')
        self.add_output('shell_buckling1', np.zeros(nPoints), desc='Shell buckling constraint load case #1.  Should be < 1 for feasibility.  Includes safety factors')
        self.add_output('shell_buckling2', np.zeros(nPoints), desc='Shell buckling constraint load case #2.  Should be < 1 for feasibility.  Includes safety factors')
        self.add_output('global_buckling1', np.zeros(nPoints), desc='Global buckling constraint load case #1.  Should be < 1 for feasibility.  Includes safety factors')
        self.add_output('global_buckling2', np.zeros(nPoints), desc='Global buckling constraint load case #2.  Should be < 1 for feasibility.  Includes safety factors')
        self.add_output('damage', np.zeros(nPoints), desc='Fatigue damage at each tower section')
        self.add_output('weldability', np.zeros(nPoints))
        self.add_output('manufacturability', np.zeros(nPoints))
        """


        self.add('geometry', TowerDiscretization(nPoints, nFull), promotes=['*'])
        # two load cases.  TODO: use a case iterator
        self.add('wind1', WindBase(nFull))
        self.add('wind2', WindBase(nFull))
        self.add('wave1', WaveBase(nFull))
        self.add('wave2', WaveBase(nFull))
        self.add('windLoads1', TowerWindDrag(nFull))
        self.add('windLoads2', TowerWindDrag(nFull))
        self.add('waveLoads1', TowerWaveDrag(nFull))
        self.add('waveLoads2', TowerWaveDrag(nFull))
        self.add('distLoads1', AeroHydroLoads(nFull))
        self.add('distLoads2', AeroHydroLoads(nFull))
        self.add('props', CylindricalShellProperties(nFull))
        self.add('tower1', TowerFrame3DD(nFull))
        self.add('tower2', TowerFrame3DD(nFull))
        self.add('gc', GeometricConstraints(nPoints))
    
        
        # connections to geometry
        """
        z_param
        d_param
        t_param
        z_full
        d_full
        t_full
        """
        # promotes = '*'
        """
        self.connect('z_param', 'geometry.z_param')
        self.connect('d_param', 'geometry.d_param')
        self.connect('t_param', 'geometry.t_param')
        self.connect('z_full', 'geometry.z_full')
        """
        # connections to wind1
        self.connect('z_full', 'wind1.z')
        # TODO self.connect('wind_Uref1', 'wind1.Uref')
        # TODO self.connect('wind_zref', 'wind1.zref')
        # TODO self.connect('wind_z0', 'wind1.z0')
     
        # connections to wind2
        self.connect('z_full', 'wind2.z')
        # TODO self.connect('wind_Uref2', 'wind2.Uref')
        # TODO self.connect('wind_zref', 'wind2.zref')
        # TODO self.connect('wind_z0', 'wind2.z0')

        # connections to wave1 and wave2
        self.connect('z_full', 'wave1.z')
        self.connect('z_full', 'wave2.z')

        # connections to windLoads1
        self.connect('wind1.U', 'windLoads1.U')
        self.connect('z_full', 'windLoads1.z')
        self.connect('d_full', 'windLoads1.d')
        self.connect('wind1.beta', 'windLoads1.beta')

        # connections to windLoads2
        self.connect('wind2.U', 'windLoads2.U')
        self.connect('z_full', 'windLoads2.z')
        self.connect('d_full', 'windLoads2.d')
        self.connect('wind2.beta', 'windLoads2.beta')

        # connect windLoads
        self.connect('windLoads1.rho', 'windLoads2.rho')
        self.connect('windLoads1.mu', 'windLoads2.mu')
        self.connect('windLoads1.cd_usr', 'windLoads2.cd_usr')

        # connections to waveLoads1
        self.connect('wave1.U', 'waveLoads1.U')
        self.connect('wave1.A', 'waveLoads1.A')
        self.connect('z_full', 'waveLoads1.z')
        self.connect('d_full', 'waveLoads1.d')
        self.connect('wave1.beta', 'waveLoads1.beta')

        # connections to waveLoads2
        self.connect('wave2.U', 'waveLoads2.U')
        self.connect('wave2.A', 'waveLoads2.A')
        self.connect('z_full', 'waveLoads2.z')
        self.connect('d_full', 'waveLoads2.d')
        self.connect('wave2.beta', 'waveLoads2.beta')

        # connect waveLoads
        self.connect('waveLoads1.rho', 'waveLoads2.rho')
        self.connect('waveLoads1.mu', 'waveLoads2.mu')
        self.connect('waveLoads1.cm', 'waveLoads2.cm')
        self.connect('waveLoads1.cd_usr', 'waveLoads2.cd_usr')

        # connections to distLoads1
        self.connect('windLoads1.windLoads:Px', 'distLoads1.windLoads:Px')
        self.connect('windLoads1.windLoads:Py', 'distLoads1.windLoads:Py')
        self.connect('windLoads1.windLoads:Pz', 'distLoads1.windLoads:Pz')
        self.connect('windLoads1.windLoads:qdyn', 'distLoads1.windLoads:qdyn')
        self.connect('windLoads1.windLoads:z', 'distLoads1.windLoads:z')
        self.connect('windLoads1.windLoads:d', 'distLoads1.windLoads:d')
        self.connect('windLoads1.windLoads:beta', 'distLoads1.windLoads:beta')
        self.connect('windLoads1.windLoads:Px0', 'distLoads1.windLoads:Px0')
        self.connect('windLoads1.windLoads:Py0', 'distLoads1.windLoads:Py0')
        self.connect('windLoads1.windLoads:Pz0', 'distLoads1.windLoads:Pz0')
        self.connect('windLoads1.windLoads:qdyn0', 'distLoads1.windLoads:qdyn0')
        self.connect('windLoads1.windLoads:beta0', 'distLoads1.windLoads:beta0')

        self.connect('waveLoads1.waveLoads:Px', 'distLoads1.waveLoads:Px')
        self.connect('waveLoads1.waveLoads:Py', 'distLoads1.waveLoads:Py')
        self.connect('waveLoads1.waveLoads:Pz', 'distLoads1.waveLoads:Pz')
        self.connect('waveLoads1.waveLoads:qdyn', 'distLoads1.waveLoads:qdyn')
        self.connect('waveLoads1.waveLoads:z', 'distLoads1.waveLoads:z')
        self.connect('waveLoads1.waveLoads:d', 'distLoads1.waveLoads:d')
        self.connect('waveLoads1.waveLoads:beta', 'distLoads1.waveLoads:beta')
        self.connect('waveLoads1.waveLoads:Px0', 'distLoads1.waveLoads:Px0')
        self.connect('waveLoads1.waveLoads:Py0', 'distLoads1.waveLoads:Py0')
        self.connect('waveLoads1.waveLoads:Pz0', 'distLoads1.waveLoads:Pz0')
        self.connect('waveLoads1.waveLoads:qdyn0', 'distLoads1.waveLoads:qdyn0')
        self.connect('waveLoads1.waveLoads:beta0', 'distLoads1.waveLoads:beta0')
        self.connect('z_full', 'distLoads1.z')

        # connections to distLoads2
        self.connect('windLoads2.windLoads:Px', 'distLoads2.windLoads:Px')
        self.connect('windLoads2.windLoads:Py', 'distLoads2.windLoads:Py')
        self.connect('windLoads2.windLoads:Pz', 'distLoads2.windLoads:Pz')
        self.connect('windLoads2.windLoads:qdyn', 'distLoads2.windLoads:qdyn')
        self.connect('windLoads2.windLoads:z', 'distLoads2.windLoads:z')
        self.connect('windLoads2.windLoads:d', 'distLoads2.windLoads:d')
        self.connect('windLoads2.windLoads:beta', 'distLoads2.windLoads:beta')
        self.connect('windLoads2.windLoads:Px0', 'distLoads2.windLoads:Px0')
        self.connect('windLoads2.windLoads:Py0', 'distLoads2.windLoads:Py0')
        self.connect('windLoads2.windLoads:Pz0', 'distLoads2.windLoads:Pz0')
        self.connect('windLoads2.windLoads:qdyn0', 'distLoads2.windLoads:qdyn0')
        self.connect('windLoads2.windLoads:beta0', 'distLoads2.windLoads:beta0')

        self.connect('waveLoads2.waveLoads:Px', 'distLoads2.waveLoads:Px')
        self.connect('waveLoads2.waveLoads:Py', 'distLoads2.waveLoads:Py')
        self.connect('waveLoads2.waveLoads:Pz', 'distLoads2.waveLoads:Pz')
        self.connect('waveLoads2.waveLoads:qdyn', 'distLoads2.waveLoads:qdyn')
        self.connect('waveLoads2.waveLoads:z', 'distLoads2.waveLoads:z')
        self.connect('waveLoads2.waveLoads:d', 'distLoads2.waveLoads:d')
        self.connect('waveLoads2.waveLoads:beta', 'distLoads2.waveLoads:beta')
        self.connect('waveLoads2.waveLoads:Px0', 'distLoads2.waveLoads:Px0')
        self.connect('waveLoads2.waveLoads:Py0', 'distLoads2.waveLoads:Py0')
        self.connect('waveLoads2.waveLoads:Pz0', 'distLoads2.waveLoads:Pz0')
        self.connect('waveLoads2.waveLoads:qdyn0', 'distLoads2.waveLoads:qdyn0')
        self.connect('waveLoads2.waveLoads:beta0', 'distLoads2.waveLoads:beta0')
        self.connect('z_full', 'distLoads2.z')
        
        # connect yaw
        self.connect('distLoads1.yaw', 'distLoads2.yaw')

        # connections to props
        self.connect('d_full', 'props.d')
        self.connect('t_full', 'props.t')

        # connect to tower1
        self.connect('z_full', 'tower1.z')
        self.connect('props.Az', 'tower1.Az')
        self.connect('props.Asx', 'tower1.Asx')
        self.connect('props.Asy', 'tower1.Asy')
        self.connect('props.Jz', 'tower1.Jz')
        self.connect('props.Ixx', 'tower1.Ixx')
        self.connect('props.Iyy', 'tower1.Iyy')
      
        self.connect('d_full', 'tower1.d')
        self.connect('t_full', 'tower1.t')
        
        self.connect('distLoads1.Px',   'tower1.Px')
        self.connect('distLoads1.Py',   'tower1.Py')
        self.connect('distLoads1.Pz',   'tower1.Pz')
        self.connect('distLoads1.qdyn', 'tower1.qdyn')
        #self.connect('distLoads1.outloads', 'tower1.WWloads')

        # connect to tower2
        self.connect('z_full', 'tower2.z')
        self.connect('props.Az', 'tower2.Az')
        self.connect('props.Asx', 'tower2.Asx')
        self.connect('props.Asy', 'tower2.Asy')
        self.connect('props.Jz', 'tower2.Jz')
        self.connect('props.Ixx', 'tower2.Ixx')
        self.connect('props.Iyy', 'tower2.Iyy')
        
        self.connect('d_full', 'tower2.d')
        self.connect('t_full', 'tower2.t')
        
        self.connect('distLoads2.Px', 'tower2.Px')
        self.connect('distLoads2.Py', 'tower2.Py')
        self.connect('distLoads2.Pz', 'tower2.Pz')
        self.connect('distLoads2.qdyn', 'tower2.qdyn')
        #elf.connect('distLoads2.outloads', 'tower2.WWloads')

        # connect tower1 and tower2
        self.connect('tower1.E', 'tower2.E')
        self.connect('tower1.G', 'tower2.G')
        self.connect('tower1.rho', 'tower2.rho')
        self.connect('tower1.sigma_y', 'tower2.sigma_y')
        self.connect('tower1.L_reinforced', 'tower2.L_reinforced')
        # self.connect('tower1.theta_stress', 'tower2.theta_stress')
        self.connect('tower1.kidx', 'tower2.kidx')
        self.connect('tower1.kx', 'tower2.kx')
        self.connect('tower1.ky', 'tower2.ky')
        self.connect('tower1.kz', 'tower2.kz')
        self.connect('tower1.ktx', 'tower2.ktx')
        self.connect('tower1.kty', 'tower2.kty')
        self.connect('tower1.ktz', 'tower2.ktz')
        self.connect('tower1.midx', 'tower2.midx')
        self.connect('tower1.m', 'tower2.m')
        self.connect('tower1.mIxx', 'tower2.mIxx')
        self.connect('tower1.mIyy', 'tower2.mIyy')
        self.connect('tower1.mIzz', 'tower2.mIzz')
        self.connect('tower1.mIxy', 'tower2.mIxy')
        self.connect('tower1.mIxz', 'tower2.mIxz')
        self.connect('tower1.mIyz', 'tower2.mIyz')
        self.connect('tower1.mrhox', 'tower2.mrhox')
        self.connect('tower1.mrhoy', 'tower2.mrhoy')
        self.connect('tower1.mrhoz', 'tower2.mrhoz')
        self.connect('tower1.addGravityLoadForExtraMass', 'tower2.addGravityLoadForExtraMass')
        self.connect('tower1.g', 'tower2.g')
        self.connect('tower1.plidx', 'tower2.plidx')
        self.connect('tower1.Fx', 'tower2.Fx')
        self.connect('tower1.Fy', 'tower2.Fy')
        self.connect('tower1.Fz', 'tower2.Fz')
        self.connect('tower1.Mxx', 'tower2.Mxx')
        self.connect('tower1.Myy', 'tower2.Myy')
        self.connect('tower1.Mzz', 'tower2.Mzz')
        self.connect('tower1.gamma_f', 'tower2.gamma_f')
        self.connect('tower1.gamma_m', 'tower2.gamma_m')
        self.connect('tower1.gamma_n', 'tower2.gamma_n')
        self.connect('tower1.gamma_b', 'tower2.gamma_b')
        self.connect('tower1.life', 'tower2.life')
        self.connect('tower1.m_SN', 'tower2.m_SN')
        self.connect('tower1.DC', 'tower2.DC')
        self.connect('tower1.z_DEL', 'tower2.z_DEL')
        self.connect('tower1.M_DEL', 'tower2.M_DEL')
        self.connect('tower1.gamma_fatigue', 'tower2.gamma_fatigue')
        self.connect('tower1.shear', 'tower2.shear')
        self.connect('tower1.geom', 'tower2.geom')
        self.connect('tower1.dx', 'tower2.dx')
        self.connect('tower1.nM', 'tower2.nM')
        self.connect('tower1.Mmethod', 'tower2.Mmethod')
        self.connect('tower1.lump', 'tower2.lump')
        self.connect('tower1.tol', 'tower2.tol')
        self.connect('tower1.shift', 'tower2.shift')

        # connections to gc
        self.connect('d_param', 'gc.d')
        self.connect('t_param', 'gc.t')
        # TODO self.connect('min_d_to_t', 'gc.min_d_to_t')
        # TODO self.connect('min_taper', 'gc.min_taper')


        # outputs TODO
        """
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
        """

if __name__ == '__main__':
    # --- tower setup ------
    from commonse.environment import PowerWind

    # --- geometry ----
    z_param = [0.0, 43.8, 87.6]
    d_param = [6.0, 4.935, 3.87]
    t_param = [0.027*1.3, 0.023*1.3, 0.019*1.3]
    n = 15
    z_full = np.linspace(0.0, 87.6, n)
    L_reinforced = 30.0*np.ones(n)  # [m] buckling length
    theta_stress = 0.0*np.ones(n)
    yaw = 0.0

    # --- material props ---
    E = 210e9*np.ones(n)
    G = 80.8e9*np.ones(n)
    rho = 8500.0*np.ones(n)
    sigma_y = 450.0e6*np.ones(n)

    # --- spring reaction data.  Use float('inf') for rigid constraints. ---
    kidx = [0]  # applied at base
    kx = [float('inf')]
    ky = [float('inf')]
    kz = [float('inf')]
    ktx = [float('inf')]
    kty = [float('inf')]
    ktz = [float('inf')]

    # --- extra mass ----
    midx = [n-1]  # RNA mass at top
    m = [285598.8]
    mIxx = [1.14930678e+08]
    mIyy = [2.20354030e+07]
    mIzz = [1.87597425e+07]
    mIxy = [0.00000000e+00]
    mIxz = [5.03710467e+05]
    mIyz = [0.00000000e+00]
    mrhox = [-1.13197635]
    mrhoy = [0.]
    mrhoz = [0.50875268]
    addGravityLoadForExtraMass = True
    # -----------

    # --- wind ---
    wind_zref = 90.0
    wind_z0 = 0.0
    # ---------------

    # if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also
    # # --- loading case 1: max Thrust ---
    wind_Uref1 = 11.73732
    plidx1 = [n-1]  # at  top
    Fx1 = [1284744.19620519]
    Fy1 = [0.]
    Fz1 = [-2914124.84400512]
    Mxx1 = [3963732.76208099]
    Myy1 = [-2275104.79420872]
    Mzz1 = [-346781.68192839]
    # # ---------------

    # # --- loading case 2: max wind speed ---
    wind_Uref2 = 70.0
    plidx1 = [n-1]  # at  top
    Fx1 = [930198.60063279]
    Fy1 = [0.]
    Fz1 = [-2883106.12368949]
    Mxx1 = [-1683669.22411597]
    Myy1 = [-2522475.34625363]
    Mzz1 = [147301.97023764]
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

    nPoints = len(z_param)
    nFull = len(z_full)

    prob = Problem(root=TowerSE(nPoints, nFull))

    prob.setup()

    wind1.shearExp = 0.2
    wind2.shearExp = 0.2

    
    # ---- tower ------
    tower.replace('wind1', PowerWind())
    tower.replace('wind2', PowerWind())
    # onshore (no waves)

    


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

