#!/usr/bin/env python
# encoding: utf-8
"""
towerstruc.py

Created by Andrew Ning on 2012-01-20.
Copyright (c) NREL. All rights reserved.

HISTORY:  2012 created
          -7/2014:  Bugs found in the call to shellBucklingEurocode from towerwithFrame3DD. Fixed.
                    Also set_as_top added.
          -10/2014: Merged back with some changes Andrew did on his end.
          -12/2014: fixed some errors from the merge (redundant drag calc).  pep8 compliance.  removed several unneccesary variables and imports (including set_as_top)
To DO:   1.FIX treatment of L_reinorced between tower.py and tower_supplement.py, for now it works with an array with all same elements. \n
         2.Get a better handle of the RNA weight effect on the moment at the top, to be conservative if desired when upwind. For now that moment is ignored. \n
 """

import math
import sys
import numpy as np
from openmdao.main.api import VariableTree, Component, Assembly
from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Slot, Instance

from commonse.utilities import sind, cosd, linspace_with_deriv, interp_with_deriv, hstack, vstack
from commonse.csystem import DirectionVector
from commonse.environment import WindBase, WaveBase, SoilBase
from commonse.Material import Material
from commonse import Tube

from jacketse.VarTrees import Frame3DDaux

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
    d = Array(units='m', desc='corresponding diameters')

    beta = Array(units='deg', desc='wind/wave angle relative to inertia c.s.')
    Px0= Float(units='N/m', desc='Distributed load at z=0 MSL')
    Py0= Float(units='N/m', desc='Distributed load at z=0 MSL')
    Pz0= Float(units='N/m', desc='Distributed load at z=0 MSL')
    q0 = Float(units='N/m**2', desc='dynamic pressure at z=0 MSL')
    beta0 = Float(units='deg', desc='wind/wave angle relative to inertia c.s.')

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

    U0= Float(iotype='in', units='m/s', desc='magnitude of wave speed at z=0 MSL')
    A = Array(iotype='in', units='m/s**2', desc='magnitude of wave acceleration')
    A0= Float(iotype='in', units='m/s**2', desc='magnitude of wave acceleration at z=0 MSL')
    z = Array(iotype='in', units='m', desc='heights where wave speed was computed')
    d = Array(iotype='in', units='m', desc='corresponding diameter of cylinder section')

    # parameters

    wlevel = Float(iotype='in', units='m', desc='Water Level, to assess z w.r.t. MSL')
    beta = Array(iotype='in', units='deg', desc='corresponding wave angles relative to inertial coordinate system')
    beta0= Float(iotype='in', units='deg', desc='corresponding wave angles relative to inertial coordinate system at z=0 MSL')
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
        U0 = self.U0
        d = self.d
        zrel= self.z-self.wlevel
        mu = self.mu
        beta = self.beta
        beta0= self.beta0

        # dynamic pressure
        q = 0.5*rho*U**2
        q0= 0.5*rho*U0**2

        # Reynolds number and drag
        if self.cd_usr:
            cd = self.cd_usr*np.ones_like(self.d)
            Re = 1.0
            dcd_dRe = 0.0
        else:
            Re = rho*U*d/mu
            cd, dcd_dRe = cylinderDrag(Re)

        d = self.d
        mu = self.mu
        beta = self.beta

        # inertial and drag forces
        Fi = rho*self.cm*math.pi/4.0*d**2*self.A  # Morrison's equation
        Fd = q*cd*d
        Fp = Fi + Fd

        #FORCES [N/m] AT z=0 m
        idx0 = np.abs(zrel).argmin()  # closest index to z=0, used to find d at z=0
        d0 = d[idx0]  # initialize
        cd0 = cd[idx0]  # initialize
        if (zrel[idx0]<0.) and (idx0< (zrel.size-1)):       # point below water
            d0 = np.mean(d[idx0:idx0+2])
            cd0 = np.mean(cd[idx0:idx0+2])
        elif (zrel[idx0]>0.) and (idx0>0):     # point above water
            d0 = np.mean(d[idx0-1:idx0+1])
            cd0 = np.mean(cd[idx0-1:idx0+1])
        Fi0 = rho*self.cm*math.pi/4.0*d0**2*self.A0  # Morrison's equation
        Fd0 = q0*cd0*d0
        Fp0 = Fi0 + Fd0

        # components of distributed loads
        Px = Fp*cosd(beta)
        Py = Fp*sind(beta)
        Pz = 0.*Fp

        Px0 = Fp0*cosd(beta0)
        Py0 = Fp0*sind(beta0)
        Pz0 = 0.*Fp0

        #Store qties at z=0 MSL
        self.waveLoads.Px0 = Px0
        self.waveLoads.Py0 = Py0
        self.waveLoads.Pz0 = Pz0
        self.waveLoads.q0 = q0
        self.waveLoads.beta0 = beta0

        # pack data
        self.waveLoads.Px = Px
        self.waveLoads.Py = Py
        self.waveLoads.Pz = Pz
        self.waveLoads.q = q
        self.waveLoads.z = self.z
        self.waveLoads.beta = beta
        self.waveLoads.d = d


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

    #inputs

    # variables
    twrlength = Float(iotype='in', units='m', desc='Tower-only length.')
    SubLength = Float(0.0, iotype='in', units='m', desc='Substructure-only length.')

    z = Array(iotype='in', desc='Locations of tower-only stations along unit MP-tower, linear lofting between.')
    d = Array(iotype='in', units='m', desc='Tower-only diameters at corresponding locations.')
    t = Array(iotype='in', units='m', desc='Tower-only shell thickness at corresponding locations.')
    L_reinforced = Array(iotype='in', units='m',desc='Buckling lengths for the various segments of Tower-only. Length of array = ``len(z)``.')

    zsub = Array(iotype='in', desc='Locations along unit MP-tower of the substructure only, linear lofting between')
    dsub = Array(iotype='in', units='m', desc='Substructure-only outer diameters at corresponding locations')
    tsub = Array(iotype='in', units='m', desc='Substructure-only wall thickness at corresponding locations')
    subLreinforced = Array(iotype='in', units='m',desc='Buckling lengths for the various segments of substructure-only.')

    ##d_monopile = Float(iotype='in', units='m')
    ##t_monopile = Float(iotype='in', units='m')
    # parameters
    n    = Array(iotype='in', dtype=np.int, desc='Number of finite elements between tower-only sections.  Array length should be ``len(z)-1``')
    nsub = Array(iotype='in', dtype=np.int, desc='number of finite elements between substructure-only sections.  Array length should be ``len(zsub)`` to account for sub-to-twr conneciton as well.')
    ##nsub = Int(iotype='in', desc='must be a minimum of 1 (top and bottom)')


    # outputs
    ntot = Array(iotype='out', dtype=np.int, desc='number of finite elements for the entire structure sections.  Array length should be ``len(zsub)+len(z)-1``.')
    z_node = Array(iotype='out', units='m', desc='All Locations along MP-tower, linear lofting between')
    d_node = Array(iotype='out', units='m', desc='All outer diameters at corresponding locations')
    t_node = Array(iotype='out', units='m', desc='All wall  thicknesses at corresponding locations')

    L_reinforced_node = Array(iotype='out', units='m')

    idxtwr = Int(iotype='out',desc='Starting Index into z_node for actual tower (no TP or MP).')
    idxol0 = Int(iotype='out',desc='Starting Index into z_node for overlap start. It assumes thickness is  constnat in MP, and constant in TP.')
    idxol1 = Int(iotype='out',desc='Starting Index into z_node for overlap end (top of MP). It assumes thickness is  constnat in MP, and constant in TP.')

    zall = Array(iotype='out', desc='Locations of MP+tower stations along unit MP-tower, overlap is removed.')
    dall = Array(iotype='out', units='m', desc='MP+Tower diameters at all z locations, overlap is removed.')
    tall = Array(iotype='out', units='m', desc='MP+Tower wall thickness at at all z locations, overlap is removed.')



    missing_deriv_policy = 'assume_zero'


    def execute(self):
        #RRD
        MPtowerlgth = self.twrlength+self.SubLength
        self.z = (self.z*self.twrlength+self.SubLength)/MPtowerlgth
        self.idxtwr=0 #initialize
        self.idxol0=0 #initialize
        self.idxl1=0  #initialize
        self.ntot=self.n #initialize

        nz_sub= self.zsub.size

        if nz_sub: #Augment in case substructure is present
            self.z = np.concatenate([self.zsub,self.z]) #still dimensionless
            self.d = np.concatenate([self.dsub,self.d])
            self.t = np.concatenate([self.tsub,self.t])
            self.L_reinforced =np.concatenate([self.subLreinforced,self.L_reinforced])
            self.ntot =np.concatenate([self.nsub,self.n])

        n1 = sum(self.ntot) + 1
        n2 = len(self.z)
        self.dznode_dz = np.zeros((n1, n2))

        # compute nodal locations (and gradients)
        self.z_node = np.array([self.z[0]])

        nlast = 0
        for i in range(len(self.ntot)):
            znode, dznode_dzi, dznode_dzip = linspace_with_deriv(self.z[i], self.z[i+1], self.ntot[i]+1)
            self.z_node = np.r_[self.z_node, znode[1:]]

            # gradients
            self.dznode_dz[nlast:nlast+self.ntot[i]+1, i] = dznode_dzi
            self.dznode_dz[nlast:nlast+self.ntot[i]+1, i+1] = dznode_dzip
            nlast += self.ntot[i]


        # interpolate (and gradients)
        self.d_node = np.interp(self.z_node, self.z, self.d)
        self.t_node = np.interp(self.z_node, self.z, self.t)

        self.d_node, ddnode_dznode, ddnode_dz, self.ddnode_dd = interp_with_deriv(self.z_node, self.z, self.d)
        self.t_node, dtnode_dznode, dtnode_dz, self.dtnode_dt = interp_with_deriv(self.z_node, self.z, self.t)

       #also take care of some outputs
        self.idxtwr=(np.nonzero(self.z_node >= self.z[nz_sub])[0])[0]  #tower starts here  TO CHECK TO DO
        idxs=np.nonzero(np.roll(self.t_node,-1) > self.t_node)[0]
        self.idxol0=idxs[0]+1 #overlap starts here
        self.idxol1=self.idxtwr-1  #overlap ends
            #print 'from towerdiscretization idxol0=', self.idxol0 #debug

        # chain rule
        self.ddnode_dz = ddnode_dz + np.dot(ddnode_dznode, self.dznode_dz)
        self.dtnode_dz = dtnode_dz + np.dot(dtnode_dznode, self.dznode_dz)


        # reinforcement distances (assumed not to change during optimization---no gradients are derived for Lreinforced)
        self.L_reinforced_node = np.interp(self.z_node, self.z, self.L_reinforced)

        # self.z_reinforced, dzr_dz0, dzr_dzend = linspace_with_deriv(self.z[0], self.z[-1], self.n_reinforced+1)
        # self.dzr_dz = np.zeros((len(self.z_reinforced), n2))
        # self.dzr_dz[:, 0] = dzr_dz0
        # self.dzr_dz[:, -1] = dzr_dzend

        # make dimensional

        self.z_node *= MPtowerlgth
        self.dznode_dz *= MPtowerlgth
        # self.z_reinforced *= MPtowerlgth
        # self.dzr_dz *= MPtowerlgth

        self.z_node_save = self.z_node

            #Create new arrays to check the DTR and taperratio
        self.zall=np.copy(self.z)
        self.dall=np.copy(self.d)
        self.tall=np.copy(self.t)
        #Need to acount for the fake overlap member, so remove those stations for the time being
        nzall=len(self.zall)
        idx=np.array(range(0,2)+range(4,nzall))
        self.zall=self.zall[idx]
        self.dall=self.dall[idx]
        self.tall=self.tall[idx]

##        if self.monopileHeight > 0:
##            z_monopile = np.linspace(self.z[0] - self.monopileHeight, self.z[0], self.n_monopile)
            ##d_monopile = self.d_monopile * np.ones_like(z_monopile)
            ##t_monopile = self.t_monopile * np.ones_like(z_monopile)
            # L_monopile = self.L_reinforced * np.ones_like(z_monopile)

##            self.z_node = np.concatenate([z_monopile[:-1], self.z_node])
##            self.d_node = np.concatenate([d_Submonopile[:-1], self.d_node])
##            self.t_node = np.concatenate([t_monopile[:-1], self.t_node])

##            self.L_reinforced_node = np.interp(self.z_node, self.z, self.L_reinforced)
            # self.L_reinforced_node = np.concatenate([L_monopile[:-1], self.L_reinforced_node])
            # self.z_reinforced = np.concatenate([[z_monopile[0]], self.z_reinforced])

##            nprev = len(self.z_node_save)
##            self.dznode_dmonoHeight = np.concatenate([np.linspace(-1, 0, self.n_monopile)[:-1], np.zeros(nprev)])
##            self.dzmp_dz0 = np.ones(len(z_monopile)-1)
##            self.ddnode_ddmono = np.concatenate([np.ones(len(z_monopile)-1), np.zeros(nprev)])
##            self.dtnode_dtmono = np.concatenate([np.ones(len(z_monopile)-1), np.zeros(nprev)])


    def list_deriv_vars(self):

        if self.SubLength > 0:
            inputs = ('twrlength', 'z', 'd', 't', 'SubLength')  #, 'd_monopile', 't_monopile'
        else:
            inputs = ('twrlength', 'z', 'd', 't')
        outputs = ('z_node', 'd_node', 't_node')  # , 'z_reinforced')

        return inputs, outputs

    def provideJ(self):

        n = len(self.z_node_save)
        m = len(self.z)
        # n2 = len(self.z_reinforced)

        #print 'got here'
        dzn = hstack([self.z_node_save/MPtowerlgth, self.dznode_dz, np.zeros((n, 2*m))])
        ddn = hstack([np.zeros(n), self.ddnode_dz, self.ddnode_dd, np.zeros((n, m))])
        dtn = hstack([np.zeros(n), self.dtnode_dz, np.zeros((n, m)), self.dtnode_dt])

##        if self.monopileHeight > 0:
##            nfull = len(self.z_node)
##
##            dzmp = np.zeros((nfull-n, 1 + 3*m))
##            dzmp[:, 1] = self.dzmp_dz0
##
##            dzn = vstack([dzmp, dzn])
##            ddn = vstack([np.zeros((nfull-n, 1 + 3*m)), ddn])
##            dtn = vstack([np.zeros((nfull-n, 1 + 3*m)), dtn])
##
##
##            dzn = hstack([dzn, self.dznode_dmonoHeight, np.zeros((nfull, 2))])
##            ddn = hstack([ddn, np.zeros((nfull, 1)), self.ddnode_ddmono, np.zeros((nfull, 1))])
##            dtn = hstack([dtn, np.zeros((nfull, 2)), self.dtnode_dtmono])


        J = np.vstack([dzn, ddn, dtn])

        return J


class GeometricConstraints(Component):
    """docstring for OtherConstraints"""

    d = Array(iotype='in')
    t = Array(iotype='in')
    min_d_to_t = Float(120.0, iotype='in')
    min_taper = Float(0.4, iotype='in')

    weldability = Array(iotype='out')
    ##manufacturability = Float(iotype='out')
    manufacturability = Array(iotype='out')

    def execute(self):

        self.weldability = (self.min_d_to_t - self.d/self.t) / self.min_d_to_t
        ##self.manufacturability = self.min_taper - self.d[-1] / self.d[0]  # taper ratio
        manufacturability = self.min_taper - (np.roll(self.d,-1)/self.d)[0:-1]  # taper ratio)
        self.manufacturability = np.hstack((manufacturability,manufacturability[-1]))

    def list_deriv_vars(self):

        inputs = ('d', 't')
        outputs = ('weldability', 'manufacturability')
        return inputs, outputs


    def provideJ(self):

        nts=len(self.t)

        dw_dd = np.diag(-1.0/self.t/self.min_d_to_t)
        dw_dt = np.diag(self.d/self.t**2/self.min_d_to_t)

        dw = np.hstack([dw_dd, dw_dt])


        #dm_dd = np.zeros_like(self.d)
        #dm_dd[0] = self.d[-1]/self.d[0]**2
        #dm_dd[-1] = -1.0/self.d[0]

        dm_dd=np.zeros([nts,nts]) #initialize, since nts=nds; matrix is double banded; assume constraint assigned ot the bottom section of the segment
        dm_dd[0:-1,0:-1]=self.d[1:]/self.d[0:-1]**2
        dm_dd[0:-1,1:]  =-1./self.d[1:]
        dm_dd[-1,-1]=dm_dd[-2,-2] #augment the last section
        ##dm = np.hstack([dm_dd, np.zeros(len(self.t))])

        dm = np.hstack([dm_dd, np.zeros([nts,nts])])

        J = np.vstack([dw, dm])

        return J

class AllConstraints(Component):
    """This component allows the for changing the size of the arrays involved in the constraints"""
    #inputs
    stress1in = Array(iotype='in', units='N/m**2', desc='Von Mises stress along tower on downwind side (yaw-aligned +x).  Normalized by yield stress.  Includes safety factors.')
    stress2in = Array(iotype='in', units='N/m**2', desc='Von Mises stress along tower on downwind side (yaw-aligned +x).  Normalized by yield stress.  Includes safety factors.')
    shellBuckling1in = Array(iotype='in', desc='Shell buckling constraint load case #1.  Should be < 1 for feasibility.  Includes safety factors')
    shellBuckling2in = Array(iotype='in', desc='Shell buckling constraint load case #2.  Should be < 1 for feasibility.  Includes safety factors')
    buckling1in = Array(iotype='in', desc='Global buckling constraint load case #1.  Should be < 1 for feasibility.  Includes safety factors')
    buckling2in = Array(iotype='in', desc='Global buckling constraint load case #2.  Should be < 1 for feasibility.  Includes safety factors')
    damagein = Array(iotype='in', desc='Fatigue damage at each tower section')
    weldabilityin = Array(iotype='in')
    manufacturabilityin = Array(iotype='in')


    sttctrt =Int(0,iotype='in',desc='Starting Index into z_node for constraints based on nodes (e.g., buckling).')
    endctrt =Int(-1,iotype='in',desc='Ending Index into z_node for constraints based on nodes (e.g., buckling).')
    sttzctrt =Int(0,iotype='in',desc='Starting Index into zall for constraints based on dimensionless z (e.g., weldability).')
    endzctrt =Int(-1,iotype='in',desc='Ending Index into zall for constraints based on dimensionless (e.g., weldability).')

    #outputs
    stress1 = Array(iotype='out', units='N/m**2', desc='Von Mises stress along tower on downwind side (yaw-aligned +x).  Normalized by yield stress.  Includes safety factors.')
    stress2 = Array(iotype='out', units='N/m**2', desc='Von Mises stress along tower on downwind side (yaw-aligned +x).  Normalized by yield stress.  Includes safety factors.')
    shellBuckling1 = Array(iotype='out', desc='Shell buckling constraint load case #1.  Should be < 1 for feasibility.  Includes safety factors')
    shellBuckling2 = Array(iotype='out', desc='Shell buckling constraint load case #2.  Should be < 1 for feasibility.  Includes safety factors')
    buckling1 = Array(iotype='out', desc='Global buckling constraint load case #1.  Should be < 1 for feasibility.  Includes safety factors')
    buckling2 = Array(iotype='out', desc='Global buckling constraint load case #2.  Should be < 1 for feasibility.  Includes safety factors')
    damage = Array(iotype='out', desc='Fatigue damage at each requested tower section')
    weldability = Array(iotype='out')
    manufacturability = Array(iotype='out')

    def execute(self):
        idx0=self.sttctrt
        idx1=self.endctrt
        if idx1 ==-1:
            idx1 =None

        self.stress1 = self.stress1in[idx0:idx1]
        self.stress2 = self.stress2in[idx0:idx1]
        self.shellBuckling1 = self.shellBuckling1in[idx0:idx1]
        self.shellBuckling2 = self.shellBuckling2in[idx0:idx1]
        self.buckling1 = self.buckling1in[idx0:idx1]
        self.buckling2 = self.buckling2in[idx0:idx1]
        self.damage = self.damagein[idx0:idx1]
        idx0=self.sttzctrt
        idx1=self.endzctrt
        if idx1 ==-1:
            idx1 =None

        self.manufacturability = self.manufacturabilityin[idx0:idx1]
        self.weldability = self.weldabilityin[idx0:idx1]



class TowerBase(Component):
    """structural analysis of cylindrical tower

    all forces, moments, distances should be given (and returned) in the yaw-aligned coordinate system
    """

    # geometry
    #towerHeight = Float(iotype='in', units='m')
    z =       Array(iotype='in', unit='m', desc='Node z-coordinates  [m] along MP-tower')
    d =       Array(iotype='in', units='m', desc='Tower diameter at nodes')
    t =       Array(iotype='in', units='m', desc='Wall thickness at nodes')

    idxtwr = Int(iotype='in',desc='Starting Index into z (nodes) for actual tower (no TP or MP).')
    idxol0 = Int(iotype='in',desc='Starting Index into z (nodes) for overlap start.')
    idxol1 = Int(iotype='in',desc='Starting Index into z (nodes) for overlap end (top of MP).')

    L_reinforced = Array(iotype='in', units='m', desc='Reinforcement spacings for tower-only. Length of array = ``len(z)``.')
    yaw = Float(0.0, iotype='in', units='deg')

    # wind/wave loads
    windLoads = VarTree(AeroLoads(), iotype='in')
    waveLoads = VarTree(AeroLoads(), iotype='in')
    g = Float(9.81, iotype='in', units='m/s**2')

    # top mass
    top_F = Array(iotype='in', desc='Forces from the Rotor Nacelle Assembly at the tower top centerline (no weight)')
    top_M = Array(iotype='in', desc='Forces from the Rotor Nacelle Assembly at the tower top centerline (no weight contribution)')
    top_m = Float(iotype='in', desc='Mass of the RNA')
    top_I = Array(iotype='in', units='kg*m**2', desc='Mass moments of inertia. order: (xx, yy, zz, xy, xz, yz)')
    top_cm = Array(iotype='in', desc='RNA CM location w.r.t. to tower top centerline.')

    # soil
    k_soil = Array(iotype='in', desc='Stiffness properties at base of foundation')

    # material properties
    twr_material = Instance(Material(matname='HeavySteel', E=2.1E11, G=8.08E10, rho=8500., fy=345.e6), iotype='in', desc='Material properties for the tower shell.')
    mp_material =  Instance(Material(matname='HeavySteel', E=2.1E11, G=8.08E10, rho=8741., fy=345.e6), iotype='in', desc='Material properties for the pile and TP.')
    ol_material =  Instance(Material(matname='OLSteel'),                                               iotype='in', desc='Material properties for the Overlap Region.')

##    E = Float(210e9, iotype='in', units='N/m**2', desc='material modulus of elasticity')
##    G = Float(80.8e9, iotype='in', units='N/m**2', desc='material shear modulus')
##    rho = Float(8500.0, iotype='in', units='kg/m**3', desc='material density')
##    sigma_y = Float(345.0e6, iotype='in', units='N/m**2', desc='yield stress')

    # safety factors
    gamma_f = Float(1.35, iotype='in', desc='Safety factor on loads')
    gamma_m = Float(1.1, iotype='in', desc='Safety factor on materials')
    gamma_n = Float(1.0, iotype='in', desc='Safety factor on consequence of failure')
    gamma_b = Float(1.1, iotype='in', desc='Buckling safety factor')

    life = Float(20.0, iotype='in', desc='Fatigue life of tower')
    m_SN = Int(4, iotype='in', desc='Slope of S/N curve')
    DC = Float(80.0, iotype='in', desc='Standard value of stress')
    gamma_fatigue = Float(1.755, iotype='in', desc='Total safety factor for fatigue')
    z_DEL = Array(iotype='in')
    M_DEL = Array(iotype='in')

    # outputs
    mass = Float(iotype='out')
    f1 = Float(iotype='out', units='Hz', desc='First natural frequency')
    f2 = Float(iotype='out', units='Hz', desc='Second natural frequency')
    top_deflection = Float(iotype='out', units='m', desc='Deflection of tower top in yaw-aligned +x direction')
    stress = Array(iotype='out', units='N/m**2', desc='Von Mises stress along tower on downwind side (yaw-aligned +x).  Normalized by yield stress.  Includes safety factors.')
    # z_buckling = Array(iotype='out', units='m', desc='z-locations along tower where shell buckling is evaluted')
    shellBuckling = Array(iotype='out', desc='Shell buckling utilization.  should be < 1 for feasibility.  Includes safety factors')
    buckling = Array(iotype='out', desc='Global buckling utilization.  should be < 1 for feasibility.  Includes safety factors')
    damage = Array(iotype='out', desc='Fatigue damage at each tower section')


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

        damage=np.zeros(z.size)
        if any(self.M_DEL):
            M_DEL = np.interp(z, self.z_DEL, self.M_DEL)

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
        Pz_weight = -self.material.rho*self.g*math.pi*d*t
        Pz += Pz_weight

        # pBEAM loads object
        loads = _pBEAM.Loads(nodes, Px, Py, Pz)

        # material properties
        mat = _pBEAM.Material(self.material.E, self.material.G, self.material.rho)

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
        axial_stress = self.material.E*tower.axialStrain(nodes, d/2.0, 0.0*d, z)

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
            self.gamma_f*self.gamma_m*self.gamma_n, self.material.fy)

        # buckling
        self.shellBuckling = shellBucklingEurocode(d, t, axial_stress, hoop_stress, shear_stress,
            self.L_reinforced, self.material.E*np.ones(nodes), self.material.fy*np.ones(nodes),
            self.gamma_f, self.gamma_b)

        tower_height = self.z[-1] - self.z[0]
        self.buckling = bucklingGL(d, t, Fz, My, tower_height, self.material.E*np.ones(nodes),
            self.material.fy*np.ones(nodes), self.gamma_f, self.gamma_b)

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

    FrameAuxIns= VarTree(Frame3DDaux(),  iotype='in', desc='Auxiliary Data for Frame3DD')

    def __init__(self):
        import frame3dd  # import only if instantiated so that Frame3DD is not required for all users
        self.frame3dd = frame3dd
        super(TowerWithFrame3DD, self).__init__()


    def execute(self):

        #print 'from towerithframe', self.idxol0  #debug
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
        #Now account for materials at nodes
        E = self.mp_material.E*np.ones(n)
        G = self.mp_material.G*np.ones(n)
        fy = self.mp_material.fy*np.ones(n)
        rho = self.mp_material.rho*np.ones(n)
        #Now fix materials for overlap and tower
        E[self.idxol0:self.idxol1+1]=self.ol_material.E
        E[self.idxol1+1:]=self.twr_material.E
        G[self.idxol0:self.idxol1+1]=self.ol_material.G
        G[self.idxol1+1:]=self.twr_material.G
        rho[self.idxol0:self.idxol1+1]=self.ol_material.rho
        rho[self.idxol1+1:]=self.twr_material.rho
        fy[self.idxol0:self.idxol1+1]=self.ol_material.fy
        fy[self.idxol1+1:]=self.twr_material.fy
        #Now create E,G,fy for elements (n-1)
        Eel= np.mean(np.vstack((np.roll(E,-1),E)),0)[0:-1]
        Gel= np.mean(np.vstack((np.roll(G,-1),G)),0)[0:-1]
        fyel= np.mean(np.vstack((np.roll(fy,-1),fy)),0)[0:-1]
        rhoel= np.mean(np.vstack((np.roll(rho,-1),rho)),0)[0:-1]
        roll = np.zeros(n-1)

        elements = frame3dd.ElementData(element, N1, N2, Ax, Asy, Asz, Jx, Iy, Iz, Eel, Gel, roll, rhoel)

        # -----------------------------------


        # ------ options ------------

        shear = self.FrameAuxIns.sh_fg        # default= True or 1. 1=include shear deformation
        geom = self.FrameAuxIns.geo_fg        # default=False  or 0 # 1=include geometric stiffness
        dx = self.FrameAuxIns.deltaz          # default =5. x-axis increment for internal forces
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
        addGravityLoad = True

        tower.changeExtraNodeMass(N, EMs, EMx, EMy, EMz, EMxy, EMxz, EMyz, rhox, rhoy, rhoz, addGravityLoad)
        # ------------------------------------

        # ------- enable dynamic analysis ----------

        nM = self.FrameAuxIns.nModes              # default=6. Number of desired dynamic modes of vibration (below only necessary if nM > 0)
        Mmethod = self.FrameAuxIns.Mmethod        # default=1. 1: subspace Jacobi     2: Stodola
        lump = self.FrameAuxIns.lump              # default=0. 0: consistent mass ... 1: lumped mass matrix
        tol = self.FrameAuxIns.tol                # default=1e-9. Mode shape tolerance
        shift = self.FrameAuxIns.shift            # default=0. Shift value ... for unrestrained structures
        tower.enableDynamics(nM, Mmethod, lump, tol, shift)

        # ----------------------------


        # ------ static load case 1 ------------

        # gravity in the X, Y, Z, directions (global)
        gx = 0.0
        gy = 0.0
        gz = -self.g

        load = frame3dd.StaticLoadCase(gx, gy, gz)


        # tower top load  (have removed RNA weight from these forces since Frame3DD accounts for it)

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
        Vy = forces.Vy[iCase, :]
        Vx = -forces.Vz[iCase, :]

        Myy = forces.Myy[iCase, :]
        Mxx = -forces.Mzz[iCase, :]
        Mzz = forces.Txx[iCase, :]

        # one per element (first negative b.c. need reaction)
        Fz = np.concatenate([[-Fz[0]], Fz[1::2]])
        Vx = np.concatenate([[-Vx[0]], Vx[1::2]])
        Vy = np.concatenate([[-Vy[0]], Vy[1::2]])
        Mxx = np.concatenate([[-Mxx[0]], Mxx[1::2]])
        Myy = np.concatenate([[-Myy[0]], Myy[1::2]])
        Mzz = np.concatenate([[-Mzz[0]], Mzz[1::2]])

        # axial and shear stress (all stress evaluated on +x yaw side)
        #A = math.pi * self.d * self.t
        A = math.pi/4 * (self.d**2-(self.d-2 * self.t)**2)
        #Iyy = math.pi/8.0 * self.d**3 * self.t
        Iyy = math.pi/64.0 * (self.d**4 -(self.d- 2*self.t)**4)

        axial_stress = Fz/A - np.sqrt(Mxx**2+Myy**2)/Iyy*self.d/2.0
        shear_stress = 2. * np.sqrt(Vx**2+Vy**2) / A

        # hoop_stress (Eurocode method)
        hoop_stress = hoopStressEurocode(self.windLoads, self.waveLoads,
            self.z, self.d, self.t, self.L_reinforced)


        # von mises stress
        self.stress = vonMisesStressUtilization(axial_stress, hoop_stress, shear_stress,
            self.gamma_f*self.gamma_m*self.gamma_n, fy)

        # buckling
        ones_array = np.ones(nodes.node.size)
        # global buckling changed d and t to self.d and self.t
        self.shellBuckling = shellBucklingEurocode(self.d, self.t, axial_stress, hoop_stress, shear_stress,
            #self.L_reinforced, self.material.E*ones_array, fy*ones_array,
            self.L_reinforced, E, fy,
            self.gamma_f, self.gamma_b)

        tower_height = self.z[-1] - self.z[0]
        self.buckling = bucklingGL(self.d, self.t, Fz, Myy, tower_height,\
                       #self.material.E*ones_array, self.material.fy*ones_array,\
                       E, fy,\
                       self.gamma_f, self.gamma_b)

        # fatigue
        self.damage = self.fatigue()





# -----------------
#  Assembly
# -----------------


class TowerSE(Assembly):

    # geometry
    towerHeight = Float(iotype='in', units='m', desc='height of tower')
    monopileHeight = Float(0.0, iotype='in', units='m', desc='height of monopile (0.0 if no monopile)')
    d_monopile = Float(iotype='in', units='m', desc='diameter of monopile')
    t_monopile = Float(iotype='in', units='m', desc='wall thickness of monopile')
    z = Array(iotype='in', desc='locations along unit tower, linear lofting between')
    d = Array(iotype='in', units='m', desc='tower diameter at corresponding locations')
    t = Array(iotype='in', units='m', desc='shell thickness at corresponding locations')
    idxtwr = Int(0,iotype='in',desc='Starting Index into z for actual tower (no TP or MP)')
    idxol0 = Int(0,iotype='in',desc='Starting Index into z for overlap start')
    idxol1 = Int(0,iotype='in',desc='Starting Index into z for overlap end (top of MP)')

    n = Array(iotype='in', dtype=np.int, desc='Number of finite elements between tower-only sections.  Array length should be ``len(z)-1``.')
    L_reinforced = Array(iotype='in', units='m', desc='Reinforcement spacings for tower-only.  Length of array = ``len(z)``.')
    n_monopile = Int(iotype='in', desc='Number of finite elements in monopile, must be a minimum of 1 (top and bottom nodes).')
    yaw = Float(0.0, iotype='in', units='deg', desc='yaw angle')

    # rna
    top_m = Float(iotype='in', units='kg', desc='RNA (tower top) mass')
    top_I = Array(iotype='in', units='kg*m**2', desc='mass moments of inertia. order: (xx, yy, zz, xy, xz, yz)')
    top_cm = Array(iotype='in', units='m', desc='RNA center of mass')
    top1_F = Array(iotype='in', units='N', desc='Aerodynamic forces')
    top1_M = Array(iotype='in', units='N*m', desc='Aerodynamic moments')
    top2_F = Array(iotype='in', units='N', desc='Aerodynamic forces')
    top2_M = Array(iotype='in', units='N*m', desc='Aerodynamic moments')

    # environment
    wind_rho = Float(1.225, iotype='in', units='kg/m**3', desc='air density')
    wind_mu = Float(1.7934e-5, iotype='in', units='kg/(m*s)', desc='dynamic viscosity of air')
    wind_Uref1 = Float(iotype='in', units='m/s', desc='reference wind speed (usually at hub height)')
    wind_Uref2 = Float(iotype='in', units='m/s', desc='reference wind speed (usually at hub height)')
    wind_zref = Float(iotype='in', units='m', desc='corresponding reference height')
    wind_z0 = Float(0.0, iotype='in', units='m', desc='bottom of wind profile (height of ground/sea)')
    wind_cd= Float(iotype='in', desc='Cd coefficient, if left blank it will be calculated based on cylinder Re')  # -RRD

    wave_rho = Float(1027.0, iotype='in', units='kg/m**3', desc='water density')
    wave_mu = Float(1.3351e-3, iotype='in', units='kg/(m*s)', desc='dynamic viscosity of water')
    wave_cm = Float(2.0, iotype='in', desc='mass coefficient')
    wave_cd= Float(iotype='in', desc='Cd coefficient, if left blank it will be calculated based on cylinder Re')  # -RRD

    g = Float(9.81, iotype='in', units='m/s**2', desc='acceleration of gravity')

    # constraint parameters
    min_d_to_t = Float(120.0, iotype='in', desc='minimum allowable diameter to thickness ratio')
    min_taper = Float(0.4, iotype='in', desc='minimum allowable taper ratio from tower top to tower bottom')

    # material properties
    #material = Instance(Material(matname='HeavySteel', E=2.1E11, G=8.08E10, rho=8500., fy=345.e6), iotype='in', desc='Material properties for the tower shell.')
    twr_material = Instance(Material(matname='HeavySteel', E=2.1E11, G=8.08E10, rho=8500., fy=345.e6),    iotype='in', desc='Material properties for the tower shell.')
    mp_material = Instance(Material(matname='HeavySteel', E=2.1E11, G=8.08E10, rho=8741., fy=345.e6), iotype='in', desc='Material properties for the pile and TP.')
    grt_material = Instance(Material(matname='Grout'),                                                iotype='in', desc='Material properties for the Grout.')

    # safety factors
    gamma_f = Float(1.35, iotype='in', desc='safety factor on loads')
    gamma_m = Float(1.1, iotype='in', desc='safety factor on materials')
    gamma_n = Float(1.0, iotype='in', desc='safety factor on consequence of failure')
    gamma_b = Float(1.1, iotype='in', desc='buckling safety factor on consequence of failure')

    # fatigue parameters
    life = Float(20.0, iotype='in', desc='fatigue life of tower')
    m_SN = Int(4, iotype='in', desc='slope of S/N curve')
    DC = Float(80.0, iotype='in', desc='standard value of stress')
    gamma_fatigue = Float(1.755, iotype='in', desc='total safety factor for fatigue')
    z_DEL = Array(iotype='in', desc='locations along unit tower for M_DEL')
    M_DEL = Array(iotype='in', units='N*m', desc='damage equivalent moments along tower')

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
    f1 = Float(iotype='out', units='Hz', desc='First natural frequency')
    f2 = Float(iotype='out', units='Hz', desc='Second natural frequency')
    top_deflection1 = Float(iotype='out', units='m', desc='Deflection of tower top in yaw-aligned +x direction')
    top_deflection2 = Float(iotype='out', units='m', desc='Deflection of tower top in yaw-aligned +x direction')
    z_nodes = Array(iotype='out', units='m')
    stress1 = Array(iotype='out', units='N/m**2', desc='Von Mises stress along tower on downwind side (yaw-aligned +x).  Normalized by yield stress.  Includes safety factors.')
    stress2 = Array(iotype='out', units='N/m**2', desc='Von Mises stress along tower on downwind side (yaw-aligned +x).  Normalized by yield stress.  Includes safety factors.')
    shellBuckling1 = Array(iotype='out', desc='Shell buckling constraint load case #1.  Should be < 1 for feasibility.  Includes safety factors')
    shellBuckling2 = Array(iotype='out', desc='Shell buckling constraint load case #2.  Should be < 1 for feasibility.  Includes safety factors')
    buckling1 = Array(iotype='out', desc='Global buckling constraint load case #1.  Should be < 1 for feasibility.  Includes safety factors')
    buckling2 = Array(iotype='out', desc='Global buckling constraint load case #2.  Should be < 1 for feasibility.  Includes safety factors')
    damage = Array(iotype='out', desc='Fatigue damage at each tower section')
    weldability = Array(iotype='out')
    manufacturability = Array(iotype='out')


    def configure(self):

        #self.add('MPprep', MPprep())
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
        self.add('tower1', TowerBase())
        self.add('tower2', TowerBase())
        self.add('gc', GeometricConstraints())

        # self.driver.workflow.add(['geometry', 'wind', 'wave', 'windLoads', 'waveLoads', 'soil', 'rna', 'rotorloads', 'tower'])
        self.driver.workflow.add(['geometry', 'wind1', 'wind2', 'wave1', 'wave2',
            'windLoads1', 'windLoads2', 'waveLoads1', 'waveLoads2', 'soil', 'tower1', 'tower2', 'gc'])
        # TODO: probably better to do this with a driver or something rather than manually setting 2 cases


        # connections to geometry
        self.connect('towerHeight', 'geometry.towerHeight')
        self.connect('monopileHeight', 'geometry.monopileHeight')
        #self.connect('d_monopile', 'geometry.d_monopile')
        #self.connect('t_monopile', 'geometry.t_monopile')
        self.connect('z', 'geometry.z')
        self.connect('d', 'geometry.d')
        self.connect('MPprep.tmptptwr', 'geometry.t')
        self.connect('n', 'geometry.n')
        #self.connect('n_monopile', 'geometry.n_monopile')
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

        self.connect('wind_cd', 'windLoads1.cd_usr')  # -RRD
        self.connect('geometry.z_node', 'windLoads1.z')
        self.connect('geometry.d_node', 'windLoads1.d')

        # connections to windLoads2
        self.connect('wind2.U', 'windLoads2.U')
        self.connect('wind2.beta', 'windLoads2.beta')
        self.connect('wind_rho', 'windLoads2.rho')
        self.connect('wind_mu', 'windLoads2.mu')

        self.connect('wind_cd', 'windLoads2.cd_usr')  # -RRD
        self.connect('geometry.z_node', 'windLoads2.z')
        self.connect('geometry.d_node', 'windLoads2.d')

        # connections to waveLoads1
        self.connect('wave1.U', 'waveLoads1.U')
        self.connect('wave1.U0', 'waveLoads1.U0')
        self.connect('wave1.A', 'waveLoads1.A')
        self.connect('wave1.A0', 'waveLoads1.A0')
        self.connect('wave1.beta', 'waveLoads1.beta')
        self.connect('wave1.beta0', 'waveLoads1.beta0')
        self.connect('wave_rho', 'waveLoads1.rho')
        self.connect('wave_mu', 'waveLoads1.mu')
        self.connect('wave_cm', 'waveLoads1.cm')
        self.connect('wave_cd', 'waveLoads1.cd_usr')  # -RRD
        self.connect('geometry.z_node', 'waveLoads1.z')
        self.connect('geometry.d_node', 'waveLoads1.d')


        # connections to waveLoads2
        self.connect('wave2.U', 'waveLoads2.U')
        self.connect('wave2.U0', 'waveLoads2.U0')
        self.connect('wave2.A', 'waveLoads2.A')
        self.connect('wave2.beta', 'waveLoads2.beta')
        self.connect('wave2.beta0', 'waveLoads2.beta0')
        self.connect('wave2.A0', 'waveLoads2.A0')
        self.connect('wave_rho', 'waveLoads2.rho')
        self.connect('wave_mu', 'waveLoads2.mu')
        self.connect('wave_cm', 'waveLoads2.cm')
        self.connect('wave_cd', 'waveLoads2.cd_usr')  # -RRD
        self.connect('geometry.z_node', 'waveLoads2.z')
        self.connect('geometry.d_node', 'waveLoads2.d')

        # connections to tower
        self.connect('towerHeight', 'tower1.towerHeight')
        self.connect('geometry.z_node', 'tower1.z')
        self.connect('geometry.d_node', 'tower1.d')
        self.connect('geometry.t_node', 'tower1.t')

        self.connect('geometry.idxol0', 'tower1.idxol0')
        self.connect('geometry.idxol1', 'tower1.idxol1')
        self.connect('geometry.idxtwr', 'tower1.idxtwr')
        self.connect('MPprep.ol_material', 'tower1.ol_material')
        self.connect('twr_material',       'tower1.twr_material')
        self.connect('mp_material',        'tower1.mp_material')
##        self.connect('G', 'tower1.G')
##        self.connect('rho', 'tower1.rho')
##        self.connect('sigma_y', 'tower1.sigma_y')

        self.connect('geometry.L_reinforced_node', 'tower1.L_reinforced')
        self.connect('windLoads1.windLoads', 'tower1.windLoads')
        self.connect('waveLoads1.waveLoads', 'tower1.waveLoads')
        self.connect('soil.k', 'tower1.k_soil')
        self.connect('yaw', 'tower1.yaw')
        self.connect('top_m', 'tower1.top_m')
        self.connect('top_cm', 'tower1.top_cm')
        self.connect('top_I', 'tower1.top_I')
        self.connect('top1_F', 'tower1.top_F')
        self.connect('top1_M', 'tower1.top_M')
        self.connect('g', 'tower1.g')
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

        self.connect('geometry.idxol0', 'tower2.idxol0')
        self.connect('geometry.idxol1', 'tower2.idxol1')
        self.connect('geometry.idxtwr', 'tower2.idxtwr')

        self.connect('MPprep.ol_material', 'tower2.ol_material')
        self.connect('twr_material',       'tower2.twr_material')
        self.connect('mp_material',        'tower2.mp_material')
##        self.connect('E', 'tower2.E')
##        self.connect('G', 'tower2.G')
##        self.connect('rho', 'tower2.rho')
##        self.connect('sigma_y', 'tower2.sigma_y')

        self.connect('geometry.L_reinforced_node', 'tower2.L_reinforced')
        self.connect('windLoads2.windLoads', 'tower2.windLoads')
        self.connect('waveLoads2.waveLoads', 'tower2.waveLoads')
        self.connect('soil.k', 'tower2.k_soil')
        self.connect('yaw', 'tower2.yaw')
        self.connect('top_m', 'tower2.top_m')
        self.connect('top_cm', 'tower2.top_cm')
        self.connect('top_I', 'tower2.top_I')
        self.connect('top2_F', 'tower2.top_F')
        self.connect('top2_M', 'tower2.top_M')
        self.connect('g', 'tower2.g')


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
        self.connect('gc.manufacturability', 'manufacturability')


#### ------------- Additional Component and Assembly for Monopile/Jacket structure

class MPprep(Component):
    """This component is in charge of calculating properties for overlap member. It also builds building blocks to pass to TowerDiscretization.
    """
    # inputs
    DMP=Float( iotype='in', units='m', desc='MP outer diameter.')
    tMP=Float( iotype='in', units='m', desc='MP wall thickness.')
    DTP=Float( iotype='in', units='m', desc='TP outer diameter.')
    tTP=Float( iotype='in', units='m', desc='TP wall thickness.')

    twr_material = Instance(Material(matname='HeavySteel', E=2.1E11, G=8.08E10, rho=8500., fy=345.e6),    iotype='in', desc='Material properties for the tower shell.')
    mp_material = Instance(Material(matname='HeavySteel', E=2.1E11, G=8.08E10, rho=8741., fy=345.e6), iotype='in', desc='Material properties for the pile and TP.')
    grt_material = Instance(Material(matname='Grout'),                                                iotype='in', desc='Material properties for the Grout.')

    wdepth = Float( iotype='in', units='m', desc='Water depth')
    MP2MSL = Float( iotype='in', units='m', desc='Top of monopile above MSL (usually 5 m)')
    ol_fact = Float( iotype='in', units=None, desc='Overlap extent in MP ODs (usually 1.5)')
    embd_fact = Float( iotype='in', units=None, desc='Embedment length in terms of MP OD (usually 6.)')

    twrlength = Float(iotype='in', units='m', desc='Length of tower component')
    dck_botz  = Float(iotype='in', units='m', desc='Height of tower-bottom flange above MSL.')
    nsubin = Array(iotype='in', dtype=np.int, desc='MUST INPUT THIS as maxsubel auses constraints to change size and that''s not allowed. Number of finite elements between substructure-only sections.  Array length should be ``len(zsub)`` to account for sub-to-twr conneciton as well. ')

    rho_ol = Float(iotype='in', units='kg/m**3', desc='Effective density for overlap member.')
    E_ol = Float(iotype='in', units='N/m**2', desc='Effective Young''s modulus for overlap member.')
    G_ol = Float(iotype='in', units='N/m**2', desc='Effective Shear Modulus for overlap member.')
    t_ol = Float(iotype='in', units='m', desc='Effective wall thickness for overlap member.')

    # outputs
    zsub = Array(iotype='out', desc='Locations along unit MP-tower of MP and TP only, linear lofting between. Last station is 0.01m before top of TP')
    dsub = Array(iotype='out', units='m', desc='MP ODs at corresponding locations.') #Note for the overlap region: OD=DTP and an approximate t=t_ol, this component will calculate t_ol
    tsub = Array(iotype='out', units='m', desc='MP wall thickness at corresponding locations.')
    SubLength = Float(iotype='out', units='m', desc='Length of Substructure, from seabed to tower flange.')
    TPLength = Float(iotype='out', units='m', desc='Length of TP.')
    Emdlength = Float(iotype='out', units='m', desc='Length of Pile Embedment.')
    olap    = Float(iotype='out', units='m', desc='Length of TP-MP overlap [m].')
    subLreinforced = Array(iotype='out', units='m',desc='Buckling lengths for the various segments of substructure-only.  Length of array = ``len(zsub)``')
    nsub = Array(iotype='out', dtype=np.int, desc='Number of finite elements between substructure-only sections.  Array length should be ``len(zsub)`` to account for sub-to-twr conneciton as well.')

    ol_material=Instance(Material(matname='HeavySteel', E=2.1E11, G=8.08E10, rho=8741., fy=345.e6), iotype='out', desc='Material properties for the overlap region between pile and TP.')
    fout = Array(iotype='out',  desc='Equations to be satisfied (as in fout=0) to create overlap equivalent member.  See ol_equations function.')

    MPmass =   Float(iotype='out', units='kg', desc='Pile (not including embedment) Mass.')
    TPmass =   Float(iotype='out', units='kg', desc='TP Steel-only Mass.')
    GRTmass =  Float(iotype='out', units='kg', desc='Grout Mass.')
    Submass = Float(iotype='out', units='kg', desc='TP+grout+Pile (not including embedment) Mass.')
    Emdmass = Float(iotype='out', units='kg', desc='Pile Embedment Mass. For now based simply on a multiplier, in the future it will be calculated based on capacity.' )

    def __init__(self):

        super(MPprep, self).__init__()

        #controls what happens if derivatives are missing
        self.missing_deriv_policy = 'assume_zero'

    def execute(self):
        from scipy.optimize import fsolve

        wdepth=self.wdepth #[m]
        MP2MSL= self.MP2MSL #[m], top of MP aobve MSL
        self.olap = self.ol_fact*self.DMP #[m], overlap extent between TP and MP
        twrlength=self.twrlength #tower length
        dck_botz=self.dck_botz #tower-base flange level above MSL
        DMP = self.DMP
        tMP = self.tMP
        DTP = self.DTP
        tTP=  self.tTP

        #Calculate needed grout to fill the gap
        tgrt= (DTP-2.*tTP-DMP)/2.
        print 'tgrt=',tgrt
        #Calculate Overlap Properties
        def ol_equations(x,ODs,ts,mpmat,grtmat):
            """Insert here the equations to solve. x (array) is the unknown. \n

               x= [rho_ol,E_ol,G_ol,t_ol] equivalent overlap quantities. \n
                ODs- float(2), [DMP, DTP] , outer diameters. Note it is assumed that: Dgrt=DMP+2*tgrout. \n
                ts- float(3), [tMP, tTP, tgrt] , wall thicknesses. Dgrt=DMP+2*tgrout. \n
               mpmat  - object of class material, for the MP/TP material \n
               grtmat - object of class material, for the grout material \n
                All the other parameters are arguments that could be customized as needed."""
               #Here add auxiliary stuff that can be used by the equations below


            MP=Tube(ODs[0],ts[0],mat=mpmat)
            TP=Tube(ODs[1],ts[1],mat=mpmat)
            grout=Tube(ODs[0]+2*ts[2],ts[2],mat=grtmat)
            #LHS of the equations E_ol Jxx_ol= E1Jxx1+E2Jxx2+E3Jxx3
            C1=MP.mat.rho*MP.Area+TP.mat.rho*TP.Area+grout.mat.rho*grout.Area #mass
            C2=MP.mat.E*MP.Jxx +TP.mat.E*TP.Jxx#+grout.mat.E*grout.Jxx #Jxx do not include grout in bending stiffness
            C3=MP.mat.E*MP.Area+TP.mat.E*TP.Area#+grout.mat.E*grout.Area #Axial Stiffness EA, do not include grout in axial stiffness
            C4=1./(MP.mat.G*MP.J0)+1./(TP.mat.G*TP.J0)+1./(grout.mat.G*grout.J0) #1/GJ0_eq these are springs in series, the others in parallel
            #I wish I could use tube for the object to be found as well
            def Area(t_ol):
                return np.pi/4*(ODs[1]**2-(ODs[1]-2.*t_ol)**2)  # X-sectional area
            def Jxx(t_ol):
                return np.pi/64*(ODs[1]**4-(ODs[1]-2.*t_ol)**4)  # 2nd moment of inertia
            def J0(t_ol):
                return 2.*Jxx(t_ol)  # polar moment of inertia of hollow cylinder

            #REal Equations go here
                                    #1st Equation
            fout=(x[0]*Area(x[3])-C1,\
                                    #2nd Equation
                x[1]*Jxx(x[3])-C2,
                                    #3rd Equation
                x[1]*Area(x[3])-C3,\
                                    #4th Equation
                x[2]*J0(x[3])-1./C4,\
                )

            return fout

        #Initial guess
        guess=np.array([self.mp_material.rho,self.mp_material.E,self.mp_material.G,tMP+tTP])
        #print equations(guess,ODs,ts,matnames,rhos)  #test function works as expected

        #print DMP, DTP, tMP, tTP  #debug

        #outputs
            #        !!!WATCH!!!
        #x=np.array([self.rho_ol,self.E_ol,self.G_ol,self.t_ol])  #Use this when using fout externally solved
        x=fsolve(ol_equations,guess,args=(np.array([DMP, DTP]),np.array([tMP,tTP,tgrt]),self.mp_material,self.grt_material))#Use this when using fout internally solved  !!!!
        #Return fout for external minimization
        self.fout=ol_equations(x,np.array([DMP, DTP]),np.array([tMP,tTP,tgrt]),self.mp_material,self.grt_material)

        self.ol_material=Material(rho=x[0],E=x[1],G=x[2], fy=self.mp_material.fy, matname='EqSteel')
        self.SubLength = wdepth + dck_botz #substructure Length
        self.TPlength  = self.olap-MP2MSL+dck_botz #TP length
        mptwrlength=self.SubLength+twrlength   #overall length from seabed to top flange of tower
        #               [seabed, base of TP -0.01,               TP base, Top of MP (overlap), Top of MP+0.01,  Top of TP,         Top of TP +0.01 (tower base), Tower waist, Tower Top]
        self.zsub = (wdepth+np.array([-wdepth, MP2MSL-self.olap-0.01, MP2MSL-self.olap, MP2MSL,            MP2MSL +0.01,    dck_botz-0.01]) )
        self.dsub =         np.array([DMP,       DMP,             DTP,       DTP,                 DTP,           DTP])
        self.tsub =         np.array([tMP,       tMP,             x[3],      x[3],                tTP,           tTP])
        # then the discretization based on max element size, add the last nsub from the end of the substructure to the first point of the tower
        #self.nsub= np.ceil( np.hstack(((np.roll(self.zsub,-1)-self.zsub)[0:-1],(wdepth+dck_botz-self.zsub[-1])))/self.maxsubel)
        if self.nsubin.size:
            self.nsub=self.nsubin
        #print 'nsub=',self.nsub

    #Somehow changing nsub messes up
        self.subLreinforced = (wdepth+MP2MSL-self.olap)*np.ones_like(self.zsub) #[m] L_buckling for the substructure

        #MP and TP mass, and substructure mass
        self.MPmass=  np.float(Tube(DMP,tMP,Lgth=self.zsub[3],mat=self.mp_material).pseudomass)
        self.TPmass=  np.float(Tube(DTP,tTP,Lgth=dck_botz+self.olap-MP2MSL,mat=self.mp_material).pseudomass) #float needed as L might be an array
        self.GRTmass= np.float(Tube(DMP+2*tgrt,tgrt,Lgth=self.olap,mat=self.grt_material).pseudomass)
        self.Submass= self.MPmass+self.TPmass+self.GRTmass
        #next two may change when we set the pile capacity properly
        self.Emdlength=self.embd_fact*DMP
        self.Emdmass= np.float(Tube(DMP,tMP,Lgth=self.Emdlength,mat=self.mp_material).pseudomass)#float needed as L might be an array

        #zsub must be dimensionless
        self.zsub /= mptwrlength
        #_____________________________________________#


#____________________________________________________________________________________________________________#

##class JacketPositioning(Component):
##
##    # inputs
##    sea_depth = Float(20.0, iotype='in', units='m', desc='average sea depth for the site or for the particular monopile position')
##    tower_length = Float(iotype='in', units='m', desc='length of tower')
##    tower2hub = Float(iotype='in', units='m', desc='vertical distance of tower top to shaft connection to hub')
##    MP2MSL = Float(5.0,iotype='in', units='m', desc='top of monopile above MSL')
##    deck_height = Float(iotype='in', units='m', desc='height of deck above mean sea level. For the MP this is the bottom tower flange elevation MSL.')
##    d_monopile = Float(iotype='in', units='m', desc='diameter of monopile')
##    t_monopile = Float(iotype='in', units='m', desc='wall thickness of monopile')
##    t_grout= Float(iotype='in', units='m', desc=' thickness of monopile')   # t_jacket = Float(iotype='in', units='m', desc=' thickness of monopile') ??
##    d_tower_base = Float(iotype='in', units='m', desc='diameter at the tower base')
##    d_tower_top = Float(iotype='in', units='m', desc='diameter at the tower top')
##    t_tower_base = Float(iotype='in', units='m', desc='wall thickness of tower base')
##    t_tower_top = Float(iotype='in', units='m', desc='wall thickness of tower top')
##
##    # outputs
##    # tower
##    towerHeight = Float(iotype='out', units='m', desc='height of tower')
##    monopileHeight = Float(0.0, iotype='out', units='m', desc='height of monopile (0.0 if no monopile)')
##    z = Array(iotype='out', desc='locations along unit tower, linear lofting between')
##    d = Array(iotype='out', units='m', desc='tower diameter at corresponding locations')
##    t = Array(iotype='out', units='m', desc='shell thickness at corresponding locations')
##    n = Array(iotype='out', dtype=np.int, desc='number of finite elements between sections.  array length should be ``len(z)-1``')
##    L_reinforced = Array(iotype='out', units='m', desc='distance along tower for reinforcements used in buckling calc')
##    n_monopile = Int(iotype='out', desc='number of finite elements in monopile, must be a minimum of 1 (top and bottom nodes)')
##    # environment
##    wind_zref = Float(iotype='out', units='m', desc='corresponding reference height')
##    wind_z0 = Float(0.0, iotype='out', units='m', desc='bottom of wind profile (height of ground/sea)')
##    z_surface = Float(iotype='out', units='m', desc='vertical location of water surface')
##    z_floor = Float(0.0, iotype='out', units='m', desc='vertical location of sea floor')
##
##    def __init__(self):
##
##        super(JacketPositioning, self).__init__()
##
##        #controls what happens if derivatives are missing
##        self.missing_deriv_policy = 'assume_zero'
##
##    def execute(self):
##
##        # tower/monopile positioning
##        self.towerHeight = self.tower_length + 1.5*self.d_monopile - self.MP2MSL # account for extra length below sea level- This is now the length from the base of the TP
##
##        self.z = np.array([0.0, 1.5*self.d_monopile, 1.5*self.d_monopile+0.001, 1.5*self.d_monopile +self.deck_height - self.MP2MSL, \
##                            1.5*self.d_monopile + (self.deck_height - self.MP2MSL) + 0.001, \
##                            ((self.deck_height + 1.5*self.d_monopile - self.MP2MSL + 0.1)/self.towerHeight + 1.0)/2.0, self.towerHeight])/self.towerHeight
##
###        self.d = np.array([self.d_monopile, self.d_monopile, self.d_monopile+2.*self.t_monopile+2.*self.t_grout, \
###                           self.d_monopile+2.*self.t_monopile+2.*self.t_grout, self.d_tower_base, (self.d_tower_base + self.d_tower_top)/2., \
###                           self.d_tower_top])
##        self.d = np.array([self.d_TP, self.d_TP, self.d_monopile+2.*self.t_monopile+2.*self.t_grout, \
##                           self.d_monopile+2.*self.t_monopile+2.*self.t_grout, self.d_tower_base, (self.d_tower_base + self.d_tower_top)/2., \
##                           self.d_tower_top])
##
##        self.t = [self.t_monopile,  self.t_monopile+self.t_grout, self.t_grout, self.t_grout, self.t_tower_base, \
##                (self.t_tower_base + self.t_tower_top)/2.0, self.t_tower_top]
##        self.n = [4, 2, 6, 2, 5, 5] # varying n based on positioning due to asymetric node spacing
##        self.n_reinforced = 7
##        self.L_reinforced = np.array([30., 30., 30., 30., 30., 30., 30.])  # [m] buckling length
##        # monopile
##        self.monopileHeight = self.sea_depth - (1.5*self.d_monopile - self.MP2MSL)
##        self.n_monopile = 10
##
##        # environment positioning
##        #wind
##        self.wind_zref = self.towerHeight + self.tower2hub
##        self.wind_z0 = 1.5*self.d_monopile - self.MP2MSL
##        #wave
##        self.z_surface = 1.5*self.d_monopile - self.MP2MSL
##        self.z_floor = 1.5*self.d_monopile - self.MP2MSL - self.sea_depth
##
##        # derivatives
##        d_tH_d_tl = 1.0
##        d_tH_d_md = 1.5
##        d_tH_d_me = -1.0
##
##        d_z_d_md = np.array([0.0,(self.towerHeight*1.5 - 1.5*self.d_monopile*1.5)/(self.towerHeight**2.),(self.towerHeight*1.5 - (1.5*self.d_monopile+0.1)*1.5)/(self.towerHeight**2.), \
##                             (1.5*self.towerHeight - (self.deck_height + 1.5*self.d_monopile - self.Mp2MSL)*1.5)/(self.towerHeight**2.), \
##                             (1.5*self.towerHeight - (self.deck_height + 1.5*self.d_monopile - self.Mp2MSL + 0.1)*1.5)/(self.towerHeight**2.), \
##                             0.5*(1.5*self.towerHeight - (self.deck_height + 1.5*self.d_monopile - self.Mp2MSL + 0.1)*1.5)/(self.towerHeight**2.), 0.0])
##        d_z_d_tl = np.array([0.0,-(1.5*self.d_monopile)/(self.towerHeight**2.),-(1.5*self.d_monopile+0.1)/(self.towerHeight**2.), \
##                             -(self.deck_height + 1.5*self.d_monopile - self.Mp2MSL)/(self.towerHeight**2.), \
##                             -(self.deck_height + 1.5*self.d_monopile - self.Mp2MSL + 0.1)/(self.towerHeight**2.), \
##                             -0.5*(self.deck_height + 1.5*self.d_monopile - self.Mp2MSL + 0.1)/(self.towerHeight**2.), 0.0])
##        d_z_d_dH =np.array([0.0, 0.0, 0.0, 1./self.towerHeight, 1./self.towerHeight, 0.5/self.towerHeight, 0.0])
##        d_z_d_me = np.array([0.0,(1.5*self.d_monopile)/(self.towerHeight**2.), (1.5*self.d_monopile+0.1)/(self.towerHeight**2.), \
##                             (-self.towerHeight+(self.deck_height + 1.5*self.d_monopile - self.Mp2MSL))/(self.towerHeight**2.), \
##                             (-self.towerHeight+(self.deck_height + 1.5*self.d_monopile - self.Mp2MSL + 0.1))/(self.towerHeight**2.), \
##                             0.5*(-self.towerHeight+(self.deck_height + 1.5*self.d_monopile - self.Mp2MSL + 0.1))/(self.towerHeight**2.), 0.0])
##
##        d_d_d_md = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
##        d_d_d_mt = np.array([0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0])
##        d_d_d_jt = np.array([0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0])
##        d_d_d_tbd = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0])
##        d_d_d_ttd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0])
##
##        d_t_d_mt = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
##        d_t_d_jt = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
##        d_t_d_ttb = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0])
##        d_t_d_ttt = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0])
##
##        d_mH_d_md = -1.5
##        d_mH_d_sd = 1.0
##        d_mH_d_me = 1.0
##
##        d_zref_d_tl = d_tH_d_tl
##        d_zref_d_md = d_tH_d_md
##        d_zref_d_me = d_tH_d_me
##        d_zref_d_ts = 1.0
##
##        d_z0_d_md = 1.5
##        d_z0_d_me = -1.0
##
##        d_zsurf_d_md = 1.5
##        d_zsurf_d_me = -1.0
##
##        d_zfl_d_md = 1.5
##        d_zfl_d_me = -1.0
##        d_zfl_d_sd = -1.0
##
##        self.J = np.array([[0.0, d_tH_d_tl, 0.0, d_tH_d_me, 0.0, d_tH_d_md, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
##                           [d_mH_d_sd, 0.0, 0.0, d_mH_d_me, 0.0, d_mH_d_md, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
##                           [0.0, d_zref_d_tl, d_zref_d_ts, d_zref_d_me, 0.0, d_zref_d_md, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
##                           [0.0, 0.0, 0.0, d_z0_d_me, 0.0, d_z0_d_md, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
##                           [0.0, 0.0, 0.0, d_zsurf_d_me, 0.0, d_zsurf_d_md, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
##                           [d_zfl_d_sd, 0.0, 0.0, d_zfl_d_me, 0.0, d_zfl_d_md, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
##                           [0.0, 0.0, 0.0, 0.0, 0.0, d_d_d_md[0], d_d_d_mt[0], d_d_d_jt[0], d_d_d_tbd[0], d_d_d_ttd[0], 0.0, 0.0], \
##                           [0.0, 0.0, 0.0, 0.0, 0.0, d_d_d_md[1], d_d_d_mt[1], d_d_d_jt[1], d_d_d_tbd[1], d_d_d_ttd[1], 0.0, 0.0], \
##                           [0.0, 0.0, 0.0, 0.0, 0.0, d_d_d_md[2], d_d_d_mt[2], d_d_d_jt[2], d_d_d_tbd[2], d_d_d_ttd[2], 0.0, 0.0], \
##                           [0.0, 0.0, 0.0, 0.0, 0.0, d_d_d_md[3], d_d_d_mt[3], d_d_d_jt[3], d_d_d_tbd[3], d_d_d_ttd[3], 0.0, 0.0], \
##                           [0.0, 0.0, 0.0, 0.0, 0.0, d_d_d_md[4], d_d_d_mt[4], d_d_d_jt[4], d_d_d_tbd[4], d_d_d_ttd[4], 0.0, 0.0], \
##                           [0.0, 0.0, 0.0, 0.0, 0.0, d_d_d_md[5], d_d_d_mt[5], d_d_d_jt[5], d_d_d_tbd[5], d_d_d_ttd[5], 0.0, 0.0], \
##                           [0.0, 0.0, 0.0, 0.0, 0.0, d_d_d_md[6], d_d_d_mt[6], d_d_d_jt[6], d_d_d_tbd[6], d_d_d_ttd[6], 0.0, 0.0], \
##                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, d_t_d_mt[0], d_t_d_jt[0], 0.0, 0.0, d_t_d_ttb[0], d_t_d_ttt[0]], \
##                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, d_t_d_mt[1], d_t_d_jt[1], 0.0, 0.0, d_t_d_ttb[1], d_t_d_ttt[1]], \
##                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, d_t_d_mt[2], d_t_d_jt[2], 0.0, 0.0, d_t_d_ttb[2], d_t_d_ttt[2]], \
##                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, d_t_d_mt[3], d_t_d_jt[3], 0.0, 0.0, d_t_d_ttb[3], d_t_d_ttt[3]], \
##                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, d_t_d_mt[4], d_t_d_jt[4], 0.0, 0.0, d_t_d_ttb[4], d_t_d_ttt[4]], \
##                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, d_t_d_mt[5], d_t_d_jt[5], 0.0, 0.0, d_t_d_ttb[5], d_t_d_ttt[5]], \
##                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, d_t_d_mt[6], d_t_d_jt[6], 0.0, 0.0, d_t_d_ttb[6], d_t_d_ttt[6]], \
##                           [0.0, d_z_d_tl[0], 0.0, d_z_d_me[0], d_z_d_dH[0], d_z_d_md[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
##                           [0.0, d_z_d_tl[1], 0.0, d_z_d_me[1], d_z_d_dH[1], d_z_d_md[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
##                           [0.0, d_z_d_tl[2], 0.0, d_z_d_me[2], d_z_d_dH[2], d_z_d_md[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
##                           [0.0, d_z_d_tl[3], 0.0, d_z_d_me[3], d_z_d_dH[3], d_z_d_md[3], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
##                           [0.0, d_z_d_tl[4], 0.0, d_z_d_me[4], d_z_d_dH[4], d_z_d_md[4], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
##                           [0.0, d_z_d_tl[5], 0.0, d_z_d_me[5], d_z_d_dH[5], d_z_d_md[5], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
##                           [0.0, d_z_d_tl[6], 0.0, d_z_d_me[6], d_z_d_dH[6], d_z_d_md[6], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
##
##
##    def list_deriv_vars(self):
##
##        inputs = ('sea_depth', 'tower_length','tower2hub','Mp2MSL', 'deck_height','d_monopile','t_monopile',\
##                  't_grout', 'd_tower_base', 'd_tower_top', 't_tower_base', 't_tower_top')
##        outputs = ('towerHeight', 'monopileHeight', 'wind_zref','wind_z0', 'z_surface','z_floor','d', 't', 'z')
##        return inputs, outputs
##
##    def provideJ(self):
##
##        return self.J


class TowerMonopileSE(Assembly):

    # geometry
    wdepth = Float( iotype='in', units='m', desc='Water depth')
    MP2MSL = Float( iotype='in', units='m', desc='Top of monopile above MSL (usually 5 m)')
    ol_fact = Float( iotype='in', units=None, desc='Overlap extent in MP ODs (usually 1.5)')
    embd_fact = Float( iotype='in', units=None, desc='Embedment length in terms of MP OD (usually 6.)')

    twrlength = Float(iotype='in', units='m', desc='Length of tower component')
    dck_botz  = Float(iotype='in', units='m', desc='Height of tower-bottom flange above MSL.')

    DMP=Float( iotype='in', units='m', desc='MP outer diameter.')
    tMP=Float( iotype='in', units='m', desc='MP wall thickness.')
    DTP=Float( iotype='in', units='m', desc='TP outer diameter.')
    tTP=Float( iotype='in', units='m', desc='TP wall thickness.')

    nsubin = Array(iotype='in', dtype=np.int, desc='Number of finite elements between substructure-only sections.  Array length should be ``len(zsub)`` to account for sub-to-twr conneciton as well. If given the mzxsubel is overriden')
##    d_tower_base = Float(iotype='in', units='m', desc='diameter at the tower base')
##    d_tower_top = Float(iotype='in', units='m', desc='diameter at the tower top')
##    t_tower_base = Float(iotype='in', units='m', desc='wall thickness of tower base')
##    t_tower_top = Float(iotype='in', units='m', desc='wall thickness of tower top')
    z = Array(iotype='in', desc='Locations along unit tower (tower only), linear lofting between')
    d = Array(iotype='in', units='m', desc='Tower diameter at corresponding locations')
    t = Array(iotype='in', units='m', desc='Wall thickness at corresponding locations')
    n = Array(iotype='in', dtype=np.int, desc='Number of finite elements between tower sections.  Array length should be ``len(z)-1``')

    L_reinforced = Array(iotype='in', units='m', desc='Reinforcement spacings used in buckling calculations along the tower-only. Length of array = ``len(z)``.  Units are [m].')

    yaw = Float(0.0, iotype='in', units='deg', desc='yaw angle')

    # rna
    top_m = Float(iotype='in', units='kg', desc='RNA (tower top) mass')
    top_I = Array(iotype='in', units='kg*m**2', desc='mass moments of inertia. order: (xx, yy, zz, xy, xz, yz)')
    top_cm = Array(iotype='in', units='m', desc='RNA center of mass')
    top1_F = Array(iotype='in', units='N', desc='Aerodynamic forces')
    top1_M = Array(iotype='in', units='N*m', desc='Aerodynamic moments')
    top2_F = Array(iotype='in', units='N', desc='Aerodynamic forces')
    top2_M = Array(iotype='in', units='N*m', desc='Aerodynamic moments')

    # environment
    wdepth = Float(20.0, iotype='in', units='m', desc='Water depth.')
    wind_rho = Float(1.225, iotype='in', units='kg/m**3', desc='Air density.')
    wind_mu = Float(1.7934e-5, iotype='in', units='kg/(m*s)', desc='Dynamic viscosity of air.')
    wind_Uref1 = Float(iotype='in', units='m/s', desc='Reference wind speed (usually at hub height).')
    wind_Uref2 = Float(iotype='in', units='m/s', desc='Reference wind speed (usually at hub height).')
    wind_zref = Float(iotype='in', units='m', desc='corresponding reference height')
    wind_z0 = Float(0.0, iotype='in', units='m', desc='bottom of wind profile (height of ground/sea)')
    wind_cd= Float(iotype='in', desc='Cd coefficient, if left blank it will be calculated based on cylinder Re')  # -RRD

    wave_rho = Float(1027.0, iotype='in', units='kg/m**3', desc='water density')
    wave_mu = Float(1.3351e-3, iotype='in', units='kg/(m*s)', desc='dynamic viscosity of water')
    wave_cm = Float(2.0, iotype='in', desc='mass coefficient')
    wave_cd= Float(iotype='in', desc='Cd coefficient, if left blank it will be calculated based on cylinder Re')  # -RRD
    z_surface= Float(iotype='in', units='m', desc='MSL coordinate [m] with respect to overall MP-tower coordinate system.')
    z_floor  = Float(iotype='in', units='m', desc='Seafloor coordinate with respect to overall unit MP-tower.')

    g = Float(9.81, iotype='in', units='m/s**2', desc='acceleration of gravity')

    # constraint parameters
    min_d_to_t = Float(120.0, iotype='in', desc='minimum allowable diameter to thickness ratio')
    min_taper = Float(0.4, iotype='in', desc='minimum allowable taper ratio from tower top to tower bottom')
    sttctrt =Int(0,iotype='in',desc='Starting Index into z_node for constraints based on nodes (e.g., buckling).')
    endctrt =Int(-1,iotype='in',desc='Ending Index into z_node for constraints based on nodes (e.g., buckling).')
    sttzctrt =Int(0,iotype='in',desc='Starting Index into zall for constraints based on dimensionless z (e.g., weldability).')
    endzctrt =Int(-1,iotype='in',desc='Ending Index into zall for constraints based on dimensionless (e.g., weldability).')

    # material properties
    twr_material = Instance(Material(matname='HeavySteel', E=2.1E11, G=8.08E10, rho=8500., fy=355.e6), iotype='in', desc='Material properties for the tower shell.')
    mp_material = Instance(Material(matname='HeavySteel', E=2.1E11, G=8.08E10, rho=8741., fy=345.e6), iotype='in', desc='Material properties for the pile and TP.')
    grt_material = Instance(Material(matname='Grout'), iotype='in', desc='Material properties for the Grout.')

    # safety factors
    gamma_f = Float(1.35, iotype='in', desc='safety factor on loads')
    gamma_m = Float(1.1, iotype='in', desc='safety factor on materials')
    gamma_n = Float(1.0, iotype='in', desc='safety factor on consequence of failure')
    gamma_b = Float(1.1, iotype='in', desc='buckling safety factor on consequence of failure')

    # fatigue parameters
    life = Float(20.0, iotype='in', desc='fatigue life of tower')
    m_SN = Int(4, iotype='in', desc='slope of S/N curve')
    DC = Float(80.0, iotype='in', desc='standard value of stress')
    gamma_fatigue = Float(1.755, iotype='in', desc='total safety factor for fatigue')
    z_DEL = Array(iotype='in', desc='locations along unit tower for M_DEL')
    M_DEL = Array(iotype='in', units='N*m', desc='damage equivalent moments along tower')

    # replace
    wind1 = Slot(WindBase)
    wind2 = Slot(WindBase)
    wave1 = Slot(WaveBase)
    wave2 = Slot(WaveBase)
    soil = Slot(SoilBase)
    tower1 = Slot(TowerBase)
    tower2 = Slot(TowerBase)

    # outputs
    mass = Float(iotype='out', units='kg', desc='MP-tower (not including embedment) Mass.')
    MPmass  = Float(iotype='out', units='kg', desc='Pile (not including embedment) Mass.')
    TPmass  = Float(iotype='out', units='kg', desc='TP Mass.')
    Submass = Float(iotype='out', units='kg', desc='TP+grout+Pile (not including embedment) Mass.')
    Emdmass = Float(iotype='out', units='kg', desc='Pile Embedment Mass. For now based simply on a multiplier, in the future it will be calculated based on capacity.' )

    f1 = Float(iotype='out', units='Hz', desc='First natural frequency')
    f2 = Float(iotype='out', units='Hz', desc='Second natural frequency')
    top_deflection1 = Float(iotype='out', units='m', desc='Deflection of tower top in yaw-aligned +x direction')
    top_deflection2 = Float(iotype='out', units='m', desc='Deflection of tower top in yaw-aligned +x direction')
    z_nodes = Array(iotype='out', units='m')
    stress1 = Array(iotype='out', units='N/m**2', desc='Von Mises stress along tower on downwind side (yaw-aligned +x).  Normalized by yield stress.  Includes safety factors.')
    stress2 = Array(iotype='out', units='N/m**2', desc='Von Mises stress along tower on downwind side (yaw-aligned +x).  Normalized by yield stress.  Includes safety factors.')
    shellBuckling1 = Array(iotype='out', desc='Shell buckling constraint load case #1.  Should be < 1 for feasibility.  Includes safety factors')
    shellBuckling2 = Array(iotype='out', desc='Shell buckling constraint load case #2.  Should be < 1 for feasibility.  Includes safety factors')
    buckling1 = Array(iotype='out', desc='Global buckling constraint load case #1.  Should be < 1 for feasibility.  Includes safety factors')
    buckling2 = Array(iotype='out', desc='Global buckling constraint load case #2.  Should be < 1 for feasibility.  Includes safety factors')
    damage = Array(iotype='out', desc='Fatigue damage at each tower section')
    weldability = Array(iotype='out')
    manufacturability = Array(iotype='out')


    def configure(self):
        self.add('MPprep', MPprep())
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
        self.add('tower1', TowerBase())
        self.add('tower2', TowerBase())
        self.add('gc', GeometricConstraints())
        self.add('CNTRT', AllConstraints())  #This component is to return a set of shrunk arrays for the constraints to be used by optimization, nowhere else in here
        #self.add('jp', JacketPositioning())

        # self.driver.workflow.add(['MPprep','geometry', 'wind', 'wave', 'windLoads', 'waveLoads', 'soil', 'rna', 'rotorloads', 'tower'])
        self.driver.workflow.add(['MPprep','geometry', 'wind1', 'wind2', 'wave1', 'wave2',
            'windLoads1', 'windLoads2', 'waveLoads1', 'waveLoads2', 'soil', 'tower1', 'tower2', 'gc','CNTRT']) #, 'jp'
        # TODO: probably better to do this with a driver or something rather than manually setting 2 cases

        # connections to geometry (from positioning module)
##        self.connect('jp.towerHeight', 'geometry.towerHeight')
##        self.connect('jp.monopileHeight', 'geometry.monopileHeight')
##        self.connect('jp.d_monopile', 'geometry.d_monopile')
##        self.connect('jp.t_monopile', 'geometry.t_monopile')
##        self.connect('jp.z', 'geometry.z')
##        self.connect('jp.d', 'geometry.d')
##        self.connect('jp.t', 'geometry.t')
##        self.connect('jp.n', 'geometry.n')
##        self.connect('jp.n_monopile', 'geometry.n_monopile')
##        self.connect('jp.L_reinforced', 'geometry.L_reinforced')

        # connections to MPprep
        self.connect('mp_material', 'MPprep.mp_material')
        self.connect('grt_material', 'MPprep.grt_material')
        self.connect('DMP', 'MPprep.DMP')
        self.connect('DTP', 'MPprep.DTP')
        self.connect('tMP', 'MPprep.tMP')
        self.connect('tTP', 'MPprep.tTP')
        self.connect('wdepth', 'MPprep.wdepth')
        self.connect('MP2MSL', 'MPprep.MP2MSL')
        self.connect('ol_fact', 'MPprep.ol_fact')
        self.connect('embd_fact', 'MPprep.embd_fact')
        self.connect('twrlength', 'MPprep.twrlength')
        self.connect('dck_botz', 'MPprep.dck_botz')
        self.connect('nsubin', 'MPprep.nsubin')

        # connections to geometry
        self.connect('MPprep.SubLength', 'geometry.SubLength')
        self.connect('twrlength',        'geometry.twrlength')
        self.connect('z', 'geometry.z')
        self.connect('d', 'geometry.d')
        self.connect('t', 'geometry.t')
        self.connect('n', 'geometry.n')
        self.connect('L_reinforced', 'geometry.L_reinforced')
        self.connect('MPprep.zsub', 'geometry.zsub')
        self.connect('MPprep.dsub', 'geometry.dsub')
        self.connect('MPprep.tsub', 'geometry.tsub')
        self.connect('MPprep.nsub', 'geometry.nsub')
        self.connect('MPprep.subLreinforced', 'geometry.subLreinforced')

         # connections to wind1
        self.connect('geometry.z_node', 'wind1.z')
        self.connect('wind_Uref1', 'wind1.Uref')
        self.connect('wind_zref', 'wind1.zref') # from positioning component
        self.connect('wind_z0', 'wind1.z0') # from positioning component

        # connections to wind2
        self.connect('geometry.z_node', 'wind2.z')
        self.connect('wind_Uref2', 'wind2.Uref')
        self.connect('wind_zref', 'wind2.zref') # from positioning component
        self.connect('wind_z0', 'wind2.z0') # from positioning component

        # connections to wave1
        self.connect('geometry.z_node', 'wave1.z')

        # connections to wave2
        self.connect('geometry.z_node', 'wave2.z')

        # connections to windLoads1
        self.connect('wind1.U', 'windLoads1.U')
        self.connect('wind1.beta', 'windLoads1.beta')
        self.connect('wind_rho', 'windLoads1.rho')
        self.connect('wind_mu', 'windLoads1.mu')

        self.connect('wind_cd', 'windLoads1.cd_usr')  # -RRD
        self.connect('geometry.z_node', 'windLoads1.z')
        self.connect('geometry.d_node', 'windLoads1.d')

        # connections to windLoads2
        self.connect('wind2.U', 'windLoads2.U')
        self.connect('wind2.beta', 'windLoads2.beta')
        self.connect('wind_rho', 'windLoads2.rho')
        self.connect('wind_mu', 'windLoads2.mu')

        self.connect('wind_cd', 'windLoads2.cd_usr')  # -RRD
        self.connect('geometry.z_node', 'windLoads2.z')
        self.connect('geometry.d_node', 'windLoads2.d')

        # connections to waveLoads1
        self.connect('wave1.U', 'waveLoads1.U')
        self.connect('wave1.U0', 'waveLoads1.U0')
        self.connect('wave1.A', 'waveLoads1.A')
        self.connect('wave1.A0', 'waveLoads1.A0')
        self.connect('wave1.beta', 'waveLoads1.beta')
        self.connect('wave1.beta0', 'waveLoads1.beta0')
        self.connect('wave1.z_surface', 'waveLoads1.wlevel')
        self.connect('wave_rho', 'waveLoads1.rho')
        self.connect('wave_mu', 'waveLoads1.mu')
        self.connect('wave_cm', 'waveLoads1.cm')
        self.connect('wave_cd', 'waveLoads1.cd_usr')  # -RRD
        self.connect('geometry.z_node', 'waveLoads1.z')
        self.connect('geometry.d_node', 'waveLoads1.d')

##        self.connect('jp.z_surface','wave1.z_surface')  # from positioning component
##        self.connect('jp.z_floor','wave1.z_floor')  # from positioning component
##        self.connect('z_surface','wave1.z_surface')
##        self.connect('z_floor','wave1.z_floor')


        # connections to waveLoads2
        self.connect('wave2.U', 'waveLoads2.U')
        self.connect('wave2.U0', 'waveLoads2.U0')
        self.connect('wave2.A', 'waveLoads2.A')
        self.connect('wave2.beta', 'waveLoads2.beta')
        self.connect('wave2.beta0', 'waveLoads2.beta0')
        self.connect('wave2.A0', 'waveLoads2.A0')
        self.connect('wave2.z_surface', 'waveLoads2.wlevel')
        self.connect('wave_rho', 'waveLoads2.rho')
        self.connect('wave_mu', 'waveLoads2.mu')
        self.connect('wave_cm', 'waveLoads2.cm')
        self.connect('wave_cd', 'waveLoads2.cd_usr')  # -RRD
        self.connect('geometry.z_node', 'waveLoads2.z')
        self.connect('geometry.d_node', 'waveLoads2.d')
##        self.connect('jp.z_surface','wave2.z_surface')  # from positioning component
##        self.connect('jp.z_floor','wave2.z_floor')  # from positioning component
##        self.connect('z_surface','wave2.z_surface')
##        self.connect('z_floor','wave2.z_floor')

        # connections to tower
        ##self.connect('jp.towerHeight', 'tower1.towerHeight')  # from positioning component
        self.connect('geometry.z_node', 'tower1.z')
        self.connect('geometry.d_node', 'tower1.d')
        self.connect('geometry.t_node', 'tower1.t')
        self.connect('geometry.L_reinforced_node', 'tower1.L_reinforced')
        self.connect('windLoads1.windLoads', 'tower1.windLoads')
        self.connect('waveLoads1.waveLoads', 'tower1.waveLoads')
        self.connect('soil.k', 'tower1.k_soil')
        self.connect('yaw', 'tower1.yaw')
        self.connect('top_m', 'tower1.top_m')
        self.connect('top_cm', 'tower1.top_cm')
        self.connect('top_I', 'tower1.top_I')
        self.connect('top1_F', 'tower1.top_F')
        self.connect('top1_M', 'tower1.top_M')
        self.connect('g', 'tower1.g')
        self.connect('twr_material', 'tower1.twr_material')
        self.connect('mp_material', 'tower1.mp_material')
        self.connect('MPprep.ol_material', 'tower1.ol_material')
        self.connect('geometry.idxol0', 'tower1.idxol0')
        self.connect('geometry.idxol1', 'tower1.idxol1')
        self.connect('geometry.idxtwr', 'tower1.idxtwr')


##        self.connect('G', 'tower1.G')
##        self.connect('rho', 'tower1.rho')
##        self.connect('sigma_y', 'tower1.sigma_y')
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
        ##self.connect('jp.towerHeight', 'tower2.towerHeight')  # from positioning component
        self.connect('geometry.z_node', 'tower2.z')
        self.connect('geometry.d_node', 'tower2.d')
        self.connect('geometry.t_node', 'tower2.t')
        self.connect('geometry.L_reinforced_node', 'tower2.L_reinforced')
        self.connect('windLoads2.windLoads', 'tower2.windLoads')
        self.connect('waveLoads2.waveLoads', 'tower2.waveLoads')
        self.connect('soil.k', 'tower2.k_soil')
        self.connect('yaw', 'tower2.yaw')
        self.connect('top_m', 'tower2.top_m')
        self.connect('top_cm', 'tower2.top_cm')
        self.connect('top_I', 'tower2.top_I')
        self.connect('top2_F', 'tower2.top_F')
        self.connect('top2_M', 'tower2.top_M')
        self.connect('g', 'tower2.g')
        self.connect('twr_material', 'tower2.twr_material')
        self.connect('mp_material',  'tower2.mp_material')
        self.connect('MPprep.ol_material', 'tower2.ol_material')
        self.connect('geometry.idxol0', 'tower2.idxol0')
        self.connect('geometry.idxol1', 'tower2.idxol1')
        self.connect('geometry.idxtwr', 'tower2.idxtwr')


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
##        self.connect('jp.d', 'gc.d')
##        self.connect('jp.t', 'gc.t')
        self.connect('geometry.dall', 'gc.d')
        self.connect('geometry.tall', 'gc.t')

        self.connect('min_d_to_t', 'gc.min_d_to_t')
        self.connect('min_taper', 'gc.min_taper')

##        # connections to jp
##        self.connect('sea_depth','jp.sea_depth')
##        self.connect('tower_length','jp.tower_length')
##        self.connect('tower2hub','jp.tower2hub')
##        self.connect('deck_height','jp.deck_height')
##        self.connect('Mp2MSL','jp.Mp2MSL')
##        self.connect('d_monopile','jp.d_monopile')
##        self.connect('t_monopile','jp.t_monopile')
##        self.connect('t_grout','jp.t_grout')
##        self.connect('d_tower_base','jp.d_tower_base')
##        self.connect('d_tower_top','jp.d_tower_top')
##        self.connect('t_tower_base','jp.t_tower_base')
##        self.connect('t_tower_top','jp.t_tower_top')

        #connections to Allconstraints
        self.connect('sttctrt', 'CNTRT.sttctrt')
        self.connect('endctrt', 'CNTRT.endctrt')
        self.connect('sttzctrt', 'CNTRT.sttzctrt')
        self.connect('endzctrt', 'CNTRT.endzctrt')

        self.connect('tower1.stress', 'CNTRT.stress1in')
        self.connect('tower2.stress', 'CNTRT.stress2in')
        self.connect('tower1.buckling', 'CNTRT.buckling1in')
        self.connect('tower2.buckling', 'CNTRT.buckling2in')
        self.connect('tower1.shellBuckling', 'CNTRT.shellBuckling1in')
        self.connect('tower2.shellBuckling', 'CNTRT.shellBuckling2in')
        self.connect('tower1.damage', 'CNTRT.damagein')
        self.connect('gc.weldability', 'CNTRT.weldabilityin')
        self.connect('gc.manufacturability', 'CNTRT.manufacturabilityin')


        # connections to outputs
        self.connect('tower1.mass', 'mass')
        self.connect('MPprep.MPmass',  'MPmass')
        self.connect('MPprep.TPmass',  'TPmass')
        self.connect('MPprep.Submass', 'Submass')
        self.connect('MPprep.Emdmass', 'Emdmass')


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
        self.connect('gc.manufacturability', 'manufacturability')

if __name__ == '__main__':


    optimize = False

    # --- tower setup ------
    from commonse.environment import PowerWind, TowerSoil

    tower = TowerSE()


    # ---- tower ------
    tower.replace('wind1', PowerWind())
    tower.replace('wind2', PowerWind())
    # onshore (no waves)

    tower.replace('soil', TowerSoil())
    #tower.replace('tower1', TowerWithpBEAM())
    #tower.replace('tower2', TowerWithpBEAM())
    tower.replace('tower1', TowerWithFrame3DD())
    tower.replace('tower2', TowerWithFrame3DD())

    # geometry
    tower.towerHeight = 87.6
    tower.z = np.array([0.0, 0.5, 1.0])
    tower.d = [6.0, 4.935, 3.87]
    tower.t = [0.027*1.3, 0.023*1.3, 0.019*1.3]
    tower.n = [10, 10]
    tower.L_reinforced = np.array([30., 30., 30.])  # [m] buckling length
    tower.yaw = 0.0
    tower.tilt = 5.0
    # ---------------

    '''# --- blades ---
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
    # ---------------'''

    # --- rna ---
    tower.top_m = 285598.8  # Float(iotype='in', units='m', desc='RNA (tower top) mass')
    tower.top_I = np.array([1.14930678e+08, 2.20354030e+07, 1.87597425e+07, 0.00000000e+00, 5.03710467e+05, 0.00000000e+00])  # Array(iotype='in', units='kg*m**2', desc='mass moments of inertia. order: (xx, yy, zz, xy, xz, yz)')
    tower.top_cm = np.array([-1.13197635, 0., 0.50875268])  # Array(iotype='in', units='m', desc='RNA center of mass')
    # -----------

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
    tower.wind_Uref1 = 11.73732
    tower.top1_F = np.array([1284744.19620519, 0., -2914124.84400512])  # Array(iotype='in', units='N', desc='Aerodynamic forces')
    tower.top1_M = np.array([3963732.76208099, -2275104.79420872, -346781.68192839])  # Array(iotype='in', units='N*m', desc='Aerodynamic moments')
    # ---------------

    # --- loading case 2: max wind speed ---
    tower.wind_Uref2 = 70.0
    tower.top2_F = np.array([930198.60063279, 0., -2883106.12368949])  # Array(iotype='in', units='N', desc='Aerodynamic forces')
    tower.top2_M = np.array([-1683669.22411597, -2522475.34625363, 147301.97023764])  # Array(iotype='in', units='N*m', desc='Aerodynamic moments')
    # ---------------

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
    print 'manufacturability =', tower.manufacturability
    print 'stress1 =', tower.stress1
    print 'stress1 =', tower.stress2
    print 'zs=', tower.tower1.z
    print 'ds=', tower.tower1.d
    print 'ts=', tower.tower1.t
    print 'GL buckling =', tower.buckling1
    print 'GL buckling =', tower.buckling2
    print 'Shell buckling =', tower.shellBuckling1
    print 'Shell buckling =', tower.shellBuckling2

    print 'damage =', tower.damage

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.0, 3.5))
    plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
    plt.plot(tower.stress1, tower.z_nodes, label='stress1')
    plt.plot(tower.stress2, tower.z_nodes, label='stress2')
    plt.plot(tower.shellBuckling1, tower.z_nodes, label='shell buckling 1')
    plt.plot(tower.shellBuckling2, tower.z_nodes, label='shell buckling 2')
    plt.plot(tower.buckling1, tower.z_nodes, label='global buckling 1')
    plt.plot(tower.buckling2, tower.z_nodes, label='global buckling 2')
    plt.plot(tower.damage, tower.z_nodes, label='damage')
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
        tower.driver.add_parameter('z[1]', low=0.25, high=0.75)
        tower.driver.add_parameter('d[:-1]', low=3.87, high=20.0)
        tower.driver.add_parameter('t', low=0.005, high=0.2)
        # ----------------------

        # --- recorder ---
        tower.recorders = [DumpCaseRecorder()]
        # ----------------------

        # --- Constraints ---
        tower.driver.add_constraint('tower1.stress <= 1.0')
        tower.driver.add_constraint('tower2.stress <= 1.0')
        tower.driver.add_constraint('tower1.buckling <= 1.0')
        tower.driver.add_constraint('tower2.buckling <= 1.0')
        tower.driver.add_constraint('tower1.shellBuckling <= 1.0')
        tower.driver.add_constraint('tower2.shellBuckling <= 1.0')
        tower.driver.add_constraint('tower1.damage <= 1.0')
        tower.driver.add_constraint('gc.weldability <= 0.0')
        tower.driver.add_constraint('gc.manufacturability <= 0.0')
        freq1p = 0.2  # 1P freq in Hz
        tower.driver.add_constraint('tower1.f1 >= 1.1*%f' % freq1p)
        # ----------------------

        # --- run opt ---
        tower.run()
        # ---------------

