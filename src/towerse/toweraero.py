#!/usr/bin/env python
# encoding: utf-8
"""
toweraero.py

Created by Andrew Ning on 2012-01-20.
Copyright (c) NREL. All rights reserved.
"""

import math
import numpy as np
from scipy.optimize import brentq
from openmdao.main.api import Component, VariableTree
from openmdao.main.datatypes.api import Array, Float, VarTree

from wisdem.common import _akima  # replace this later
from utilities import sind, cosd, Vector


class AeroLoads(VariableTree):

    P = VarTree(Vector())
    q = Array()
    z = Array()
    beta = Float()



class Wind(Component):

    z = Array(iotype='in', units='m', desc='heights where wind speed was computed')

    U = Array(iotype='out', units='m/s', desc='magnitude of wind speed at each z location')
    beta = Array(iotype='out', units='deg', desc='corresponding wind angles relative to inertial coordinate system')


class Wave(Component):

    z = Array(iotype='in', units='m', desc='heights where wave speed was computed')

    U = Array(iotype='out', units='m/s', desc='magnitude of wave speed at each z location')
    A = Array(iotype='out', units='m/s**2', desc='magnitude of wave acceleration at each z location')
    beta = Array(iotype='out', units='deg', desc='corresponding wave angles relative to inertial coordinate system')



class PowerWind(Wind):

    # ---------- in -----------------
    Uref = Float(iotype='in', units='m/s', desc='reference velocity of power-law model')
    zref = Float(iotype='in', units='m', desc='corresponding reference height')
    z0 = Float(0.0, iotype='in', units='m', desc='bottom of wind profile (height of ground/sea)')
    shearExp = Float(0.2, iotype='in', desc='shear exponent')
    betaWind = Float(0.0, iotype='in', units='deg', desc='wind angle relative to inertial coordinate system')


    def execute(self):

        # rename
        z = self.z
        zref = self.zref
        z0 = self.z0

        # velocity
        self.U = np.zeros_like(z)
        idx = [z > z0]
        self.U[idx] = self.Uref*((z[idx] - z0)/(zref - z0))**self.shearExp
        self.beta = self.betaWind*np.ones_like(z)



class LogWind(Wind):

    # ---------- in -----------------
    Uref = Float(iotype='in', units='m/s', desc='reference velocity of power-law model')
    zref = Float(iotype='in', units='m', desc='corresponding reference height')
    z0 = Float(0.0, iotype='in', units='m', desc='bottom of wind profile (height of ground/sea)')
    z_roughness = Float(10.0, iotype='in', units='mm', desc='surface roughness length')
    betaWind = Float(0.0, iotype='in', units='deg', desc='wind angle relative to inertial coordinate system')


    def execute(self):

        # rename
        z = self.z
        zref = self.zref
        z0 = self.z0
        z_roughness = self.z_roughness

        # find velocity
        self.U = np.zeros_like(z)
        idx = [z > z0]
        self.U[idx] = self.Uref*(np.log((z[idx] - z0)/z_roughness) / math.log((zref - z0)/z_roughness))
        self.beta = self.betaWind*np.ones_like(z)


class LinearWaves(Wave):

    # ---------- in -------------
    hs = Float(iotype='in', units='m', desc='significant wave height (crest-to-trough)')
    T = Float(iotype='in', units='s', desc='period of waves')
    g = Float(9.81, iotype='in', units='m/s**2', desc='acceleration of gravity')
    Uc = Float(iotype='in', units='m/s', desc='mean current speed')
    betaWave = Float(0.0, iotype='in', units='deg', desc='wave angle relative to inertial coordinate system')
    z_surface = Float(iotype='in', units='m', desc='vertical location of water surface')
    z_floor = Float(0.0, iotype='in', units='m', desc='vertical location of sea floor')


    def execute(self):

        # water depth
        d = self.z_surface - self.z_floor

        # design wave height
        h = 1.1*self.hs

        # circular frequency
        omega = 2.0*math.pi/self.T

        # compute wave number from dispersion relationship
        k = brentq(lambda k: omega**2 - self.g*k*math.tanh(d*k), 0, 10*omega**2/self.g)

        # zero at surface
        z_rel = self.z - self.z_surface

        # maximum velocity
        self.U = h/2.0*omega*np.cosh(k*(z_rel + d))/math.sinh(k*d) + self.Uc

        # check heights
        self.U[np.logical_or(self.z < self.z_floor, self.z > self.z_surface)] = 0

        self.A = self.U * omega

        self.beta = self.betaWave*np.ones_like(self.z)




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



class TowerWindDrag(Component):

    rho = Float(iotype='in', units='kg/m**3', desc='air density')
    mu = Float(iotype='in', units='kg/(m*s)', desc='dynamic viscosity of air')
    U = Array(iotype='in', units='m/s', desc='magnitude of wind speed')
    z = Array(iotype='in', units='m', desc='heights where wind speed was computed')
    d = Array(iotype='in', units='m', desc='corresponding diameter of cylinder section')
    beta = Array(iotype='in', units='deg', desc='corresponding wind angles relative to inertial coordinate system')

    windLoads = VarTree(AeroLoads(), desc='wind loads in inertial coordinate system')

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

    rho = Float(iotype='in', units='kg/m**3', desc='water density')
    mu = Float(iotype='in', units='kg/(m*s)', desc='dynamic viscosity of water')
    cm = Float(2.0, iotype='in', desc='mass coefficient')
    U = Array(iotype='in', units='m/s', desc='magnitude of wave speed')
    A = Array(iotype='in', units='m/s', desc='magnitude of wave acceleration')
    z = Array(iotype='in', units='m', desc='heights where wave speed was computed')
    d = Array(iotype='in', units='m', desc='corresponding diameter of cylinder section')
    beta = Array(iotype='in', units='deg', desc='corresponding wave angles relative to inertial coordinate system')

    waveLoads = VarTree(AeroLoads(), desc='wave loads in inertial coordinate system')

    def execute(self):

        # dynamic pressure
        q = 0.5*self.rho*self.U**2

        # Reynolds number and drag
        Re = self.rho*self.U*self.d/self.mu
        cd = cylinderDrag(Re)

        # inertial and drag forces
        Fi = self.rho*self.cm*math.pi/4.0*self.d**2*self.A
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





if __name__ == '__main__':

    wind = PowerWind()

    wind.z = np.linspace(0, 20)

    wind.beta = 0.0
    wind.Uref = 15.0
    wind.zref = 10.0
    wind.z0 = 3.0
    wind.shearExp = 0.2


    wind.run()

    import matplotlib.pyplot as plt
    plt.plot(wind.V.x, wind.z)
    plt.show()


    wave = LinearWaves()
    wave.hs = 5.0
    wave.T = 10.0
    wave.z = np.linspace(0, 20)
    wave.Uc = 3.0
    wave.z_surface = 10.0
    wave.z_floor = 0.0

    wave.run()

    import matplotlib.pyplot as plt
    plt.plot(wave.V.x, wave.z)
    plt.show()
