#-------------------------------------------------------------------------------
# Name:        MyTowerInputs.py
# Purpose:     This module is  a template to set up a Tower Assembly with basic input & optimization parameters (to be set (harwired here))
#              THIS IS JUST A TEMPLATE:
#              in order to modify this: 1. Copy this file and rename as 'yourinputfile.py'
#                                       2. Edit all the tower inputs till the line that prohibits to further edit
#                                       3. Launch TowerSEOpt_Py&MDAOopt.py pointing to 'yourinputfile.py'
# Author:      rdamiani
#
# Created:     11/24/2014
# Copyright:   (c) rdamiani 2014
# Licence:     Apache (2014)
#-------------------------------------------------------------------------------
import numpy as np
from scipy.interpolate import interp1d

from openmdao.main.api import set_as_top

from commonse.environment import PowerWind, TowerSoil, TowerSoilK, LinearWaves
from commonse.Material import Material

from towerse.tower import TowerSE,TowerWithFrame3DD

from jacketse.VarTrees import Frame3DDaux


def main(): #\
    """Function to Instantiate a TowerSE Assembly: \n
       INPUTS \n
             All hardwired, so edit the quantities below all the way to the line "#________________ DO NOT MODIFY THE FOLLOWING ________________#" \n
             -See TowerSEOpt_Py&MDAOopt.py for more information. \n
       OUTPUTS \n
             mytwr -tower assembly instance \n\n

             Optimization parameters:    \n\n

             f0          -float, target frequency [Hz]
             f0epsilon   -float,  f0*(1+f0epsilon) will not be exceeded \n
             guesses     -Float(n), guesses for all design variables check out DesVar class. \n
             bounds      -Float(n,2), bounds for all design variables check out DesVar class. \n\n
             SAMPLE CALLS: \n
             1.OPTIMIZATION: python towerOpt_ExtCobyla.py C:\RRD\PYTHON\WISDEM\towerSE\src\towerse\MyTowerInputs.py \n
             2.OPTIMIZATION: python TowerSEOpt_Py&MDAOopt.py C:\RRD\PYTHON\WISDEM\towerSE\src\towerse\MytowerInputs.py True \n
             3.BUILD Tower: python >>> mytwr=C:\RRD\PYTHON\WISDEM\TowerSE\src\towerse\MyTowerInputs.py \n
        """

    # I need to put this at the top as it is not set up nicely as jacket: do not modify next line and go to inputs below
    mytwr=set_as_top(TowerSE())


    # __________Environment___________#

    mytwr.replace('wind1', PowerWind())
    mytwr.replace('wind2', PowerWind())

    # wind
    towerToShaft = 2.0
    mytwr.wind_zref = mytwr.towerHeight + towerToShaft
    mytwr.wind_z0 = 0.0
    mytwr.wind1.shearExp = 0.2
    mytwr.wind2.shearExp = 0.2

    # waves
    sea_depth = 20.0
    mytwr.replace('wave1', LinearWaves())
    mytwr.replace('wave2', LinearWaves())

    mytwr.wave1.Uc = 0.0
    mytwr.wave1.hs = 8.0 * 1.86
    mytwr.wave1.T = 10.0
    mytwr.wave1.z_surface = 0.0
    mytwr.wave1.z_floor = -sea_depth
    mytwr.wave1.g = 9.81
    mytwr.wave1.betaWave = 0.0

    mytwr.wave2.Uc = 0.0
    mytwr.wave2.hs = 8.0 * 1.86
    mytwr.wave2.T = 10.0
    mytwr.wave2.z_surface = 0.0
    mytwr.wave2.z_floor = -sea_depth
    mytwr.wave2.g = 9.81
    mytwr.wave2.betaWave = 0.0

    # __________Soil___________#
    mytwr.replace('soil', TowerSoil())

    mytwr.soil.rigid = 6*[True]

    # OR DIRECTLY IMPORT THE STIFFNESS CONSTANTS

    #mytwr.replace('soil', TowerSoilK())  #Provide soil constants bypassing the TowerSoil Component
    #mytwr.soil.kin = np.array([5.8e7,4.4e10,5.8e7,4.4e10,1.3E9,6.46e9])#


    # __________Frame3DD or PBeam___________#
    #mytwr.replace('mytwr1', mytwrWithpBEAM())
    #mytwr.replace('mytwr2', mytwrWithpBEAM())
    mytwr.replace('tower1', TowerWithFrame3DD())
    mytwr.replace('tower2', TowerWithFrame3DD())

    # __________Material___________#
    mytwr.material=Material(matname='heavysteel',E=2.1e11,nu=0.33)

    # __________Geometry___________#
    mytwr.towerHeight = 115.63
    mytwr.z = np.array([0.0, 0.5, 1.0])
    mytwr.d = [6.0, 4.935, 3.87]
    mytwr.t = [0.027*1.3, 0.023*1.3, 0.019*1.3]
    mytwr.n = [10, 10]
    mytwr.n_reinforced = 3
    mytwr.L_reinforced=np.array([30.])#,30.,30.]) #[m] buckling length
    mytwr.yaw = 0.0
    mytwr.tilt = 5.0
    # monopile
    mytwr.monopileHeight = sea_depth
    mytwr.n_monopile = 5
    mytwr.d_monopile = 6.0
    mytwr.t_monopile = 6.0/80.0

    #________RNA mass Properties_________#
    mytwr.top_m = 285598.8 #Float(iotype='in', units='m', desc='RNA (tower top) mass')
    mytwr.top_I = np.array([1.14930678e+08, 2.20354030e+07, 1.87597425e+07, 0.00000000e+00, 5.03710467e+05, 0.00000000e+00]) #Array(iotype='in', units='kg*m**2', desc='mass moments of inertia. order: (xx, yy, zz, xy, xz, yz)')
    mytwr.top_cm = np.array([-1.13197635, 0., 0.50875268]) #Array(iotype='in', units='m', desc='RNA center of mass')

    #_______________Loads________________#

    # max Thrust case
    mytwr.wind_Uref1 = 11.73732
    mytwr.top1_F = np.array([1284744.19620519, 0., -2914124.84400512]) #Array(iotype='in', units='N', desc='Aerodynamic forces')
    mytwr.top1_M = np.array([3963732.76208099, -2275104.79420872, -346781.68192839]) #Array(iotype='in', units='N*m', desc='Aerodynamic moments')

    # max wind speed case
    mytwr.wind_Uref2 = 70.0
    mytwr.top2_F = np.array([930198.60063279, 0., -2883106.12368949]) #Array(iotype='in', units='N', desc='Aerodynamic forces')
    mytwr.top2_M = np.array([-1683669.22411597, -2522475.34625363, 147301.97023764]) #Array(iotype='in', units='N*m', desc='Aerodynamic moments')

    # fatigue
    mytwr.z_DEL = np.array([0.000]) #, 1.327, 3.982, 6.636, 9.291, 11.945, 14.600, 17.255, 19.909, 22.564, 25.218, 27.873, 30.527, 33.182, 35.836, 38.491, 41.145, 43.800, 46.455, 49.109, 51.764, 54.418, 57.073, 59.727, 62.382, 65.036, 67.691, 70.345, 73.000, 75.655, 78.309, 80.964, 83.618, 86.273, 87.600])
    mytwr.M_DEL = 1e3*np.array([0.0]) #8.2940E+003, 8.1518E+003, 7.8831E+003, 7.6099E+003, 7.3359E+003, 7.0577E+003, 6.7821E+003, 6.5119E+003, 6.2391E+003, 5.9707E+003, 5.7070E+003, 5.4500E+003, 5.2015E+003, 4.9588E+003, 4.7202E+003, 4.4884E+003, 4.2577E+003, 4.0246E+003, 3.7942E+003, 3.5664E+003, 3.3406E+003, 3.1184E+003, 2.8977E+003, 2.6811E+003, 2.4719E+003, 2.2663E+003, 2.0673E+003, 1.8769E+003, 1.7017E+003, 1.5479E+003, 1.4207E+003, 1.3304E+003, 1.2780E+003, 1.2673E+003, 1.2761E+003])
    mytwr.gamma_fatigue = 1.35*1.3*1.0
    mytwr.life = 20.0
    mytwr.m_SN = 4

    # Frame3DD parameters
    FrameAuxIns=Frame3DDaux()
    FrameAuxIns.sh_fg=1               #shear flag-->Timoshenko
    FrameAuxIns.deltaz=5.
    FrameAuxIns.geo_fg=0
    FrameAuxIns.nModes = 6             # number of desired dynamic modes of vibration
    FrameAuxIns.Mmethod = 1            # 1: subspace Jacobi     2: Stodola
    FrameAuxIns.lump = 0               # 0: consistent mass ... 1: lumped mass matrix
    FrameAuxIns.tol = 1e-9             # mode shape tolerance
    FrameAuxIns.shift = 0.0            # shift value ... for unrestrained structures
    FrameAuxIns.gvector=np.array([0.,0.,-9.8065])    #GRAVITY


    #_____ Safety Factors______#
    mytwr.gamma_f = 1.35
    mytwr.gamma_m = 1.3
    mytwr.gamma_n = 1.0
    mytwr.gamma_b = 1.1

    #______________________________________________#
    #______________________________________________#

# OTHER AUXILIARY CONSTRAINTS AND TARGETS FOR OPTIMIZATION #
    #______________________________________________#
    #______________________________________________#

    # _______Geometric constraints__________#

    mytwr.min_taper = 0.4

    DTRsdiff = True  #Set whether or not DTRt=DTRb

    #______Set target frequency [Hz] and f0epsilon, i.e. fmax=(1+f0eps)*f0_________#
    f0=0.28
    f0epsilon=0.1

    #________Set Optimization Bounds and guesses for the various variables_________#
    #          x=  [      Db,   DTRb   Dt,   DTRt   Htwr2fac  ]
    MnCnst = np.array([  5.,   120.,  3.,   120.,     0.05  ])
    MxCnst = np.array([  7.,   200.,  4.,   200.,     0.25])
    guesses= np.array([  6.,   140.,  3.5,  150.,     0.2 ])


   #_____________________________________________________________#
   #________________ DO NOT MODIFY THE FOLLOWING ________________#
   #_____________________________________________________________#

    mytwr.min_d_to_t = np.min(MnCnst[[1,3]])
    bounds=np.vstack((MnCnst,MxCnst))
    desvarmeans=np.mean(bounds,1)

    mytwr.FrameAuxIns=FrameAuxIns

    return mytwr,f0,f0epsilon,DTRsdiff,guesses,bounds.T



if __name__ == '__main__':
    from PlotTower import main as PlotTower  #COMMENT THIS ONE OUT FOR PEREGRINE"S SAKE

    mytwr= main()[0]
    #--- RUN JACKET ---#
    mytwr.run()
    # ---------------- #

    #_____________________________________#
    #Now show results of modal analysis
    print('First two Freqs.= {:5.4f} and {:5.4f} Hz'.format(mytwr.tower1.f1,mytwr.tower1.f2))
    #print component masses

    print('tower mass [kg] = {:6.0f}'.format(mytwr.mass))

    #print tower top displacement
    print('Tower Top Displacement (load cases 1 and 2) in Global Coordinate System [m] ={:5.4f} & {:5.4f}'.format(mytwr.top_deflection1,mytwr.top_deflection2))
    #print max Utilizations
    print('MAX GL buckling = {:5.4f}'.format(np.max((mytwr.buckling1,mytwr.buckling2))))
    print('MAX Shell buckling = {:5.4f}'.format(np.max((mytwr.shellBuckling1,mytwr.shellBuckling2))))


    PlotTower(mytwr,util=True)