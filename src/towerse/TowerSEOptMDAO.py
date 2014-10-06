#-------------------------------------------------------------------------------
# Name:        TowerSEOptMDAO.py
# Purpose:     It solves for minimum mass problem with constraints on max utilization and frequency.
#              It uses Andrew's TowerSE and works within the OPENMDAO framework for the optimizer as well.
#              The main routine can be called with two inputs: a filename for the recorder, and a flag to indicate whether snopt (True) or Cobyla(false) will be used.
# Author:      rdamiani
#
# Created:     7/2014 based off of JacketOptMDAO.py
# Copyright:   (c) rdamiani 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import scipy.optimize
from math import pi             # >>>
import os

from openmdao.main.api import Assembly, set_as_top
from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Bool, Dict,Instance
from openmdao.lib.drivers.api import COBYLAdriver
from pyopt_driver.pyopt_driver import pyOptDriver # >>>

from openmdao.lib.casehandlers.api import JSONCaseRecorder,DumpCaseRecorder
from openmdao.main.api import enable_console
#enable_console()
import logging
logging.getLogger().setLevel(logging.DEBUG) # >>> (?)

#from jacket import JacketAsmly,JcktGeoInputs,LegGeoInputs,XBrcGeoInputs,MudBrcGeoInputs,HBrcGeoInputs,PileGeoInputs,TPGeoInputs,TwrGeoInputs,RNAprops,TPlumpMass,WaterInputs,WindInputs,SoilGeoInputs,Frame3DDaux,IEC_PSFS
from towerse.tower import TowerSE       # >>>
from commonse.environment import PowerWind, TowerSoil   # >>>
from towerse.tower import TowerWithFrame3DD             # >>>


def main(recordfile,SNOPTflag=False):

    #1.Set bounds for optimization variables

    #          x=  [     H2frac,  OD    t,    ]
    MnCnst=np.array([    0.01,      4.,  0.020 ])
    MxCnst=np.array([    0.2,     9.,  0.050 ])
    avgcnst=(MnCnst+MxCnst)/2.
    factors=1./avgcnst

    #2. Set up main variables and create assembly with main inputs via SetTowerInputs.py
    #
    # ** should SetTowerInputs get the input file as an argument???
    # to call the function
    #
    mytwr = set_as_top(TowerSE()) # >> creates instance of tower assembly,(!) maybe make it "top"
    #execfile(r'C:\PROJECTS\OFFSHORE_WIND\AML\twrInputs_maintower.py')
    execfile(r'C:\PROJECTS\OFFSHORE_WIND\AML\twrInputs_FSGtower.py')
    #execfile(r'D:\RRD_ENGINEERING\PROJECTS\NREL\OFFSHOREWIND\AML\twrInputs_maintower.py')
##    mytwr=SetTowerInputs.main(avgcnst[0],avgcnst[1],avgcnst[2],avgcnst[3],avgcnst[4],avgcnst[5],avgcnst[6],avgcnst[7], avgcnst[8],avgcnst[9],avgcnst[10],avgcnst[11],avgcnst[12],avgcnst[13],avgcnst[14])

    mytwr.run()   # >>> (!) I can't yet run the assembly
    print 'mass =', mytwr.mass
    print 'f1 =', mytwr.f1
    print 'f2 =', mytwr.f2
    print 'top_deflection =', mytwr.top_deflection1
    print 'top_deflection =', mytwr.top_deflection2
    print 'stress =', mytwr.stress1
    print 'stress =', mytwr.stress2
    print 'zs=', mytwr.tower1.z
    print 'ds=', mytwr.tower1.d
    print 'ts=', mytwr.tower1.t
    print 'global buckling =', mytwr.buckling1
    print 'global buckling =', mytwr.buckling2
    print 'shell_buckling =', mytwr.shellBuckling1
    print 'shell_buckling =', mytwr.shellBuckling2

    print 'damage =', mytwr.damage

   #3. Replace driver in main assembly and specify optimizer parameters
    if not(SNOPTflag):
        mytwr.replace('driver',COBYLAdriver())
    else:
        mytwr.replace('driver',pyOptDriver())
        mytwr.driver.optimizer = 'SNOPT'
        mytwr.driver.options = {'Major feasibility tolerance': 1e-6,
                               'Minor feasibility tolerance': 1e-6,
                               'Major optimality tolerance': 1e-5,
                               'Function precision': 1e-8}
    mytwr.driver.iprint = 1
    #mytwr.driver.rhoend = 0.1
    mytwr.driver.rhobeg=0.1
    mytwr.driver.disp=1

    #4. Objective and target frequency
    mytwr.driver.add_objective('tower1.mass / 500.e3')
    f0 = 0.25  # Target Frequency

    #5. Design Variables

    mytwr.driver.add_parameter('z[1]',     low=MnCnst[0], high=MxCnst[0])  #This H2
    mytwr.driver.add_parameter('d',        low=MnCnst[1], high=MxCnst[1])   #This is OD at three stations
    mytwr.driver.add_parameter('t',        low=MnCnst[2], high=MxCnst[2])   #This is t at three stations

    #6. Constraitns
    mytwr.driver.add_constraint('tower1.stress <= 0.0')
    mytwr.driver.add_constraint('tower2.stress <= 0.0')
    mytwr.driver.add_constraint('tower1.buckling <= 0.0')
    mytwr.driver.add_constraint('tower2.buckling <= 0.0')
    mytwr.driver.add_constraint('tower1.shellBuckling <= 0.0')
    mytwr.driver.add_constraint('tower2.shellBuckling <= 0.0')
    mytwr.driver.add_constraint('tower1.damage <= 1.0')
    mytwr.driver.add_constraint('gc.weldability <= 0.0')
    mytwr.driver.add_constraint('gc.manufactuability <= 0.0')

    mytwr.driver.add_constraint('tower1.f1 >= %f' % (0.95*f0))
    mytwr.driver.add_constraint('tower1.f1 <= %f' % (1.05*f0))
    #7. recorder

    #if (isinstance(recordfile,str) and (os.path.exists(recordfile) or  os.access(os.path.dirname(recordfile), os.W_OK))):
    fileID=open(recordfile,'a')
    #mytwr.driver.recorders = [DumpCaseRecorder(fileID),DumpCaseRecorder()]
    mytwr.recorders = [JSONCaseRecorder(fileID)]
    import time
    tt = time.time()

    #8. run
    mytwr.run()

    print "\n"
    print "Minimum found at Db=%f, Dt=%f, tb=%f, tt=%f; mass= (%f)" % (mytwr.tower1.d[0],mytwr.tower1.d[-1],mytwr.tower1.t[0],mytwr.tower1.t[-1],mytwr.tower1.mass)
    print "Minimum found at z1=%f, D1=%f, t1=%f" % (mytwr.tower1.z[1],mytwr.tower1.d[1],mytwr.tower1.t[1])
    print "Minimum found at DTRb DTRt(%f, %f)" % (mytwr.tower1.d[0]/mytwr.tower1.t[0],mytwr.tower1.d[-1]/mytwr.tower1.t[-1])
    print "Minimum found at Freq %f"  % (mytwr.tower1.f1)
    print "Minimum found at GLutil EUutil %f %f"  % (np.max(np.vstack((mytwr.tower1.buckling,mytwr.tower2.buckling))),np.max(np.vstack((mytwr.tower1.shellBuckling,mytwr.tower2.shellBuckling))))
    print "Minimum found at GLutil 1 and 2"  , mytwr.tower1.buckling,mytwr.tower2.buckling
    print "Minimum found at EUutil 1 and 2"  , mytwr.tower1.shellBuckling,mytwr.tower2.shellBuckling
    print "Elapsed time: ", time.time()-tt, "seconds"
    print "Execution count: ", mytwr.exec_count
    fileID.close()

    #9. store data for future
##    import cPickle as pickle;
##    pickle.dump(mytwr,open("C:\PROJECTS\OFFSHORE_WIND\AML\PYTHON\TwrOptdata.p","w"))
##
##    #10. recall data
##    mytwr=pickle.load(open("C:\PROJECTS\OFFSHORE_WIND\AML\PYTHON\TwrOptdata.p","r"))


if __name__ == '__main__':
    #This is how you call this function
    main('C:\PROJECTS\OFFSHORE_WIND\AML\SNOPTrecorderBaseline.txt',True)
