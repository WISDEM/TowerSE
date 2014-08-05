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
import os

from openmdao.main.api import Assembly
from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Bool, Dict,Instance
from openmdao.lib.drivers.api import COBYLAdriver, pyOptDriver
from openmdao.lib.casehandlers.api import DumpCaseRecorder
from openmdao.main.api import enable_console
#enable_console()
import logging
logging.getLogger().setLevel(logging.DEBUG)

#from jacket import JacketAsmly,JcktGeoInputs,LegGeoInputs,XBrcGeoInputs,MudBrcGeoInputs,HBrcGeoInputs,PileGeoInputs,TPGeoInputs,TwrGeoInputs,RNAprops,TPlumpMass,WaterInputs,WindInputs,SoilGeoInputs,Frame3DDaux,IEC_PSFS
import SetTowerInputs

def main(recordfile,SNOPTflag=False):

    #1.Set bounds for optimization variables

    #          x=  [     H2frac,  OD    t,    ]
    MnCnst=np.array([    0.,      3.,   0.005 ])
    MxCnst=np.array([    0.5,     20.,  0.2,  ])
    avgcnst=(MnCnst+MxCnst)/2.
    factors=1./avgcnst

    #2. Set up main variables and create assembly with main inputs via SetJacketInputs.py
    mytwr=SetTowerInputs.main(avgcnst[0],avgcnst[1],avgcnst[2],avgcnst[3],avgcnst[4],avgcnst[5],avgcnst[6],avgcnst[7], avgcnst[8],avgcnst[9],avgcnst[10],avgcnst[11],avgcnst[12],avgcnst[13],avgcnst[14])


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
    mytwr.driver.add_constraint('tower1.stress <= 1.0')
    mytwr.driver.add_constraint('tower2.stress <= 1.0')
    mytwr.driver.add_constraint('tower1.buckling <= 1.0')
    mytwr.driver.add_constraint('tower2.buckling <= 1.0')
    mytwr.driver.add_constraint('tower1.shellBuckling <= 1.0')
    mytwr.driver.add_constraint('tower2.shellBuckling <= 1.0')
    mytwr.driver.add_constraint('tower1.damage <= 1.0')
    mytwr.driver.add_constraint('gc.weldability <= 0.0')
    mytwr.driver.add_constraint('gc.manufactuability <= 0.0')

    mytwr.driver.add_constraint('tower1.f1 >= %f' % f0)

    #7. recorder

    #if (isinstance(recordfile,str) and (os.path.exists(recordfile) or  os.access(os.path.dirname(recordfile), os.W_OK))):
    fileID=open(recordfile,'a')
    mytwr.driver.recorders = [DumpCaseRecorder(fileID),DumpCaseRecorder()]

    import time
    tt = time.time()

    #8. run
    mytwr.run()

    print "\n"
    print "Minimum found at Db=%f, Dt=%f, tb=%f, tt=%f; mass= (%f)" % (mytwr.tower1.d[0],mytwr.tower1.d[-1],mytwr.tower1.t[0],mytwr.tower1.t[-1],mytwr.tower1.mass)
    print "Minimum found at DTRb DTRt(%f, %f)" % (mytwr.tower1.d[0]/mytwr.tower1.t[0],mytwr.tower1.d[-1]/mytwr.tower1.t[-1])
    print "Minimum found at Freq %f"  % (mytwr.tower1.f1)
    print "Minimum found at GLutil EUutil %f %f"  % (np.max(np.vstack((mytwr.tower1.buckling,mytwr.tower2.buckling))),np.max(np.vstack((mytwr.tower1.shellBuckling,mytwr.tower2.shellBuckling))))
    print "Elapsed time: ", time.time()-tt, "seconds"
    print "Execution count: ", mytwr.exec_count


    #9. store data for future
    import cPickle as pickle;
    pickle.dump(res,open("C:\PROJECTS\OFFSHORE_WIND\AML\PYTHON\TwrOptdata.p","w"))

    #10. recall data
    res=pickle.load(open("C:\PROJECTS\OFFSHORE_WIND\AML\PYTHON\TwrOptdata.p","r"))
    fileID.close()

if __name__ == '__main__':
    #This is how you call this function
    main('twropt_test.txt',False)
