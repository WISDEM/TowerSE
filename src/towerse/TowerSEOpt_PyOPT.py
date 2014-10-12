#-------------------------------------------------------------------------------
# Name:        TowerSEOpt_PyOPT.py variant of JacketOpt_PyOPT.py for tower only
# Purpose:     It Expects to read data from input file and then calls the optimizer based on that.
#              It runs only one case (not a table of cases).
#              It all started from JacketOpt_OldStyle.py.
#                     NOTE HERE DESVARS ARE NOT MADE DIMENSIOLESS!!! JUST objfunc and cosntraints! vs JacketOPt_Peregrine
#              This routine will execute the optimization from outside the OpenMdao framework.
# Author:      rdamiani
#
# Created:     7/2014 based off of JacketOpt_PyOPT.py
# Copyright:   (c) rdamiani 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import scipy.optimize

import pyOpt

MPIFlag=True  #INitialize
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
except:
    MPIFlag=False
    #raise ImportError('mpi4py is required for parallelization')

import time
import sys
import os

from PlotJacket import main as PlotJacket  #COMMENT THIS ONE OUT FOR PEREGRINE"S SAKE

#__________________________________________________________#

class TwrDesVars(object): #Design Variables Grouped in a structure
    """Class containing the values of the optimization design variables."""
#   x 0     1     2         3     4
#   Db,DTRb,   Dt,    DTRt,     H2frac,
    def __init__(self,**kwargs):

        dvars=collections.OrderedDict([ ('Db',5.),('DTRb',120.),('Dt',3.),('DTRt',120.),('Htwr2frac',0.25) ]) #SI Units
        dvarunits=collections.OrderedDict([('Db','[m]'),   ('DTRb','[-]'),('Dt','[m]'),('DTRt','[-]'),('Htwr2frac','[-]') ]) #SI Units

        dvars.update(kwargs) #update in case user put some new params in

        self.dvars=dvars  #This allows me to save the orer of the design variables for loops elsewhere
        self.dvarunits=dvarunits  #This allows me to save the orer of the design variables units for loops elsewhere

        for key in dvars:
            setattr(self,key,dvars[key])

#__________________________________________________________#

class TwrDesVarBounds(TwrDesVars): #Design Variable Bounds
    """Class containing the bounds to the optimization design variables."""
    def __init__(self,**kwargs):

        prms={'Db':np.array([5.,7.]),'DTRb':np.array([120.,200.]),'Dt':np.array([3.,4.5]),'DTRt':np.array([120.,200.]),\
              'Htwr2frac':np.array([0.,0.25])\
               } #SI Units

        prms.update(kwargs) #update in case user put some new params in
        for key in prms:  #Initialize material object with these parameters, possibly updated by user' stuff
            setattr(self,key,prms[key])

#__________________________________________________________#

class TwrDesPrms(object): #Design Parameters
    """Class containing the Main Design Parameters (won't be optimized) that may change from case to case."""

    def __init__(self,**kwargs):


        prms={'TurbRating':3., 'wdepth':20.,'HW50':30,'Tp50':12.,'dck_botz':16.,'HH':100.,'U50HH':70., \
              'RNA_F':np.array([1.5*1700.e3,0.,0.,12564863.93,0.,0.]),'RNAins':RNAprops(),'f0':0.35} #SI Units excpet for MW for the rating and legndiv

        prms.update(kwargs) #update in case user put some new params in
        for key in prms:  #Initialize material object with these parameters, possibly updated by user' stuff
            setattr(self,key,prms[key])

#__________________________________________________________#


#__________________________________________________________#


              #  OPTIMIZATION FUNCTION STARTS HERE #

#__________________________________________________________#

def twrOptReadInput(twrInputs,resfile,desbds,f0epsilon=0.05,SNOPTflag=False):
    """Optimizer which reads and sets inputs from a file:\n
    INPUT \n
        twrInputs -string, complete path+filename to tower input file. \n
        resfile   -string, complete path+filename to tower output file. \n
        desbds    -array, design variables bounds. \n
    OPTIONALS \n
        f0epsilon -float, (1+f0eps)*f0 will be the upper bound for the f0 constraint. \n
        SNOPTflag -boolean, if True SNOPT will be used
    OUTPUT \n
        It will create a final configuration optimized for the design parameters in the twrInputs. \n
        Also, it will save text and excel file for design variables, and summary for report, respectively. \n
        Figures will also be made of configuration and tower utilization. (not in peregrine where you can then use reconall.py \n

        mytwr   -OPenMdao assembly of JacketSE, final configuration after optimization
        \n
        """
    #This is the optimizer that will act on 1 case from a table
    global f0,f0eps, DTRsdiff
    import SetTowerInputs
    import ntpath

    desvars=TwrDesVars() #instance of design variables
    desprms=TwrDesPrms() #instance of design params
    desvarmeans=np.mean(desbds,0)


    #Then set up an initial guess at the Design variables and use it also to instantiate the assembly

    for ii,key in enumerate(desvars.dvars):
        #print key  #debug
        setattr(desvars,key,desvarmeans[ii])

    #Then build the initial assembly
    mytwr=SetTowerInputs.main(desprms,desvars)

#   x       0     1       2       3         4
#             Db,  DTRb,   Dt,    DTRt,     H2frac,
    #guess=[  6.8,  124., 3.5,    124.,      0.2]
    guess=desvarmeans

    varlist=desvars.dvars.keys()
    if not(DTRsdiff):#DTRs FIXED TO EACH OTHER
        idx=varlist.index('DTRt')
        varlist.pop(idx)
        guess=np.delete(guess,idx)
        desvarmeans=np.delete(desvarmeans,idx)
        desvarbds=np.delete(desvarbds,idx,0)


    #Target frequency
    f0=desprms.f0
    f0eps=f0epsilon


    #SET UP THE PROBLEM
    opt_prob=pyOpt.Optimization('Tower Optimization ', objfunc)

    opt_prob.addObj('mass')

    for ii,key in enumerate(varlist):
        opt_prob.addVar(key,'c',lower=desvarbds[ii,0],upper=desvarbds[ii,1],value=guess[ii])

    opt_prob.addConGroup('cnstrts',5,type='i')


    print opt_prob

    #Finally call the optimizer
    args=(mytwr,desvarmeans,desvarbds)


    if SNOPTflag:
        opt_prob.write2file(outfile=os.path.join(os.path.dirname(casefile),'pyopt_snopt_'+str(caseno).zfill(2)+'.hst'), disp_sols=False, solutions=[])
        #set some strings for MPI incase
        mpistr=''
        printfstr='pyopt_snopt_print_'
        summfstr='pyopt_snopt_summary_'
        if MPIFlag:
            mpistr='pgc'
            printfstr='pyopt_mpisnopt_print_'
            summfstr='pyopt_mpisnopt_summary_'

        opt =pyOpt.SNOPT()  #non-MPI here always
        opt.setOption('Minor print level',1)
        opt.setOption('Major feasibility tolerance',1.e-3)
        opt.setOption('Major optimality tolerance',1.e-3)
        opt.setOption('Minor feasibility tolerance',1.e-3)
        opt.setOption('Print file',os.path.join(os.path.dirname(twrInputs),printfstr+str(caseno).zfill(2)+'.out'))
        opt.setOption('Summary file',os.path.join(os.path.dirname(twrInputs),summfstr+str(caseno).zfill(2)+'.out'))
        opt.setOption('Solution','Yes')
        #Solve
        tt = time.time()

        [fstr, xstr, inform]=opt(opt_prob,'FD',True,True,True,False,mpistr,{},*args)  #for parallel gradient calculations
        #
        #opt_problem={}, sens_type='FD', store_sol=True, disp_opts=False, store_hst=False, hot_start=False, sens_mode='', sens_step={}, *args, **kwargs)

    #COBYLA
    else:
        mpistr=None
        ifilestr='pyopt_cobyla_'
        if MPIFlag:
            mpistr='POA'
            ifilestr='pyopt_mpicobyla_'

        opt =pyOpt.COBYLA(pll_type=mpistr)

        #opt.setOption('RHOBEG',0.01)
        opt.setOption('RHOEND',1.e-3)
        opt.setOption('MAXFUN',2000)
        opt.setOption('IPRINT',1)
        opt.setOption('IFILE',os.path.join(os.path.dirname(twrInputs),ifilestr+str(caseno).zfill(2)+'.hst') ) #store cobyla output
        [fstr, xstr, inform]=opt(opt_prob,  True,           False,             True,          False, *args)
    ###opt_problem={}, store_sol=True, disp_opts=False, store_hst=False, hot_start=False


    print opt_prob.solution(0)

    print "\n"
    print "Minimum mass Tower Mass = %f " %(mytwr.mass)
    print "Minimum found at Db=%f DTRb=%f Dt=%f DTRt=%f H2frac=%f " % (mytwr.Tower.Twrins.Db,mytwr.Tower.Twrins.DTRb,mytwr.Tower.Twrins.Dt,mytwr.Tower.Twrins.DTRt,mytwr.Tower.Twrins.Htwr2frac)
    print "Minimum found at Freq %f"  % (mytwr.Frameouts2.Freqs[0])
    print "Minimum found at GLutil=%f EUutil=%f"  % (np.nanmax(mytwr.tower_utilization.GLUtil),np.nanmax(mytwr.tower_utilization.EUshUtil))
    print "Elapsed time: ", time.time()-tt, "seconds"
    print "Execution count: ", mytwr.exec_count

    #STORE RESULTS
    if not(DTRsdiff):#DTRs FIXED TO EACH OTHER, reexpand array
        idx_DTRb=desvars.dvars.keys().index('DTRb')
        xstr=np.insert(xstr,idx_DTRt,xstr[idx_DTRb])

    #Plot
    #PlotJacket(mytwr,util=True,savefileroot=outdir+'\\'+casename)

    return mytwr
#__________________________________________________________#



def TwrWrapper(x,mytwr,desvarmeans,desvarbds):

    """This function builds up the actual model and calculates Jacket stresses and utilization.
    INPUT
        x         -list(N), as in DesVars with that order, but ALL NORMALIZED BY THEIR AVERAGES!!!! \n
        mytwr    -assembly of TowerSE.\n
        desvarmeans -array(N), average values for each design variable.\n
        """
    global DTRsdiff
    global  xlast,f1,max_GLUtil,max_EUUtil

#   x 0     1     2         3         4
#     Db,  DTRb,   Dt,      DTRt,     H2frac,

    mytwr.d[0]     =x[0]#  Db
    mytwr.t[0]     =x[0]/x[1]# Db/DTRb
    mytwr.d[-1]     =x[2]# Dt
    mytwr.t[-1]     =x[2]/x[1]# Dt/DTRb  assuming DTRb=DTRt
    if DTRsdiff:
        mytwr.t[-1]     =x[2]/x[3]# Dt/DTRt

    mytwr.z[1]      =x[4-int(DTRsdiff)]# H2frac

    #Run the assembly and get main output
    mytwr.run()

    #Get Frame3dd mass
    mass=mytwr.tower1.mass  #Total structural mass
    #Get Frame3dd-calculated 1st natural frequency
    f1=mytwr.tower1.f1

    #Get Utilizations
    max_GLUtil=np.nanmax(np.vstack((mytwr.tower1.buckling, mytwr.tower2.buckling)))
    max_EUUtil=np.nanmax(np.vstack((mytwr.tower1.shellBuckling, mytwr.tower2.shellBuckling)))

    #Pile Embedment Criteria
    #Lp0rat=mytwr.Lp0rat
    #__________________________________________#

    #calc width at seabed proportional to stiffness

    xlast=x.copy() #update the latest set of input params to the current


    print('from TwrWrapper Db={:5.2f}, DTRb={:5.2f}, Dt={:5.2f}, DTRt={:5.2f},H2twrfrac={:5.2f},  Twrmass={:6.3f}'.\
            format(mytwr.tower1.d[0],mytwr.tower1.d[0]/tower1.t[0],mytwr.tower1.d[-1],\
                   mytwr.tower1.d[-1]/tower1.t[-1],mytwr.tower1.z[1],mytwr.tower1.mass))
    print(' \n')

    sys.stdout.flush()  #This for peregrine
    return mass,f1,max_GLUtil,max_EUUtil


#__________________#
#   OBJ FUNCTION   #
#__________________#

def objfunc(x,mytwr,desvarmeans,desvarbds):

    mass = TwrWrapper(x,mytwr,desvarmeans,desvarbds)[0]/300.e3

    cnstrts=[0.0]*4 #given as negatives (at the end), since PYOPT wants <0
    cnstrts[0]=f0Cnstrt1(x,mytwr,desvarmeans,desvarbds)
    cnstrts[1]=f0Cnstrt2(x,mytwr,desvarmeans,desvarbds)
    cnstrts[2]=GLCnstrt(x,mytwr,desvarmeans,desvarbds)
    cnstrts[3]=EUCnstrt(x,mytwr,desvarmeans,desvarbds)
    #cnstrts[4]=LpCnstrt(x,mytwr,desvarmeans,desvarbds)

    cnstrts =-cnstrts   #since PYOPT wants <0 !!!!!!!!

    fail=0
    return mass,cnstrts,fail

#__________________#
#    CONSTRAINTS   #
#__________________#


def f0Cnstrt1(x,mytwr,desvarmeans,desvarbds):  #f1>f0
    global xlast,f1

    if xlast==None or np.any([x != xlast]):
        #print('call TwrWrapper from const')
        mass,f1,max_GLUtil,max_EUUtil=TwrWrapper(x,mytwr,desvarmeans,desvarbds)
        xlast=x.copy()
    cnstrt=(f1-f0)/f0
    print('f0Cnstrt1=',cnstrt)
    return cnstrt

def f0Cnstrt2(x,mytwr,desvarmeans,desvarbds): #f1<(f0*(1+f0eps))
    global xlast,f1,f0eps

    if xlast==None or np.any([x != xlast]):
        #print('call TwrWrapper from const')
        mass,f1,max_GLUtil,max_EUUtil=TwrWrapper(x,mytwr,desvarmeans,desvarbds)

        xlast=x.copy()
    cnstrt=(-f1+f0*(1+f0eps))/f0
    print('f0Cnstrt2=',cnstrt)
    return cnstrt


def GLCnstrt(x,mytwr,desvarmeans,desvarbds): #GLUtil<1
    global xlast,max_GLUtil

    if xlast==None or np.any([x != xlast]):
        #print('call TwrWrapper from const')
        mass,f1,max_GLUtil,max_EUUtil=TwrWrapper(x,mytwr,desvarmeans,desvarbds)

        xlast=x.copy()
    cnstrt=1.-max_GLUtil
    print('GL constraint=',cnstrt)
    return cnstrt

def EUCnstrt(x,mytwr,desvarmeans,desvarbds): #EUUtil<1
    global xlast,max_EUUtil

    if xlast==None or np.any([x != xlast]):
        #print('call TwrWrapper from const')
        mass,f1,max_GLUtil,max_EUUtil=TwrWrapper(x,mytwr,desvarmeans,desvarbds)

        xlast=x.copy()
    cnstrt=1.-max_EUUtil
    print('EU constraint=',cnstrt)
    return cnstrt



 #Tower constraints
def TwrCnstrt01(x,mytwr,desvarmeans,desvarbds): #Maximum tower Diameter
    idx=12
    cnstrt=maxcnstrt(x,idx,desvarmeans,desvarbds)
    print('Db TwrCnstrt01=',cnstrt )
    return cnstrt
def TwrCnstrt02(x,mytwr,desvarmeans,desvarbds): #Maximum tower Diameter at the top
    idx=14
    cnstrt=maxcnstrt(x,idx,desvarmeans,desvarbds)
    print('Dt TwrCnstrt02=', cnstrt)
    return cnstrt
def TwrCnstrt03(x,mytwr,desvarmeans,desvarbds): #Minimum tower Diameter at the top
    idx=14
    cnstrt=mincnstrt(x,idx,desvarmeans,desvarbds)
    print('Dt TwrCnstrt03=', cnstrt)
    return cnstrt
def TwrCnstrt04(x,mytwr,desvarmeans,desvarbds):  #Minimum DTRb>120
    idx=13
    cnstrt=mincnstrt(x,idx,desvarmeans,desvarbds)
    #print('DTR min TwrCnstrt04=', cnstrt)
    return cnstrt
def TwrCnstrt05(x,mytwr,desvarmeans,desvarbds):  #Max DTRb<200
    idx=13
    cnstrt=maxcnstrt(x,idx,desvarmeans,desvarbds)
    #print('DTR max TwrCnstrt05=',cnstrt)
    return cnstrt
def TwrCnstrt06(x,mytwr,desvarmeans,desvarbds):  #Minimum DTRt>120
    idx=15
    cnstrt=mincnstrt(x,idx,desvarmeans,desvarbds)
    #print('DTR min TwrCnstrt06=', cnstrt)
    return cnstrt
def TwrCnstrt07(x,mytwr,desvarmeans,desvarbds):  #Max DTRt<200
    idx=15
    cnstrt=maxcnstrt(x,idx,desvarmeans,desvarbds)
    #print('DTR max TwrCnstrt07='cnstrt)
    return cnstrt
def TwrCnstrt08(x,mytwr,desvarmeans,desvarbds):  #Maximum Htwr2 < Htwr/4
    global    DTRsdiff
    idx=15+int(DTRsdiff)
    cnstrt=maxcnstrt(x,idx,desvarmeans,desvarbds)
    print('Htwr2 max TwrCnstrt08=',cnstrt)
    return cnstrt
def TwrCnstrt09(x,mytwr,desvarmeans,desvarbds):  #Minimum Htwr2 >0.005
    global    DTRsdiff
    idx=15+int(DTRsdiff)
    cnstrt=mincnstrt(x,idx,desvarmeans,desvarbds)
    print('Htwr2 min TwrCnstrt09=',cnstrt)
    return cnstrt



#Embedment length constraint
def LpCnstrt(x,mytwr,desvarmeans,desvarbds):  #Maximum Htwr2 < Htwr/4
    global xlast,Lp0rat
    if xlast==None or np.any([x != xlast]):
        #print('call TwrWrapper from const')
        mass,f1,max_tutil,max_cbutil,max_KjntUtil,max_XjntUtil,max_GLUtil,max_EUUtil,\
        MudCrit01,MudCrit02,MudCrit03,MudCrit04,MudCrit05,\
        XBrcCrit01,XBrcCrit02,XBrcCrit03,XBrcCrit04,XBrcCrit05,Lp0rat=TwrWrapper(x,mytwr,desvarmeans,desvarbds)
        xlast=x.copy()
    print('Lp0rat constraint=',Lp0rat)
    return Lp0rat

#______________________________________________________________________________#

# AUXILIARY FUNCTIONS
def maxcnstrt(x,idx,desvarmeans,desvarbds):
    return (-x[idx]+desvarbds[idx,1])/desvarmeans[idx]
def mincnstrt(x,idx,desvarmeans,desvarbds):
    return  (x[idx]-desvarbds[idx,0])/desvarmeans[idx]


#______________________________________________________________________________#

def main(inpfile,resfile,desbds=desbds,f0eps=0.05,SNOPTflag=False):
    global    DTRsdiff

    DTRsdiff=False    ##SET THIS TO TRUE IF YOU WANT DTRs to be different between base and top
    f0epsilon=f0eps #upper f0 allowed

    #Default values of bounds for des vars
    MnCnst=np.array([    5.,  120.,     3.,  120., 0.005 ])
    MxCnst=np.array([    7.5, 200.,     4.,  200.,  0.25 ])
    desbds=np.vstack((MnCnst,MxCnst))

    inpfile=sys.argv[1]
    if len(sys.argv)>2:
        resfile=sys.argv[2]
        SNOPTflag=False
        if len(sys.argv)>3:
            SNOPTflag= (sys.argv[3].lower() == 'true')

    else:
        inpfile=r'C:\PROJECTS\OFFSHORE_WIND\AML\twrinputs.inp'
        resfile=r'C:\PROJECTS\OFFSHORE_WIND\AML\teoweropt.out'
        SNOPTflag=False


    print('#____________________________________________#\n')
    print(('#TowerSEOpt_PYOPT NOW PROCESSING CASE ') ) #print(('#TowerSEOpt_PYOPT NOW PROCESSING CASE #\n').format(caseno) )
    print('#____________________________________________#\n')

    mytwr=twrOptReadInput(inpfile,resfile,desbds,f0epsilon=f0epsilon, SNOPTflag=SNOPTflag)

    sys.stdout.flush()  #This for peregrine

    ##SAMPLE CALL FROM OUTSIDE ENVIRONMENT
    ##python TowerSEOpt_PyOpt.py Twrinputs AMLopt.out False




if __name__ == '__main__':

    #START FROM SETTING BOUNDS of DEISGN VARIABLES
                        #Db,  DTRb,,  Dt, DTRt, H2frac
    MnCnst=np.array([    5.,  120.,     3.,  120., 0.005 ])
    MxCnst=np.array([    7.5, 200.,     4.,  200.,  0.25 ])
    avgcnst=(MnCnst+MxCnst)/2.
    factors=1./avgcnst

    inpfile=r'C:\PROJECTS\OFFSHORE_WIND\AML\twrinputs.inp'
    resfile=r'C:\PROJECTS\OFFSHORE_WIND\AML\teoweropt.out'
    SNOPTflag=False
    f0epsilon=0.05
    desbds=np.vstack((MnCnst,MxCnst))

    main(inpfile,resfile,desbds,f0eps=f0epsilon,SNOPTflag=SNOPTflag)
