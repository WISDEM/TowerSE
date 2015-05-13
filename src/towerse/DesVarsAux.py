#-------------------------------------------------------------------------------
# Name:        DesVarsAux.py
# Purpose:     It constains classes that can be used in various forms, optimization included for TowerSE and JacketSE.
#
# Author:      rdamiani
#
# Created:     25/11/2014
# Copyright:   (c) rdamiani 2014
# Licence:     <APACHE 2014>
#-------------------------------------------------------------------------------
import numpy as np
import collections
from jacketse.VarTrees import RNAprops

def main():
    pass


class TwrDesVars(object): #Design Variables Grouped in a structure
    """Class containing the values of the optimization design variables."""
#   x 0     1     2         3     4       5      6     7     8
#   Db,DTRb,   Dt,    DTRt,     H2frac,  DMP , tMP   ,DTP,  tTP
    def __init__(self,**kwargs):

        dvars=collections.OrderedDict([ ('Db',5.),('DTRb',120.),('Dt',3.),('DTRt',120.),('Dw',5.),('DTRw',120.),('Htwr2frac',0.25),('DMP',6.),('tMP',0.06),('DTP',6.2),('tTP',0.05) ]) #SI Units
        dvarunits=collections.OrderedDict([('Db','[m]'),   ('DTRb','[-]'),('Dt','[m]'),('DTRt','[-]'),('Dw','[m]'),('DTRw','[-]'),('Htwr2frac','[-]'),('DMP','[m]'),('tMP','[m]'),('DTP','[m]'),('tTP','[m]') ]) #SI Units

        dvars.update(kwargs) #update in case user put some new params in

        self.dvars=dvars  #This allows me to save the orer of the design variables for loops elsewhere
        self.dvarunits=dvarunits  #This allows me to save the orer of the design variables units for loops elsewhere

        for key in dvars:
            setattr(self,key,dvars[key])

#__________________________________________________________#

class TwrDesVarBounds(TwrDesVars): #Design Variable Bounds
    """Class containing the bounds to the optimization design variables."""
    def __init__(self,**kwargs):

        prms={'Db':np.array([5.,7.]),'DTRb':np.array([120.,200.]),'Dt':np.array([3.,4.5]),'DTRt':np.array([120.,200.]),'Dw':np.array([5.,7.]),'DTRw':np.array([120.,200.]),\
              'Htwr2frac':np.array([0.,0.25]),'DMP':np.array([6.,7.]),'tMP':np.array([2*0.0254,4*0.0254]),\
               'DTP':np.array([6.2,7.23]),'tTP':np.array([2*0.0254,4*0.0254])} #SI Units

        prms.update(kwargs) #update in case user put some new params in
        for key in prms:  #Initialize material object with these parameters, possibly updated by user' stuff
            setattr(self,key,prms[key])

#__________________________________________________________#

class TwrDesPrms(object): #Design Parameters
    """Class containing the Main Design Parameters (won't be optimized) that may change from case to case."""

    def __init__(self,**kwargs):


        prms={'TurbRating':3., 'wdepth':0.,'HW50':30,'Tp50':12.,'HW50_2':30,'Tp50_2':12.,'dck_botz':0.,'HH':100.,'U50HH':30.,'U50HH_2':70., \
              'RNA_F':np.array([1.5*1700.e3,0.,0.,12564863.93,0.,0.]),'RNA_F2':np.array([700.e3,0.,0.,0.,0.,0.]),'RNAins':RNAprops(),'f0':0.35} #SI Units excpet for MW for the rating and legndiv

        prms.update(kwargs) #update in case user put some new params in
        for key in prms:  #Initialize material object with these parameters, possibly updated by user' stuff
            setattr(self,key,prms[key])

#__________________________________________________________#

if __name__ == '__main__':
    main()
