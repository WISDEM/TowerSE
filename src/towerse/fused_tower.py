
# TowerSE model set
from tower_components import TowerDiscretization, TowerMass, TurbineMass, TowerPreFrame, TowerPostFrame

# FUSED helper functions and interface defintions
from fusedwind.fused_wind import FUSED_Object
from fusedwind.windio_plant_costs import fifc_aep, fifc_tcc_costs, fifc_bos_costs, fifc_opex, fifc_finance

import numpy as np

### FUSED-wrapper file 
class TowerDiscretization_fused(FUSED_Object):
    def __init__(self):
        super(TowerDiscretization_fused, self).__init__()

        