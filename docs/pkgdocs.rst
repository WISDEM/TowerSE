
================
Package Metadata
================

- **author:** S. Andrew Ning

- **author-email:** andrew.ning@nrel.gov

- **classifier**:: 

    Intended Audience :: Science/Research
    Topic :: Scientific/Engineering

- **dependency_links:** ['https://github.com/NREL-WISDEM/pBEAM/tarball/master#egg=pBEAM-0.1.0', 'https://github.nrel.gov/sning/CommonSE/tarball/master#egg=wisdem.commonse-0.1.0']

- **description-file:** README.md

- **entry_points**:: 

    [openmdao.component]
    towerse.tower.TowerWaveDrag=towerse.tower:TowerWaveDrag
    towerse.tower.TowerDiscretization=towerse.tower:TowerDiscretization
    towerse.tower.LogWind=towerse.tower:LogWind
    towerse.tower.Wind=towerse.tower:Wind
    towerse.tower.TowerSoil=towerse.tower:TowerSoil
    towerse.tower.Tower=towerse.tower:Tower
    towerse.tower.TowerWindDrag=towerse.tower:TowerWindDrag
    towerse.tower.PowerWind=towerse.tower:PowerWind
    towerse.tower.Wave=towerse.tower:Wave
    towerse.tower.LinearWaves=towerse.tower:LinearWaves
    towerse.tower.TowerStruc=towerse.tower:TowerStruc
    towerse.tower.Soil=towerse.tower:Soil
    [openmdao.container]
    towerse.tower.TowerWaveDrag=towerse.tower:TowerWaveDrag
    towerse.tower.TowerDiscretization=towerse.tower:TowerDiscretization
    towerse.tower.AeroLoads=towerse.tower:AeroLoads
    towerse.tower.Wind=towerse.tower:Wind
    towerse.tower.TowerSoil=towerse.tower:TowerSoil
    towerse.tower.Tower=towerse.tower:Tower
    towerse.tower.TowerWindDrag=towerse.tower:TowerWindDrag
    towerse.tower.PowerWind=towerse.tower:PowerWind
    towerse.tower.Wave=towerse.tower:Wave
    towerse.tower.LinearWaves=towerse.tower:LinearWaves
    towerse.tower.Soil=towerse.tower:Soil
    towerse.tower.TowerStruc=towerse.tower:TowerStruc
    towerse.tower.LogWind=towerse.tower:LogWind

- **install_requires:** ['openmdao.main', 'pBEAM', 'wisdem.commonse']

- **keywords:** openmdao

- **license:** Apache License, Version 2.0

- **name:** towerse

- **requires-dist:** openmdao.main

- **requires-python**:: 

    >=2.6
    <3.0

- **static_path:** [ '_static' ]

- **summary:** Tower Systems Engineering Model for NREL WISDEM

- **version:** 0.1

