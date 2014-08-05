TowerSE
=======

Systems Engineering Model of Wind Turbine Towers (OpenMDAO plugin)

To install (for development purposes, the packaged version does all these steps for you):

1) activate OpenMDAO

    source openmdao/bin/activate

2) install CommonSE 
    
    git clone https://github.nrel.gov/sning/CommonSE
    cd CommonSE
    plugin install

3) install pBEAM (requires C compiler and Boost)

    plugin install -f https://github.com/NREL-WISDEM/pBEAM/tarball/master#egg=pBEAM-0.1.0 pBEAM

4) install TowerSE
    
    git clone https://github.nrel.gov/sning/TowerSE
    cd TowerSE
    plugin install
