# TowerSE Changelog

## 0.2.0 ([04/01/2019])

[Garrett Barter](mailto: garrett.barter@nrel.gov)

- OpenMDAO1 release

## 0.1.0 ([09/30/2014])

[Andrew Ning](mailto: aning@byu.edu)

- initial release

## 0.1.1 ([11/13/2014])

[Katherine Dykes](mailto: katherine.dykes@nrel.gov)

[CHANGE]:

- new import of rna mass properties and rotor loads module from CommonSE (was in TowerSE) due to common use by TowerSE and JacketSE

- updates to inputs on materials to use materials from CommonSE (were specified as individual variables)

- updated to use "utilization" rather than "constraint" nomenclature

## 0.1.2 ([12/04/2014])

[Katherine Dykes](mailto: katherine.dykes@nrel.gov)

[CHANGE]:

- updating tower to use new material fy variable instead of f_y

- fixing some "utilization" variable issues

- updating tower main section for new input set so that tower model main executes

## 0.1.3 ([12/18/2014])

[Andrew Ning](mailto: aning@byu.edu)

[CHANGE]:

- updated gradients to match the changes in 0.1.2 so all unit tests should pass


## 0.1.4 ([12/04/2014])

[Katherine Dykes](mailto: katherine.dykes@nrel.gov)

[NEW]:

- added TowerMonopileSE assembly and JacketPositioning component for crude representation of tower/jacket/monopile substructure

[CHANGE]:

- updated gradient tests to include test for new JacketPositioning component

## 0.1.5 ([07/08/2015])

[Andrew Ning](mailto: aning@byu.edu)

[CHANGE]:

- significant changes made to overall tower.py structure: eliminated use of pBEAM (now only using frame3DD), eliminated separation of monopile from overall tower specification, restructured i/o for main tower assembly
