.. _documentation-label:

.. currentmodule:: towerse.tower

Documentation
--------

.. autoclass:: TowerSE

.. literalinclude:: ../src/towerse/tower.py
    :language: python
    :start-after: TowerSE(Assembly)
    :end-before: def configure(self)
    :prepend: class TowerSE(Assembly):

=============== =========== =============== =========== ========================================================================
name            type        default value   units       description
=============== =========== =============== =========== ========================================================================
min_d_to_t      Float       120.0
towerHeight     Float       0.0             m
hub_mass        Float       0.0             kg          mass of hub
z_DEL           Array
wind_z0         Float       0.0             m           bottom of wind profile (height of ground/sea)
DC              Float       80.0                        standard value of stress
L_reinforced    Float       0.0             m
t_monopile      Float       0.0             m
hub_cm          Array                       m           location of hub center of mass relative to tower top in yaw-aligned c.s.
rotorM1         Array                                   moments in hub-aligned coordinate system
rotorM2         Array                                   moments in hub-aligned coordinate system
m_SN            Int         4                           slope of S/N curve
rotorT2         Float       0.0                         thrust in hub-aligned coordinate system
rotorT1         Float       0.0                         thrust in hub-aligned coordinate system
gamma_fatigue   Float       1.755                       total safety factor for fatigue
sigma_y         Float       450000000.0     N/m**2      yield stress
gamma_n         Float       1.0                         safety factor on consequence of failure
printvars       List        []                          List of extra variables to output in the recorders.
nac_I           Array                       kg*m**2     mass moments of inertia of nacelle about its center of mass
gamma_m         Float       1.1                         safety factor on materials
tilt            Float       0.0             deg
yaw             Float       0.0             deg
gamma_f         Float       1.35                        safety factor on loads
gamma_b         Float       1.1                         buckling safety factor
n_monopile      Int         0                           must be a minimum of 1 (top and bottom)
wind_mu         Float       1.7934e-05      kg/(m*s)    dynamic viscosity of air
wave_rho        Float       1027.0          kg/m**3     water density
rho             Float       8500.0          kg/m**3     material density
hub_I           Array                       kg*m**2     mass moments of inertia of hub about its center of mass
wind_Uref1      Float       0.0             m/s         reference wind speed (usually at hub height)
rotorQ1         Float       0.0                         torque in hub-aligned coordinate system
rotorQ2         Float       0.0                         torque in hub-aligned coordinate system
wind_Uref2      Float       0.0             m/s         reference wind speed (usually at hub height)
E               Float       2.1e+11         N/m**2      material modulus of elasticity
nac_mass        Float       0.0             kg          mass of nacelle
G               Float       80800000000.0   N/m**2      material shear modulus
rotorF1         Array                                   forces in hub-aligned coordinate system
M_DEL           Array
rotorF2         Array                                   forces in hub-aligned coordinate system
downwind        Bool        False
monopileHeight  Float       0.0             m
blades_mass     Float       0.0             kg          mass of all blade
wind_rho        Float       1.225           kg/m**3     air density
blades_I        Array                       kg*m**2     mass moments of inertia of all blades about hub center
d_monopile      Float       0.0             m
wave_cm         Float       2.0                         mass coefficient
d               Array                       m           tower diameter at corresponding locations
g               Float       9.81            m/s**2
life            Float       20.0                        fatigue life of tower
min_taper       Float       0.4
n               Array                                   number of finite elements between sections.  array length should be ``len(z)-1``
nac_cm          Array                       m           location of nacelle center of mass relative to tower top in yaw-aligned c.s.
t               Array                       m           shell thickness at corresponding locations
wave_mu         Float       0.0013351       kg/(m*s)    dynamic viscosity of water
z               Array                                   locations along unit tower, linear lofting between
wind_zref       Float       0.0             m           corresponding reference height
=============== =========== =============== =========== ========================================================================
