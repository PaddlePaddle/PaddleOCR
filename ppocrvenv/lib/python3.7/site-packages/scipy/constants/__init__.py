r"""
==================================
Constants (:mod:`scipy.constants`)
==================================

.. currentmodule:: scipy.constants

Physical and mathematical constants and units.


Mathematical constants
======================

================  =================================================================
``pi``            Pi
``golden``        Golden ratio
``golden_ratio``  Golden ratio
================  =================================================================


Physical constants
==================

===========================  =================================================================
``c``                        speed of light in vacuum
``speed_of_light``           speed of light in vacuum
``mu_0``                     the magnetic constant :math:`\mu_0`
``epsilon_0``                the electric constant (vacuum permittivity), :math:`\epsilon_0`
``h``                        the Planck constant :math:`h`
``Planck``                   the Planck constant :math:`h`
``hbar``                     :math:`\hbar = h/(2\pi)`
``G``                        Newtonian constant of gravitation
``gravitational_constant``   Newtonian constant of gravitation
``g``                        standard acceleration of gravity
``e``                        elementary charge
``elementary_charge``        elementary charge
``R``                        molar gas constant
``gas_constant``             molar gas constant
``alpha``                    fine-structure constant
``fine_structure``           fine-structure constant
``N_A``                      Avogadro constant
``Avogadro``                 Avogadro constant
``k``                        Boltzmann constant
``Boltzmann``                Boltzmann constant
``sigma``                    Stefan-Boltzmann constant :math:`\sigma`
``Stefan_Boltzmann``         Stefan-Boltzmann constant :math:`\sigma`
``Wien``                     Wien displacement law constant
``Rydberg``                  Rydberg constant
``m_e``                      electron mass
``electron_mass``            electron mass
``m_p``                      proton mass
``proton_mass``              proton mass
``m_n``                      neutron mass
``neutron_mass``             neutron mass
===========================  =================================================================


Constants database
------------------

In addition to the above variables, :mod:`scipy.constants` also contains the
2018 CODATA recommended values [CODATA2018]_ database containing more physical
constants.

.. autosummary::
   :toctree: generated/

   value      -- Value in physical_constants indexed by key
   unit       -- Unit in physical_constants indexed by key
   precision  -- Relative precision in physical_constants indexed by key
   find       -- Return list of physical_constant keys with a given string
   ConstantWarning -- Constant sought not in newest CODATA data set

.. data:: physical_constants

   Dictionary of physical constants, of the format
   ``physical_constants[name] = (value, unit, uncertainty)``.

Available constants:

======================================================================  ====
%(constant_names)s
======================================================================  ====


Units
=====

SI prefixes
-----------

============  =================================================================
``yotta``     :math:`10^{24}`
``zetta``     :math:`10^{21}`
``exa``       :math:`10^{18}`
``peta``      :math:`10^{15}`
``tera``      :math:`10^{12}`
``giga``      :math:`10^{9}`
``mega``      :math:`10^{6}`
``kilo``      :math:`10^{3}`
``hecto``     :math:`10^{2}`
``deka``      :math:`10^{1}`
``deci``      :math:`10^{-1}`
``centi``     :math:`10^{-2}`
``milli``     :math:`10^{-3}`
``micro``     :math:`10^{-6}`
``nano``      :math:`10^{-9}`
``pico``      :math:`10^{-12}`
``femto``     :math:`10^{-15}`
``atto``      :math:`10^{-18}`
``zepto``     :math:`10^{-21}`
============  =================================================================

Binary prefixes
---------------

============  =================================================================
``kibi``      :math:`2^{10}`
``mebi``      :math:`2^{20}`
``gibi``      :math:`2^{30}`
``tebi``      :math:`2^{40}`
``pebi``      :math:`2^{50}`
``exbi``      :math:`2^{60}`
``zebi``      :math:`2^{70}`
``yobi``      :math:`2^{80}`
============  =================================================================

Mass
----

=================  ============================================================
``gram``           :math:`10^{-3}` kg
``metric_ton``     :math:`10^{3}` kg
``grain``          one grain in kg
``lb``             one pound (avoirdupous) in kg
``pound``          one pound (avoirdupous) in kg
``blob``           one inch version of a slug in kg (added in 1.0.0)
``slinch``         one inch version of a slug in kg (added in 1.0.0)
``slug``           one slug in kg (added in 1.0.0)
``oz``             one ounce in kg
``ounce``          one ounce in kg
``stone``          one stone in kg
``grain``          one grain in kg
``long_ton``       one long ton in kg
``short_ton``      one short ton in kg
``troy_ounce``     one Troy ounce in kg
``troy_pound``     one Troy pound in kg
``carat``          one carat in kg
``m_u``            atomic mass constant (in kg)
``u``              atomic mass constant (in kg)
``atomic_mass``    atomic mass constant (in kg)
=================  ============================================================

Angle
-----

=================  ============================================================
``degree``         degree in radians
``arcmin``         arc minute in radians
``arcminute``      arc minute in radians
``arcsec``         arc second in radians
``arcsecond``      arc second in radians
=================  ============================================================


Time
----

=================  ============================================================
``minute``         one minute in seconds
``hour``           one hour in seconds
``day``            one day in seconds
``week``           one week in seconds
``year``           one year (365 days) in seconds
``Julian_year``    one Julian year (365.25 days) in seconds
=================  ============================================================


Length
------

=====================  ============================================================
``inch``               one inch in meters
``foot``               one foot in meters
``yard``               one yard in meters
``mile``               one mile in meters
``mil``                one mil in meters
``pt``                 one point in meters
``point``              one point in meters
``survey_foot``        one survey foot in meters
``survey_mile``        one survey mile in meters
``nautical_mile``      one nautical mile in meters
``fermi``              one Fermi in meters
``angstrom``           one Angstrom in meters
``micron``             one micron in meters
``au``                 one astronomical unit in meters
``astronomical_unit``  one astronomical unit in meters
``light_year``         one light year in meters
``parsec``             one parsec in meters
=====================  ============================================================

Pressure
--------

=================  ============================================================
``atm``            standard atmosphere in pascals
``atmosphere``     standard atmosphere in pascals
``bar``            one bar in pascals
``torr``           one torr (mmHg) in pascals
``mmHg``           one torr (mmHg) in pascals
``psi``            one psi in pascals
=================  ============================================================

Area
----

=================  ============================================================
``hectare``        one hectare in square meters
``acre``           one acre in square meters
=================  ============================================================


Volume
------

===================    ========================================================
``liter``              one liter in cubic meters
``litre``              one liter in cubic meters
``gallon``             one gallon (US) in cubic meters
``gallon_US``          one gallon (US) in cubic meters
``gallon_imp``         one gallon (UK) in cubic meters
``fluid_ounce``        one fluid ounce (US) in cubic meters
``fluid_ounce_US``     one fluid ounce (US) in cubic meters
``fluid_ounce_imp``    one fluid ounce (UK) in cubic meters
``bbl``                one barrel in cubic meters
``barrel``             one barrel in cubic meters
===================    ========================================================

Speed
-----

==================    ==========================================================
``kmh``               kilometers per hour in meters per second
``mph``               miles per hour in meters per second
``mach``              one Mach (approx., at 15 C, 1 atm) in meters per second
``speed_of_sound``    one Mach (approx., at 15 C, 1 atm) in meters per second
``knot``              one knot in meters per second
==================    ==========================================================


Temperature
-----------

=====================  =======================================================
``zero_Celsius``       zero of Celsius scale in Kelvin
``degree_Fahrenheit``  one Fahrenheit (only differences) in Kelvins
=====================  =======================================================

.. autosummary::
   :toctree: generated/

   convert_temperature

Energy
------

====================  =======================================================
``eV``                one electron volt in Joules
``electron_volt``     one electron volt in Joules
``calorie``           one calorie (thermochemical) in Joules
``calorie_th``        one calorie (thermochemical) in Joules
``calorie_IT``        one calorie (International Steam Table calorie, 1956) in Joules
``erg``               one erg in Joules
``Btu``               one British thermal unit (International Steam Table) in Joules
``Btu_IT``            one British thermal unit (International Steam Table) in Joules
``Btu_th``            one British thermal unit (thermochemical) in Joules
``ton_TNT``           one ton of TNT in Joules
====================  =======================================================

Power
-----

====================  =======================================================
``hp``                one horsepower in watts
``horsepower``        one horsepower in watts
====================  =======================================================

Force
-----

====================  =======================================================
``dyn``               one dyne in newtons
``dyne``              one dyne in newtons
``lbf``               one pound force in newtons
``pound_force``       one pound force in newtons
``kgf``               one kilogram force in newtons
``kilogram_force``    one kilogram force in newtons
====================  =======================================================

Optics
------

.. autosummary::
   :toctree: generated/

   lambda2nu
   nu2lambda

References
==========

.. [CODATA2018] CODATA Recommended Values of the Fundamental
   Physical Constants 2018.

   https://physics.nist.gov/cuu/Constants/

"""
# Modules contributed by BasSw (wegwerp@gmail.com)
from .codata import *
from .constants import *
from .codata import _obsolete_constants

_constant_names = [(_k.lower(), _k, _v)
                   for _k, _v in physical_constants.items()
                   if _k not in _obsolete_constants]
_constant_names = "\n".join(["``%s``%s  %s %s" % (_x[1], " "*(66-len(_x[1])),
                                                  _x[2][0], _x[2][1])
                             for _x in sorted(_constant_names)])
if __doc__:
    __doc__ = __doc__ % dict(constant_names=_constant_names)

del _constant_names

__all__ = [s for s in dir() if not s.startswith('_')]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
