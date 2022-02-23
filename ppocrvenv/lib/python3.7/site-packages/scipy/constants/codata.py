# Compiled by Charles Harris, dated October 3, 2002
# updated to 2002 values by BasSw, 2006
# Updated to 2006 values by Vincent Davis June 2010
# Updated to 2014 values by Joseph Booker, 2015
# Updated to 2018 values by Jakob Jakobson, 2019

"""
Fundamental Physical Constants
------------------------------

These constants are taken from CODATA Recommended Values of the Fundamental
Physical Constants 2018.

Object
------
physical_constants : dict
    A dictionary containing physical constants. Keys are the names of physical
    constants, values are tuples (value, units, precision).

Functions
---------
value(key):
    Returns the value of the physical constant(key).
unit(key):
    Returns the units of the physical constant(key).
precision(key):
    Returns the relative precision of the physical constant(key).
find(sub):
    Prints or returns list of keys containing the string sub, default is all.

Source
------
The values of the constants provided at this site are recommended for
international use by CODATA and are the latest available. Termed the "2018
CODATA recommended values," they are generally recognized worldwide for use in
all fields of science and technology. The values became available on 20 May
2019 and replaced the 2014 CODATA set. Also available is an introduction to the
constants for non-experts at

https://physics.nist.gov/cuu/Constants/introduction.html

References
----------
Theoretical and experimental publications relevant to the fundamental constants
and closely related precision measurements published since the mid 1980s, but
also including many older papers of particular interest, some of which date
back to the 1800s. To search the bibliography, visit

https://physics.nist.gov/cuu/Constants/

"""
import warnings
from math import pi, sqrt

__all__ = ['physical_constants', 'value', 'unit', 'precision', 'find',
           'ConstantWarning']

"""
Source:  https://physics.nist.gov/cuu/Constants/

The values of the constants provided at this site are recommended for
international use by CODATA and are the latest available. Termed the "2018
CODATA recommended values," they are generally recognized worldwide for use in
all fields of science and technology. The values became available on 20 May
2019 and replaced the 2014 CODATA set.
"""

#
# Source:  https://physics.nist.gov/cuu/Constants/
#

# Quantity                                             Value                 Uncertainty          Unit
# ---------------------------------------------------- --------------------- -------------------- -------------
txt2002 = """\
Wien displacement law constant                         2.897 7685e-3         0.000 0051e-3         m K
atomic unit of 1st hyperpolarizablity                  3.206 361 51e-53      0.000 000 28e-53      C^3 m^3 J^-2
atomic unit of 2nd hyperpolarizablity                  6.235 3808e-65        0.000 0011e-65        C^4 m^4 J^-3
atomic unit of electric dipole moment                  8.478 353 09e-30      0.000 000 73e-30      C m
atomic unit of electric polarizablity                  1.648 777 274e-41     0.000 000 016e-41     C^2 m^2 J^-1
atomic unit of electric quadrupole moment              4.486 551 24e-40      0.000 000 39e-40      C m^2
atomic unit of magn. dipole moment                     1.854 801 90e-23      0.000 000 16e-23      J T^-1
atomic unit of magn. flux density                      2.350 517 42e5        0.000 000 20e5        T
deuteron magn. moment                                  0.433 073 482e-26     0.000 000 038e-26     J T^-1
deuteron magn. moment to Bohr magneton ratio           0.466 975 4567e-3     0.000 000 0050e-3
deuteron magn. moment to nuclear magneton ratio        0.857 438 2329        0.000 000 0092
deuteron-electron magn. moment ratio                   -4.664 345 548e-4     0.000 000 050e-4
deuteron-proton magn. moment ratio                     0.307 012 2084        0.000 000 0045
deuteron-neutron magn. moment ratio                    -0.448 206 52         0.000 000 11
electron gyromagn. ratio                               1.760 859 74e11       0.000 000 15e11       s^-1 T^-1
electron gyromagn. ratio over 2 pi                     28 024.9532           0.0024                MHz T^-1
electron magn. moment                                  -928.476 412e-26      0.000 080e-26         J T^-1
electron magn. moment to Bohr magneton ratio           -1.001 159 652 1859   0.000 000 000 0038
electron magn. moment to nuclear magneton ratio        -1838.281 971 07      0.000 000 85
electron magn. moment anomaly                          1.159 652 1859e-3     0.000 000 0038e-3
electron to shielded proton magn. moment ratio         -658.227 5956         0.000 0071
electron to shielded helion magn. moment ratio         864.058 255           0.000 010
electron-deuteron magn. moment ratio                   -2143.923 493         0.000 023
electron-muon magn. moment ratio                       206.766 9894          0.000 0054
electron-neutron magn. moment ratio                    960.920 50            0.000 23
electron-proton magn. moment ratio                     -658.210 6862         0.000 0066
magn. constant                                         12.566 370 614...e-7  0                     N A^-2
magn. flux quantum                                     2.067 833 72e-15      0.000 000 18e-15      Wb
muon magn. moment                                      -4.490 447 99e-26     0.000 000 40e-26      J T^-1
muon magn. moment to Bohr magneton ratio               -4.841 970 45e-3      0.000 000 13e-3
muon magn. moment to nuclear magneton ratio            -8.890 596 98         0.000 000 23
muon-proton magn. moment ratio                         -3.183 345 118        0.000 000 089
neutron gyromagn. ratio                                1.832 471 83e8        0.000 000 46e8        s^-1 T^-1
neutron gyromagn. ratio over 2 pi                      29.164 6950           0.000 0073            MHz T^-1
neutron magn. moment                                   -0.966 236 45e-26     0.000 000 24e-26      J T^-1
neutron magn. moment to Bohr magneton ratio            -1.041 875 63e-3      0.000 000 25e-3
neutron magn. moment to nuclear magneton ratio         -1.913 042 73         0.000 000 45
neutron to shielded proton magn. moment ratio          -0.684 996 94         0.000 000 16
neutron-electron magn. moment ratio                    1.040 668 82e-3       0.000 000 25e-3
neutron-proton magn. moment ratio                      -0.684 979 34         0.000 000 16
proton gyromagn. ratio                                 2.675 222 05e8        0.000 000 23e8        s^-1 T^-1
proton gyromagn. ratio over 2 pi                       42.577 4813           0.000 0037            MHz T^-1
proton magn. moment                                    1.410 606 71e-26      0.000 000 12e-26      J T^-1
proton magn. moment to Bohr magneton ratio             1.521 032 206e-3      0.000 000 015e-3
proton magn. moment to nuclear magneton ratio          2.792 847 351         0.000 000 028
proton magn. shielding correction                      25.689e-6             0.015e-6
proton-neutron magn. moment ratio                      -1.459 898 05         0.000 000 34
shielded helion gyromagn. ratio                        2.037 894 70e8        0.000 000 18e8        s^-1 T^-1
shielded helion gyromagn. ratio over 2 pi              32.434 1015           0.000 0028            MHz T^-1
shielded helion magn. moment                           -1.074 553 024e-26    0.000 000 093e-26     J T^-1
shielded helion magn. moment to Bohr magneton ratio    -1.158 671 474e-3     0.000 000 014e-3
shielded helion magn. moment to nuclear magneton ratio -2.127 497 723        0.000 000 025
shielded helion to proton magn. moment ratio           -0.761 766 562        0.000 000 012
shielded helion to shielded proton magn. moment ratio  -0.761 786 1313       0.000 000 0033
shielded helion gyromagn. ratio                        2.037 894 70e8        0.000 000 18e8        s^-1 T^-1
shielded helion gyromagn. ratio over 2 pi              32.434 1015           0.000 0028            MHz T^-1
shielded proton magn. moment                           1.410 570 47e-26      0.000 000 12e-26      J T^-1
shielded proton magn. moment to Bohr magneton ratio    1.520 993 132e-3      0.000 000 016e-3
shielded proton magn. moment to nuclear magneton ratio 2.792 775 604         0.000 000 030
{220} lattice spacing of silicon                       192.015 5965e-12      0.000 0070e-12        m"""

txt2006 = """\
lattice spacing of silicon                             192.015 5762 e-12     0.000 0050 e-12       m
alpha particle-electron mass ratio                     7294.299 5365         0.000 0031
alpha particle mass                                    6.644 656 20 e-27     0.000 000 33 e-27     kg
alpha particle mass energy equivalent                  5.971 919 17 e-10     0.000 000 30 e-10     J
alpha particle mass energy equivalent in MeV           3727.379 109          0.000 093             MeV
alpha particle mass in u                               4.001 506 179 127     0.000 000 000 062     u
alpha particle molar mass                              4.001 506 179 127 e-3 0.000 000 000 062 e-3 kg mol^-1
alpha particle-proton mass ratio                       3.972 599 689 51      0.000 000 000 41
Angstrom star                                          1.000 014 98 e-10     0.000 000 90 e-10     m
atomic mass constant                                   1.660 538 782 e-27    0.000 000 083 e-27    kg
atomic mass constant energy equivalent                 1.492 417 830 e-10    0.000 000 074 e-10    J
atomic mass constant energy equivalent in MeV          931.494 028           0.000 023             MeV
atomic mass unit-electron volt relationship            931.494 028 e6        0.000 023 e6          eV
atomic mass unit-hartree relationship                  3.423 177 7149 e7     0.000 000 0049 e7     E_h
atomic mass unit-hertz relationship                    2.252 342 7369 e23    0.000 000 0032 e23    Hz
atomic mass unit-inverse meter relationship            7.513 006 671 e14     0.000 000 011 e14     m^-1
atomic mass unit-joule relationship                    1.492 417 830 e-10    0.000 000 074 e-10    J
atomic mass unit-kelvin relationship                   1.080 9527 e13        0.000 0019 e13        K
atomic mass unit-kilogram relationship                 1.660 538 782 e-27    0.000 000 083 e-27    kg
atomic unit of 1st hyperpolarizability                 3.206 361 533 e-53    0.000 000 081 e-53    C^3 m^3 J^-2
atomic unit of 2nd hyperpolarizability                 6.235 380 95 e-65     0.000 000 31 e-65     C^4 m^4 J^-3
atomic unit of action                                  1.054 571 628 e-34    0.000 000 053 e-34    J s
atomic unit of charge                                  1.602 176 487 e-19    0.000 000 040 e-19    C
atomic unit of charge density                          1.081 202 300 e12     0.000 000 027 e12     C m^-3
atomic unit of current                                 6.623 617 63 e-3      0.000 000 17 e-3      A
atomic unit of electric dipole mom.                    8.478 352 81 e-30     0.000 000 21 e-30     C m
atomic unit of electric field                          5.142 206 32 e11      0.000 000 13 e11      V m^-1
atomic unit of electric field gradient                 9.717 361 66 e21      0.000 000 24 e21      V m^-2
atomic unit of electric polarizability                 1.648 777 2536 e-41   0.000 000 0034 e-41   C^2 m^2 J^-1
atomic unit of electric potential                      27.211 383 86         0.000 000 68          V
atomic unit of electric quadrupole mom.                4.486 551 07 e-40     0.000 000 11 e-40     C m^2
atomic unit of energy                                  4.359 743 94 e-18     0.000 000 22 e-18     J
atomic unit of force                                   8.238 722 06 e-8      0.000 000 41 e-8      N
atomic unit of length                                  0.529 177 208 59 e-10 0.000 000 000 36 e-10 m
atomic unit of mag. dipole mom.                        1.854 801 830 e-23    0.000 000 046 e-23    J T^-1
atomic unit of mag. flux density                       2.350 517 382 e5      0.000 000 059 e5      T
atomic unit of magnetizability                         7.891 036 433 e-29    0.000 000 027 e-29    J T^-2
atomic unit of mass                                    9.109 382 15 e-31     0.000 000 45 e-31     kg
atomic unit of momentum                                1.992 851 565 e-24    0.000 000 099 e-24    kg m s^-1
atomic unit of permittivity                            1.112 650 056... e-10 (exact)               F m^-1
atomic unit of time                                    2.418 884 326 505 e-17 0.000 000 000 016 e-17 s
atomic unit of velocity                                2.187 691 2541 e6     0.000 000 0015 e6     m s^-1
Avogadro constant                                      6.022 141 79 e23      0.000 000 30 e23      mol^-1
Bohr magneton                                          927.400 915 e-26      0.000 023 e-26        J T^-1
Bohr magneton in eV/T                                  5.788 381 7555 e-5    0.000 000 0079 e-5    eV T^-1
Bohr magneton in Hz/T                                  13.996 246 04 e9      0.000 000 35 e9       Hz T^-1
Bohr magneton in inverse meters per tesla              46.686 4515           0.000 0012            m^-1 T^-1
Bohr magneton in K/T                                   0.671 7131            0.000 0012            K T^-1
Bohr radius                                            0.529 177 208 59 e-10 0.000 000 000 36 e-10 m
Boltzmann constant                                     1.380 6504 e-23       0.000 0024 e-23       J K^-1
Boltzmann constant in eV/K                             8.617 343 e-5         0.000 015 e-5         eV K^-1
Boltzmann constant in Hz/K                             2.083 6644 e10        0.000 0036 e10        Hz K^-1
Boltzmann constant in inverse meters per kelvin        69.503 56             0.000 12              m^-1 K^-1
characteristic impedance of vacuum                     376.730 313 461...    (exact)               ohm
classical electron radius                              2.817 940 2894 e-15   0.000 000 0058 e-15   m
Compton wavelength                                     2.426 310 2175 e-12   0.000 000 0033 e-12   m
Compton wavelength over 2 pi                           386.159 264 59 e-15   0.000 000 53 e-15     m
conductance quantum                                    7.748 091 7004 e-5    0.000 000 0053 e-5    S
conventional value of Josephson constant               483 597.9 e9          (exact)               Hz V^-1
conventional value of von Klitzing constant            25 812.807            (exact)               ohm
Cu x unit                                              1.002 076 99 e-13     0.000 000 28 e-13     m
deuteron-electron mag. mom. ratio                      -4.664 345 537 e-4    0.000 000 039 e-4
deuteron-electron mass ratio                           3670.482 9654         0.000 0016
deuteron g factor                                      0.857 438 2308        0.000 000 0072
deuteron mag. mom.                                     0.433 073 465 e-26    0.000 000 011 e-26    J T^-1
deuteron mag. mom. to Bohr magneton ratio              0.466 975 4556 e-3    0.000 000 0039 e-3
deuteron mag. mom. to nuclear magneton ratio           0.857 438 2308        0.000 000 0072
deuteron mass                                          3.343 583 20 e-27     0.000 000 17 e-27     kg
deuteron mass energy equivalent                        3.005 062 72 e-10     0.000 000 15 e-10     J
deuteron mass energy equivalent in MeV                 1875.612 793          0.000 047             MeV
deuteron mass in u                                     2.013 553 212 724     0.000 000 000 078     u
deuteron molar mass                                    2.013 553 212 724 e-3 0.000 000 000 078 e-3 kg mol^-1
deuteron-neutron mag. mom. ratio                       -0.448 206 52         0.000 000 11
deuteron-proton mag. mom. ratio                        0.307 012 2070        0.000 000 0024
deuteron-proton mass ratio                             1.999 007 501 08      0.000 000 000 22
deuteron rms charge radius                             2.1402 e-15           0.0028 e-15           m
electric constant                                      8.854 187 817... e-12 (exact)               F m^-1
electron charge to mass quotient                       -1.758 820 150 e11    0.000 000 044 e11     C kg^-1
electron-deuteron mag. mom. ratio                      -2143.923 498         0.000 018
electron-deuteron mass ratio                           2.724 437 1093 e-4    0.000 000 0012 e-4
electron g factor                                      -2.002 319 304 3622   0.000 000 000 0015
electron gyromag. ratio                                1.760 859 770 e11     0.000 000 044 e11     s^-1 T^-1
electron gyromag. ratio over 2 pi                      28 024.953 64         0.000 70              MHz T^-1
electron mag. mom.                                     -928.476 377 e-26     0.000 023 e-26        J T^-1
electron mag. mom. anomaly                             1.159 652 181 11 e-3  0.000 000 000 74 e-3
electron mag. mom. to Bohr magneton ratio              -1.001 159 652 181 11 0.000 000 000 000 74
electron mag. mom. to nuclear magneton ratio           -1838.281 970 92      0.000 000 80
electron mass                                          9.109 382 15 e-31     0.000 000 45 e-31     kg
electron mass energy equivalent                        8.187 104 38 e-14     0.000 000 41 e-14     J
electron mass energy equivalent in MeV                 0.510 998 910         0.000 000 013         MeV
electron mass in u                                     5.485 799 0943 e-4    0.000 000 0023 e-4    u
electron molar mass                                    5.485 799 0943 e-7    0.000 000 0023 e-7    kg mol^-1
electron-muon mag. mom. ratio                          206.766 9877          0.000 0052
electron-muon mass ratio                               4.836 331 71 e-3      0.000 000 12 e-3
electron-neutron mag. mom. ratio                       960.920 50            0.000 23
electron-neutron mass ratio                            5.438 673 4459 e-4    0.000 000 0033 e-4
electron-proton mag. mom. ratio                        -658.210 6848         0.000 0054
electron-proton mass ratio                             5.446 170 2177 e-4    0.000 000 0024 e-4
electron-tau mass ratio                                2.875 64 e-4          0.000 47 e-4
electron to alpha particle mass ratio                  1.370 933 555 70 e-4  0.000 000 000 58 e-4
electron to shielded helion mag. mom. ratio            864.058 257           0.000 010
electron to shielded proton mag. mom. ratio            -658.227 5971         0.000 0072
electron volt                                          1.602 176 487 e-19    0.000 000 040 e-19    J
electron volt-atomic mass unit relationship            1.073 544 188 e-9     0.000 000 027 e-9     u
electron volt-hartree relationship                     3.674 932 540 e-2     0.000 000 092 e-2     E_h
electron volt-hertz relationship                       2.417 989 454 e14     0.000 000 060 e14     Hz
electron volt-inverse meter relationship               8.065 544 65 e5       0.000 000 20 e5       m^-1
electron volt-joule relationship                       1.602 176 487 e-19    0.000 000 040 e-19    J
electron volt-kelvin relationship                      1.160 4505 e4         0.000 0020 e4         K
electron volt-kilogram relationship                    1.782 661 758 e-36    0.000 000 044 e-36    kg
elementary charge                                      1.602 176 487 e-19    0.000 000 040 e-19    C
elementary charge over h                               2.417 989 454 e14     0.000 000 060 e14     A J^-1
Faraday constant                                       96 485.3399           0.0024                C mol^-1
Faraday constant for conventional electric current     96 485.3401           0.0048                C_90 mol^-1
Fermi coupling constant                                1.166 37 e-5          0.000 01 e-5          GeV^-2
fine-structure constant                                7.297 352 5376 e-3    0.000 000 0050 e-3
first radiation constant                               3.741 771 18 e-16     0.000 000 19 e-16     W m^2
first radiation constant for spectral radiance         1.191 042 759 e-16    0.000 000 059 e-16    W m^2 sr^-1
hartree-atomic mass unit relationship                  2.921 262 2986 e-8    0.000 000 0042 e-8    u
hartree-electron volt relationship                     27.211 383 86         0.000 000 68          eV
Hartree energy                                         4.359 743 94 e-18     0.000 000 22 e-18     J
Hartree energy in eV                                   27.211 383 86         0.000 000 68          eV
hartree-hertz relationship                             6.579 683 920 722 e15 0.000 000 000 044 e15 Hz
hartree-inverse meter relationship                     2.194 746 313 705 e7  0.000 000 000 015 e7  m^-1
hartree-joule relationship                             4.359 743 94 e-18     0.000 000 22 e-18     J
hartree-kelvin relationship                            3.157 7465 e5         0.000 0055 e5         K
hartree-kilogram relationship                          4.850 869 34 e-35     0.000 000 24 e-35     kg
helion-electron mass ratio                             5495.885 2765         0.000 0052
helion mass                                            5.006 411 92 e-27     0.000 000 25 e-27     kg
helion mass energy equivalent                          4.499 538 64 e-10     0.000 000 22 e-10     J
helion mass energy equivalent in MeV                   2808.391 383          0.000 070             MeV
helion mass in u                                       3.014 932 2473        0.000 000 0026        u
helion molar mass                                      3.014 932 2473 e-3    0.000 000 0026 e-3    kg mol^-1
helion-proton mass ratio                               2.993 152 6713        0.000 000 0026
hertz-atomic mass unit relationship                    4.439 821 6294 e-24   0.000 000 0064 e-24   u
hertz-electron volt relationship                       4.135 667 33 e-15     0.000 000 10 e-15     eV
hertz-hartree relationship                             1.519 829 846 006 e-16 0.000 000 000010e-16 E_h
hertz-inverse meter relationship                       3.335 640 951... e-9  (exact)               m^-1
hertz-joule relationship                               6.626 068 96 e-34     0.000 000 33 e-34     J
hertz-kelvin relationship                              4.799 2374 e-11       0.000 0084 e-11       K
hertz-kilogram relationship                            7.372 496 00 e-51     0.000 000 37 e-51     kg
inverse fine-structure constant                        137.035 999 679       0.000 000 094
inverse meter-atomic mass unit relationship            1.331 025 0394 e-15   0.000 000 0019 e-15   u
inverse meter-electron volt relationship               1.239 841 875 e-6     0.000 000 031 e-6     eV
inverse meter-hartree relationship                     4.556 335 252 760 e-8 0.000 000 000 030 e-8 E_h
inverse meter-hertz relationship                       299 792 458           (exact)               Hz
inverse meter-joule relationship                       1.986 445 501 e-25    0.000 000 099 e-25    J
inverse meter-kelvin relationship                      1.438 7752 e-2        0.000 0025 e-2        K
inverse meter-kilogram relationship                    2.210 218 70 e-42     0.000 000 11 e-42     kg
inverse of conductance quantum                         12 906.403 7787       0.000 0088            ohm
Josephson constant                                     483 597.891 e9        0.012 e9              Hz V^-1
joule-atomic mass unit relationship                    6.700 536 41 e9       0.000 000 33 e9       u
joule-electron volt relationship                       6.241 509 65 e18      0.000 000 16 e18      eV
joule-hartree relationship                             2.293 712 69 e17      0.000 000 11 e17      E_h
joule-hertz relationship                               1.509 190 450 e33     0.000 000 075 e33     Hz
joule-inverse meter relationship                       5.034 117 47 e24      0.000 000 25 e24      m^-1
joule-kelvin relationship                              7.242 963 e22         0.000 013 e22         K
joule-kilogram relationship                            1.112 650 056... e-17 (exact)               kg
kelvin-atomic mass unit relationship                   9.251 098 e-14        0.000 016 e-14        u
kelvin-electron volt relationship                      8.617 343 e-5         0.000 015 e-5         eV
kelvin-hartree relationship                            3.166 8153 e-6        0.000 0055 e-6        E_h
kelvin-hertz relationship                              2.083 6644 e10        0.000 0036 e10        Hz
kelvin-inverse meter relationship                      69.503 56             0.000 12              m^-1
kelvin-joule relationship                              1.380 6504 e-23       0.000 0024 e-23       J
kelvin-kilogram relationship                           1.536 1807 e-40       0.000 0027 e-40       kg
kilogram-atomic mass unit relationship                 6.022 141 79 e26      0.000 000 30 e26      u
kilogram-electron volt relationship                    5.609 589 12 e35      0.000 000 14 e35      eV
kilogram-hartree relationship                          2.061 486 16 e34      0.000 000 10 e34      E_h
kilogram-hertz relationship                            1.356 392 733 e50     0.000 000 068 e50     Hz
kilogram-inverse meter relationship                    4.524 439 15 e41      0.000 000 23 e41      m^-1
kilogram-joule relationship                            8.987 551 787... e16  (exact)               J
kilogram-kelvin relationship                           6.509 651 e39         0.000 011 e39         K
lattice parameter of silicon                           543.102 064 e-12      0.000 014 e-12        m
Loschmidt constant (273.15 K, 101.325 kPa)             2.686 7774 e25        0.000 0047 e25        m^-3
mag. constant                                          12.566 370 614... e-7 (exact)               N A^-2
mag. flux quantum                                      2.067 833 667 e-15    0.000 000 052 e-15    Wb
molar gas constant                                     8.314 472             0.000 015             J mol^-1 K^-1
molar mass constant                                    1 e-3                 (exact)               kg mol^-1
molar mass of carbon-12                                12 e-3                (exact)               kg mol^-1
molar Planck constant                                  3.990 312 6821 e-10   0.000 000 0057 e-10   J s mol^-1
molar Planck constant times c                          0.119 626 564 72      0.000 000 000 17      J m mol^-1
molar volume of ideal gas (273.15 K, 100 kPa)          22.710 981 e-3        0.000 040 e-3         m^3 mol^-1
molar volume of ideal gas (273.15 K, 101.325 kPa)      22.413 996 e-3        0.000 039 e-3         m^3 mol^-1
molar volume of silicon                                12.058 8349 e-6       0.000 0011 e-6        m^3 mol^-1
Mo x unit                                              1.002 099 55 e-13     0.000 000 53 e-13     m
muon Compton wavelength                                11.734 441 04 e-15    0.000 000 30 e-15     m
muon Compton wavelength over 2 pi                      1.867 594 295 e-15    0.000 000 047 e-15    m
muon-electron mass ratio                               206.768 2823          0.000 0052
muon g factor                                          -2.002 331 8414       0.000 000 0012
muon mag. mom.                                         -4.490 447 86 e-26    0.000 000 16 e-26     J T^-1
muon mag. mom. anomaly                                 1.165 920 69 e-3      0.000 000 60 e-3
muon mag. mom. to Bohr magneton ratio                  -4.841 970 49 e-3     0.000 000 12 e-3
muon mag. mom. to nuclear magneton ratio               -8.890 597 05         0.000 000 23
muon mass                                              1.883 531 30 e-28     0.000 000 11 e-28     kg
muon mass energy equivalent                            1.692 833 510 e-11    0.000 000 095 e-11    J
muon mass energy equivalent in MeV                     105.658 3668          0.000 0038            MeV
muon mass in u                                         0.113 428 9256        0.000 000 0029        u
muon molar mass                                        0.113 428 9256 e-3    0.000 000 0029 e-3    kg mol^-1
muon-neutron mass ratio                                0.112 454 5167        0.000 000 0029
muon-proton mag. mom. ratio                            -3.183 345 137        0.000 000 085
muon-proton mass ratio                                 0.112 609 5261        0.000 000 0029
muon-tau mass ratio                                    5.945 92 e-2          0.000 97 e-2
natural unit of action                                 1.054 571 628 e-34    0.000 000 053 e-34    J s
natural unit of action in eV s                         6.582 118 99 e-16     0.000 000 16 e-16     eV s
natural unit of energy                                 8.187 104 38 e-14     0.000 000 41 e-14     J
natural unit of energy in MeV                          0.510 998 910         0.000 000 013         MeV
natural unit of length                                 386.159 264 59 e-15   0.000 000 53 e-15     m
natural unit of mass                                   9.109 382 15 e-31     0.000 000 45 e-31     kg
natural unit of momentum                               2.730 924 06 e-22     0.000 000 14 e-22     kg m s^-1
natural unit of momentum in MeV/c                      0.510 998 910         0.000 000 013         MeV/c
natural unit of time                                   1.288 088 6570 e-21   0.000 000 0018 e-21   s
natural unit of velocity                               299 792 458           (exact)               m s^-1
neutron Compton wavelength                             1.319 590 8951 e-15   0.000 000 0020 e-15   m
neutron Compton wavelength over 2 pi                   0.210 019 413 82 e-15 0.000 000 000 31 e-15 m
neutron-electron mag. mom. ratio                       1.040 668 82 e-3      0.000 000 25 e-3
neutron-electron mass ratio                            1838.683 6605         0.000 0011
neutron g factor                                       -3.826 085 45         0.000 000 90
neutron gyromag. ratio                                 1.832 471 85 e8       0.000 000 43 e8       s^-1 T^-1
neutron gyromag. ratio over 2 pi                       29.164 6954           0.000 0069            MHz T^-1
neutron mag. mom.                                      -0.966 236 41 e-26    0.000 000 23 e-26     J T^-1
neutron mag. mom. to Bohr magneton ratio               -1.041 875 63 e-3     0.000 000 25 e-3
neutron mag. mom. to nuclear magneton ratio            -1.913 042 73         0.000 000 45
neutron mass                                           1.674 927 211 e-27    0.000 000 084 e-27    kg
neutron mass energy equivalent                         1.505 349 505 e-10    0.000 000 075 e-10    J
neutron mass energy equivalent in MeV                  939.565 346           0.000 023             MeV
neutron mass in u                                      1.008 664 915 97      0.000 000 000 43      u
neutron molar mass                                     1.008 664 915 97 e-3  0.000 000 000 43 e-3  kg mol^-1
neutron-muon mass ratio                                8.892 484 09          0.000 000 23
neutron-proton mag. mom. ratio                         -0.684 979 34         0.000 000 16
neutron-proton mass ratio                              1.001 378 419 18      0.000 000 000 46
neutron-tau mass ratio                                 0.528 740             0.000 086
neutron to shielded proton mag. mom. ratio             -0.684 996 94         0.000 000 16
Newtonian constant of gravitation                      6.674 28 e-11         0.000 67 e-11         m^3 kg^-1 s^-2
Newtonian constant of gravitation over h-bar c         6.708 81 e-39         0.000 67 e-39         (GeV/c^2)^-2
nuclear magneton                                       5.050 783 24 e-27     0.000 000 13 e-27     J T^-1
nuclear magneton in eV/T                               3.152 451 2326 e-8    0.000 000 0045 e-8    eV T^-1
nuclear magneton in inverse meters per tesla           2.542 623 616 e-2     0.000 000 064 e-2     m^-1 T^-1
nuclear magneton in K/T                                3.658 2637 e-4        0.000 0064 e-4        K T^-1
nuclear magneton in MHz/T                              7.622 593 84          0.000 000 19          MHz T^-1
Planck constant                                        6.626 068 96 e-34     0.000 000 33 e-34     J s
Planck constant in eV s                                4.135 667 33 e-15     0.000 000 10 e-15     eV s
Planck constant over 2 pi                              1.054 571 628 e-34    0.000 000 053 e-34    J s
Planck constant over 2 pi in eV s                      6.582 118 99 e-16     0.000 000 16 e-16     eV s
Planck constant over 2 pi times c in MeV fm            197.326 9631          0.000 0049            MeV fm
Planck length                                          1.616 252 e-35        0.000 081 e-35        m
Planck mass                                            2.176 44 e-8          0.000 11 e-8          kg
Planck mass energy equivalent in GeV                   1.220 892 e19         0.000 061 e19         GeV
Planck temperature                                     1.416 785 e32         0.000 071 e32         K
Planck time                                            5.391 24 e-44         0.000 27 e-44         s
proton charge to mass quotient                         9.578 833 92 e7       0.000 000 24 e7       C kg^-1
proton Compton wavelength                              1.321 409 8446 e-15   0.000 000 0019 e-15   m
proton Compton wavelength over 2 pi                    0.210 308 908 61 e-15 0.000 000 000 30 e-15 m
proton-electron mass ratio                             1836.152 672 47       0.000 000 80
proton g factor                                        5.585 694 713         0.000 000 046
proton gyromag. ratio                                  2.675 222 099 e8      0.000 000 070 e8      s^-1 T^-1
proton gyromag. ratio over 2 pi                        42.577 4821           0.000 0011            MHz T^-1
proton mag. mom.                                       1.410 606 662 e-26    0.000 000 037 e-26    J T^-1
proton mag. mom. to Bohr magneton ratio                1.521 032 209 e-3     0.000 000 012 e-3
proton mag. mom. to nuclear magneton ratio             2.792 847 356         0.000 000 023
proton mag. shielding correction                       25.694 e-6            0.014 e-6
proton mass                                            1.672 621 637 e-27    0.000 000 083 e-27    kg
proton mass energy equivalent                          1.503 277 359 e-10    0.000 000 075 e-10    J
proton mass energy equivalent in MeV                   938.272 013           0.000 023             MeV
proton mass in u                                       1.007 276 466 77      0.000 000 000 10      u
proton molar mass                                      1.007 276 466 77 e-3  0.000 000 000 10 e-3  kg mol^-1
proton-muon mass ratio                                 8.880 243 39          0.000 000 23
proton-neutron mag. mom. ratio                         -1.459 898 06         0.000 000 34
proton-neutron mass ratio                              0.998 623 478 24      0.000 000 000 46
proton rms charge radius                               0.8768 e-15           0.0069 e-15           m
proton-tau mass ratio                                  0.528 012             0.000 086
quantum of circulation                                 3.636 947 5199 e-4    0.000 000 0050 e-4    m^2 s^-1
quantum of circulation times 2                         7.273 895 040 e-4     0.000 000 010 e-4     m^2 s^-1
Rydberg constant                                       10 973 731.568 527    0.000 073             m^-1
Rydberg constant times c in Hz                         3.289 841 960 361 e15 0.000 000 000 022 e15 Hz
Rydberg constant times hc in eV                        13.605 691 93         0.000 000 34          eV
Rydberg constant times hc in J                         2.179 871 97 e-18     0.000 000 11 e-18     J
Sackur-Tetrode constant (1 K, 100 kPa)                 -1.151 7047           0.000 0044
Sackur-Tetrode constant (1 K, 101.325 kPa)             -1.164 8677           0.000 0044
second radiation constant                              1.438 7752 e-2        0.000 0025 e-2        m K
shielded helion gyromag. ratio                         2.037 894 730 e8      0.000 000 056 e8      s^-1 T^-1
shielded helion gyromag. ratio over 2 pi               32.434 101 98         0.000 000 90          MHz T^-1
shielded helion mag. mom.                              -1.074 552 982 e-26   0.000 000 030 e-26    J T^-1
shielded helion mag. mom. to Bohr magneton ratio       -1.158 671 471 e-3    0.000 000 014 e-3
shielded helion mag. mom. to nuclear magneton ratio    -2.127 497 718        0.000 000 025
shielded helion to proton mag. mom. ratio              -0.761 766 558        0.000 000 011
shielded helion to shielded proton mag. mom. ratio     -0.761 786 1313       0.000 000 0033
shielded proton gyromag. ratio                         2.675 153 362 e8      0.000 000 073 e8      s^-1 T^-1
shielded proton gyromag. ratio over 2 pi               42.576 3881           0.000 0012            MHz T^-1
shielded proton mag. mom.                              1.410 570 419 e-26    0.000 000 038 e-26    J T^-1
shielded proton mag. mom. to Bohr magneton ratio       1.520 993 128 e-3     0.000 000 017 e-3
shielded proton mag. mom. to nuclear magneton ratio    2.792 775 598         0.000 000 030
speed of light in vacuum                               299 792 458           (exact)               m s^-1
standard acceleration of gravity                       9.806 65              (exact)               m s^-2
standard atmosphere                                    101 325               (exact)               Pa
Stefan-Boltzmann constant                              5.670 400 e-8         0.000 040 e-8         W m^-2 K^-4
tau Compton wavelength                                 0.697 72 e-15         0.000 11 e-15         m
tau Compton wavelength over 2 pi                       0.111 046 e-15        0.000 018 e-15        m
tau-electron mass ratio                                3477.48               0.57
tau mass                                               3.167 77 e-27         0.000 52 e-27         kg
tau mass energy equivalent                             2.847 05 e-10         0.000 46 e-10         J
tau mass energy equivalent in MeV                      1776.99               0.29                  MeV
tau mass in u                                          1.907 68              0.000 31              u
tau molar mass                                         1.907 68 e-3          0.000 31 e-3          kg mol^-1
tau-muon mass ratio                                    16.8183               0.0027
tau-neutron mass ratio                                 1.891 29              0.000 31
tau-proton mass ratio                                  1.893 90              0.000 31
Thomson cross section                                  0.665 245 8558 e-28   0.000 000 0027 e-28   m^2
triton-electron mag. mom. ratio                        -1.620 514 423 e-3    0.000 000 021 e-3
triton-electron mass ratio                             5496.921 5269         0.000 0051
triton g factor                                        5.957 924 896         0.000 000 076
triton mag. mom.                                       1.504 609 361 e-26    0.000 000 042 e-26    J T^-1
triton mag. mom. to Bohr magneton ratio                1.622 393 657 e-3     0.000 000 021 e-3
triton mag. mom. to nuclear magneton ratio             2.978 962 448         0.000 000 038
triton mass                                            5.007 355 88 e-27     0.000 000 25 e-27     kg
triton mass energy equivalent                          4.500 387 03 e-10     0.000 000 22 e-10     J
triton mass energy equivalent in MeV                   2808.920 906          0.000 070             MeV
triton mass in u                                       3.015 500 7134        0.000 000 0025        u
triton molar mass                                      3.015 500 7134 e-3    0.000 000 0025 e-3    kg mol^-1
triton-neutron mag. mom. ratio                         -1.557 185 53         0.000 000 37
triton-proton mag. mom. ratio                          1.066 639 908         0.000 000 010
triton-proton mass ratio                               2.993 717 0309        0.000 000 0025
unified atomic mass unit                               1.660 538 782 e-27    0.000 000 083 e-27    kg
von Klitzing constant                                  25 812.807 557        0.000 018             ohm
weak mixing angle                                      0.222 55              0.000 56
Wien frequency displacement law constant               5.878 933 e10         0.000 010 e10         Hz K^-1
Wien wavelength displacement law constant              2.897 7685 e-3        0.000 0051 e-3        m K"""

txt2010 = """\
{220} lattice spacing of silicon                       192.015 5714 e-12     0.000 0032 e-12       m
alpha particle-electron mass ratio                     7294.299 5361         0.000 0029
alpha particle mass                                    6.644 656 75 e-27     0.000 000 29 e-27     kg
alpha particle mass energy equivalent                  5.971 919 67 e-10     0.000 000 26 e-10     J
alpha particle mass energy equivalent in MeV           3727.379 240          0.000 082             MeV
alpha particle mass in u                               4.001 506 179 125     0.000 000 000 062     u
alpha particle molar mass                              4.001 506 179 125 e-3 0.000 000 000 062 e-3 kg mol^-1
alpha particle-proton mass ratio                       3.972 599 689 33      0.000 000 000 36
Angstrom star                                          1.000 014 95 e-10     0.000 000 90 e-10     m
atomic mass constant                                   1.660 538 921 e-27    0.000 000 073 e-27    kg
atomic mass constant energy equivalent                 1.492 417 954 e-10    0.000 000 066 e-10    J
atomic mass constant energy equivalent in MeV          931.494 061           0.000 021             MeV
atomic mass unit-electron volt relationship            931.494 061 e6        0.000 021 e6          eV
atomic mass unit-hartree relationship                  3.423 177 6845 e7     0.000 000 0024 e7     E_h
atomic mass unit-hertz relationship                    2.252 342 7168 e23    0.000 000 0016 e23    Hz
atomic mass unit-inverse meter relationship            7.513 006 6042 e14    0.000 000 0053 e14    m^-1
atomic mass unit-joule relationship                    1.492 417 954 e-10    0.000 000 066 e-10    J
atomic mass unit-kelvin relationship                   1.080 954 08 e13      0.000 000 98 e13      K
atomic mass unit-kilogram relationship                 1.660 538 921 e-27    0.000 000 073 e-27    kg
atomic unit of 1st hyperpolarizability                 3.206 361 449 e-53    0.000 000 071 e-53    C^3 m^3 J^-2
atomic unit of 2nd hyperpolarizability                 6.235 380 54 e-65     0.000 000 28 e-65     C^4 m^4 J^-3
atomic unit of action                                  1.054 571 726 e-34    0.000 000 047 e-34    J s
atomic unit of charge                                  1.602 176 565 e-19    0.000 000 035 e-19    C
atomic unit of charge density                          1.081 202 338 e12     0.000 000 024 e12     C m^-3
atomic unit of current                                 6.623 617 95 e-3      0.000 000 15 e-3      A
atomic unit of electric dipole mom.                    8.478 353 26 e-30     0.000 000 19 e-30     C m
atomic unit of electric field                          5.142 206 52 e11      0.000 000 11 e11      V m^-1
atomic unit of electric field gradient                 9.717 362 00 e21      0.000 000 21 e21      V m^-2
atomic unit of electric polarizability                 1.648 777 2754 e-41   0.000 000 0016 e-41   C^2 m^2 J^-1
atomic unit of electric potential                      27.211 385 05         0.000 000 60          V
atomic unit of electric quadrupole mom.                4.486 551 331 e-40    0.000 000 099 e-40    C m^2
atomic unit of energy                                  4.359 744 34 e-18     0.000 000 19 e-18     J
atomic unit of force                                   8.238 722 78 e-8      0.000 000 36 e-8      N
atomic unit of length                                  0.529 177 210 92 e-10 0.000 000 000 17 e-10 m
atomic unit of mag. dipole mom.                        1.854 801 936 e-23    0.000 000 041 e-23    J T^-1
atomic unit of mag. flux density                       2.350 517 464 e5      0.000 000 052 e5      T
atomic unit of magnetizability                         7.891 036 607 e-29    0.000 000 013 e-29    J T^-2
atomic unit of mass                                    9.109 382 91 e-31     0.000 000 40 e-31     kg
atomic unit of mom.um                                  1.992 851 740 e-24    0.000 000 088 e-24    kg m s^-1
atomic unit of permittivity                            1.112 650 056... e-10 (exact)               F m^-1
atomic unit of time                                    2.418 884 326 502e-17 0.000 000 000 012e-17 s
atomic unit of velocity                                2.187 691 263 79 e6   0.000 000 000 71 e6   m s^-1
Avogadro constant                                      6.022 141 29 e23      0.000 000 27 e23      mol^-1
Bohr magneton                                          927.400 968 e-26      0.000 020 e-26        J T^-1
Bohr magneton in eV/T                                  5.788 381 8066 e-5    0.000 000 0038 e-5    eV T^-1
Bohr magneton in Hz/T                                  13.996 245 55 e9      0.000 000 31 e9       Hz T^-1
Bohr magneton in inverse meters per tesla              46.686 4498           0.000 0010            m^-1 T^-1
Bohr magneton in K/T                                   0.671 713 88          0.000 000 61          K T^-1
Bohr radius                                            0.529 177 210 92 e-10 0.000 000 000 17 e-10 m
Boltzmann constant                                     1.380 6488 e-23       0.000 0013 e-23       J K^-1
Boltzmann constant in eV/K                             8.617 3324 e-5        0.000 0078 e-5        eV K^-1
Boltzmann constant in Hz/K                             2.083 6618 e10        0.000 0019 e10        Hz K^-1
Boltzmann constant in inverse meters per kelvin        69.503 476            0.000 063             m^-1 K^-1
characteristic impedance of vacuum                     376.730 313 461...    (exact)               ohm
classical electron radius                              2.817 940 3267 e-15   0.000 000 0027 e-15   m
Compton wavelength                                     2.426 310 2389 e-12   0.000 000 0016 e-12   m
Compton wavelength over 2 pi                           386.159 268 00 e-15   0.000 000 25 e-15     m
conductance quantum                                    7.748 091 7346 e-5    0.000 000 0025 e-5    S
conventional value of Josephson constant               483 597.9 e9          (exact)               Hz V^-1
conventional value of von Klitzing constant            25 812.807            (exact)               ohm
Cu x unit                                              1.002 076 97 e-13     0.000 000 28 e-13     m
deuteron-electron mag. mom. ratio                      -4.664 345 537 e-4    0.000 000 039 e-4
deuteron-electron mass ratio                           3670.482 9652         0.000 0015
deuteron g factor                                      0.857 438 2308        0.000 000 0072
deuteron mag. mom.                                     0.433 073 489 e-26    0.000 000 010 e-26    J T^-1
deuteron mag. mom. to Bohr magneton ratio              0.466 975 4556 e-3    0.000 000 0039 e-3
deuteron mag. mom. to nuclear magneton ratio           0.857 438 2308        0.000 000 0072
deuteron mass                                          3.343 583 48 e-27     0.000 000 15 e-27     kg
deuteron mass energy equivalent                        3.005 062 97 e-10     0.000 000 13 e-10     J
deuteron mass energy equivalent in MeV                 1875.612 859          0.000 041             MeV
deuteron mass in u                                     2.013 553 212 712     0.000 000 000 077     u
deuteron molar mass                                    2.013 553 212 712 e-3 0.000 000 000 077 e-3 kg mol^-1
deuteron-neutron mag. mom. ratio                       -0.448 206 52         0.000 000 11
deuteron-proton mag. mom. ratio                        0.307 012 2070        0.000 000 0024
deuteron-proton mass ratio                             1.999 007 500 97      0.000 000 000 18
deuteron rms charge radius                             2.1424 e-15           0.0021 e-15           m
electric constant                                      8.854 187 817... e-12 (exact)               F m^-1
electron charge to mass quotient                       -1.758 820 088 e11    0.000 000 039 e11     C kg^-1
electron-deuteron mag. mom. ratio                      -2143.923 498         0.000 018
electron-deuteron mass ratio                           2.724 437 1095 e-4    0.000 000 0011 e-4
electron g factor                                      -2.002 319 304 361 53 0.000 000 000 000 53
electron gyromag. ratio                                1.760 859 708 e11     0.000 000 039 e11     s^-1 T^-1
electron gyromag. ratio over 2 pi                      28 024.952 66         0.000 62              MHz T^-1
electron-helion mass ratio                             1.819 543 0761 e-4    0.000 000 0017 e-4
electron mag. mom.                                     -928.476 430 e-26     0.000 021 e-26        J T^-1
electron mag. mom. anomaly                             1.159 652 180 76 e-3  0.000 000 000 27 e-3
electron mag. mom. to Bohr magneton ratio              -1.001 159 652 180 76 0.000 000 000 000 27
electron mag. mom. to nuclear magneton ratio           -1838.281 970 90      0.000 000 75
electron mass                                          9.109 382 91 e-31     0.000 000 40 e-31     kg
electron mass energy equivalent                        8.187 105 06 e-14     0.000 000 36 e-14     J
electron mass energy equivalent in MeV                 0.510 998 928         0.000 000 011         MeV
electron mass in u                                     5.485 799 0946 e-4    0.000 000 0022 e-4    u
electron molar mass                                    5.485 799 0946 e-7    0.000 000 0022 e-7    kg mol^-1
electron-muon mag. mom. ratio                          206.766 9896          0.000 0052
electron-muon mass ratio                               4.836 331 66 e-3      0.000 000 12 e-3
electron-neutron mag. mom. ratio                       960.920 50            0.000 23
electron-neutron mass ratio                            5.438 673 4461 e-4    0.000 000 0032 e-4
electron-proton mag. mom. ratio                        -658.210 6848         0.000 0054
electron-proton mass ratio                             5.446 170 2178 e-4    0.000 000 0022 e-4
electron-tau mass ratio                                2.875 92 e-4          0.000 26 e-4
electron to alpha particle mass ratio                  1.370 933 555 78 e-4  0.000 000 000 55 e-4
electron to shielded helion mag. mom. ratio            864.058 257           0.000 010
electron to shielded proton mag. mom. ratio            -658.227 5971         0.000 0072
electron-triton mass ratio                             1.819 200 0653 e-4    0.000 000 0017 e-4
electron volt                                          1.602 176 565 e-19    0.000 000 035 e-19    J
electron volt-atomic mass unit relationship            1.073 544 150 e-9     0.000 000 024 e-9     u
electron volt-hartree relationship                     3.674 932 379 e-2     0.000 000 081 e-2     E_h
electron volt-hertz relationship                       2.417 989 348 e14     0.000 000 053 e14     Hz
electron volt-inverse meter relationship               8.065 544 29 e5       0.000 000 18 e5       m^-1
electron volt-joule relationship                       1.602 176 565 e-19    0.000 000 035 e-19    J
electron volt-kelvin relationship                      1.160 4519 e4         0.000 0011 e4         K
electron volt-kilogram relationship                    1.782 661 845 e-36    0.000 000 039 e-36    kg
elementary charge                                      1.602 176 565 e-19    0.000 000 035 e-19    C
elementary charge over h                               2.417 989 348 e14     0.000 000 053 e14     A J^-1
Faraday constant                                       96 485.3365           0.0021                C mol^-1
Faraday constant for conventional electric current     96 485.3321           0.0043                C_90 mol^-1
Fermi coupling constant                                1.166 364 e-5         0.000 005 e-5         GeV^-2
fine-structure constant                                7.297 352 5698 e-3    0.000 000 0024 e-3
first radiation constant                               3.741 771 53 e-16     0.000 000 17 e-16     W m^2
first radiation constant for spectral radiance         1.191 042 869 e-16    0.000 000 053 e-16    W m^2 sr^-1
hartree-atomic mass unit relationship                  2.921 262 3246 e-8    0.000 000 0021 e-8    u
hartree-electron volt relationship                     27.211 385 05         0.000 000 60          eV
Hartree energy                                         4.359 744 34 e-18     0.000 000 19 e-18     J
Hartree energy in eV                                   27.211 385 05         0.000 000 60          eV
hartree-hertz relationship                             6.579 683 920 729 e15 0.000 000 000 033 e15 Hz
hartree-inverse meter relationship                     2.194 746 313 708 e7  0.000 000 000 011 e7  m^-1
hartree-joule relationship                             4.359 744 34 e-18     0.000 000 19 e-18     J
hartree-kelvin relationship                            3.157 7504 e5         0.000 0029 e5         K
hartree-kilogram relationship                          4.850 869 79 e-35     0.000 000 21 e-35     kg
helion-electron mass ratio                             5495.885 2754         0.000 0050
helion g factor                                        -4.255 250 613        0.000 000 050
helion mag. mom.                                       -1.074 617 486 e-26   0.000 000 027 e-26    J T^-1
helion mag. mom. to Bohr magneton ratio                -1.158 740 958 e-3    0.000 000 014 e-3
helion mag. mom. to nuclear magneton ratio             -2.127 625 306        0.000 000 025
helion mass                                            5.006 412 34 e-27     0.000 000 22 e-27     kg
helion mass energy equivalent                          4.499 539 02 e-10     0.000 000 20 e-10     J
helion mass energy equivalent in MeV                   2808.391 482          0.000 062             MeV
helion mass in u                                       3.014 932 2468        0.000 000 0025        u
helion molar mass                                      3.014 932 2468 e-3    0.000 000 0025 e-3    kg mol^-1
helion-proton mass ratio                               2.993 152 6707        0.000 000 0025
hertz-atomic mass unit relationship                    4.439 821 6689 e-24   0.000 000 0031 e-24   u
hertz-electron volt relationship                       4.135 667 516 e-15    0.000 000 091 e-15    eV
hertz-hartree relationship                             1.519 829 8460045e-16 0.000 000 0000076e-16 E_h
hertz-inverse meter relationship                       3.335 640 951... e-9  (exact)               m^-1
hertz-joule relationship                               6.626 069 57 e-34     0.000 000 29 e-34     J
hertz-kelvin relationship                              4.799 2434 e-11       0.000 0044 e-11       K
hertz-kilogram relationship                            7.372 496 68 e-51     0.000 000 33 e-51     kg
inverse fine-structure constant                        137.035 999 074       0.000 000 044
inverse meter-atomic mass unit relationship            1.331 025 051 20 e-15 0.000 000 000 94 e-15 u
inverse meter-electron volt relationship               1.239 841 930 e-6     0.000 000 027 e-6     eV
inverse meter-hartree relationship                     4.556 335 252 755 e-8 0.000 000 000 023 e-8 E_h
inverse meter-hertz relationship                       299 792 458           (exact)               Hz
inverse meter-joule relationship                       1.986 445 684 e-25    0.000 000 088 e-25    J
inverse meter-kelvin relationship                      1.438 7770 e-2        0.000 0013 e-2        K
inverse meter-kilogram relationship                    2.210 218 902 e-42    0.000 000 098 e-42    kg
inverse of conductance quantum                         12 906.403 7217       0.000 0042            ohm
Josephson constant                                     483 597.870 e9        0.011 e9              Hz V^-1
joule-atomic mass unit relationship                    6.700 535 85 e9       0.000 000 30 e9       u
joule-electron volt relationship                       6.241 509 34 e18      0.000 000 14 e18      eV
joule-hartree relationship                             2.293 712 48 e17      0.000 000 10 e17      E_h
joule-hertz relationship                               1.509 190 311 e33     0.000 000 067 e33     Hz
joule-inverse meter relationship                       5.034 117 01 e24      0.000 000 22 e24      m^-1
joule-kelvin relationship                              7.242 9716 e22        0.000 0066 e22        K
joule-kilogram relationship                            1.112 650 056... e-17 (exact)               kg
kelvin-atomic mass unit relationship                   9.251 0868 e-14       0.000 0084 e-14       u
kelvin-electron volt relationship                      8.617 3324 e-5        0.000 0078 e-5        eV
kelvin-hartree relationship                            3.166 8114 e-6        0.000 0029 e-6        E_h
kelvin-hertz relationship                              2.083 6618 e10        0.000 0019 e10        Hz
kelvin-inverse meter relationship                      69.503 476            0.000 063             m^-1
kelvin-joule relationship                              1.380 6488 e-23       0.000 0013 e-23       J
kelvin-kilogram relationship                           1.536 1790 e-40       0.000 0014 e-40       kg
kilogram-atomic mass unit relationship                 6.022 141 29 e26      0.000 000 27 e26      u
kilogram-electron volt relationship                    5.609 588 85 e35      0.000 000 12 e35      eV
kilogram-hartree relationship                          2.061 485 968 e34     0.000 000 091 e34     E_h
kilogram-hertz relationship                            1.356 392 608 e50     0.000 000 060 e50     Hz
kilogram-inverse meter relationship                    4.524 438 73 e41      0.000 000 20 e41      m^-1
kilogram-joule relationship                            8.987 551 787... e16  (exact)               J
kilogram-kelvin relationship                           6.509 6582 e39        0.000 0059 e39        K
lattice parameter of silicon                           543.102 0504 e-12     0.000 0089 e-12       m
Loschmidt constant (273.15 K, 100 kPa)                 2.651 6462 e25        0.000 0024 e25        m^-3
Loschmidt constant (273.15 K, 101.325 kPa)             2.686 7805 e25        0.000 0024 e25        m^-3
mag. constant                                          12.566 370 614... e-7 (exact)               N A^-2
mag. flux quantum                                      2.067 833 758 e-15    0.000 000 046 e-15    Wb
molar gas constant                                     8.314 4621            0.000 0075            J mol^-1 K^-1
molar mass constant                                    1 e-3                 (exact)               kg mol^-1
molar mass of carbon-12                                12 e-3                (exact)               kg mol^-1
molar Planck constant                                  3.990 312 7176 e-10   0.000 000 0028 e-10   J s mol^-1
molar Planck constant times c                          0.119 626 565 779     0.000 000 000 084     J m mol^-1
molar volume of ideal gas (273.15 K, 100 kPa)          22.710 953 e-3        0.000 021 e-3         m^3 mol^-1
molar volume of ideal gas (273.15 K, 101.325 kPa)      22.413 968 e-3        0.000 020 e-3         m^3 mol^-1
molar volume of silicon                                12.058 833 01 e-6     0.000 000 80 e-6      m^3 mol^-1
Mo x unit                                              1.002 099 52 e-13     0.000 000 53 e-13     m
muon Compton wavelength                                11.734 441 03 e-15    0.000 000 30 e-15     m
muon Compton wavelength over 2 pi                      1.867 594 294 e-15    0.000 000 047 e-15    m
muon-electron mass ratio                               206.768 2843          0.000 0052
muon g factor                                          -2.002 331 8418       0.000 000 0013
muon mag. mom.                                         -4.490 448 07 e-26    0.000 000 15 e-26     J T^-1
muon mag. mom. anomaly                                 1.165 920 91 e-3      0.000 000 63 e-3
muon mag. mom. to Bohr magneton ratio                  -4.841 970 44 e-3     0.000 000 12 e-3
muon mag. mom. to nuclear magneton ratio               -8.890 596 97         0.000 000 22
muon mass                                              1.883 531 475 e-28    0.000 000 096 e-28    kg
muon mass energy equivalent                            1.692 833 667 e-11    0.000 000 086 e-11    J
muon mass energy equivalent in MeV                     105.658 3715          0.000 0035            MeV
muon mass in u                                         0.113 428 9267        0.000 000 0029        u
muon molar mass                                        0.113 428 9267 e-3    0.000 000 0029 e-3    kg mol^-1
muon-neutron mass ratio                                0.112 454 5177        0.000 000 0028
muon-proton mag. mom. ratio                            -3.183 345 107        0.000 000 084
muon-proton mass ratio                                 0.112 609 5272        0.000 000 0028
muon-tau mass ratio                                    5.946 49 e-2          0.000 54 e-2
natural unit of action                                 1.054 571 726 e-34    0.000 000 047 e-34    J s
natural unit of action in eV s                         6.582 119 28 e-16     0.000 000 15 e-16     eV s
natural unit of energy                                 8.187 105 06 e-14     0.000 000 36 e-14     J
natural unit of energy in MeV                          0.510 998 928         0.000 000 011         MeV
natural unit of length                                 386.159 268 00 e-15   0.000 000 25 e-15     m
natural unit of mass                                   9.109 382 91 e-31     0.000 000 40 e-31     kg
natural unit of mom.um                                 2.730 924 29 e-22     0.000 000 12 e-22     kg m s^-1
natural unit of mom.um in MeV/c                        0.510 998 928         0.000 000 011         MeV/c
natural unit of time                                   1.288 088 668 33 e-21 0.000 000 000 83 e-21 s
natural unit of velocity                               299 792 458           (exact)               m s^-1
neutron Compton wavelength                             1.319 590 9068 e-15   0.000 000 0011 e-15   m
neutron Compton wavelength over 2 pi                   0.210 019 415 68 e-15 0.000 000 000 17 e-15 m
neutron-electron mag. mom. ratio                       1.040 668 82 e-3      0.000 000 25 e-3
neutron-electron mass ratio                            1838.683 6605         0.000 0011
neutron g factor                                       -3.826 085 45         0.000 000 90
neutron gyromag. ratio                                 1.832 471 79 e8       0.000 000 43 e8       s^-1 T^-1
neutron gyromag. ratio over 2 pi                       29.164 6943           0.000 0069            MHz T^-1
neutron mag. mom.                                      -0.966 236 47 e-26    0.000 000 23 e-26     J T^-1
neutron mag. mom. to Bohr magneton ratio               -1.041 875 63 e-3     0.000 000 25 e-3
neutron mag. mom. to nuclear magneton ratio            -1.913 042 72         0.000 000 45
neutron mass                                           1.674 927 351 e-27    0.000 000 074 e-27    kg
neutron mass energy equivalent                         1.505 349 631 e-10    0.000 000 066 e-10    J
neutron mass energy equivalent in MeV                  939.565 379           0.000 021             MeV
neutron mass in u                                      1.008 664 916 00      0.000 000 000 43      u
neutron molar mass                                     1.008 664 916 00 e-3  0.000 000 000 43 e-3  kg mol^-1
neutron-muon mass ratio                                8.892 484 00          0.000 000 22
neutron-proton mag. mom. ratio                         -0.684 979 34         0.000 000 16
neutron-proton mass difference                         2.305 573 92 e-30     0.000 000 76 e-30
neutron-proton mass difference energy equivalent       2.072 146 50 e-13     0.000 000 68 e-13
neutron-proton mass difference energy equivalent in MeV 1.293 332 17          0.000 000 42
neutron-proton mass difference in u                    0.001 388 449 19      0.000 000 000 45
neutron-proton mass ratio                              1.001 378 419 17      0.000 000 000 45
neutron-tau mass ratio                                 0.528 790             0.000 048
neutron to shielded proton mag. mom. ratio             -0.684 996 94         0.000 000 16
Newtonian constant of gravitation                      6.673 84 e-11         0.000 80 e-11         m^3 kg^-1 s^-2
Newtonian constant of gravitation over h-bar c         6.708 37 e-39         0.000 80 e-39         (GeV/c^2)^-2
nuclear magneton                                       5.050 783 53 e-27     0.000 000 11 e-27     J T^-1
nuclear magneton in eV/T                               3.152 451 2605 e-8    0.000 000 0022 e-8    eV T^-1
nuclear magneton in inverse meters per tesla           2.542 623 527 e-2     0.000 000 056 e-2     m^-1 T^-1
nuclear magneton in K/T                                3.658 2682 e-4        0.000 0033 e-4        K T^-1
nuclear magneton in MHz/T                              7.622 593 57          0.000 000 17          MHz T^-1
Planck constant                                        6.626 069 57 e-34     0.000 000 29 e-34     J s
Planck constant in eV s                                4.135 667 516 e-15    0.000 000 091 e-15    eV s
Planck constant over 2 pi                              1.054 571 726 e-34    0.000 000 047 e-34    J s
Planck constant over 2 pi in eV s                      6.582 119 28 e-16     0.000 000 15 e-16     eV s
Planck constant over 2 pi times c in MeV fm            197.326 9718          0.000 0044            MeV fm
Planck length                                          1.616 199 e-35        0.000 097 e-35        m
Planck mass                                            2.176 51 e-8          0.000 13 e-8          kg
Planck mass energy equivalent in GeV                   1.220 932 e19         0.000 073 e19         GeV
Planck temperature                                     1.416 833 e32         0.000 085 e32         K
Planck time                                            5.391 06 e-44         0.000 32 e-44         s
proton charge to mass quotient                         9.578 833 58 e7       0.000 000 21 e7       C kg^-1
proton Compton wavelength                              1.321 409 856 23 e-15 0.000 000 000 94 e-15 m
proton Compton wavelength over 2 pi                    0.210 308 910 47 e-15 0.000 000 000 15 e-15 m
proton-electron mass ratio                             1836.152 672 45       0.000 000 75
proton g factor                                        5.585 694 713         0.000 000 046
proton gyromag. ratio                                  2.675 222 005 e8      0.000 000 063 e8      s^-1 T^-1
proton gyromag. ratio over 2 pi                        42.577 4806           0.000 0010            MHz T^-1
proton mag. mom.                                       1.410 606 743 e-26    0.000 000 033 e-26    J T^-1
proton mag. mom. to Bohr magneton ratio                1.521 032 210 e-3     0.000 000 012 e-3
proton mag. mom. to nuclear magneton ratio             2.792 847 356         0.000 000 023
proton mag. shielding correction                       25.694 e-6            0.014 e-6
proton mass                                            1.672 621 777 e-27    0.000 000 074 e-27    kg
proton mass energy equivalent                          1.503 277 484 e-10    0.000 000 066 e-10    J
proton mass energy equivalent in MeV                   938.272 046           0.000 021             MeV
proton mass in u                                       1.007 276 466 812     0.000 000 000 090     u
proton molar mass                                      1.007 276 466 812 e-3 0.000 000 000 090 e-3 kg mol^-1
proton-muon mass ratio                                 8.880 243 31          0.000 000 22
proton-neutron mag. mom. ratio                         -1.459 898 06         0.000 000 34
proton-neutron mass ratio                              0.998 623 478 26      0.000 000 000 45
proton rms charge radius                               0.8775 e-15           0.0051 e-15           m
proton-tau mass ratio                                  0.528 063             0.000 048
quantum of circulation                                 3.636 947 5520 e-4    0.000 000 0024 e-4    m^2 s^-1
quantum of circulation times 2                         7.273 895 1040 e-4    0.000 000 0047 e-4    m^2 s^-1
Rydberg constant                                       10 973 731.568 539    0.000 055             m^-1
Rydberg constant times c in Hz                         3.289 841 960 364 e15 0.000 000 000 017 e15 Hz
Rydberg constant times hc in eV                        13.605 692 53         0.000 000 30          eV
Rydberg constant times hc in J                         2.179 872 171 e-18    0.000 000 096 e-18    J
Sackur-Tetrode constant (1 K, 100 kPa)                 -1.151 7078           0.000 0023
Sackur-Tetrode constant (1 K, 101.325 kPa)             -1.164 8708           0.000 0023
second radiation constant                              1.438 7770 e-2        0.000 0013 e-2        m K
shielded helion gyromag. ratio                         2.037 894 659 e8      0.000 000 051 e8      s^-1 T^-1
shielded helion gyromag. ratio over 2 pi               32.434 100 84         0.000 000 81          MHz T^-1
shielded helion mag. mom.                              -1.074 553 044 e-26   0.000 000 027 e-26    J T^-1
shielded helion mag. mom. to Bohr magneton ratio       -1.158 671 471 e-3    0.000 000 014 e-3
shielded helion mag. mom. to nuclear magneton ratio    -2.127 497 718        0.000 000 025
shielded helion to proton mag. mom. ratio              -0.761 766 558        0.000 000 011
shielded helion to shielded proton mag. mom. ratio     -0.761 786 1313       0.000 000 0033
shielded proton gyromag. ratio                         2.675 153 268 e8      0.000 000 066 e8      s^-1 T^-1
shielded proton gyromag. ratio over 2 pi               42.576 3866           0.000 0010            MHz T^-1
shielded proton mag. mom.                              1.410 570 499 e-26    0.000 000 035 e-26    J T^-1
shielded proton mag. mom. to Bohr magneton ratio       1.520 993 128 e-3     0.000 000 017 e-3
shielded proton mag. mom. to nuclear magneton ratio    2.792 775 598         0.000 000 030
speed of light in vacuum                               299 792 458           (exact)               m s^-1
standard acceleration of gravity                       9.806 65              (exact)               m s^-2
standard atmosphere                                    101 325               (exact)               Pa
standard-state pressure                                100 000               (exact)               Pa
Stefan-Boltzmann constant                              5.670 373 e-8         0.000 021 e-8         W m^-2 K^-4
tau Compton wavelength                                 0.697 787 e-15        0.000 063 e-15        m
tau Compton wavelength over 2 pi                       0.111 056 e-15        0.000 010 e-15        m
tau-electron mass ratio                                3477.15               0.31
tau mass                                               3.167 47 e-27         0.000 29 e-27         kg
tau mass energy equivalent                             2.846 78 e-10         0.000 26 e-10         J
tau mass energy equivalent in MeV                      1776.82               0.16                  MeV
tau mass in u                                          1.907 49              0.000 17              u
tau molar mass                                         1.907 49 e-3          0.000 17 e-3          kg mol^-1
tau-muon mass ratio                                    16.8167               0.0015
tau-neutron mass ratio                                 1.891 11              0.000 17
tau-proton mass ratio                                  1.893 72              0.000 17
Thomson cross section                                  0.665 245 8734 e-28   0.000 000 0013 e-28   m^2
triton-electron mass ratio                             5496.921 5267         0.000 0050
triton g factor                                        5.957 924 896         0.000 000 076
triton mag. mom.                                       1.504 609 447 e-26    0.000 000 038 e-26    J T^-1
triton mag. mom. to Bohr magneton ratio                1.622 393 657 e-3     0.000 000 021 e-3
triton mag. mom. to nuclear magneton ratio             2.978 962 448         0.000 000 038
triton mass                                            5.007 356 30 e-27     0.000 000 22 e-27     kg
triton mass energy equivalent                          4.500 387 41 e-10     0.000 000 20 e-10     J
triton mass energy equivalent in MeV                   2808.921 005          0.000 062             MeV
triton mass in u                                       3.015 500 7134        0.000 000 0025        u
triton molar mass                                      3.015 500 7134 e-3    0.000 000 0025 e-3    kg mol^-1
triton-proton mass ratio                               2.993 717 0308        0.000 000 0025
unified atomic mass unit                               1.660 538 921 e-27    0.000 000 073 e-27    kg
von Klitzing constant                                  25 812.807 4434       0.000 0084            ohm
weak mixing angle                                      0.2223                0.0021
Wien frequency displacement law constant               5.878 9254 e10        0.000 0053 e10        Hz K^-1
Wien wavelength displacement law constant              2.897 7721 e-3        0.000 0026 e-3        m K"""

txt2014 = """\
{220} lattice spacing of silicon                       192.015 5714 e-12     0.000 0032 e-12       m
alpha particle-electron mass ratio                     7294.299 541 36       0.000 000 24
alpha particle mass                                    6.644 657 230 e-27    0.000 000 082 e-27    kg
alpha particle mass energy equivalent                  5.971 920 097 e-10    0.000 000 073 e-10    J
alpha particle mass energy equivalent in MeV           3727.379 378          0.000 023             MeV
alpha particle mass in u                               4.001 506 179 127     0.000 000 000 063     u
alpha particle molar mass                              4.001 506 179 127 e-3 0.000 000 000 063 e-3 kg mol^-1
alpha particle-proton mass ratio                       3.972 599 689 07      0.000 000 000 36
Angstrom star                                          1.000 014 95 e-10     0.000 000 90 e-10     m
atomic mass constant                                   1.660 539 040 e-27    0.000 000 020 e-27    kg
atomic mass constant energy equivalent                 1.492 418 062 e-10    0.000 000 018 e-10    J
atomic mass constant energy equivalent in MeV          931.494 0954          0.000 0057            MeV
atomic mass unit-electron volt relationship            931.494 0954 e6       0.000 0057 e6         eV
atomic mass unit-hartree relationship                  3.423 177 6902 e7     0.000 000 0016 e7     E_h
atomic mass unit-hertz relationship                    2.252 342 7206 e23    0.000 000 0010 e23    Hz
atomic mass unit-inverse meter relationship            7.513 006 6166 e14    0.000 000 0034 e14    m^-1
atomic mass unit-joule relationship                    1.492 418 062 e-10    0.000 000 018 e-10    J
atomic mass unit-kelvin relationship                   1.080 954 38 e13      0.000 000 62 e13      K
atomic mass unit-kilogram relationship                 1.660 539 040 e-27    0.000 000 020 e-27    kg
atomic unit of 1st hyperpolarizability                 3.206 361 329 e-53    0.000 000 020 e-53    C^3 m^3 J^-2
atomic unit of 2nd hyperpolarizability                 6.235 380 085 e-65    0.000 000 077 e-65    C^4 m^4 J^-3
atomic unit of action                                  1.054 571 800 e-34    0.000 000 013 e-34    J s
atomic unit of charge                                  1.602 176 6208 e-19   0.000 000 0098 e-19   C
atomic unit of charge density                          1.081 202 3770 e12    0.000 000 0067 e12    C m^-3
atomic unit of current                                 6.623 618 183 e-3     0.000 000 041 e-3     A
atomic unit of electric dipole mom.                    8.478 353 552 e-30    0.000 000 052 e-30    C m
atomic unit of electric field                          5.142 206 707 e11     0.000 000 032 e11     V m^-1
atomic unit of electric field gradient                 9.717 362 356 e21     0.000 000 060 e21     V m^-2
atomic unit of electric polarizability                 1.648 777 2731 e-41   0.000 000 0011 e-41   C^2 m^2 J^-1
atomic unit of electric potential                      27.211 386 02         0.000 000 17          V
atomic unit of electric quadrupole mom.                4.486 551 484 e-40    0.000 000 028 e-40    C m^2
atomic unit of energy                                  4.359 744 650 e-18    0.000 000 054 e-18    J
atomic unit of force                                   8.238 723 36 e-8      0.000 000 10 e-8      N
atomic unit of length                                  0.529 177 210 67 e-10 0.000 000 000 12 e-10 m
atomic unit of mag. dipole mom.                        1.854 801 999 e-23    0.000 000 011 e-23    J T^-1
atomic unit of mag. flux density                       2.350 517 550 e5      0.000 000 014 e5      T
atomic unit of magnetizability                         7.891 036 5886 e-29   0.000 000 0090 e-29   J T^-2
atomic unit of mass                                    9.109 383 56 e-31     0.000 000 11 e-31     kg
atomic unit of mom.um                                  1.992 851 882 e-24    0.000 000 024 e-24    kg m s^-1
atomic unit of permittivity                            1.112 650 056... e-10 (exact)               F m^-1
atomic unit of time                                    2.418 884 326509e-17  0.000 000 000014e-17  s
atomic unit of velocity                                2.187 691 262 77 e6   0.000 000 000 50 e6   m s^-1
Avogadro constant                                      6.022 140 857 e23     0.000 000 074 e23     mol^-1
Bohr magneton                                          927.400 9994 e-26     0.000 0057 e-26       J T^-1
Bohr magneton in eV/T                                  5.788 381 8012 e-5    0.000 000 0026 e-5    eV T^-1
Bohr magneton in Hz/T                                  13.996 245 042 e9     0.000 000 086 e9      Hz T^-1
Bohr magneton in inverse meters per tesla              46.686 448 14         0.000 000 29          m^-1 T^-1
Bohr magneton in K/T                                   0.671 714 05          0.000 000 39          K T^-1
Bohr radius                                            0.529 177 210 67 e-10 0.000 000 000 12 e-10 m
Boltzmann constant                                     1.380 648 52 e-23     0.000 000 79 e-23     J K^-1
Boltzmann constant in eV/K                             8.617 3303 e-5        0.000 0050 e-5        eV K^-1
Boltzmann constant in Hz/K                             2.083 6612 e10        0.000 0012 e10        Hz K^-1
Boltzmann constant in inverse meters per kelvin        69.503 457            0.000 040             m^-1 K^-1
characteristic impedance of vacuum                     376.730 313 461...    (exact)               ohm
classical electron radius                              2.817 940 3227 e-15   0.000 000 0019 e-15   m
Compton wavelength                                     2.426 310 2367 e-12   0.000 000 0011 e-12   m
Compton wavelength over 2 pi                           386.159 267 64 e-15   0.000 000 18 e-15     m
conductance quantum                                    7.748 091 7310 e-5    0.000 000 0018 e-5    S
conventional value of Josephson constant               483 597.9 e9          (exact)               Hz V^-1
conventional value of von Klitzing constant            25 812.807            (exact)               ohm
Cu x unit                                              1.002 076 97 e-13     0.000 000 28 e-13     m
deuteron-electron mag. mom. ratio                      -4.664 345 535 e-4    0.000 000 026 e-4
deuteron-electron mass ratio                           3670.482 967 85       0.000 000 13
deuteron g factor                                      0.857 438 2311        0.000 000 0048
deuteron mag. mom.                                     0.433 073 5040 e-26   0.000 000 0036 e-26   J T^-1
deuteron mag. mom. to Bohr magneton ratio              0.466 975 4554 e-3    0.000 000 0026 e-3
deuteron mag. mom. to nuclear magneton ratio           0.857 438 2311        0.000 000 0048
deuteron mass                                          3.343 583 719 e-27    0.000 000 041 e-27    kg
deuteron mass energy equivalent                        3.005 063 183 e-10    0.000 000 037 e-10    J
deuteron mass energy equivalent in MeV                 1875.612 928          0.000 012             MeV
deuteron mass in u                                     2.013 553 212 745     0.000 000 000 040     u
deuteron molar mass                                    2.013 553 212 745 e-3 0.000 000 000 040 e-3 kg mol^-1
deuteron-neutron mag. mom. ratio                       -0.448 206 52         0.000 000 11
deuteron-proton mag. mom. ratio                        0.307 012 2077        0.000 000 0015
deuteron-proton mass ratio                             1.999 007 500 87      0.000 000 000 19
deuteron rms charge radius                             2.1413 e-15           0.0025 e-15           m
electric constant                                      8.854 187 817... e-12 (exact)               F m^-1
electron charge to mass quotient                       -1.758 820 024 e11    0.000 000 011 e11     C kg^-1
electron-deuteron mag. mom. ratio                      -2143.923 499         0.000 012
electron-deuteron mass ratio                           2.724 437 107 484 e-4 0.000 000 000 096 e-4
electron g factor                                      -2.002 319 304 361 82 0.000 000 000 000 52
electron gyromag. ratio                                1.760 859 644 e11     0.000 000 011 e11     s^-1 T^-1
electron gyromag. ratio over 2 pi                      28 024.951 64         0.000 17              MHz T^-1
electron-helion mass ratio                             1.819 543 074 854 e-4 0.000 000 000 088 e-4
electron mag. mom.                                     -928.476 4620 e-26    0.000 0057 e-26       J T^-1
electron mag. mom. anomaly                             1.159 652 180 91 e-3  0.000 000 000 26 e-3
electron mag. mom. to Bohr magneton ratio              -1.001 159 652 180 91 0.000 000 000 000 26
electron mag. mom. to nuclear magneton ratio           -1838.281 972 34      0.000 000 17
electron mass                                          9.109 383 56 e-31     0.000 000 11 e-31     kg
electron mass energy equivalent                        8.187 105 65 e-14     0.000 000 10 e-14     J
electron mass energy equivalent in MeV                 0.510 998 9461        0.000 000 0031        MeV
electron mass in u                                     5.485 799 090 70 e-4  0.000 000 000 16 e-4  u
electron molar mass                                    5.485 799 090 70 e-7  0.000 000 000 16 e-7  kg mol^-1
electron-muon mag. mom. ratio                          206.766 9880          0.000 0046
electron-muon mass ratio                               4.836 331 70 e-3      0.000 000 11 e-3
electron-neutron mag. mom. ratio                       960.920 50            0.000 23
electron-neutron mass ratio                            5.438 673 4428 e-4    0.000 000 0027 e-4
electron-proton mag. mom. ratio                        -658.210 6866         0.000 0020
electron-proton mass ratio                             5.446 170 213 52 e-4  0.000 000 000 52 e-4
electron-tau mass ratio                                2.875 92 e-4          0.000 26 e-4
electron to alpha particle mass ratio                  1.370 933 554 798 e-4 0.000 000 000 045 e-4
electron to shielded helion mag. mom. ratio            864.058 257           0.000 010
electron to shielded proton mag. mom. ratio            -658.227 5971         0.000 0072
electron-triton mass ratio                             1.819 200 062 203 e-4 0.000 000 000 084 e-4
electron volt                                          1.602 176 6208 e-19   0.000 000 0098 e-19   J
electron volt-atomic mass unit relationship            1.073 544 1105 e-9    0.000 000 0066 e-9    u
electron volt-hartree relationship                     3.674 932 248 e-2     0.000 000 023 e-2     E_h
electron volt-hertz relationship                       2.417 989 262 e14     0.000 000 015 e14     Hz
electron volt-inverse meter relationship               8.065 544 005 e5      0.000 000 050 e5      m^-1
electron volt-joule relationship                       1.602 176 6208 e-19   0.000 000 0098 e-19   J
electron volt-kelvin relationship                      1.160 452 21 e4       0.000 000 67 e4       K
electron volt-kilogram relationship                    1.782 661 907 e-36    0.000 000 011 e-36    kg
elementary charge                                      1.602 176 6208 e-19   0.000 000 0098 e-19   C
elementary charge over h                               2.417 989 262 e14     0.000 000 015 e14     A J^-1
Faraday constant                                       96 485.332 89         0.000 59              C mol^-1
Faraday constant for conventional electric current     96 485.3251           0.0012                C_90 mol^-1
Fermi coupling constant                                1.166 3787 e-5        0.000 0006 e-5        GeV^-2
fine-structure constant                                7.297 352 5664 e-3    0.000 000 0017 e-3
first radiation constant                               3.741 771 790 e-16    0.000 000 046 e-16    W m^2
first radiation constant for spectral radiance         1.191 042 953 e-16    0.000 000 015 e-16    W m^2 sr^-1
hartree-atomic mass unit relationship                  2.921 262 3197 e-8    0.000 000 0013 e-8    u
hartree-electron volt relationship                     27.211 386 02         0.000 000 17          eV
Hartree energy                                         4.359 744 650 e-18    0.000 000 054 e-18    J
Hartree energy in eV                                   27.211 386 02         0.000 000 17          eV
hartree-hertz relationship                             6.579 683 920 711 e15 0.000 000 000 039 e15 Hz
hartree-inverse meter relationship                     2.194 746 313 702 e7  0.000 000 000 013 e7  m^-1
hartree-joule relationship                             4.359 744 650 e-18    0.000 000 054 e-18    J
hartree-kelvin relationship                            3.157 7513 e5         0.000 0018 e5         K
hartree-kilogram relationship                          4.850 870 129 e-35    0.000 000 060 e-35    kg
helion-electron mass ratio                             5495.885 279 22       0.000 000 27
helion g factor                                        -4.255 250 616        0.000 000 050
helion mag. mom.                                       -1.074 617 522 e-26   0.000 000 014 e-26    J T^-1
helion mag. mom. to Bohr magneton ratio                -1.158 740 958 e-3    0.000 000 014 e-3
helion mag. mom. to nuclear magneton ratio             -2.127 625 308        0.000 000 025
helion mass                                            5.006 412 700 e-27    0.000 000 062 e-27    kg
helion mass energy equivalent                          4.499 539 341 e-10    0.000 000 055 e-10    J
helion mass energy equivalent in MeV                   2808.391 586          0.000 017             MeV
helion mass in u                                       3.014 932 246 73      0.000 000 000 12      u
helion molar mass                                      3.014 932 246 73 e-3  0.000 000 000 12 e-3  kg mol^-1
helion-proton mass ratio                               2.993 152 670 46      0.000 000 000 29
hertz-atomic mass unit relationship                    4.439 821 6616 e-24   0.000 000 0020 e-24   u
hertz-electron volt relationship                       4.135 667 662 e-15    0.000 000 025 e-15    eV
hertz-hartree relationship                             1.5198298460088 e-16  0.0000000000090e-16   E_h
hertz-inverse meter relationship                       3.335 640 951... e-9  (exact)               m^-1
hertz-joule relationship                               6.626 070 040 e-34    0.000 000 081 e-34    J
hertz-kelvin relationship                              4.799 2447 e-11       0.000 0028 e-11       K
hertz-kilogram relationship                            7.372 497 201 e-51    0.000 000 091 e-51    kg
inverse fine-structure constant                        137.035 999 139       0.000 000 031
inverse meter-atomic mass unit relationship            1.331 025 049 00 e-15 0.000 000 000 61 e-15 u
inverse meter-electron volt relationship               1.239 841 9739 e-6    0.000 000 0076 e-6    eV
inverse meter-hartree relationship                     4.556 335 252 767 e-8 0.000 000 000 027 e-8 E_h
inverse meter-hertz relationship                       299 792 458           (exact)               Hz
inverse meter-joule relationship                       1.986 445 824 e-25    0.000 000 024 e-25    J
inverse meter-kelvin relationship                      1.438 777 36 e-2      0.000 000 83 e-2      K
inverse meter-kilogram relationship                    2.210 219 057 e-42    0.000 000 027 e-42    kg
inverse of conductance quantum                         12 906.403 7278       0.000 0029            ohm
Josephson constant                                     483 597.8525 e9       0.0030 e9             Hz V^-1
joule-atomic mass unit relationship                    6.700 535 363 e9      0.000 000 082 e9      u
joule-electron volt relationship                       6.241 509 126 e18     0.000 000 038 e18     eV
joule-hartree relationship                             2.293 712 317 e17     0.000 000 028 e17     E_h
joule-hertz relationship                               1.509 190 205 e33     0.000 000 019 e33     Hz
joule-inverse meter relationship                       5.034 116 651 e24     0.000 000 062 e24     m^-1
joule-kelvin relationship                              7.242 9731 e22        0.000 0042 e22        K
joule-kilogram relationship                            1.112 650 056... e-17 (exact)               kg
kelvin-atomic mass unit relationship                   9.251 0842 e-14       0.000 0053 e-14       u
kelvin-electron volt relationship                      8.617 3303 e-5        0.000 0050 e-5        eV
kelvin-hartree relationship                            3.166 8105 e-6        0.000 0018 e-6        E_h
kelvin-hertz relationship                              2.083 6612 e10        0.000 0012 e10        Hz
kelvin-inverse meter relationship                      69.503 457            0.000 040             m^-1
kelvin-joule relationship                              1.380 648 52 e-23     0.000 000 79 e-23     J
kelvin-kilogram relationship                           1.536 178 65 e-40     0.000 000 88 e-40     kg
kilogram-atomic mass unit relationship                 6.022 140 857 e26     0.000 000 074 e26     u
kilogram-electron volt relationship                    5.609 588 650 e35     0.000 000 034 e35     eV
kilogram-hartree relationship                          2.061 485 823 e34     0.000 000 025 e34     E_h
kilogram-hertz relationship                            1.356 392 512 e50     0.000 000 017 e50     Hz
kilogram-inverse meter relationship                    4.524 438 411 e41     0.000 000 056 e41     m^-1
kilogram-joule relationship                            8.987 551 787... e16  (exact)               J
kilogram-kelvin relationship                           6.509 6595 e39        0.000 0037 e39        K
lattice parameter of silicon                           543.102 0504 e-12     0.000 0089 e-12       m
Loschmidt constant (273.15 K, 100 kPa)                 2.651 6467 e25        0.000 0015 e25        m^-3
Loschmidt constant (273.15 K, 101.325 kPa)             2.686 7811 e25        0.000 0015 e25        m^-3
mag. constant                                          12.566 370 614... e-7 (exact)               N A^-2
mag. flux quantum                                      2.067 833 831 e-15    0.000 000 013 e-15    Wb
molar gas constant                                     8.314 4598            0.000 0048            J mol^-1 K^-1
molar mass constant                                    1 e-3                 (exact)               kg mol^-1
molar mass of carbon-12                                12 e-3                (exact)               kg mol^-1
molar Planck constant                                  3.990 312 7110 e-10   0.000 000 0018 e-10   J s mol^-1
molar Planck constant times c                          0.119 626 565 582     0.000 000 000 054     J m mol^-1
molar volume of ideal gas (273.15 K, 100 kPa)          22.710 947 e-3        0.000 013 e-3         m^3 mol^-1
molar volume of ideal gas (273.15 K, 101.325 kPa)      22.413 962 e-3        0.000 013 e-3         m^3 mol^-1
molar volume of silicon                                12.058 832 14 e-6     0.000 000 61 e-6      m^3 mol^-1
Mo x unit                                              1.002 099 52 e-13     0.000 000 53 e-13     m
muon Compton wavelength                                11.734 441 11 e-15    0.000 000 26 e-15     m
muon Compton wavelength over 2 pi                      1.867 594 308 e-15    0.000 000 042 e-15    m
muon-electron mass ratio                               206.768 2826          0.000 0046
muon g factor                                          -2.002 331 8418       0.000 000 0013
muon mag. mom.                                         -4.490 448 26 e-26    0.000 000 10 e-26     J T^-1
muon mag. mom. anomaly                                 1.165 920 89 e-3      0.000 000 63 e-3
muon mag. mom. to Bohr magneton ratio                  -4.841 970 48 e-3     0.000 000 11 e-3
muon mag. mom. to nuclear magneton ratio               -8.890 597 05         0.000 000 20
muon mass                                              1.883 531 594 e-28    0.000 000 048 e-28    kg
muon mass energy equivalent                            1.692 833 774 e-11    0.000 000 043 e-11    J
muon mass energy equivalent in MeV                     105.658 3745          0.000 0024            MeV
muon mass in u                                         0.113 428 9257        0.000 000 0025        u
muon molar mass                                        0.113 428 9257 e-3    0.000 000 0025 e-3    kg mol^-1
muon-neutron mass ratio                                0.112 454 5167        0.000 000 0025
muon-proton mag. mom. ratio                            -3.183 345 142        0.000 000 071
muon-proton mass ratio                                 0.112 609 5262        0.000 000 0025
muon-tau mass ratio                                    5.946 49 e-2          0.000 54 e-2
natural unit of action                                 1.054 571 800 e-34    0.000 000 013 e-34    J s
natural unit of action in eV s                         6.582 119 514 e-16    0.000 000 040 e-16    eV s
natural unit of energy                                 8.187 105 65 e-14     0.000 000 10 e-14     J
natural unit of energy in MeV                          0.510 998 9461        0.000 000 0031        MeV
natural unit of length                                 386.159 267 64 e-15   0.000 000 18 e-15     m
natural unit of mass                                   9.109 383 56 e-31     0.000 000 11 e-31     kg
natural unit of mom.um                                 2.730 924 488 e-22    0.000 000 034 e-22    kg m s^-1
natural unit of mom.um in MeV/c                        0.510 998 9461        0.000 000 0031        MeV/c
natural unit of time                                   1.288 088 667 12 e-21 0.000 000 000 58 e-21 s
natural unit of velocity                               299 792 458           (exact)               m s^-1
neutron Compton wavelength                             1.319 590 904 81 e-15 0.000 000 000 88 e-15 m
neutron Compton wavelength over 2 pi                   0.210 019 415 36 e-15 0.000 000 000 14 e-15 m
neutron-electron mag. mom. ratio                       1.040 668 82 e-3      0.000 000 25 e-3
neutron-electron mass ratio                            1838.683 661 58       0.000 000 90
neutron g factor                                       -3.826 085 45         0.000 000 90
neutron gyromag. ratio                                 1.832 471 72 e8       0.000 000 43 e8       s^-1 T^-1
neutron gyromag. ratio over 2 pi                       29.164 6933           0.000 0069            MHz T^-1
neutron mag. mom.                                      -0.966 236 50 e-26    0.000 000 23 e-26     J T^-1
neutron mag. mom. to Bohr magneton ratio               -1.041 875 63 e-3     0.000 000 25 e-3
neutron mag. mom. to nuclear magneton ratio            -1.913 042 73         0.000 000 45
neutron mass                                           1.674 927 471 e-27    0.000 000 021 e-27    kg
neutron mass energy equivalent                         1.505 349 739 e-10    0.000 000 019 e-10    J
neutron mass energy equivalent in MeV                  939.565 4133          0.000 0058            MeV
neutron mass in u                                      1.008 664 915 88      0.000 000 000 49      u
neutron molar mass                                     1.008 664 915 88 e-3  0.000 000 000 49 e-3  kg mol^-1
neutron-muon mass ratio                                8.892 484 08          0.000 000 20
neutron-proton mag. mom. ratio                         -0.684 979 34         0.000 000 16
neutron-proton mass difference                         2.305 573 77 e-30     0.000 000 85 e-30
neutron-proton mass difference energy equivalent       2.072 146 37 e-13     0.000 000 76 e-13
neutron-proton mass difference energy equivalent in MeV 1.293 332 05         0.000 000 48
neutron-proton mass difference in u                    0.001 388 449 00      0.000 000 000 51
neutron-proton mass ratio                              1.001 378 418 98      0.000 000 000 51
neutron-tau mass ratio                                 0.528 790             0.000 048
neutron to shielded proton mag. mom. ratio             -0.684 996 94         0.000 000 16
Newtonian constant of gravitation                      6.674 08 e-11         0.000 31 e-11         m^3 kg^-1 s^-2
Newtonian constant of gravitation over h-bar c         6.708 61 e-39         0.000 31 e-39         (GeV/c^2)^-2
nuclear magneton                                       5.050 783 699 e-27    0.000 000 031 e-27    J T^-1
nuclear magneton in eV/T                               3.152 451 2550 e-8    0.000 000 0015 e-8    eV T^-1
nuclear magneton in inverse meters per tesla           2.542 623 432 e-2     0.000 000 016 e-2     m^-1 T^-1
nuclear magneton in K/T                                3.658 2690 e-4        0.000 0021 e-4        K T^-1
nuclear magneton in MHz/T                              7.622 593 285         0.000 000 047         MHz T^-1
Planck constant                                        6.626 070 040 e-34    0.000 000 081 e-34    J s
Planck constant in eV s                                4.135 667 662 e-15    0.000 000 025 e-15    eV s
Planck constant over 2 pi                              1.054 571 800 e-34    0.000 000 013 e-34    J s
Planck constant over 2 pi in eV s                      6.582 119 514 e-16    0.000 000 040 e-16    eV s
Planck constant over 2 pi times c in MeV fm            197.326 9788          0.000 0012            MeV fm
Planck length                                          1.616 229 e-35        0.000 038 e-35        m
Planck mass                                            2.176 470 e-8         0.000 051 e-8         kg
Planck mass energy equivalent in GeV                   1.220 910 e19         0.000 029 e19         GeV
Planck temperature                                     1.416 808 e32         0.000 033 e32         K
Planck time                                            5.391 16 e-44         0.000 13 e-44         s
proton charge to mass quotient                         9.578 833 226 e7      0.000 000 059 e7      C kg^-1
proton Compton wavelength                              1.321 409 853 96 e-15 0.000 000 000 61 e-15 m
proton Compton wavelength over 2 pi                    0.210 308910109e-15   0.000 000 000097e-15  m
proton-electron mass ratio                             1836.152 673 89       0.000 000 17
proton g factor                                        5.585 694 702         0.000 000 017
proton gyromag. ratio                                  2.675 221 900 e8      0.000 000 018 e8      s^-1 T^-1
proton gyromag. ratio over 2 pi                        42.577 478 92         0.000 000 29          MHz T^-1
proton mag. mom.                                       1.410 606 7873 e-26   0.000 000 0097 e-26   J T^-1
proton mag. mom. to Bohr magneton ratio                1.521 032 2053 e-3    0.000 000 0046 e-3
proton mag. mom. to nuclear magneton ratio             2.792 847 3508        0.000 000 0085
proton mag. shielding correction                       25.691 e-6            0.011 e-6
proton mass                                            1.672 621 898 e-27    0.000 000 021 e-27    kg
proton mass energy equivalent                          1.503 277 593 e-10    0.000 000 018 e-10    J
proton mass energy equivalent in MeV                   938.272 0813          0.000 0058            MeV
proton mass in u                                       1.007 276 466 879     0.000 000 000 091     u
proton molar mass                                      1.007 276 466 879 e-3 0.000 000 000 091 e-3 kg mol^-1
proton-muon mass ratio                                 8.880 243 38          0.000 000 20
proton-neutron mag. mom. ratio                         -1.459 898 05         0.000 000 34
proton-neutron mass ratio                              0.998 623 478 44      0.000 000 000 51
proton rms charge radius                               0.8751 e-15           0.0061 e-15           m
proton-tau mass ratio                                  0.528 063             0.000 048
quantum of circulation                                 3.636 947 5486 e-4    0.000 000 0017 e-4    m^2 s^-1
quantum of circulation times 2                         7.273 895 0972 e-4    0.000 000 0033 e-4    m^2 s^-1
Rydberg constant                                       10 973 731.568 508    0.000 065             m^-1
Rydberg constant times c in Hz                         3.289 841 960 355 e15 0.000 000 000 019 e15 Hz
Rydberg constant times hc in eV                        13.605 693 009        0.000 000 084         eV
Rydberg constant times hc in J                         2.179 872 325 e-18    0.000 000 027 e-18    J
Sackur-Tetrode constant (1 K, 100 kPa)                 -1.151 7084           0.000 0014
Sackur-Tetrode constant (1 K, 101.325 kPa)             -1.164 8714           0.000 0014
second radiation constant                              1.438 777 36 e-2      0.000 000 83 e-2      m K
shielded helion gyromag. ratio                         2.037 894 585 e8      0.000 000 027 e8      s^-1 T^-1
shielded helion gyromag. ratio over 2 pi               32.434 099 66         0.000 000 43          MHz T^-1
shielded helion mag. mom.                              -1.074 553 080 e-26   0.000 000 014 e-26    J T^-1
shielded helion mag. mom. to Bohr magneton ratio       -1.158 671 471 e-3    0.000 000 014 e-3
shielded helion mag. mom. to nuclear magneton ratio    -2.127 497 720        0.000 000 025
shielded helion to proton mag. mom. ratio              -0.761 766 5603       0.000 000 0092
shielded helion to shielded proton mag. mom. ratio     -0.761 786 1313       0.000 000 0033
shielded proton gyromag. ratio                         2.675 153 171 e8      0.000 000 033 e8      s^-1 T^-1
shielded proton gyromag. ratio over 2 pi               42.576 385 07         0.000 000 53          MHz T^-1
shielded proton mag. mom.                              1.410 570 547 e-26    0.000 000 018 e-26    J T^-1
shielded proton mag. mom. to Bohr magneton ratio       1.520 993 128 e-3     0.000 000 017 e-3
shielded proton mag. mom. to nuclear magneton ratio    2.792 775 600         0.000 000 030
speed of light in vacuum                               299 792 458           (exact)               m s^-1
standard acceleration of gravity                       9.806 65              (exact)               m s^-2
standard atmosphere                                    101 325               (exact)               Pa
standard-state pressure                                100 000               (exact)               Pa
Stefan-Boltzmann constant                              5.670 367 e-8         0.000 013 e-8         W m^-2 K^-4
tau Compton wavelength                                 0.697 787 e-15        0.000 063 e-15        m
tau Compton wavelength over 2 pi                       0.111 056 e-15        0.000 010 e-15        m
tau-electron mass ratio                                3477.15               0.31
tau mass                                               3.167 47 e-27         0.000 29 e-27         kg
tau mass energy equivalent                             2.846 78 e-10         0.000 26 e-10         J
tau mass energy equivalent in MeV                      1776.82               0.16                  MeV
tau mass in u                                          1.907 49              0.000 17              u
tau molar mass                                         1.907 49 e-3          0.000 17 e-3          kg mol^-1
tau-muon mass ratio                                    16.8167               0.0015
tau-neutron mass ratio                                 1.891 11              0.000 17
tau-proton mass ratio                                  1.893 72              0.000 17
Thomson cross section                                  0.665 245 871 58 e-28 0.000 000 000 91 e-28 m^2
triton-electron mass ratio                             5496.921 535 88       0.000 000 26
triton g factor                                        5.957 924 920         0.000 000 028
triton mag. mom.                                       1.504 609 503 e-26    0.000 000 012 e-26    J T^-1
triton mag. mom. to Bohr magneton ratio                1.622 393 6616 e-3    0.000 000 0076 e-3
triton mag. mom. to nuclear magneton ratio             2.978 962 460         0.000 000 014
triton mass                                            5.007 356 665 e-27    0.000 000 062 e-27    kg
triton mass energy equivalent                          4.500 387 735 e-10    0.000 000 055 e-10    J
triton mass energy equivalent in MeV                   2808.921 112          0.000 017             MeV
triton mass in u                                       3.015 500 716 32      0.000 000 000 11      u
triton molar mass                                      3.015 500 716 32 e-3  0.000 000 000 11 e-3  kg mol^-1
triton-proton mass ratio                               2.993 717 033 48      0.000 000 000 22
unified atomic mass unit                               1.660 539 040 e-27    0.000 000 020 e-27    kg
von Klitzing constant                                  25 812.807 4555       0.000 0059            ohm
weak mixing angle                                      0.2223                0.0021
Wien frequency displacement law constant               5.878 9238 e10        0.000 0034 e10        Hz K^-1
Wien wavelength displacement law constant              2.897 7729 e-3        0.000 0017 e-3        m K"""

txt2018 = """\
alpha particle-electron mass ratio                          7294.299 541 42          0.000 000 24
alpha particle mass                                         6.644 657 3357 e-27      0.000 000 0020 e-27      kg
alpha particle mass energy equivalent                       5.971 920 1914 e-10      0.000 000 0018 e-10      J
alpha particle mass energy equivalent in MeV                3727.379 4066            0.000 0011               MeV
alpha particle mass in u                                    4.001 506 179 127        0.000 000 000 063        u
alpha particle molar mass                                   4.001 506 1777 e-3       0.000 000 0012 e-3       kg mol^-1
alpha particle-proton mass ratio                            3.972 599 690 09         0.000 000 000 22
alpha particle relative atomic mass                         4.001 506 179 127        0.000 000 000 063
Angstrom star                                               1.000 014 95 e-10        0.000 000 90 e-10        m
atomic mass constant                                        1.660 539 066 60 e-27    0.000 000 000 50 e-27    kg
atomic mass constant energy equivalent                      1.492 418 085 60 e-10    0.000 000 000 45 e-10    J
atomic mass constant energy equivalent in MeV               931.494 102 42           0.000 000 28             MeV
atomic mass unit-electron volt relationship                 9.314 941 0242 e8        0.000 000 0028 e8        eV
atomic mass unit-hartree relationship                       3.423 177 6874 e7        0.000 000 0010 e7        E_h
atomic mass unit-hertz relationship                         2.252 342 718 71 e23     0.000 000 000 68 e23     Hz
atomic mass unit-inverse meter relationship                 7.513 006 6104 e14       0.000 000 0023 e14       m^-1
atomic mass unit-joule relationship                         1.492 418 085 60 e-10    0.000 000 000 45 e-10    J
atomic mass unit-kelvin relationship                        1.080 954 019 16 e13     0.000 000 000 33 e13     K
atomic mass unit-kilogram relationship                      1.660 539 066 60 e-27    0.000 000 000 50 e-27    kg
atomic unit of 1st hyperpolarizability                      3.206 361 3061 e-53      0.000 000 0015 e-53      C^3 m^3 J^-2
atomic unit of 2nd hyperpolarizability                      6.235 379 9905 e-65      0.000 000 0038 e-65      C^4 m^4 J^-3
atomic unit of action                                       1.054 571 817... e-34    (exact)                  J s
atomic unit of charge                                       1.602 176 634 e-19       (exact)                  C
atomic unit of charge density                               1.081 202 384 57 e12     0.000 000 000 49 e12     C m^-3
atomic unit of current                                      6.623 618 237 510 e-3    0.000 000 000 013 e-3    A
atomic unit of electric dipole mom.                         8.478 353 6255 e-30      0.000 000 0013 e-30      C m
atomic unit of electric field                               5.142 206 747 63 e11     0.000 000 000 78 e11     V m^-1
atomic unit of electric field gradient                      9.717 362 4292 e21       0.000 000 0029 e21       V m^-2
atomic unit of electric polarizability                      1.648 777 274 36 e-41    0.000 000 000 50 e-41    C^2 m^2 J^-1
atomic unit of electric potential                           27.211 386 245 988       0.000 000 000 053        V
atomic unit of electric quadrupole mom.                     4.486 551 5246 e-40      0.000 000 0014 e-40      C m^2
atomic unit of energy                                       4.359 744 722 2071 e-18  0.000 000 000 0085 e-18  J
atomic unit of force                                        8.238 723 4983 e-8       0.000 000 0012 e-8       N
atomic unit of length                                       5.291 772 109 03 e-11    0.000 000 000 80 e-11    m
atomic unit of mag. dipole mom.                             1.854 802 015 66 e-23    0.000 000 000 56 e-23    J T^-1
atomic unit of mag. flux density                            2.350 517 567 58 e5      0.000 000 000 71 e5      T
atomic unit of magnetizability                              7.891 036 6008 e-29      0.000 000 0048 e-29      J T^-2
atomic unit of mass                                         9.109 383 7015 e-31      0.000 000 0028 e-31      kg
atomic unit of momentum                                     1.992 851 914 10 e-24    0.000 000 000 30 e-24    kg m s^-1
atomic unit of permittivity                                 1.112 650 055 45 e-10    0.000 000 000 17 e-10    F m^-1
atomic unit of time                                         2.418 884 326 5857 e-17  0.000 000 000 0047 e-17  s
atomic unit of velocity                                     2.187 691 263 64 e6      0.000 000 000 33 e6      m s^-1
Avogadro constant                                           6.022 140 76 e23         (exact)                  mol^-1
Bohr magneton                                               9.274 010 0783 e-24      0.000 000 0028 e-24      J T^-1
Bohr magneton in eV/T                                       5.788 381 8060 e-5       0.000 000 0017 e-5       eV T^-1
Bohr magneton in Hz/T                                       1.399 624 493 61 e10     0.000 000 000 42 e10     Hz T^-1
Bohr magneton in inverse meter per tesla                    46.686 447 783           0.000 000 014            m^-1 T^-1
Bohr magneton in K/T                                        0.671 713 815 63         0.000 000 000 20         K T^-1
Bohr radius                                                 5.291 772 109 03 e-11    0.000 000 000 80 e-11    m
Boltzmann constant                                          1.380 649 e-23           (exact)                  J K^-1
Boltzmann constant in eV/K                                  8.617 333 262... e-5     (exact)                  eV K^-1
Boltzmann constant in Hz/K                                  2.083 661 912... e10     (exact)                  Hz K^-1
Boltzmann constant in inverse meter per kelvin              69.503 480 04...         (exact)                  m^-1 K^-1
classical electron radius                                   2.817 940 3262 e-15      0.000 000 0013 e-15      m
Compton wavelength                                          2.426 310 238 67 e-12    0.000 000 000 73 e-12    m
conductance quantum                                         7.748 091 729... e-5     (exact)                  S
conventional value of ampere-90                             1.000 000 088 87...      (exact)                  A
conventional value of coulomb-90                            1.000 000 088 87...      (exact)                  C
conventional value of farad-90                              0.999 999 982 20...      (exact)                  F
conventional value of henry-90                              1.000 000 017 79...      (exact)                  H
conventional value of Josephson constant                    483 597.9 e9             (exact)                  Hz V^-1
conventional value of ohm-90                                1.000 000 017 79...      (exact)                  ohm
conventional value of volt-90                               1.000 000 106 66...      (exact)                  V
conventional value of von Klitzing constant                 25 812.807               (exact)                  ohm
conventional value of watt-90                               1.000 000 195 53...      (exact)                  W
Cu x unit                                                   1.002 076 97 e-13        0.000 000 28 e-13        m
deuteron-electron mag. mom. ratio                           -4.664 345 551 e-4       0.000 000 012 e-4
deuteron-electron mass ratio                                3670.482 967 88          0.000 000 13
deuteron g factor                                           0.857 438 2338           0.000 000 0022
deuteron mag. mom.                                          4.330 735 094 e-27       0.000 000 011 e-27       J T^-1
deuteron mag. mom. to Bohr magneton ratio                   4.669 754 570 e-4        0.000 000 012 e-4
deuteron mag. mom. to nuclear magneton ratio                0.857 438 2338           0.000 000 0022
deuteron mass                                               3.343 583 7724 e-27      0.000 000 0010 e-27      kg
deuteron mass energy equivalent                             3.005 063 231 02 e-10    0.000 000 000 91 e-10    J
deuteron mass energy equivalent in MeV                      1875.612 942 57          0.000 000 57             MeV
deuteron mass in u                                          2.013 553 212 745        0.000 000 000 040        u
deuteron molar mass                                         2.013 553 212 05 e-3     0.000 000 000 61 e-3     kg mol^-1
deuteron-neutron mag. mom. ratio                            -0.448 206 53            0.000 000 11
deuteron-proton mag. mom. ratio                             0.307 012 209 39         0.000 000 000 79
deuteron-proton mass ratio                                  1.999 007 501 39         0.000 000 000 11
deuteron relative atomic mass                               2.013 553 212 745        0.000 000 000 040
deuteron rms charge radius                                  2.127 99 e-15            0.000 74 e-15            m
electron charge to mass quotient                            -1.758 820 010 76 e11    0.000 000 000 53 e11     C kg^-1
electron-deuteron mag. mom. ratio                           -2143.923 4915           0.000 0056
electron-deuteron mass ratio                                2.724 437 107 462 e-4    0.000 000 000 096 e-4
electron g factor                                           -2.002 319 304 362 56    0.000 000 000 000 35
electron gyromag. ratio                                     1.760 859 630 23 e11     0.000 000 000 53 e11     s^-1 T^-1
electron gyromag. ratio in MHz/T                            28 024.951 4242          0.000 0085               MHz T^-1
electron-helion mass ratio                                  1.819 543 074 573 e-4    0.000 000 000 079 e-4
electron mag. mom.                                          -9.284 764 7043 e-24     0.000 000 0028 e-24      J T^-1
electron mag. mom. anomaly                                  1.159 652 181 28 e-3     0.000 000 000 18 e-3
electron mag. mom. to Bohr magneton ratio                   -1.001 159 652 181 28    0.000 000 000 000 18
electron mag. mom. to nuclear magneton ratio                -1838.281 971 88         0.000 000 11
electron mass                                               9.109 383 7015 e-31      0.000 000 0028 e-31      kg
electron mass energy equivalent                             8.187 105 7769 e-14      0.000 000 0025 e-14      J
electron mass energy equivalent in MeV                      0.510 998 950 00         0.000 000 000 15         MeV
electron mass in u                                          5.485 799 090 65 e-4     0.000 000 000 16 e-4     u
electron molar mass                                         5.485 799 0888 e-7       0.000 000 0017 e-7       kg mol^-1
electron-muon mag. mom. ratio                               206.766 9883             0.000 0046
electron-muon mass ratio                                    4.836 331 69 e-3         0.000 000 11 e-3
electron-neutron mag. mom. ratio                            960.920 50               0.000 23
electron-neutron mass ratio                                 5.438 673 4424 e-4       0.000 000 0026 e-4
electron-proton mag. mom. ratio                             -658.210 687 89          0.000 000 20
electron-proton mass ratio                                  5.446 170 214 87 e-4     0.000 000 000 33 e-4
electron relative atomic mass                               5.485 799 090 65 e-4     0.000 000 000 16 e-4
electron-tau mass ratio                                     2.875 85 e-4             0.000 19 e-4
electron to alpha particle mass ratio                       1.370 933 554 787 e-4    0.000 000 000 045 e-4
electron to shielded helion mag. mom. ratio                 864.058 257              0.000 010
electron to shielded proton mag. mom. ratio                 -658.227 5971            0.000 0072
electron-triton mass ratio                                  1.819 200 062 251 e-4    0.000 000 000 090 e-4
electron volt                                               1.602 176 634 e-19       (exact)                  J
electron volt-atomic mass unit relationship                 1.073 544 102 33 e-9     0.000 000 000 32 e-9     u
electron volt-hartree relationship                          3.674 932 217 5655 e-2   0.000 000 000 0071 e-2   E_h
electron volt-hertz relationship                            2.417 989 242... e14     (exact)                  Hz
electron volt-inverse meter relationship                    8.065 543 937... e5      (exact)                  m^-1
electron volt-joule relationship                            1.602 176 634 e-19       (exact)                  J
electron volt-kelvin relationship                           1.160 451 812... e4      (exact)                  K
electron volt-kilogram relationship                         1.782 661 921... e-36    (exact)                  kg
elementary charge                                           1.602 176 634 e-19       (exact)                  C
elementary charge over h-bar                                1.519 267 447... e15     (exact)                  A J^-1
Faraday constant                                            96 485.332 12...         (exact)                  C mol^-1
Fermi coupling constant                                     1.166 3787 e-5           0.000 0006 e-5           GeV^-2
fine-structure constant                                     7.297 352 5693 e-3       0.000 000 0011 e-3
first radiation constant                                    3.741 771 852... e-16    (exact)                  W m^2
first radiation constant for spectral radiance              1.191 042 972... e-16    (exact)                  W m^2 sr^-1
hartree-atomic mass unit relationship                       2.921 262 322 05 e-8     0.000 000 000 88 e-8     u
hartree-electron volt relationship                          27.211 386 245 988       0.000 000 000 053        eV
Hartree energy                                              4.359 744 722 2071 e-18  0.000 000 000 0085 e-18  J
Hartree energy in eV                                        27.211 386 245 988       0.000 000 000 053        eV
hartree-hertz relationship                                  6.579 683 920 502 e15    0.000 000 000 013 e15    Hz
hartree-inverse meter relationship                          2.194 746 313 6320 e7    0.000 000 000 0043 e7    m^-1
hartree-joule relationship                                  4.359 744 722 2071 e-18  0.000 000 000 0085 e-18  J
hartree-kelvin relationship                                 3.157 750 248 0407 e5    0.000 000 000 0061 e5    K
hartree-kilogram relationship                               4.850 870 209 5432 e-35  0.000 000 000 0094 e-35  kg
helion-electron mass ratio                                  5495.885 280 07          0.000 000 24
helion g factor                                             -4.255 250 615           0.000 000 050
helion mag. mom.                                            -1.074 617 532 e-26      0.000 000 013 e-26       J T^-1
helion mag. mom. to Bohr magneton ratio                     -1.158 740 958 e-3       0.000 000 014 e-3
helion mag. mom. to nuclear magneton ratio                  -2.127 625 307           0.000 000 025
helion mass                                                 5.006 412 7796 e-27      0.000 000 0015 e-27      kg
helion mass energy equivalent                               4.499 539 4125 e-10      0.000 000 0014 e-10      J
helion mass energy equivalent in MeV                        2808.391 607 43          0.000 000 85             MeV
helion mass in u                                            3.014 932 247 175        0.000 000 000 097        u
helion molar mass                                           3.014 932 246 13 e-3     0.000 000 000 91 e-3     kg mol^-1
helion-proton mass ratio                                    2.993 152 671 67         0.000 000 000 13
helion relative atomic mass                                 3.014 932 247 175        0.000 000 000 097
helion shielding shift                                      5.996 743 e-5            0.000 010 e-5
hertz-atomic mass unit relationship                         4.439 821 6652 e-24      0.000 000 0013 e-24      u
hertz-electron volt relationship                            4.135 667 696... e-15    (exact)                  eV
hertz-hartree relationship                                  1.519 829 846 0570 e-16  0.000 000 000 0029 e-16  E_h
hertz-inverse meter relationship                            3.335 640 951... e-9     (exact)                  m^-1
hertz-joule relationship                                    6.626 070 15 e-34        (exact)                  J
hertz-kelvin relationship                                   4.799 243 073... e-11    (exact)                  K
hertz-kilogram relationship                                 7.372 497 323... e-51    (exact)                  kg
hyperfine transition frequency of Cs-133                    9 192 631 770            (exact)                  Hz
inverse fine-structure constant                             137.035 999 084          0.000 000 021
inverse meter-atomic mass unit relationship                 1.331 025 050 10 e-15    0.000 000 000 40 e-15    u
inverse meter-electron volt relationship                    1.239 841 984... e-6     (exact)                  eV
inverse meter-hartree relationship                          4.556 335 252 9120 e-8   0.000 000 000 0088 e-8   E_h
inverse meter-hertz relationship                            299 792 458              (exact)                  Hz
inverse meter-joule relationship                            1.986 445 857... e-25    (exact)                  J
inverse meter-kelvin relationship                           1.438 776 877... e-2     (exact)                  K
inverse meter-kilogram relationship                         2.210 219 094... e-42    (exact)                  kg
inverse of conductance quantum                              12 906.403 72...         (exact)                  ohm
Josephson constant                                          483 597.848 4... e9      (exact)                  Hz V^-1
joule-atomic mass unit relationship                         6.700 535 2565 e9        0.000 000 0020 e9        u
joule-electron volt relationship                            6.241 509 074... e18     (exact)                  eV
joule-hartree relationship                                  2.293 712 278 3963 e17   0.000 000 000 0045 e17   E_h
joule-hertz relationship                                    1.509 190 179... e33     (exact)                  Hz
joule-inverse meter relationship                            5.034 116 567... e24     (exact)                  m^-1
joule-kelvin relationship                                   7.242 970 516... e22     (exact)                  K
joule-kilogram relationship                                 1.112 650 056... e-17    (exact)                  kg
kelvin-atomic mass unit relationship                        9.251 087 3014 e-14      0.000 000 0028 e-14      u
kelvin-electron volt relationship                           8.617 333 262... e-5     (exact)                  eV
kelvin-hartree relationship                                 3.166 811 563 4556 e-6   0.000 000 000 0061 e-6   E_h
kelvin-hertz relationship                                   2.083 661 912... e10     (exact)                  Hz
kelvin-inverse meter relationship                           69.503 480 04...         (exact)                  m^-1
kelvin-joule relationship                                   1.380 649 e-23           (exact)                  J
kelvin-kilogram relationship                                1.536 179 187... e-40    (exact)                  kg
kilogram-atomic mass unit relationship                      6.022 140 7621 e26       0.000 000 0018 e26       u
kilogram-electron volt relationship                         5.609 588 603... e35     (exact)                  eV
kilogram-hartree relationship                               2.061 485 788 7409 e34   0.000 000 000 0040 e34   E_h
kilogram-hertz relationship                                 1.356 392 489... e50     (exact)                  Hz
kilogram-inverse meter relationship                         4.524 438 335... e41     (exact)                  m^-1
kilogram-joule relationship                                 8.987 551 787... e16     (exact)                  J
kilogram-kelvin relationship                                6.509 657 260... e39     (exact)                  K
lattice parameter of silicon                                5.431 020 511 e-10       0.000 000 089 e-10       m
lattice spacing of ideal Si (220)                           1.920 155 716 e-10       0.000 000 032 e-10       m
Loschmidt constant (273.15 K, 100 kPa)                      2.651 645 804... e25     (exact)                  m^-3
Loschmidt constant (273.15 K, 101.325 kPa)                  2.686 780 111... e25     (exact)                  m^-3
luminous efficacy                                           683                      (exact)                  lm W^-1
mag. flux quantum                                           2.067 833 848... e-15    (exact)                  Wb
molar gas constant                                          8.314 462 618...         (exact)                  J mol^-1 K^-1
molar mass constant                                         0.999 999 999 65 e-3     0.000 000 000 30 e-3     kg mol^-1
molar mass of carbon-12                                     11.999 999 9958 e-3      0.000 000 0036 e-3       kg mol^-1
molar Planck constant                                       3.990 312 712... e-10    (exact)                  J Hz^-1 mol^-1
molar volume of ideal gas (273.15 K, 100 kPa)               22.710 954 64... e-3     (exact)                  m^3 mol^-1
molar volume of ideal gas (273.15 K, 101.325 kPa)           22.413 969 54... e-3     (exact)                  m^3 mol^-1
molar volume of silicon                                     1.205 883 199 e-5        0.000 000 060 e-5        m^3 mol^-1
Mo x unit                                                   1.002 099 52 e-13        0.000 000 53 e-13        m
muon Compton wavelength                                     1.173 444 110 e-14       0.000 000 026 e-14       m
muon-electron mass ratio                                    206.768 2830             0.000 0046
muon g factor                                               -2.002 331 8418          0.000 000 0013
muon mag. mom.                                              -4.490 448 30 e-26       0.000 000 10 e-26        J T^-1
muon mag. mom. anomaly                                      1.165 920 89 e-3         0.000 000 63 e-3
muon mag. mom. to Bohr magneton ratio                       -4.841 970 47 e-3        0.000 000 11 e-3
muon mag. mom. to nuclear magneton ratio                    -8.890 597 03            0.000 000 20
muon mass                                                   1.883 531 627 e-28       0.000 000 042 e-28       kg
muon mass energy equivalent                                 1.692 833 804 e-11       0.000 000 038 e-11       J
muon mass energy equivalent in MeV                          105.658 3755             0.000 0023               MeV
muon mass in u                                              0.113 428 9259           0.000 000 0025           u
muon molar mass                                             1.134 289 259 e-4        0.000 000 025 e-4        kg mol^-1
muon-neutron mass ratio                                     0.112 454 5170           0.000 000 0025
muon-proton mag. mom. ratio                                 -3.183 345 142           0.000 000 071
muon-proton mass ratio                                      0.112 609 5264           0.000 000 0025
muon-tau mass ratio                                         5.946 35 e-2             0.000 40 e-2
natural unit of action                                      1.054 571 817... e-34    (exact)                  J s
natural unit of action in eV s                              6.582 119 569... e-16    (exact)                  eV s
natural unit of energy                                      8.187 105 7769 e-14      0.000 000 0025 e-14      J
natural unit of energy in MeV                               0.510 998 950 00         0.000 000 000 15         MeV
natural unit of length                                      3.861 592 6796 e-13      0.000 000 0012 e-13      m
natural unit of mass                                        9.109 383 7015 e-31      0.000 000 0028 e-31      kg
natural unit of momentum                                    2.730 924 530 75 e-22    0.000 000 000 82 e-22    kg m s^-1
natural unit of momentum in MeV/c                           0.510 998 950 00         0.000 000 000 15         MeV/c
natural unit of time                                        1.288 088 668 19 e-21    0.000 000 000 39 e-21    s
natural unit of velocity                                    299 792 458              (exact)                  m s^-1
neutron Compton wavelength                                  1.319 590 905 81 e-15    0.000 000 000 75 e-15    m
neutron-electron mag. mom. ratio                            1.040 668 82 e-3         0.000 000 25 e-3
neutron-electron mass ratio                                 1838.683 661 73          0.000 000 89
neutron g factor                                            -3.826 085 45            0.000 000 90
neutron gyromag. ratio                                      1.832 471 71 e8          0.000 000 43 e8          s^-1 T^-1
neutron gyromag. ratio in MHz/T                             29.164 6931              0.000 0069               MHz T^-1
neutron mag. mom.                                           -9.662 3651 e-27         0.000 0023 e-27          J T^-1
neutron mag. mom. to Bohr magneton ratio                    -1.041 875 63 e-3        0.000 000 25 e-3
neutron mag. mom. to nuclear magneton ratio                 -1.913 042 73            0.000 000 45
neutron mass                                                1.674 927 498 04 e-27    0.000 000 000 95 e-27    kg
neutron mass energy equivalent                              1.505 349 762 87 e-10    0.000 000 000 86 e-10    J
neutron mass energy equivalent in MeV                       939.565 420 52           0.000 000 54             MeV
neutron mass in u                                           1.008 664 915 95         0.000 000 000 49         u
neutron molar mass                                          1.008 664 915 60 e-3     0.000 000 000 57 e-3     kg mol^-1
neutron-muon mass ratio                                     8.892 484 06             0.000 000 20
neutron-proton mag. mom. ratio                              -0.684 979 34            0.000 000 16
neutron-proton mass difference                              2.305 574 35 e-30        0.000 000 82 e-30        kg
neutron-proton mass difference energy equivalent            2.072 146 89 e-13        0.000 000 74 e-13        J
neutron-proton mass difference energy equivalent in MeV     1.293 332 36             0.000 000 46             MeV
neutron-proton mass difference in u                         1.388 449 33 e-3         0.000 000 49 e-3         u
neutron-proton mass ratio                                   1.001 378 419 31         0.000 000 000 49
neutron relative atomic mass                                1.008 664 915 95         0.000 000 000 49
neutron-tau mass ratio                                      0.528 779                0.000 036
neutron to shielded proton mag. mom. ratio                  -0.684 996 94            0.000 000 16
Newtonian constant of gravitation                           6.674 30 e-11            0.000 15 e-11            m^3 kg^-1 s^-2
Newtonian constant of gravitation over h-bar c              6.708 83 e-39            0.000 15 e-39            (GeV/c^2)^-2
nuclear magneton                                            5.050 783 7461 e-27      0.000 000 0015 e-27      J T^-1
nuclear magneton in eV/T                                    3.152 451 258 44 e-8     0.000 000 000 96 e-8     eV T^-1
nuclear magneton in inverse meter per tesla                 2.542 623 413 53 e-2     0.000 000 000 78 e-2     m^-1 T^-1
nuclear magneton in K/T                                     3.658 267 7756 e-4       0.000 000 0011 e-4       K T^-1
nuclear magneton in MHz/T                                   7.622 593 2291           0.000 000 0023           MHz T^-1
Planck constant                                             6.626 070 15 e-34        (exact)                  J Hz^-1
Planck constant in eV/Hz                                    4.135 667 696... e-15    (exact)                  eV Hz^-1
Planck length                                               1.616 255 e-35           0.000 018 e-35           m
Planck mass                                                 2.176 434 e-8            0.000 024 e-8            kg
Planck mass energy equivalent in GeV                        1.220 890 e19            0.000 014 e19            GeV
Planck temperature                                          1.416 784 e32            0.000 016 e32            K
Planck time                                                 5.391 247 e-44           0.000 060 e-44           s
proton charge to mass quotient                              9.578 833 1560 e7        0.000 000 0029 e7        C kg^-1
proton Compton wavelength                                   1.321 409 855 39 e-15    0.000 000 000 40 e-15    m
proton-electron mass ratio                                  1836.152 673 43          0.000 000 11
proton g factor                                             5.585 694 6893           0.000 000 0016
proton gyromag. ratio                                       2.675 221 8744 e8        0.000 000 0011 e8        s^-1 T^-1
proton gyromag. ratio in MHz/T                              42.577 478 518           0.000 000 018            MHz T^-1
proton mag. mom.                                            1.410 606 797 36 e-26    0.000 000 000 60 e-26    J T^-1
proton mag. mom. to Bohr magneton ratio                     1.521 032 202 30 e-3     0.000 000 000 46 e-3
proton mag. mom. to nuclear magneton ratio                  2.792 847 344 63         0.000 000 000 82
proton mag. shielding correction                            2.5689 e-5               0.0011 e-5
proton mass                                                 1.672 621 923 69 e-27    0.000 000 000 51 e-27    kg
proton mass energy equivalent                               1.503 277 615 98 e-10    0.000 000 000 46 e-10    J
proton mass energy equivalent in MeV                        938.272 088 16           0.000 000 29             MeV
proton mass in u                                            1.007 276 466 621        0.000 000 000 053        u
proton molar mass                                           1.007 276 466 27 e-3     0.000 000 000 31 e-3     kg mol^-1
proton-muon mass ratio                                      8.880 243 37             0.000 000 20
proton-neutron mag. mom. ratio                              -1.459 898 05            0.000 000 34
proton-neutron mass ratio                                   0.998 623 478 12         0.000 000 000 49
proton relative atomic mass                                 1.007 276 466 621        0.000 000 000 053
proton rms charge radius                                    8.414 e-16               0.019 e-16               m
proton-tau mass ratio                                       0.528 051                0.000 036
quantum of circulation                                      3.636 947 5516 e-4       0.000 000 0011 e-4       m^2 s^-1
quantum of circulation times 2                              7.273 895 1032 e-4       0.000 000 0022 e-4       m^2 s^-1
reduced Compton wavelength                                  3.861 592 6796 e-13      0.000 000 0012 e-13      m
reduced muon Compton wavelength                             1.867 594 306 e-15       0.000 000 042 e-15       m
reduced neutron Compton wavelength                          2.100 194 1552 e-16      0.000 000 0012 e-16      m
reduced Planck constant                                     1.054 571 817... e-34    (exact)                  J s
reduced Planck constant in eV s                             6.582 119 569... e-16    (exact)                  eV s
reduced Planck constant times c in MeV fm                   197.326 980 4...         (exact)                  MeV fm
reduced proton Compton wavelength                           2.103 089 103 36 e-16    0.000 000 000 64 e-16    m
reduced tau Compton wavelength                              1.110 538 e-16           0.000 075 e-16           m
Rydberg constant                                            10 973 731.568 160       0.000 021                m^-1
Rydberg constant times c in Hz                              3.289 841 960 2508 e15   0.000 000 000 0064 e15   Hz
Rydberg constant times hc in eV                             13.605 693 122 994       0.000 000 000 026        eV
Rydberg constant times hc in J                              2.179 872 361 1035 e-18  0.000 000 000 0042 e-18  J
Sackur-Tetrode constant (1 K, 100 kPa)                      -1.151 707 537 06        0.000 000 000 45
Sackur-Tetrode constant (1 K, 101.325 kPa)                  -1.164 870 523 58        0.000 000 000 45
second radiation constant                                   1.438 776 877... e-2     (exact)                  m K
shielded helion gyromag. ratio                              2.037 894 569 e8         0.000 000 024 e8         s^-1 T^-1
shielded helion gyromag. ratio in MHz/T                     32.434 099 42            0.000 000 38             MHz T^-1
shielded helion mag. mom.                                   -1.074 553 090 e-26      0.000 000 013 e-26       J T^-1
shielded helion mag. mom. to Bohr magneton ratio            -1.158 671 471 e-3       0.000 000 014 e-3
shielded helion mag. mom. to nuclear magneton ratio         -2.127 497 719           0.000 000 025
shielded helion to proton mag. mom. ratio                   -0.761 766 5618          0.000 000 0089
shielded helion to shielded proton mag. mom. ratio          -0.761 786 1313          0.000 000 0033
shielded proton gyromag. ratio                              2.675 153 151 e8         0.000 000 029 e8         s^-1 T^-1
shielded proton gyromag. ratio in MHz/T                     42.576 384 74            0.000 000 46             MHz T^-1
shielded proton mag. mom.                                   1.410 570 560 e-26       0.000 000 015 e-26       J T^-1
shielded proton mag. mom. to Bohr magneton ratio            1.520 993 128 e-3        0.000 000 017 e-3
shielded proton mag. mom. to nuclear magneton ratio         2.792 775 599            0.000 000 030
shielding difference of d and p in HD                       2.0200 e-8               0.0020 e-8
shielding difference of t and p in HT                       2.4140 e-8               0.0020 e-8
speed of light in vacuum                                    299 792 458              (exact)                  m s^-1
standard acceleration of gravity                            9.806 65                 (exact)                  m s^-2
standard atmosphere                                         101 325                  (exact)                  Pa
standard-state pressure                                     100 000                  (exact)                  Pa
Stefan-Boltzmann constant                                   5.670 374 419... e-8     (exact)                  W m^-2 K^-4
tau Compton wavelength                                      6.977 71 e-16            0.000 47 e-16            m
tau-electron mass ratio                                     3477.23                  0.23
tau energy equivalent                                       1776.86                  0.12                     MeV
tau mass                                                    3.167 54 e-27            0.000 21 e-27            kg
tau mass energy equivalent                                  2.846 84 e-10            0.000 19 e-10            J
tau mass in u                                               1.907 54                 0.000 13                 u
tau molar mass                                              1.907 54 e-3             0.000 13 e-3             kg mol^-1
tau-muon mass ratio                                         16.8170                  0.0011
tau-neutron mass ratio                                      1.891 15                 0.000 13
tau-proton mass ratio                                       1.893 76                 0.000 13
Thomson cross section                                       6.652 458 7321 e-29      0.000 000 0060 e-29      m^2
triton-electron mass ratio                                  5496.921 535 73          0.000 000 27
triton g factor                                             5.957 924 931            0.000 000 012
triton mag. mom.                                            1.504 609 5202 e-26      0.000 000 0030 e-26      J T^-1
triton mag. mom. to Bohr magneton ratio                     1.622 393 6651 e-3       0.000 000 0032 e-3
triton mag. mom. to nuclear magneton ratio                  2.978 962 4656           0.000 000 0059
triton mass                                                 5.007 356 7446 e-27      0.000 000 0015 e-27      kg
triton mass energy equivalent                               4.500 387 8060 e-10      0.000 000 0014 e-10      J
triton mass energy equivalent in MeV                        2808.921 132 98          0.000 000 85             MeV
triton mass in u                                            3.015 500 716 21         0.000 000 000 12         u
triton molar mass                                           3.015 500 715 17 e-3     0.000 000 000 92 e-3     kg mol^-1
triton-proton mass ratio                                    2.993 717 034 14         0.000 000 000 15
triton relative atomic mass                                 3.015 500 716 21         0.000 000 000 12
triton to proton mag. mom. ratio                            1.066 639 9191           0.000 000 0021
unified atomic mass unit                                    1.660 539 066 60 e-27    0.000 000 000 50 e-27    kg
vacuum electric permittivity                                8.854 187 8128 e-12      0.000 000 0013 e-12      F m^-1
vacuum mag. permeability                                    1.256 637 062 12 e-6     0.000 000 000 19 e-6     N A^-2
von Klitzing constant                                       25 812.807 45...         (exact)                  ohm
weak mixing angle                                           0.222 90                 0.000 30
Wien frequency displacement law constant                    5.878 925 757... e10     (exact)                  Hz K^-1
Wien wavelength displacement law constant                   2.897 771 955... e-3     (exact)                  m K
W to Z mass ratio                                           0.881 53                 0.000 17                   """

# -----------------------------------------------------------------------------

physical_constants = {}


def parse_constants_2002to2014(d):
    constants = {}
    for line in d.split('\n'):
        name = line[:55].rstrip()
        val = line[55:77].replace(' ', '').replace('...', '')
        val = float(val)
        uncert = line[77:99].replace(' ', '').replace('(exact)', '0')
        uncert = float(uncert)
        units = line[99:].rstrip()
        constants[name] = (val, units, uncert)
    return constants

def parse_constants_2018toXXXX(d):
    constants = {}
    for line in d.split('\n'):
        name = line[:60].rstrip()
        val = line[60:85].replace(' ', '').replace('...', '')
        val = float(val)
        uncert = line[85:110].replace(' ', '').replace('(exact)', '0')
        uncert = float(uncert)
        units = line[110:].rstrip()
        constants[name] = (val, units, uncert)
    return constants


_physical_constants_2002 = parse_constants_2002to2014(txt2002)
_physical_constants_2006 = parse_constants_2002to2014(txt2006)
_physical_constants_2010 = parse_constants_2002to2014(txt2010)
_physical_constants_2014 = parse_constants_2002to2014(txt2014)
_physical_constants_2018 = parse_constants_2018toXXXX(txt2018)


physical_constants.update(_physical_constants_2002)
physical_constants.update(_physical_constants_2006)
physical_constants.update(_physical_constants_2010)
physical_constants.update(_physical_constants_2014)
physical_constants.update(_physical_constants_2018)
_current_constants = _physical_constants_2018
_current_codata = "CODATA 2018"

# check obsolete values
_obsolete_constants = {}
for k in physical_constants:
    if k not in _current_constants:
        _obsolete_constants[k] = True

# generate some additional aliases
_aliases = {}
for k in _physical_constants_2002:
    if 'magn.' in k:
        _aliases[k] = k.replace('magn.', 'mag.')
for k in _physical_constants_2006:
    if 'momentum' in k:
        _aliases[k] = k.replace('momentum', 'mom.um')
for k in _physical_constants_2018:
    if 'momentum' in k:
        _aliases[k] = k.replace('momentum', 'mom.um')

# CODATA 2018: renamed and no longer exact; use as aliases
_aliases['mag. constant'] = 'vacuum mag. permeability'
_aliases['electric constant'] = 'vacuum electric permittivity'


class ConstantWarning(DeprecationWarning):
    """Accessing a constant no longer in current CODATA data set"""
    pass


def _check_obsolete(key):
    if key in _obsolete_constants and key not in _aliases:
        warnings.warn("Constant '%s' is not in current %s data set" % (
            key, _current_codata), ConstantWarning)


def value(key):
    """
    Value in physical_constants indexed by key

    Parameters
    ----------
    key : Python string or unicode
        Key in dictionary `physical_constants`

    Returns
    -------
    value : float
        Value in `physical_constants` corresponding to `key`

    Examples
    --------
    >>> from scipy import constants
    >>> constants.value(u'elementary charge')
    1.602176634e-19

    """
    _check_obsolete(key)
    return physical_constants[key][0]


def unit(key):
    """
    Unit in physical_constants indexed by key

    Parameters
    ----------
    key : Python string or unicode
        Key in dictionary `physical_constants`

    Returns
    -------
    unit : Python string
        Unit in `physical_constants` corresponding to `key`

    Examples
    --------
    >>> from scipy import constants
    >>> constants.unit(u'proton mass')
    'kg'

    """
    _check_obsolete(key)
    return physical_constants[key][1]


def precision(key):
    """
    Relative precision in physical_constants indexed by key

    Parameters
    ----------
    key : Python string or unicode
        Key in dictionary `physical_constants`

    Returns
    -------
    prec : float
        Relative precision in `physical_constants` corresponding to `key`

    Examples
    --------
    >>> from scipy import constants
    >>> constants.precision(u'proton mass')
    5.1e-37

    """
    _check_obsolete(key)
    return physical_constants[key][2] / physical_constants[key][0]


def find(sub=None, disp=False):
    """
    Return list of physical_constant keys containing a given string.

    Parameters
    ----------
    sub : str, unicode
        Sub-string to search keys for. By default, return all keys.
    disp : bool
        If True, print the keys that are found and return None.
        Otherwise, return the list of keys without printing anything.

    Returns
    -------
    keys : list or None
        If `disp` is False, the list of keys is returned.
        Otherwise, None is returned.

    Examples
    --------
    >>> from scipy.constants import find, physical_constants

    Which keys in the ``physical_constants`` dictionary contain 'boltzmann'?

    >>> find('boltzmann')
    ['Boltzmann constant',
     'Boltzmann constant in Hz/K',
     'Boltzmann constant in eV/K',
     'Boltzmann constant in inverse meter per kelvin',
     'Stefan-Boltzmann constant']

    Get the constant called 'Boltzmann constant in Hz/K':

    >>> physical_constants['Boltzmann constant in Hz/K']
    (20836619120.0, 'Hz K^-1', 0.0)

    Find constants with 'radius' in the key:

    >>> find('radius')
    ['Bohr radius',
     'classical electron radius',
     'deuteron rms charge radius',
     'proton rms charge radius']
    >>> physical_constants['classical electron radius']
    (2.8179403262e-15, 'm', 1.3e-24)

    """
    if sub is None:
        result = list(_current_constants.keys())
    else:
        result = [key for key in _current_constants
                  if sub.lower() in key.lower()]

    result.sort()
    if disp:
        for key in result:
            print(key)
        return
    else:
        return result

    
c = value('speed of light in vacuum')
mu0 = value('vacuum mag. permeability')
epsilon0 = value('vacuum electric permittivity')

# Table is lacking some digits for exact values: calculate from definition
exact_values = {
    'joule-kilogram relationship': (1 / (c * c), 'kg', 0.0),
    'kilogram-joule relationship': (c * c, 'J', 0.0),
    'hertz-inverse meter relationship': (1 / c, 'm^-1', 0.0),

    # The following derived quantities are no longer exact (CODATA2018):
    # specify separately
    'characteristic impedance of vacuum': (
        sqrt(mu0 / epsilon0), 'ohm',
        sqrt(mu0 / epsilon0) * 0.5 * (
            physical_constants['vacuum mag. permeability'][2] / mu0
            + physical_constants['vacuum electric permittivity'][2] / epsilon0))
}

# sanity check
for key in exact_values:
    val = physical_constants[key][0]
    if abs(exact_values[key][0] - val) / val > 1e-9:
        raise ValueError("Constants.codata: exact values too far off.")
    if exact_values[key][2] == 0 and physical_constants[key][2] != 0:
        raise ValueError("Constants.codata: value not exact")

physical_constants.update(exact_values)

_tested_keys = ['natural unit of velocity',
                'natural unit of action',
                'natural unit of action in eV s',
                'natural unit of mass',
                'natural unit of energy',
                'natural unit of energy in MeV',
                'natural unit of mom.um',
                'natural unit of mom.um in MeV/c',
                'natural unit of length',
                'natural unit of time']

# finally, insert aliases for values
for k, v in list(_aliases.items()):
    if v in _current_constants or v in _tested_keys:
        physical_constants[k] = physical_constants[v]
    else:
        del _aliases[k]
