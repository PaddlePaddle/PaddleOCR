"""
Statistics-related constants.

"""
import numpy as np


# The smallest representable positive number such that 1.0 + _EPS != 1.0.
_EPS = np.finfo(float).eps

# The largest [in magnitude] usable floating value.
_XMAX = np.finfo(float).max

# The log of the largest usable floating value; useful for knowing
# when exp(something) will overflow
_LOGXMAX = np.log(_XMAX)

# The smallest [in magnitude] usable floating value.
_XMIN = np.finfo(float).tiny

# -special.psi(1)
_EULER = 0.577215664901532860606512090082402431042

# special.zeta(3, 1)  Apery's constant
_ZETA3 = 1.202056903159594285399738161511449990765

# sqrt(2/pi)
_SQRT_2_OVER_PI = 0.7978845608028654

# log(sqrt(2/pi))
_LOG_SQRT_2_OVER_PI = -0.22579135264472744
