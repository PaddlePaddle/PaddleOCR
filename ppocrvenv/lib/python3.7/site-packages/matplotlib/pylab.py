"""
.. warning::
   Since heavily importing into the global namespace may result in unexpected
   behavior, the use of pylab is strongly discouraged. Use `matplotlib.pyplot`
   instead.

`pylab` is a module that includes `matplotlib.pyplot`, `numpy`, `numpy.fft`,
`numpy.linalg`, `numpy.random`, and some additional functions, all within
a single namespace. Its original purpose was to mimic a MATLAB-like way
of working by importing all functions into the global namespace. This is
considered bad style nowadays.
"""

from matplotlib.cbook import flatten, silent_list

import matplotlib as mpl

from matplotlib.dates import (
    date2num, num2date, datestr2num, drange, epoch2num,
    num2epoch, DateFormatter, DateLocator,
    RRuleLocator, YearLocator, MonthLocator, WeekdayLocator, DayLocator,
    HourLocator, MinuteLocator, SecondLocator, rrule, MO, TU, WE, TH, FR,
    SA, SU, YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY,
    relativedelta)

# bring all the symbols in so folks can import them from
# pylab in one fell swoop

## We are still importing too many things from mlab; more cleanup is needed.

from matplotlib.mlab import (
    detrend, detrend_linear, detrend_mean, detrend_none, window_hanning,
    window_none)

from matplotlib import cbook, mlab, pyplot as plt
from matplotlib.pyplot import *

from numpy import *
from numpy.fft import *
from numpy.random import *
from numpy.linalg import *

import numpy as np
import numpy.ma as ma

# don't let numpy's datetime hide stdlib
import datetime

# This is needed, or bytes will be numpy.random.bytes from
# "from numpy.random import *" above
bytes = __import__("builtins").bytes
