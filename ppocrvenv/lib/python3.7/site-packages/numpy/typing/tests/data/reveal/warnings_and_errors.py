from typing import Type

import numpy as np

reveal_type(np.ModuleDeprecationWarning())  # E: numpy.ModuleDeprecationWarning
reveal_type(np.VisibleDeprecationWarning())  # E: numpy.VisibleDeprecationWarning
reveal_type(np.ComplexWarning())  # E: numpy.ComplexWarning
reveal_type(np.RankWarning())  # E: numpy.RankWarning
reveal_type(np.TooHardError())  # E: numpy.TooHardError
reveal_type(np.AxisError(1))  # E: numpy.AxisError
