# imports are for backward compatibility
from matplotlib._api.deprecation import (
    MatplotlibDeprecationWarning, mplDeprecation, warn_deprecated, deprecated)

warn_deprecated("3.4",
                message="The module matplotlib.cbook.deprecation is "
                "considered internal and it will be made private in the "
                "future.")
