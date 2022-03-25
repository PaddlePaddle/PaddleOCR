try:
    from ._version import version as __version__
except ImportError:
    # broken installation, we don't even try
    # unknown only works because we do poor mans version compare
    __version__ = "unknown"

__all__ = [
    "PluginManager",
    "PluginValidationError",
    "HookCallError",
    "HookspecMarker",
    "HookimplMarker",
]

from ._manager import PluginManager, PluginValidationError
from ._callers import HookCallError
from ._hooks import HookspecMarker, HookimplMarker
