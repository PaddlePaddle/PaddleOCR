r"""The :mod:`loky` module manages a pool of worker that can be re-used across time.
It provides a robust and dynamic implementation os the
:class:`ProcessPoolExecutor` and a function :func:`get_reusable_executor` which
hide the pool management under the hood.
"""
from ._base import Executor, Future
from ._base import wait, as_completed
from ._base import TimeoutError, CancelledError
from ._base import ALL_COMPLETED, FIRST_COMPLETED, FIRST_EXCEPTION

from .backend.context import cpu_count
from .backend.reduction import set_loky_pickler
from .reusable_executor import get_reusable_executor
from .cloudpickle_wrapper import wrap_non_picklable_objects
from .process_executor import BrokenProcessPool, ProcessPoolExecutor


__all__ = ["get_reusable_executor", "cpu_count", "wait", "as_completed",
           "Future", "Executor", "ProcessPoolExecutor",
           "BrokenProcessPool", "CancelledError", "TimeoutError",
           "FIRST_COMPLETED", "FIRST_EXCEPTION", "ALL_COMPLETED",
           "wrap_non_picklable_objects", "set_loky_pickler"]


__version__ = '3.0.0'
