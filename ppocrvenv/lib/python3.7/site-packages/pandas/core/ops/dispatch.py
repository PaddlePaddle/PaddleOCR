"""
Functions for defining unary operations.
"""
from typing import Any

from pandas._typing import ArrayLike

from pandas.core.dtypes.generic import ABCExtensionArray


def should_extension_dispatch(left: ArrayLike, right: Any) -> bool:
    """
    Identify cases where Series operation should dispatch to ExtensionArray method.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
    right : object

    Returns
    -------
    bool
    """
    return isinstance(left, ABCExtensionArray) or isinstance(right, ABCExtensionArray)
