"""
Functions to generate methods and pin them to the appropriate classes.
"""
import operator

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

from pandas.core.ops import roperator


def _get_method_wrappers(cls):
    """
    Find the appropriate operation-wrappers to use when defining flex/special
    arithmetic, boolean, and comparison operations with the given class.

    Parameters
    ----------
    cls : class

    Returns
    -------
    arith_flex : function or None
    comp_flex : function or None
    """
    # TODO: make these non-runtime imports once the relevant functions
    #  are no longer in __init__
    from pandas.core.ops import (
        flex_arith_method_FRAME,
        flex_comp_method_FRAME,
        flex_method_SERIES,
    )

    if issubclass(cls, ABCSeries):
        # Just Series
        arith_flex = flex_method_SERIES
        comp_flex = flex_method_SERIES
    elif issubclass(cls, ABCDataFrame):
        arith_flex = flex_arith_method_FRAME
        comp_flex = flex_comp_method_FRAME
    return arith_flex, comp_flex


def add_flex_arithmetic_methods(cls):
    """
    Adds the full suite of flex arithmetic methods (``pow``, ``mul``, ``add``)
    to the class.

    Parameters
    ----------
    cls : class
        flex methods will be defined and pinned to this class
    """
    flex_arith_method, flex_comp_method = _get_method_wrappers(cls)
    new_methods = _create_methods(cls, flex_arith_method, flex_comp_method)
    new_methods.update(
        {
            "multiply": new_methods["mul"],
            "subtract": new_methods["sub"],
            "divide": new_methods["div"],
        }
    )
    # opt out of bool flex methods for now
    assert not any(kname in new_methods for kname in ("ror_", "rxor", "rand_"))

    _add_methods(cls, new_methods=new_methods)


def _create_methods(cls, arith_method, comp_method):
    # creates actual flex methods based upon arithmetic, and comp method
    # constructors.

    have_divmod = issubclass(cls, ABCSeries)
    # divmod is available for Series

    new_methods = {}

    new_methods.update(
        {
            "add": arith_method(operator.add),
            "radd": arith_method(roperator.radd),
            "sub": arith_method(operator.sub),
            "mul": arith_method(operator.mul),
            "truediv": arith_method(operator.truediv),
            "floordiv": arith_method(operator.floordiv),
            "mod": arith_method(operator.mod),
            "pow": arith_method(operator.pow),
            "rmul": arith_method(roperator.rmul),
            "rsub": arith_method(roperator.rsub),
            "rtruediv": arith_method(roperator.rtruediv),
            "rfloordiv": arith_method(roperator.rfloordiv),
            "rpow": arith_method(roperator.rpow),
            "rmod": arith_method(roperator.rmod),
        }
    )
    new_methods["div"] = new_methods["truediv"]
    new_methods["rdiv"] = new_methods["rtruediv"]
    if have_divmod:
        # divmod doesn't have an op that is supported by numexpr
        new_methods["divmod"] = arith_method(divmod)
        new_methods["rdivmod"] = arith_method(roperator.rdivmod)

    new_methods.update(
        {
            "eq": comp_method(operator.eq),
            "ne": comp_method(operator.ne),
            "lt": comp_method(operator.lt),
            "gt": comp_method(operator.gt),
            "le": comp_method(operator.le),
            "ge": comp_method(operator.ge),
        }
    )

    new_methods = {k.strip("_"): v for k, v in new_methods.items()}
    return new_methods


def _add_methods(cls, new_methods):
    for name, method in new_methods.items():
        setattr(cls, name, method)
