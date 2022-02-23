DISPATCHED_UFUNCS = {
    "add",
    "sub",
    "mul",
    "pow",
    "mod",
    "floordiv",
    "truediv",
    "divmod",
    "eq",
    "ne",
    "lt",
    "gt",
    "le",
    "ge",
    "remainder",
    "matmul",
    "or",
    "xor",
    "and",
}
UFUNC_ALIASES = {
    "subtract": "sub",
    "multiply": "mul",
    "floor_divide": "floordiv",
    "true_divide": "truediv",
    "power": "pow",
    "remainder": "mod",
    "divide": "truediv",
    "equal": "eq",
    "not_equal": "ne",
    "less": "lt",
    "less_equal": "le",
    "greater": "gt",
    "greater_equal": "ge",
    "bitwise_or": "or",
    "bitwise_and": "and",
    "bitwise_xor": "xor",
}

# For op(., Array) -> Array.__r{op}__
REVERSED_NAMES = {
    "lt": "__gt__",
    "le": "__ge__",
    "gt": "__lt__",
    "ge": "__le__",
    "eq": "__eq__",
    "ne": "__ne__",
}


def maybe_dispatch_ufunc_to_dunder_op(
    object self, object ufunc, str method, *inputs, **kwargs
):
    """
    Dispatch a ufunc to the equivalent dunder method.

    Parameters
    ----------
    self : ArrayLike
        The array whose dunder method we dispatch to
    ufunc : Callable
        A NumPy ufunc
    method : {'reduce', 'accumulate', 'reduceat', 'outer', 'at', '__call__'}
    inputs : ArrayLike
        The input arrays.
    kwargs : Any
        The additional keyword arguments, e.g. ``out``.

    Returns
    -------
    result : Any
        The result of applying the ufunc
    """
    # special has the ufuncs we dispatch to the dunder op on

    op_name = ufunc.__name__
    op_name = UFUNC_ALIASES.get(op_name, op_name)

    def not_implemented(*args, **kwargs):
        return NotImplemented

    if (method == "__call__"
            and op_name in DISPATCHED_UFUNCS
            and kwargs.get("out") is None):
        if isinstance(inputs[0], type(self)):
            name = f"__{op_name}__"
            return getattr(self, name, not_implemented)(inputs[1])
        else:
            name = REVERSED_NAMES.get(op_name, f"__r{op_name}__")
            result = getattr(self, name, not_implemented)(inputs[0])
            return result
    else:
        return NotImplemented
