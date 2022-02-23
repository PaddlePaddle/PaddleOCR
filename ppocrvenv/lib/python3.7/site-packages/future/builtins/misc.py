"""
A module that brings in equivalents of various modified Python 3 builtins
into Py2. Has no effect on Py3.

The builtin functions are:

- ``ascii`` (from Py2's future_builtins module)
- ``hex`` (from Py2's future_builtins module)
- ``oct`` (from Py2's future_builtins module)
- ``chr`` (equivalent to ``unichr`` on Py2)
- ``input`` (equivalent to ``raw_input`` on Py2)
- ``next`` (calls ``__next__`` if it exists, else ``next`` method)
- ``open`` (equivalent to io.open on Py2)
- ``super`` (backport of Py3's magic zero-argument super() function
- ``round`` (new "Banker's Rounding" behaviour from Py3)
- ``max`` (new default option from Py3.4)
- ``min`` (new default option from Py3.4)

``isinstance`` is also currently exported for backwards compatibility
with v0.8.2, although this has been deprecated since v0.9.


input()
-------
Like the new ``input()`` function from Python 3 (without eval()), except
that it returns bytes. Equivalent to Python 2's ``raw_input()``.

Warning: By default, importing this module *removes* the old Python 2
input() function entirely from ``__builtin__`` for safety. This is
because forgetting to import the new ``input`` from ``future`` might
otherwise lead to a security vulnerability (shell injection) on Python 2.

To restore it, you can retrieve it yourself from
``__builtin__._old_input``.

Fortunately, ``input()`` seems to be seldom used in the wild in Python
2...

"""

from future import utils


if utils.PY2:
    from io import open
    from future_builtins import ascii, oct, hex
    from __builtin__ import unichr as chr, pow as _builtin_pow
    import __builtin__

    # Only for backward compatibility with future v0.8.2:
    isinstance = __builtin__.isinstance

    # Warning: Python 2's input() is unsafe and MUST not be able to be used
    # accidentally by someone who expects Python 3 semantics but forgets
    # to import it on Python 2. Versions of ``future`` prior to 0.11
    # deleted it from __builtin__.  Now we keep in __builtin__ but shadow
    # the name like all others. Just be sure to import ``input``.

    input = raw_input

    from future.builtins.newnext import newnext as next
    from future.builtins.newround import newround as round
    from future.builtins.newsuper import newsuper as super
    from future.builtins.new_min_max import newmax as max
    from future.builtins.new_min_max import newmin as min
    from future.types.newint import newint

    _SENTINEL = object()

    def pow(x, y, z=_SENTINEL):
        """
        pow(x, y[, z]) -> number

        With two arguments, equivalent to x**y.  With three arguments,
        equivalent to (x**y) % z, but may be more efficient (e.g. for ints).
        """
        # Handle newints
        if isinstance(x, newint):
            x = long(x)
        if isinstance(y, newint):
            y = long(y)
        if isinstance(z, newint):
            z = long(z)

        try:
            if z == _SENTINEL:
                return _builtin_pow(x, y)
            else:
                return _builtin_pow(x, y, z)
        except ValueError:
            if z == _SENTINEL:
                return _builtin_pow(x+0j, y)
            else:
                return _builtin_pow(x+0j, y, z)


    # ``future`` doesn't support Py3.0/3.1. If we ever did, we'd add this:
    #     callable = __builtin__.callable

    __all__ = ['ascii', 'chr', 'hex', 'input', 'isinstance', 'next', 'oct',
               'open', 'pow', 'round', 'super', 'max', 'min']

else:
    import builtins
    ascii = builtins.ascii
    chr = builtins.chr
    hex = builtins.hex
    input = builtins.input
    next = builtins.next
    # Only for backward compatibility with future v0.8.2:
    isinstance = builtins.isinstance
    oct = builtins.oct
    open = builtins.open
    pow = builtins.pow
    round = builtins.round
    super = builtins.super
    if utils.PY34_PLUS:
        max = builtins.max
        min = builtins.min
        __all__ = []
    else:
        from future.builtins.new_min_max import newmax as max
        from future.builtins.new_min_max import newmin as min
        __all__ = ['min', 'max']

    # The callable() function was removed from Py3.0 and 3.1 and
    # reintroduced into Py3.2+. ``future`` doesn't support Py3.0/3.1. If we ever
    # did, we'd add this:
    # try:
    #     callable = builtins.callable
    # except AttributeError:
    #     # Definition from Pandas
    #     def callable(obj):
    #         return any("__call__" in klass.__dict__ for klass in type(obj).__mro__)
    #     __all__.append('callable')
