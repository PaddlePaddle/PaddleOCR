"""
This file is separate from ``_add_newdocs.py`` so that it can be mocked out by
our sphinx ``conf.py`` during doc builds, where we want to avoid showing
platform-dependent information.
"""
from numpy.core import dtype
from numpy.core import numerictypes as _numerictypes
from numpy.core.function_base import add_newdoc
import platform

##############################################################################
#
# Documentation for concrete scalar classes
#
##############################################################################

def numeric_type_aliases(aliases):
    def type_aliases_gen():
        for alias, doc in aliases:
            try:
                alias_type = getattr(_numerictypes, alias)
            except AttributeError:
                # The set of aliases that actually exist varies between platforms
                pass
            else:
                yield (alias_type, alias, doc)
    return list(type_aliases_gen())


possible_aliases = numeric_type_aliases([
    ('int8', '8-bit signed integer (``-128`` to ``127``)'),
    ('int16', '16-bit signed integer (``-32_768`` to ``32_767``)'),
    ('int32', '32-bit signed integer (``-2_147_483_648`` to ``2_147_483_647``)'),
    ('int64', '64-bit signed integer (``-9_223_372_036_854_775_808`` to ``9_223_372_036_854_775_807``)'),
    ('intp', 'Signed integer large enough to fit pointer, compatible with C ``intptr_t``'),
    ('uint8', '8-bit unsigned integer (``0`` to ``255``)'),
    ('uint16', '16-bit unsigned integer (``0`` to ``65_535``)'),
    ('uint32', '32-bit unsigned integer (``0`` to ``4_294_967_295``)'),
    ('uint64', '64-bit unsigned integer (``0`` to ``18_446_744_073_709_551_615``)'),
    ('uintp', 'Unsigned integer large enough to fit pointer, compatible with C ``uintptr_t``'),
    ('float16', '16-bit-precision floating-point number type: sign bit, 5 bits exponent, 10 bits mantissa'),
    ('float32', '32-bit-precision floating-point number type: sign bit, 8 bits exponent, 23 bits mantissa'),
    ('float64', '64-bit precision floating-point number type: sign bit, 11 bits exponent, 52 bits mantissa'),
    ('float96', '96-bit extended-precision floating-point number type'),
    ('float128', '128-bit extended-precision floating-point number type'),
    ('complex64', 'Complex number type composed of 2 32-bit-precision floating-point numbers'),
    ('complex128', 'Complex number type composed of 2 64-bit-precision floating-point numbers'),
    ('complex192', 'Complex number type composed of 2 96-bit extended-precision floating-point numbers'),
    ('complex256', 'Complex number type composed of 2 128-bit extended-precision floating-point numbers'),
    ])




def add_newdoc_for_scalar_type(obj, fixed_aliases, doc):
    # note: `:field: value` is rST syntax which renders as field lists.
    o = getattr(_numerictypes, obj)

    character_code = dtype(o).char
    canonical_name_doc = "" if obj == o.__name__ else ":Canonical name: `numpy.{}`\n    ".format(obj)
    alias_doc = ''.join(":Alias: `numpy.{}`\n    ".format(alias) for alias in fixed_aliases)
    alias_doc += ''.join(":Alias on this platform ({} {}): `numpy.{}`: {}.\n    ".format(platform.system(), platform.machine(), alias, doc)
                         for (alias_type, alias, doc) in possible_aliases if alias_type is o)
    docstring = """
    {doc}

    :Character code: ``'{character_code}'``
    {canonical_name_doc}{alias_doc}
    """.format(doc=doc.strip(), character_code=character_code,
               canonical_name_doc=canonical_name_doc, alias_doc=alias_doc)

    add_newdoc('numpy.core.numerictypes', obj, docstring)


add_newdoc_for_scalar_type('bool_', ['bool8'],
    """
    Boolean type (True or False), stored as a byte.

    .. warning::

       The :class:`bool_` type is not a subclass of the :class:`int_` type
       (the :class:`bool_` is not even a number type). This is different
       than Python's default implementation of :class:`bool` as a
       sub-class of :class:`int`.
    """)

add_newdoc_for_scalar_type('byte', [],
    """
    Signed integer type, compatible with C ``char``.
    """)

add_newdoc_for_scalar_type('short', [],
    """
    Signed integer type, compatible with C ``short``.
    """)

add_newdoc_for_scalar_type('intc', [],
    """
    Signed integer type, compatible with C ``int``.
    """)

add_newdoc_for_scalar_type('int_', [],
    """
    Signed integer type, compatible with Python `int` and C ``long``.
    """)

add_newdoc_for_scalar_type('longlong', [],
    """
    Signed integer type, compatible with C ``long long``.
    """)

add_newdoc_for_scalar_type('ubyte', [],
    """
    Unsigned integer type, compatible with C ``unsigned char``.
    """)

add_newdoc_for_scalar_type('ushort', [],
    """
    Unsigned integer type, compatible with C ``unsigned short``.
    """)

add_newdoc_for_scalar_type('uintc', [],
    """
    Unsigned integer type, compatible with C ``unsigned int``.
    """)

add_newdoc_for_scalar_type('uint', [],
    """
    Unsigned integer type, compatible with C ``unsigned long``.
    """)

add_newdoc_for_scalar_type('ulonglong', [],
    """
    Signed integer type, compatible with C ``unsigned long long``.
    """)

add_newdoc_for_scalar_type('half', [],
    """
    Half-precision floating-point number type.
    """)

add_newdoc_for_scalar_type('single', [],
    """
    Single-precision floating-point number type, compatible with C ``float``.
    """)

add_newdoc_for_scalar_type('double', ['float_'],
    """
    Double-precision floating-point number type, compatible with Python `float`
    and C ``double``.
    """)

add_newdoc_for_scalar_type('longdouble', ['longfloat'],
    """
    Extended-precision floating-point number type, compatible with C
    ``long double`` but not necessarily with IEEE 754 quadruple-precision.
    """)

add_newdoc_for_scalar_type('csingle', ['singlecomplex'],
    """
    Complex number type composed of two single-precision floating-point
    numbers.
    """)

add_newdoc_for_scalar_type('cdouble', ['cfloat', 'complex_'],
    """
    Complex number type composed of two double-precision floating-point
    numbers, compatible with Python `complex`.
    """)

add_newdoc_for_scalar_type('clongdouble', ['clongfloat', 'longcomplex'],
    """
    Complex number type composed of two extended-precision floating-point
    numbers.
    """)

add_newdoc_for_scalar_type('object_', [],
    """
    Any Python object.
    """)

add_newdoc_for_scalar_type('str_', ['unicode_'],
    r"""
    A unicode string.

    When used in arrays, this type strips trailing null codepoints.

    Unlike the builtin `str`, this supports the :ref:`python:bufferobjects`, exposing its
    contents as UCS4:

    >>> m = memoryview(np.str_("abc"))
    >>> m.format
    '3w'
    >>> m.tobytes()
    b'a\x00\x00\x00b\x00\x00\x00c\x00\x00\x00'
    """)

add_newdoc_for_scalar_type('bytes_', ['string_'],
    r"""
    A byte string.

    When used in arrays, this type strips trailing null bytes.
    """)

add_newdoc_for_scalar_type('void', [],
    r"""
    Either an opaque sequence of bytes, or a structure.
    
    >>> np.void(b'abcd')
    void(b'\x61\x62\x63\x64')
    
    Structured `void` scalars can only be constructed via extraction from :ref:`structured_arrays`:
    
    >>> arr = np.array((1, 2), dtype=[('x', np.int8), ('y', np.int8)])
    >>> arr[()]
    (1, 2)  # looks like a tuple, but is `np.void`
    """)

add_newdoc_for_scalar_type('datetime64', [],
    """
    If created from a 64-bit integer, it represents an offset from
    ``1970-01-01T00:00:00``.
    If created from string, the string can be in ISO 8601 date
    or datetime format.

    >>> np.datetime64(10, 'Y')
    numpy.datetime64('1980')
    >>> np.datetime64('1980', 'Y')
    numpy.datetime64('1980')   
    >>> np.datetime64(10, 'D')
    numpy.datetime64('1970-01-11')
    
    See :ref:`arrays.datetime` for more information.
    """)

add_newdoc_for_scalar_type('timedelta64', [],
    """
    A timedelta stored as a 64-bit integer.
    
    See :ref:`arrays.datetime` for more information.
    """)

# TODO: work out how to put this on the base class, np.floating
for float_name in ('half', 'single', 'double', 'longdouble'):
    add_newdoc('numpy.core.numerictypes', float_name, ('as_integer_ratio',
        """
        {ftype}.as_integer_ratio() -> (int, int)

        Return a pair of integers, whose ratio is exactly equal to the original
        floating point number, and with a positive denominator.
        Raise `OverflowError` on infinities and a `ValueError` on NaNs.

        >>> np.{ftype}(10.0).as_integer_ratio()
        (10, 1)
        >>> np.{ftype}(0.0).as_integer_ratio()
        (0, 1)
        >>> np.{ftype}(-.25).as_integer_ratio()
        (-1, 4)
        """.format(ftype=float_name)))
