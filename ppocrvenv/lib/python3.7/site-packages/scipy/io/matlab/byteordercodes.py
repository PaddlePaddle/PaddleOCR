''' Byteorder utilities for system - numpy byteorder encoding

Converts a variety of string codes for little endian, big endian,
native byte order and swapped byte order to explicit NumPy endian
codes - one of '<' (little endian) or '>' (big endian)

'''
import sys

sys_is_le = sys.byteorder == 'little'
native_code = sys_is_le and '<' or '>'
swapped_code = sys_is_le and '>' or '<'

aliases = {'little': ('little', '<', 'l', 'le'),
           'big': ('big', '>', 'b', 'be'),
           'native': ('native', '='),
           'swapped': ('swapped', 'S')}


def to_numpy_code(code):
    """
    Convert various order codings to NumPy format.

    Parameters
    ----------
    code : str
        The code to convert. It is converted to lower case before parsing.
        Legal values are:
        'little', 'big', 'l', 'b', 'le', 'be', '<', '>', 'native', '=',
        'swapped', 's'.

    Returns
    -------
    out_code : {'<', '>'}
        Here '<' is the numpy dtype code for little endian,
        and '>' is the code for big endian.

    Examples
    --------
    >>> import sys
    >>> sys_is_le == (sys.byteorder == 'little')
    True
    >>> to_numpy_code('big')
    '>'
    >>> to_numpy_code('little')
    '<'
    >>> nc = to_numpy_code('native')
    >>> nc == '<' if sys_is_le else nc == '>'
    True
    >>> sc = to_numpy_code('swapped')
    >>> sc == '>' if sys_is_le else sc == '<'
    True

    """
    code = code.lower()
    if code is None:
        return native_code
    if code in aliases['little']:
        return '<'
    elif code in aliases['big']:
        return '>'
    elif code in aliases['native']:
        return native_code
    elif code in aliases['swapped']:
        return swapped_code
    else:
        raise ValueError(
            'We cannot handle byte order %s' % code)
