# -*- coding: utf-8 -*-
"""
Along with ns_enum.py, this module contains all of the
natsort public API.

The majority of the "work" is defined in utils.py.
"""

import platform
from functools import partial
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

import natsort.compat.locale
from natsort import utils
from natsort.ns_enum import NSType, NS_DUMB, ns
from natsort.utils import (
    KeyType,
    MaybeKeyType,
    NatsortInType,
    NatsortOutType,
    StrBytesNum,
    StrBytesPathNum,
)

# Common input and output types
Iter_ns = Iterable[NatsortInType]
Iter_any = Iterable[Any]
List_ns = List[NatsortInType]
List_any = List[Any]
List_int = List[int]

# The type that natsort_key returns
NatsortKeyType = Callable[[NatsortInType], NatsortOutType]

# Types for os_sorted
OSSortInType = Iterable[Optional[StrBytesPathNum]]
OSSortOutType = Tuple[Union[StrBytesNum, Tuple[StrBytesNum, ...]], ...]
OSSortKeyType = Callable[[Optional[StrBytesPathNum]], OSSortOutType]
Iter_path = Iterable[Optional[StrBytesPathNum]]
List_path = List[StrBytesPathNum]


def decoder(encoding: str) -> Callable[[NatsortInType], NatsortInType]:
    """
    Return a function that can be used to decode bytes to unicode.

    Parameters
    ----------
    encoding : str
        The codec to use for decoding. This must be a valid unicode codec.

    Returns
    -------
    decode_function
        A function that takes a single argument and attempts to decode
        it using the supplied codec. Any `UnicodeErrors` are raised.
        If the argument was not of `bytes` type, it is simply returned
        as-is.

    See Also
    --------
    as_ascii
    as_utf8

    Examples
    --------

        >>> f = decoder('utf8')
        >>> f(b'bytes') == 'bytes'
        True
        >>> f(12345) == 12345
        True
        >>> # On Python 3, without decoder this would return [b'a10', b'a2']
        >>> natsorted([b'a10', b'a2'], key=decoder('utf8')) == [b'a2', b'a10']
        True
        >>> # On Python 3, without decoder this would raise a TypeError.
        >>> natsorted([b'a10', 'a2'], key=decoder('utf8')) == ['a2', b'a10']
        True

    """
    return partial(utils.do_decoding, encoding=encoding)


def as_ascii(s: NatsortInType) -> NatsortInType:
    """
    Function to decode an input with the ASCII codec, or return as-is.

    Parameters
    ----------
    s : object

    Returns
    -------
    output
        If the input was of type `bytes`, the return value is a `str` decoded
        with the ASCII codec. Otherwise, the return value is identically the
        input.

    See Also
    --------
    decoder

    """
    return utils.do_decoding(s, "ascii")


def as_utf8(s: NatsortInType) -> NatsortInType:
    """
    Function to decode an input with the UTF-8 codec, or return as-is.

    Parameters
    ----------
    s : object

    Returns
    -------
    output
        If the input was of type `bytes`, the return value is a `str` decoded
        with the UTF-8 codec. Otherwise, the return value is identically the
        input.

    See Also
    --------
    decoder

    """
    return utils.do_decoding(s, "utf-8")


def natsort_keygen(
    key: MaybeKeyType = None, alg: NSType = ns.DEFAULT
) -> NatsortKeyType:
    """
    Generate a key to sort strings and numbers naturally.

    This key is designed for use as the `key` argument to
    functions such as the `sorted` builtin.

    The user may customize the generated function with the
    arguments to `natsort_keygen`, including an optional
    `key` function.

    Parameters
    ----------
    key : callable, optional
        A key used to manipulate the input value before parsing for
        numbers. It is **not** applied recursively.
        It should accept a single argument and return a single value.

    alg : ns enum, optional
        This option is used to control which algorithm `natsort`
        uses when sorting. For details into these options, please see
        the :class:`ns` class documentation. The default is `ns.INT`.

    Returns
    -------
    out : function
        A function that parses input for natural sorting that is
        suitable for passing as the `key` argument to functions
        such as `sorted`.

    See Also
    --------
    natsorted
    natsort_key

    Examples
    --------
    `natsort_keygen` is a convenient way to create a custom key
    to sort lists in-place (for example).::

        >>> a = ['num5.10', 'num-3', 'num5.3', 'num2']
        >>> a.sort(key=natsort_keygen(alg=ns.REAL))
        >>> a
        ['num-3', 'num2', 'num5.10', 'num5.3']

    """
    try:
        ns.DEFAULT | alg
    except TypeError:
        msg = "natsort_keygen: 'alg' argument must be from the enum 'ns'"
        raise ValueError(msg + ", got {}".format(str(alg)))

    # Add the NS_DUMB option if the locale library is broken.
    if alg & ns.LOCALEALPHA and natsort.compat.locale.dumb_sort():
        alg |= NS_DUMB

    # Set some variables that will be passed to the factory functions
    if alg & ns.NUMAFTER:
        if alg & ns.LOCALEALPHA:
            sep = natsort.compat.locale.null_string_locale_max
        else:
            sep = natsort.compat.locale.null_string_max
        pre_sep = natsort.compat.locale.null_string_max
    else:
        if alg & ns.LOCALEALPHA:
            sep = natsort.compat.locale.null_string_locale
        else:
            sep = natsort.compat.locale.null_string
        pre_sep = natsort.compat.locale.null_string
    regex = utils.regex_chooser(alg)

    # Create the functions that will be used to split strings.
    input_transform = utils.input_string_transform_factory(alg)
    component_transform = utils.string_component_transform_factory(alg)
    final_transform = utils.final_data_transform_factory(alg, sep, pre_sep)

    # Create the high-level parsing functions for strings, bytes, and numbers.
    string_func = utils.parse_string_factory(
        alg, sep, regex.split, input_transform, component_transform, final_transform
    )
    if alg & ns.PATH:
        string_func = utils.parse_path_factory(string_func)
    bytes_func = utils.parse_bytes_factory(alg)
    num_func = utils.parse_number_or_none_factory(alg, sep, pre_sep)

    # Return the natsort key with the parsing path pre-chosen.
    return partial(
        utils.natsort_key,
        key=key,
        string_func=string_func,
        bytes_func=bytes_func,
        num_func=num_func,
    )


# Exposed for simplicity if one needs the default natsort key.
natsort_key = natsort_keygen()
natsort_key.__doc__ = """\
natsort_key(val)
The default natural sorting key.

This is the output of :func:`natsort_keygen` with default values.

See Also
--------
natsort_keygen

"""


@overload
def natsorted(
    seq: Iter_ns, key: None = None, reverse: bool = False, alg: NSType = ns.DEFAULT
) -> List_ns:
    ...


@overload
def natsorted(
    seq: Iter_any, key: KeyType, reverse: bool = False, alg: NSType = ns.DEFAULT
) -> List_any:
    ...


def natsorted(
    seq: Iter_any,
    key: MaybeKeyType = None,
    reverse: bool = False,
    alg: NSType = ns.DEFAULT,
) -> List_any:
    """
    Sorts an iterable naturally.

    Parameters
    ----------
    seq : iterable
        The input to sort.

    key : callable, optional
        A key used to determine how to sort each element of the iterable.
        It is **not** applied recursively.
        It should accept a single argument and return a single value.

    reverse : {{True, False}}, optional
        Return the list in reversed sorted order. The default is
        `False`.

    alg : ns enum, optional
        This option is used to control which algorithm `natsort`
        uses when sorting. For details into these options, please see
        the :class:`ns` class documentation. The default is `ns.INT`.

    Returns
    -------
    out: list
        The sorted input.

    See Also
    --------
    natsort_keygen : Generates the key that makes natural sorting possible.
    realsorted : A wrapper for ``natsorted(seq, alg=ns.REAL)``.
    humansorted : A wrapper for ``natsorted(seq, alg=ns.LOCALE)``.
    index_natsorted : Returns the sorted indexes from `natsorted`.
    os_sorted : Sort according to your operating system's rules.

    Examples
    --------
    Use `natsorted` just like the builtin `sorted`::

        >>> a = ['num3', 'num5', 'num2']
        >>> natsorted(a)
        ['num2', 'num3', 'num5']

    """
    return sorted(seq, reverse=reverse, key=natsort_keygen(key, alg))


@overload
def humansorted(
    seq: Iter_ns, key: None = None, reverse: bool = False, alg: NSType = ns.DEFAULT
) -> List_ns:
    ...


@overload
def humansorted(
    seq: Iter_any, key: KeyType, reverse: bool = False, alg: NSType = ns.DEFAULT
) -> List_any:
    ...


def humansorted(
    seq: Iter_any,
    key: MaybeKeyType = None,
    reverse: bool = False,
    alg: NSType = ns.DEFAULT,
) -> List_any:
    """
    Convenience function to properly sort non-numeric characters.

    This is a wrapper around ``natsorted(seq, alg=ns.LOCALE)``.

    Parameters
    ----------
    seq : iterable
        The input to sort.

    key : callable, optional
        A key used to determine how to sort each element of the sequence.
        It is **not** applied recursively.
        It should accept a single argument and return a single value.

    reverse : {{True, False}}, optional
        Return the list in reversed sorted order. The default is
        `False`.

    alg : ns enum, optional
        This option is used to control which algorithm `natsort`
        uses when sorting. For details into these options, please see
        the :class:`ns` class documentation. The default is `ns.LOCALE`.

    Returns
    -------
    out : list
        The sorted input.

    See Also
    --------
    index_humansorted : Returns the sorted indexes from `humansorted`.

    Notes
    -----
    Please read :ref:`locale_issues` before using `humansorted`.

    Examples
    --------
    Use `humansorted` just like the builtin `sorted`::

        >>> a = ['Apple', 'Banana', 'apple', 'banana']
        >>> natsorted(a)
        ['Apple', 'Banana', 'apple', 'banana']
        >>> humansorted(a)
        ['apple', 'Apple', 'banana', 'Banana']

    """
    return natsorted(seq, key, reverse, alg | ns.LOCALE)


@overload
def realsorted(
    seq: Iter_ns, key: None = None, reverse: bool = False, alg: NSType = ns.DEFAULT
) -> List_ns:
    ...


@overload
def realsorted(
    seq: Iter_any, key: KeyType, reverse: bool = False, alg: NSType = ns.DEFAULT
) -> List_any:
    ...


def realsorted(
    seq: Iter_any,
    key: MaybeKeyType = None,
    reverse: bool = False,
    alg: NSType = ns.DEFAULT,
) -> List_any:
    """
    Convenience function to properly sort signed floats.

    A signed float in a string could be "a-5.7". This is a wrapper around
    ``natsorted(seq, alg=ns.REAL)``.

    The behavior of :func:`realsorted` for `natsort` version >= 4.0.0
    was the default behavior of :func:`natsorted` for `natsort`
    version < 4.0.0.

    Parameters
    ----------
    seq : iterable
        The input to sort.

    key : callable, optional
        A key used to determine how to sort each element of the sequence.
        It is **not** applied recursively.
        It should accept a single argument and return a single value.

    reverse : {{True, False}}, optional
        Return the list in reversed sorted order. The default is
        `False`.

    alg : ns enum, optional
        This option is used to control which algorithm `natsort`
        uses when sorting. For details into these options, please see
        the :class:`ns` class documentation. The default is `ns.REAL`.

    Returns
    -------
    out : list
        The sorted input.

    See Also
    --------
    index_realsorted : Returns the sorted indexes from `realsorted`.

    Examples
    --------
    Use `realsorted` just like the builtin `sorted`::

        >>> a = ['num5.10', 'num-3', 'num5.3', 'num2']
        >>> natsorted(a)
        ['num2', 'num5.3', 'num5.10', 'num-3']
        >>> realsorted(a)
        ['num-3', 'num2', 'num5.10', 'num5.3']

    """
    return natsorted(seq, key, reverse, alg | ns.REAL)


@overload
def index_natsorted(
    seq: Iter_ns, key: None = None, reverse: bool = False, alg: NSType = ns.DEFAULT
) -> List_int:
    ...


@overload
def index_natsorted(
    seq: Iter_any, key: KeyType, reverse: bool = False, alg: NSType = ns.DEFAULT
) -> List_int:
    ...


def index_natsorted(
    seq: Iter_any,
    key: MaybeKeyType = None,
    reverse: bool = False,
    alg: NSType = ns.DEFAULT,
) -> List_int:
    """
    Determine the list of the indexes used to sort the input sequence.

    Sorts a sequence naturally, but returns a list of sorted the
    indexes and not the sorted list itself. This list of indexes
    can be used to sort multiple lists by the sorted order of the
    given sequence.

    Parameters
    ----------
    seq : iterable
        The input to sort.

    key : callable, optional
        A key used to determine how to sort each element of the sequence.
        It is **not** applied recursively.
        It should accept a single argument and return a single value.

    reverse : {{True, False}}, optional
        Return the list in reversed sorted order. The default is
        `False`.

    alg : ns enum, optional
        This option is used to control which algorithm `natsort`
        uses when sorting. For details into these options, please see
        the :class:`ns` class documentation. The default is `ns.INT`.

    Returns
    -------
    out : tuple
        The ordered indexes of the input.

    See Also
    --------
    natsorted
    order_by_index

    Examples
    --------

    Use index_natsorted if you want to sort multiple lists by the
    sorted order of one list::

        >>> a = ['num3', 'num5', 'num2']
        >>> b = ['foo', 'bar', 'baz']
        >>> index = index_natsorted(a)
        >>> index
        [2, 0, 1]
        >>> # Sort both lists by the sort order of a
        >>> order_by_index(a, index)
        ['num2', 'num3', 'num5']
        >>> order_by_index(b, index)
        ['baz', 'foo', 'bar']

    """
    newkey: KeyType
    if key is None:
        newkey = itemgetter(1)
    else:

        def newkey(x: Any) -> NatsortInType:
            return cast(KeyType, key)(itemgetter(1)(x))

    # Pair the index and sequence together, then sort by element
    index_seq_pair = [(x, y) for x, y in enumerate(seq)]
    index_seq_pair.sort(reverse=reverse, key=natsort_keygen(newkey, alg))
    return [x for x, _ in index_seq_pair]


@overload
def index_humansorted(
    seq: Iter_ns, key: None = None, reverse: bool = False, alg: NSType = ns.DEFAULT
) -> List_int:
    ...


@overload
def index_humansorted(
    seq: Iter_any, key: KeyType, reverse: bool = False, alg: NSType = ns.DEFAULT
) -> List_int:
    ...


def index_humansorted(
    seq: Iter_any,
    key: MaybeKeyType = None,
    reverse: bool = False,
    alg: NSType = ns.DEFAULT,
) -> List_int:
    """
    This is a wrapper around ``index_natsorted(seq, alg=ns.LOCALE)``.

    Parameters
    ----------
    seq: iterable
        The input to sort.

    key: callable, optional
        A key used to determine how to sort each element of the sequence.
        It is **not** applied recursively.
        It should accept a single argument and return a single value.

    reverse : {{True, False}}, optional
        Return the list in reversed sorted order. The default is
        `False`.

    alg : ns enum, optional
        This option is used to control which algorithm `natsort`
        uses when sorting. For details into these options, please see
        the :class:`ns` class documentation. The default is `ns.LOCALE`.

    Returns
    -------
    out : tuple
        The ordered indexes of the input.

    See Also
    --------
    humansorted
    order_by_index

    Notes
    -----
    Please read :ref:`locale_issues` before using `humansorted`.

    Examples
    --------
    Use `index_humansorted` just like the builtin `sorted`::

        >>> a = ['Apple', 'Banana', 'apple', 'banana']
        >>> index_humansorted(a)
        [2, 0, 3, 1]

    """
    return index_natsorted(seq, key, reverse, alg | ns.LOCALE)


@overload
def index_realsorted(
    seq: Iter_ns, key: None = None, reverse: bool = False, alg: NSType = ns.DEFAULT
) -> List_int:
    ...


@overload
def index_realsorted(
    seq: Iter_any, key: KeyType, reverse: bool = False, alg: NSType = ns.DEFAULT
) -> List_int:
    ...


def index_realsorted(
    seq: Iter_any,
    key: MaybeKeyType = None,
    reverse: bool = False,
    alg: NSType = ns.DEFAULT,
) -> List_int:
    """
    This is a wrapper around ``index_natsorted(seq, alg=ns.REAL)``.

    Parameters
    ----------
    seq: iterable
        The input to sort.

    key: callable, optional
        A key used to determine how to sort each element of the sequence.
        It is **not** applied recursively.
        It should accept a single argument and return a single value.

    reverse : {{True, False}}, optional
        Return the list in reversed sorted order. The default is
        `False`.

    alg : ns enum, optional
        This option is used to control which algorithm `natsort`
        uses when sorting. For details into these options, please see
        the :class:`ns` class documentation. The default is `ns.REAL`.

    Returns
    -------
    out : tuple
        The ordered indexes of the input.

    See Also
    --------
    realsorted
    order_by_index

    Examples
    --------
    Use `index_realsorted` just like the builtin `sorted`::

        >>> a = ['num5.10', 'num-3', 'num5.3', 'num2']
        >>> index_realsorted(a)
        [1, 3, 0, 2]

    """
    return index_natsorted(seq, key, reverse, alg | ns.REAL)


# noinspection PyShadowingBuiltins,PyUnresolvedReferences
def order_by_index(
    seq: Sequence[Any], index: Iterable[int], iter: bool = False
) -> Iter_any:
    """
    Order a given sequence by an index sequence.

    The output of `index_natsorted` is a
    sequence of integers (index) that correspond to how its input
    sequence **would** be sorted. The idea is that this index can
    be used to reorder multiple sequences by the sorted order of the
    first sequence. This function is a convenient wrapper to
    apply this ordering to a sequence.

    Parameters
    ----------
    seq : sequence
        The sequence to order.

    index : iterable
        The iterable that indicates how to order `seq`.
        It should be the same length as `seq` and consist
        of integers only.

    iter : {{True, False}}, optional
        If `True`, the ordered sequence is returned as a
        iterator; otherwise it is returned as a
        list. The default is `False`.

    Returns
    -------
    out : {{list, iterator}}
        The sequence ordered by `index`, as a `list` or as an
        iterator (depending on the value of `iter`).

    See Also
    --------
    index_natsorted
    index_humansorted
    index_realsorted

    Examples
    --------

    `order_by_index` is a convenience function that helps you apply
    the result of `index_natsorted`::

        >>> a = ['num3', 'num5', 'num2']
        >>> b = ['foo', 'bar', 'baz']
        >>> index = index_natsorted(a)
        >>> index
        [2, 0, 1]
        >>> # Sort both lists by the sort order of a
        >>> order_by_index(a, index)
        ['num2', 'num3', 'num5']
        >>> order_by_index(b, index)
        ['baz', 'foo', 'bar']

    """
    return (seq[i] for i in index) if iter else [seq[i] for i in index]


def numeric_regex_chooser(alg: NSType) -> str:
    """
    Select an appropriate regex for the type of number of interest.

    Parameters
    ----------
    alg : ns enum
        Used to indicate the regular expression to select.

    Returns
    -------
    regex : str
        Regular expression string that matches the desired number type.

    """
    # Remove the leading and trailing parens
    return utils.regex_chooser(alg).pattern[1:-1]


def _split_apply(v: Any, key: MaybeKeyType = None) -> Iterator[str]:
    if key is not None:
        v = key(v)
    return utils.path_splitter(str(v))


# Choose the implementation based on the host OS
if platform.system() == "Windows":

    from ctypes import wintypes, windll  # type: ignore
    from functools import cmp_to_key

    _windows_sort_cmp = windll.Shlwapi.StrCmpLogicalW
    _windows_sort_cmp.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _windows_sort_cmp.restype = wintypes.INT
    _winsort_key = cmp_to_key(_windows_sort_cmp)

    def os_sort_keygen(key: MaybeKeyType = None) -> OSSortKeyType:
        return cast(
            OSSortKeyType, lambda x: tuple(map(_winsort_key, _split_apply(x, key)))
        )

else:

    # For UNIX-based platforms, ICU performs MUCH better than locale
    # at replicating the file explorer's sort order. We will use
    # ICU's ability to do basic natural sorting as it also better
    # replicates than what natsort does by default.
    #
    # However, if the user does not have ICU installed then fall back
    # on natsort's default handling for paths with locale turned on
    # which will give good results in most cases (e.g. when there aren't
    # a bunch of special characters).
    try:
        import icu

    except ImportError:
        # No ICU installed
        def os_sort_keygen(key: MaybeKeyType = None) -> OSSortKeyType:
            return cast(
                OSSortKeyType,
                natsort_keygen(key=key, alg=ns.LOCALE | ns.PATH | ns.IGNORECASE),
            )

    else:
        # ICU installed
        def os_sort_keygen(key: MaybeKeyType = None) -> OSSortKeyType:
            loc = natsort.compat.locale.get_icu_locale()
            collator = icu.Collator.createInstance(loc)
            collator.setAttribute(
                icu.UCollAttribute.NUMERIC_COLLATION, icu.UCollAttributeValue.ON
            )
            return lambda x: tuple(map(collator.getSortKey, _split_apply(x, key)))


os_sort_keygen.__doc__ = """
Generate a sorting key to replicate your file browser's sort order

See :func:`os_sorted` for description and caveats.

Returns
-------
out : function
    A function that parses input for OS path sorting that is
    suitable for passing as the `key` argument to functions
    such as `sorted`.

See Also
--------
os_sort_key
os_sorted

Notes
-----
On Windows, this will implicitly coerce all inputs to str before
collating.

"""

os_sort_key = os_sort_keygen()
os_sort_key.__doc__ = """
os_sort_key(val)
The default key to replicate your file browser's sort order

This is the output of :func:`os_sort_keygen` with default values.

See Also
--------
os_sort_keygen

"""


@overload
def os_sorted(seq: Iter_path, key: None = None, reverse: bool = False) -> List_path:
    ...


@overload
def os_sorted(seq: Iter_any, key: KeyType, reverse: bool = False) -> List_any:
    ...


def os_sorted(
    seq: Iter_any, key: MaybeKeyType = None, reverse: bool = False
) -> List_any:
    """
    Sort elements in the same order as your operating system's file browser

    .. warning::

        The resulting function will generate results that will be
        different depending on your platform. This is intentional.

    On Windows, this will sort with the same order as Windows Explorer.

    On MacOS/Linux, you will get different results depending on whether
    or not you have :mod:`pyicu` installed.

    - If you have :mod:`pyicu` installed, you will get results that are
      the same as (or very close to) the same order as your operating
      system's file browser.
    - If you do not have :mod:`pyicu` installed, then this will give
      the same results as if you used ``ns.LOCALE``, ``ns.PATH``,
      and ``ns.IGNORECASE`` with :func:`natsorted`. If you do not have
      special characters this will give correct results, but once
      special characters are added you should lower your expectations.

    It is *strongly* recommended to have :mod:`pyicu` installed on
    MacOS/Linux if you want correct sort results.

    It does *not* take into account if a path is a directory or a file
    when sorting.

    Parameters
    ----------
    seq : iterable
        The input to sort. Each element must be of type str.

    key : callable, optional
        A key used to determine how to sort each element of the sequence.
        It should accept a single argument and return a single value.

    reverse : {{True, False}}, optional
        Return the list in reversed sorted order. The default is
        `False`.

    Returns
    -------
    out : list
        The sorted input.

    See Also
    --------
    natsorted
    os_sort_keygen

    Notes
    -----
    This will implicitly coerce all inputs to str before collating.

    """
    return sorted(seq, key=os_sort_keygen(key), reverse=reverse)
