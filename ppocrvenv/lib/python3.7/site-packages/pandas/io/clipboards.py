""" io on the clipboard """
from io import StringIO
import warnings

from pandas.core.dtypes.generic import ABCDataFrame

from pandas import (
    get_option,
    option_context,
)


def read_clipboard(sep=r"\s+", **kwargs):  # pragma: no cover
    r"""
    Read text from clipboard and pass to read_csv.

    Parameters
    ----------
    sep : str, default '\s+'
        A string or regex delimiter. The default of '\s+' denotes
        one or more whitespace characters.

    **kwargs
        See read_csv for the full argument list.

    Returns
    -------
    DataFrame
        A parsed DataFrame object.
    """
    encoding = kwargs.pop("encoding", "utf-8")

    # only utf-8 is valid for passed value because that's what clipboard
    # supports
    if encoding is not None and encoding.lower().replace("-", "") != "utf8":
        raise NotImplementedError("reading from clipboard only supports utf-8 encoding")

    from pandas.io.clipboard import clipboard_get
    from pandas.io.parsers import read_csv

    text = clipboard_get()

    # Try to decode (if needed, as "text" might already be a string here).
    try:
        text = text.decode(kwargs.get("encoding") or get_option("display.encoding"))
    except AttributeError:
        pass

    # Excel copies into clipboard with \t separation
    # inspect no more then the 10 first lines, if they
    # all contain an equal number (>0) of tabs, infer
    # that this came from excel and set 'sep' accordingly
    lines = text[:10000].split("\n")[:-1][:10]

    # Need to remove leading white space, since read_csv
    # accepts:
    #    a  b
    # 0  1  2
    # 1  3  4

    counts = {x.lstrip(" ").count("\t") for x in lines}
    if len(lines) > 1 and len(counts) == 1 and counts.pop() != 0:
        sep = "\t"
        # check the number of leading tabs in the first line
        # to account for index columns
        index_length = len(lines[0]) - len(lines[0].lstrip(" \t"))
        if index_length != 0:
            kwargs.setdefault("index_col", list(range(index_length)))

    # Edge case where sep is specified to be None, return to default
    if sep is None and kwargs.get("delim_whitespace") is None:
        sep = r"\s+"

    # Regex separator currently only works with python engine.
    # Default to python if separator is multi-character (regex)
    if len(sep) > 1 and kwargs.get("engine") is None:
        kwargs["engine"] = "python"
    elif len(sep) > 1 and kwargs.get("engine") == "c":
        warnings.warn(
            "read_clipboard with regex separator does not work properly with c engine"
        )

    return read_csv(StringIO(text), sep=sep, **kwargs)


def to_clipboard(obj, excel=True, sep=None, **kwargs):  # pragma: no cover
    """
    Attempt to write text representation of object to the system clipboard
    The clipboard can be then pasted into Excel for example.

    Parameters
    ----------
    obj : the object to write to the clipboard
    excel : bool, defaults to True
            if True, use the provided separator, writing in a csv
            format for allowing easy pasting into excel.
            if False, write a string representation of the object
            to the clipboard
    sep : optional, defaults to tab
    other keywords are passed to to_csv

    Notes
    -----
    Requirements for your platform
      - Linux: xclip, or xsel (with PyQt4 modules)
      - Windows:
      - OS X:
    """
    encoding = kwargs.pop("encoding", "utf-8")

    # testing if an invalid encoding is passed to clipboard
    if encoding is not None and encoding.lower().replace("-", "") != "utf8":
        raise ValueError("clipboard only supports utf-8 encoding")

    from pandas.io.clipboard import clipboard_set

    if excel is None:
        excel = True

    if excel:
        try:
            if sep is None:
                sep = "\t"
            buf = StringIO()

            # clipboard_set (pyperclip) expects unicode
            obj.to_csv(buf, sep=sep, encoding="utf-8", **kwargs)
            text = buf.getvalue()

            clipboard_set(text)
            return
        except TypeError:
            warnings.warn(
                "to_clipboard in excel mode requires a single character separator."
            )
    elif sep is not None:
        warnings.warn("to_clipboard with excel=False ignores the sep argument")

    if isinstance(obj, ABCDataFrame):
        # str(df) has various unhelpful defaults, like truncation
        with option_context("display.max_colwidth", None):
            objstr = obj.to_string(**kwargs)
    else:
        objstr = str(obj)
    clipboard_set(objstr)
