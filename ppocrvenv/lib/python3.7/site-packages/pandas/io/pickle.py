""" pickle compat """
import pickle
from typing import Any
import warnings

from pandas._typing import (
    CompressionOptions,
    FilePathOrBuffer,
    StorageOptions,
)
from pandas.compat import pickle_compat as pc
from pandas.util._decorators import doc

from pandas.core import generic

from pandas.io.common import get_handle


@doc(storage_options=generic._shared_docs["storage_options"])
def to_pickle(
    obj: Any,
    filepath_or_buffer: FilePathOrBuffer,
    compression: CompressionOptions = "infer",
    protocol: int = pickle.HIGHEST_PROTOCOL,
    storage_options: StorageOptions = None,
):
    """
    Pickle (serialize) object to file.

    Parameters
    ----------
    obj : any object
        Any python object.
    filepath_or_buffer : str, path object or file-like object
        File path, URL, or buffer where the pickled object will be stored.

        .. versionchanged:: 1.0.0
           Accept URL. URL has to be of S3 or GCS.

    compression : {{'infer', 'gzip', 'bz2', 'zip', 'xz', None}}, default 'infer'
        If 'infer' and 'path_or_url' is path-like, then detect compression from
        the following extensions: '.gz', '.bz2', '.zip', or '.xz' (otherwise no
        compression) If 'infer' and 'path_or_url' is not path-like, then use
        None (= no decompression).
    protocol : int
        Int which indicates which protocol should be used by the pickler,
        default HIGHEST_PROTOCOL (see [1], paragraph 12.1.2). The possible
        values for this parameter depend on the version of Python. For Python
        2.x, possible values are 0, 1, 2. For Python>=3.0, 3 is a valid value.
        For Python >= 3.4, 4 is a valid value. A negative value for the
        protocol parameter is equivalent to setting its value to
        HIGHEST_PROTOCOL.

    {storage_options}

        .. versionadded:: 1.2.0

        .. [1] https://docs.python.org/3/library/pickle.html

    See Also
    --------
    read_pickle : Load pickled pandas object (or any object) from file.
    DataFrame.to_hdf : Write DataFrame to an HDF5 file.
    DataFrame.to_sql : Write DataFrame to a SQL database.
    DataFrame.to_parquet : Write a DataFrame to the binary parquet format.

    Examples
    --------
    >>> original_df = pd.DataFrame({{"foo": range(5), "bar": range(5, 10)}})
    >>> original_df
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> pd.to_pickle(original_df, "./dummy.pkl")

    >>> unpickled_df = pd.read_pickle("./dummy.pkl")
    >>> unpickled_df
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9

    >>> import os
    >>> os.remove("./dummy.pkl")
    """
    if protocol < 0:
        protocol = pickle.HIGHEST_PROTOCOL

    with get_handle(
        filepath_or_buffer,
        "wb",
        compression=compression,
        is_text=False,
        storage_options=storage_options,
    ) as handles:
        if handles.compression["method"] in ("bz2", "xz") and protocol >= 5:
            # some weird TypeError GH#39002 with pickle 5: fallback to letting
            # pickle create the entire object and then write it to the buffer.
            # "zip" would also be here if pandas.io.common._BytesZipFile
            # wouldn't buffer write calls
            handles.handle.write(
                # error: Argument 1 to "write" of "TextIOBase" has incompatible type
                # "bytes"; expected "str"
                pickle.dumps(obj, protocol=protocol)  # type: ignore[arg-type]
            )
        else:
            # letting pickle write directly to the buffer is more memory-efficient
            pickle.dump(
                # error: Argument 2 to "dump" has incompatible type "Union[IO[Any],
                # RawIOBase, BufferedIOBase, TextIOBase, TextIOWrapper, mmap]"; expected
                # "IO[bytes]"
                obj,
                handles.handle,  # type: ignore[arg-type]
                protocol=protocol,
            )


@doc(storage_options=generic._shared_docs["storage_options"])
def read_pickle(
    filepath_or_buffer: FilePathOrBuffer,
    compression: CompressionOptions = "infer",
    storage_options: StorageOptions = None,
):
    """
    Load pickled pandas object (or any object) from file.

    .. warning::

       Loading pickled data received from untrusted sources can be
       unsafe. See `here <https://docs.python.org/3/library/pickle.html>`__.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        File path, URL, or buffer where the pickled object will be loaded from.

        .. versionchanged:: 1.0.0
           Accept URL. URL is not limited to S3 and GCS.

    compression : {{'infer', 'gzip', 'bz2', 'zip', 'xz', None}}, default 'infer'
        If 'infer' and 'path_or_url' is path-like, then detect compression from
        the following extensions: '.gz', '.bz2', '.zip', or '.xz' (otherwise no
        compression) If 'infer' and 'path_or_url' is not path-like, then use
        None (= no decompression).

    {storage_options}

        .. versionadded:: 1.2.0

    Returns
    -------
    unpickled : same type as object stored in file

    See Also
    --------
    DataFrame.to_pickle : Pickle (serialize) DataFrame object to file.
    Series.to_pickle : Pickle (serialize) Series object to file.
    read_hdf : Read HDF5 file into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.
    read_parquet : Load a parquet object, returning a DataFrame.

    Notes
    -----
    read_pickle is only guaranteed to be backwards compatible to pandas 0.20.3.

    Examples
    --------
    >>> original_df = pd.DataFrame({{"foo": range(5), "bar": range(5, 10)}})
    >>> original_df
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> pd.to_pickle(original_df, "./dummy.pkl")

    >>> unpickled_df = pd.read_pickle("./dummy.pkl")
    >>> unpickled_df
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9

    >>> import os
    >>> os.remove("./dummy.pkl")
    """
    excs_to_catch = (AttributeError, ImportError, ModuleNotFoundError, TypeError)
    with get_handle(
        filepath_or_buffer,
        "rb",
        compression=compression,
        is_text=False,
        storage_options=storage_options,
    ) as handles:

        # 1) try standard library Pickle
        # 2) try pickle_compat (older pandas version) to handle subclass changes
        # 3) try pickle_compat with latin-1 encoding upon a UnicodeDecodeError

        try:
            # TypeError for Cython complaints about object.__new__ vs Tick.__new__
            try:
                with warnings.catch_warnings(record=True):
                    # We want to silence any warnings about, e.g. moved modules.
                    warnings.simplefilter("ignore", Warning)
                    # error: Argument 1 to "load" has incompatible type "Union[IO[Any],
                    # RawIOBase, BufferedIOBase, TextIOBase, TextIOWrapper, mmap]";
                    # expected "IO[bytes]"
                    return pickle.load(handles.handle)  # type: ignore[arg-type]
            except excs_to_catch:
                # e.g.
                #  "No module named 'pandas.core.sparse.series'"
                #  "Can't get attribute '__nat_unpickle' on <module 'pandas._libs.tslib"
                return pc.load(handles.handle, encoding=None)
        except UnicodeDecodeError:
            # e.g. can occur for files written in py27; see GH#28645 and GH#31988
            return pc.load(handles.handle, encoding="latin-1")
