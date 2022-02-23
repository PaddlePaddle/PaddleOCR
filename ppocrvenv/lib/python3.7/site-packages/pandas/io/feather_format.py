""" feather-format compat """

from typing import AnyStr

from pandas._typing import (
    FilePathOrBuffer,
    StorageOptions,
)
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc

from pandas import (
    DataFrame,
    Int64Index,
    RangeIndex,
)
from pandas.core import generic

from pandas.io.common import get_handle


@doc(storage_options=generic._shared_docs["storage_options"])
def to_feather(
    df: DataFrame,
    path: FilePathOrBuffer[AnyStr],
    storage_options: StorageOptions = None,
    **kwargs,
):
    """
    Write a DataFrame to the binary Feather format.

    Parameters
    ----------
    df : DataFrame
    path : string file path, or file-like object
    {storage_options}

        .. versionadded:: 1.2.0

    **kwargs :
        Additional keywords passed to `pyarrow.feather.write_feather`.

        .. versionadded:: 1.1.0
    """
    import_optional_dependency("pyarrow")
    from pyarrow import feather

    if not isinstance(df, DataFrame):
        raise ValueError("feather only support IO with DataFrames")

    valid_types = {"string", "unicode"}

    # validate index
    # --------------

    # validate that we have only a default index
    # raise on anything else as we don't serialize the index

    if not isinstance(df.index, (Int64Index, RangeIndex)):
        typ = type(df.index)
        raise ValueError(
            f"feather does not support serializing {typ} "
            "for the index; you can .reset_index() to make the index into column(s)"
        )

    if not df.index.equals(RangeIndex.from_range(range(len(df)))):
        raise ValueError(
            "feather does not support serializing a non-default index for the index; "
            "you can .reset_index() to make the index into column(s)"
        )

    if df.index.name is not None:
        raise ValueError(
            "feather does not serialize index meta-data on a default index"
        )

    # validate columns
    # ----------------

    # must have value column names (strings only)
    if df.columns.inferred_type not in valid_types:
        raise ValueError("feather must have string column names")

    with get_handle(
        path, "wb", storage_options=storage_options, is_text=False
    ) as handles:
        feather.write_feather(df, handles.handle, **kwargs)


@doc(storage_options=generic._shared_docs["storage_options"])
def read_feather(
    path, columns=None, use_threads: bool = True, storage_options: StorageOptions = None
):
    """
    Load a feather-format object from the file path.

    Parameters
    ----------
    path : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.feather``.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handle (e.g. via builtin ``open`` function)
        or ``StringIO``.
    columns : sequence, default None
        If not provided, all columns are read.
    use_threads : bool, default True
        Whether to parallelize reading using multiple threads.
    {storage_options}

        .. versionadded:: 1.2.0

    Returns
    -------
    type of object stored in file
    """
    import_optional_dependency("pyarrow")
    from pyarrow import feather

    with get_handle(
        path, "rb", storage_options=storage_options, is_text=False
    ) as handles:

        return feather.read_feather(
            handles.handle, columns=columns, use_threads=bool(use_threads)
        )
