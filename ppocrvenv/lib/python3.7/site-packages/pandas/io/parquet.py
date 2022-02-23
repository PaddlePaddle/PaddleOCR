""" parquet compat """
from __future__ import annotations

import io
import os
from typing import (
    Any,
    AnyStr,
)
from warnings import catch_warnings

from pandas._typing import (
    FilePathOrBuffer,
    StorageOptions,
)
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc

from pandas import (
    DataFrame,
    MultiIndex,
    get_option,
)
from pandas.core import generic
from pandas.util.version import Version

from pandas.io.common import (
    IOHandles,
    get_handle,
    is_fsspec_url,
    is_url,
    stringify_path,
)


def get_engine(engine: str) -> BaseImpl:
    """return our implementation"""
    if engine == "auto":
        engine = get_option("io.parquet.engine")

    if engine == "auto":
        # try engines in this order
        engine_classes = [PyArrowImpl, FastParquetImpl]

        error_msgs = ""
        for engine_class in engine_classes:
            try:
                return engine_class()
            except ImportError as err:
                error_msgs += "\n - " + str(err)

        raise ImportError(
            "Unable to find a usable engine; "
            "tried using: 'pyarrow', 'fastparquet'.\n"
            "A suitable version of "
            "pyarrow or fastparquet is required for parquet "
            "support.\n"
            "Trying to import the above resulted in these errors:"
            f"{error_msgs}"
        )

    if engine == "pyarrow":
        return PyArrowImpl()
    elif engine == "fastparquet":
        return FastParquetImpl()

    raise ValueError("engine must be one of 'pyarrow', 'fastparquet'")


def _get_path_or_handle(
    path: FilePathOrBuffer,
    fs: Any,
    storage_options: StorageOptions = None,
    mode: str = "rb",
    is_dir: bool = False,
) -> tuple[FilePathOrBuffer, IOHandles | None, Any]:
    """File handling for PyArrow."""
    path_or_handle = stringify_path(path)
    if is_fsspec_url(path_or_handle) and fs is None:
        fsspec = import_optional_dependency("fsspec")

        fs, path_or_handle = fsspec.core.url_to_fs(
            path_or_handle, **(storage_options or {})
        )
    elif storage_options and (not is_url(path_or_handle) or mode != "rb"):
        # can't write to a remote url
        # without making use of fsspec at the moment
        raise ValueError("storage_options passed with buffer, or non-supported URL")

    handles = None
    if (
        not fs
        and not is_dir
        and isinstance(path_or_handle, str)
        and not os.path.isdir(path_or_handle)
    ):
        # use get_handle only when we are very certain that it is not a directory
        # fsspec resources can also point to directories
        # this branch is used for example when reading from non-fsspec URLs
        handles = get_handle(
            path_or_handle, mode, is_text=False, storage_options=storage_options
        )
        fs = None
        path_or_handle = handles.handle
    return path_or_handle, handles, fs


class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame):

        if not isinstance(df, DataFrame):
            raise ValueError("to_parquet only supports IO with DataFrames")

        # must have value column names for all index levels (strings only)
        if isinstance(df.columns, MultiIndex):
            if not all(
                x.inferred_type in {"string", "empty"} for x in df.columns.levels
            ):
                raise ValueError(
                    """
                    parquet must have string column names for all values in
                     each level of the MultiIndex
                    """
                )
        else:
            if df.columns.inferred_type not in {"string", "empty"}:
                raise ValueError("parquet must have string column names")

        # index level names must be strings
        valid_names = all(
            isinstance(name, str) for name in df.index.names if name is not None
        )
        if not valid_names:
            raise ValueError("Index level names must be strings")

    def write(self, df: DataFrame, path, compression, **kwargs):
        raise AbstractMethodError(self)

    def read(self, path, columns=None, **kwargs):
        raise AbstractMethodError(self)


class PyArrowImpl(BaseImpl):
    def __init__(self):
        import_optional_dependency(
            "pyarrow", extra="pyarrow is required for parquet support."
        )
        import pyarrow.parquet

        # import utils to register the pyarrow extension types
        import pandas.core.arrays._arrow_utils  # noqa

        self.api = pyarrow

    def write(
        self,
        df: DataFrame,
        path: FilePathOrBuffer[AnyStr],
        compression: str | None = "snappy",
        index: bool | None = None,
        storage_options: StorageOptions = None,
        partition_cols: list[str] | None = None,
        **kwargs,
    ):
        self.validate_dataframe(df)

        from_pandas_kwargs: dict[str, Any] = {"schema": kwargs.pop("schema", None)}
        if index is not None:
            from_pandas_kwargs["preserve_index"] = index

        table = self.api.Table.from_pandas(df, **from_pandas_kwargs)

        path_or_handle, handles, kwargs["filesystem"] = _get_path_or_handle(
            path,
            kwargs.pop("filesystem", None),
            storage_options=storage_options,
            mode="wb",
            is_dir=partition_cols is not None,
        )
        try:
            if partition_cols is not None:
                # writes to multiple files under the given path
                self.api.parquet.write_to_dataset(
                    table,
                    path_or_handle,
                    compression=compression,
                    partition_cols=partition_cols,
                    **kwargs,
                )
            else:
                # write to single output file
                self.api.parquet.write_table(
                    table, path_or_handle, compression=compression, **kwargs
                )
        finally:
            if handles is not None:
                handles.close()

    def read(
        self,
        path,
        columns=None,
        use_nullable_dtypes=False,
        storage_options: StorageOptions = None,
        **kwargs,
    ):
        kwargs["use_pandas_metadata"] = True

        to_pandas_kwargs = {}
        if use_nullable_dtypes:
            import pandas as pd

            mapping = {
                self.api.int8(): pd.Int8Dtype(),
                self.api.int16(): pd.Int16Dtype(),
                self.api.int32(): pd.Int32Dtype(),
                self.api.int64(): pd.Int64Dtype(),
                self.api.uint8(): pd.UInt8Dtype(),
                self.api.uint16(): pd.UInt16Dtype(),
                self.api.uint32(): pd.UInt32Dtype(),
                self.api.uint64(): pd.UInt64Dtype(),
                self.api.bool_(): pd.BooleanDtype(),
                self.api.string(): pd.StringDtype(),
            }
            to_pandas_kwargs["types_mapper"] = mapping.get
        manager = get_option("mode.data_manager")
        if manager == "array":
            to_pandas_kwargs["split_blocks"] = True  # type: ignore[assignment]

        path_or_handle, handles, kwargs["filesystem"] = _get_path_or_handle(
            path,
            kwargs.pop("filesystem", None),
            storage_options=storage_options,
            mode="rb",
        )
        try:
            result = self.api.parquet.read_table(
                path_or_handle, columns=columns, **kwargs
            ).to_pandas(**to_pandas_kwargs)
            if manager == "array":
                result = result._as_manager("array", copy=False)
            return result
        finally:
            if handles is not None:
                handles.close()


class FastParquetImpl(BaseImpl):
    def __init__(self):
        # since pandas is a dependency of fastparquet
        # we need to import on first use
        fastparquet = import_optional_dependency(
            "fastparquet", extra="fastparquet is required for parquet support."
        )
        self.api = fastparquet

    def write(
        self,
        df: DataFrame,
        path,
        compression="snappy",
        index=None,
        partition_cols=None,
        storage_options: StorageOptions = None,
        **kwargs,
    ):
        self.validate_dataframe(df)
        # thriftpy/protocol/compact.py:339:
        # DeprecationWarning: tostring() is deprecated.
        # Use tobytes() instead.

        if "partition_on" in kwargs and partition_cols is not None:
            raise ValueError(
                "Cannot use both partition_on and "
                "partition_cols. Use partition_cols for partitioning data"
            )
        elif "partition_on" in kwargs:
            partition_cols = kwargs.pop("partition_on")

        if partition_cols is not None:
            kwargs["file_scheme"] = "hive"

        # cannot use get_handle as write() does not accept file buffers
        path = stringify_path(path)
        if is_fsspec_url(path):
            fsspec = import_optional_dependency("fsspec")

            # if filesystem is provided by fsspec, file must be opened in 'wb' mode.
            kwargs["open_with"] = lambda path, _: fsspec.open(
                path, "wb", **(storage_options or {})
            ).open()
        elif storage_options:
            raise ValueError(
                "storage_options passed with file object or non-fsspec file path"
            )

        with catch_warnings(record=True):
            self.api.write(
                path,
                df,
                compression=compression,
                write_index=index,
                partition_on=partition_cols,
                **kwargs,
            )

    def read(
        self, path, columns=None, storage_options: StorageOptions = None, **kwargs
    ):
        parquet_kwargs: dict[str, Any] = {}
        use_nullable_dtypes = kwargs.pop("use_nullable_dtypes", False)
        if Version(self.api.__version__) >= Version("0.7.1"):
            # We are disabling nullable dtypes for fastparquet pending discussion
            parquet_kwargs["pandas_nulls"] = False
        if use_nullable_dtypes:
            raise ValueError(
                "The 'use_nullable_dtypes' argument is not supported for the "
                "fastparquet engine"
            )
        path = stringify_path(path)
        handles = None
        if is_fsspec_url(path):
            fsspec = import_optional_dependency("fsspec")

            if Version(self.api.__version__) > Version("0.6.1"):
                parquet_kwargs["fs"] = fsspec.open(
                    path, "rb", **(storage_options or {})
                ).fs
            else:
                parquet_kwargs["open_with"] = lambda path, _: fsspec.open(
                    path, "rb", **(storage_options or {})
                ).open()
        elif isinstance(path, str) and not os.path.isdir(path):
            # use get_handle only when we are very certain that it is not a directory
            # fsspec resources can also point to directories
            # this branch is used for example when reading from non-fsspec URLs
            handles = get_handle(
                path, "rb", is_text=False, storage_options=storage_options
            )
            path = handles.handle

        parquet_file = self.api.ParquetFile(path, **parquet_kwargs)

        result = parquet_file.to_pandas(columns=columns, **kwargs)

        if handles is not None:
            handles.close()
        return result


@doc(storage_options=generic._shared_docs["storage_options"])
def to_parquet(
    df: DataFrame,
    path: FilePathOrBuffer | None = None,
    engine: str = "auto",
    compression: str | None = "snappy",
    index: bool | None = None,
    storage_options: StorageOptions = None,
    partition_cols: list[str] | None = None,
    **kwargs,
) -> bytes | None:
    """
    Write a DataFrame to the parquet format.

    Parameters
    ----------
    df : DataFrame
    path : str or file-like object, default None
        If a string, it will be used as Root Directory path
        when writing a partitioned dataset. By file-like object,
        we refer to objects with a write() method, such as a file handle
        (e.g. via builtin open function) or io.BytesIO. The engine
        fastparquet does not accept file-like objects. If path is None,
        a bytes object is returned.

        .. versionchanged:: 1.2.0

    engine : {{'auto', 'pyarrow', 'fastparquet'}}, default 'auto'
        Parquet library to use. If 'auto', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try 'pyarrow', falling back to 'fastparquet' if
        'pyarrow' is unavailable.
    compression : {{'snappy', 'gzip', 'brotli', None}}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.
    index : bool, default None
        If ``True``, include the dataframe's index(es) in the file output. If
        ``False``, they will not be written to the file.
        If ``None``, similar to ``True`` the dataframe's index(es)
        will be saved. However, instead of being saved as values,
        the RangeIndex will be stored as a range in the metadata so it
        doesn't require much space and is faster. Other indexes will
        be included as columns in the file output.
    partition_cols : str or list, optional, default None
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.
        Must be None if path is not a string.
    {storage_options}

        .. versionadded:: 1.2.0

    kwargs
        Additional keyword arguments passed to the engine

    Returns
    -------
    bytes if no path argument is provided else None
    """
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]
    impl = get_engine(engine)

    path_or_buf: FilePathOrBuffer = io.BytesIO() if path is None else path

    impl.write(
        df,
        path_or_buf,
        compression=compression,
        index=index,
        partition_cols=partition_cols,
        storage_options=storage_options,
        **kwargs,
    )

    if path is None:
        assert isinstance(path_or_buf, io.BytesIO)
        return path_or_buf.getvalue()
    else:
        return None


@doc(storage_options=generic._shared_docs["storage_options"])
def read_parquet(
    path,
    engine: str = "auto",
    columns=None,
    storage_options: StorageOptions = None,
    use_nullable_dtypes: bool = False,
    **kwargs,
):
    """
    Load a parquet object from the file path, returning a DataFrame.

    Parameters
    ----------
    path : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, gs, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.parquet``.
        A file URL can also be a path to a directory that contains multiple
        partitioned parquet files. Both pyarrow and fastparquet support
        paths to directories as well as file URLs. A directory path could be:
        ``file://localhost/path/to/tables`` or ``s3://bucket/partition_dir``

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handle (e.g. via builtin ``open`` function)
        or ``StringIO``.
    engine : {{'auto', 'pyarrow', 'fastparquet'}}, default 'auto'
        Parquet library to use. If 'auto', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try 'pyarrow', falling back to 'fastparquet' if
        'pyarrow' is unavailable.
    columns : list, default=None
        If not None, only these columns will be read from the file.

    {storage_options}

        .. versionadded:: 1.3.0

    use_nullable_dtypes : bool, default False
        If True, use dtypes that use ``pd.NA`` as missing value indicator
        for the resulting DataFrame. (only applicable for the ``pyarrow``
        engine)
        As new dtypes are added that support ``pd.NA`` in the future, the
        output with this option will change to use those dtypes.
        Note: this is an experimental option, and behaviour (e.g. additional
        support dtypes) may change without notice.

        .. versionadded:: 1.2.0

    **kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    DataFrame
    """
    impl = get_engine(engine)

    return impl.read(
        path,
        columns=columns,
        storage_options=storage_options,
        use_nullable_dtypes=use_nullable_dtypes,
        **kwargs,
    )
