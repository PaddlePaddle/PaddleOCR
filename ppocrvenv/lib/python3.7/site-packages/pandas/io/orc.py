""" orc compat """
from __future__ import annotations

from typing import TYPE_CHECKING

from pandas._typing import FilePathOrBuffer
from pandas.compat._optional import import_optional_dependency

from pandas.io.common import get_handle

if TYPE_CHECKING:
    from pandas import DataFrame


def read_orc(
    path: FilePathOrBuffer, columns: list[str] | None = None, **kwargs
) -> DataFrame:
    """
    Load an ORC object from the file path, returning a DataFrame.

    .. versionadded:: 1.0.0

    Parameters
    ----------
    path : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.orc``.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handle (e.g. via builtin ``open`` function)
        or ``StringIO``.
    columns : list, default None
        If not None, only these columns will be read from the file.
    **kwargs
        Any additional kwargs are passed to pyarrow.

    Returns
    -------
    DataFrame

    Notes
    -------
    Before using this function you should read the :ref:`user guide about ORC <io.orc>`
    and :ref:`install optional dependencies <install.warn_orc>`.
    """
    # we require a newer version of pyarrow than we support for parquet

    orc = import_optional_dependency("pyarrow.orc")

    with get_handle(path, "rb", is_text=False) as handles:
        orc_file = orc.ORCFile(handles.handle)
        return orc_file.read(columns=columns, **kwargs).to_pandas()
