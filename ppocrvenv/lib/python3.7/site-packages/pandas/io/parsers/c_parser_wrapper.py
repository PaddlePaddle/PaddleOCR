from __future__ import annotations

import warnings

import numpy as np

import pandas._libs.parsers as parsers
from pandas._typing import (
    ArrayLike,
    FilePathOrBuffer,
)
from pandas.errors import DtypeWarning

from pandas.core.dtypes.common import (
    is_categorical_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.concat import union_categoricals
from pandas.core.dtypes.dtypes import ExtensionDtype

from pandas.core.indexes.api import ensure_index_from_sequences

from pandas.io.parsers.base_parser import (
    ParserBase,
    is_index_col,
)


class CParserWrapper(ParserBase):
    low_memory: bool
    _reader: parsers.TextReader

    def __init__(self, src: FilePathOrBuffer, **kwds):
        self.kwds = kwds
        kwds = kwds.copy()

        ParserBase.__init__(self, kwds)

        self.low_memory = kwds.pop("low_memory", False)

        # #2442
        # error: Cannot determine type of 'index_col'
        kwds["allow_leading_cols"] = (
            self.index_col is not False  # type: ignore[has-type]
        )

        # GH20529, validate usecol arg before TextReader
        kwds["usecols"] = self.usecols

        # open handles
        self._open_handles(src, kwds)
        assert self.handles is not None

        # Have to pass int, would break tests using TextReader directly otherwise :(
        kwds["on_bad_lines"] = self.on_bad_lines.value

        for key in (
            "storage_options",
            "encoding",
            "memory_map",
            "compression",
            "error_bad_lines",
            "warn_bad_lines",
        ):
            kwds.pop(key, None)

        kwds["dtype"] = ensure_dtype_objs(kwds.get("dtype", None))
        try:
            self._reader = parsers.TextReader(self.handles.handle, **kwds)
        except Exception:
            self.handles.close()
            raise

        self.unnamed_cols = self._reader.unnamed_cols

        # error: Cannot determine type of 'names'
        passed_names = self.names is None  # type: ignore[has-type]

        if self._reader.header is None:
            self.names = None
        else:
            if len(self._reader.header) > 1:
                # we have a multi index in the columns
                # error: Cannot determine type of 'names'
                # error: Cannot determine type of 'index_names'
                # error: Cannot determine type of 'col_names'
                (
                    self.names,  # type: ignore[has-type]
                    self.index_names,
                    self.col_names,
                    passed_names,
                ) = self._extract_multi_indexer_columns(
                    self._reader.header,
                    self.index_names,  # type: ignore[has-type]
                    self.col_names,  # type: ignore[has-type]
                    passed_names,
                )
            else:
                # error: Cannot determine type of 'names'
                self.names = list(self._reader.header[0])  # type: ignore[has-type]

        # error: Cannot determine type of 'names'
        if self.names is None:  # type: ignore[has-type]
            if self.prefix:
                # error: Cannot determine type of 'names'
                self.names = [  # type: ignore[has-type]
                    f"{self.prefix}{i}" for i in range(self._reader.table_width)
                ]
            else:
                # error: Cannot determine type of 'names'
                self.names = list(  # type: ignore[has-type]
                    range(self._reader.table_width)
                )

        # gh-9755
        #
        # need to set orig_names here first
        # so that proper indexing can be done
        # with _set_noconvert_columns
        #
        # once names has been filtered, we will
        # then set orig_names again to names
        # error: Cannot determine type of 'names'
        self.orig_names = self.names[:]  # type: ignore[has-type]

        if self.usecols:
            usecols = self._evaluate_usecols(self.usecols, self.orig_names)

            # GH 14671
            # assert for mypy, orig_names is List or None, None would error in issubset
            assert self.orig_names is not None
            if self.usecols_dtype == "string" and not set(usecols).issubset(
                self.orig_names
            ):
                self._validate_usecols_names(usecols, self.orig_names)

            # error: Cannot determine type of 'names'
            if len(self.names) > len(usecols):  # type: ignore[has-type]
                # error: Cannot determine type of 'names'
                self.names = [  # type: ignore[has-type]
                    n
                    # error: Cannot determine type of 'names'
                    for i, n in enumerate(self.names)  # type: ignore[has-type]
                    if (i in usecols or n in usecols)
                ]

            # error: Cannot determine type of 'names'
            if len(self.names) < len(usecols):  # type: ignore[has-type]
                # error: Cannot determine type of 'names'
                self._validate_usecols_names(
                    usecols,
                    self.names,  # type: ignore[has-type]
                )

        # error: Cannot determine type of 'names'
        self._validate_parse_dates_presence(self.names)  # type: ignore[has-type]
        self._set_noconvert_columns()

        # error: Cannot determine type of 'names'
        self.orig_names = self.names  # type: ignore[has-type]

        if not self._has_complex_date_col:
            # error: Cannot determine type of 'index_col'
            if self._reader.leading_cols == 0 and is_index_col(
                self.index_col  # type: ignore[has-type]
            ):

                self._name_processed = True
                (
                    index_names,
                    # error: Cannot determine type of 'names'
                    self.names,  # type: ignore[has-type]
                    self.index_col,
                ) = self._clean_index_names(
                    # error: Cannot determine type of 'names'
                    self.names,  # type: ignore[has-type]
                    # error: Cannot determine type of 'index_col'
                    self.index_col,  # type: ignore[has-type]
                    self.unnamed_cols,
                )

                if self.index_names is None:
                    self.index_names = index_names

            if self._reader.header is None and not passed_names:
                assert self.index_names is not None
                self.index_names = [None] * len(self.index_names)

        self._implicit_index = self._reader.leading_cols > 0

    def close(self) -> None:
        super().close()

        # close additional handles opened by C parser
        try:
            self._reader.close()
        except ValueError:
            pass

    def _set_noconvert_columns(self):
        """
        Set the columns that should not undergo dtype conversions.

        Currently, any column that is involved with date parsing will not
        undergo such conversions.
        """
        assert self.orig_names is not None
        # error: Cannot determine type of 'names'

        # much faster than using orig_names.index(x) xref GH#44106
        names_dict = {x: i for i, x in enumerate(self.orig_names)}
        col_indices = [names_dict[x] for x in self.names]  # type: ignore[has-type]
        # error: Cannot determine type of 'names'
        noconvert_columns = self._set_noconvert_dtype_columns(
            col_indices,
            self.names,  # type: ignore[has-type]
        )
        for col in noconvert_columns:
            self._reader.set_noconvert(col)

    def read(self, nrows=None):
        try:
            if self.low_memory:
                chunks = self._reader.read_low_memory(nrows)
                # destructive to chunks
                data = _concatenate_chunks(chunks)

            else:
                data = self._reader.read(nrows)
        except StopIteration:
            if self._first_chunk:
                self._first_chunk = False
                names = self._maybe_dedup_names(self.orig_names)
                index, columns, col_dict = self._get_empty_meta(
                    names,
                    self.index_col,
                    self.index_names,
                    dtype=self.kwds.get("dtype"),
                )
                columns = self._maybe_make_multi_index_columns(columns, self.col_names)

                if self.usecols is not None:
                    columns = self._filter_usecols(columns)

                col_dict = {k: v for k, v in col_dict.items() if k in columns}

                return index, columns, col_dict

            else:
                self.close()
                raise

        # Done with first read, next time raise StopIteration
        self._first_chunk = False

        # error: Cannot determine type of 'names'
        names = self.names  # type: ignore[has-type]

        if self._reader.leading_cols:
            if self._has_complex_date_col:
                raise NotImplementedError("file structure not yet supported")

            # implicit index, no index names
            arrays = []

            for i in range(self._reader.leading_cols):
                if self.index_col is None:
                    values = data.pop(i)
                else:
                    values = data.pop(self.index_col[i])

                values = self._maybe_parse_dates(values, i, try_parse_dates=True)
                arrays.append(values)

            index = ensure_index_from_sequences(arrays)

            if self.usecols is not None:
                names = self._filter_usecols(names)

            names = self._maybe_dedup_names(names)

            # rename dict keys
            data_tups = sorted(data.items())
            data = {k: v for k, (i, v) in zip(names, data_tups)}

            names, data = self._do_date_conversions(names, data)

        else:
            # rename dict keys
            data_tups = sorted(data.items())

            # ugh, mutation

            # assert for mypy, orig_names is List or None, None would error in list(...)
            assert self.orig_names is not None
            names = list(self.orig_names)
            names = self._maybe_dedup_names(names)

            if self.usecols is not None:
                names = self._filter_usecols(names)

            # columns as list
            alldata = [x[1] for x in data_tups]
            if self.usecols is None:
                self._check_data_length(names, alldata)

            data = {k: v for k, (i, v) in zip(names, data_tups)}

            names, data = self._do_date_conversions(names, data)
            index, names = self._make_index(data, alldata, names)

        # maybe create a mi on the columns
        names = self._maybe_make_multi_index_columns(names, self.col_names)

        return index, names, data

    def _filter_usecols(self, names):
        # hackish
        usecols = self._evaluate_usecols(self.usecols, names)
        if usecols is not None and len(names) != len(usecols):
            names = [
                name for i, name in enumerate(names) if i in usecols or name in usecols
            ]
        return names

    def _get_index_names(self):
        names = list(self._reader.header[0])
        idx_names = None

        if self._reader.leading_cols == 0 and self.index_col is not None:
            (idx_names, names, self.index_col) = self._clean_index_names(
                names, self.index_col, self.unnamed_cols
            )

        return names, idx_names

    def _maybe_parse_dates(self, values, index: int, try_parse_dates: bool = True):
        if try_parse_dates and self._should_parse_dates(index):
            values = self._date_conv(values)
        return values


def _concatenate_chunks(chunks: list[dict[int, ArrayLike]]) -> dict:
    """
    Concatenate chunks of data read with low_memory=True.

    The tricky part is handling Categoricals, where different chunks
    may have different inferred categories.
    """
    names = list(chunks[0].keys())
    warning_columns = []

    result = {}
    for name in names:
        arrs = [chunk.pop(name) for chunk in chunks]
        # Check each arr for consistent types.
        dtypes = {a.dtype for a in arrs}
        # TODO: shouldn't we exclude all EA dtypes here?
        numpy_dtypes = {x for x in dtypes if not is_categorical_dtype(x)}
        if len(numpy_dtypes) > 1:
            # error: Argument 1 to "find_common_type" has incompatible type
            # "Set[Any]"; expected "Sequence[Union[dtype[Any], None, type,
            # _SupportsDType, str, Union[Tuple[Any, int], Tuple[Any,
            # Union[int, Sequence[int]]], List[Any], _DTypeDict, Tuple[Any, Any]]]]"
            common_type = np.find_common_type(
                numpy_dtypes,  # type: ignore[arg-type]
                [],
            )
            # error: Non-overlapping equality check (left operand type: "dtype[Any]",
            # right operand type: "Type[object]")
            if common_type == object:  # type: ignore[comparison-overlap]
                warning_columns.append(str(name))

        dtype = dtypes.pop()
        if is_categorical_dtype(dtype):
            result[name] = union_categoricals(arrs, sort_categories=False)
        else:
            if isinstance(dtype, ExtensionDtype):
                # TODO: concat_compat?
                array_type = dtype.construct_array_type()
                # error: Argument 1 to "_concat_same_type" of "ExtensionArray"
                # has incompatible type "List[Union[ExtensionArray, ndarray]]";
                # expected "Sequence[ExtensionArray]"
                result[name] = array_type._concat_same_type(
                    arrs  # type: ignore[arg-type]
                )
            else:
                result[name] = np.concatenate(arrs)

    if warning_columns:
        warning_names = ",".join(warning_columns)
        warning_message = " ".join(
            [
                f"Columns ({warning_names}) have mixed types."
                f"Specify dtype option on import or set low_memory=False."
            ]
        )
        warnings.warn(warning_message, DtypeWarning, stacklevel=8)
    return result


def ensure_dtype_objs(dtype):
    """
    Ensure we have either None, a dtype object, or a dictionary mapping to
    dtype objects.
    """
    if isinstance(dtype, dict):
        dtype = {k: pandas_dtype(dtype[k]) for k in dtype}
    elif dtype is not None:
        dtype = pandas_dtype(dtype)
    return dtype
