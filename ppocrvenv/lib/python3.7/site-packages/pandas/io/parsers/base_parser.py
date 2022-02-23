from __future__ import annotations

from collections import defaultdict
import csv
import datetime
from enum import Enum
import itertools
from typing import (
    Any,
    Callable,
    DefaultDict,
    Iterable,
    Sequence,
    cast,
)
import warnings

import numpy as np

import pandas._libs.lib as lib
import pandas._libs.ops as libops
import pandas._libs.parsers as parsers
from pandas._libs.parsers import STR_NA_VALUES
from pandas._libs.tslibs import parsing
from pandas._typing import (
    ArrayLike,
    DtypeArg,
    FilePathOrBuffer,
    final,
)
from pandas.errors import (
    ParserError,
    ParserWarning,
)

from pandas.core.dtypes.cast import astype_nansafe
from pandas.core.dtypes.common import (
    ensure_object,
    ensure_str,
    is_bool_dtype,
    is_categorical_dtype,
    is_dict_like,
    is_dtype_equal,
    is_extension_array_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.missing import isna

from pandas.core import algorithms
from pandas.core.arrays import Categorical
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    ensure_index_from_sequences,
)
from pandas.core.series import Series
from pandas.core.tools import datetimes as tools

from pandas.io.common import (
    IOHandles,
    get_handle,
)
from pandas.io.date_converters import generic_parser

parser_defaults = {
    "delimiter": None,
    "escapechar": None,
    "quotechar": '"',
    "quoting": csv.QUOTE_MINIMAL,
    "doublequote": True,
    "skipinitialspace": False,
    "lineterminator": None,
    "header": "infer",
    "index_col": None,
    "names": None,
    "prefix": None,
    "skiprows": None,
    "skipfooter": 0,
    "nrows": None,
    "na_values": None,
    "keep_default_na": True,
    "true_values": None,
    "false_values": None,
    "converters": None,
    "dtype": None,
    "cache_dates": True,
    "thousands": None,
    "comment": None,
    "decimal": ".",
    # 'engine': 'c',
    "parse_dates": False,
    "keep_date_col": False,
    "dayfirst": False,
    "date_parser": None,
    "usecols": None,
    # 'iterator': False,
    "chunksize": None,
    "verbose": False,
    "encoding": None,
    "squeeze": False,
    "compression": None,
    "mangle_dupe_cols": True,
    "infer_datetime_format": False,
    "skip_blank_lines": True,
    "encoding_errors": "strict",
    "on_bad_lines": "error",
}


class ParserBase:
    class BadLineHandleMethod(Enum):
        ERROR = 0
        WARN = 1
        SKIP = 2

    _implicit_index: bool = False
    _first_chunk: bool

    def __init__(self, kwds):

        self.names = kwds.get("names")
        self.orig_names: list | None = None
        self.prefix = kwds.pop("prefix", None)

        self.index_col = kwds.get("index_col", None)
        self.unnamed_cols: set = set()
        self.index_names: list | None = None
        self.col_names = None

        self.parse_dates = _validate_parse_dates_arg(kwds.pop("parse_dates", False))
        self.date_parser = kwds.pop("date_parser", None)
        self.dayfirst = kwds.pop("dayfirst", False)
        self.keep_date_col = kwds.pop("keep_date_col", False)

        self.na_values = kwds.get("na_values")
        self.na_fvalues = kwds.get("na_fvalues")
        self.na_filter = kwds.get("na_filter", False)
        self.keep_default_na = kwds.get("keep_default_na", True)

        self.true_values = kwds.get("true_values")
        self.false_values = kwds.get("false_values")
        self.mangle_dupe_cols = kwds.get("mangle_dupe_cols", True)
        self.infer_datetime_format = kwds.pop("infer_datetime_format", False)
        self.cache_dates = kwds.pop("cache_dates", True)

        self._date_conv = _make_date_converter(
            date_parser=self.date_parser,
            dayfirst=self.dayfirst,
            infer_datetime_format=self.infer_datetime_format,
            cache_dates=self.cache_dates,
        )

        # validate header options for mi
        self.header = kwds.get("header")
        if isinstance(self.header, (list, tuple, np.ndarray)):
            if not all(map(is_integer, self.header)):
                raise ValueError("header must be integer or list of integers")
            if any(i < 0 for i in self.header):
                raise ValueError(
                    "cannot specify multi-index header with negative integers"
                )
            if kwds.get("usecols"):
                raise ValueError(
                    "cannot specify usecols when specifying a multi-index header"
                )
            if kwds.get("names"):
                raise ValueError(
                    "cannot specify names when specifying a multi-index header"
                )

            # validate index_col that only contains integers
            if self.index_col is not None:
                is_sequence = isinstance(self.index_col, (list, tuple, np.ndarray))
                if not (
                    is_sequence
                    and all(map(is_integer, self.index_col))
                    or is_integer(self.index_col)
                ):
                    raise ValueError(
                        "index_col must only contain row numbers "
                        "when specifying a multi-index header"
                    )
        elif self.header is not None:
            # GH 27394
            if self.prefix is not None:
                raise ValueError(
                    "Argument prefix must be None if argument header is not None"
                )
            # GH 16338
            elif not is_integer(self.header):
                raise ValueError("header must be integer or list of integers")
            # GH 27779
            elif self.header < 0:
                raise ValueError(
                    "Passing negative integer to header is invalid. "
                    "For no header, use header=None instead"
                )

        self._name_processed = False

        self._first_chunk = True

        self.usecols, self.usecols_dtype = self._validate_usecols_arg(kwds["usecols"])

        self.handles: IOHandles | None = None

        # Fallback to error to pass a sketchy test(test_override_set_noconvert_columns)
        # Normally, this arg would get pre-processed earlier on
        self.on_bad_lines = kwds.get("on_bad_lines", self.BadLineHandleMethod.ERROR)

    def _open_handles(self, src: FilePathOrBuffer, kwds: dict[str, Any]) -> None:
        """
        Let the readers open IOHandles after they are done with their potential raises.
        """
        self.handles = get_handle(
            src,
            "r",
            encoding=kwds.get("encoding", None),
            compression=kwds.get("compression", None),
            memory_map=kwds.get("memory_map", False),
            storage_options=kwds.get("storage_options", None),
            errors=kwds.get("encoding_errors", "strict"),
        )

    def _validate_parse_dates_presence(self, columns: list[str]) -> None:
        """
        Check if parse_dates are in columns.

        If user has provided names for parse_dates, check if those columns
        are available.

        Parameters
        ----------
        columns : list
            List of names of the dataframe.

        Raises
        ------
        ValueError
            If column to parse_date is not in dataframe.

        """
        cols_needed: Iterable
        if is_dict_like(self.parse_dates):
            cols_needed = itertools.chain(*self.parse_dates.values())
        elif is_list_like(self.parse_dates):
            # a column in parse_dates could be represented
            # ColReference = Union[int, str]
            # DateGroups = List[ColReference]
            # ParseDates = Union[DateGroups, List[DateGroups],
            #     Dict[ColReference, DateGroups]]
            cols_needed = itertools.chain.from_iterable(
                col if is_list_like(col) else [col] for col in self.parse_dates
            )
        else:
            cols_needed = []

        # get only columns that are references using names (str), not by index
        missing_cols = ", ".join(
            sorted(
                {
                    col
                    for col in cols_needed
                    if isinstance(col, str) and col not in columns
                }
            )
        )
        if missing_cols:
            raise ValueError(
                f"Missing column provided to 'parse_dates': '{missing_cols}'"
            )

    def close(self):
        if self.handles is not None:
            self.handles.close()

    @final
    @property
    def _has_complex_date_col(self) -> bool:
        return isinstance(self.parse_dates, dict) or (
            isinstance(self.parse_dates, list)
            and len(self.parse_dates) > 0
            and isinstance(self.parse_dates[0], list)
        )

    @final
    def _should_parse_dates(self, i: int) -> bool:
        if isinstance(self.parse_dates, bool):
            return self.parse_dates
        else:
            if self.index_names is not None:
                name = self.index_names[i]
            else:
                name = None
            j = i if self.index_col is None else self.index_col[i]

            if is_scalar(self.parse_dates):
                return (j == self.parse_dates) or (
                    name is not None and name == self.parse_dates
                )
            else:
                return (j in self.parse_dates) or (
                    name is not None and name in self.parse_dates
                )

    @final
    def _extract_multi_indexer_columns(
        self, header, index_names, col_names, passed_names: bool = False
    ):
        """
        extract and return the names, index_names, col_names
        header is a list-of-lists returned from the parsers
        """
        if len(header) < 2:
            return header[0], index_names, col_names, passed_names

        # the names are the tuples of the header that are not the index cols
        # 0 is the name of the index, assuming index_col is a list of column
        # numbers
        ic = self.index_col
        if ic is None:
            ic = []

        if not isinstance(ic, (list, tuple, np.ndarray)):
            ic = [ic]
        sic = set(ic)

        # clean the index_names
        index_names = header.pop(-1)
        index_names, _, _ = self._clean_index_names(
            index_names, self.index_col, self.unnamed_cols
        )

        # extract the columns
        field_count = len(header[0])

        def extract(r):
            return tuple(r[i] for i in range(field_count) if i not in sic)

        columns = list(zip(*(extract(r) for r in header)))
        names = ic + columns

        # If we find unnamed columns all in a single
        # level, then our header was too long.
        for n in range(len(columns[0])):
            if all(ensure_str(col[n]) in self.unnamed_cols for col in columns):
                header = ",".join(str(x) for x in self.header)
                raise ParserError(
                    f"Passed header=[{header}] are too many rows "
                    "for this multi_index of columns"
                )

        # Clean the column names (if we have an index_col).
        if len(ic):
            col_names = [
                r[0] if ((r[0] is not None) and r[0] not in self.unnamed_cols) else None
                for r in header
            ]
        else:
            col_names = [None] * len(header)

        passed_names = True

        return names, index_names, col_names, passed_names

    @final
    def _maybe_dedup_names(self, names):
        # see gh-7160 and gh-9424: this helps to provide
        # immediate alleviation of the duplicate names
        # issue and appears to be satisfactory to users,
        # but ultimately, not needing to butcher the names
        # would be nice!
        if self.mangle_dupe_cols:
            names = list(names)  # so we can index
            counts: DefaultDict[int | str | tuple, int] = defaultdict(int)
            is_potential_mi = _is_potential_multi_index(names, self.index_col)

            for i, col in enumerate(names):
                cur_count = counts[col]

                while cur_count > 0:
                    counts[col] = cur_count + 1

                    if is_potential_mi:
                        col = col[:-1] + (f"{col[-1]}.{cur_count}",)
                    else:
                        col = f"{col}.{cur_count}"
                    cur_count = counts[col]

                names[i] = col
                counts[col] = cur_count + 1

        return names

    @final
    def _maybe_make_multi_index_columns(self, columns, col_names=None):
        # possibly create a column mi here
        if _is_potential_multi_index(columns):
            columns = MultiIndex.from_tuples(columns, names=col_names)
        return columns

    @final
    def _make_index(self, data, alldata, columns, indexnamerow=False):
        if not is_index_col(self.index_col) or not self.index_col:
            index = None

        elif not self._has_complex_date_col:
            index = self._get_simple_index(alldata, columns)
            index = self._agg_index(index)
        elif self._has_complex_date_col:
            if not self._name_processed:
                (self.index_names, _, self.index_col) = self._clean_index_names(
                    list(columns), self.index_col, self.unnamed_cols
                )
                self._name_processed = True
            index = self._get_complex_date_index(data, columns)
            index = self._agg_index(index, try_parse_dates=False)

        # add names for the index
        if indexnamerow:
            coffset = len(indexnamerow) - len(columns)
            assert index is not None
            index = index.set_names(indexnamerow[:coffset])

        # maybe create a mi on the columns
        columns = self._maybe_make_multi_index_columns(columns, self.col_names)

        return index, columns

    @final
    def _get_simple_index(self, data, columns):
        def ix(col):
            if not isinstance(col, str):
                return col
            raise ValueError(f"Index {col} invalid")

        to_remove = []
        index = []
        for idx in self.index_col:
            i = ix(idx)
            to_remove.append(i)
            index.append(data[i])

        # remove index items from content and columns, don't pop in
        # loop
        for i in sorted(to_remove, reverse=True):
            data.pop(i)
            if not self._implicit_index:
                columns.pop(i)

        return index

    @final
    def _get_complex_date_index(self, data, col_names):
        def _get_name(icol):
            if isinstance(icol, str):
                return icol

            if col_names is None:
                raise ValueError(f"Must supply column order to use {icol!s} as index")

            for i, c in enumerate(col_names):
                if i == icol:
                    return c

        to_remove = []
        index = []
        for idx in self.index_col:
            name = _get_name(idx)
            to_remove.append(name)
            index.append(data[name])

        # remove index items from content and columns, don't pop in
        # loop
        for c in sorted(to_remove, reverse=True):
            data.pop(c)
            col_names.remove(c)

        return index

    @final
    def _agg_index(self, index, try_parse_dates: bool = True) -> Index:
        arrays = []

        for i, arr in enumerate(index):

            if try_parse_dates and self._should_parse_dates(i):
                arr = self._date_conv(arr)

            if self.na_filter:
                col_na_values = self.na_values
                col_na_fvalues = self.na_fvalues
            else:
                col_na_values = set()
                col_na_fvalues = set()

            if isinstance(self.na_values, dict):
                assert self.index_names is not None
                col_name = self.index_names[i]
                if col_name is not None:
                    col_na_values, col_na_fvalues = _get_na_values(
                        col_name, self.na_values, self.na_fvalues, self.keep_default_na
                    )

            arr, _ = self._infer_types(arr, col_na_values | col_na_fvalues)
            arrays.append(arr)

        names = self.index_names
        index = ensure_index_from_sequences(arrays, names)

        return index

    @final
    def _convert_to_ndarrays(
        self,
        dct: dict,
        na_values,
        na_fvalues,
        verbose: bool = False,
        converters=None,
        dtypes=None,
    ):
        result = {}
        for c, values in dct.items():
            conv_f = None if converters is None else converters.get(c, None)
            if isinstance(dtypes, dict):
                cast_type = dtypes.get(c, None)
            else:
                # single dtype or None
                cast_type = dtypes

            if self.na_filter:
                col_na_values, col_na_fvalues = _get_na_values(
                    c, na_values, na_fvalues, self.keep_default_na
                )
            else:
                col_na_values, col_na_fvalues = set(), set()

            if conv_f is not None:
                # conv_f applied to data before inference
                if cast_type is not None:
                    warnings.warn(
                        (
                            "Both a converter and dtype were specified "
                            f"for column {c} - only the converter will be used"
                        ),
                        ParserWarning,
                        stacklevel=7,
                    )

                try:
                    values = lib.map_infer(values, conv_f)
                except ValueError:
                    # error: Argument 2 to "isin" has incompatible type "List[Any]";
                    # expected "Union[Union[ExtensionArray, ndarray], Index, Series]"
                    mask = algorithms.isin(
                        values, list(na_values)  # type: ignore[arg-type]
                    ).view(np.uint8)
                    values = lib.map_infer_mask(values, conv_f, mask)

                cvals, na_count = self._infer_types(
                    values, set(col_na_values) | col_na_fvalues, try_num_bool=False
                )
            else:
                is_ea = is_extension_array_dtype(cast_type)
                is_str_or_ea_dtype = is_ea or is_string_dtype(cast_type)
                # skip inference if specified dtype is object
                # or casting to an EA
                try_num_bool = not (cast_type and is_str_or_ea_dtype)

                # general type inference and conversion
                cvals, na_count = self._infer_types(
                    values, set(col_na_values) | col_na_fvalues, try_num_bool
                )

                # type specified in dtype param or cast_type is an EA
                if cast_type and (
                    not is_dtype_equal(cvals, cast_type)
                    or is_extension_array_dtype(cast_type)
                ):
                    if not is_ea and na_count > 0:
                        try:
                            if is_bool_dtype(cast_type):
                                raise ValueError(
                                    f"Bool column has NA values in column {c}"
                                )
                        except (AttributeError, TypeError):
                            # invalid input to is_bool_dtype
                            pass
                    cast_type = pandas_dtype(cast_type)
                    cvals = self._cast_types(cvals, cast_type, c)

            result[c] = cvals
            if verbose and na_count:
                print(f"Filled {na_count} NA values in column {c!s}")
        return result

    @final
    def _set_noconvert_dtype_columns(
        self, col_indices: list[int], names: list[int | str | tuple]
    ) -> set[int]:
        """
        Set the columns that should not undergo dtype conversions.

        Currently, any column that is involved with date parsing will not
        undergo such conversions. If usecols is specified, the positions of the columns
        not to cast is relative to the usecols not to all columns.

        Parameters
        ----------
        col_indices: The indices specifying order and positions of the columns
        names: The column names which order is corresponding with the order
               of col_indices

        Returns
        -------
        A set of integers containing the positions of the columns not to convert.
        """
        usecols: list[int] | list[str] | None
        noconvert_columns = set()
        if self.usecols_dtype == "integer":
            # A set of integers will be converted to a list in
            # the correct order every single time.
            usecols = sorted(self.usecols)
        elif callable(self.usecols) or self.usecols_dtype not in ("empty", None):
            # The names attribute should have the correct columns
            # in the proper order for indexing with parse_dates.
            usecols = col_indices
        else:
            # Usecols is empty.
            usecols = None

        def _set(x) -> int:
            if usecols is not None and is_integer(x):
                x = usecols[x]

            if not is_integer(x):
                x = col_indices[names.index(x)]

            return x

        if isinstance(self.parse_dates, list):
            for val in self.parse_dates:
                if isinstance(val, list):
                    for k in val:
                        noconvert_columns.add(_set(k))
                else:
                    noconvert_columns.add(_set(val))

        elif isinstance(self.parse_dates, dict):
            for val in self.parse_dates.values():
                if isinstance(val, list):
                    for k in val:
                        noconvert_columns.add(_set(k))
                else:
                    noconvert_columns.add(_set(val))

        elif self.parse_dates:
            if isinstance(self.index_col, list):
                for k in self.index_col:
                    noconvert_columns.add(_set(k))
            elif self.index_col is not None:
                noconvert_columns.add(_set(self.index_col))

        return noconvert_columns

    def _infer_types(self, values, na_values, try_num_bool=True):
        """
        Infer types of values, possibly casting

        Parameters
        ----------
        values : ndarray
        na_values : set
        try_num_bool : bool, default try
           try to cast values to numeric (first preference) or boolean

        Returns
        -------
        converted : ndarray
        na_count : int
        """
        na_count = 0
        if issubclass(values.dtype.type, (np.number, np.bool_)):
            # error: Argument 2 to "isin" has incompatible type "List[Any]"; expected
            # "Union[Union[ExtensionArray, ndarray], Index, Series]"
            mask = algorithms.isin(values, list(na_values))  # type: ignore[arg-type]
            na_count = mask.sum()
            if na_count > 0:
                if is_integer_dtype(values):
                    values = values.astype(np.float64)
                np.putmask(values, mask, np.nan)
            return values, na_count

        if try_num_bool and is_object_dtype(values.dtype):
            # exclude e.g DatetimeIndex here
            try:
                result, _ = lib.maybe_convert_numeric(values, na_values, False)
            except (ValueError, TypeError):
                # e.g. encountering datetime string gets ValueError
                #  TypeError can be raised in floatify
                result = values
                na_count = parsers.sanitize_objects(result, na_values, False)
            else:
                na_count = isna(result).sum()
        else:
            result = values
            if values.dtype == np.object_:
                na_count = parsers.sanitize_objects(values, na_values, False)

        if result.dtype == np.object_ and try_num_bool:
            result, _ = libops.maybe_convert_bool(
                np.asarray(values),
                true_values=self.true_values,
                false_values=self.false_values,
            )

        return result, na_count

    def _cast_types(self, values, cast_type, column):
        """
        Cast values to specified type

        Parameters
        ----------
        values : ndarray
        cast_type : string or np.dtype
           dtype to cast values to
        column : string
            column name - used only for error reporting

        Returns
        -------
        converted : ndarray
        """
        if is_categorical_dtype(cast_type):
            known_cats = (
                isinstance(cast_type, CategoricalDtype)
                and cast_type.categories is not None
            )

            if not is_object_dtype(values) and not known_cats:
                # TODO: this is for consistency with
                # c-parser which parses all categories
                # as strings

                values = astype_nansafe(values, np.dtype(str))

            cats = Index(values).unique().dropna()
            values = Categorical._from_inferred_categories(
                cats, cats.get_indexer(values), cast_type, true_values=self.true_values
            )

        # use the EA's implementation of casting
        elif is_extension_array_dtype(cast_type):
            # ensure cast_type is an actual dtype and not a string
            cast_type = pandas_dtype(cast_type)
            array_type = cast_type.construct_array_type()
            try:
                if is_bool_dtype(cast_type):
                    return array_type._from_sequence_of_strings(
                        values,
                        dtype=cast_type,
                        true_values=self.true_values,
                        false_values=self.false_values,
                    )
                else:
                    return array_type._from_sequence_of_strings(values, dtype=cast_type)
            except NotImplementedError as err:
                raise NotImplementedError(
                    f"Extension Array: {array_type} must implement "
                    "_from_sequence_of_strings in order to be used in parser methods"
                ) from err

        else:
            try:
                values = astype_nansafe(values, cast_type, copy=True, skipna=True)
            except ValueError as err:
                raise ValueError(
                    f"Unable to convert column {column} to type {cast_type}"
                ) from err
        return values

    def _do_date_conversions(self, names, data):
        # returns data, columns

        if self.parse_dates is not None:
            data, names = _process_date_conversion(
                data,
                self._date_conv,
                self.parse_dates,
                self.index_col,
                self.index_names,
                names,
                keep_date_col=self.keep_date_col,
            )

        return names, data

    def _check_data_length(self, columns: list[str], data: list[ArrayLike]) -> None:
        """Checks if length of data is equal to length of column names.

        One set of trailing commas is allowed. self.index_col not False
        results in a ParserError previously when lengths do not match.

        Parameters
        ----------
        columns: list of column names
        data: list of array-likes containing the data column-wise.
        """
        if not self.index_col and len(columns) != len(data) and columns:
            if len(columns) == len(data) - 1 and np.all(
                (is_object_dtype(data[-1]) and data[-1] == "") | isna(data[-1])
            ):
                return
            warnings.warn(
                "Length of header or names does not match length of data. This leads "
                "to a loss of data with index_col=False.",
                ParserWarning,
                stacklevel=6,
            )

    def _evaluate_usecols(self, usecols, names):
        """
        Check whether or not the 'usecols' parameter
        is a callable.  If so, enumerates the 'names'
        parameter and returns a set of indices for
        each entry in 'names' that evaluates to True.
        If not a callable, returns 'usecols'.
        """
        if callable(usecols):
            return {i for i, name in enumerate(names) if usecols(name)}
        return usecols

    def _validate_usecols_names(self, usecols, names):
        """
        Validates that all usecols are present in a given
        list of names. If not, raise a ValueError that
        shows what usecols are missing.

        Parameters
        ----------
        usecols : iterable of usecols
            The columns to validate are present in names.
        names : iterable of names
            The column names to check against.

        Returns
        -------
        usecols : iterable of usecols
            The `usecols` parameter if the validation succeeds.

        Raises
        ------
        ValueError : Columns were missing. Error message will list them.
        """
        missing = [c for c in usecols if c not in names]
        if len(missing) > 0:
            raise ValueError(
                f"Usecols do not match columns, columns expected but not found: "
                f"{missing}"
            )

        return usecols

    def _validate_usecols_arg(self, usecols):
        """
        Validate the 'usecols' parameter.

        Checks whether or not the 'usecols' parameter contains all integers
        (column selection by index), strings (column by name) or is a callable.
        Raises a ValueError if that is not the case.

        Parameters
        ----------
        usecols : list-like, callable, or None
            List of columns to use when parsing or a callable that can be used
            to filter a list of table columns.

        Returns
        -------
        usecols_tuple : tuple
            A tuple of (verified_usecols, usecols_dtype).

            'verified_usecols' is either a set if an array-like is passed in or
            'usecols' if a callable or None is passed in.

            'usecols_dtype` is the inferred dtype of 'usecols' if an array-like
            is passed in or None if a callable or None is passed in.
        """
        msg = (
            "'usecols' must either be list-like of all strings, all unicode, "
            "all integers or a callable."
        )
        if usecols is not None:
            if callable(usecols):
                return usecols, None

            if not is_list_like(usecols):
                # see gh-20529
                #
                # Ensure it is iterable container but not string.
                raise ValueError(msg)

            usecols_dtype = lib.infer_dtype(usecols, skipna=False)

            if usecols_dtype not in ("empty", "integer", "string"):
                raise ValueError(msg)

            usecols = set(usecols)

            return usecols, usecols_dtype
        return usecols, None

    def _clean_index_names(self, columns, index_col, unnamed_cols):
        if not is_index_col(index_col):
            return None, columns, index_col

        columns = list(columns)

        # In case of no rows and multiindex columns we have to set index_names to
        # list of Nones GH#38292
        if not columns:
            return [None] * len(index_col), columns, index_col

        cp_cols = list(columns)
        index_names: list[str | int | None] = []

        # don't mutate
        index_col = list(index_col)

        for i, c in enumerate(index_col):
            if isinstance(c, str):
                index_names.append(c)
                for j, name in enumerate(cp_cols):
                    if name == c:
                        index_col[i] = j
                        columns.remove(name)
                        break
            else:
                name = cp_cols[c]
                columns.remove(name)
                index_names.append(name)

        # Only clean index names that were placeholders.
        for i, name in enumerate(index_names):
            if isinstance(name, str) and name in unnamed_cols:
                index_names[i] = None

        return index_names, columns, index_col

    def _get_empty_meta(
        self, columns, index_col, index_names, dtype: DtypeArg | None = None
    ):
        columns = list(columns)

        # Convert `dtype` to a defaultdict of some kind.
        # This will enable us to write `dtype[col_name]`
        # without worrying about KeyError issues later on.
        if not is_dict_like(dtype):
            # if dtype == None, default will be object.
            default_dtype = dtype or object
            # error: Argument 1 to "defaultdict" has incompatible type "Callable[[],
            # Union[ExtensionDtype, str, dtype[Any], Type[object], Dict[Hashable,
            # Union[ExtensionDtype, Union[str, dtype[Any]], Type[str], Type[float],
            # Type[int], Type[complex], Type[bool], Type[object]]]]]"; expected
            # "Optional[Callable[[], Union[ExtensionDtype, str, dtype[Any],
            # Type[object]]]]"
            # error: Incompatible return value type (got "Union[ExtensionDtype, str,
            # dtype[Any], Type[object], Dict[Hashable, Union[ExtensionDtype, Union[str,
            # dtype[Any]], Type[str], Type[float], Type[int], Type[complex], Type[bool],
            # Type[object]]]]", expected "Union[ExtensionDtype, str, dtype[Any],
            # Type[object]]")
            dtype = defaultdict(
                lambda: default_dtype  # type: ignore[arg-type, return-value]
            )
        else:
            dtype = cast(dict, dtype)
            dtype = defaultdict(
                lambda: object,
                {columns[k] if is_integer(k) else k: v for k, v in dtype.items()},
            )

        # Even though we have no data, the "index" of the empty DataFrame
        # could for example still be an empty MultiIndex. Thus, we need to
        # check whether we have any index columns specified, via either:
        #
        # 1) index_col (column indices)
        # 2) index_names (column names)
        #
        # Both must be non-null to ensure a successful construction. Otherwise,
        # we have to create a generic empty Index.
        if (index_col is None or index_col is False) or index_names is None:
            index = Index([])
        else:
            data = [Series([], dtype=dtype[name]) for name in index_names]
            index = ensure_index_from_sequences(data, names=index_names)
            index_col.sort()

            for i, n in enumerate(index_col):
                columns.pop(n - i)

        col_dict = {col_name: Series([], dtype=dtype[col_name]) for col_name in columns}

        return index, columns, col_dict


def _make_date_converter(
    date_parser=None, dayfirst=False, infer_datetime_format=False, cache_dates=True
):
    def converter(*date_cols):
        if date_parser is None:
            strs = parsing.concat_date_cols(date_cols)

            try:
                return tools.to_datetime(
                    ensure_object(strs),
                    utc=None,
                    dayfirst=dayfirst,
                    errors="ignore",
                    infer_datetime_format=infer_datetime_format,
                    cache=cache_dates,
                ).to_numpy()

            except ValueError:
                return tools.to_datetime(
                    parsing.try_parse_dates(strs, dayfirst=dayfirst), cache=cache_dates
                )
        else:
            try:
                result = tools.to_datetime(
                    date_parser(*date_cols), errors="ignore", cache=cache_dates
                )
                if isinstance(result, datetime.datetime):
                    raise Exception("scalar parser")
                return result
            except Exception:
                try:
                    return tools.to_datetime(
                        parsing.try_parse_dates(
                            parsing.concat_date_cols(date_cols),
                            parser=date_parser,
                            dayfirst=dayfirst,
                        ),
                        errors="ignore",
                    )
                except Exception:
                    return generic_parser(date_parser, *date_cols)

    return converter


def _process_date_conversion(
    data_dict,
    converter: Callable,
    parse_spec,
    index_col,
    index_names,
    columns,
    keep_date_col: bool = False,
):
    def _isindex(colspec):
        return (isinstance(index_col, list) and colspec in index_col) or (
            isinstance(index_names, list) and colspec in index_names
        )

    new_cols = []
    new_data = {}

    orig_names = columns
    columns = list(columns)

    date_cols = set()

    if parse_spec is None or isinstance(parse_spec, bool):
        return data_dict, columns

    if isinstance(parse_spec, list):
        # list of column lists
        for colspec in parse_spec:
            if is_scalar(colspec):
                if isinstance(colspec, int) and colspec not in data_dict:
                    colspec = orig_names[colspec]
                if _isindex(colspec):
                    continue
                data_dict[colspec] = converter(data_dict[colspec])
            else:
                new_name, col, old_names = _try_convert_dates(
                    converter, colspec, data_dict, orig_names
                )
                if new_name in data_dict:
                    raise ValueError(f"New date column already in dict {new_name}")
                new_data[new_name] = col
                new_cols.append(new_name)
                date_cols.update(old_names)

    elif isinstance(parse_spec, dict):
        # dict of new name to column list
        for new_name, colspec in parse_spec.items():
            if new_name in data_dict:
                raise ValueError(f"Date column {new_name} already in dict")

            _, col, old_names = _try_convert_dates(
                converter, colspec, data_dict, orig_names
            )

            new_data[new_name] = col
            new_cols.append(new_name)
            date_cols.update(old_names)

    data_dict.update(new_data)
    new_cols.extend(columns)

    if not keep_date_col:
        for c in list(date_cols):
            data_dict.pop(c)
            new_cols.remove(c)

    return data_dict, new_cols


def _try_convert_dates(parser: Callable, colspec, data_dict, columns):
    colset = set(columns)
    colnames = []

    for c in colspec:
        if c in colset:
            colnames.append(c)
        elif isinstance(c, int) and c not in columns:
            colnames.append(columns[c])
        else:
            colnames.append(c)

    new_name = "_".join(str(x) for x in colnames)
    to_parse = [data_dict[c] for c in colnames if c in data_dict]

    new_col = parser(*to_parse)
    return new_name, new_col, colnames


def _get_na_values(col, na_values, na_fvalues, keep_default_na):
    """
    Get the NaN values for a given column.

    Parameters
    ----------
    col : str
        The name of the column.
    na_values : array-like, dict
        The object listing the NaN values as strings.
    na_fvalues : array-like, dict
        The object listing the NaN values as floats.
    keep_default_na : bool
        If `na_values` is a dict, and the column is not mapped in the
        dictionary, whether to return the default NaN values or the empty set.

    Returns
    -------
    nan_tuple : A length-two tuple composed of

        1) na_values : the string NaN values for that column.
        2) na_fvalues : the float NaN values for that column.
    """
    if isinstance(na_values, dict):
        if col in na_values:
            return na_values[col], na_fvalues[col]
        else:
            if keep_default_na:
                return STR_NA_VALUES, set()

            return set(), set()
    else:
        return na_values, na_fvalues


def _is_potential_multi_index(
    columns, index_col: bool | Sequence[int] | None = None
) -> bool:
    """
    Check whether or not the `columns` parameter
    could be converted into a MultiIndex.

    Parameters
    ----------
    columns : array-like
        Object which may or may not be convertible into a MultiIndex
    index_col : None, bool or list, optional
        Column or columns to use as the (possibly hierarchical) index

    Returns
    -------
    bool : Whether or not columns could become a MultiIndex
    """
    if index_col is None or isinstance(index_col, bool):
        index_col = []

    return bool(
        len(columns)
        and not isinstance(columns, MultiIndex)
        and all(isinstance(c, tuple) for c in columns if c not in list(index_col))
    )


def _validate_parse_dates_arg(parse_dates):
    """
    Check whether or not the 'parse_dates' parameter
    is a non-boolean scalar. Raises a ValueError if
    that is the case.
    """
    msg = (
        "Only booleans, lists, and dictionaries are accepted "
        "for the 'parse_dates' parameter"
    )

    if parse_dates is not None:
        if is_scalar(parse_dates):
            if not lib.is_bool(parse_dates):
                raise TypeError(msg)

        elif not isinstance(parse_dates, (list, dict)):
            raise TypeError(msg)

    return parse_dates


def is_index_col(col) -> bool:
    return col is not None and col is not False
