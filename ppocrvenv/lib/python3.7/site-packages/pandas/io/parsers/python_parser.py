from __future__ import annotations

from collections import (
    abc,
    defaultdict,
)
from copy import copy
import csv
from io import StringIO
import re
import sys
from typing import (
    DefaultDict,
    Iterator,
    cast,
)
import warnings

import numpy as np

import pandas._libs.lib as lib
from pandas._typing import FilePathOrBuffer
from pandas.errors import (
    EmptyDataError,
    ParserError,
)

from pandas.core.dtypes.common import is_integer
from pandas.core.dtypes.inference import is_dict_like

from pandas.io.parsers.base_parser import (
    ParserBase,
    parser_defaults,
)

# BOM character (byte order mark)
# This exists at the beginning of a file to indicate endianness
# of a file (stream). Unfortunately, this marker screws up parsing,
# so we need to remove it if we see it.
_BOM = "\ufeff"


class PythonParser(ParserBase):
    def __init__(self, f: FilePathOrBuffer | list, **kwds):
        """
        Workhorse function for processing nested list into DataFrame
        """
        ParserBase.__init__(self, kwds)

        self.data: Iterator[str] | None = None
        self.buf: list = []
        self.pos = 0
        self.line_pos = 0

        self.skiprows = kwds["skiprows"]

        if callable(self.skiprows):
            self.skipfunc = self.skiprows
        else:
            self.skipfunc = lambda x: x in self.skiprows

        self.skipfooter = _validate_skipfooter_arg(kwds["skipfooter"])
        self.delimiter = kwds["delimiter"]

        self.quotechar = kwds["quotechar"]
        if isinstance(self.quotechar, str):
            self.quotechar = str(self.quotechar)

        self.escapechar = kwds["escapechar"]
        self.doublequote = kwds["doublequote"]
        self.skipinitialspace = kwds["skipinitialspace"]
        self.lineterminator = kwds["lineterminator"]
        self.quoting = kwds["quoting"]
        self.skip_blank_lines = kwds["skip_blank_lines"]

        self.names_passed = kwds["names"] or None

        self.has_index_names = False
        if "has_index_names" in kwds:
            self.has_index_names = kwds["has_index_names"]

        self.verbose = kwds["verbose"]
        self.converters = kwds["converters"]

        self.dtype = copy(kwds["dtype"])
        self.thousands = kwds["thousands"]
        self.decimal = kwds["decimal"]

        self.comment = kwds["comment"]

        # Set self.data to something that can read lines.
        if isinstance(f, list):
            # read_excel: f is a list
            self.data = cast(Iterator[str], f)
        else:
            self._open_handles(f, kwds)
            assert self.handles is not None
            assert hasattr(self.handles.handle, "readline")
            try:
                self._make_reader(self.handles.handle)
            except (csv.Error, UnicodeDecodeError):
                self.close()
                raise

        # Get columns in two steps: infer from data, then
        # infer column indices from self.usecols if it is specified.
        self._col_indices: list[int] | None = None
        try:
            (
                self.columns,
                self.num_original_columns,
                self.unnamed_cols,
            ) = self._infer_columns()
        except (TypeError, ValueError):
            self.close()
            raise

        # Now self.columns has the set of columns that we will process.
        # The original set is stored in self.original_columns.
        if len(self.columns) > 1:
            # we are processing a multi index column
            # error: Cannot determine type of 'index_names'
            # error: Cannot determine type of 'col_names'
            (
                self.columns,
                self.index_names,
                self.col_names,
                _,
            ) = self._extract_multi_indexer_columns(
                self.columns,
                self.index_names,  # type: ignore[has-type]
                self.col_names,  # type: ignore[has-type]
            )
            # Update list of original names to include all indices.
            self.num_original_columns = len(self.columns)
        else:
            self.columns = self.columns[0]

        # get popped off for index
        self.orig_names: list[int | str | tuple] = list(self.columns)

        # needs to be cleaned/refactored
        # multiple date column thing turning into a real spaghetti factory

        if not self._has_complex_date_col:
            (index_names, self.orig_names, self.columns) = self._get_index_name(
                self.columns
            )
            self._name_processed = True
            if self.index_names is None:
                self.index_names = index_names

        if self._col_indices is None:
            self._col_indices = list(range(len(self.columns)))

        self._validate_parse_dates_presence(self.columns)
        no_thousands_columns: set[int] | None = None
        if self.parse_dates:
            no_thousands_columns = self._set_noconvert_dtype_columns(
                self._col_indices, self.columns
            )
        self._no_thousands_columns = no_thousands_columns

        if len(self.decimal) != 1:
            raise ValueError("Only length-1 decimal markers supported")

        decimal = re.escape(self.decimal)
        if self.thousands is None:
            regex = fr"^[\-\+]?[0-9]*({decimal}[0-9]*)?([0-9]?(E|e)\-?[0-9]+)?$"
        else:
            thousands = re.escape(self.thousands)
            regex = (
                fr"^[\-\+]?([0-9]+{thousands}|[0-9])*({decimal}[0-9]*)?"
                fr"([0-9]?(E|e)\-?[0-9]+)?$"
            )
        self.num = re.compile(regex)

    def _make_reader(self, f):
        sep = self.delimiter

        if sep is None or len(sep) == 1:
            if self.lineterminator:
                raise ValueError(
                    "Custom line terminators not supported in python parser (yet)"
                )

            class MyDialect(csv.Dialect):
                delimiter = self.delimiter
                quotechar = self.quotechar
                escapechar = self.escapechar
                doublequote = self.doublequote
                skipinitialspace = self.skipinitialspace
                quoting = self.quoting
                lineterminator = "\n"

            dia = MyDialect

            if sep is not None:
                dia.delimiter = sep
            else:
                # attempt to sniff the delimiter from the first valid line,
                # i.e. no comment line and not in skiprows
                line = f.readline()
                lines = self._check_comments([[line]])[0]
                while self.skipfunc(self.pos) or not lines:
                    self.pos += 1
                    line = f.readline()
                    lines = self._check_comments([[line]])[0]

                # since `line` was a string, lines will be a list containing
                # only a single string
                line = lines[0]

                self.pos += 1
                self.line_pos += 1
                sniffed = csv.Sniffer().sniff(line)
                dia.delimiter = sniffed.delimiter

                # Note: encoding is irrelevant here
                line_rdr = csv.reader(StringIO(line), dialect=dia)
                self.buf.extend(list(line_rdr))

            # Note: encoding is irrelevant here
            reader = csv.reader(f, dialect=dia, strict=True)

        else:

            def _read():
                line = f.readline()
                pat = re.compile(sep)

                yield pat.split(line.strip())

                for line in f:
                    yield pat.split(line.strip())

            reader = _read()

        # error: Incompatible types in assignment (expression has type "_reader",
        # variable has type "Union[IO[Any], RawIOBase, BufferedIOBase, TextIOBase,
        # TextIOWrapper, mmap, None]")
        self.data = reader  # type: ignore[assignment]

    def read(self, rows=None):
        try:
            content = self._get_lines(rows)
        except StopIteration:
            if self._first_chunk:
                content = []
            else:
                self.close()
                raise

        # done with first read, next time raise StopIteration
        self._first_chunk = False

        columns = list(self.orig_names)
        if not len(content):  # pragma: no cover
            # DataFrame with the right metadata, even though it's length 0
            names = self._maybe_dedup_names(self.orig_names)
            # error: Cannot determine type of 'index_col'
            index, columns, col_dict = self._get_empty_meta(
                names,
                self.index_col,  # type: ignore[has-type]
                self.index_names,
                self.dtype,
            )
            columns = self._maybe_make_multi_index_columns(columns, self.col_names)
            return index, columns, col_dict

        # handle new style for names in index
        count_empty_content_vals = count_empty_vals(content[0])
        indexnamerow = None
        if self.has_index_names and count_empty_content_vals == len(columns):
            indexnamerow = content[0]
            content = content[1:]

        alldata = self._rows_to_cols(content)
        data, columns = self._exclude_implicit_index(alldata)

        columns, data = self._do_date_conversions(columns, data)

        data = self._convert_data(data)
        index, columns = self._make_index(data, alldata, columns, indexnamerow)

        return index, columns, data

    def _exclude_implicit_index(self, alldata):
        names = self._maybe_dedup_names(self.orig_names)

        offset = 0
        if self._implicit_index:
            # error: Cannot determine type of 'index_col'
            offset = len(self.index_col)  # type: ignore[has-type]

        len_alldata = len(alldata)
        self._check_data_length(names, alldata)

        return {
            name: alldata[i + offset] for i, name in enumerate(names) if i < len_alldata
        }, names

    # legacy
    def get_chunk(self, size=None):
        if size is None:
            # error: "PythonParser" has no attribute "chunksize"
            size = self.chunksize  # type: ignore[attr-defined]
        return self.read(rows=size)

    def _convert_data(self, data):
        # apply converters
        def _clean_mapping(mapping):
            """converts col numbers to names"""
            clean = {}
            for col, v in mapping.items():
                if isinstance(col, int) and col not in self.orig_names:
                    col = self.orig_names[col]
                clean[col] = v
            return clean

        clean_conv = _clean_mapping(self.converters)
        if not isinstance(self.dtype, dict):
            # handles single dtype applied to all columns
            clean_dtypes = self.dtype
        else:
            clean_dtypes = _clean_mapping(self.dtype)

        # Apply NA values.
        clean_na_values = {}
        clean_na_fvalues = {}

        if isinstance(self.na_values, dict):
            for col in self.na_values:
                na_value = self.na_values[col]
                na_fvalue = self.na_fvalues[col]

                if isinstance(col, int) and col not in self.orig_names:
                    col = self.orig_names[col]

                clean_na_values[col] = na_value
                clean_na_fvalues[col] = na_fvalue
        else:
            clean_na_values = self.na_values
            clean_na_fvalues = self.na_fvalues

        return self._convert_to_ndarrays(
            data,
            clean_na_values,
            clean_na_fvalues,
            self.verbose,
            clean_conv,
            clean_dtypes,
        )

    def _infer_columns(self):
        names = self.names
        num_original_columns = 0
        clear_buffer = True
        unnamed_cols: set[str | int | None] = set()

        if self.header is not None:
            header = self.header

            if isinstance(header, (list, tuple, np.ndarray)):
                have_mi_columns = len(header) > 1
                # we have a mi columns, so read an extra line
                if have_mi_columns:
                    header = list(header) + [header[-1] + 1]
            else:
                have_mi_columns = False
                header = [header]

            columns: list[list[int | str | None]] = []
            for level, hr in enumerate(header):
                try:
                    line = self._buffered_line()

                    while self.line_pos <= hr:
                        line = self._next_line()

                except StopIteration as err:
                    if self.line_pos < hr:
                        raise ValueError(
                            f"Passed header={hr} but only {self.line_pos + 1} lines in "
                            "file"
                        ) from err

                    # We have an empty file, so check
                    # if columns are provided. That will
                    # serve as the 'line' for parsing
                    if have_mi_columns and hr > 0:
                        if clear_buffer:
                            self._clear_buffer()
                        columns.append([None] * len(columns[-1]))
                        return columns, num_original_columns, unnamed_cols

                    if not self.names:
                        raise EmptyDataError("No columns to parse from file") from err

                    line = self.names[:]

                this_columns: list[int | str | None] = []
                this_unnamed_cols = []

                for i, c in enumerate(line):
                    if c == "":
                        if have_mi_columns:
                            col_name = f"Unnamed: {i}_level_{level}"
                        else:
                            col_name = f"Unnamed: {i}"

                        this_unnamed_cols.append(i)
                        this_columns.append(col_name)
                    else:
                        this_columns.append(c)

                if not have_mi_columns and self.mangle_dupe_cols:
                    counts: DefaultDict = defaultdict(int)

                    for i, col in enumerate(this_columns):
                        old_col = col
                        cur_count = counts[col]

                        if cur_count > 0:
                            while cur_count > 0:
                                counts[col] = cur_count + 1
                                col = f"{col}.{cur_count}"
                                cur_count = counts[col]
                            if (
                                self.dtype is not None
                                and is_dict_like(self.dtype)
                                and self.dtype.get(old_col) is not None
                                and self.dtype.get(col) is None
                            ):
                                self.dtype.update({col: self.dtype.get(old_col)})
                        this_columns[i] = col
                        counts[col] = cur_count + 1
                elif have_mi_columns:

                    # if we have grabbed an extra line, but its not in our
                    # format so save in the buffer, and create an blank extra
                    # line for the rest of the parsing code
                    if hr == header[-1]:
                        lc = len(this_columns)
                        # error: Cannot determine type of 'index_col'
                        sic = self.index_col  # type: ignore[has-type]
                        ic = len(sic) if sic is not None else 0
                        unnamed_count = len(this_unnamed_cols)

                        # if wrong number of blanks or no index, not our format
                        if (lc != unnamed_count and lc - ic > unnamed_count) or ic == 0:
                            clear_buffer = False
                            this_columns = [None] * lc
                            self.buf = [self.buf[-1]]

                columns.append(this_columns)
                unnamed_cols.update({this_columns[i] for i in this_unnamed_cols})

                if len(columns) == 1:
                    num_original_columns = len(this_columns)

            if clear_buffer:
                self._clear_buffer()

            if names is not None:
                if len(names) > len(columns[0]):
                    raise ValueError(
                        "Number of passed names did not match "
                        "number of header fields in the file"
                    )
                if len(columns) > 1:
                    raise TypeError("Cannot pass names with multi-index columns")

                if self.usecols is not None:
                    # Set _use_cols. We don't store columns because they are
                    # overwritten.
                    self._handle_usecols(columns, names, num_original_columns)
                else:
                    num_original_columns = len(names)
                if self._col_indices is not None and len(names) != len(
                    self._col_indices
                ):
                    columns = [[names[i] for i in sorted(self._col_indices)]]
                else:
                    columns = [names]
            else:
                columns = self._handle_usecols(
                    columns, columns[0], num_original_columns
                )
        else:
            try:
                line = self._buffered_line()

            except StopIteration as err:
                if not names:
                    raise EmptyDataError("No columns to parse from file") from err

                line = names[:]

            ncols = len(line)
            num_original_columns = ncols

            if not names:
                if self.prefix:
                    columns = [[f"{self.prefix}{i}" for i in range(ncols)]]
                else:
                    columns = [list(range(ncols))]
                columns = self._handle_usecols(
                    columns, columns[0], num_original_columns
                )
            else:
                if self.usecols is None or len(names) >= num_original_columns:
                    columns = self._handle_usecols([names], names, num_original_columns)
                    num_original_columns = len(names)
                else:
                    if not callable(self.usecols) and len(names) != len(self.usecols):
                        raise ValueError(
                            "Number of passed names did not match number of "
                            "header fields in the file"
                        )
                    # Ignore output but set used columns.
                    self._handle_usecols([names], names, ncols)
                    columns = [names]
                    num_original_columns = ncols

        return columns, num_original_columns, unnamed_cols

    def _handle_usecols(
        self,
        columns: list[list[str | int | None]],
        usecols_key: list[str | int | None],
        num_original_columns: int,
    ):
        """
        Sets self._col_indices

        usecols_key is used if there are string usecols.
        """
        if self.usecols is not None:
            if callable(self.usecols):
                col_indices = self._evaluate_usecols(self.usecols, usecols_key)
            elif any(isinstance(u, str) for u in self.usecols):
                if len(columns) > 1:
                    raise ValueError(
                        "If using multiple headers, usecols must be integers."
                    )
                col_indices = []

                for col in self.usecols:
                    if isinstance(col, str):
                        try:
                            col_indices.append(usecols_key.index(col))
                        except ValueError:
                            self._validate_usecols_names(self.usecols, usecols_key)
                    else:
                        col_indices.append(col)
            else:
                missing_usecols = [
                    col for col in self.usecols if col >= num_original_columns
                ]
                if missing_usecols:
                    warnings.warn(
                        "Defining usecols with out of bounds indices is deprecated "
                        "and will raise a ParserError in a future version.",
                        FutureWarning,
                        stacklevel=8,
                    )
                col_indices = self.usecols

            columns = [
                [n for i, n in enumerate(column) if i in col_indices]
                for column in columns
            ]
            self._col_indices = sorted(col_indices)
        return columns

    def _buffered_line(self):
        """
        Return a line from buffer, filling buffer if required.
        """
        if len(self.buf) > 0:
            return self.buf[0]
        else:
            return self._next_line()

    def _check_for_bom(self, first_row):
        """
        Checks whether the file begins with the BOM character.
        If it does, remove it. In addition, if there is quoting
        in the field subsequent to the BOM, remove it as well
        because it technically takes place at the beginning of
        the name, not the middle of it.
        """
        # first_row will be a list, so we need to check
        # that that list is not empty before proceeding.
        if not first_row:
            return first_row

        # The first element of this row is the one that could have the
        # BOM that we want to remove. Check that the first element is a
        # string before proceeding.
        if not isinstance(first_row[0], str):
            return first_row

        # Check that the string is not empty, as that would
        # obviously not have a BOM at the start of it.
        if not first_row[0]:
            return first_row

        # Since the string is non-empty, check that it does
        # in fact begin with a BOM.
        first_elt = first_row[0][0]
        if first_elt != _BOM:
            return first_row

        first_row_bom = first_row[0]

        if len(first_row_bom) > 1 and first_row_bom[1] == self.quotechar:
            start = 2
            quote = first_row_bom[1]
            end = first_row_bom[2:].index(quote) + 2

            # Extract the data between the quotation marks
            new_row = first_row_bom[start:end]

            # Extract any remaining data after the second
            # quotation mark.
            if len(first_row_bom) > end + 1:
                new_row += first_row_bom[end + 1 :]

        else:

            # No quotation so just remove BOM from first element
            new_row = first_row_bom[1:]
        return [new_row] + first_row[1:]

    def _is_line_empty(self, line):
        """
        Check if a line is empty or not.

        Parameters
        ----------
        line : str, array-like
            The line of data to check.

        Returns
        -------
        boolean : Whether or not the line is empty.
        """
        return not line or all(not x for x in line)

    def _next_line(self):
        if isinstance(self.data, list):
            while self.skipfunc(self.pos):
                self.pos += 1

            while True:
                try:
                    line = self._check_comments([self.data[self.pos]])[0]
                    self.pos += 1
                    # either uncommented or blank to begin with
                    if not self.skip_blank_lines and (
                        self._is_line_empty(self.data[self.pos - 1]) or line
                    ):
                        break
                    elif self.skip_blank_lines:
                        ret = self._remove_empty_lines([line])
                        if ret:
                            line = ret[0]
                            break
                except IndexError:
                    raise StopIteration
        else:
            while self.skipfunc(self.pos):
                self.pos += 1
                # assert for mypy, data is Iterator[str] or None, would error in next
                assert self.data is not None
                next(self.data)

            while True:
                orig_line = self._next_iter_line(row_num=self.pos + 1)
                self.pos += 1

                if orig_line is not None:
                    line = self._check_comments([orig_line])[0]

                    if self.skip_blank_lines:
                        ret = self._remove_empty_lines([line])

                        if ret:
                            line = ret[0]
                            break
                    elif self._is_line_empty(orig_line) or line:
                        break

        # This was the first line of the file,
        # which could contain the BOM at the
        # beginning of it.
        if self.pos == 1:
            line = self._check_for_bom(line)

        self.line_pos += 1
        self.buf.append(line)
        return line

    def _alert_malformed(self, msg, row_num):
        """
        Alert a user about a malformed row, depending on value of
        `self.on_bad_lines` enum.

        If `self.on_bad_lines` is ERROR, the alert will be `ParserError`.
        If `self.on_bad_lines` is WARN, the alert will be printed out.

        Parameters
        ----------
        msg : The error message to display.
        row_num : The row number where the parsing error occurred.
                  Because this row number is displayed, we 1-index,
                  even though we 0-index internally.
        """
        if self.on_bad_lines == self.BadLineHandleMethod.ERROR:
            raise ParserError(msg)
        elif self.on_bad_lines == self.BadLineHandleMethod.WARN:
            base = f"Skipping line {row_num}: "
            sys.stderr.write(base + msg + "\n")

    def _next_iter_line(self, row_num):
        """
        Wrapper around iterating through `self.data` (CSV source).

        When a CSV error is raised, we check for specific
        error messages that allow us to customize the
        error message displayed to the user.

        Parameters
        ----------
        row_num : The row number of the line being parsed.
        """
        try:
            # assert for mypy, data is Iterator[str] or None, would error in next
            assert self.data is not None
            return next(self.data)
        except csv.Error as e:
            if (
                self.on_bad_lines == self.BadLineHandleMethod.ERROR
                or self.on_bad_lines == self.BadLineHandleMethod.WARN
            ):
                msg = str(e)

                if "NULL byte" in msg or "line contains NUL" in msg:
                    msg = (
                        "NULL byte detected. This byte "
                        "cannot be processed in Python's "
                        "native csv library at the moment, "
                        "so please pass in engine='c' instead"
                    )

                if self.skipfooter > 0:
                    reason = (
                        "Error could possibly be due to "
                        "parsing errors in the skipped footer rows "
                        "(the skipfooter keyword is only applied "
                        "after Python's csv library has parsed "
                        "all rows)."
                    )
                    msg += ". " + reason

                self._alert_malformed(msg, row_num)
            return None

    def _check_comments(self, lines):
        if self.comment is None:
            return lines
        ret = []
        for line in lines:
            rl = []
            for x in line:
                if (
                    not isinstance(x, str)
                    or self.comment not in x
                    or x in self.na_values
                ):
                    rl.append(x)
                else:
                    x = x[: x.find(self.comment)]
                    if len(x) > 0:
                        rl.append(x)
                    break
            ret.append(rl)
        return ret

    def _remove_empty_lines(self, lines):
        """
        Iterate through the lines and remove any that are
        either empty or contain only one whitespace value

        Parameters
        ----------
        lines : array-like
            The array of lines that we are to filter.

        Returns
        -------
        filtered_lines : array-like
            The same array of lines with the "empty" ones removed.
        """
        ret = []
        for line in lines:
            # Remove empty lines and lines with only one whitespace value
            if (
                len(line) > 1
                or len(line) == 1
                and (not isinstance(line[0], str) or line[0].strip())
            ):
                ret.append(line)
        return ret

    def _check_thousands(self, lines):
        if self.thousands is None:
            return lines

        return self._search_replace_num_columns(
            lines=lines, search=self.thousands, replace=""
        )

    def _search_replace_num_columns(self, lines, search, replace):
        ret = []
        for line in lines:
            rl = []
            for i, x in enumerate(line):
                if (
                    not isinstance(x, str)
                    or search not in x
                    or (self._no_thousands_columns and i in self._no_thousands_columns)
                    or not self.num.search(x.strip())
                ):
                    rl.append(x)
                else:
                    rl.append(x.replace(search, replace))
            ret.append(rl)
        return ret

    def _check_decimal(self, lines):
        if self.decimal == parser_defaults["decimal"]:
            return lines

        return self._search_replace_num_columns(
            lines=lines, search=self.decimal, replace="."
        )

    def _clear_buffer(self):
        self.buf = []

    _implicit_index = False

    def _get_index_name(self, columns):
        """
        Try several cases to get lines:

        0) There are headers on row 0 and row 1 and their
        total summed lengths equals the length of the next line.
        Treat row 0 as columns and row 1 as indices
        1) Look for implicit index: there are more columns
        on row 1 than row 0. If this is true, assume that row
        1 lists index columns and row 0 lists normal columns.
        2) Get index from the columns if it was listed.
        """
        orig_names = list(columns)
        columns = list(columns)

        try:
            line = self._next_line()
        except StopIteration:
            line = None

        try:
            next_line = self._next_line()
        except StopIteration:
            next_line = None

        # implicitly index_col=0 b/c 1 fewer column names
        implicit_first_cols = 0
        if line is not None:
            # leave it 0, #2442
            # Case 1
            # error: Cannot determine type of 'index_col'
            index_col = self.index_col  # type: ignore[has-type]
            if index_col is not False:
                implicit_first_cols = len(line) - self.num_original_columns

            # Case 0
            if next_line is not None:
                if len(next_line) == len(line) + self.num_original_columns:
                    # column and index names on diff rows
                    self.index_col = list(range(len(line)))
                    self.buf = self.buf[1:]

                    for c in reversed(line):
                        columns.insert(0, c)

                    # Update list of original names to include all indices.
                    orig_names = list(columns)
                    self.num_original_columns = len(columns)
                    return line, orig_names, columns

        if implicit_first_cols > 0:
            # Case 1
            self._implicit_index = True
            if self.index_col is None:
                self.index_col = list(range(implicit_first_cols))

            index_name = None

        else:
            # Case 2
            (index_name, columns_, self.index_col) = self._clean_index_names(
                columns, self.index_col, self.unnamed_cols
            )

        return index_name, orig_names, columns

    def _rows_to_cols(self, content):
        col_len = self.num_original_columns

        if self._implicit_index:
            col_len += len(self.index_col)

        max_len = max(len(row) for row in content)

        # Check that there are no rows with too many
        # elements in their row (rows with too few
        # elements are padded with NaN).
        # error: Non-overlapping identity check (left operand type: "List[int]",
        # right operand type: "Literal[False]")
        if (
            max_len > col_len
            and self.index_col is not False  # type: ignore[comparison-overlap]
            and self.usecols is None
        ):

            footers = self.skipfooter if self.skipfooter else 0
            bad_lines = []

            iter_content = enumerate(content)
            content_len = len(content)
            content = []

            for (i, l) in iter_content:
                actual_len = len(l)

                if actual_len > col_len:
                    if (
                        self.on_bad_lines == self.BadLineHandleMethod.ERROR
                        or self.on_bad_lines == self.BadLineHandleMethod.WARN
                    ):
                        row_num = self.pos - (content_len - i + footers)
                        bad_lines.append((row_num, actual_len))

                        if self.on_bad_lines == self.BadLineHandleMethod.ERROR:
                            break
                else:
                    content.append(l)

            for row_num, actual_len in bad_lines:
                msg = (
                    f"Expected {col_len} fields in line {row_num + 1}, saw "
                    f"{actual_len}"
                )
                if (
                    self.delimiter
                    and len(self.delimiter) > 1
                    and self.quoting != csv.QUOTE_NONE
                ):
                    # see gh-13374
                    reason = (
                        "Error could possibly be due to quotes being "
                        "ignored when a multi-char delimiter is used."
                    )
                    msg += ". " + reason

                self._alert_malformed(msg, row_num + 1)

        # see gh-13320
        zipped_content = list(lib.to_object_array(content, min_width=col_len).T)

        if self.usecols:
            assert self._col_indices is not None
            col_indices = self._col_indices

            if self._implicit_index:
                zipped_content = [
                    a
                    for i, a in enumerate(zipped_content)
                    if (
                        i < len(self.index_col)
                        or i - len(self.index_col) in col_indices
                    )
                ]
            else:
                zipped_content = [
                    a for i, a in enumerate(zipped_content) if i in col_indices
                ]
        return zipped_content

    def _get_lines(self, rows=None):
        lines = self.buf
        new_rows = None

        # already fetched some number
        if rows is not None:
            # we already have the lines in the buffer
            if len(self.buf) >= rows:
                new_rows, self.buf = self.buf[:rows], self.buf[rows:]

            # need some lines
            else:
                rows -= len(self.buf)

        if new_rows is None:
            if isinstance(self.data, list):
                if self.pos > len(self.data):
                    raise StopIteration
                if rows is None:
                    new_rows = self.data[self.pos :]
                    new_pos = len(self.data)
                else:
                    new_rows = self.data[self.pos : self.pos + rows]
                    new_pos = self.pos + rows

                # Check for stop rows. n.b.: self.skiprows is a set.
                if self.skiprows:
                    new_rows = [
                        row
                        for i, row in enumerate(new_rows)
                        if not self.skipfunc(i + self.pos)
                    ]

                lines.extend(new_rows)
                self.pos = new_pos

            else:
                new_rows = []
                try:
                    if rows is not None:
                        for _ in range(rows):
                            # assert for mypy, data is Iterator[str] or None, would
                            # error in next
                            assert self.data is not None
                            new_rows.append(next(self.data))
                        lines.extend(new_rows)
                    else:
                        rows = 0

                        while True:
                            new_row = self._next_iter_line(row_num=self.pos + rows + 1)
                            rows += 1

                            if new_row is not None:
                                new_rows.append(new_row)

                except StopIteration:
                    if self.skiprows:
                        new_rows = [
                            row
                            for i, row in enumerate(new_rows)
                            if not self.skipfunc(i + self.pos)
                        ]
                    lines.extend(new_rows)
                    if len(lines) == 0:
                        raise
                self.pos += len(new_rows)

            self.buf = []
        else:
            lines = new_rows

        if self.skipfooter:
            lines = lines[: -self.skipfooter]

        lines = self._check_comments(lines)
        if self.skip_blank_lines:
            lines = self._remove_empty_lines(lines)
        lines = self._check_thousands(lines)
        return self._check_decimal(lines)


class FixedWidthReader(abc.Iterator):
    """
    A reader of fixed-width lines.
    """

    def __init__(self, f, colspecs, delimiter, comment, skiprows=None, infer_nrows=100):
        self.f = f
        self.buffer = None
        self.delimiter = "\r\n" + delimiter if delimiter else "\n\r\t "
        self.comment = comment
        if colspecs == "infer":
            self.colspecs = self.detect_colspecs(
                infer_nrows=infer_nrows, skiprows=skiprows
            )
        else:
            self.colspecs = colspecs

        if not isinstance(self.colspecs, (tuple, list)):
            raise TypeError(
                "column specifications must be a list or tuple, "
                f"input was a {type(colspecs).__name__}"
            )

        for colspec in self.colspecs:
            if not (
                isinstance(colspec, (tuple, list))
                and len(colspec) == 2
                and isinstance(colspec[0], (int, np.integer, type(None)))
                and isinstance(colspec[1], (int, np.integer, type(None)))
            ):
                raise TypeError(
                    "Each column specification must be "
                    "2 element tuple or list of integers"
                )

    def get_rows(self, infer_nrows, skiprows=None):
        """
        Read rows from self.f, skipping as specified.

        We distinguish buffer_rows (the first <= infer_nrows
        lines) from the rows returned to detect_colspecs
        because it's simpler to leave the other locations
        with skiprows logic alone than to modify them to
        deal with the fact we skipped some rows here as
        well.

        Parameters
        ----------
        infer_nrows : int
            Number of rows to read from self.f, not counting
            rows that are skipped.
        skiprows: set, optional
            Indices of rows to skip.

        Returns
        -------
        detect_rows : list of str
            A list containing the rows to read.

        """
        if skiprows is None:
            skiprows = set()
        buffer_rows = []
        detect_rows = []
        for i, row in enumerate(self.f):
            if i not in skiprows:
                detect_rows.append(row)
            buffer_rows.append(row)
            if len(detect_rows) >= infer_nrows:
                break
        self.buffer = iter(buffer_rows)
        return detect_rows

    def detect_colspecs(self, infer_nrows=100, skiprows=None):
        # Regex escape the delimiters
        delimiters = "".join(fr"\{x}" for x in self.delimiter)
        pattern = re.compile(f"([^{delimiters}]+)")
        rows = self.get_rows(infer_nrows, skiprows)
        if not rows:
            raise EmptyDataError("No rows from which to infer column width")
        max_len = max(map(len, rows))
        mask = np.zeros(max_len + 1, dtype=int)
        if self.comment is not None:
            rows = [row.partition(self.comment)[0] for row in rows]
        for row in rows:
            for m in pattern.finditer(row):
                mask[m.start() : m.end()] = 1
        shifted = np.roll(mask, 1)
        shifted[0] = 0
        edges = np.where((mask ^ shifted) == 1)[0]
        edge_pairs = list(zip(edges[::2], edges[1::2]))
        return edge_pairs

    def __next__(self):
        if self.buffer is not None:
            try:
                line = next(self.buffer)
            except StopIteration:
                self.buffer = None
                line = next(self.f)
        else:
            line = next(self.f)
        # Note: 'colspecs' is a sequence of half-open intervals.
        return [line[fromm:to].strip(self.delimiter) for (fromm, to) in self.colspecs]


class FixedWidthFieldParser(PythonParser):
    """
    Specialization that Converts fixed-width fields into DataFrames.
    See PythonParser for details.
    """

    def __init__(self, f, **kwds):
        # Support iterators, convert to a list.
        self.colspecs = kwds.pop("colspecs")
        self.infer_nrows = kwds.pop("infer_nrows")
        PythonParser.__init__(self, f, **kwds)

    def _make_reader(self, f):
        self.data = FixedWidthReader(
            f,
            self.colspecs,
            self.delimiter,
            self.comment,
            self.skiprows,
            self.infer_nrows,
        )

    def _remove_empty_lines(self, lines) -> list:
        """
        Returns the list of lines without the empty ones. With fixed-width
        fields, empty lines become arrays of empty strings.

        See PythonParser._remove_empty_lines.
        """
        return [
            line
            for line in lines
            if any(not isinstance(e, str) or e.strip() for e in line)
        ]


def count_empty_vals(vals) -> int:
    return sum(1 for v in vals if v == "" or v is None)


def _validate_skipfooter_arg(skipfooter: int) -> int:
    """
    Validate the 'skipfooter' parameter.

    Checks whether 'skipfooter' is a non-negative integer.
    Raises a ValueError if that is not the case.

    Parameters
    ----------
    skipfooter : non-negative integer
        The number of rows to skip at the end of the file.

    Returns
    -------
    validated_skipfooter : non-negative integer
        The original input if the validation succeeds.

    Raises
    ------
    ValueError : 'skipfooter' was not a non-negative integer.
    """
    if not is_integer(skipfooter):
        raise ValueError("skipfooter must be an integer")

    if skipfooter < 0:
        raise ValueError("skipfooter cannot be negative")

    return skipfooter
