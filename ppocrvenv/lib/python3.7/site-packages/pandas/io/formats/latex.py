"""
Module for formatting output data in Latex.
"""
from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Iterator,
    Sequence,
)

import numpy as np

from pandas.core.dtypes.generic import ABCMultiIndex

from pandas.io.formats.format import DataFrameFormatter


def _split_into_full_short_caption(
    caption: str | tuple[str, str] | None
) -> tuple[str, str]:
    """Extract full and short captions from caption string/tuple.

    Parameters
    ----------
    caption : str or tuple, optional
        Either table caption string or tuple (full_caption, short_caption).
        If string is provided, then it is treated as table full caption,
        while short_caption is considered an empty string.

    Returns
    -------
    full_caption, short_caption : tuple
        Tuple of full_caption, short_caption strings.
    """
    if caption:
        if isinstance(caption, str):
            full_caption = caption
            short_caption = ""
        else:
            try:
                full_caption, short_caption = caption
            except ValueError as err:
                msg = "caption must be either a string or a tuple of two strings"
                raise ValueError(msg) from err
    else:
        full_caption = ""
        short_caption = ""
    return full_caption, short_caption


class RowStringConverter(ABC):
    r"""Converter for dataframe rows into LaTeX strings.

    Parameters
    ----------
    formatter : `DataFrameFormatter`
        Instance of `DataFrameFormatter`.
    multicolumn: bool, optional
        Whether to use \multicolumn macro.
    multicolumn_format: str, optional
        Multicolumn format.
    multirow: bool, optional
        Whether to use \multirow macro.

    """

    def __init__(
        self,
        formatter: DataFrameFormatter,
        multicolumn: bool = False,
        multicolumn_format: str | None = None,
        multirow: bool = False,
    ):
        self.fmt = formatter
        self.frame = self.fmt.frame
        self.multicolumn = multicolumn
        self.multicolumn_format = multicolumn_format
        self.multirow = multirow
        self.clinebuf: list[list[int]] = []
        self.strcols = self._get_strcols()
        self.strrows = list(zip(*self.strcols))

    def get_strrow(self, row_num: int) -> str:
        """Get string representation of the row."""
        row = self.strrows[row_num]

        is_multicol = (
            row_num < self.column_levels and self.fmt.header and self.multicolumn
        )

        is_multirow = (
            row_num >= self.header_levels
            and self.fmt.index
            and self.multirow
            and self.index_levels > 1
        )

        is_cline_maybe_required = is_multirow and row_num < len(self.strrows) - 1

        crow = self._preprocess_row(row)

        if is_multicol:
            crow = self._format_multicolumn(crow)
        if is_multirow:
            crow = self._format_multirow(crow, row_num)

        lst = []
        lst.append(" & ".join(crow))
        lst.append(" \\\\")
        if is_cline_maybe_required:
            cline = self._compose_cline(row_num, len(self.strcols))
            lst.append(cline)
        return "".join(lst)

    @property
    def _header_row_num(self) -> int:
        """Number of rows in header."""
        return self.header_levels if self.fmt.header else 0

    @property
    def index_levels(self) -> int:
        """Integer number of levels in index."""
        return self.frame.index.nlevels

    @property
    def column_levels(self) -> int:
        return self.frame.columns.nlevels

    @property
    def header_levels(self) -> int:
        nlevels = self.column_levels
        if self.fmt.has_index_names and self.fmt.show_index_names:
            nlevels += 1
        return nlevels

    def _get_strcols(self) -> list[list[str]]:
        """String representation of the columns."""
        if self.fmt.frame.empty:
            strcols = [[self._empty_info_line]]
        else:
            strcols = self.fmt.get_strcols()

        # reestablish the MultiIndex that has been joined by get_strcols()
        if self.fmt.index and isinstance(self.frame.index, ABCMultiIndex):
            out = self.frame.index.format(
                adjoin=False,
                sparsify=self.fmt.sparsify,
                names=self.fmt.has_index_names,
                na_rep=self.fmt.na_rep,
            )

            # index.format will sparsify repeated entries with empty strings
            # so pad these with some empty space
            def pad_empties(x):
                for pad in reversed(x):
                    if pad:
                        break
                return [x[0]] + [i if i else " " * len(pad) for i in x[1:]]

            gen = (pad_empties(i) for i in out)

            # Add empty spaces for each column level
            clevels = self.frame.columns.nlevels
            out = [[" " * len(i[-1])] * clevels + i for i in gen]

            # Add the column names to the last index column
            cnames = self.frame.columns.names
            if any(cnames):
                new_names = [i if i else "{}" for i in cnames]
                out[self.frame.index.nlevels - 1][:clevels] = new_names

            # Get rid of old multiindex column and add new ones
            strcols = out + strcols[1:]
        return strcols

    @property
    def _empty_info_line(self):
        return (
            f"Empty {type(self.frame).__name__}\n"
            f"Columns: {self.frame.columns}\n"
            f"Index: {self.frame.index}"
        )

    def _preprocess_row(self, row: Sequence[str]) -> list[str]:
        """Preprocess elements of the row."""
        if self.fmt.escape:
            crow = _escape_symbols(row)
        else:
            crow = [x if x else "{}" for x in row]
        if self.fmt.bold_rows and self.fmt.index:
            crow = _convert_to_bold(crow, self.index_levels)
        return crow

    def _format_multicolumn(self, row: list[str]) -> list[str]:
        r"""
        Combine columns belonging to a group to a single multicolumn entry
        according to self.multicolumn_format

        e.g.:
        a &  &  & b & c &
        will become
        \multicolumn{3}{l}{a} & b & \multicolumn{2}{l}{c}
        """
        row2 = row[: self.index_levels]
        ncol = 1
        coltext = ""

        def append_col():
            # write multicolumn if needed
            if ncol > 1:
                row2.append(
                    f"\\multicolumn{{{ncol:d}}}{{{self.multicolumn_format}}}"
                    f"{{{coltext.strip()}}}"
                )
            # don't modify where not needed
            else:
                row2.append(coltext)

        for c in row[self.index_levels :]:
            # if next col has text, write the previous
            if c.strip():
                if coltext:
                    append_col()
                coltext = c
                ncol = 1
            # if not, add it to the previous multicolumn
            else:
                ncol += 1
        # write last column name
        if coltext:
            append_col()
        return row2

    def _format_multirow(self, row: list[str], i: int) -> list[str]:
        r"""
        Check following rows, whether row should be a multirow

        e.g.:     becomes:
        a & 0 &   \multirow{2}{*}{a} & 0 &
          & 1 &     & 1 &
        b & 0 &   \cline{1-2}
                  b & 0 &
        """
        for j in range(self.index_levels):
            if row[j].strip():
                nrow = 1
                for r in self.strrows[i + 1 :]:
                    if not r[j].strip():
                        nrow += 1
                    else:
                        break
                if nrow > 1:
                    # overwrite non-multirow entry
                    row[j] = f"\\multirow{{{nrow:d}}}{{*}}{{{row[j].strip()}}}"
                    # save when to end the current block with \cline
                    self.clinebuf.append([i + nrow - 1, j + 1])
        return row

    def _compose_cline(self, i: int, icol: int) -> str:
        """
        Create clines after multirow-blocks are finished.
        """
        lst = []
        for cl in self.clinebuf:
            if cl[0] == i:
                lst.append(f"\n\\cline{{{cl[1]:d}-{icol:d}}}")
                # remove entries that have been written to buffer
                self.clinebuf = [x for x in self.clinebuf if x[0] != i]
        return "".join(lst)


class RowStringIterator(RowStringConverter):
    """Iterator over rows of the header or the body of the table."""

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        """Iterate over LaTeX string representations of rows."""


class RowHeaderIterator(RowStringIterator):
    """Iterator for the table header rows."""

    def __iter__(self) -> Iterator[str]:
        for row_num in range(len(self.strrows)):
            if row_num < self._header_row_num:
                yield self.get_strrow(row_num)


class RowBodyIterator(RowStringIterator):
    """Iterator for the table body rows."""

    def __iter__(self) -> Iterator[str]:
        for row_num in range(len(self.strrows)):
            if row_num >= self._header_row_num:
                yield self.get_strrow(row_num)


class TableBuilderAbstract(ABC):
    """
    Abstract table builder producing string representation of LaTeX table.

    Parameters
    ----------
    formatter : `DataFrameFormatter`
        Instance of `DataFrameFormatter`.
    column_format: str, optional
        Column format, for example, 'rcl' for three columns.
    multicolumn: bool, optional
        Use multicolumn to enhance MultiIndex columns.
    multicolumn_format: str, optional
        The alignment for multicolumns, similar to column_format.
    multirow: bool, optional
        Use multirow to enhance MultiIndex rows.
    caption: str, optional
        Table caption.
    short_caption: str, optional
        Table short caption.
    label: str, optional
        LaTeX label.
    position: str, optional
        Float placement specifier, for example, 'htb'.
    """

    def __init__(
        self,
        formatter: DataFrameFormatter,
        column_format: str | None = None,
        multicolumn: bool = False,
        multicolumn_format: str | None = None,
        multirow: bool = False,
        caption: str | None = None,
        short_caption: str | None = None,
        label: str | None = None,
        position: str | None = None,
    ):
        self.fmt = formatter
        self.column_format = column_format
        self.multicolumn = multicolumn
        self.multicolumn_format = multicolumn_format
        self.multirow = multirow
        self.caption = caption
        self.short_caption = short_caption
        self.label = label
        self.position = position

    def get_result(self) -> str:
        """String representation of LaTeX table."""
        elements = [
            self.env_begin,
            self.top_separator,
            self.header,
            self.middle_separator,
            self.env_body,
            self.bottom_separator,
            self.env_end,
        ]
        result = "\n".join(item for item in elements if item)
        trailing_newline = "\n"
        result += trailing_newline
        return result

    @property
    @abstractmethod
    def env_begin(self) -> str:
        """Beginning of the environment."""

    @property
    @abstractmethod
    def top_separator(self) -> str:
        """Top level separator."""

    @property
    @abstractmethod
    def header(self) -> str:
        """Header lines."""

    @property
    @abstractmethod
    def middle_separator(self) -> str:
        """Middle level separator."""

    @property
    @abstractmethod
    def env_body(self) -> str:
        """Environment body."""

    @property
    @abstractmethod
    def bottom_separator(self) -> str:
        """Bottom level separator."""

    @property
    @abstractmethod
    def env_end(self) -> str:
        """End of the environment."""


class GenericTableBuilder(TableBuilderAbstract):
    """Table builder producing string representation of LaTeX table."""

    @property
    def header(self) -> str:
        iterator = self._create_row_iterator(over="header")
        return "\n".join(list(iterator))

    @property
    def top_separator(self) -> str:
        return "\\toprule"

    @property
    def middle_separator(self) -> str:
        return "\\midrule" if self._is_separator_required() else ""

    @property
    def env_body(self) -> str:
        iterator = self._create_row_iterator(over="body")
        return "\n".join(list(iterator))

    def _is_separator_required(self) -> bool:
        return bool(self.header and self.env_body)

    @property
    def _position_macro(self) -> str:
        r"""Position macro, extracted from self.position, like [h]."""
        return f"[{self.position}]" if self.position else ""

    @property
    def _caption_macro(self) -> str:
        r"""Caption macro, extracted from self.caption.

        With short caption:
            \caption[short_caption]{caption_string}.

        Without short caption:
            \caption{caption_string}.
        """
        if self.caption:
            return "".join(
                [
                    r"\caption",
                    f"[{self.short_caption}]" if self.short_caption else "",
                    f"{{{self.caption}}}",
                ]
            )
        return ""

    @property
    def _label_macro(self) -> str:
        r"""Label macro, extracted from self.label, like \label{ref}."""
        return f"\\label{{{self.label}}}" if self.label else ""

    def _create_row_iterator(self, over: str) -> RowStringIterator:
        """Create iterator over header or body of the table.

        Parameters
        ----------
        over : {'body', 'header'}
            Over what to iterate.

        Returns
        -------
        RowStringIterator
            Iterator over body or header.
        """
        iterator_kind = self._select_iterator(over)
        return iterator_kind(
            formatter=self.fmt,
            multicolumn=self.multicolumn,
            multicolumn_format=self.multicolumn_format,
            multirow=self.multirow,
        )

    def _select_iterator(self, over: str) -> type[RowStringIterator]:
        """Select proper iterator over table rows."""
        if over == "header":
            return RowHeaderIterator
        elif over == "body":
            return RowBodyIterator
        else:
            msg = f"'over' must be either 'header' or 'body', but {over} was provided"
            raise ValueError(msg)


class LongTableBuilder(GenericTableBuilder):
    """Concrete table builder for longtable.

    >>> from pandas import DataFrame
    >>> from pandas.io.formats import format as fmt
    >>> df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
    >>> formatter = fmt.DataFrameFormatter(df)
    >>> builder = LongTableBuilder(formatter, caption='a long table',
    ...                            label='tab:long', column_format='lrl')
    >>> table = builder.get_result()
    >>> print(table)
    \\begin{longtable}{lrl}
    \\caption{a long table}
    \\label{tab:long}\\\\
    \\toprule
    {} &  a &   b \\\\
    \\midrule
    \\endfirsthead
    \\caption[]{a long table} \\\\
    \\toprule
    {} &  a &   b \\\\
    \\midrule
    \\endhead
    \\midrule
    \\multicolumn{3}{r}{{Continued on next page}} \\\\
    \\midrule
    \\endfoot
    <BLANKLINE>
    \\bottomrule
    \\endlastfoot
    0 &  1 &  b1 \\\\
    1 &  2 &  b2 \\\\
    \\end{longtable}
    <BLANKLINE>
    """

    @property
    def env_begin(self) -> str:
        first_row = (
            f"\\begin{{longtable}}{self._position_macro}{{{self.column_format}}}"
        )
        elements = [first_row, f"{self._caption_and_label()}"]
        return "\n".join(item for item in elements if item)

    def _caption_and_label(self) -> str:
        if self.caption or self.label:
            double_backslash = "\\\\"
            elements = [f"{self._caption_macro}", f"{self._label_macro}"]
            caption_and_label = "\n".join(item for item in elements if item)
            caption_and_label += double_backslash
            return caption_and_label
        else:
            return ""

    @property
    def middle_separator(self) -> str:
        iterator = self._create_row_iterator(over="header")

        # the content between \endfirsthead and \endhead commands
        # mitigates repeated List of Tables entries in the final LaTeX
        # document when dealing with longtable environments; GH #34360
        elements = [
            "\\midrule",
            "\\endfirsthead",
            f"\\caption[]{{{self.caption}}} \\\\" if self.caption else "",
            self.top_separator,
            self.header,
            "\\midrule",
            "\\endhead",
            "\\midrule",
            f"\\multicolumn{{{len(iterator.strcols)}}}{{r}}"
            "{{Continued on next page}} \\\\",
            "\\midrule",
            "\\endfoot\n",
            "\\bottomrule",
            "\\endlastfoot",
        ]
        if self._is_separator_required():
            return "\n".join(elements)
        return ""

    @property
    def bottom_separator(self) -> str:
        return ""

    @property
    def env_end(self) -> str:
        return "\\end{longtable}"


class RegularTableBuilder(GenericTableBuilder):
    """Concrete table builder for regular table.

    >>> from pandas import DataFrame
    >>> from pandas.io.formats import format as fmt
    >>> df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
    >>> formatter = fmt.DataFrameFormatter(df)
    >>> builder = RegularTableBuilder(formatter, caption='caption', label='lab',
    ...                               column_format='lrc')
    >>> table = builder.get_result()
    >>> print(table)
    \\begin{table}
    \\centering
    \\caption{caption}
    \\label{lab}
    \\begin{tabular}{lrc}
    \\toprule
    {} &  a &   b \\\\
    \\midrule
    0 &  1 &  b1 \\\\
    1 &  2 &  b2 \\\\
    \\bottomrule
    \\end{tabular}
    \\end{table}
    <BLANKLINE>
    """

    @property
    def env_begin(self) -> str:
        elements = [
            f"\\begin{{table}}{self._position_macro}",
            "\\centering",
            f"{self._caption_macro}",
            f"{self._label_macro}",
            f"\\begin{{tabular}}{{{self.column_format}}}",
        ]
        return "\n".join(item for item in elements if item)

    @property
    def bottom_separator(self) -> str:
        return "\\bottomrule"

    @property
    def env_end(self) -> str:
        return "\n".join(["\\end{tabular}", "\\end{table}"])


class TabularBuilder(GenericTableBuilder):
    """Concrete table builder for tabular environment.

    >>> from pandas import DataFrame
    >>> from pandas.io.formats import format as fmt
    >>> df = DataFrame({"a": [1, 2], "b": ["b1", "b2"]})
    >>> formatter = fmt.DataFrameFormatter(df)
    >>> builder = TabularBuilder(formatter, column_format='lrc')
    >>> table = builder.get_result()
    >>> print(table)
    \\begin{tabular}{lrc}
    \\toprule
    {} &  a &   b \\\\
    \\midrule
    0 &  1 &  b1 \\\\
    1 &  2 &  b2 \\\\
    \\bottomrule
    \\end{tabular}
    <BLANKLINE>
    """

    @property
    def env_begin(self) -> str:
        return f"\\begin{{tabular}}{{{self.column_format}}}"

    @property
    def bottom_separator(self) -> str:
        return "\\bottomrule"

    @property
    def env_end(self) -> str:
        return "\\end{tabular}"


class LatexFormatter:
    r"""
    Used to render a DataFrame to a LaTeX tabular/longtable environment output.

    Parameters
    ----------
    formatter : `DataFrameFormatter`
    longtable : bool, default False
        Use longtable environment.
    column_format : str, default None
        The columns format as specified in `LaTeX table format
        <https://en.wikibooks.org/wiki/LaTeX/Tables>`__ e.g 'rcl' for 3 columns
    multicolumn : bool, default False
        Use \multicolumn to enhance MultiIndex columns.
    multicolumn_format : str, default 'l'
        The alignment for multicolumns, similar to `column_format`
    multirow : bool, default False
        Use \multirow to enhance MultiIndex rows.
    caption : str or tuple, optional
        Tuple (full_caption, short_caption),
        which results in \caption[short_caption]{full_caption};
        if a single string is passed, no short caption will be set.
    label : str, optional
        The LaTeX label to be placed inside ``\label{}`` in the output.
    position : str, optional
        The LaTeX positional argument for tables, to be placed after
        ``\begin{}`` in the output.

    See Also
    --------
    HTMLFormatter
    """

    def __init__(
        self,
        formatter: DataFrameFormatter,
        longtable: bool = False,
        column_format: str | None = None,
        multicolumn: bool = False,
        multicolumn_format: str | None = None,
        multirow: bool = False,
        caption: str | tuple[str, str] | None = None,
        label: str | None = None,
        position: str | None = None,
    ):
        self.fmt = formatter
        self.frame = self.fmt.frame
        self.longtable = longtable
        self.column_format = column_format
        self.multicolumn = multicolumn
        self.multicolumn_format = multicolumn_format
        self.multirow = multirow
        self.caption, self.short_caption = _split_into_full_short_caption(caption)
        self.label = label
        self.position = position

    def to_string(self) -> str:
        """
        Render a DataFrame to a LaTeX tabular, longtable, or table/tabular
        environment output.
        """
        return self.builder.get_result()

    @property
    def builder(self) -> TableBuilderAbstract:
        """Concrete table builder.

        Returns
        -------
        TableBuilder
        """
        builder = self._select_builder()
        return builder(
            formatter=self.fmt,
            column_format=self.column_format,
            multicolumn=self.multicolumn,
            multicolumn_format=self.multicolumn_format,
            multirow=self.multirow,
            caption=self.caption,
            short_caption=self.short_caption,
            label=self.label,
            position=self.position,
        )

    def _select_builder(self) -> type[TableBuilderAbstract]:
        """Select proper table builder."""
        if self.longtable:
            return LongTableBuilder
        if any([self.caption, self.label, self.position]):
            return RegularTableBuilder
        return TabularBuilder

    @property
    def column_format(self) -> str | None:
        """Column format."""
        return self._column_format

    @column_format.setter
    def column_format(self, input_column_format: str | None) -> None:
        """Setter for column format."""
        if input_column_format is None:
            self._column_format = (
                self._get_index_format() + self._get_column_format_based_on_dtypes()
            )
        elif not isinstance(input_column_format, str):
            raise ValueError(
                f"column_format must be str or unicode, "
                f"not {type(input_column_format)}"
            )
        else:
            self._column_format = input_column_format

    def _get_column_format_based_on_dtypes(self) -> str:
        """Get column format based on data type.

        Right alignment for numbers and left - for strings.
        """

        def get_col_type(dtype):
            if issubclass(dtype.type, np.number):
                return "r"
            return "l"

        dtypes = self.frame.dtypes._values
        return "".join(map(get_col_type, dtypes))

    def _get_index_format(self) -> str:
        """Get index column format."""
        return "l" * self.frame.index.nlevels if self.fmt.index else ""


def _escape_symbols(row: Sequence[str]) -> list[str]:
    """Carry out string replacements for special symbols.

    Parameters
    ----------
    row : list
        List of string, that may contain special symbols.

    Returns
    -------
    list
        list of strings with the special symbols replaced.
    """
    return [
        (
            x.replace("\\", "\\textbackslash ")
            .replace("_", "\\_")
            .replace("%", "\\%")
            .replace("$", "\\$")
            .replace("#", "\\#")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("~", "\\textasciitilde ")
            .replace("^", "\\textasciicircum ")
            .replace("&", "\\&")
            if (x and x != "{}")
            else "{}"
        )
        for x in row
    ]


def _convert_to_bold(crow: Sequence[str], ilevels: int) -> list[str]:
    """Convert elements in ``crow`` to bold."""
    return [
        f"\\textbf{{{x}}}" if j < ilevels and x.strip() not in ["", "{}"] else x
        for j, x in enumerate(crow)
    ]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
