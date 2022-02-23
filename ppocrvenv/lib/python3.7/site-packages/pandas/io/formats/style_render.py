from __future__ import annotations

from collections import defaultdict
from functools import partial
import re
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from uuid import uuid4

import numpy as np

from pandas._config import get_option

from pandas._libs import lib
from pandas._typing import (
    FrameOrSeriesUnion,
    TypedDict,
)
from pandas.compat._optional import import_optional_dependency

from pandas.core.dtypes.generic import ABCSeries

from pandas import (
    DataFrame,
    Index,
    IndexSlice,
    MultiIndex,
    Series,
    isna,
)
from pandas.api.types import is_list_like
import pandas.core.common as com

jinja2 = import_optional_dependency("jinja2", extra="DataFrame.style requires jinja2.")
from markupsafe import escape as escape_html  # markupsafe is jinja2 dependency

BaseFormatter = Union[str, Callable]
ExtFormatter = Union[BaseFormatter, Dict[Any, Optional[BaseFormatter]]]
CSSPair = Tuple[str, Union[str, int, float]]
CSSList = List[CSSPair]
CSSProperties = Union[str, CSSList]


class CSSDict(TypedDict):
    selector: str
    props: CSSProperties


CSSStyles = List[CSSDict]
Subset = Union[slice, Sequence, Index]


class StylerRenderer:
    """
    Base class to process rendering a Styler with a specified jinja2 template.
    """

    loader = jinja2.PackageLoader("pandas", "io/formats/templates")
    env = jinja2.Environment(loader=loader, trim_blocks=True)
    template_html = env.get_template("html.tpl")
    template_html_table = env.get_template("html_table.tpl")
    template_html_style = env.get_template("html_style.tpl")
    template_latex = env.get_template("latex.tpl")

    def __init__(
        self,
        data: FrameOrSeriesUnion,
        uuid: str | None = None,
        uuid_len: int = 5,
        table_styles: CSSStyles | None = None,
        table_attributes: str | None = None,
        caption: str | tuple | None = None,
        cell_ids: bool = True,
    ):

        # validate ordered args
        if isinstance(data, Series):
            data = data.to_frame()
        if not isinstance(data, DataFrame):
            raise TypeError("``data`` must be a Series or DataFrame")
        self.data: DataFrame = data
        self.index: Index = data.index
        self.columns: Index = data.columns
        if not isinstance(uuid_len, int) or not uuid_len >= 0:
            raise TypeError("``uuid_len`` must be an integer in range [0, 32].")
        self.uuid_len = min(32, uuid_len)
        self.uuid = (uuid or uuid4().hex[: self.uuid_len]) + "_"
        self.table_styles = table_styles
        self.table_attributes = table_attributes
        self.caption = caption
        self.cell_ids = cell_ids

        # add rendering variables
        self.hide_index_: bool = False  # bools for hiding col/row headers
        self.hide_columns_: bool = False
        self.hidden_rows: Sequence[int] = []  # sequence for specific hidden rows/cols
        self.hidden_columns: Sequence[int] = []
        self.ctx: DefaultDict[tuple[int, int], CSSList] = defaultdict(list)
        self.cell_context: DefaultDict[tuple[int, int], str] = defaultdict(str)
        self._todo: list[tuple[Callable, tuple, dict]] = []
        self.tooltips: Tooltips | None = None
        def_precision = get_option("display.precision")
        self._display_funcs: DefaultDict[  # maps (row, col) -> formatting function
            tuple[int, int], Callable[[Any], str]
        ] = defaultdict(lambda: partial(_default_formatter, precision=def_precision))

    def _render_html(self, sparse_index: bool, sparse_columns: bool, **kwargs) -> str:
        """
        Renders the ``Styler`` including all applied styles to HTML.
        Generates a dict with necessary kwargs passed to jinja2 template.
        """
        self._compute()
        # TODO: namespace all the pandas keys
        d = self._translate(sparse_index, sparse_columns)
        d.update(kwargs)
        return self.template_html.render(
            **d,
            html_table_tpl=self.template_html_table,
            html_style_tpl=self.template_html_style,
        )

    def _render_latex(self, sparse_index: bool, sparse_columns: bool, **kwargs) -> str:
        """
        Render a Styler in latex format
        """
        self._compute()

        d = self._translate(sparse_index, sparse_columns, blank="")
        self._translate_latex(d)

        self.template_latex.globals["parse_wrap"] = _parse_latex_table_wrapping
        self.template_latex.globals["parse_table"] = _parse_latex_table_styles
        self.template_latex.globals["parse_cell"] = _parse_latex_cell_styles
        self.template_latex.globals["parse_header"] = _parse_latex_header_span

        d.update(kwargs)
        return self.template_latex.render(**d)

    def _compute(self):
        """
        Execute the style functions built up in `self._todo`.

        Relies on the conventions that all style functions go through
        .apply or .applymap. The append styles to apply as tuples of

        (application method, *args, **kwargs)
        """
        self.ctx.clear()
        r = self
        for func, args, kwargs in self._todo:
            r = func(self)(*args, **kwargs)
        return r

    def _translate(self, sparse_index: bool, sparse_cols: bool, blank: str = "&nbsp;"):
        """
        Process Styler data and settings into a dict for template rendering.

        Convert data and settings from ``Styler`` attributes such as ``self.data``,
        ``self.tooltips`` including applying any methods in ``self._todo``.

        Parameters
        ----------
        sparse_index : bool
            Whether to sparsify the index or print all hierarchical index elements.
            Upstream defaults are typically to `pandas.options.styler.sparse.index`.
        sparse_cols : bool
            Whether to sparsify the columns or print all hierarchical column elements.
            Upstream defaults are typically to `pandas.options.styler.sparse.columns`.

        Returns
        -------
        d : dict
            The following structure: {uuid, table_styles, caption, head, body,
            cellstyle, table_attributes}
        """
        ROW_HEADING_CLASS = "row_heading"
        COL_HEADING_CLASS = "col_heading"
        INDEX_NAME_CLASS = "index_name"
        TRIMMED_COL_CLASS = "col_trim"
        TRIMMED_ROW_CLASS = "row_trim"

        DATA_CLASS = "data"
        BLANK_CLASS = "blank"
        BLANK_VALUE = blank

        # construct render dict
        d = {
            "uuid": self.uuid,
            "table_styles": _format_table_styles(self.table_styles or []),
            "caption": self.caption,
        }

        max_elements = get_option("styler.render.max_elements")
        max_rows, max_cols = _get_trimming_maximums(
            len(self.data.index), len(self.data.columns), max_elements
        )

        head = self._translate_header(
            BLANK_CLASS,
            BLANK_VALUE,
            INDEX_NAME_CLASS,
            COL_HEADING_CLASS,
            sparse_cols,
            max_cols,
            TRIMMED_COL_CLASS,
        )
        d.update({"head": head})

        self.cellstyle_map: DefaultDict[tuple[CSSPair, ...], list[str]] = defaultdict(
            list
        )
        body = self._translate_body(
            DATA_CLASS,
            ROW_HEADING_CLASS,
            sparse_index,
            max_rows,
            max_cols,
            TRIMMED_ROW_CLASS,
            TRIMMED_COL_CLASS,
        )
        d.update({"body": body})

        cellstyle: list[dict[str, CSSList | list[str]]] = [
            {"props": list(props), "selectors": selectors}
            for props, selectors in self.cellstyle_map.items()
        ]
        d.update({"cellstyle": cellstyle})

        table_attr = self.table_attributes
        use_mathjax = get_option("display.html.use_mathjax")
        if not use_mathjax:
            table_attr = table_attr or ""
            if 'class="' in table_attr:
                table_attr = table_attr.replace('class="', 'class="tex2jax_ignore ')
            else:
                table_attr += ' class="tex2jax_ignore"'
        d.update({"table_attributes": table_attr})

        if self.tooltips:
            d = self.tooltips._translate(self.data, self.uuid, d)

        return d

    def _translate_header(
        self,
        blank_class: str,
        blank_value: str,
        index_name_class: str,
        col_heading_class: str,
        sparsify_cols: bool,
        max_cols: int,
        trimmed_col_class: str,
    ):
        """
        Build each <tr> within table <head> as a list

        Using the structure:
             +----------------------------+---------------+---------------------------+
             |  index_blanks ...          | column_name_0 |  column_headers (level_0) |
          1) |       ..                   |       ..      |             ..            |
             |  index_blanks ...          | column_name_n |  column_headers (level_n) |
             +----------------------------+---------------+---------------------------+
          2) |  index_names (level_0 to level_n) ...      | column_blanks ...         |
             +----------------------------+---------------+---------------------------+

        Parameters
        ----------
        blank_class : str
            CSS class added to elements within blank sections of the structure.
        blank_value : str
            HTML display value given to elements within blank sections of the structure.
        index_name_class : str
            CSS class added to elements within the index_names section of the structure.
        col_heading_class : str
            CSS class added to elements within the column_names section of structure.
        sparsify_cols : bool
            Whether column_headers section will add colspan attributes (>1) to elements.
        max_cols : int
            Maximum number of columns to render. If exceeded will contain `...` filler.
        trimmed_col_class : str
            CSS class added to elements within a column including `...` trimmed vals.

        Returns
        -------
        head : list
            The associated HTML elements needed for template rendering.
        """
        # for sparsifying a MultiIndex
        col_lengths = _get_level_lengths(
            self.columns, sparsify_cols, max_cols, self.hidden_columns
        )

        clabels = self.data.columns.tolist()[:max_cols]  # slice to allow trimming
        if self.data.columns.nlevels == 1:
            clabels = [[x] for x in clabels]
        clabels = list(zip(*clabels))

        head = []
        # 1) column headers
        if not self.hide_columns_:
            for r in range(self.data.columns.nlevels):
                index_blanks = [
                    _element("th", blank_class, blank_value, not self.hide_index_)
                ] * (self.data.index.nlevels - 1)

                name = self.data.columns.names[r]
                column_name = [
                    _element(
                        "th",
                        f"{blank_class if name is None else index_name_class} level{r}",
                        name if name is not None else blank_value,
                        not self.hide_index_,
                    )
                ]

                if clabels:
                    column_headers = [
                        _element(
                            "th",
                            f"{col_heading_class} level{r} col{c}",
                            value,
                            _is_visible(c, r, col_lengths),
                            attributes=(
                                f'colspan="{col_lengths.get((r, c), 0)}"'
                                if col_lengths.get((r, c), 0) > 1
                                else ""
                            ),
                        )
                        for c, value in enumerate(clabels[r])
                    ]

                    if len(self.data.columns) > max_cols:
                        # add an extra column with `...` value to indicate trimming
                        column_headers.append(
                            _element(
                                "th",
                                f"{col_heading_class} level{r} {trimmed_col_class}",
                                "...",
                                True,
                                attributes="",
                            )
                        )
                    head.append(index_blanks + column_name + column_headers)

        # 2) index names
        if (
            self.data.index.names
            and com.any_not_none(*self.data.index.names)
            and not self.hide_index_
            and not self.hide_columns_
        ):
            index_names = [
                _element(
                    "th",
                    f"{index_name_class} level{c}",
                    blank_value if name is None else name,
                    True,
                )
                for c, name in enumerate(self.data.index.names)
            ]

            if len(self.data.columns) <= max_cols:
                blank_len = len(clabels[0])
            else:
                blank_len = len(clabels[0]) + 1  # to allow room for `...` trim col

            column_blanks = [
                _element(
                    "th",
                    f"{blank_class} col{c}",
                    blank_value,
                    c not in self.hidden_columns,
                )
                for c in range(blank_len)
            ]
            head.append(index_names + column_blanks)

        return head

    def _translate_body(
        self,
        data_class: str,
        row_heading_class: str,
        sparsify_index: bool,
        max_rows: int,
        max_cols: int,
        trimmed_row_class: str,
        trimmed_col_class: str,
    ):
        """
        Build each <tr> within table <body> as a list

        Use the following structure:
          +--------------------------------------------+---------------------------+
          |  index_header_0    ...    index_header_n   |  data_by_column           |
          +--------------------------------------------+---------------------------+

        Also add elements to the cellstyle_map for more efficient grouped elements in
        <style></style> block

        Parameters
        ----------
        data_class : str
            CSS class added to elements within data_by_column sections of the structure.
        row_heading_class : str
            CSS class added to elements within the index_header section of structure.
        sparsify_index : bool
            Whether index_headers section will add rowspan attributes (>1) to elements.

        Returns
        -------
        body : list
            The associated HTML elements needed for template rendering.
        """
        # for sparsifying a MultiIndex
        idx_lengths = _get_level_lengths(
            self.index, sparsify_index, max_rows, self.hidden_rows
        )

        rlabels = self.data.index.tolist()[:max_rows]  # slice to allow trimming
        if self.data.index.nlevels == 1:
            rlabels = [[x] for x in rlabels]

        body = []
        for r, row_tup in enumerate(self.data.itertuples()):
            if r >= max_rows:  # used only to add a '...' trimmed row:
                index_headers = [
                    _element(
                        "th",
                        f"{row_heading_class} level{c} {trimmed_row_class}",
                        "...",
                        not self.hide_index_,
                        attributes="",
                    )
                    for c in range(self.data.index.nlevels)
                ]

                data = [
                    _element(
                        "td",
                        f"{data_class} col{c} {trimmed_row_class}",
                        "...",
                        (c not in self.hidden_columns),
                        attributes="",
                    )
                    for c in range(max_cols)
                ]

                if len(self.data.columns) > max_cols:
                    # columns are also trimmed so we add the final element
                    data.append(
                        _element(
                            "td",
                            f"{data_class} {trimmed_row_class} {trimmed_col_class}",
                            "...",
                            True,
                            attributes="",
                        )
                    )

                body.append(index_headers + data)
                break

            index_headers = [
                _element(
                    "th",
                    f"{row_heading_class} level{c} row{r}",
                    value,
                    (_is_visible(r, c, idx_lengths) and not self.hide_index_),
                    id=f"level{c}_row{r}",
                    attributes=(
                        f'rowspan="{idx_lengths.get((c, r), 0)}"'
                        if idx_lengths.get((c, r), 0) > 1
                        else ""
                    ),
                )
                for c, value in enumerate(rlabels[r])
            ]

            data = []
            for c, value in enumerate(row_tup[1:]):
                if c >= max_cols:
                    data.append(
                        _element(
                            "td",
                            f"{data_class} row{r} {trimmed_col_class}",
                            "...",
                            True,
                            attributes="",
                        )
                    )
                    break

                # add custom classes from cell context
                cls = ""
                if (r, c) in self.cell_context:
                    cls = " " + self.cell_context[r, c]

                data_element = _element(
                    "td",
                    f"{data_class} row{r} col{c}{cls}",
                    value,
                    (c not in self.hidden_columns and r not in self.hidden_rows),
                    attributes="",
                    display_value=self._display_funcs[(r, c)](value),
                )

                # only add an id if the cell has a style
                if self.cell_ids or (r, c) in self.ctx:
                    data_element["id"] = f"row{r}_col{c}"
                    if (r, c) in self.ctx and self.ctx[r, c]:  # only add  if non-empty
                        self.cellstyle_map[tuple(self.ctx[r, c])].append(
                            f"row{r}_col{c}"
                        )

                data.append(data_element)

            body.append(index_headers + data)
        return body

    def _translate_latex(self, d: dict) -> None:
        r"""
        Post-process the default render dict for the LaTeX template format.

        Processing items included are:
          - Remove hidden columns from the non-headers part of the body.
          - Place cellstyles directly in td cells rather than use cellstyle_map.
          - Remove hidden indexes or reinsert missing th elements if part of multiindex
            or multirow sparsification (so that \multirow and \multicol work correctly).
        """
        d["head"] = [[col for col in row if col["is_visible"]] for row in d["head"]]
        body = []
        for r, row in enumerate(d["body"]):
            if self.hide_index_:
                row_body_headers = []
            else:
                row_body_headers = [
                    {
                        **col,
                        "display_value": col["display_value"]
                        if col["is_visible"]
                        else "",
                    }
                    for col in row
                    if col["type"] == "th"
                ]

            row_body_cells = [
                {**col, "cellstyle": self.ctx[r, c - self.data.index.nlevels]}
                for c, col in enumerate(row)
                if (col["is_visible"] and col["type"] == "td")
            ]

            body.append(row_body_headers + row_body_cells)
        d["body"] = body

    def format(
        self,
        formatter: ExtFormatter | None = None,
        subset: Subset | None = None,
        na_rep: str | None = None,
        precision: int | None = None,
        decimal: str = ".",
        thousands: str | None = None,
        escape: str | None = None,
    ) -> StylerRenderer:
        r"""
        Format the text display value of cells.

        Parameters
        ----------
        formatter : str, callable, dict or None
            Object to define how values are displayed. See notes.
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        na_rep : str, optional
            Representation for missing values.
            If ``na_rep`` is None, no special formatting is applied.

            .. versionadded:: 1.0.0

        precision : int, optional
            Floating point precision to use for display purposes, if not determined by
            the specified ``formatter``.

            .. versionadded:: 1.3.0

        decimal : str, default "."
            Character used as decimal separator for floats, complex and integers

            .. versionadded:: 1.3.0

        thousands : str, optional, default None
            Character used as thousands separator for floats, complex and integers

            .. versionadded:: 1.3.0

        escape : str, optional
            Use 'html' to replace the characters ``&``, ``<``, ``>``, ``'``, and ``"``
            in cell display string with HTML-safe sequences.
            Use 'latex' to replace the characters ``&``, ``%``, ``$``, ``#``, ``_``,
            ``{``, ``}``, ``~``, ``^``, and ``\`` in the cell display string with
            LaTeX-safe sequences.
            Escaping is done before ``formatter``.

            .. versionadded:: 1.3.0

        Returns
        -------
        self : Styler

        Notes
        -----
        This method assigns a formatting function, ``formatter``, to each cell in the
        DataFrame. If ``formatter`` is ``None``, then the default formatter is used.
        If a callable then that function should take a data value as input and return
        a displayable representation, such as a string. If ``formatter`` is
        given as a string this is assumed to be a valid Python format specification
        and is wrapped to a callable as ``string.format(x)``. If a ``dict`` is given,
        keys should correspond to column names, and values should be string or
        callable, as above.

        The default formatter currently expresses floats and complex numbers with the
        pandas display precision unless using the ``precision`` argument here. The
        default formatter does not adjust the representation of missing values unless
        the ``na_rep`` argument is used.

        The ``subset`` argument defines which region to apply the formatting function
        to. If the ``formatter`` argument is given in dict form but does not include
        all columns within the subset then these columns will have the default formatter
        applied. Any columns in the formatter dict excluded from the subset will
        be ignored.

        When using a ``formatter`` string the dtypes must be compatible, otherwise a
        `ValueError` will be raised.

        Examples
        --------
        Using ``na_rep`` and ``precision`` with the default ``formatter``

        >>> df = pd.DataFrame([[np.nan, 1.0, 'A'], [2.0, np.nan, 3.0]])
        >>> df.style.format(na_rep='MISS', precision=3)
                0       1       2
        0    MISS   1.000       A
        1   2.000    MISS   3.000

        Using a ``formatter`` specification on consistent column dtypes

        >>> df.style.format('{:.2f}', na_rep='MISS', subset=[0,1])
                0      1          2
        0    MISS   1.00          A
        1    2.00   MISS   3.000000

        Using the default ``formatter`` for unspecified columns

        >>> df.style.format({0: '{:.2f}', 1: '£ {:.1f}'}, na_rep='MISS', precision=1)
                 0      1     2
        0    MISS   £ 1.0     A
        1    2.00    MISS   3.0

        Multiple ``na_rep`` or ``precision`` specifications under the default
        ``formatter``.

        >>> df.style.format(na_rep='MISS', precision=1, subset=[0])
        ...     .format(na_rep='PASS', precision=2, subset=[1, 2])
                0      1      2
        0    MISS   1.00      A
        1     2.0   PASS   3.00

        Using a callable ``formatter`` function.

        >>> func = lambda s: 'STRING' if isinstance(s, str) else 'FLOAT'
        >>> df.style.format({0: '{:.1f}', 2: func}, precision=4, na_rep='MISS')
                0        1        2
        0    MISS   1.0000   STRING
        1     2.0     MISS    FLOAT

        Using a ``formatter`` with HTML ``escape`` and ``na_rep``.

        >>> df = pd.DataFrame([['<div></div>', '"A&B"', None]])
        >>> s = df.style.format(
        ...     '<a href="a.com/{0}">{0}</a>', escape="html", na_rep="NA"
        ...     )
        >>> s.render()
        ...
        <td .. ><a href="a.com/&lt;div&gt;&lt;/div&gt;">&lt;div&gt;&lt;/div&gt;</a></td>
        <td .. ><a href="a.com/&#34;A&amp;B&#34;">&#34;A&amp;B&#34;</a></td>
        <td .. >NA</td>
        ...

        Using a ``formatter`` with LaTeX ``escape``.

        >>> df = pd.DataFrame([["123"], ["~ ^"], ["$%#"]])
        >>> s = df.style.format("\\textbf{{{}}}", escape="latex").to_latex()
        \begin{tabular}{ll}
        {} & {0} \\
        0 & \textbf{123} \\
        1 & \textbf{\textasciitilde \space \textasciicircum } \\
        2 & \textbf{\$\%\#} \\
        \end{tabular}
        """
        if all(
            (
                formatter is None,
                subset is None,
                precision is None,
                decimal == ".",
                thousands is None,
                na_rep is None,
                escape is None,
            )
        ):
            self._display_funcs.clear()
            return self  # clear the formatter / revert to default and avoid looping

        subset = slice(None) if subset is None else subset
        subset = non_reducing_slice(subset)
        data = self.data.loc[subset]

        if not isinstance(formatter, dict):
            formatter = {col: formatter for col in data.columns}

        cis = self.columns.get_indexer_for(data.columns)
        ris = self.index.get_indexer_for(data.index)
        for ci in cis:
            format_func = _maybe_wrap_formatter(
                formatter.get(self.columns[ci]),
                na_rep=na_rep,
                precision=precision,
                decimal=decimal,
                thousands=thousands,
                escape=escape,
            )
            for ri in ris:
                self._display_funcs[(ri, ci)] = format_func

        return self


def _element(
    html_element: str,
    html_class: str,
    value: Any,
    is_visible: bool,
    **kwargs,
) -> dict:
    """
    Template to return container with information for a <td></td> or <th></th> element.
    """
    if "display_value" not in kwargs:
        kwargs["display_value"] = value
    return {
        "type": html_element,
        "value": value,
        "class": html_class,
        "is_visible": is_visible,
        **kwargs,
    }


def _get_trimming_maximums(rn, cn, max_elements, scaling_factor=0.8):
    """
    Recursively reduce the number of rows and columns to satisfy max elements.

    Parameters
    ----------
    rn, cn : int
        The number of input rows / columns
    max_elements : int
        The number of allowable elements

    Returns
    -------
    rn, cn : tuple
        New rn and cn values that satisfy the max_elements constraint
    """

    def scale_down(rn, cn):
        if cn >= rn:
            return rn, int(cn * scaling_factor)
        else:
            return int(rn * scaling_factor), cn

    while rn * cn > max_elements:
        rn, cn = scale_down(rn, cn)

    return rn, cn


def _get_level_lengths(
    index: Index,
    sparsify: bool,
    max_index: int,
    hidden_elements: Sequence[int] | None = None,
):
    """
    Given an index, find the level length for each element.

    Parameters
    ----------
    index : Index
        Index or columns to determine lengths of each element
    sparsify : bool
        Whether to hide or show each distinct element in a MultiIndex
    max_index : int
        The maximum number of elements to analyse along the index due to trimming
    hidden_elements : sequence of int
        Index positions of elements hidden from display in the index affecting
        length

    Returns
    -------
    Dict :
        Result is a dictionary of (level, initial_position): span
    """
    if isinstance(index, MultiIndex):
        levels = index.format(sparsify=lib.no_default, adjoin=False)
    else:
        levels = index.format()

    if hidden_elements is None:
        hidden_elements = []

    lengths = {}
    if index.nlevels == 1:
        for i, value in enumerate(levels):
            if i not in hidden_elements:
                lengths[(0, i)] = 1
        return lengths

    for i, lvl in enumerate(levels):
        for j, row in enumerate(lvl):
            if j >= max_index:
                # stop the loop due to display trimming
                break
            if not sparsify:
                lengths[(i, j)] = 1
            elif (row is not lib.no_default) and (j not in hidden_elements):
                last_label = j
                lengths[(i, last_label)] = 1
            elif row is not lib.no_default:
                # even if its hidden, keep track of it in case
                # length >1 and later elements are visible
                last_label = j
                lengths[(i, last_label)] = 0
            elif j not in hidden_elements:
                if lengths[(i, last_label)] == 0:
                    # if the previous iteration was first-of-kind but hidden then offset
                    last_label = j
                    lengths[(i, last_label)] = 1
                else:
                    # else add to previous iteration
                    lengths[(i, last_label)] += 1

    non_zero_lengths = {
        element: length for element, length in lengths.items() if length >= 1
    }

    return non_zero_lengths


def _is_visible(idx_row, idx_col, lengths) -> bool:
    """
    Index -> {(idx_row, idx_col): bool}).
    """
    return (idx_col, idx_row) in lengths


def _format_table_styles(styles: CSSStyles) -> CSSStyles:
    """
    looks for multiple CSS selectors and separates them:
    [{'selector': 'td, th', 'props': 'a:v;'}]
        ---> [{'selector': 'td', 'props': 'a:v;'},
              {'selector': 'th', 'props': 'a:v;'}]
    """
    return [
        {"selector": selector, "props": css_dict["props"]}
        for css_dict in styles
        for selector in css_dict["selector"].split(",")
    ]


def _default_formatter(x: Any, precision: int, thousands: bool = False) -> Any:
    """
    Format the display of a value

    Parameters
    ----------
    x : Any
        Input variable to be formatted
    precision : Int
        Floating point precision used if ``x`` is float or complex.
    thousands : bool, default False
        Whether to group digits with thousands separated with ",".

    Returns
    -------
    value : Any
        Matches input type, or string if input is float or complex or int with sep.
    """
    if isinstance(x, (float, complex)):
        if thousands:
            return f"{x:,.{precision}f}"
        return f"{x:.{precision}f}"
    elif isinstance(x, int) and thousands:
        return f"{x:,.0f}"
    return x


def _wrap_decimal_thousands(
    formatter: Callable, decimal: str, thousands: str | None
) -> Callable:
    """
    Takes a string formatting function and wraps logic to deal with thousands and
    decimal parameters, in the case that they are non-standard and that the input
    is a (float, complex, int).
    """

    def wrapper(x):
        if isinstance(x, (float, complex, int)):
            if decimal != "." and thousands is not None and thousands != ",":
                return (
                    formatter(x)
                    .replace(",", "§_§-")  # rare string to avoid "," <-> "." clash.
                    .replace(".", decimal)
                    .replace("§_§-", thousands)
                )
            elif decimal != "." and (thousands is None or thousands == ","):
                return formatter(x).replace(".", decimal)
            elif decimal == "." and thousands is not None and thousands != ",":
                return formatter(x).replace(",", thousands)
        return formatter(x)

    return wrapper


def _str_escape(x, escape):
    """if escaping: only use on str, else return input"""
    if isinstance(x, str):
        if escape == "html":
            return escape_html(x)
        elif escape == "latex":
            return _escape_latex(x)
        else:
            raise ValueError(
                f"`escape` only permitted in {{'html', 'latex'}}, got {escape}"
            )
    return x


def _maybe_wrap_formatter(
    formatter: BaseFormatter | None = None,
    na_rep: str | None = None,
    precision: int | None = None,
    decimal: str = ".",
    thousands: str | None = None,
    escape: str | None = None,
) -> Callable:
    """
    Allows formatters to be expressed as str, callable or None, where None returns
    a default formatting function. wraps with na_rep, and precision where they are
    available.
    """
    # Get initial func from input string, input callable, or from default factory
    if isinstance(formatter, str):
        func_0 = lambda x: formatter.format(x)
    elif callable(formatter):
        func_0 = formatter
    elif formatter is None:
        precision = get_option("display.precision") if precision is None else precision
        func_0 = partial(
            _default_formatter, precision=precision, thousands=(thousands is not None)
        )
    else:
        raise TypeError(f"'formatter' expected str or callable, got {type(formatter)}")

    # Replace chars if escaping
    if escape is not None:
        func_1 = lambda x: func_0(_str_escape(x, escape=escape))
    else:
        func_1 = func_0

    # Replace decimals and thousands if non-standard inputs detected
    if decimal != "." or (thousands is not None and thousands != ","):
        func_2 = _wrap_decimal_thousands(func_1, decimal=decimal, thousands=thousands)
    else:
        func_2 = func_1

    # Replace missing values if na_rep
    if na_rep is None:
        return func_2
    else:
        return lambda x: na_rep if isna(x) else func_2(x)


def non_reducing_slice(slice_: Subset):
    """
    Ensure that a slice doesn't reduce to a Series or Scalar.

    Any user-passed `subset` should have this called on it
    to make sure we're always working with DataFrames.
    """
    # default to column slice, like DataFrame
    # ['A', 'B'] -> IndexSlices[:, ['A', 'B']]
    kinds = (ABCSeries, np.ndarray, Index, list, str)
    if isinstance(slice_, kinds):
        slice_ = IndexSlice[:, slice_]

    def pred(part) -> bool:
        """
        Returns
        -------
        bool
            True if slice does *not* reduce,
            False if `part` is a tuple.
        """
        # true when slice does *not* reduce, False when part is a tuple,
        # i.e. MultiIndex slice
        if isinstance(part, tuple):
            # GH#39421 check for sub-slice:
            return any((isinstance(s, slice) or is_list_like(s)) for s in part)
        else:
            return isinstance(part, slice) or is_list_like(part)

    if not is_list_like(slice_):
        if not isinstance(slice_, slice):
            # a 1-d slice, like df.loc[1]
            slice_ = [[slice_]]
        else:
            # slice(a, b, c)
            slice_ = [slice_]  # to tuplize later
    else:
        # error: Item "slice" of "Union[slice, Sequence[Any]]" has no attribute
        # "__iter__" (not iterable) -> is specifically list_like in conditional
        slice_ = [p if pred(p) else [p] for p in slice_]  # type: ignore[union-attr]
    return tuple(slice_)


def maybe_convert_css_to_tuples(style: CSSProperties) -> CSSList:
    """
    Convert css-string to sequence of tuples format if needed.
    'color:red; border:1px solid black;' -> [('color', 'red'),
                                             ('border','1px solid red')]
    """
    if isinstance(style, str):
        s = style.split(";")
        try:
            return [
                (x.split(":")[0].strip(), x.split(":")[1].strip())
                for x in s
                if x.strip() != ""
            ]
        except IndexError:
            raise ValueError(
                "Styles supplied as string must follow CSS rule formats, "
                f"for example 'attr: val;'. '{style}' was given."
            )
    return style


class Tooltips:
    """
    An extension to ``Styler`` that allows for and manipulates tooltips on hover
    of ``<td>`` cells in the HTML result.

    Parameters
    ----------
    css_name: str, default "pd-t"
        Name of the CSS class that controls visualisation of tooltips.
    css_props: list-like, default; see Notes
        List of (attr, value) tuples defining properties of the CSS class.
    tooltips: DataFrame, default empty
        DataFrame of strings aligned with underlying Styler data for tooltip
        display.

    Notes
    -----
    The default properties for the tooltip CSS class are:

        - visibility: hidden
        - position: absolute
        - z-index: 1
        - background-color: black
        - color: white
        - transform: translate(-20px, -20px)

    Hidden visibility is a key prerequisite to the hover functionality, and should
    always be included in any manual properties specification.
    """

    def __init__(
        self,
        css_props: CSSProperties = [
            ("visibility", "hidden"),
            ("position", "absolute"),
            ("z-index", 1),
            ("background-color", "black"),
            ("color", "white"),
            ("transform", "translate(-20px, -20px)"),
        ],
        css_name: str = "pd-t",
        tooltips: DataFrame = DataFrame(),
    ):
        self.class_name = css_name
        self.class_properties = css_props
        self.tt_data = tooltips
        self.table_styles: CSSStyles = []

    @property
    def _class_styles(self):
        """
        Combine the ``_Tooltips`` CSS class name and CSS properties to the format
        required to extend the underlying ``Styler`` `table_styles` to allow
        tooltips to render in HTML.

        Returns
        -------
        styles : List
        """
        return [
            {
                "selector": f".{self.class_name}",
                "props": maybe_convert_css_to_tuples(self.class_properties),
            }
        ]

    def _pseudo_css(self, uuid: str, name: str, row: int, col: int, text: str):
        """
        For every table data-cell that has a valid tooltip (not None, NaN or
        empty string) must create two pseudo CSS entries for the specific
        <td> element id which are added to overall table styles:
        an on hover visibility change and a content change
        dependent upon the user's chosen display string.

        For example:
            [{"selector": "T__row1_col1:hover .pd-t",
             "props": [("visibility", "visible")]},
            {"selector": "T__row1_col1 .pd-t::after",
             "props": [("content", "Some Valid Text String")]}]

        Parameters
        ----------
        uuid: str
            The uuid of the Styler instance
        name: str
            The css-name of the class used for styling tooltips
        row : int
            The row index of the specified tooltip string data
        col : int
            The col index of the specified tooltip string data
        text : str
            The textual content of the tooltip to be displayed in HTML.

        Returns
        -------
        pseudo_css : List
        """
        selector_id = "#T_" + uuid + "row" + str(row) + "_col" + str(col)
        return [
            {
                "selector": selector_id + f":hover .{name}",
                "props": [("visibility", "visible")],
            },
            {
                "selector": selector_id + f" .{name}::after",
                "props": [("content", f'"{text}"')],
            },
        ]

    def _translate(self, styler_data: FrameOrSeriesUnion, uuid: str, d: dict):
        """
        Mutate the render dictionary to allow for tooltips:

        - Add ``<span>`` HTML element to each data cells ``display_value``. Ignores
          headers.
        - Add table level CSS styles to control pseudo classes.

        Parameters
        ----------
        styler_data : DataFrame
            Underlying ``Styler`` DataFrame used for reindexing.
        uuid : str
            The underlying ``Styler`` uuid for CSS id.
        d : dict
            The dictionary prior to final render

        Returns
        -------
        render_dict : Dict
        """
        self.tt_data = self.tt_data.reindex_like(styler_data)

        if self.tt_data.empty:
            return d

        name = self.class_name

        mask = (self.tt_data.isna()) | (self.tt_data.eq(""))  # empty string = no ttip
        self.table_styles = [
            style
            for sublist in [
                self._pseudo_css(uuid, name, i, j, str(self.tt_data.iloc[i, j]))
                for i in range(len(self.tt_data.index))
                for j in range(len(self.tt_data.columns))
                if not mask.iloc[i, j]
            ]
            for style in sublist
        ]

        if self.table_styles:
            # add span class to every cell only if at least 1 non-empty tooltip
            for row in d["body"]:
                for item in row:
                    if item["type"] == "td":
                        item["display_value"] = (
                            str(item["display_value"])
                            + f'<span class="{self.class_name}"></span>'
                        )
            d["table_styles"].extend(self._class_styles)
            d["table_styles"].extend(self.table_styles)

        return d


def _parse_latex_table_wrapping(table_styles: CSSStyles, caption: str | None) -> bool:
    """
    Indicate whether LaTeX {tabular} should be wrapped with a {table} environment.

    Parses the `table_styles` and detects any selectors which must be included outside
    of {tabular}, i.e. indicating that wrapping must occur, and therefore return True,
    or if a caption exists and requires similar.
    """
    IGNORED_WRAPPERS = ["toprule", "midrule", "bottomrule", "column_format"]
    # ignored selectors are included with {tabular} so do not need wrapping
    return (
        table_styles is not None
        and any(d["selector"] not in IGNORED_WRAPPERS for d in table_styles)
    ) or caption is not None


def _parse_latex_table_styles(table_styles: CSSStyles, selector: str) -> str | None:
    """
    Return the first 'props' 'value' from ``tables_styles`` identified by ``selector``.

    Examples
    --------
    >>> table_styles = [{'selector': 'foo', 'props': [('attr','value')],
    ...                 {'selector': 'bar', 'props': [('attr', 'overwritten')]},
    ...                 {'selector': 'bar', 'props': [('a1', 'baz'), ('a2', 'ignore')]}]
    >>> _parse_latex_table_styles(table_styles, selector='bar')
    'baz'

    Notes
    -----
    The replacement of "§" with ":" is to avoid the CSS problem where ":" has structural
    significance and cannot be used in LaTeX labels, but is often required by them.
    """
    for style in table_styles[::-1]:  # in reverse for most recently applied style
        if style["selector"] == selector:
            return str(style["props"][0][1]).replace("§", ":")
    return None


def _parse_latex_cell_styles(
    latex_styles: CSSList, display_value: str, convert_css: bool = False
) -> str:
    r"""
    Mutate the ``display_value`` string including LaTeX commands from ``latex_styles``.

    This method builds a recursive latex chain of commands based on the
    CSSList input, nested around ``display_value``.

    If a CSS style is given as ('<command>', '<options>') this is translated to
    '\<command><options>{display_value}', and this value is treated as the
    display value for the next iteration.

    The most recent style forms the inner component, for example for styles:
    `[('c1', 'o1'), ('c2', 'o2')]` this returns: `\c1o1{\c2o2{display_value}}`

    Sometimes latex commands have to be wrapped with curly braces in different ways:
    We create some parsing flags to identify the different behaviours:

     - `--rwrap`        : `\<command><options>{<display_value>}`
     - `--wrap`         : `{\<command><options> <display_value>}`
     - `--nowrap`       : `\<command><options> <display_value>`
     - `--lwrap`        : `{\<command><options>} <display_value>`
     - `--dwrap`        : `{\<command><options>}{<display_value>}`

    For example for styles:
    `[('c1', 'o1--wrap'), ('c2', 'o2')]` this returns: `{\c1o1 \c2o2{display_value}}
    """
    if convert_css:
        latex_styles = _parse_latex_css_conversion(latex_styles)
    for (command, options) in latex_styles[::-1]:  # in reverse for most recent style
        formatter = {
            "--wrap": f"{{\\{command}--to_parse {display_value}}}",
            "--nowrap": f"\\{command}--to_parse {display_value}",
            "--lwrap": f"{{\\{command}--to_parse}} {display_value}",
            "--rwrap": f"\\{command}--to_parse{{{display_value}}}",
            "--dwrap": f"{{\\{command}--to_parse}}{{{display_value}}}",
        }
        display_value = f"\\{command}{options} {display_value}"
        for arg in ["--nowrap", "--wrap", "--lwrap", "--rwrap", "--dwrap"]:
            if arg in str(options):
                display_value = formatter[arg].replace(
                    "--to_parse", _parse_latex_options_strip(value=options, arg=arg)
                )
                break  # only ever one purposeful entry
    return display_value


def _parse_latex_header_span(
    cell: dict[str, Any], multirow_align: str, multicol_align: str, wrap: bool = False
) -> str:
    r"""
    Refactor the cell `display_value` if a 'colspan' or 'rowspan' attribute is present.

    'rowspan' and 'colspan' do not occur simultaneouly. If they are detected then
    the `display_value` is altered to a LaTeX `multirow` or `multicol` command
    respectively, with the appropriate cell-span.

    ``wrap`` is used to enclose the `display_value` in braces which is needed for
    column headers using an siunitx package.

    Requires the package {multirow}, whereas multicol support is usually built in
    to the {tabular} environment.

    Examples
    --------
    >>> cell = {'display_vale':'text', 'attributes': 'colspan="3"'}
    >>> _parse_latex_header_span(cell, 't', 'c')
    '\multicol{3}{c}{text}'
    """
    if "attributes" in cell:
        attrs = cell["attributes"]
        if 'colspan="' in attrs:
            colspan = attrs[attrs.find('colspan="') + 9 :]  # len('colspan="') = 9
            colspan = int(colspan[: colspan.find('"')])
            return (
                f"\\multicolumn{{{colspan}}}{{{multicol_align}}}"
                f"{{{cell['display_value']}}}"
            )
        elif 'rowspan="' in attrs:
            rowspan = attrs[attrs.find('rowspan="') + 9 :]
            rowspan = int(rowspan[: rowspan.find('"')])
            return (
                f"\\multirow[{multirow_align}]{{{rowspan}}}{{*}}"
                f"{{{cell['display_value']}}}"
            )
    if wrap:
        return f"{{{cell['display_value']}}}"
    else:
        return cell["display_value"]


def _parse_latex_options_strip(value: str | int | float, arg: str) -> str:
    """
    Strip a css_value which may have latex wrapping arguments, css comment identifiers,
    and whitespaces, to a valid string for latex options parsing.

    For example: 'red /* --wrap */  ' --> 'red'
    """
    return str(value).replace(arg, "").replace("/*", "").replace("*/", "").strip()


def _parse_latex_css_conversion(styles: CSSList) -> CSSList:
    """
    Convert CSS (attribute,value) pairs to equivalent LaTeX (command,options) pairs.

    Ignore conversion if tagged with `--latex` option, skipped if no conversion found.
    """

    def font_weight(value, arg):
        if value == "bold" or value == "bolder":
            return "bfseries", f"{arg}"
        return None

    def font_style(value, arg):
        if value == "italic":
            return "itshape", f"{arg}"
        elif value == "oblique":
            return "slshape", f"{arg}"
        return None

    def color(value, user_arg, command, comm_arg):
        """
        CSS colors have 5 formats to process:

         - 6 digit hex code: "#ff23ee"     --> [HTML]{FF23EE}
         - 3 digit hex code: "#f0e"        --> [HTML]{FF00EE}
         - rgba: rgba(128, 255, 0, 0.5)    --> [rgb]{0.502, 1.000, 0.000}
         - rgb: rgb(128, 255, 0,)          --> [rbg]{0.502, 1.000, 0.000}
         - string: red                     --> {red}

        Additionally rgb or rgba can be expressed in % which is also parsed.
        """
        arg = user_arg if user_arg != "" else comm_arg

        if value[0] == "#" and len(value) == 7:  # color is hex code
            return command, f"[HTML]{{{value[1:].upper()}}}{arg}"
        if value[0] == "#" and len(value) == 4:  # color is short hex code
            val = f"{value[1].upper()*2}{value[2].upper()*2}{value[3].upper()*2}"
            return command, f"[HTML]{{{val}}}{arg}"
        elif value[:3] == "rgb":  # color is rgb or rgba
            r = re.findall("(?<=\\()[0-9\\s%]+(?=,)", value)[0].strip()
            r = float(r[:-1]) / 100 if "%" in r else int(r) / 255
            g = re.findall("(?<=,)[0-9\\s%]+(?=,)", value)[0].strip()
            g = float(g[:-1]) / 100 if "%" in g else int(g) / 255
            if value[3] == "a":  # color is rgba
                b = re.findall("(?<=,)[0-9\\s%]+(?=,)", value)[1].strip()
            else:  # color is rgb
                b = re.findall("(?<=,)[0-9\\s%]+(?=\\))", value)[0].strip()
            b = float(b[:-1]) / 100 if "%" in b else int(b) / 255
            return command, f"[rgb]{{{r:.3f}, {g:.3f}, {b:.3f}}}{arg}"
        else:
            return command, f"{{{value}}}{arg}"  # color is likely string-named

    CONVERTED_ATTRIBUTES: dict[str, Callable] = {
        "font-weight": font_weight,
        "background-color": partial(color, command="cellcolor", comm_arg="--lwrap"),
        "color": partial(color, command="color", comm_arg=""),
        "font-style": font_style,
    }

    latex_styles: CSSList = []
    for (attribute, value) in styles:
        if isinstance(value, str) and "--latex" in value:
            # return the style without conversion but drop '--latex'
            latex_styles.append((attribute, value.replace("--latex", "")))
        if attribute in CONVERTED_ATTRIBUTES.keys():
            arg = ""
            for x in ["--wrap", "--nowrap", "--lwrap", "--dwrap", "--rwrap"]:
                if x in str(value):
                    arg, value = x, _parse_latex_options_strip(value, x)
                    break
            latex_style = CONVERTED_ATTRIBUTES[attribute](value, arg)
            if latex_style is not None:
                latex_styles.extend([latex_style])
    return latex_styles


def _escape_latex(s):
    r"""
    Replace the characters ``&``, ``%``, ``$``, ``#``, ``_``, ``{``, ``}``,
    ``~``, ``^``, and ``\`` in the string with LaTeX-safe sequences.

    Use this if you need to display text that might contain such characters in LaTeX.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    return (
        s.replace("\\", "ab2§=§8yz")  # rare string for final conversion: avoid \\ clash
        .replace("ab2§=§8yz ", "ab2§=§8yz\\space ")  # since \backslash gobbles spaces
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~ ", "~\\space ")  # since \textasciitilde gobbles spaces
        .replace("~", "\\textasciitilde ")
        .replace("^ ", "^\\space ")  # since \textasciicircum gobbles spaces
        .replace("^", "\\textasciicircum ")
        .replace("ab2§=§8yz", "\\textbackslash ")
    )
