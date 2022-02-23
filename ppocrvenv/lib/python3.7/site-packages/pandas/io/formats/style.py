"""
Module for applying conditional formatting to DataFrames and Series.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from functools import partial
import operator
from typing import (
    Any,
    Callable,
    Hashable,
    Sequence,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._typing import (
    Axis,
    FilePathOrBuffer,
    FrameOrSeries,
    FrameOrSeriesUnion,
    IndexLabel,
    Scalar,
)
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc

import pandas as pd
from pandas import (
    IndexSlice,
    RangeIndex,
)
from pandas.api.types import is_list_like
from pandas.core import generic
import pandas.core.common as com
from pandas.core.frame import (
    DataFrame,
    Series,
)
from pandas.core.generic import NDFrame

from pandas.io.formats.format import save_to_buffer

jinja2 = import_optional_dependency("jinja2", extra="DataFrame.style requires jinja2.")

from pandas.io.formats.style_render import (
    CSSProperties,
    CSSStyles,
    StylerRenderer,
    Subset,
    Tooltips,
    maybe_convert_css_to_tuples,
    non_reducing_slice,
)

try:
    from matplotlib import colors
    import matplotlib.pyplot as plt

    has_mpl = True
except ImportError:
    has_mpl = False
    no_mpl_message = "{0} requires matplotlib."


@contextmanager
def _mpl(func: Callable):
    if has_mpl:
        yield plt, colors
    else:
        raise ImportError(no_mpl_message.format(func.__name__))


class Styler(StylerRenderer):
    r"""
    Helps style a DataFrame or Series according to the data with HTML and CSS.

    Parameters
    ----------
    data : Series or DataFrame
        Data to be styled - either a Series or DataFrame.
    precision : int
        Precision to round floats to, defaults to pd.options.display.precision.
    table_styles : list-like, default None
        List of {selector: (attr, value)} dicts; see Notes.
    uuid : str, default None
        A unique identifier to avoid CSS collisions; generated automatically.
    caption : str, tuple, default None
        String caption to attach to the table. Tuple only used for LaTeX dual captions.
    table_attributes : str, default None
        Items that show up in the opening ``<table>`` tag
        in addition to automatic (by default) id.
    cell_ids : bool, default True
        If True, each cell will have an ``id`` attribute in their HTML tag.
        The ``id`` takes the form ``T_<uuid>_row<num_row>_col<num_col>``
        where ``<uuid>`` is the unique identifier, ``<num_row>`` is the row
        number and ``<num_col>`` is the column number.
    na_rep : str, optional
        Representation for missing values.
        If ``na_rep`` is None, no special formatting is applied.

        .. versionadded:: 1.0.0

    uuid_len : int, default 5
        If ``uuid`` is not specified, the length of the ``uuid`` to randomly generate
        expressed in hex characters, in range [0, 32].

        .. versionadded:: 1.2.0

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

        .. versionadded:: 1.3.0

    Attributes
    ----------
    env : Jinja2 jinja2.Environment
    template : Jinja2 Template
    loader : Jinja2 Loader

    See Also
    --------
    DataFrame.style : Return a Styler object containing methods for building
        a styled HTML representation for the DataFrame.

    Notes
    -----
    Most styling will be done by passing style functions into
    ``Styler.apply`` or ``Styler.applymap``. Style functions should
    return values with strings containing CSS ``'attr: value'`` that will
    be applied to the indicated cells.

    If using in the Jupyter notebook, Styler has defined a ``_repr_html_``
    to automatically render itself. Otherwise call Styler.render to get
    the generated HTML.

    CSS classes are attached to the generated HTML

    * Index and Column names include ``index_name`` and ``level<k>``
      where `k` is its level in a MultiIndex
    * Index label cells include

      * ``row_heading``
      * ``row<n>`` where `n` is the numeric position of the row
      * ``level<k>`` where `k` is the level in a MultiIndex

    * Column label cells include
      * ``col_heading``
      * ``col<n>`` where `n` is the numeric position of the column
      * ``level<k>`` where `k` is the level in a MultiIndex

    * Blank cells include ``blank``
    * Data cells include ``data``
    """

    def __init__(
        self,
        data: FrameOrSeriesUnion,
        precision: int | None = None,
        table_styles: CSSStyles | None = None,
        uuid: str | None = None,
        caption: str | tuple | None = None,
        table_attributes: str | None = None,
        cell_ids: bool = True,
        na_rep: str | None = None,
        uuid_len: int = 5,
        decimal: str = ".",
        thousands: str | None = None,
        escape: str | None = None,
    ):
        super().__init__(
            data=data,
            uuid=uuid,
            uuid_len=uuid_len,
            table_styles=table_styles,
            table_attributes=table_attributes,
            caption=caption,
            cell_ids=cell_ids,
        )

        # validate ordered args
        self.precision = precision  # can be removed on set_precision depr cycle
        self.na_rep = na_rep  # can be removed on set_na_rep depr cycle
        self.format(
            formatter=None,
            precision=precision,
            na_rep=na_rep,
            escape=escape,
            decimal=decimal,
            thousands=thousands,
        )

    def _repr_html_(self) -> str:
        """
        Hooks into Jupyter notebook rich display system.
        """
        return self.render()

    def render(
        self,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        **kwargs,
    ) -> str:
        """
        Render the ``Styler`` including all applied styles to HTML.

        Parameters
        ----------
        sparse_index : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.index`` value.
        sparse_columns : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.columns`` value.
        **kwargs
            Any additional keyword arguments are passed
            through to ``self.template.render``.
            This is useful when you need to provide
            additional variables for a custom template.

        Returns
        -------
        rendered : str
            The rendered HTML.

        Notes
        -----
        Styler objects have defined the ``_repr_html_`` method
        which automatically calls ``self.render()`` when it's the
        last item in a Notebook cell. When calling ``Styler.render()``
        directly, wrap the result in ``IPython.display.HTML`` to view
        the rendered HTML in the notebook.

        Pandas uses the following keys in render. Arguments passed
        in ``**kwargs`` take precedence, so think carefully if you want
        to override them:

        * head
        * cellstyle
        * body
        * uuid
        * table_styles
        * caption
        * table_attributes
        """
        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")
        return self._render_html(sparse_index, sparse_columns, **kwargs)

    def set_tooltips(
        self,
        ttips: DataFrame,
        props: CSSProperties | None = None,
        css_class: str | None = None,
    ) -> Styler:
        """
        Set the DataFrame of strings on ``Styler`` generating ``:hover`` tooltips.

        These string based tooltips are only applicable to ``<td>`` HTML elements,
        and cannot be used for column or index headers.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        ttips : DataFrame
            DataFrame containing strings that will be translated to tooltips, mapped
            by identical column and index values that must exist on the underlying
            Styler data. None, NaN values, and empty strings will be ignored and
            not affect the rendered HTML.
        props : list-like or str, optional
            List of (attr, value) tuples or a valid CSS string. If ``None`` adopts
            the internal default values described in notes.
        css_class : str, optional
            Name of the tooltip class used in CSS, should conform to HTML standards.
            Only useful if integrating tooltips with external CSS. If ``None`` uses the
            internal default value 'pd-t'.

        Returns
        -------
        self : Styler

        Notes
        -----
        Tooltips are created by adding `<span class="pd-t"></span>` to each data cell
        and then manipulating the table level CSS to attach pseudo hover and pseudo
        after selectors to produce the required the results.

        The default properties for the tooltip CSS class are:

        - visibility: hidden
        - position: absolute
        - z-index: 1
        - background-color: black
        - color: white
        - transform: translate(-20px, -20px)

        The property 'visibility: hidden;' is a key prerequisite to the hover
        functionality, and should always be included in any manual properties
        specification, using the ``props`` argument.

        Tooltips are not designed to be efficient, and can add large amounts of
        additional HTML for larger tables, since they also require that ``cell_ids``
        is forced to `True`.

        Examples
        --------
        Basic application

        >>> df = pd.DataFrame(data=[[0, 1], [2, 3]])
        >>> ttips = pd.DataFrame(
        ...    data=[["Min", ""], [np.nan, "Max"]], columns=df.columns, index=df.index
        ... )
        >>> s = df.style.set_tooltips(ttips).render()

        Optionally controlling the tooltip visual display

        >>> df.style.set_tooltips(ttips, css_class='tt-add', props=[
        ...     ('visibility', 'hidden'),
        ...     ('position', 'absolute'),
        ...     ('z-index', 1)])
        >>> df.style.set_tooltips(ttips, css_class='tt-add',
        ...     props='visibility:hidden; position:absolute; z-index:1;')
        """
        if not self.cell_ids:
            # tooltips not optimised for individual cell check. requires reasonable
            # redesign and more extensive code for a feature that might be rarely used.
            raise NotImplementedError(
                "Tooltips can only render with 'cell_ids' is True."
            )
        if not ttips.index.is_unique or not ttips.columns.is_unique:
            raise KeyError(
                "Tooltips render only if `ttips` has unique index and columns."
            )
        if self.tooltips is None:  # create a default instance if necessary
            self.tooltips = Tooltips()
        self.tooltips.tt_data = ttips
        if props:
            self.tooltips.class_properties = props
        if css_class:
            self.tooltips.class_name = css_class

        return self

    @doc(
        NDFrame.to_excel,
        klass="Styler",
        storage_options=generic._shared_docs["storage_options"],
    )
    def to_excel(
        self,
        excel_writer,
        sheet_name: str = "Sheet1",
        na_rep: str = "",
        float_format: str | None = None,
        columns: Sequence[Hashable] | None = None,
        header: Sequence[Hashable] | bool = True,
        index: bool = True,
        index_label: IndexLabel | None = None,
        startrow: int = 0,
        startcol: int = 0,
        engine: str | None = None,
        merge_cells: bool = True,
        encoding: str | None = None,
        inf_rep: str = "inf",
        verbose: bool = True,
        freeze_panes: tuple[int, int] | None = None,
    ) -> None:

        from pandas.io.formats.excel import ExcelFormatter

        formatter = ExcelFormatter(
            self,
            na_rep=na_rep,
            cols=columns,
            header=header,
            float_format=float_format,
            index=index,
            index_label=index_label,
            merge_cells=merge_cells,
            inf_rep=inf_rep,
        )
        formatter.write(
            excel_writer,
            sheet_name=sheet_name,
            startrow=startrow,
            startcol=startcol,
            freeze_panes=freeze_panes,
            engine=engine,
        )

    def to_latex(
        self,
        buf: FilePathOrBuffer[str] | None = None,
        *,
        column_format: str | None = None,
        position: str | None = None,
        position_float: str | None = None,
        hrules: bool = False,
        label: str | None = None,
        caption: str | tuple | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        multirow_align: str = "c",
        multicol_align: str = "r",
        siunitx: bool = False,
        encoding: str | None = None,
        convert_css: bool = False,
    ):
        r"""
        Write Styler to a file, buffer or string in LaTeX format.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        buf : str, Path, or StringIO-like, optional, default None
            Buffer to write to. If ``None``, the output is returned as a string.
        column_format : str, optional
            The LaTeX column specification placed in location:

            \\begin{tabular}{<column_format>}

            Defaults to 'l' for index and
            non-numeric data columns, and, for numeric data columns,
            to 'r' by default, or 'S' if ``siunitx`` is ``True``.
        position : str, optional
            The LaTeX positional argument (e.g. 'h!') for tables, placed in location:

            \\begin{table}[<position>]
        position_float : {"centering", "raggedleft", "raggedright"}, optional
            The LaTeX float command placed in location:

            \\begin{table}[<position>]

            \\<position_float>
        hrules : bool, default False
            Set to `True` to add \\toprule, \\midrule and \\bottomrule from the
            {booktabs} LaTeX package.
        label : str, optional
            The LaTeX label included as: \\label{<label>}.
            This is used with \\ref{<label>} in the main .tex file.
        caption : str, tuple, optional
            If string, the LaTeX table caption included as: \\caption{<caption>}.
            If tuple, i.e ("full caption", "short caption"), the caption included
            as: \\caption[<caption[1]>]{<caption[0]>}.
        sparse_index : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.index`` value.
        sparse_columns : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.columns`` value.
        multirow_align : {"c", "t", "b"}
            If sparsifying hierarchical MultiIndexes whether to align text centrally,
            at the top or bottom.
        multicol_align : {"r", "c", "l"}
            If sparsifying hierarchical MultiIndex columns whether to align text at
            the left, centrally, or at the right.
        siunitx : bool, default False
            Set to ``True`` to structure LaTeX compatible with the {siunitx} package.
        encoding : str, default "utf-8"
            Character encoding setting.
        convert_css : bool, default False
            Convert simple cell-styles from CSS to LaTeX format. Any CSS not found in
            conversion table is dropped. A style can be forced by adding option
            `--latex`. See notes.

        Returns
        -------
        str or None
            If `buf` is None, returns the result as a string. Otherwise returns `None`.

        See Also
        --------
        Styler.format: Format the text display value of cells.

        Notes
        -----
        **Latex Packages**

        For the following features we recommend the following LaTeX inclusions:

        ===================== ==========================================================
        Feature               Inclusion
        ===================== ==========================================================
        sparse columns        none: included within default {tabular} environment
        sparse rows           \\usepackage{multirow}
        hrules                \\usepackage{booktabs}
        colors                \\usepackage[table]{xcolor}
        siunitx               \\usepackage{siunitx}
        bold (with siunitx)   | \\usepackage{etoolbox}
                              | \\robustify\\bfseries
                              | \\sisetup{detect-all = true}  *(within {document})*
        italic (with siunitx) | \\usepackage{etoolbox}
                              | \\robustify\\itshape
                              | \\sisetup{detect-all = true}  *(within {document})*
        ===================== ==========================================================

        **Cell Styles**

        LaTeX styling can only be rendered if the accompanying styling functions have
        been constructed with appropriate LaTeX commands. All styling
        functionality is built around the concept of a CSS ``(<attribute>, <value>)``
        pair (see `Table Visualization <../../user_guide/style.ipynb>`_), and this
        should be replaced by a LaTeX
        ``(<command>, <options>)`` approach. Each cell will be styled individually
        using nested LaTeX commands with their accompanied options.

        For example the following code will highlight and bold a cell in HTML-CSS:

        >>> df = pd.DataFrame([[1,2], [3,4]])
        >>> s = df.style.highlight_max(axis=None,
        ...                            props='background-color:red; font-weight:bold;')
        >>> s.render()

        The equivalent using LaTeX only commands is the following:

        >>> s = df.style.highlight_max(axis=None,
        ...                            props='cellcolor:{red}; bfseries: ;')
        >>> s.to_latex()

        Internally these structured LaTeX ``(<command>, <options>)`` pairs
        are translated to the
        ``display_value`` with the default structure:
        ``\<command><options> <display_value>``.
        Where there are multiple commands the latter is nested recursively, so that
        the above example highlighed cell is rendered as
        ``\cellcolor{red} \bfseries 4``.

        Occasionally this format does not suit the applied command, or
        combination of LaTeX packages that is in use, so additional flags can be
        added to the ``<options>``, within the tuple, to result in different
        positions of required braces (the **default** being the same as ``--nowrap``):

        =================================== ============================================
        Tuple Format                           Output Structure
        =================================== ============================================
        (<command>,<options>)               \\<command><options> <display_value>
        (<command>,<options> ``--nowrap``)  \\<command><options> <display_value>
        (<command>,<options> ``--rwrap``)   \\<command><options>{<display_value>}
        (<command>,<options> ``--wrap``)    {\\<command><options> <display_value>}
        (<command>,<options> ``--lwrap``)   {\\<command><options>} <display_value>
        (<command>,<options> ``--dwrap``)   {\\<command><options>}{<display_value>}
        =================================== ============================================

        For example the `textbf` command for font-weight
        should always be used with `--rwrap` so ``('textbf', '--rwrap')`` will render a
        working cell, wrapped with braces, as ``\textbf{<display_value>}``.

        A more comprehensive example is as follows:

        >>> df = pd.DataFrame([[1, 2.2, "dogs"], [3, 4.4, "cats"], [2, 6.6, "cows"]],
        ...                   index=["ix1", "ix2", "ix3"],
        ...                   columns=["Integers", "Floats", "Strings"])
        >>> s = df.style.highlight_max(
        ...     props='cellcolor:[HTML]{FFFF00}; color:{red};'
        ...           'textit:--rwrap; textbf:--rwrap;'
        ... )
        >>> s.to_latex()

        .. figure:: ../../_static/style/latex_1.png

        **Table Styles**

        Internally Styler uses its ``table_styles`` object to parse the
        ``column_format``, ``position``, ``position_float``, and ``label``
        input arguments. These arguments are added to table styles in the format:

        .. code-block:: python

            set_table_styles([
                {"selector": "column_format", "props": f":{column_format};"},
                {"selector": "position", "props": f":{position};"},
                {"selector": "position_float", "props": f":{position_float};"},
                {"selector": "label", "props": f":{{{label.replace(':','§')}}};"}
            ], overwrite=False)

        Exception is made for the ``hrules`` argument which, in fact, controls all three
        commands: ``toprule``, ``bottomrule`` and ``midrule`` simultaneously. Instead of
        setting ``hrules`` to ``True``, it is also possible to set each
        individual rule definition, by manually setting the ``table_styles``,
        for example below we set a regular ``toprule``, set an ``hline`` for
        ``bottomrule`` and exclude the ``midrule``:

        .. code-block:: python

            set_table_styles([
                {'selector': 'toprule', 'props': ':toprule;'},
                {'selector': 'bottomrule', 'props': ':hline;'},
            ], overwrite=False)

        If other ``commands`` are added to table styles they will be detected, and
        positioned immediately above the '\\begin{tabular}' command. For example to
        add odd and even row coloring, from the {colortbl} package, in format
        ``\rowcolors{1}{pink}{red}``, use:

        .. code-block:: python

            set_table_styles([
                {'selector': 'rowcolors', 'props': ':{1}{pink}{red};'}
            ], overwrite=False)

        A more comprehensive example using these arguments is as follows:

        >>> df.columns = pd.MultiIndex.from_tuples([
        ...     ("Numeric", "Integers"),
        ...     ("Numeric", "Floats"),
        ...     ("Non-Numeric", "Strings")
        ... ])
        >>> df.index = pd.MultiIndex.from_tuples([
        ...     ("L0", "ix1"), ("L0", "ix2"), ("L1", "ix3")
        ... ])
        >>> s = df.style.highlight_max(
        ...     props='cellcolor:[HTML]{FFFF00}; color:{red}; itshape:; bfseries:;'
        ... )
        >>> s.to_latex(
        ...     column_format="rrrrr", position="h", position_float="centering",
        ...     hrules=True, label="table:5", caption="Styled LaTeX Table",
        ...     multirow_align="t", multicol_align="r"
        ... )

        .. figure:: ../../_static/style/latex_2.png

        **Formatting**

        To format values :meth:`Styler.format` should be used prior to calling
        `Styler.to_latex`, as well as other methods such as :meth:`Styler.hide_index`
        or :meth:`Styler.hide_columns`, for example:

        >>> s.clear()
        >>> s.table_styles = []
        >>> s.caption = None
        >>> s.format({
        ...    ("Numeric", "Integers"): '\${}',
        ...    ("Numeric", "Floats"): '{:.3f}',
        ...    ("Non-Numeric", "Strings"): str.upper
        ... })
        >>> s.to_latex()
        \begin{tabular}{llrrl}
        {} & {} & \multicolumn{2}{r}{Numeric} & {Non-Numeric} \\
        {} & {} & {Integers} & {Floats} & {Strings} \\
        \multirow[c]{2}{*}{L0} & ix1 & \\$1 & 2.200 & DOGS \\
         & ix2 & \$3 & 4.400 & CATS \\
        L1 & ix3 & \$2 & 6.600 & COWS \\
        \end{tabular}

        **CSS Conversion**

        This method can convert a Styler constructured with HTML-CSS to LaTeX using
        the following limited conversions.

        ================== ==================== ============= ==========================
        CSS Attribute      CSS value            LaTeX Command LaTeX Options
        ================== ==================== ============= ==========================
        font-weight        | bold               | bfseries
                           | bolder             | bfseries
        font-style         | italic             | itshape
                           | oblique            | slshape
        background-color   | red                cellcolor     | {red}--lwrap
                           | #fe01ea                          | [HTML]{FE01EA}--lwrap
                           | #f0e                             | [HTML]{FF00EE}--lwrap
                           | rgb(128,255,0)                   | [rgb]{0.5,1,0}--lwrap
                           | rgba(128,0,0,0.5)                | [rgb]{0.5,0,0}--lwrap
                           | rgb(25%,255,50%)                 | [rgb]{0.25,1,0.5}--lwrap
        color              | red                color         | {red}
                           | #fe01ea                          | [HTML]{FE01EA}
                           | #f0e                             | [HTML]{FF00EE}
                           | rgb(128,255,0)                   | [rgb]{0.5,1,0}
                           | rgba(128,0,0,0.5)                | [rgb]{0.5,0,0}
                           | rgb(25%,255,50%)                 | [rgb]{0.25,1,0.5}
        ================== ==================== ============= ==========================

        It is also possible to add user-defined LaTeX only styles to a HTML-CSS Styler
        using the ``--latex`` flag, and to add LaTeX parsing options that the
        converter will detect within a CSS-comment.

        >>> df = pd.DataFrame([[1]])
        >>> df.style.set_properties(
        ...     **{"font-weight": "bold /* --dwrap */", "Huge": "--latex--rwrap"}
        ... ).to_latex(convert_css=True)
        \begin{tabular}{lr}
        {} & {0} \\
        0 & {\bfseries}{\Huge{1}} \\
        \end{tabular}
        """
        obj = self._copy(deepcopy=True)  # manipulate table_styles on obj, not self

        table_selectors = (
            [style["selector"] for style in self.table_styles]
            if self.table_styles is not None
            else []
        )

        if column_format is not None:
            # add more recent setting to table_styles
            obj.set_table_styles(
                [{"selector": "column_format", "props": f":{column_format}"}],
                overwrite=False,
            )
        elif "column_format" in table_selectors:
            pass  # adopt what has been previously set in table_styles
        else:
            # create a default: set float, complex, int cols to 'r' ('S'), index to 'l'
            _original_columns = self.data.columns
            self.data.columns = RangeIndex(stop=len(self.data.columns))
            numeric_cols = self.data._get_numeric_data().columns.to_list()
            self.data.columns = _original_columns
            column_format = "" if self.hide_index_ else "l" * self.data.index.nlevels
            for ci, _ in enumerate(self.data.columns):
                if ci not in self.hidden_columns:
                    column_format += (
                        ("r" if not siunitx else "S") if ci in numeric_cols else "l"
                    )
            obj.set_table_styles(
                [{"selector": "column_format", "props": f":{column_format}"}],
                overwrite=False,
            )

        if position:
            obj.set_table_styles(
                [{"selector": "position", "props": f":{position}"}],
                overwrite=False,
            )

        if position_float:
            if position_float not in ["raggedright", "raggedleft", "centering"]:
                raise ValueError(
                    f"`position_float` should be one of "
                    f"'raggedright', 'raggedleft', 'centering', "
                    f"got: '{position_float}'"
                )
            obj.set_table_styles(
                [{"selector": "position_float", "props": f":{position_float}"}],
                overwrite=False,
            )

        if hrules:
            obj.set_table_styles(
                [
                    {"selector": "toprule", "props": ":toprule"},
                    {"selector": "midrule", "props": ":midrule"},
                    {"selector": "bottomrule", "props": ":bottomrule"},
                ],
                overwrite=False,
            )

        if label:
            obj.set_table_styles(
                [{"selector": "label", "props": f":{{{label.replace(':', '§')}}}"}],
                overwrite=False,
            )

        if caption:
            obj.set_caption(caption)

        if sparse_index is None:
            sparse_index = get_option("styler.sparse.index")
        if sparse_columns is None:
            sparse_columns = get_option("styler.sparse.columns")

        latex = obj._render_latex(
            sparse_index=sparse_index,
            sparse_columns=sparse_columns,
            multirow_align=multirow_align,
            multicol_align=multicol_align,
            convert_css=convert_css,
        )

        return save_to_buffer(latex, buf=buf, encoding=encoding)

    def to_html(
        self,
        buf: FilePathOrBuffer[str] | None = None,
        *,
        table_uuid: str | None = None,
        table_attributes: str | None = None,
        encoding: str | None = None,
        doctype_html: bool = False,
        exclude_styles: bool = False,
    ):
        """
        Write Styler to a file, buffer or string in HTML-CSS format.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        buf : str, Path, or StringIO-like, optional, default None
            Buffer to write to. If ``None``, the output is returned as a string.
        table_uuid : str, optional
            Id attribute assigned to the <table> HTML element in the format:

            ``<table id="T_<table_uuid>" ..>``

            If not given uses Styler's initially assigned value.
        table_attributes : str, optional
            Attributes to assign within the `<table>` HTML element in the format:

            ``<table .. <table_attributes> >``

            If not given defaults to Styler's preexisting value.
        encoding : str, optional
            Character encoding setting for file output, and HTML meta tags,
            defaults to "utf-8" if None.
        doctype_html : bool, default False
            Whether to output a fully structured HTML file including all
            HTML elements, or just the core ``<style>`` and ``<table>`` elements.
        exclude_styles : bool, default False
            Whether to include the ``<style>`` element and all associated element
            ``class`` and ``id`` identifiers, or solely the ``<table>`` element without
            styling identifiers.

        Returns
        -------
        str or None
            If `buf` is None, returns the result as a string. Otherwise returns `None`.

        See Also
        --------
        DataFrame.to_html: Write a DataFrame to a file, buffer or string in HTML format.
        """
        if table_uuid:
            self.set_uuid(table_uuid)

        if table_attributes:
            self.set_table_attributes(table_attributes)

        # Build HTML string..
        html = self.render(
            exclude_styles=exclude_styles,
            encoding=encoding if encoding else "utf-8",
            doctype_html=doctype_html,
        )

        return save_to_buffer(
            html, buf=buf, encoding=(encoding if buf is not None else None)
        )

    def set_td_classes(self, classes: DataFrame) -> Styler:
        """
        Set the DataFrame of strings added to the ``class`` attribute of ``<td>``
        HTML elements.

        Parameters
        ----------
        classes : DataFrame
            DataFrame containing strings that will be translated to CSS classes,
            mapped by identical column and index key values that must exist on the
            underlying Styler data. None, NaN values, and empty strings will
            be ignored and not affect the rendered HTML.

        Returns
        -------
        self : Styler

        See Also
        --------
        Styler.set_table_styles: Set the table styles included within the ``<style>``
            HTML element.
        Styler.set_table_attributes: Set the table attributes added to the ``<table>``
            HTML element.

        Notes
        -----
        Can be used in combination with ``Styler.set_table_styles`` to define an
        internal CSS solution without reference to external CSS files.

        Examples
        --------
        >>> df = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
        >>> classes = pd.DataFrame([
        ...     ["min-val red", "", "blue"],
        ...     ["red", None, "blue max-val"]
        ... ], index=df.index, columns=df.columns)
        >>> df.style.set_td_classes(classes)

        Using `MultiIndex` columns and a `classes` `DataFrame` as a subset of the
        underlying,

        >>> df = pd.DataFrame([[1,2],[3,4]], index=["a", "b"],
        ...     columns=[["level0", "level0"], ["level1a", "level1b"]])
        >>> classes = pd.DataFrame(["min-val"], index=["a"],
        ...     columns=[["level0"],["level1a"]])
        >>> df.style.set_td_classes(classes)

        Form of the output with new additional css classes,

        >>> df = pd.DataFrame([[1]])
        >>> css = pd.DataFrame([["other-class"]])
        >>> s = Styler(df, uuid="_", cell_ids=False).set_td_classes(css)
        >>> s.hide_index().render()
        '<style type="text/css"></style>'
        '<table id="T__">'
        '  <thead>'
        '    <tr><th class="col_heading level0 col0" >0</th></tr>'
        '  </thead>'
        '  <tbody>'
        '    <tr><td class="data row0 col0 other-class" >1</td></tr>'
        '  </tbody>'
        '</table>'
        """
        if not classes.index.is_unique or not classes.columns.is_unique:
            raise KeyError(
                "Classes render only if `classes` has unique index and columns."
            )
        classes = classes.reindex_like(self.data)

        for r, row_tup in enumerate(classes.itertuples()):
            for c, value in enumerate(row_tup[1:]):
                if not (pd.isna(value) or value == ""):
                    self.cell_context[(r, c)] = str(value)

        return self

    def _update_ctx(self, attrs: DataFrame) -> None:
        """
        Update the state of the ``Styler`` for data cells.

        Collects a mapping of {index_label: [('<property>', '<value>'), ..]}.

        Parameters
        ----------
        attrs : DataFrame
            should contain strings of '<property>: <value>;<prop2>: <val2>'
            Whitespace shouldn't matter and the final trailing ';' shouldn't
            matter.
        """
        if not self.index.is_unique or not self.columns.is_unique:
            raise KeyError(
                "`Styler.apply` and `.applymap` are not compatible "
                "with non-unique index or columns."
            )

        for cn in attrs.columns:
            for rn, c in attrs[[cn]].itertuples():
                if not c:
                    continue
                css_list = maybe_convert_css_to_tuples(c)
                i, j = self.index.get_loc(rn), self.columns.get_loc(cn)
                self.ctx[(i, j)].extend(css_list)

    def _copy(self, deepcopy: bool = False) -> Styler:
        """
        Copies a Styler, allowing for deepcopy or shallow copy

        Copying a Styler aims to recreate a new Styler object which contains the same
        data and styles as the original.

        Data dependent attributes [copied and NOT exported]:
          - formatting (._display_funcs)
          - hidden index values or column values (.hidden_rows, .hidden_columns)
          - tooltips
          - cell_context (cell css classes)
          - ctx (cell css styles)
          - caption

        Non-data dependent attributes [copied and exported]:
          - hidden index state and hidden columns state (.hide_index_, .hide_columns_)
          - table_attributes
          - table_styles
          - applied styles (_todo)

        """
        # GH 40675
        styler = Styler(
            self.data,  # populates attributes 'data', 'columns', 'index' as shallow
            uuid_len=self.uuid_len,
        )
        shallow = [  # simple string or boolean immutables
            "hide_index_",
            "hide_columns_",
            "table_attributes",
            "cell_ids",
            "caption",
        ]
        deep = [  # nested lists or dicts
            "_display_funcs",
            "hidden_rows",
            "hidden_columns",
            "ctx",
            "cell_context",
            "_todo",
            "table_styles",
            "tooltips",
        ]

        for attr in shallow:
            setattr(styler, attr, getattr(self, attr))

        for attr in deep:
            val = getattr(self, attr)
            setattr(styler, attr, copy.deepcopy(val) if deepcopy else val)

        return styler

    def __copy__(self) -> Styler:
        return self._copy(deepcopy=False)

    def __deepcopy__(self, memo) -> Styler:
        return self._copy(deepcopy=True)

    def clear(self) -> None:
        """
        Reset the ``Styler``, removing any previously applied styles.

        Returns None.
        """
        self.ctx.clear()
        self.tooltips = None
        self.cell_context.clear()
        self._todo.clear()

        self.hide_index_ = False
        self.hidden_columns = []
        # self.format and self.table_styles may be dependent on user
        # input in self.__init__()

    def _apply(
        self,
        func: Callable[..., Styler],
        axis: Axis | None = 0,
        subset: Subset | None = None,
        **kwargs,
    ) -> Styler:
        subset = slice(None) if subset is None else subset
        subset = non_reducing_slice(subset)
        data = self.data.loc[subset]
        if axis is not None:
            result = data.apply(func, axis=axis, result_type="expand", **kwargs)
            result.columns = data.columns
        else:
            result = func(data, **kwargs)
            if not isinstance(result, DataFrame):
                if not isinstance(result, np.ndarray):
                    raise TypeError(
                        f"Function {repr(func)} must return a DataFrame or ndarray "
                        f"when passed to `Styler.apply` with axis=None"
                    )
                if not (data.shape == result.shape):
                    raise ValueError(
                        f"Function {repr(func)} returned ndarray with wrong shape.\n"
                        f"Result has shape: {result.shape}\n"
                        f"Expected shape: {data.shape}"
                    )
                result = DataFrame(result, index=data.index, columns=data.columns)
            elif not (
                result.index.equals(data.index) and result.columns.equals(data.columns)
            ):
                raise ValueError(
                    f"Result of {repr(func)} must have identical "
                    f"index and columns as the input"
                )

        if result.shape != data.shape:
            raise ValueError(
                f"Function {repr(func)} returned the wrong shape.\n"
                f"Result has shape: {result.shape}\n"
                f"Expected shape:   {data.shape}"
            )
        self._update_ctx(result)
        return self

    def apply(
        self,
        func: Callable[..., Styler],
        axis: Axis | None = 0,
        subset: Subset | None = None,
        **kwargs,
    ) -> Styler:
        """
        Apply a CSS-styling function column-wise, row-wise, or table-wise.

        Updates the HTML representation with the result.

        Parameters
        ----------
        func : function
            ``func`` should take a Series if ``axis`` in [0,1] and return an object
            of same length, also with identical index if the object is a Series.
            ``func`` should take a DataFrame if ``axis`` is ``None`` and return either
            an ndarray with the same shape or a DataFrame with identical columns and
            index.

            .. versionchanged:: 1.3.0

        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        self : Styler

        See Also
        --------
        Styler.applymap: Apply a CSS-styling function elementwise.

        Notes
        -----
        The elements of the output of ``func`` should be CSS styles as strings, in the
        format 'attribute: value; attribute2: value2; ...' or,
        if nothing is to be applied to that element, an empty string or ``None``.

        This is similar to ``DataFrame.apply``, except that ``axis=None``
        applies the function to the entire DataFrame at once,
        rather than column-wise or row-wise.

        Examples
        --------
        >>> def highlight_max(x, color):
        ...     return np.where(x == np.nanmax(x.to_numpy()), f"color: {color};", None)
        >>> df = pd.DataFrame(np.random.randn(5, 2), columns=["A", "B"])
        >>> df.style.apply(highlight_max, color='red')
        >>> df.style.apply(highlight_max, color='blue', axis=1)
        >>> df.style.apply(highlight_max, color='green', axis=None)

        Using ``subset`` to restrict application to a single column or multiple columns

        >>> df.style.apply(highlight_max, color='red', subset="A")
        >>> df.style.apply(highlight_max, color='red', subset=["A", "B"])

        Using a 2d input to ``subset`` to select rows in addition to columns

        >>> df.style.apply(highlight_max, color='red', subset=([0,1,2], slice(None))
        >>> df.style.apply(highlight_max, color='red', subset=(slice(0,5,2), "A")
        """
        self._todo.append(
            (lambda instance: getattr(instance, "_apply"), (func, axis, subset), kwargs)
        )
        return self

    def _applymap(
        self, func: Callable, subset: Subset | None = None, **kwargs
    ) -> Styler:
        func = partial(func, **kwargs)  # applymap doesn't take kwargs?
        if subset is None:
            subset = IndexSlice[:]
        subset = non_reducing_slice(subset)
        result = self.data.loc[subset].applymap(func)
        self._update_ctx(result)
        return self

    def applymap(
        self, func: Callable, subset: Subset | None = None, **kwargs
    ) -> Styler:
        """
        Apply a CSS-styling function elementwise.

        Updates the HTML representation with the result.

        Parameters
        ----------
        func : function
            ``func`` should take a scalar and return a scalar.
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        **kwargs : dict
            Pass along to ``func``.

        Returns
        -------
        self : Styler

        See Also
        --------
        Styler.apply: Apply a CSS-styling function column-wise, row-wise, or table-wise.

        Notes
        -----
        The elements of the output of ``func`` should be CSS styles as strings, in the
        format 'attribute: value; attribute2: value2; ...' or,
        if nothing is to be applied to that element, an empty string or ``None``.

        Examples
        --------
        >>> def color_negative(v, color):
        ...     return f"color: {color};" if v < 0 else None
        >>> df = pd.DataFrame(np.random.randn(5, 2), columns=["A", "B"])
        >>> df.style.applymap(color_negative, color='red')

        Using ``subset`` to restrict application to a single column or multiple columns

        >>> df.style.applymap(color_negative, color='red', subset="A")
        >>> df.style.applymap(color_negative, color='red', subset=["A", "B"])

        Using a 2d input to ``subset`` to select rows in addition to columns

        >>> df.style.applymap(color_negative, color='red', subset=([0,1,2], slice(None))
        >>> df.style.applymap(color_negative, color='red', subset=(slice(0,5,2), "A")
        """
        self._todo.append(
            (lambda instance: getattr(instance, "_applymap"), (func, subset), kwargs)
        )
        return self

    def where(
        self,
        cond: Callable,
        value: str,
        other: str | None = None,
        subset: Subset | None = None,
        **kwargs,
    ) -> Styler:
        """
        Apply CSS-styles based on a conditional function elementwise.

        .. deprecated:: 1.3.0

        Updates the HTML representation with a style which is
        selected in accordance with the return value of a function.

        Parameters
        ----------
        cond : callable
            ``cond`` should take a scalar, and optional keyword arguments, and return
            a boolean.
        value : str
            Applied when ``cond`` returns true.
        other : str
            Applied when ``cond`` returns false.
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        **kwargs : dict
            Pass along to ``cond``.

        Returns
        -------
        self : Styler

        See Also
        --------
        Styler.applymap: Apply a CSS-styling function elementwise.
        Styler.apply: Apply a CSS-styling function column-wise, row-wise, or table-wise.

        Notes
        -----
        This method is deprecated.

        This method is a convenience wrapper for :meth:`Styler.applymap`, which we
        recommend using instead.

        The example:

        >>> df = pd.DataFrame([[1, 2], [3, 4]])
        >>> def cond(v, limit=4):
        ...     return v > 1 and v != limit
        >>> df.style.where(cond, value='color:green;', other='color:red;')

        should be refactored to:

        >>> def style_func(v, value, other, limit=4):
        ...     cond = v > 1 and v != limit
        ...     return value if cond else other
        >>> df.style.applymap(style_func, value='color:green;', other='color:red;')
        """
        warnings.warn(
            "this method is deprecated in favour of `Styler.applymap()`",
            FutureWarning,
            stacklevel=2,
        )

        if other is None:
            other = ""

        return self.applymap(
            lambda val: value if cond(val, **kwargs) else other,
            subset=subset,
        )

    def set_precision(self, precision: int) -> StylerRenderer:
        """
        Set the precision used to display values.

        .. deprecated:: 1.3.0

        Parameters
        ----------
        precision : int

        Returns
        -------
        self : Styler

        Notes
        -----
        This method is deprecated see `Styler.format`.
        """
        warnings.warn(
            "this method is deprecated in favour of `Styler.format(precision=..)`",
            FutureWarning,
            stacklevel=2,
        )
        self.precision = precision
        return self.format(precision=precision, na_rep=self.na_rep)

    def set_table_attributes(self, attributes: str) -> Styler:
        """
        Set the table attributes added to the ``<table>`` HTML element.

        These are items in addition to automatic (by default) ``id`` attribute.

        Parameters
        ----------
        attributes : str

        Returns
        -------
        self : Styler

        See Also
        --------
        Styler.set_table_styles: Set the table styles included within the ``<style>``
            HTML element.
        Styler.set_td_classes: Set the DataFrame of strings added to the ``class``
            attribute of ``<td>`` HTML elements.

        Examples
        --------
        >>> df = pd.DataFrame(np.random.randn(10, 4))
        >>> df.style.set_table_attributes('class="pure-table"')
        # ... <table class="pure-table"> ...
        """
        self.table_attributes = attributes
        return self

    def export(self) -> list[tuple[Callable, tuple, dict]]:
        """
        Export the styles applied to the current ``Styler``.

        Can be applied to a second Styler with ``Styler.use``.

        Returns
        -------
        styles : list

        See Also
        --------
        Styler.use: Set the styles on the current ``Styler``.
        """
        return self._todo

    def use(self, styles: list[tuple[Callable, tuple, dict]]) -> Styler:
        """
        Set the styles on the current ``Styler``.

        Possibly uses styles from ``Styler.export``.

        Parameters
        ----------
        styles : list
            List of style functions.

        Returns
        -------
        self : Styler

        See Also
        --------
        Styler.export : Export the styles to applied to the current ``Styler``.
        """
        self._todo.extend(styles)
        return self

    def set_uuid(self, uuid: str) -> Styler:
        """
        Set the uuid applied to ``id`` attributes of HTML elements.

        Parameters
        ----------
        uuid : str

        Returns
        -------
        self : Styler

        Notes
        -----
        Almost all HTML elements within the table, and including the ``<table>`` element
        are assigned ``id`` attributes. The format is ``T_uuid_<extra>`` where
        ``<extra>`` is typically a more specific identifier, such as ``row1_col2``.
        """
        self.uuid = uuid
        return self

    def set_caption(self, caption: str | tuple) -> Styler:
        """
        Set the text added to a ``<caption>`` HTML element.

        Parameters
        ----------
        caption : str, tuple
            For HTML output either the string input is used or the first element of the
            tuple. For LaTeX the string input provides a caption and the additional
            tuple input allows for full captions and short captions, in that order.

        Returns
        -------
        self : Styler
        """
        self.caption = caption
        return self

    def set_sticky(
        self,
        axis: Axis = 0,
        pixel_size: int | None = None,
        levels: list[int] | None = None,
    ) -> Styler:
        """
        Add CSS to permanently display the index or column headers in a scrolling frame.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Whether to make the index or column headers sticky.
        pixel_size : int, optional
            Required to configure the width of index cells or the height of column
            header cells when sticking a MultiIndex (or with a named Index).
            Defaults to 75 and 25 respectively.
        levels : list of int
            If ``axis`` is a MultiIndex the specific levels to stick. If ``None`` will
            stick all levels.

        Returns
        -------
        self : Styler

        Notes
        -----
        This method uses the CSS 'position: sticky;' property to display. It is
        designed to work with visible axes, therefore both:

          - `styler.set_sticky(axis="index").hide_index()`
          - `styler.set_sticky(axis="columns").hide_columns()`

        may produce strange behaviour due to CSS controls with missing elements.
        """
        if axis in [0, "index"]:
            axis, obj = 0, self.data.index
            pixel_size = 75 if not pixel_size else pixel_size
        elif axis in [1, "columns"]:
            axis, obj = 1, self.data.columns
            pixel_size = 25 if not pixel_size else pixel_size
        else:
            raise ValueError("`axis` must be one of {0, 1, 'index', 'columns'}")

        props = "position:sticky; background-color:white;"
        if not isinstance(obj, pd.MultiIndex):
            # handling MultiIndexes requires different CSS

            if axis == 1:
                # stick the first <tr> of <head> and, if index names, the second <tr>
                # if self._hide_columns then no <thead><tr> here will exist: no conflict
                styles: CSSStyles = [
                    {
                        "selector": "thead tr:nth-child(1) th",
                        "props": props + "top:0px; z-index:2;",
                    }
                ]
                if not self.index.names[0] is None:
                    styles[0]["props"] = (
                        props + f"top:0px; z-index:2; height:{pixel_size}px;"
                    )
                    styles.append(
                        {
                            "selector": "thead tr:nth-child(2) th",
                            "props": props
                            + f"top:{pixel_size}px; z-index:2; height:{pixel_size}px; ",
                        }
                    )
            else:
                # stick the first <th> of each <tr> in both <thead> and <tbody>
                # if self._hide_index then no <th> will exist in <tbody>: no conflict
                # but <th> will exist in <thead>: conflict with initial element
                styles = [
                    {
                        "selector": "thead tr th:nth-child(1)",
                        "props": props + "left:0px; z-index:3 !important;",
                    },
                    {
                        "selector": "tbody tr th:nth-child(1)",
                        "props": props + "left:0px; z-index:1;",
                    },
                ]

        else:
            # handle the MultiIndex case
            range_idx = list(range(obj.nlevels))
            levels = sorted(levels) if levels else range_idx

            if axis == 1:
                styles = []
                for i, level in enumerate(levels):
                    styles.append(
                        {
                            "selector": f"thead tr:nth-child({level+1}) th",
                            "props": props
                            + (
                                f"top:{i * pixel_size}px; height:{pixel_size}px; "
                                "z-index:2;"
                            ),
                        }
                    )
                if not all(name is None for name in self.index.names):
                    styles.append(
                        {
                            "selector": f"thead tr:nth-child({obj.nlevels+1}) th",
                            "props": props
                            + (
                                f"top:{(i+1) * pixel_size}px; height:{pixel_size}px; "
                                "z-index:2;"
                            ),
                        }
                    )

            else:
                styles = []
                for i, level in enumerate(levels):
                    props_ = props + (
                        f"left:{i * pixel_size}px; "
                        f"min-width:{pixel_size}px; "
                        f"max-width:{pixel_size}px; "
                    )
                    styles.extend(
                        [
                            {
                                "selector": f"thead tr th:nth-child({level+1})",
                                "props": props_ + "z-index:3 !important;",
                            },
                            {
                                "selector": f"tbody tr th.level{level}",
                                "props": props_ + "z-index:1;",
                            },
                        ]
                    )

        return self.set_table_styles(styles, overwrite=False)

    def set_table_styles(
        self,
        table_styles: dict[Any, CSSStyles] | CSSStyles,
        axis: int = 0,
        overwrite: bool = True,
    ) -> Styler:
        """
        Set the table styles included within the ``<style>`` HTML element.

        This function can be used to style the entire table, columns, rows or
        specific HTML selectors.

        Parameters
        ----------
        table_styles : list or dict
            If supplying a list, each individual table_style should be a
            dictionary with ``selector`` and ``props`` keys. ``selector``
            should be a CSS selector that the style will be applied to
            (automatically prefixed by the table's UUID) and ``props``
            should be a list of tuples with ``(attribute, value)``.
            If supplying a dict, the dict keys should correspond to
            column names or index values, depending upon the specified
            `axis` argument. These will be mapped to row or col CSS
            selectors. MultiIndex values as dict keys should be
            in their respective tuple form. The dict values should be
            a list as specified in the form with CSS selectors and
            props that will be applied to the specified row or column.

            .. versionchanged:: 1.2.0

        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``). Only used if `table_styles` is
            dict.

            .. versionadded:: 1.2.0

        overwrite : bool, default True
            Styles are replaced if `True`, or extended if `False`. CSS
            rules are preserved so most recent styles set will dominate
            if selectors intersect.

            .. versionadded:: 1.2.0

        Returns
        -------
        self : Styler

        See Also
        --------
        Styler.set_td_classes: Set the DataFrame of strings added to the ``class``
            attribute of ``<td>`` HTML elements.
        Styler.set_table_attributes: Set the table attributes added to the ``<table>``
            HTML element.

        Examples
        --------
        >>> df = pd.DataFrame(np.random.randn(10, 4),
        ...                   columns=['A', 'B', 'C', 'D'])
        >>> df.style.set_table_styles(
        ...     [{'selector': 'tr:hover',
        ...       'props': [('background-color', 'yellow')]}]
        ... )

        Or with CSS strings

        >>> df.style.set_table_styles(
        ...     [{'selector': 'tr:hover',
        ...       'props': 'background-color: yellow; font-size: 1em;']}]
        ... )

        Adding column styling by name

        >>> df.style.set_table_styles({
        ...     'A': [{'selector': '',
        ...            'props': [('color', 'red')]}],
        ...     'B': [{'selector': 'td',
        ...            'props': 'color: blue;']}]
        ... }, overwrite=False)

        Adding row styling

        >>> df.style.set_table_styles({
        ...     0: [{'selector': 'td:hover',
        ...          'props': [('font-size', '25px')]}]
        ... }, axis=1, overwrite=False)
        """
        if isinstance(table_styles, dict):
            if axis in [0, "index"]:
                obj, idf = self.data.columns, ".col"
            else:
                obj, idf = self.data.index, ".row"

            table_styles = [
                {
                    "selector": str(s["selector"]) + idf + str(idx),
                    "props": maybe_convert_css_to_tuples(s["props"]),
                }
                for key, styles in table_styles.items()
                for idx in obj.get_indexer_for([key])
                for s in styles
            ]
        else:
            table_styles = [
                {
                    "selector": s["selector"],
                    "props": maybe_convert_css_to_tuples(s["props"]),
                }
                for s in table_styles
            ]

        if not overwrite and self.table_styles is not None:
            self.table_styles.extend(table_styles)
        else:
            self.table_styles = table_styles
        return self

    def set_na_rep(self, na_rep: str) -> StylerRenderer:
        """
        Set the missing data representation on a ``Styler``.

        .. versionadded:: 1.0.0

        .. deprecated:: 1.3.0

        Parameters
        ----------
        na_rep : str

        Returns
        -------
        self : Styler

        Notes
        -----
        This method is deprecated. See `Styler.format()`
        """
        warnings.warn(
            "this method is deprecated in favour of `Styler.format(na_rep=..)`",
            FutureWarning,
            stacklevel=2,
        )
        self.na_rep = na_rep
        return self.format(na_rep=na_rep, precision=self.precision)

    def hide_index(self, subset: Subset | None = None) -> Styler:
        """
        Hide the entire index, or specific keys in the index from rendering.

        This method has dual functionality:

          - if ``subset`` is ``None`` then the entire index will be hidden whilst
            displaying all data-rows.
          - if a ``subset`` is given then those specific rows will be hidden whilst the
            index itself remains visible.

        .. versionchanged:: 1.3.0

        Parameters
        ----------
        subset : label, array-like, IndexSlice, optional
            A valid 1d input or single key along the index axis within
            `DataFrame.loc[<subset>, :]`, to limit ``data`` to *before* applying
            the function.

        Returns
        -------
        self : Styler

        See Also
        --------
        Styler.hide_columns: Hide the entire column headers row, or specific columns.

        Examples
        --------
        Simple application hiding specific rows:

        >>> df = pd.DataFrame([[1,2], [3,4], [5,6]], index=["a", "b", "c"])
        >>> df.style.hide_index(["a", "b"])
             0    1
        c    5    6

        Hide the index and retain the data values:

        >>> midx = pd.MultiIndex.from_product([["x", "y"], ["a", "b", "c"]])
        >>> df = pd.DataFrame(np.random.randn(6,6), index=midx, columns=midx)
        >>> df.style.format("{:.1f}").hide_index()
                         x                    y
           a      b      c      a      b      c
         0.1    0.0    0.4    1.3    0.6   -1.4
         0.7    1.0    1.3    1.5   -0.0   -0.2
         1.4   -0.8    1.6   -0.2   -0.4   -0.3
         0.4    1.0   -0.2   -0.8   -1.2    1.1
        -0.6    1.2    1.8    1.9    0.3    0.3
         0.8    0.5   -0.3    1.2    2.2   -0.8

        Hide specific rows but retain the index:

        >>> df.style.format("{:.1f}").hide_index(subset=(slice(None), ["a", "c"]))
                                 x                    y
                   a      b      c      a      b      c
        x   b    0.7    1.0    1.3    1.5   -0.0   -0.2
        y   b   -0.6    1.2    1.8    1.9    0.3    0.3

        Hide specific rows and the index:

        >>> df.style.format("{:.1f}").hide_index(subset=(slice(None), ["a", "c"]))
        ...     .hide_index()
                         x                    y
           a      b      c      a      b      c
         0.7    1.0    1.3    1.5   -0.0   -0.2
        -0.6    1.2    1.8    1.9    0.3    0.3
        """
        if subset is None:
            self.hide_index_ = True
        else:
            subset_ = IndexSlice[subset, :]  # new var so mypy reads not Optional
            subset = non_reducing_slice(subset_)
            hide = self.data.loc[subset]
            hrows = self.index.get_indexer_for(hide.index)
            # error: Incompatible types in assignment (expression has type
            # "ndarray", variable has type "Sequence[int]")
            self.hidden_rows = hrows  # type: ignore[assignment]
        return self

    def hide_columns(self, subset: Subset | None = None) -> Styler:
        """
        Hide the column headers or specific keys in the columns from rendering.

        This method has dual functionality:

          - if ``subset`` is ``None`` then the entire column headers row will be hidden
            whilst the data-values remain visible.
          - if a ``subset`` is given then those specific columns, including the
            data-values will be hidden, whilst the column headers row remains visible.

        .. versionchanged:: 1.3.0

        Parameters
        ----------
        subset : label, array-like, IndexSlice, optional
            A valid 1d input or single key along the columns axis within
            `DataFrame.loc[:, <subset>]`, to limit ``data`` to *before* applying
            the function.

        Returns
        -------
        self : Styler

        See Also
        --------
        Styler.hide_index: Hide the entire index, or specific keys in the index.

        Examples
        --------
        Simple application hiding specific columns:

        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
        >>> df.style.hide_columns(["a", "b"])
             c
        0    3
        1    6

        Hide column headers and retain the data values:

        >>> midx = pd.MultiIndex.from_product([["x", "y"], ["a", "b", "c"]])
        >>> df = pd.DataFrame(np.random.randn(6,6), index=midx, columns=midx)
        >>> df.style.format("{:.1f}").hide_columns()
        x   d    0.1    0.0    0.4    1.3    0.6   -1.4
            e    0.7    1.0    1.3    1.5   -0.0   -0.2
            f    1.4   -0.8    1.6   -0.2   -0.4   -0.3
        y   d    0.4    1.0   -0.2   -0.8   -1.2    1.1
            e   -0.6    1.2    1.8    1.9    0.3    0.3
            f    0.8    0.5   -0.3    1.2    2.2   -0.8

        Hide specific columns but retain the column headers:

        >>> df.style.format("{:.1f}").hide_columns(subset=(slice(None), ["a", "c"]))
                   x      y
                   b      b
        x   a    0.0    0.6
            b    1.0   -0.0
            c   -0.8   -0.4
        y   a    1.0   -1.2
            b    1.2    0.3
            c    0.5    2.2

        Hide specific columns and the column headers:

        >>> df.style.format("{:.1f}").hide_columns(subset=(slice(None), ["a", "c"]))
        ...     .hide_columns()
        x   a    0.0    0.6
            b    1.0   -0.0
            c   -0.8   -0.4
        y   a    1.0   -1.2
            b    1.2    0.3
            c    0.5    2.2
        """
        if subset is None:
            self.hide_columns_ = True
        else:
            subset_ = IndexSlice[:, subset]  # new var so mypy reads not Optional
            subset = non_reducing_slice(subset_)
            hide = self.data.loc[subset]
            hcols = self.columns.get_indexer_for(hide.columns)
            # error: Incompatible types in assignment (expression has type
            # "ndarray", variable has type "Sequence[int]")
            self.hidden_columns = hcols  # type: ignore[assignment]
        return self

    # -----------------------------------------------------------------------
    # A collection of "builtin" styles
    # -----------------------------------------------------------------------

    @doc(
        name="background",
        alt="text",
        image_prefix="bg",
        axis="{0 or 'index', 1 or 'columns', None}",
        text_threshold="",
    )
    def background_gradient(
        self,
        cmap="PuBu",
        low: float = 0,
        high: float = 0,
        axis: Axis | None = 0,
        subset: Subset | None = None,
        text_color_threshold: float = 0.408,
        vmin: float | None = None,
        vmax: float | None = None,
        gmap: Sequence | None = None,
    ) -> Styler:
        """
        Color the {name} in a gradient style.

        The {name} color is determined according
        to the data in each column, row or frame, or by a given
        gradient map. Requires matplotlib.

        Parameters
        ----------
        cmap : str or colormap
            Matplotlib colormap.
        low : float
            Compress the color range at the low end. This is a multiple of the data
            range to extend below the minimum; good values usually in [0, 1],
            defaults to 0.
        high : float
            Compress the color range at the high end. This is a multiple of the data
            range to extend above the maximum; good values usually in [0, 1],
            defaults to 0.
        axis : {axis}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        text_color_threshold : float or int
            {text_threshold}
            Luminance threshold for determining text color in [0, 1]. Facilitates text
            visibility across varying background colors. All text is dark if 0, and
            light if 1, defaults to 0.408.
        vmin : float, optional
            Minimum data value that corresponds to colormap minimum value.
            If not specified the minimum value of the data (or gmap) will be used.

            .. versionadded:: 1.0.0

        vmax : float, optional
            Maximum data value that corresponds to colormap maximum value.
            If not specified the maximum value of the data (or gmap) will be used.

            .. versionadded:: 1.0.0

        gmap : array-like, optional
            Gradient map for determining the {name} colors. If not supplied
            will use the underlying data from rows, columns or frame. If given as an
            ndarray or list-like must be an identical shape to the underlying data
            considering ``axis`` and ``subset``. If given as DataFrame or Series must
            have same index and column labels considering ``axis`` and ``subset``.
            If supplied, ``vmin`` and ``vmax`` should be given relative to this
            gradient map.

            .. versionadded:: 1.3.0

        Returns
        -------
        self : Styler

        See Also
        --------
        Styler.{alt}_gradient: Color the {alt} in a gradient style.

        Notes
        -----
        When using ``low`` and ``high`` the range
        of the gradient, given by the data if ``gmap`` is not given or by ``gmap``,
        is extended at the low end effectively by
        `map.min - low * map.range` and at the high end by
        `map.max + high * map.range` before the colors are normalized and determined.

        If combining with ``vmin`` and ``vmax`` the `map.min`, `map.max` and
        `map.range` are replaced by values according to the values derived from
        ``vmin`` and ``vmax``.

        This method will preselect numeric columns and ignore non-numeric columns
        unless a ``gmap`` is supplied in which case no preselection occurs.

        Examples
        --------
        >>> df = pd.DataFrame(columns=["City", "Temp (c)", "Rain (mm)", "Wind (m/s)"],
        ...                   data=[["Stockholm", 21.6, 5.0, 3.2],
        ...                         ["Oslo", 22.4, 13.3, 3.1],
        ...                         ["Copenhagen", 24.5, 0.0, 6.7]])

        Shading the values column-wise, with ``axis=0``, preselecting numeric columns

        >>> df.style.{name}_gradient(axis=0)

        .. figure:: ../../_static/style/{image_prefix}_ax0.png

        Shading all values collectively using ``axis=None``

        >>> df.style.{name}_gradient(axis=None)

        .. figure:: ../../_static/style/{image_prefix}_axNone.png

        Compress the color map from the both ``low`` and ``high`` ends

        >>> df.style.{name}_gradient(axis=None, low=0.75, high=1.0)

        .. figure:: ../../_static/style/{image_prefix}_axNone_lowhigh.png

        Manually setting ``vmin`` and ``vmax`` gradient thresholds

        >>> df.style.{name}_gradient(axis=None, vmin=6.7, vmax=21.6)

        .. figure:: ../../_static/style/{image_prefix}_axNone_vminvmax.png

        Setting a ``gmap`` and applying to all columns with another ``cmap``

        >>> df.style.{name}_gradient(axis=0, gmap=df['Temp (c)'], cmap='YlOrRd')

        .. figure:: ../../_static/style/{image_prefix}_gmap.png

        Setting the gradient map for a dataframe (i.e. ``axis=None``), we need to
        explicitly state ``subset`` to match the ``gmap`` shape

        >>> gmap = np.array([[1,2,3], [2,3,4], [3,4,5]])
        >>> df.style.{name}_gradient(axis=None, gmap=gmap,
        ...     cmap='YlOrRd', subset=['Temp (c)', 'Rain (mm)', 'Wind (m/s)']
        ... )

        .. figure:: ../../_static/style/{image_prefix}_axNone_gmap.png
        """
        if subset is None and gmap is None:
            subset = self.data.select_dtypes(include=np.number).columns

        self.apply(
            _background_gradient,
            cmap=cmap,
            subset=subset,
            axis=axis,
            low=low,
            high=high,
            text_color_threshold=text_color_threshold,
            vmin=vmin,
            vmax=vmax,
            gmap=gmap,
        )
        return self

    @doc(
        background_gradient,
        name="text",
        alt="background",
        image_prefix="tg",
        axis="{0 or 'index', 1 or 'columns', None}",
        text_threshold="This argument is ignored (only used in `background_gradient`).",
    )
    def text_gradient(
        self,
        cmap="PuBu",
        low: float = 0,
        high: float = 0,
        axis: Axis | None = 0,
        subset: Subset | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        gmap: Sequence | None = None,
    ) -> Styler:
        if subset is None and gmap is None:
            subset = self.data.select_dtypes(include=np.number).columns

        return self.apply(
            _background_gradient,
            cmap=cmap,
            subset=subset,
            axis=axis,
            low=low,
            high=high,
            vmin=vmin,
            vmax=vmax,
            gmap=gmap,
            text_only=True,
        )

    def set_properties(self, subset: Subset | None = None, **kwargs) -> Styler:
        """
        Set defined CSS-properties to each ``<td>`` HTML element within the given
        subset.

        Parameters
        ----------
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        **kwargs : dict
            A dictionary of property, value pairs to be set for each cell.

        Returns
        -------
        self : Styler

        Notes
        -----
        This is a convenience methods which wraps the :meth:`Styler.applymap` calling a
        function returning the CSS-properties independently of the data.

        Examples
        --------
        >>> df = pd.DataFrame(np.random.randn(10, 4))
        >>> df.style.set_properties(color="white", align="right")
        >>> df.style.set_properties(**{'background-color': 'yellow'})
        """
        values = "".join(f"{p}: {v};" for p, v in kwargs.items())
        return self.applymap(lambda x: values, subset=subset)

    @staticmethod
    def _bar(
        s,
        align: str,
        colors: list[str],
        width: float = 100,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        """
        Draw bar chart in dataframe cells.
        """
        # Get input value range.
        smin = np.nanmin(s.to_numpy()) if vmin is None else vmin
        smax = np.nanmax(s.to_numpy()) if vmax is None else vmax
        if align == "mid":
            smin = min(0, smin)
            smax = max(0, smax)
        elif align == "zero":
            # For "zero" mode, we want the range to be symmetrical around zero.
            smax = max(abs(smin), abs(smax))
            smin = -smax
        # Transform to percent-range of linear-gradient
        normed = width * (s.to_numpy(dtype=float) - smin) / (smax - smin + 1e-12)
        zero = -width * smin / (smax - smin + 1e-12)

        def css_bar(start: float, end: float, color: str) -> str:
            """
            Generate CSS code to draw a bar from start to end.
            """
            css = "width: 10em; height: 80%;"
            if end > start:
                css += "background: linear-gradient(90deg,"
                if start > 0:
                    css += f" transparent {start:.1f}%, {color} {start:.1f}%, "
                e = min(end, width)
                css += f"{color} {e:.1f}%, transparent {e:.1f}%)"
            return css

        def css(x):
            if pd.isna(x):
                return ""

            # avoid deprecated indexing `colors[x > zero]`
            color = colors[1] if x > zero else colors[0]

            if align == "left":
                return css_bar(0, x, color)
            else:
                return css_bar(min(x, zero), max(x, zero), color)

        if s.ndim == 1:
            return [css(x) for x in normed]
        else:
            return DataFrame(
                [[css(x) for x in row] for row in normed],
                index=s.index,
                columns=s.columns,
            )

    def bar(
        self,
        subset: Subset | None = None,
        axis: Axis | None = 0,
        color="#d65f5f",
        width: float = 100,
        align: str = "left",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> Styler:
        """
        Draw bar chart in the cell backgrounds.

        Parameters
        ----------
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        color : str or 2-tuple/list
            If a str is passed, the color is the same for both
            negative and positive numbers. If 2-tuple/list is used, the
            first element is the color_negative and the second is the
            color_positive (eg: ['#d65f5f', '#5fba7d']).
        width : float, default 100
            A number between 0 or 100. The largest value will cover `width`
            percent of the cell's width.
        align : {'left', 'zero',' mid'}, default 'left'
            How to align the bars with the cells.

            - 'left' : the min value starts at the left of the cell.
            - 'zero' : a value of zero is located at the center of the cell.
            - 'mid' : the center of the cell is at (max-min)/2, or
              if values are all negative (positive) the zero is aligned
              at the right (left) of the cell.
        vmin : float, optional
            Minimum bar value, defining the left hand limit
            of the bar drawing range, lower values are clipped to `vmin`.
            When None (default): the minimum value of the data will be used.
        vmax : float, optional
            Maximum bar value, defining the right hand limit
            of the bar drawing range, higher values are clipped to `vmax`.
            When None (default): the maximum value of the data will be used.

        Returns
        -------
        self : Styler
        """
        if align not in ("left", "zero", "mid"):
            raise ValueError("`align` must be one of {'left', 'zero',' mid'}")

        if not (is_list_like(color)):
            color = [color, color]
        elif len(color) == 1:
            color = [color[0], color[0]]
        elif len(color) > 2:
            raise ValueError(
                "`color` must be string or a list-like "
                "of length 2: [`color_neg`, `color_pos`] "
                "(eg: color=['#d65f5f', '#5fba7d'])"
            )

        if subset is None:
            subset = self.data.select_dtypes(include=np.number).columns

        self.apply(
            self._bar,
            subset=subset,
            axis=axis,
            align=align,
            colors=color,
            width=width,
            vmin=vmin,
            vmax=vmax,
        )

        return self

    def highlight_null(
        self,
        null_color: str = "red",
        subset: Subset | None = None,
        props: str | None = None,
    ) -> Styler:
        """
        Highlight missing values with a style.

        Parameters
        ----------
        null_color : str, default 'red'
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.

            .. versionadded:: 1.1.0

        props : str, default None
            CSS properties to use for highlighting. If ``props`` is given, ``color``
            is not used.

            .. versionadded:: 1.3.0

        Returns
        -------
        self : Styler

        See Also
        --------
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_between: Highlight a defined range with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.
        """

        def f(data: DataFrame, props: str) -> np.ndarray:
            return np.where(pd.isna(data).to_numpy(), props, "")

        if props is None:
            props = f"background-color: {null_color};"
        # error: Argument 1 to "apply" of "Styler" has incompatible type
        # "Callable[[DataFrame, str], ndarray]"; expected "Callable[..., Styler]"
        return self.apply(
            f, axis=None, subset=subset, props=props  # type: ignore[arg-type]
        )

    def highlight_max(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Axis | None = 0,
        props: str | None = None,
    ) -> Styler:
        """
        Highlight the maximum with a style.

        Parameters
        ----------
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        color : str, default 'yellow'
            Background color to use for highlighting.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        props : str, default None
            CSS properties to use for highlighting. If ``props`` is given, ``color``
            is not used.

            .. versionadded:: 1.3.0

        Returns
        -------
        self : Styler

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_between: Highlight a defined range with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.
        """

        if props is None:
            props = f"background-color: {color};"
        # error: Argument 1 to "apply" of "Styler" has incompatible type
        # "Callable[[FrameOrSeries, str], ndarray]"; expected "Callable[..., Styler]"
        return self.apply(
            partial(_highlight_value, op="max"),  # type: ignore[arg-type]
            axis=axis,
            subset=subset,
            props=props,
        )

    def highlight_min(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Axis | None = 0,
        props: str | None = None,
    ) -> Styler:
        """
        Highlight the minimum with a style.

        Parameters
        ----------
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        color : str, default 'yellow'
            Background color to use for highlighting.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Apply to each column (``axis=0`` or ``'index'``), to each row
            (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
            with ``axis=None``.
        props : str, default None
            CSS properties to use for highlighting. If ``props`` is given, ``color``
            is not used.

            .. versionadded:: 1.3.0

        Returns
        -------
        self : Styler

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_between: Highlight a defined range with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.
        """

        if props is None:
            props = f"background-color: {color};"
        # error: Argument 1 to "apply" of "Styler" has incompatible type
        # "Callable[[FrameOrSeries, str], ndarray]"; expected "Callable[..., Styler]"
        return self.apply(
            partial(_highlight_value, op="min"),  # type: ignore[arg-type]
            axis=axis,
            subset=subset,
            props=props,
        )

    def highlight_between(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Axis | None = 0,
        left: Scalar | Sequence | None = None,
        right: Scalar | Sequence | None = None,
        inclusive: str = "both",
        props: str | None = None,
    ) -> Styler:
        """
        Highlight a defined range with a style.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        color : str, default 'yellow'
            Background color to use for highlighting.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            If ``left`` or ``right`` given as sequence, axis along which to apply those
            boundaries. See examples.
        left : scalar or datetime-like, or sequence or array-like, default None
            Left bound for defining the range.
        right : scalar or datetime-like, or sequence or array-like, default None
            Right bound for defining the range.
        inclusive : {'both', 'neither', 'left', 'right'}
            Identify whether bounds are closed or open.
        props : str, default None
            CSS properties to use for highlighting. If ``props`` is given, ``color``
            is not used.

        Returns
        -------
        self : Styler

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_quantile: Highlight values defined by a quantile with a style.

        Notes
        -----
        If ``left`` is ``None`` only the right bound is applied.
        If ``right`` is ``None`` only the left bound is applied. If both are ``None``
        all values are highlighted.

        ``axis`` is only needed if ``left`` or ``right`` are provided as a sequence or
        an array-like object for aligning the shapes. If ``left`` and ``right`` are
        both scalars then all ``axis`` inputs will give the same result.

        This function only works with compatible ``dtypes``. For example a datetime-like
        region can only use equivalent datetime-like ``left`` and ``right`` arguments.
        Use ``subset`` to control regions which have multiple ``dtypes``.

        Examples
        --------
        Basic usage

        >>> df = pd.DataFrame({
        ...     'One': [1.2, 1.6, 1.5],
        ...     'Two': [2.9, 2.1, 2.5],
        ...     'Three': [3.1, 3.2, 3.8],
        ... })
        >>> df.style.highlight_between(left=2.1, right=2.9)

        .. figure:: ../../_static/style/hbetw_basic.png

        Using a range input sequnce along an ``axis``, in this case setting a ``left``
        and ``right`` for each column individually

        >>> df.style.highlight_between(left=[1.4, 2.4, 3.4], right=[1.6, 2.6, 3.6],
        ...     axis=1, color="#fffd75")

        .. figure:: ../../_static/style/hbetw_seq.png

        Using ``axis=None`` and providing the ``left`` argument as an array that
        matches the input DataFrame, with a constant ``right``

        >>> df.style.highlight_between(left=[[2,2,3],[2,2,3],[3,3,3]], right=3.5,
        ...     axis=None, color="#fffd75")

        .. figure:: ../../_static/style/hbetw_axNone.png

        Using ``props`` instead of default background coloring

        >>> df.style.highlight_between(left=1.5, right=3.5,
        ...     props='font-weight:bold;color:#e83e8c')

        .. figure:: ../../_static/style/hbetw_props.png
        """
        if props is None:
            props = f"background-color: {color};"
        return self.apply(
            _highlight_between,  # type: ignore[arg-type]
            axis=axis,
            subset=subset,
            props=props,
            left=left,
            right=right,
            inclusive=inclusive,
        )

    def highlight_quantile(
        self,
        subset: Subset | None = None,
        color: str = "yellow",
        axis: Axis | None = 0,
        q_left: float = 0.0,
        q_right: float = 1.0,
        interpolation: str = "linear",
        inclusive: str = "both",
        props: str | None = None,
    ) -> Styler:
        """
        Highlight values defined by a quantile with a style.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        color : str, default 'yellow'
            Background color to use for highlighting
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Axis along which to determine and highlight quantiles. If ``None`` quantiles
            are measured over the entire DataFrame. See examples.
        q_left : float, default 0
            Left bound, in [0, q_right), for the target quantile range.
        q_right : float, default 1
            Right bound, in (q_left, 1], for the target quantile range.
        interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
            Argument passed to ``Series.quantile`` or ``DataFrame.quantile`` for
            quantile estimation.
        inclusive : {'both', 'neither', 'left', 'right'}
            Identify whether quantile bounds are closed or open.
        props : str, default None
            CSS properties to use for highlighting. If ``props`` is given, ``color``
            is not used.

        Returns
        -------
        self : Styler

        See Also
        --------
        Styler.highlight_null: Highlight missing values with a style.
        Styler.highlight_max: Highlight the maximum with a style.
        Styler.highlight_min: Highlight the minimum with a style.
        Styler.highlight_between: Highlight a defined range with a style.

        Notes
        -----
        This function does not work with ``str`` dtypes.

        Examples
        --------
        Using ``axis=None`` and apply a quantile to all collective data

        >>> df = pd.DataFrame(np.arange(10).reshape(2,5) + 1)
        >>> df.style.highlight_quantile(axis=None, q_left=0.8, color="#fffd75")

        .. figure:: ../../_static/style/hq_axNone.png

        Or highlight quantiles row-wise or column-wise, in this case by row-wise

        >>> df.style.highlight_quantile(axis=1, q_left=0.8, color="#fffd75")

        .. figure:: ../../_static/style/hq_ax1.png

        Use ``props`` instead of default background coloring

        >>> df.style.highlight_quantile(axis=None, q_left=0.2, q_right=0.8,
        ...     props='font-weight:bold;color:#e83e8c')

        .. figure:: ../../_static/style/hq_props.png
        """
        subset_ = slice(None) if subset is None else subset
        subset_ = non_reducing_slice(subset_)
        data = self.data.loc[subset_]

        # after quantile is found along axis, e.g. along rows,
        # applying the calculated quantile to alternate axis, e.g. to each column
        kwargs = {"q": [q_left, q_right], "interpolation": interpolation}
        if axis in [0, "index"]:
            q = data.quantile(axis=axis, numeric_only=False, **kwargs)
            axis_apply: int | None = 1
        elif axis in [1, "columns"]:
            q = data.quantile(axis=axis, numeric_only=False, **kwargs)
            axis_apply = 0
        else:  # axis is None
            q = Series(data.to_numpy().ravel()).quantile(**kwargs)
            axis_apply = None

        if props is None:
            props = f"background-color: {color};"
        return self.apply(
            _highlight_between,  # type: ignore[arg-type]
            axis=axis_apply,
            subset=subset,
            props=props,
            left=q.iloc[0],
            right=q.iloc[1],
            inclusive=inclusive,
        )

    @classmethod
    def from_custom_template(
        cls, searchpath, html_table: str | None = None, html_style: str | None = None
    ):
        """
        Factory function for creating a subclass of ``Styler``.

        Uses custom templates and Jinja environment.

        .. versionchanged:: 1.3.0

        Parameters
        ----------
        searchpath : str or list
            Path or paths of directories containing the templates.
        html_table : str
            Name of your custom template to replace the html_table template.

            .. versionadded:: 1.3.0

        html_style : str
            Name of your custom template to replace the html_style template.

            .. versionadded:: 1.3.0

        Returns
        -------
        MyStyler : subclass of Styler
            Has the correct ``env``,``template_html``, ``template_html_table`` and
            ``template_html_style`` class attributes set.
        """
        loader = jinja2.ChoiceLoader([jinja2.FileSystemLoader(searchpath), cls.loader])

        # mypy doesn't like dynamically-defined classes
        # error: Variable "cls" is not valid as a type
        # error: Invalid base class "cls"
        class MyStyler(cls):  # type:ignore[valid-type,misc]
            env = jinja2.Environment(loader=loader)
            if html_table:
                template_html_table = env.get_template(html_table)
            if html_style:
                template_html_style = env.get_template(html_style)

        return MyStyler

    def pipe(self, func: Callable, *args, **kwargs):
        """
        Apply ``func(self, *args, **kwargs)``, and return the result.

        Parameters
        ----------
        func : function
            Function to apply to the Styler.  Alternatively, a
            ``(callable, keyword)`` tuple where ``keyword`` is a string
            indicating the keyword of ``callable`` that expects the Styler.
        *args : optional
            Arguments passed to `func`.
        **kwargs : optional
            A dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object :
            The value returned by ``func``.

        See Also
        --------
        DataFrame.pipe : Analogous method for DataFrame.
        Styler.apply : Apply a CSS-styling function column-wise, row-wise, or
            table-wise.

        Notes
        -----
        Like :meth:`DataFrame.pipe`, this method can simplify the
        application of several user-defined functions to a styler.  Instead
        of writing:

        .. code-block:: python

            f(g(df.style.set_precision(3), arg1=a), arg2=b, arg3=c)

        users can write:

        .. code-block:: python

            (df.style.set_precision(3)
               .pipe(g, arg1=a)
               .pipe(f, arg2=b, arg3=c))

        In particular, this allows users to define functions that take a
        styler object, along with other parameters, and return the styler after
        making styling changes (such as calling :meth:`Styler.apply` or
        :meth:`Styler.set_properties`).  Using ``.pipe``, these user-defined
        style "transformations" can be interleaved with calls to the built-in
        Styler interface.

        Examples
        --------
        >>> def format_conversion(styler):
        ...     return (styler.set_properties(**{'text-align': 'right'})
        ...                   .format({'conversion': '{:.1%}'}))

        The user-defined ``format_conversion`` function above can be called
        within a sequence of other style modifications:

        >>> df = pd.DataFrame({'trial': list(range(5)),
        ...                    'conversion': [0.75, 0.85, np.nan, 0.7, 0.72]})
        >>> (df.style
        ...    .highlight_min(subset=['conversion'], color='yellow')
        ...    .pipe(format_conversion)
        ...    .set_caption("Results with minimum conversion highlighted."))
        """
        return com.pipe(self, func, *args, **kwargs)


def _validate_apply_axis_arg(
    arg: FrameOrSeries | Sequence | np.ndarray,
    arg_name: str,
    dtype: Any | None,
    data: FrameOrSeries,
) -> np.ndarray:
    """
    For the apply-type methods, ``axis=None`` creates ``data`` as DataFrame, and for
    ``axis=[1,0]`` it creates a Series. Where ``arg`` is expected as an element
    of some operator with ``data`` we must make sure that the two are compatible shapes,
    or raise.

    Parameters
    ----------
    arg : sequence, Series or DataFrame
        the user input arg
    arg_name : string
        name of the arg for use in error messages
    dtype : numpy dtype, optional
        forced numpy dtype if given
    data : Series or DataFrame
        underling subset of Styler data on which operations are performed

    Returns
    -------
    ndarray
    """
    dtype = {"dtype": dtype} if dtype else {}
    # raise if input is wrong for axis:
    if isinstance(arg, Series) and isinstance(data, DataFrame):
        raise ValueError(
            f"'{arg_name}' is a Series but underlying data for operations "
            f"is a DataFrame since 'axis=None'"
        )
    elif isinstance(arg, DataFrame) and isinstance(data, Series):
        raise ValueError(
            f"'{arg_name}' is a DataFrame but underlying data for "
            f"operations is a Series with 'axis in [0,1]'"
        )
    elif isinstance(arg, (Series, DataFrame)):  # align indx / cols to data
        arg = arg.reindex_like(data, method=None).to_numpy(**dtype)
    else:
        arg = np.asarray(arg, **dtype)
        assert isinstance(arg, np.ndarray)  # mypy requirement
        if arg.shape != data.shape:  # check valid input
            raise ValueError(
                f"supplied '{arg_name}' is not correct shape for data over "
                f"selected 'axis': got {arg.shape}, "
                f"expected {data.shape}"
            )
    return arg


def _background_gradient(
    data,
    cmap="PuBu",
    low: float = 0,
    high: float = 0,
    text_color_threshold: float = 0.408,
    vmin: float | None = None,
    vmax: float | None = None,
    gmap: Sequence | np.ndarray | FrameOrSeries | None = None,
    text_only: bool = False,
):
    """
    Color background in a range according to the data or a gradient map
    """
    if gmap is None:  # the data is used the gmap
        gmap = data.to_numpy(dtype=float)
    else:  # else validate gmap against the underlying data
        gmap = _validate_apply_axis_arg(gmap, "gmap", float, data)

    with _mpl(Styler.background_gradient) as (plt, colors):
        smin = np.nanmin(gmap) if vmin is None else vmin
        smax = np.nanmax(gmap) if vmax is None else vmax
        rng = smax - smin
        # extend lower / upper bounds, compresses color range
        norm = colors.Normalize(smin - (rng * low), smax + (rng * high))
        rgbas = plt.cm.get_cmap(cmap)(norm(gmap))

        def relative_luminance(rgba) -> float:
            """
            Calculate relative luminance of a color.

            The calculation adheres to the W3C standards
            (https://www.w3.org/WAI/GL/wiki/Relative_luminance)

            Parameters
            ----------
            color : rgb or rgba tuple

            Returns
            -------
            float
                The relative luminance as a value from 0 to 1
            """
            r, g, b = (
                x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
                for x in rgba[:3]
            )
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        def css(rgba, text_only) -> str:
            if not text_only:
                dark = relative_luminance(rgba) < text_color_threshold
                text_color = "#f1f1f1" if dark else "#000000"
                return f"background-color: {colors.rgb2hex(rgba)};color: {text_color};"
            else:
                return f"color: {colors.rgb2hex(rgba)};"

        if data.ndim == 1:
            return [css(rgba, text_only) for rgba in rgbas]
        else:
            return DataFrame(
                [[css(rgba, text_only) for rgba in row] for row in rgbas],
                index=data.index,
                columns=data.columns,
            )


def _highlight_between(
    data: FrameOrSeries,
    props: str,
    left: Scalar | Sequence | np.ndarray | FrameOrSeries | None = None,
    right: Scalar | Sequence | np.ndarray | FrameOrSeries | None = None,
    inclusive: bool | str = True,
) -> np.ndarray:
    """
    Return an array of css props based on condition of data values within given range.
    """
    if np.iterable(left) and not isinstance(left, str):
        left = _validate_apply_axis_arg(
            left, "left", None, data  # type: ignore[arg-type]
        )

    if np.iterable(right) and not isinstance(right, str):
        right = _validate_apply_axis_arg(
            right, "right", None, data  # type: ignore[arg-type]
        )

    # get ops with correct boundary attribution
    if inclusive == "both":
        ops = (operator.ge, operator.le)
    elif inclusive == "neither":
        ops = (operator.gt, operator.lt)
    elif inclusive == "left":
        ops = (operator.ge, operator.lt)
    elif inclusive == "right":
        ops = (operator.gt, operator.le)
    else:
        raise ValueError(
            f"'inclusive' values can be 'both', 'left', 'right', or 'neither' "
            f"got {inclusive}"
        )

    g_left = (
        ops[0](data, left)
        if left is not None
        else np.full(data.shape, True, dtype=bool)
    )
    l_right = (
        ops[1](data, right)
        if right is not None
        else np.full(data.shape, True, dtype=bool)
    )
    return np.where(g_left & l_right, props, "")


def _highlight_value(data: FrameOrSeries, op: str, props: str) -> np.ndarray:
    """
    Return an array of css strings based on the condition of values matching an op.
    """
    value = getattr(data, op)(skipna=True)
    if isinstance(data, DataFrame):  # min/max must be done twice to return scalar
        value = getattr(value, op)(skipna=True)
    return np.where(data == value, props, "")
