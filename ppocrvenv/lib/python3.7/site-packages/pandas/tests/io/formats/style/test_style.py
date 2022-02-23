import copy
import re

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
)
import pandas._testing as tm

jinja2 = pytest.importorskip("jinja2")
from pandas.io.formats.style import (  # isort:skip
    Styler,
)
from pandas.io.formats.style_render import (
    _get_level_lengths,
    _get_trimming_maximums,
    maybe_convert_css_to_tuples,
    non_reducing_slice,
)


@pytest.fixture
def mi_df():
    return DataFrame(
        [[1, 2], [3, 4]],
        index=MultiIndex.from_product([["i0"], ["i1_a", "i1_b"]]),
        columns=MultiIndex.from_product([["c0"], ["c1_a", "c1_b"]]),
        dtype=int,
    )


@pytest.fixture
def mi_styler(mi_df):
    return Styler(mi_df, uuid_len=0)


@pytest.fixture
def mi_styler_comp(mi_styler):
    # comprehensively add features to mi_styler
    mi_styler.uuid_len = 5
    mi_styler.uuid = "abcde_"
    mi_styler.set_caption("capt")
    mi_styler.set_table_styles([{"selector": "a", "props": "a:v;"}])
    mi_styler.hide_columns()
    mi_styler.hide_columns([("c0", "c1_a")])
    mi_styler.hide_index()
    mi_styler.hide_index([("i0", "i1_a")])
    mi_styler.set_table_attributes('class="box"')
    mi_styler.format(na_rep="MISSING", precision=3)
    mi_styler.highlight_max(axis=None)
    mi_styler.set_td_classes(
        DataFrame(
            [["a", "b"], ["a", "c"]], index=mi_styler.index, columns=mi_styler.columns
        )
    )
    mi_styler.set_tooltips(
        DataFrame(
            [["a2", "b2"], ["a2", "c2"]],
            index=mi_styler.index,
            columns=mi_styler.columns,
        )
    )
    return mi_styler


@pytest.mark.parametrize(
    "sparse_columns, exp_cols",
    [
        (
            True,
            [
                {"is_visible": True, "attributes": 'colspan="2"', "value": "c0"},
                {"is_visible": False, "attributes": "", "value": "c0"},
            ],
        ),
        (
            False,
            [
                {"is_visible": True, "attributes": "", "value": "c0"},
                {"is_visible": True, "attributes": "", "value": "c0"},
            ],
        ),
    ],
)
def test_mi_styler_sparsify_columns(mi_styler, sparse_columns, exp_cols):
    exp_l1_c0 = {"is_visible": True, "attributes": "", "display_value": "c1_a"}
    exp_l1_c1 = {"is_visible": True, "attributes": "", "display_value": "c1_b"}

    ctx = mi_styler._translate(True, sparse_columns)

    assert exp_cols[0].items() <= ctx["head"][0][2].items()
    assert exp_cols[1].items() <= ctx["head"][0][3].items()
    assert exp_l1_c0.items() <= ctx["head"][1][2].items()
    assert exp_l1_c1.items() <= ctx["head"][1][3].items()


@pytest.mark.parametrize(
    "sparse_index, exp_rows",
    [
        (
            True,
            [
                {"is_visible": True, "attributes": 'rowspan="2"', "value": "i0"},
                {"is_visible": False, "attributes": "", "value": "i0"},
            ],
        ),
        (
            False,
            [
                {"is_visible": True, "attributes": "", "value": "i0"},
                {"is_visible": True, "attributes": "", "value": "i0"},
            ],
        ),
    ],
)
def test_mi_styler_sparsify_index(mi_styler, sparse_index, exp_rows):
    exp_l1_r0 = {"is_visible": True, "attributes": "", "display_value": "i1_a"}
    exp_l1_r1 = {"is_visible": True, "attributes": "", "display_value": "i1_b"}

    ctx = mi_styler._translate(sparse_index, True)

    assert exp_rows[0].items() <= ctx["body"][0][0].items()
    assert exp_rows[1].items() <= ctx["body"][1][0].items()
    assert exp_l1_r0.items() <= ctx["body"][0][1].items()
    assert exp_l1_r1.items() <= ctx["body"][1][1].items()


def test_mi_styler_sparsify_options(mi_styler):
    with pd.option_context("styler.sparse.index", False):
        html1 = mi_styler.render()
    with pd.option_context("styler.sparse.index", True):
        html2 = mi_styler.render()

    assert html1 != html2

    with pd.option_context("styler.sparse.columns", False):
        html1 = mi_styler.render()
    with pd.option_context("styler.sparse.columns", True):
        html2 = mi_styler.render()

    assert html1 != html2


def test_trimming_maximum():
    rn, cn = _get_trimming_maximums(100, 100, 100, scaling_factor=0.5)
    assert (rn, cn) == (12, 6)

    rn, cn = _get_trimming_maximums(1000, 3, 750, scaling_factor=0.5)
    assert (rn, cn) == (250, 3)


def test_render_trimming():
    df = DataFrame(np.arange(120).reshape(60, 2))
    with pd.option_context("styler.render.max_elements", 6):
        ctx = df.style._translate(True, True)
    assert len(ctx["head"][0]) == 3  # index + 2 data cols
    assert len(ctx["body"]) == 4  # 3 data rows + trimming row
    assert len(ctx["body"][0]) == 3  # index + 2 data cols

    df = DataFrame(np.arange(120).reshape(12, 10))
    with pd.option_context("styler.render.max_elements", 6):
        ctx = df.style._translate(True, True)
    assert len(ctx["head"][0]) == 4  # index + 2 data cols + trimming row
    assert len(ctx["body"]) == 4  # 3 data rows + trimming row
    assert len(ctx["body"][0]) == 4  # index + 2 data cols + trimming row


def test_render_trimming_mi():
    midx = MultiIndex.from_product([[1, 2], [1, 2, 3]])
    df = DataFrame(np.arange(36).reshape(6, 6), columns=midx, index=midx)
    with pd.option_context("styler.render.max_elements", 4):
        ctx = df.style._translate(True, True)

    assert len(ctx["body"][0]) == 5  # 2 indexes + 2 data cols + trimming row
    assert {"attributes": 'rowspan="2"'}.items() <= ctx["body"][0][0].items()
    assert {"class": "data row0 col_trim"}.items() <= ctx["body"][0][4].items()
    assert {"class": "data row_trim col_trim"}.items() <= ctx["body"][2][4].items()
    assert len(ctx["body"]) == 3  # 2 data rows + trimming row

    assert len(ctx["head"][0]) == 5  # 2 indexes + 2 column headers + trimming col
    assert {"attributes": 'colspan="2"'}.items() <= ctx["head"][0][2].items()


@pytest.mark.parametrize("comprehensive", [True, False])
@pytest.mark.parametrize("render", [True, False])
@pytest.mark.parametrize("deepcopy", [True, False])
def test_copy(comprehensive, render, deepcopy, mi_styler, mi_styler_comp):
    styler = mi_styler_comp if comprehensive else mi_styler
    styler.uuid_len = 5

    s2 = copy.deepcopy(styler) if deepcopy else copy.copy(styler)  # make copy and check
    assert s2 is not styler

    if render:
        styler.to_html()

    excl = ["na_rep", "precision", "uuid", "cellstyle_map"]  # deprecated or special var
    if not deepcopy:  # check memory locations are equal for all included attributes
        for attr in [a for a in styler.__dict__ if (not callable(a) and a not in excl)]:
            assert id(getattr(s2, attr)) == id(getattr(styler, attr))
    else:  # check memory locations are different for nested or mutable vars
        shallow = [
            "data",
            "columns",
            "index",
            "uuid_len",
            "caption",
            "cell_ids",
            "hide_index_",
            "hide_columns_",
            "table_attributes",
        ]
        for attr in shallow:
            assert id(getattr(s2, attr)) == id(getattr(styler, attr))

        for attr in [
            a
            for a in styler.__dict__
            if (not callable(a) and a not in excl and a not in shallow)
        ]:
            if getattr(s2, attr) is None:
                assert id(getattr(s2, attr)) == id(getattr(styler, attr))
            else:
                assert id(getattr(s2, attr)) != id(getattr(styler, attr))


class TestStyler:
    def setup_method(self, method):
        np.random.seed(24)
        self.s = DataFrame({"A": np.random.permutation(range(6))})
        self.df = DataFrame({"A": [0, 1], "B": np.random.randn(2)})
        self.f = lambda x: x
        self.g = lambda x: x

        def h(x, foo="bar"):
            return pd.Series(f"color: {foo}", index=x.index, name=x.name)

        self.h = h
        self.styler = Styler(self.df)
        self.attrs = DataFrame({"A": ["color: red", "color: blue"]})
        self.dataframes = [
            self.df,
            DataFrame(
                {"f": [1.0, 2.0], "o": ["a", "b"], "c": pd.Categorical(["a", "b"])}
            ),
        ]
        self.blank_value = "&nbsp;"

    def test_init_non_pandas(self):
        msg = "``data`` must be a Series or DataFrame"
        with pytest.raises(TypeError, match=msg):
            Styler([1, 2, 3])

    def test_init_series(self):
        result = Styler(pd.Series([1, 2]))
        assert result.data.ndim == 2

    def test_repr_html_ok(self):
        self.styler._repr_html_()

    def test_repr_html_mathjax(self):
        # gh-19824
        assert "tex2jax_ignore" not in self.styler._repr_html_()

        with pd.option_context("display.html.use_mathjax", False):
            assert "tex2jax_ignore" in self.styler._repr_html_()

    def test_update_ctx(self):
        self.styler._update_ctx(self.attrs)
        expected = {(0, 0): [("color", "red")], (1, 0): [("color", "blue")]}
        assert self.styler.ctx == expected

    def test_update_ctx_flatten_multi_and_trailing_semi(self):
        attrs = DataFrame({"A": ["color: red; foo: bar", "color:blue ; foo: baz;"]})
        self.styler._update_ctx(attrs)
        expected = {
            (0, 0): [("color", "red"), ("foo", "bar")],
            (1, 0): [("color", "blue"), ("foo", "baz")],
        }
        assert self.styler.ctx == expected

    def test_clear(self):
        # updated in GH 39396
        tt = DataFrame({"A": [None, "tt"]})
        css = DataFrame({"A": [None, "cls-a"]})
        s = self.df.style.highlight_max().set_tooltips(tt).set_td_classes(css)
        s = s.hide_index().hide_columns("A")
        # _todo, tooltips and cell_context items added to..
        assert len(s._todo) > 0
        assert s.tooltips
        assert len(s.cell_context) > 0
        assert s.hide_index_ is True
        assert len(s.hidden_columns) > 0

        s = s._compute()
        # ctx item affected when a render takes place. _todo is maintained
        assert len(s.ctx) > 0
        assert len(s._todo) > 0

        s.clear()
        # ctx, _todo, tooltips and cell_context items all revert to null state.
        assert len(s.ctx) == 0
        assert len(s._todo) == 0
        assert not s.tooltips
        assert len(s.cell_context) == 0
        assert s.hide_index_ is False
        assert len(s.hidden_columns) == 0

    def test_render(self):
        df = DataFrame({"A": [0, 1]})
        style = lambda x: pd.Series(["color: red", "color: blue"], name=x.name)
        s = Styler(df, uuid="AB").apply(style)
        s.render()
        # it worked?

    def test_multiple_render(self):
        # GH 39396
        s = Styler(self.df, uuid_len=0).applymap(lambda x: "color: red;", subset=["A"])
        s.render()  # do 2 renders to ensure css styles not duplicated
        assert (
            '<style type="text/css">\n#T__row0_col0, #T__row1_col0 {\n'
            "  color: red;\n}\n</style>" in s.render()
        )

    def test_render_empty_dfs(self):
        empty_df = DataFrame()
        es = Styler(empty_df)
        es.render()
        # An index but no columns
        DataFrame(columns=["a"]).style.render()
        # A column but no index
        DataFrame(index=["a"]).style.render()
        # No IndexError raised?

    def test_render_double(self):
        df = DataFrame({"A": [0, 1]})
        style = lambda x: pd.Series(
            ["color: red; border: 1px", "color: blue; border: 2px"], name=x.name
        )
        s = Styler(df, uuid="AB").apply(style)
        s.render()
        # it worked?

    def test_set_properties(self):
        df = DataFrame({"A": [0, 1]})
        result = df.style.set_properties(color="white", size="10px")._compute().ctx
        # order is deterministic
        v = [("color", "white"), ("size", "10px")]
        expected = {(0, 0): v, (1, 0): v}
        assert result.keys() == expected.keys()
        for v1, v2 in zip(result.values(), expected.values()):
            assert sorted(v1) == sorted(v2)

    def test_set_properties_subset(self):
        df = DataFrame({"A": [0, 1]})
        result = (
            df.style.set_properties(subset=pd.IndexSlice[0, "A"], color="white")
            ._compute()
            .ctx
        )
        expected = {(0, 0): [("color", "white")]}
        assert result == expected

    def test_empty_index_name_doesnt_display(self):
        # https://github.com/pandas-dev/pandas/pull/12090#issuecomment-180695902
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        result = df.style._translate(True, True)

        expected = [
            [
                {
                    "class": "blank level0",
                    "type": "th",
                    "value": self.blank_value,
                    "is_visible": True,
                    "display_value": self.blank_value,
                },
                {
                    "class": "col_heading level0 col0",
                    "display_value": "A",
                    "type": "th",
                    "value": "A",
                    "is_visible": True,
                    "attributes": "",
                },
                {
                    "class": "col_heading level0 col1",
                    "display_value": "B",
                    "type": "th",
                    "value": "B",
                    "is_visible": True,
                    "attributes": "",
                },
                {
                    "class": "col_heading level0 col2",
                    "display_value": "C",
                    "type": "th",
                    "value": "C",
                    "is_visible": True,
                    "attributes": "",
                },
            ]
        ]

        assert result["head"] == expected

    def test_index_name(self):
        # https://github.com/pandas-dev/pandas/issues/11655
        # TODO: this test can be minimised to address the test more directly
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        result = df.set_index("A").style._translate(True, True)

        expected = [
            [
                {
                    "class": "blank level0",
                    "type": "th",
                    "value": self.blank_value,
                    "display_value": self.blank_value,
                    "is_visible": True,
                },
                {
                    "class": "col_heading level0 col0",
                    "type": "th",
                    "value": "B",
                    "display_value": "B",
                    "is_visible": True,
                    "attributes": "",
                },
                {
                    "class": "col_heading level0 col1",
                    "type": "th",
                    "value": "C",
                    "display_value": "C",
                    "is_visible": True,
                    "attributes": "",
                },
            ],
            [
                {
                    "class": "index_name level0",
                    "type": "th",
                    "value": "A",
                    "is_visible": True,
                    "display_value": "A",
                },
                {
                    "class": "blank col0",
                    "type": "th",
                    "value": self.blank_value,
                    "is_visible": True,
                    "display_value": self.blank_value,
                },
                {
                    "class": "blank col1",
                    "type": "th",
                    "value": self.blank_value,
                    "is_visible": True,
                    "display_value": self.blank_value,
                },
            ],
        ]

        assert result["head"] == expected

    def test_multiindex_name(self):
        # https://github.com/pandas-dev/pandas/issues/11655
        # TODO: this test can be minimised to address the test more directly
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        result = df.set_index(["A", "B"]).style._translate(True, True)

        expected = [
            [
                {
                    "class": "blank",
                    "type": "th",
                    "value": self.blank_value,
                    "display_value": self.blank_value,
                    "is_visible": True,
                },
                {
                    "class": "blank level0",
                    "type": "th",
                    "value": self.blank_value,
                    "display_value": self.blank_value,
                    "is_visible": True,
                },
                {
                    "class": "col_heading level0 col0",
                    "type": "th",
                    "value": "C",
                    "display_value": "C",
                    "is_visible": True,
                    "attributes": "",
                },
            ],
            [
                {
                    "class": "index_name level0",
                    "type": "th",
                    "value": "A",
                    "is_visible": True,
                    "display_value": "A",
                },
                {
                    "class": "index_name level1",
                    "type": "th",
                    "value": "B",
                    "is_visible": True,
                    "display_value": "B",
                },
                {
                    "class": "blank col0",
                    "type": "th",
                    "value": self.blank_value,
                    "is_visible": True,
                    "display_value": self.blank_value,
                },
            ],
        ]

        assert result["head"] == expected

    def test_numeric_columns(self):
        # https://github.com/pandas-dev/pandas/issues/12125
        # smoke test for _translate
        df = DataFrame({0: [1, 2, 3]})
        df.style._translate(True, True)

    def test_apply_axis(self):
        df = DataFrame({"A": [0, 0], "B": [1, 1]})
        f = lambda x: [f"val: {x.max()}" for v in x]
        result = df.style.apply(f, axis=1)
        assert len(result._todo) == 1
        assert len(result.ctx) == 0
        result._compute()
        expected = {
            (0, 0): [("val", "1")],
            (0, 1): [("val", "1")],
            (1, 0): [("val", "1")],
            (1, 1): [("val", "1")],
        }
        assert result.ctx == expected

        result = df.style.apply(f, axis=0)
        expected = {
            (0, 0): [("val", "0")],
            (0, 1): [("val", "1")],
            (1, 0): [("val", "0")],
            (1, 1): [("val", "1")],
        }
        result._compute()
        assert result.ctx == expected
        result = df.style.apply(f)  # default
        result._compute()
        assert result.ctx == expected

    @pytest.mark.parametrize(
        "slice_",
        [
            pd.IndexSlice[:],
            pd.IndexSlice[:, ["A"]],
            pd.IndexSlice[[1], :],
            pd.IndexSlice[[1], ["A"]],
            pd.IndexSlice[:2, ["A", "B"]],
        ],
    )
    @pytest.mark.parametrize("axis", [0, 1])
    def test_apply_subset(self, slice_, axis):
        result = (
            self.df.style.apply(self.h, axis=axis, subset=slice_, foo="baz")
            ._compute()
            .ctx
        )
        expected = {
            (r, c): [("color", "baz")]
            for r, row in enumerate(self.df.index)
            for c, col in enumerate(self.df.columns)
            if row in self.df.loc[slice_].index and col in self.df.loc[slice_].columns
        }
        assert result == expected

    @pytest.mark.parametrize(
        "slice_",
        [
            pd.IndexSlice[:],
            pd.IndexSlice[:, ["A"]],
            pd.IndexSlice[[1], :],
            pd.IndexSlice[[1], ["A"]],
            pd.IndexSlice[:2, ["A", "B"]],
        ],
    )
    def test_applymap_subset(self, slice_):
        result = (
            self.df.style.applymap(lambda x: "color:baz;", subset=slice_)._compute().ctx
        )
        expected = {
            (r, c): [("color", "baz")]
            for r, row in enumerate(self.df.index)
            for c, col in enumerate(self.df.columns)
            if row in self.df.loc[slice_].index and col in self.df.loc[slice_].columns
        }
        assert result == expected

    @pytest.mark.parametrize(
        "slice_",
        [
            pd.IndexSlice[:, pd.IndexSlice["x", "A"]],
            pd.IndexSlice[:, pd.IndexSlice[:, "A"]],
            pd.IndexSlice[:, pd.IndexSlice[:, ["A", "C"]]],  # missing col element
            pd.IndexSlice[pd.IndexSlice["a", 1], :],
            pd.IndexSlice[pd.IndexSlice[:, 1], :],
            pd.IndexSlice[pd.IndexSlice[:, [1, 3]], :],  # missing row element
            pd.IndexSlice[:, ("x", "A")],
            pd.IndexSlice[("a", 1), :],
        ],
    )
    def test_applymap_subset_multiindex(self, slice_):
        # GH 19861
        # edited for GH 33562
        idx = MultiIndex.from_product([["a", "b"], [1, 2]])
        col = MultiIndex.from_product([["x", "y"], ["A", "B"]])
        df = DataFrame(np.random.rand(4, 4), columns=col, index=idx)
        df.style.applymap(lambda x: "color: red;", subset=slice_).render()

    def test_applymap_subset_multiindex_code(self):
        # https://github.com/pandas-dev/pandas/issues/25858
        # Checks styler.applymap works with multindex when codes are provided
        codes = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
        columns = MultiIndex(
            levels=[["a", "b"], ["%", "#"]], codes=codes, names=["", ""]
        )
        df = DataFrame(
            [[1, -1, 1, 1], [-1, 1, 1, 1]], index=["hello", "world"], columns=columns
        )
        pct_subset = pd.IndexSlice[:, pd.IndexSlice[:, "%":"%"]]

        def color_negative_red(val):
            color = "red" if val < 0 else "black"
            return f"color: {color}"

        df.loc[pct_subset]
        df.style.applymap(color_negative_red, subset=pct_subset)

    def test_where_with_one_style(self):
        # GH 17474
        def f(x):
            return x > 0.5

        style1 = "foo: bar"

        with tm.assert_produces_warning(FutureWarning):
            result = self.df.style.where(f, style1)._compute().ctx
        expected = {
            (r, c): [("foo", "bar")]
            for r, row in enumerate(self.df.index)
            for c, col in enumerate(self.df.columns)
            if f(self.df.loc[row, col])
        }
        assert result == expected

    @pytest.mark.parametrize(
        "slice_",
        [
            pd.IndexSlice[:],
            pd.IndexSlice[:, ["A"]],
            pd.IndexSlice[[1], :],
            pd.IndexSlice[[1], ["A"]],
            pd.IndexSlice[:2, ["A", "B"]],
        ],
    )
    def test_where_subset(self, slice_):
        # GH 17474
        def f(x):
            return x > 0.5

        style1 = "foo: bar"
        style2 = "baz: foo"

        with tm.assert_produces_warning(FutureWarning):
            res = self.df.style.where(f, style1, style2, subset=slice_)._compute().ctx
        expected = {
            (r, c): [("foo", "bar") if f(self.df.loc[row, col]) else ("baz", "foo")]
            for r, row in enumerate(self.df.index)
            for c, col in enumerate(self.df.columns)
            if row in self.df.loc[slice_].index and col in self.df.loc[slice_].columns
        }
        assert res == expected

    def test_where_subset_compare_with_applymap(self):
        # GH 17474
        def f(x):
            return x > 0.5

        style1 = "foo: bar"
        style2 = "baz: foo"

        def g(x):
            return style1 if f(x) else style2

        slices = [
            pd.IndexSlice[:],
            pd.IndexSlice[:, ["A"]],
            pd.IndexSlice[[1], :],
            pd.IndexSlice[[1], ["A"]],
            pd.IndexSlice[:2, ["A", "B"]],
        ]

        for slice_ in slices:
            with tm.assert_produces_warning(FutureWarning):
                result = (
                    self.df.style.where(f, style1, style2, subset=slice_)._compute().ctx
                )
            expected = self.df.style.applymap(g, subset=slice_)._compute().ctx
            assert result == expected

    def test_where_kwargs(self):
        df = DataFrame([[1, 2], [3, 4]])

        def f(x, val):
            return x > val

        with tm.assert_produces_warning(FutureWarning):
            res = df.style.where(f, "color:green;", "color:red;", val=2)._compute().ctx
        expected = {
            (0, 0): [("color", "red")],
            (0, 1): [("color", "red")],
            (1, 0): [("color", "green")],
            (1, 1): [("color", "green")],
        }
        assert res == expected

    def test_empty(self):
        df = DataFrame({"A": [1, 0]})
        s = df.style
        s.ctx = {(0, 0): [("color", "red")], (1, 0): [("", "")]}

        result = s._translate(True, True)["cellstyle"]
        expected = [
            {"props": [("color", "red")], "selectors": ["row0_col0"]},
            {"props": [("", "")], "selectors": ["row1_col0"]},
        ]
        assert result == expected

    def test_duplicate(self):
        df = DataFrame({"A": [1, 0]})
        s = df.style
        s.ctx = {(0, 0): [("color", "red")], (1, 0): [("color", "red")]}

        result = s._translate(True, True)["cellstyle"]
        expected = [
            {"props": [("color", "red")], "selectors": ["row0_col0", "row1_col0"]}
        ]
        assert result == expected

    def test_init_with_na_rep(self):
        # GH 21527 28358
        df = DataFrame([[None, None], [1.1, 1.2]], columns=["A", "B"])

        ctx = Styler(df, na_rep="NA")._translate(True, True)
        assert ctx["body"][0][1]["display_value"] == "NA"
        assert ctx["body"][0][2]["display_value"] == "NA"

    def test_set_na_rep(self):
        # GH 21527 28358
        df = DataFrame([[None, None], [1.1, 1.2]], columns=["A", "B"])

        with tm.assert_produces_warning(FutureWarning):
            ctx = df.style.set_na_rep("NA")._translate(True, True)
        assert ctx["body"][0][1]["display_value"] == "NA"
        assert ctx["body"][0][2]["display_value"] == "NA"

        with tm.assert_produces_warning(FutureWarning):
            ctx = (
                df.style.set_na_rep("NA")
                .format(None, na_rep="-", subset=["B"])
                ._translate(True, True)
            )
        assert ctx["body"][0][1]["display_value"] == "NA"
        assert ctx["body"][0][2]["display_value"] == "-"

    def test_caption(self):
        styler = Styler(self.df, caption="foo")
        result = styler.render()
        assert all(["caption" in result, "foo" in result])

        styler = self.df.style
        result = styler.set_caption("baz")
        assert styler is result
        assert styler.caption == "baz"

    def test_uuid(self):
        styler = Styler(self.df, uuid="abc123")
        result = styler.render()
        assert "abc123" in result

        styler = self.df.style
        result = styler.set_uuid("aaa")
        assert result is styler
        assert result.uuid == "aaa"

    def test_unique_id(self):
        # See https://github.com/pandas-dev/pandas/issues/16780
        df = DataFrame({"a": [1, 3, 5, 6], "b": [2, 4, 12, 21]})
        result = df.style.render(uuid="test")
        assert "test" in result
        ids = re.findall('id="(.*?)"', result)
        assert np.unique(ids).size == len(ids)

    def test_table_styles(self):
        style = [{"selector": "th", "props": [("foo", "bar")]}]  # default format
        styler = Styler(self.df, table_styles=style)
        result = " ".join(styler.render().split())
        assert "th { foo: bar; }" in result

        styler = self.df.style
        result = styler.set_table_styles(style)
        assert styler is result
        assert styler.table_styles == style

        # GH 39563
        style = [{"selector": "th", "props": "foo:bar;"}]  # css string format
        styler = self.df.style.set_table_styles(style)
        result = " ".join(styler.render().split())
        assert "th { foo: bar; }" in result

    def test_table_styles_multiple(self):
        ctx = self.df.style.set_table_styles(
            [
                {"selector": "th,td", "props": "color:red;"},
                {"selector": "tr", "props": "color:green;"},
            ]
        )._translate(True, True)["table_styles"]
        assert ctx == [
            {"selector": "th", "props": [("color", "red")]},
            {"selector": "td", "props": [("color", "red")]},
            {"selector": "tr", "props": [("color", "green")]},
        ]

    def test_maybe_convert_css_to_tuples(self):
        expected = [("a", "b"), ("c", "d e")]
        assert maybe_convert_css_to_tuples("a:b;c:d e;") == expected
        assert maybe_convert_css_to_tuples("a: b ;c:  d e  ") == expected
        expected = []
        assert maybe_convert_css_to_tuples("") == expected

    def test_maybe_convert_css_to_tuples_err(self):
        msg = "Styles supplied as string must follow CSS rule formats"
        with pytest.raises(ValueError, match=msg):
            maybe_convert_css_to_tuples("err")

    def test_table_attributes(self):
        attributes = 'class="foo" data-bar'
        styler = Styler(self.df, table_attributes=attributes)
        result = styler.render()
        assert 'class="foo" data-bar' in result

        result = self.df.style.set_table_attributes(attributes).render()
        assert 'class="foo" data-bar' in result

    def test_precision(self):
        s = Styler(self.df, precision=2)
        assert s.precision == 2

        with tm.assert_produces_warning(FutureWarning):
            s2 = s.set_precision(4)
        assert s is s2
        assert s.precision == 4

    def test_apply_none(self):
        def f(x):
            return DataFrame(
                np.where(x == x.max(), "color: red", ""),
                index=x.index,
                columns=x.columns,
            )

        result = DataFrame([[1, 2], [3, 4]]).style.apply(f, axis=None)._compute().ctx
        assert result[(1, 1)] == [("color", "red")]

    def test_trim(self):
        result = self.df.style.render()  # trim=True
        assert result.count("#") == 0

        result = self.df.style.highlight_max().render()
        assert result.count("#") == len(self.df.columns)

    def test_export(self):
        f = lambda x: "color: red" if x > 0 else "color: blue"
        g = lambda x, z: f"color: {z}" if x > 0 else f"color: {z}"
        style1 = self.styler
        style1.applymap(f).applymap(g, z="b").highlight_max()._compute()  # = render
        result = style1.export()
        style2 = self.df.style
        style2.use(result)
        assert style1._todo == style2._todo
        style2.render()

    def test_bad_apply_shape(self):
        df = DataFrame([[1, 2], [3, 4]])
        msg = "returned the wrong shape"
        with pytest.raises(ValueError, match=msg):
            df.style._apply(lambda x: "x", subset=pd.IndexSlice[[0, 1], :])

        with pytest.raises(ValueError, match=msg):
            df.style._apply(lambda x: [""], subset=pd.IndexSlice[[0, 1], :])

        with pytest.raises(ValueError, match=msg):
            df.style._apply(lambda x: ["", "", "", ""])

        with pytest.raises(ValueError, match=msg):
            df.style._apply(lambda x: ["", "", ""], subset=1)

        msg = "Length mismatch: Expected axis has 3 elements"
        with pytest.raises(ValueError, match=msg):
            df.style._apply(lambda x: ["", "", ""], axis=1)

        msg = "returned ndarray with wrong shape"
        with pytest.raises(ValueError, match=msg):
            df.style._apply(lambda x: np.array([[""], [""]]), axis=None)

    def test_apply_bad_return(self):
        def f(x):
            return ""

        df = DataFrame([[1, 2], [3, 4]])
        msg = (
            "must return a DataFrame or ndarray when passed to `Styler.apply` "
            "with axis=None"
        )
        with pytest.raises(TypeError, match=msg):
            df.style._apply(f, axis=None)

    def test_apply_bad_labels(self):
        def f(x):
            return DataFrame(index=[1, 2], columns=["a", "b"])

        df = DataFrame([[1, 2], [3, 4]])
        msg = "must have identical index and columns as the input"
        with pytest.raises(ValueError, match=msg):
            df.style._apply(f, axis=None)

    def test_get_level_lengths(self):
        index = MultiIndex.from_product([["a", "b"], [0, 1, 2]])
        expected = {
            (0, 0): 3,
            (0, 3): 3,
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
            (1, 4): 1,
            (1, 5): 1,
        }
        result = _get_level_lengths(index, sparsify=True, max_index=100)
        tm.assert_dict_equal(result, expected)

        expected = {
            (0, 0): 1,
            (0, 1): 1,
            (0, 2): 1,
            (0, 3): 1,
            (0, 4): 1,
            (0, 5): 1,
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
            (1, 4): 1,
            (1, 5): 1,
        }
        result = _get_level_lengths(index, sparsify=False, max_index=100)
        tm.assert_dict_equal(result, expected)

    def test_get_level_lengths_un_sorted(self):
        index = MultiIndex.from_arrays([[1, 1, 2, 1], ["a", "b", "b", "d"]])
        expected = {
            (0, 0): 2,
            (0, 2): 1,
            (0, 3): 1,
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
        }
        result = _get_level_lengths(index, sparsify=True, max_index=100)
        tm.assert_dict_equal(result, expected)

        expected = {
            (0, 0): 1,
            (0, 1): 1,
            (0, 2): 1,
            (0, 3): 1,
            (1, 0): 1,
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
        }
        result = _get_level_lengths(index, sparsify=False, max_index=100)
        tm.assert_dict_equal(result, expected)

    def test_mi_sparse_index_names(self):
        # TODO this test is verbose can be minimised to more directly target test
        df = DataFrame(
            {"A": [1, 2]},
            index=MultiIndex.from_arrays(
                [["a", "a"], [0, 1]], names=["idx_level_0", "idx_level_1"]
            ),
        )
        result = df.style._translate(True, True)
        head = result["head"][1]
        expected = [
            {
                "class": "index_name level0",
                "value": "idx_level_0",
                "type": "th",
                "is_visible": True,
                "display_value": "idx_level_0",
            },
            {
                "class": "index_name level1",
                "value": "idx_level_1",
                "type": "th",
                "is_visible": True,
                "display_value": "idx_level_1",
            },
            {
                "class": "blank col0",
                "value": self.blank_value,
                "type": "th",
                "is_visible": True,
                "display_value": self.blank_value,
            },
        ]

        assert head == expected

    def test_mi_sparse_column_names(self):
        # TODO this test is verbose - could be minimised
        df = DataFrame(
            np.arange(16).reshape(4, 4),
            index=MultiIndex.from_arrays(
                [["a", "a", "b", "a"], [0, 1, 1, 2]],
                names=["idx_level_0", "idx_level_1"],
            ),
            columns=MultiIndex.from_arrays(
                [["C1", "C1", "C2", "C2"], [1, 0, 1, 0]], names=["col_0", "col_1"]
            ),
        )
        result = df.style._translate(True, True)
        head = result["head"][1]
        expected = [
            {
                "class": "blank",
                "value": self.blank_value,
                "display_value": self.blank_value,
                "type": "th",
                "is_visible": True,
            },
            {
                "class": "index_name level1",
                "value": "col_1",
                "display_value": "col_1",
                "is_visible": True,
                "type": "th",
            },
            {
                "class": "col_heading level1 col0",
                "display_value": 1,
                "is_visible": True,
                "type": "th",
                "value": 1,
                "attributes": "",
            },
            {
                "class": "col_heading level1 col1",
                "display_value": 0,
                "is_visible": True,
                "type": "th",
                "value": 0,
                "attributes": "",
            },
            {
                "class": "col_heading level1 col2",
                "display_value": 1,
                "is_visible": True,
                "type": "th",
                "value": 1,
                "attributes": "",
            },
            {
                "class": "col_heading level1 col3",
                "display_value": 0,
                "is_visible": True,
                "type": "th",
                "value": 0,
                "attributes": "",
            },
        ]
        assert head == expected

    def test_hide_column_headers(self):
        ctx = self.styler.hide_columns()._translate(True, True)
        assert len(ctx["head"]) == 0  # no header entries with an unnamed index

        self.df.index.name = "some_name"
        ctx = self.df.style.hide_columns()._translate(True, True)
        assert len(ctx["head"]) == 0  # no header for index names, changed in #42101

    def test_hide_single_index(self):
        # GH 14194
        # single unnamed index
        ctx = self.df.style._translate(True, True)
        assert ctx["body"][0][0]["is_visible"]
        assert ctx["head"][0][0]["is_visible"]
        ctx2 = self.df.style.hide_index()._translate(True, True)
        assert not ctx2["body"][0][0]["is_visible"]
        assert not ctx2["head"][0][0]["is_visible"]

        # single named index
        ctx3 = self.df.set_index("A").style._translate(True, True)
        assert ctx3["body"][0][0]["is_visible"]
        assert len(ctx3["head"]) == 2  # 2 header levels
        assert ctx3["head"][0][0]["is_visible"]

        ctx4 = self.df.set_index("A").style.hide_index()._translate(True, True)
        assert not ctx4["body"][0][0]["is_visible"]
        assert len(ctx4["head"]) == 1  # only 1 header levels
        assert not ctx4["head"][0][0]["is_visible"]

    def test_hide_multiindex(self):
        # GH 14194
        df = DataFrame(
            {"A": [1, 2]},
            index=MultiIndex.from_arrays(
                [["a", "a"], [0, 1]], names=["idx_level_0", "idx_level_1"]
            ),
        )
        ctx1 = df.style._translate(True, True)
        # tests for 'a' and '0'
        assert ctx1["body"][0][0]["is_visible"]
        assert ctx1["body"][0][1]["is_visible"]
        # check for blank header rows
        assert ctx1["head"][0][0]["is_visible"]
        assert ctx1["head"][0][1]["is_visible"]

        ctx2 = df.style.hide_index()._translate(True, True)
        # tests for 'a' and '0'
        assert not ctx2["body"][0][0]["is_visible"]
        assert not ctx2["body"][0][1]["is_visible"]
        # check for blank header rows
        assert not ctx2["head"][0][0]["is_visible"]
        assert not ctx2["head"][0][1]["is_visible"]

    def test_hide_columns_single_level(self):
        # GH 14194
        # test hiding single column
        ctx = self.df.style._translate(True, True)
        assert ctx["head"][0][1]["is_visible"]
        assert ctx["head"][0][1]["display_value"] == "A"
        assert ctx["head"][0][2]["is_visible"]
        assert ctx["head"][0][2]["display_value"] == "B"
        assert ctx["body"][0][1]["is_visible"]  # col A, row 1
        assert ctx["body"][1][2]["is_visible"]  # col B, row 1

        ctx = self.df.style.hide_columns("A")._translate(True, True)
        assert not ctx["head"][0][1]["is_visible"]
        assert not ctx["body"][0][1]["is_visible"]  # col A, row 1
        assert ctx["body"][1][2]["is_visible"]  # col B, row 1

        # test hiding mulitiple columns
        ctx = self.df.style.hide_columns(["A", "B"])._translate(True, True)
        assert not ctx["head"][0][1]["is_visible"]
        assert not ctx["head"][0][2]["is_visible"]
        assert not ctx["body"][0][1]["is_visible"]  # col A, row 1
        assert not ctx["body"][1][2]["is_visible"]  # col B, row 1

    def test_hide_columns_index_mult_levels(self):
        # GH 14194
        # setup dataframe with multiple column levels and indices
        i1 = MultiIndex.from_arrays(
            [["a", "a"], [0, 1]], names=["idx_level_0", "idx_level_1"]
        )
        i2 = MultiIndex.from_arrays(
            [["b", "b"], [0, 1]], names=["col_level_0", "col_level_1"]
        )
        df = DataFrame([[1, 2], [3, 4]], index=i1, columns=i2)
        ctx = df.style._translate(True, True)
        # column headers
        assert ctx["head"][0][2]["is_visible"]
        assert ctx["head"][1][2]["is_visible"]
        assert ctx["head"][1][3]["display_value"] == 1
        # indices
        assert ctx["body"][0][0]["is_visible"]
        # data
        assert ctx["body"][1][2]["is_visible"]
        assert ctx["body"][1][2]["display_value"] == 3
        assert ctx["body"][1][3]["is_visible"]
        assert ctx["body"][1][3]["display_value"] == 4

        # hide top column level, which hides both columns
        ctx = df.style.hide_columns("b")._translate(True, True)
        assert not ctx["head"][0][2]["is_visible"]  # b
        assert not ctx["head"][1][2]["is_visible"]  # 0
        assert not ctx["body"][1][2]["is_visible"]  # 3
        assert ctx["body"][0][0]["is_visible"]  # index

        # hide first column only
        ctx = df.style.hide_columns([("b", 0)])._translate(True, True)
        assert not ctx["head"][0][2]["is_visible"]  # b
        assert ctx["head"][0][3]["is_visible"]  # b
        assert not ctx["head"][1][2]["is_visible"]  # 0
        assert not ctx["body"][1][2]["is_visible"]  # 3
        assert ctx["body"][1][3]["is_visible"]
        assert ctx["body"][1][3]["display_value"] == 4

        # hide second column and index
        ctx = df.style.hide_columns([("b", 1)]).hide_index()._translate(True, True)
        assert not ctx["body"][0][0]["is_visible"]  # index
        assert ctx["head"][0][2]["is_visible"]  # b
        assert ctx["head"][1][2]["is_visible"]  # 0
        assert not ctx["head"][1][3]["is_visible"]  # 1
        assert not ctx["body"][1][3]["is_visible"]  # 4
        assert ctx["body"][1][2]["is_visible"]
        assert ctx["body"][1][2]["display_value"] == 3

        # hide top row level, which hides both rows
        ctx = df.style.hide_index("a")._translate(True, True)
        for i in [0, 1, 2, 3]:
            assert not ctx["body"][0][i]["is_visible"]
            assert not ctx["body"][1][i]["is_visible"]

        # hide first row only
        ctx = df.style.hide_index(("a", 0))._translate(True, True)
        for i in [0, 1, 2, 3]:
            assert not ctx["body"][0][i]["is_visible"]
            assert ctx["body"][1][i]["is_visible"]

    def test_pipe(self):
        def set_caption_from_template(styler, a, b):
            return styler.set_caption(f"Dataframe with a = {a} and b = {b}")

        styler = self.df.style.pipe(set_caption_from_template, "A", b="B")
        assert "Dataframe with a = A and b = B" in styler.render()

        # Test with an argument that is a (callable, keyword_name) pair.
        def f(a, b, styler):
            return (a, b, styler)

        styler = self.df.style
        result = styler.pipe((f, "styler"), a=1, b=2)
        assert result == (1, 2, styler)

    def test_no_cell_ids(self):
        # GH 35588
        # GH 35663
        df = DataFrame(data=[[0]])
        styler = Styler(df, uuid="_", cell_ids=False)
        styler.render()
        s = styler.render()  # render twice to ensure ctx is not updated
        assert s.find('<td  class="data row0 col0" >') != -1

    @pytest.mark.parametrize(
        "classes",
        [
            DataFrame(
                data=[["", "test-class"], [np.nan, None]],
                columns=["A", "B"],
                index=["a", "b"],
            ),
            DataFrame(data=[["test-class"]], columns=["B"], index=["a"]),
            DataFrame(data=[["test-class", "unused"]], columns=["B", "C"], index=["a"]),
        ],
    )
    def test_set_data_classes(self, classes):
        # GH 36159
        df = DataFrame(data=[[0, 1], [2, 3]], columns=["A", "B"], index=["a", "b"])
        s = Styler(df, uuid_len=0, cell_ids=False).set_td_classes(classes).render()
        assert '<td  class="data row0 col0" >0</td>' in s
        assert '<td  class="data row0 col1 test-class" >1</td>' in s
        assert '<td  class="data row1 col0" >2</td>' in s
        assert '<td  class="data row1 col1" >3</td>' in s
        # GH 39317
        s = Styler(df, uuid_len=0, cell_ids=True).set_td_classes(classes).render()
        assert '<td id="T__row0_col0" class="data row0 col0" >0</td>' in s
        assert '<td id="T__row0_col1" class="data row0 col1 test-class" >1</td>' in s
        assert '<td id="T__row1_col0" class="data row1 col0" >2</td>' in s
        assert '<td id="T__row1_col1" class="data row1 col1" >3</td>' in s

    def test_set_data_classes_reindex(self):
        # GH 39317
        df = DataFrame(
            data=[[0, 1, 2], [3, 4, 5], [6, 7, 8]], columns=[0, 1, 2], index=[0, 1, 2]
        )
        classes = DataFrame(
            data=[["mi", "ma"], ["mu", "mo"]],
            columns=[0, 2],
            index=[0, 2],
        )
        s = Styler(df, uuid_len=0).set_td_classes(classes).render()
        assert '<td id="T__row0_col0" class="data row0 col0 mi" >0</td>' in s
        assert '<td id="T__row0_col2" class="data row0 col2 ma" >2</td>' in s
        assert '<td id="T__row1_col1" class="data row1 col1" >4</td>' in s
        assert '<td id="T__row2_col0" class="data row2 col0 mu" >6</td>' in s
        assert '<td id="T__row2_col2" class="data row2 col2 mo" >8</td>' in s

    def test_chaining_table_styles(self):
        # GH 35607
        df = DataFrame(data=[[0, 1], [1, 2]], columns=["A", "B"])
        styler = df.style.set_table_styles(
            [{"selector": "", "props": [("background-color", "yellow")]}]
        ).set_table_styles(
            [{"selector": ".col0", "props": [("background-color", "blue")]}],
            overwrite=False,
        )
        assert len(styler.table_styles) == 2

    def test_column_and_row_styling(self):
        # GH 35607
        df = DataFrame(data=[[0, 1], [1, 2]], columns=["A", "B"])
        s = Styler(df, uuid_len=0)
        s = s.set_table_styles({"A": [{"selector": "", "props": [("color", "blue")]}]})
        assert "#T__ .col0 {\n  color: blue;\n}" in s.render()
        s = s.set_table_styles(
            {0: [{"selector": "", "props": [("color", "blue")]}]}, axis=1
        )
        assert "#T__ .row0 {\n  color: blue;\n}" in s.render()

    @pytest.mark.parametrize("len_", [1, 5, 32, 33, 100])
    def test_uuid_len(self, len_):
        # GH 36345
        df = DataFrame(data=[["A"]])
        s = Styler(df, uuid_len=len_, cell_ids=False).render()
        strt = s.find('id="T_')
        end = s[strt + 6 :].find('"')
        if len_ > 32:
            assert end == 32 + 1
        else:
            assert end == len_ + 1

    @pytest.mark.parametrize("len_", [-2, "bad", None])
    def test_uuid_len_raises(self, len_):
        # GH 36345
        df = DataFrame(data=[["A"]])
        msg = "``uuid_len`` must be an integer in range \\[0, 32\\]."
        with pytest.raises(TypeError, match=msg):
            Styler(df, uuid_len=len_, cell_ids=False).render()

    @pytest.mark.parametrize(
        "slc",
        [
            pd.IndexSlice[:, :],
            pd.IndexSlice[:, 1],
            pd.IndexSlice[1, :],
            pd.IndexSlice[[1], [1]],
            pd.IndexSlice[1, [1]],
            pd.IndexSlice[[1], 1],
            pd.IndexSlice[1],
            pd.IndexSlice[1, 1],
            slice(None, None, None),
            [0, 1],
            np.array([0, 1]),
            pd.Series([0, 1]),
        ],
    )
    def test_non_reducing_slice(self, slc):
        df = DataFrame([[0, 1], [2, 3]])

        tslice_ = non_reducing_slice(slc)
        assert isinstance(df.loc[tslice_], DataFrame)

    @pytest.mark.parametrize("box", [list, pd.Series, np.array])
    def test_list_slice(self, box):
        # like dataframe getitem
        subset = box(["A"])

        df = DataFrame({"A": [1, 2], "B": [3, 4]}, index=["A", "B"])
        expected = pd.IndexSlice[:, ["A"]]

        result = non_reducing_slice(subset)
        tm.assert_frame_equal(df.loc[result], df.loc[expected])

    def test_non_reducing_slice_on_multiindex(self):
        # GH 19861
        dic = {
            ("a", "d"): [1, 4],
            ("a", "c"): [2, 3],
            ("b", "c"): [3, 2],
            ("b", "d"): [4, 1],
        }
        df = DataFrame(dic, index=[0, 1])
        idx = pd.IndexSlice
        slice_ = idx[:, idx["b", "d"]]
        tslice_ = non_reducing_slice(slice_)

        result = df.loc[tslice_]
        expected = DataFrame({("b", "d"): [4, 1]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "slice_",
        [
            pd.IndexSlice[:, :],
            # check cols
            pd.IndexSlice[:, pd.IndexSlice[["a"]]],  # inferred deeper need list
            pd.IndexSlice[:, pd.IndexSlice[["a"], ["c"]]],  # inferred deeper need list
            pd.IndexSlice[:, pd.IndexSlice["a", "c", :]],
            pd.IndexSlice[:, pd.IndexSlice["a", :, "e"]],
            pd.IndexSlice[:, pd.IndexSlice[:, "c", "e"]],
            pd.IndexSlice[:, pd.IndexSlice["a", ["c", "d"], :]],  # check list
            pd.IndexSlice[:, pd.IndexSlice["a", ["c", "d", "-"], :]],  # allow missing
            pd.IndexSlice[:, pd.IndexSlice["a", ["c", "d", "-"], "e"]],  # no slice
            # check rows
            pd.IndexSlice[pd.IndexSlice[["U"]], :],  # inferred deeper need list
            pd.IndexSlice[pd.IndexSlice[["U"], ["W"]], :],  # inferred deeper need list
            pd.IndexSlice[pd.IndexSlice["U", "W", :], :],
            pd.IndexSlice[pd.IndexSlice["U", :, "Y"], :],
            pd.IndexSlice[pd.IndexSlice[:, "W", "Y"], :],
            pd.IndexSlice[pd.IndexSlice[:, "W", ["Y", "Z"]], :],  # check list
            pd.IndexSlice[pd.IndexSlice[:, "W", ["Y", "Z", "-"]], :],  # allow missing
            pd.IndexSlice[pd.IndexSlice["U", "W", ["Y", "Z", "-"]], :],  # no slice
            # check simultaneous
            pd.IndexSlice[pd.IndexSlice[:, "W", "Y"], pd.IndexSlice["a", "c", :]],
        ],
    )
    def test_non_reducing_multi_slice_on_multiindex(self, slice_):
        # GH 33562
        cols = MultiIndex.from_product([["a", "b"], ["c", "d"], ["e", "f"]])
        idxs = MultiIndex.from_product([["U", "V"], ["W", "X"], ["Y", "Z"]])
        df = DataFrame(np.arange(64).reshape(8, 8), columns=cols, index=idxs)

        expected = df.loc[slice_]
        result = df.loc[non_reducing_slice(slice_)]
        tm.assert_frame_equal(result, expected)
