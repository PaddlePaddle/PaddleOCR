import numpy as np
import pytest

from pandas import DataFrame
import pandas._testing as tm

from pandas.io.excel import ExcelWriter
from pandas.io.formats.excel import ExcelFormatter


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(
            "xlwt",
            marks=pytest.mark.xfail(
                reason="xlwt does not support openpyxl-compatible style dicts"
            ),
        ),
        "xlsxwriter",
        "openpyxl",
    ],
)
def test_styler_to_excel(request, engine):
    def style(df):
        # TODO: RGB colors not supported in xlwt
        return DataFrame(
            [
                ["font-weight: bold", "", ""],
                ["", "color: blue", ""],
                ["", "", "text-decoration: underline"],
                ["border-style: solid", "", ""],
                ["", "font-style: italic", ""],
                ["", "", "text-align: right"],
                ["background-color: red", "", ""],
                ["number-format: 0%", "", ""],
                ["", "", ""],
                ["", "", ""],
                ["", "", ""],
            ],
            index=df.index,
            columns=df.columns,
        )

    def assert_equal_style(cell1, cell2, engine):
        if engine in ["xlsxwriter", "openpyxl"]:
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=(
                        f"GH25351: failing on some attribute comparisons in {engine}"
                    )
                )
            )
        # TODO: should find a better way to check equality
        assert cell1.alignment.__dict__ == cell2.alignment.__dict__
        assert cell1.border.__dict__ == cell2.border.__dict__
        assert cell1.fill.__dict__ == cell2.fill.__dict__
        assert cell1.font.__dict__ == cell2.font.__dict__
        assert cell1.number_format == cell2.number_format
        assert cell1.protection.__dict__ == cell2.protection.__dict__

    def custom_converter(css):
        # use bold iff there is custom style attached to the cell
        if css.strip(" \n;"):
            return {"font": {"bold": True}}
        return {}

    pytest.importorskip("jinja2")
    pytest.importorskip(engine)

    # Prepare spreadsheets

    df = DataFrame(np.random.randn(11, 3))
    with tm.ensure_clean(".xlsx" if engine != "xlwt" else ".xls") as path:
        with ExcelWriter(path, engine=engine) as writer:
            df.to_excel(writer, sheet_name="frame")
            df.style.to_excel(writer, sheet_name="unstyled")
            styled = df.style.apply(style, axis=None)
            styled.to_excel(writer, sheet_name="styled")
            ExcelFormatter(styled, style_converter=custom_converter).write(
                writer, sheet_name="custom"
            )

        if engine not in ("openpyxl", "xlsxwriter"):
            # For other engines, we only smoke test
            return
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.load_workbook(path)

        # (1) compare DataFrame.to_excel and Styler.to_excel when unstyled
        n_cells = 0
        for col1, col2 in zip(wb["frame"].columns, wb["unstyled"].columns):
            assert len(col1) == len(col2)
            for cell1, cell2 in zip(col1, col2):
                assert cell1.value == cell2.value
                assert_equal_style(cell1, cell2, engine)
                n_cells += 1

        # ensure iteration actually happened:
        assert n_cells == (11 + 1) * (3 + 1)

        # (2) check styling with default converter

        # TODO: openpyxl (as at 2.4) prefixes colors with 00, xlsxwriter with FF
        alpha = "00" if engine == "openpyxl" else "FF"

        n_cells = 0
        for col1, col2 in zip(wb["frame"].columns, wb["styled"].columns):
            assert len(col1) == len(col2)
            for cell1, cell2 in zip(col1, col2):
                ref = f"{cell2.column}{cell2.row:d}"
                # TODO: this isn't as strong a test as ideal; we should
                #      confirm that differences are exclusive
                if ref == "B2":
                    assert not cell1.font.bold
                    assert cell2.font.bold
                elif ref == "C3":
                    assert cell1.font.color.rgb != cell2.font.color.rgb
                    assert cell2.font.color.rgb == alpha + "0000FF"
                elif ref == "D4":
                    assert cell1.font.underline != cell2.font.underline
                    assert cell2.font.underline == "single"
                elif ref == "B5":
                    assert not cell1.border.left.style
                    assert (
                        cell2.border.top.style
                        == cell2.border.right.style
                        == cell2.border.bottom.style
                        == cell2.border.left.style
                        == "medium"
                    )
                elif ref == "C6":
                    assert not cell1.font.italic
                    assert cell2.font.italic
                elif ref == "D7":
                    assert cell1.alignment.horizontal != cell2.alignment.horizontal
                    assert cell2.alignment.horizontal == "right"
                elif ref == "B8":
                    assert cell1.fill.fgColor.rgb != cell2.fill.fgColor.rgb
                    assert cell1.fill.patternType != cell2.fill.patternType
                    assert cell2.fill.fgColor.rgb == alpha + "FF0000"
                    assert cell2.fill.patternType == "solid"
                elif ref == "B9":
                    assert cell1.number_format == "General"
                    assert cell2.number_format == "0%"
                else:
                    assert_equal_style(cell1, cell2, engine)

                assert cell1.value == cell2.value
                n_cells += 1

        assert n_cells == (11 + 1) * (3 + 1)

        # (3) check styling with custom converter
        n_cells = 0
        for col1, col2 in zip(wb["frame"].columns, wb["custom"].columns):
            assert len(col1) == len(col2)
            for cell1, cell2 in zip(col1, col2):
                ref = f"{cell2.column}{cell2.row:d}"
                if ref in ("B2", "C3", "D4", "B5", "C6", "D7", "B8", "B9"):
                    assert not cell1.font.bold
                    assert cell2.font.bold
                else:
                    assert_equal_style(cell1, cell2, engine)

                assert cell1.value == cell2.value
                n_cells += 1

        assert n_cells == (11 + 1) * (3 + 1)
