from io import StringIO

import pytest

import pandas as pd
import pandas._testing as tm

pytest.importorskip("tabulate")


def test_simple():
    buf = StringIO()
    df = pd.DataFrame([1, 2, 3])
    df.to_markdown(buf=buf)
    result = buf.getvalue()
    assert (
        result == "|    |   0 |\n|---:|----:|\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |"
    )


def test_empty_frame():
    buf = StringIO()
    df = pd.DataFrame({"id": [], "first_name": [], "last_name": []}).set_index("id")
    df.to_markdown(buf=buf)
    result = buf.getvalue()
    assert result == (
        "| id   | first_name   | last_name   |\n"
        "|------|--------------|-------------|"
    )


def test_other_tablefmt():
    buf = StringIO()
    df = pd.DataFrame([1, 2, 3])
    df.to_markdown(buf=buf, tablefmt="jira")
    result = buf.getvalue()
    assert result == "||    ||   0 ||\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |"


def test_other_headers():
    buf = StringIO()
    df = pd.DataFrame([1, 2, 3])
    df.to_markdown(buf=buf, headers=["foo", "bar"])
    result = buf.getvalue()
    assert result == (
        "|   foo |   bar |\n|------:|------:|\n|     0 "
        "|     1 |\n|     1 |     2 |\n|     2 |     3 |"
    )


def test_series():
    buf = StringIO()
    s = pd.Series([1, 2, 3], name="foo")
    s.to_markdown(buf=buf)
    result = buf.getvalue()
    assert result == (
        "|    |   foo |\n|---:|------:|\n|  0 |     1 "
        "|\n|  1 |     2 |\n|  2 |     3 |"
    )


def test_no_buf(capsys):
    df = pd.DataFrame([1, 2, 3])
    result = df.to_markdown()
    assert (
        result == "|    |   0 |\n|---:|----:|\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |"
    )


@pytest.mark.parametrize("index", [True, False, None])
@pytest.mark.parametrize("showindex", [True, False, None])
def test_index(index, showindex):
    # GH 32667
    kwargs = {}
    if index is not None:
        kwargs["index"] = index
    if showindex is not None:
        kwargs["showindex"] = showindex

    df = pd.DataFrame([1, 2, 3])
    yes_index_result = (
        "|    |   0 |\n|---:|----:|\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |"
    )
    no_index_result = "|   0 |\n|----:|\n|   1 |\n|   2 |\n|   3 |"

    warning = FutureWarning if "showindex" in kwargs else None
    with tm.assert_produces_warning(warning):
        result = df.to_markdown(**kwargs)

    if "showindex" in kwargs:
        # give showindex higher priority if specified
        if showindex:
            expected = yes_index_result
        else:
            expected = no_index_result
    else:
        if index in [True, None]:
            expected = yes_index_result
        else:
            expected = no_index_result
    assert result == expected
