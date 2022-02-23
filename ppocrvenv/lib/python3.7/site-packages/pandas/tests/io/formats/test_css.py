import pytest

import pandas._testing as tm

from pandas.io.formats.css import (
    CSSResolver,
    CSSWarning,
)


def assert_resolves(css, props, inherited=None):
    resolve = CSSResolver()
    actual = resolve(css, inherited=inherited)
    assert props == actual


def assert_same_resolution(css1, css2, inherited=None):
    resolve = CSSResolver()
    resolved1 = resolve(css1, inherited=inherited)
    resolved2 = resolve(css2, inherited=inherited)
    assert resolved1 == resolved2


@pytest.mark.parametrize(
    "name,norm,abnorm",
    [
        (
            "whitespace",
            "hello: world; foo: bar",
            " \t hello \t :\n  world \n  ;  \n foo: \tbar\n\n",
        ),
        ("case", "hello: world; foo: bar", "Hello: WORLD; foO: bar"),
        ("empty-decl", "hello: world; foo: bar", "; hello: world;; foo: bar;\n; ;"),
        ("empty-list", "", ";"),
    ],
)
def test_css_parse_normalisation(name, norm, abnorm):
    assert_same_resolution(norm, abnorm)


@pytest.mark.parametrize(
    "invalid_css,remainder",
    [
        # No colon
        ("hello-world", ""),
        ("border-style: solid; hello-world", "border-style: solid"),
        (
            "border-style: solid; hello-world; font-weight: bold",
            "border-style: solid; font-weight: bold",
        ),
        # Unclosed string fail
        # Invalid size
        ("font-size: blah", "font-size: 1em"),
        ("font-size: 1a2b", "font-size: 1em"),
        ("font-size: 1e5pt", "font-size: 1em"),
        ("font-size: 1+6pt", "font-size: 1em"),
        ("font-size: 1unknownunit", "font-size: 1em"),
        ("font-size: 10", "font-size: 1em"),
        ("font-size: 10 pt", "font-size: 1em"),
    ],
)
def test_css_parse_invalid(invalid_css, remainder):
    with tm.assert_produces_warning(CSSWarning):
        assert_same_resolution(invalid_css, remainder)


@pytest.mark.parametrize(
    "shorthand,expansions",
    [
        ("margin", ["margin-top", "margin-right", "margin-bottom", "margin-left"]),
        ("padding", ["padding-top", "padding-right", "padding-bottom", "padding-left"]),
        (
            "border-width",
            [
                "border-top-width",
                "border-right-width",
                "border-bottom-width",
                "border-left-width",
            ],
        ),
        (
            "border-color",
            [
                "border-top-color",
                "border-right-color",
                "border-bottom-color",
                "border-left-color",
            ],
        ),
        (
            "border-style",
            [
                "border-top-style",
                "border-right-style",
                "border-bottom-style",
                "border-left-style",
            ],
        ),
    ],
)
def test_css_side_shorthands(shorthand, expansions):
    top, right, bottom, left = expansions

    assert_resolves(
        f"{shorthand}: 1pt", {top: "1pt", right: "1pt", bottom: "1pt", left: "1pt"}
    )

    assert_resolves(
        f"{shorthand}: 1pt 4pt", {top: "1pt", right: "4pt", bottom: "1pt", left: "4pt"}
    )

    assert_resolves(
        f"{shorthand}: 1pt 4pt 2pt",
        {top: "1pt", right: "4pt", bottom: "2pt", left: "4pt"},
    )

    assert_resolves(
        f"{shorthand}: 1pt 4pt 2pt 0pt",
        {top: "1pt", right: "4pt", bottom: "2pt", left: "0pt"},
    )

    with tm.assert_produces_warning(CSSWarning):
        assert_resolves(f"{shorthand}: 1pt 1pt 1pt 1pt 1pt", {})


@pytest.mark.parametrize(
    "style,inherited,equiv",
    [
        ("margin: 1px; margin: 2px", "", "margin: 2px"),
        ("margin: 1px", "margin: 2px", "margin: 1px"),
        ("margin: 1px; margin: inherit", "margin: 2px", "margin: 2px"),
        (
            "margin: 1px; margin-top: 2px",
            "",
            "margin-left: 1px; margin-right: 1px; "
            + "margin-bottom: 1px; margin-top: 2px",
        ),
        ("margin-top: 2px", "margin: 1px", "margin: 1px; margin-top: 2px"),
        ("margin: 1px", "margin-top: 2px", "margin: 1px"),
        (
            "margin: 1px; margin-top: inherit",
            "margin: 2px",
            "margin: 1px; margin-top: 2px",
        ),
    ],
)
def test_css_precedence(style, inherited, equiv):
    resolve = CSSResolver()
    inherited_props = resolve(inherited)
    style_props = resolve(style, inherited=inherited_props)
    equiv_props = resolve(equiv)
    assert style_props == equiv_props


@pytest.mark.parametrize(
    "style,equiv",
    [
        (
            "margin: 1px; margin-top: inherit",
            "margin-bottom: 1px; margin-right: 1px; margin-left: 1px",
        ),
        ("margin-top: inherit", ""),
        ("margin-top: initial", ""),
    ],
)
def test_css_none_absent(style, equiv):
    assert_same_resolution(style, equiv)


@pytest.mark.parametrize(
    "size,resolved",
    [
        ("xx-small", "6pt"),
        ("x-small", f"{7.5:f}pt"),
        ("small", f"{9.6:f}pt"),
        ("medium", "12pt"),
        ("large", f"{13.5:f}pt"),
        ("x-large", "18pt"),
        ("xx-large", "24pt"),
        ("8px", "6pt"),
        ("1.25pc", "15pt"),
        (".25in", "18pt"),
        ("02.54cm", "72pt"),
        ("25.4mm", "72pt"),
        ("101.6q", "72pt"),
        ("101.6q", "72pt"),
    ],
)
@pytest.mark.parametrize("relative_to", [None, "16pt"])  # invariant to inherited size
def test_css_absolute_font_size(size, relative_to, resolved):
    if relative_to is None:
        inherited = None
    else:
        inherited = {"font-size": relative_to}
    assert_resolves(f"font-size: {size}", {"font-size": resolved}, inherited=inherited)


@pytest.mark.parametrize(
    "size,relative_to,resolved",
    [
        ("1em", None, "12pt"),
        ("1.0em", None, "12pt"),
        ("1.25em", None, "15pt"),
        ("1em", "16pt", "16pt"),
        ("1.0em", "16pt", "16pt"),
        ("1.25em", "16pt", "20pt"),
        ("1rem", "16pt", "12pt"),
        ("1.0rem", "16pt", "12pt"),
        ("1.25rem", "16pt", "15pt"),
        ("100%", None, "12pt"),
        ("125%", None, "15pt"),
        ("100%", "16pt", "16pt"),
        ("125%", "16pt", "20pt"),
        ("2ex", None, "12pt"),
        ("2.0ex", None, "12pt"),
        ("2.50ex", None, "15pt"),
        ("inherit", "16pt", "16pt"),
        ("smaller", None, "10pt"),
        ("smaller", "18pt", "15pt"),
        ("larger", None, f"{14.4:f}pt"),
        ("larger", "15pt", "18pt"),
    ],
)
def test_css_relative_font_size(size, relative_to, resolved):
    if relative_to is None:
        inherited = None
    else:
        inherited = {"font-size": relative_to}
    assert_resolves(f"font-size: {size}", {"font-size": resolved}, inherited=inherited)
