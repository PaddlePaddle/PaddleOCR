from __future__ import annotations

from io import (
    BytesIO,
    StringIO,
)
import os
from urllib.error import HTTPError

import numpy as np
import pytest

from pandas.compat import PY38
import pandas.util._test_decorators as td

from pandas import DataFrame
import pandas._testing as tm

from pandas.io.xml import read_xml

"""
CHECK LIST

[x] - ValueError: "Values for parser can only be lxml or etree."

etree
[X] - ImportError: "lxml not found, please install or use the etree parser."
[X] - TypeError: "expected str, bytes or os.PathLike object, not NoneType"
[X] - ValueError: "Either element or attributes can be parsed not both."
[X] - ValueError: "xpath does not return any nodes..."
[X] - SyntaxError: "You have used an incorrect or unsupported XPath"
[X] - ValueError: "names does not match length of child elements in xpath."
[X] - TypeError: "...is not a valid type for names"
[X] - ValueError: "To use stylesheet, you need lxml installed..."
[]  - URLError: (GENERAL ERROR WITH HTTPError AS SUBCLASS)
[X] - HTTPError: "HTTP Error 404: Not Found"
[]  - OSError: (GENERAL ERROR WITH FileNotFoundError AS SUBCLASS)
[X] - FileNotFoundError: "No such file or directory"
[]  - ParseError    (FAILSAFE CATCH ALL FOR VERY COMPLEX XML)
[X] - UnicodeDecodeError: "'utf-8' codec can't decode byte 0xe9..."
[X] - UnicodeError: "UTF-16 stream does not start with BOM"
[X] - BadZipFile: "File is not a zip file"
[X] - OSError: "Invalid data stream"
[X] - LZMAError: "Input format not supported by decoder"
[X] - ValueError: "Unrecognized compression type"
[X] - PermissionError: "Forbidden"

lxml
[X] - ValueError: "Either element or attributes can be parsed not both."
[X] - AttributeError: "__enter__"
[X] - XSLTApplyError: "Cannot resolve URI"
[X] - XSLTParseError: "document is not a stylesheet"
[X] - ValueError: "xpath does not return any nodes."
[X] - XPathEvalError: "Invalid expression"
[]  - XPathSyntaxError: (OLD VERSION IN lxml FOR XPATH ERRORS)
[X] - TypeError: "empty namespace prefix is not supported in XPath"
[X] - ValueError: "names does not match length of child elements in xpath."
[X] - TypeError: "...is not a valid type for names"
[X] - LookupError: "unknown encoding"
[]  - URLError: (USUALLY DUE TO NETWORKING)
[X  - HTTPError: "HTTP Error 404: Not Found"
[X] - OSError: "failed to load external entity"
[X] - XMLSyntaxError: "Start tag expected, '<' not found"
[]  - ParserError: (FAILSAFE CATCH ALL FOR VERY COMPLEX XML
[X] - ValueError: "Values for parser can only be lxml or etree."
[X] - UnicodeDecodeError: "'utf-8' codec can't decode byte 0xe9..."
[X] - UnicodeError: "UTF-16 stream does not start with BOM"
[X] - BadZipFile: "File is not a zip file"
[X] - OSError: "Invalid data stream"
[X] - LZMAError: "Input format not supported by decoder"
[X] - ValueError: "Unrecognized compression type"
[X] - PermissionError: "Forbidden"
"""

geom_df = DataFrame(
    {
        "shape": ["square", "circle", "triangle"],
        "degrees": [360, 360, 180],
        "sides": [4, np.nan, 3],
    }
)

xml_default_nmsp = """\
<?xml version='1.0' encoding='utf-8'?>
<data xmlns="http://example.com">
  <row>
    <shape>square</shape>
    <degrees>360</degrees>
    <sides>4</sides>
  </row>
  <row>
    <shape>circle</shape>
    <degrees>360</degrees>
    <sides/>
  </row>
  <row>
    <shape>triangle</shape>
    <degrees>180</degrees>
    <sides>3</sides>
  </row>
</data>"""

xml_prefix_nmsp = """\
<?xml version='1.0' encoding='utf-8'?>
<doc:data xmlns:doc="http://example.com">
  <doc:row>
    <doc:shape>square</doc:shape>
    <doc:degrees>360</doc:degrees>
    <doc:sides>4.0</doc:sides>
  </doc:row>
  <doc:row>
    <doc:shape>circle</doc:shape>
    <doc:degrees>360</doc:degrees>
    <doc:sides/>
  </doc:row>
  <doc:row>
    <doc:shape>triangle</doc:shape>
    <doc:degrees>180</doc:degrees>
    <doc:sides>3.0</doc:sides>
  </doc:row>
</doc:data>"""


df_kml = DataFrame(
    {
        "id": {
            0: "ID_00001",
            1: "ID_00002",
            2: "ID_00003",
            3: "ID_00004",
            4: "ID_00005",
        },
        "name": {
            0: "Blue Line (Forest Park)",
            1: "Red, Purple Line",
            2: "Red, Purple Line",
            3: "Red, Purple Line",
            4: "Red, Purple Line",
        },
        "styleUrl": {
            0: "#LineStyle01",
            1: "#LineStyle01",
            2: "#LineStyle01",
            3: "#LineStyle01",
            4: "#LineStyle01",
        },
        "extrude": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        "altitudeMode": {
            0: "clampedToGround",
            1: "clampedToGround",
            2: "clampedToGround",
            3: "clampedToGround",
            4: "clampedToGround",
        },
        "coordinates": {
            0: (
                "-87.77678526964958,41.8708863930319,0 "
                "-87.77826234150609,41.87097820122218,0 "
                "-87.78251583439344,41.87130129991005,0 "
                "-87.78418294588424,41.87145055520308,0 "
                "-87.7872369165933,41.8717239119163,0 "
                "-87.79160214925886,41.87210797280065,0"
            ),
            1: (
                "-87.65758750947528,41.96427269188822,0 "
                "-87.65802133507393,41.96581929055245,0 "
                "-87.65819033925305,41.96621846093642,0 "
                "-87.6583189819129,41.96650362897086,0 "
                "-87.65835858701473,41.96669002089185,0 "
                "-87.65838428411853,41.96688150295095,0 "
                "-87.65842208882658,41.96745896091846,0 "
                "-87.65846556843937,41.9683761425439,0 "
                "-87.65849296214573,41.96913893870342,0"
            ),
            2: (
                "-87.65492939166126,41.95377494531437,0 "
                "-87.65557043199591,41.95376544118533,0 "
                "-87.65606302030132,41.95376391658746,0 "
                "-87.65623502146268,41.95377379126367,0 "
                "-87.65634748981634,41.95380103566435,0 "
                "-87.65646537904269,41.95387703994676,0 "
                "-87.65656532461145,41.95396622645799,0 "
                "-87.65664760856414,41.95404201996044,0 "
                "-87.65671750555913,41.95416647054043,0 "
                "-87.65673983607117,41.95429949810849,0 "
                "-87.65673866475777,41.95441024240925,0 "
                "-87.6567690255541,41.95490657227902,0 "
                "-87.65683672482363,41.95692259283837,0 "
                "-87.6568900886376,41.95861070983142,0 "
                "-87.65699865558875,41.96181418669004,0 "
                "-87.65756347177603,41.96397045777844,0 "
                "-87.65758750947528,41.96427269188822,0"
            ),
            3: (
                "-87.65362593118043,41.94742799535678,0 "
                "-87.65363554415794,41.94819886386848,0 "
                "-87.6536456393239,41.95059994675451,0 "
                "-87.65365831235026,41.95108288489359,0 "
                "-87.6536604873874,41.9519954657554,0 "
                "-87.65362592053201,41.95245597302328,0 "
                "-87.65367158496069,41.95311153649393,0 "
                "-87.65368468595476,41.9533202828916,0 "
                "-87.65369271253692,41.95343095587119,0 "
                "-87.65373335834569,41.95351536301472,0 "
                "-87.65378605844126,41.95358212680591,0 "
                "-87.65385067928185,41.95364452823767,0 "
                "-87.6539390793817,41.95370263886964,0 "
                "-87.6540786298351,41.95373403675265,0 "
                "-87.65430648647626,41.9537535411832,0 "
                "-87.65492939166126,41.95377494531437,0"
            ),
            4: (
                "-87.65345391792157,41.94217681262115,0 "
                "-87.65342448305786,41.94237224420864,0 "
                "-87.65339745703922,41.94268217746244,0 "
                "-87.65337753982941,41.94288140770284,0 "
                "-87.65336256753105,41.94317369618263,0 "
                "-87.65338799707138,41.94357253961736,0 "
                "-87.65340240886648,41.94389158188269,0 "
                "-87.65341837392448,41.94406444407721,0 "
                "-87.65342275247338,41.94421065714904,0 "
                "-87.65347469646018,41.94434829382345,0 "
                "-87.65351486483024,41.94447699917548,0 "
                "-87.65353483605053,41.9453896864472,0 "
                "-87.65361975532807,41.94689193720703,0 "
                "-87.65362593118043,41.94742799535678,0"
            ),
        },
    }
)


@pytest.fixture(params=["rb", "r"])
def mode(request):
    return request.param


@pytest.fixture(params=[pytest.param("lxml", marks=td.skip_if_no("lxml")), "etree"])
def parser(request):
    return request.param


# FILE / URL


@td.skip_if_no("lxml")
def test_parser_consistency_file(datapath):
    filename = datapath("io", "data", "xml", "books.xml")
    df_file_lxml = read_xml(filename, parser="lxml")
    df_file_etree = read_xml(filename, parser="etree")

    tm.assert_frame_equal(df_file_lxml, df_file_etree)


@tm.network
@pytest.mark.slow
@td.skip_if_no("lxml")
@pytest.mark.skipif(
    not PY38,
    reason=("etree alpha ordered attributes < py3.8"),
)
def test_parser_consistency_url(datapath):
    url = (
        "https://data.cityofchicago.org/api/views/"
        "8pix-ypme/rows.xml?accessType=DOWNLOAD"
    )
    df_url_lxml = read_xml(url, xpath=".//row/row", parser="lxml")
    df_url_etree = read_xml(url, xpath=".//row/row", parser="etree")

    tm.assert_frame_equal(df_url_lxml, df_url_etree)


def test_file_like(datapath, parser, mode):
    filename = datapath("io", "data", "xml", "books.xml")
    with open(filename, mode) as f:
        df_file = read_xml(f, parser=parser)

    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_file, df_expected)


def test_file_io(datapath, parser, mode):
    filename = datapath("io", "data", "xml", "books.xml")
    with open(filename, mode) as f:
        xml_obj = f.read()

    df_io = read_xml(
        (BytesIO(xml_obj) if isinstance(xml_obj, bytes) else StringIO(xml_obj)),
        parser=parser,
    )

    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_io, df_expected)


def test_file_buffered_reader_string(datapath, parser, mode):
    filename = datapath("io", "data", "xml", "books.xml")
    with open(filename, mode) as f:
        xml_obj = f.read()

    df_str = read_xml(xml_obj, parser=parser)

    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_str, df_expected)


def test_file_buffered_reader_no_xml_declaration(datapath, parser, mode):
    filename = datapath("io", "data", "xml", "books.xml")
    with open(filename, mode) as f:
        next(f)
        xml_obj = f.read()

    df_str = read_xml(xml_obj, parser=parser)

    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_str, df_expected)


def test_file_handle_close(datapath, parser):
    xml_file = datapath("io", "data", "xml", "books.xml")

    with open(xml_file, "rb") as f:
        read_xml(BytesIO(f.read()), parser=parser)

        assert not f.closed


@td.skip_if_no("lxml")
@pytest.mark.parametrize("val", ["", b""])
def test_empty_string_lxml(val):
    from lxml.etree import XMLSyntaxError

    with pytest.raises(XMLSyntaxError, match="Document is empty"):
        read_xml(val, parser="lxml")


@pytest.mark.parametrize("val", ["", b""])
def test_empty_string_etree(val):
    from xml.etree.ElementTree import ParseError

    with pytest.raises(ParseError, match="no element found"):
        read_xml(val, parser="etree")


@td.skip_if_no("lxml")
def test_wrong_file_path_lxml():
    from lxml.etree import XMLSyntaxError

    filename = os.path.join("data", "html", "books.xml")

    with pytest.raises(
        XMLSyntaxError,
        match=("Start tag expected, '<' not found"),
    ):
        read_xml(filename, parser="lxml")


def test_wrong_file_path_etree():
    from xml.etree.ElementTree import ParseError

    filename = os.path.join("data", "html", "books.xml")

    with pytest.raises(
        ParseError,
        match=("not well-formed"),
    ):
        read_xml(filename, parser="etree")


@tm.network
@td.skip_if_no("lxml")
def test_url():
    url = "https://www.w3schools.com/xml/books.xml"
    df_url = read_xml(url, xpath=".//book[count(*)=4]")

    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
            "cover": [None, None, "paperback"],
        }
    )

    tm.assert_frame_equal(df_url, df_expected)


def test_wrong_url(parser):
    with pytest.raises(HTTPError, match=("HTTP Error 404: Not Found")):
        url = "https://www.w3schools.com/xml/python.xml"
        read_xml(url, xpath=".//book[count(*)=4]", parser=parser)


# XPATH


@td.skip_if_no("lxml")
def test_empty_xpath_lxml(datapath):
    filename = datapath("io", "data", "xml", "books.xml")
    with pytest.raises(ValueError, match=("xpath does not return any nodes")):
        read_xml(filename, xpath=".//python", parser="lxml")


def test_bad_xpath_etree(datapath):
    filename = datapath("io", "data", "xml", "books.xml")
    with pytest.raises(
        SyntaxError, match=("You have used an incorrect or unsupported XPath")
    ):
        read_xml(filename, xpath=".//[book]", parser="etree")


@td.skip_if_no("lxml")
def test_bad_xpath_lxml(datapath):
    from lxml.etree import XPathEvalError

    filename = datapath("io", "data", "xml", "books.xml")
    with pytest.raises(XPathEvalError, match=("Invalid expression")):
        read_xml(filename, xpath=".//[book]", parser="lxml")


# NAMESPACE


def test_default_namespace(parser):
    df_nmsp = read_xml(
        xml_default_nmsp,
        xpath=".//ns:row",
        namespaces={"ns": "http://example.com"},
        parser=parser,
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    tm.assert_frame_equal(df_nmsp, df_expected)


def test_prefix_namespace(parser):
    df_nmsp = read_xml(
        xml_prefix_nmsp,
        xpath=".//doc:row",
        namespaces={"doc": "http://example.com"},
        parser=parser,
    )

    df_expected = DataFrame(
        {
            "shape": ["square", "circle", "triangle"],
            "degrees": [360, 360, 180],
            "sides": [4.0, float("nan"), 3.0],
        }
    )

    tm.assert_frame_equal(df_nmsp, df_expected)


@td.skip_if_no("lxml")
def test_consistency_default_namespace():
    df_lxml = read_xml(
        xml_default_nmsp,
        xpath=".//ns:row",
        namespaces={"ns": "http://example.com"},
        parser="lxml",
    )

    df_etree = read_xml(
        xml_default_nmsp,
        xpath=".//doc:row",
        namespaces={"doc": "http://example.com"},
        parser="etree",
    )

    tm.assert_frame_equal(df_lxml, df_etree)


@td.skip_if_no("lxml")
def test_consistency_prefix_namespace():
    df_lxml = read_xml(
        xml_prefix_nmsp,
        xpath=".//doc:row",
        namespaces={"doc": "http://example.com"},
        parser="lxml",
    )

    df_etree = read_xml(
        xml_prefix_nmsp,
        xpath=".//doc:row",
        namespaces={"doc": "http://example.com"},
        parser="etree",
    )

    tm.assert_frame_equal(df_lxml, df_etree)


# PREFIX


def test_missing_prefix_with_default_namespace(datapath, parser):
    filename = datapath("io", "data", "xml", "books.xml")
    with pytest.raises(ValueError, match=("xpath does not return any nodes")):
        read_xml(filename, xpath=".//Placemark", parser=parser)


def test_missing_prefix_definition_etree(datapath):
    filename = datapath("io", "data", "xml", "cta_rail_lines.kml")
    with pytest.raises(SyntaxError, match=("you used an undeclared namespace prefix")):
        read_xml(filename, xpath=".//kml:Placemark", parser="etree")


@td.skip_if_no("lxml")
def test_missing_prefix_definition_lxml(datapath):
    from lxml.etree import XPathEvalError

    filename = datapath("io", "data", "xml", "cta_rail_lines.kml")
    with pytest.raises(XPathEvalError, match=("Undefined namespace prefix")):
        read_xml(filename, xpath=".//kml:Placemark", parser="lxml")


@td.skip_if_no("lxml")
@pytest.mark.parametrize("key", ["", None])
def test_none_namespace_prefix(key):
    with pytest.raises(
        TypeError, match=("empty namespace prefix is not supported in XPath")
    ):
        read_xml(
            xml_default_nmsp,
            xpath=".//kml:Placemark",
            namespaces={key: "http://www.opengis.net/kml/2.2"},
            parser="lxml",
        )


# ELEMS AND ATTRS


def test_file_elems_and_attrs(datapath, parser):
    filename = datapath("io", "data", "xml", "books.xml")
    df_file = read_xml(filename, parser=parser)
    df_expected = DataFrame(
        {
            "category": ["cooking", "children", "web"],
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_file, df_expected)


def test_file_only_attrs(datapath, parser):
    filename = datapath("io", "data", "xml", "books.xml")
    df_file = read_xml(filename, attrs_only=True, parser=parser)
    df_expected = DataFrame({"category": ["cooking", "children", "web"]})

    tm.assert_frame_equal(df_file, df_expected)


def test_file_only_elems(datapath, parser):
    filename = datapath("io", "data", "xml", "books.xml")
    df_file = read_xml(filename, elems_only=True, parser=parser)
    df_expected = DataFrame(
        {
            "title": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "author": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "year": [2005, 2005, 2003],
            "price": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_file, df_expected)


def test_elem_and_attrs_only(datapath, parser):
    filename = datapath("io", "data", "xml", "cta_rail_lines.kml")
    with pytest.raises(
        ValueError,
        match=("Either element or attributes can be parsed not both"),
    ):
        read_xml(filename, elems_only=True, attrs_only=True, parser=parser)


@td.skip_if_no("lxml")
def test_attribute_centric_xml():
    xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<TrainSchedule>
      <Stations>
         <station Name="Manhattan" coords="31,460,195,498"/>
         <station Name="Laraway Road" coords="63,409,194,455"/>
         <station Name="179th St (Orland Park)" coords="0,364,110,395"/>
         <station Name="153rd St (Orland Park)" coords="7,333,113,362"/>
         <station Name="143rd St (Orland Park)" coords="17,297,115,330"/>
         <station Name="Palos Park" coords="128,281,239,303"/>
         <station Name="Palos Heights" coords="148,257,283,279"/>
         <station Name="Worth" coords="170,230,248,255"/>
         <station Name="Chicago Ridge" coords="70,187,208,214"/>
         <station Name="Oak Lawn" coords="166,159,266,185"/>
         <station Name="Ashburn" coords="197,133,336,157"/>
         <station Name="Wrightwood" coords="219,106,340,133"/>
         <station Name="Chicago Union Sta" coords="220,0,360,43"/>
      </Stations>
</TrainSchedule>"""

    df_lxml = read_xml(xml, xpath=".//station")
    df_etree = read_xml(xml, xpath=".//station", parser="etree")

    tm.assert_frame_equal(df_lxml, df_etree)


# NAMES


def test_names_option_output(datapath, parser):
    filename = datapath("io", "data", "xml", "books.xml")
    df_file = read_xml(
        filename, names=["Col1", "Col2", "Col3", "Col4", "Col5"], parser=parser
    )

    df_expected = DataFrame(
        {
            "Col1": ["cooking", "children", "web"],
            "Col2": ["Everyday Italian", "Harry Potter", "Learning XML"],
            "Col3": ["Giada De Laurentiis", "J K. Rowling", "Erik T. Ray"],
            "Col4": [2005, 2005, 2003],
            "Col5": [30.00, 29.99, 39.95],
        }
    )

    tm.assert_frame_equal(df_file, df_expected)


def test_names_option_wrong_length(datapath, parser):
    filename = datapath("io", "data", "xml", "books.xml")

    with pytest.raises(ValueError, match=("names does not match length")):
        read_xml(filename, names=["Col1", "Col2", "Col3"], parser=parser)


def test_names_option_wrong_type(datapath, parser):
    filename = datapath("io", "data", "xml", "books.xml")

    with pytest.raises(TypeError, match=("is not a valid type for names")):
        read_xml(
            filename, names="Col1, Col2, Col3", parser=parser  # type: ignore[arg-type]
        )


# ENCODING


def test_wrong_encoding(datapath, parser):
    filename = datapath("io", "data", "xml", "baby_names.xml")
    with pytest.raises(UnicodeDecodeError, match=("'utf-8' codec can't decode")):
        read_xml(filename, parser=parser)


def test_utf16_encoding(datapath, parser):
    filename = datapath("io", "data", "xml", "baby_names.xml")
    with pytest.raises(
        UnicodeError,
        match=(
            "UTF-16 stream does not start with BOM|"
            "'utf-16-le' codec can't decode byte"
        ),
    ):
        read_xml(filename, encoding="UTF-16", parser=parser)


def test_unknown_encoding(datapath, parser):
    filename = datapath("io", "data", "xml", "baby_names.xml")
    with pytest.raises(LookupError, match=("unknown encoding: uft-8")):
        read_xml(filename, encoding="UFT-8", parser=parser)


def test_ascii_encoding(datapath, parser):
    filename = datapath("io", "data", "xml", "baby_names.xml")
    with pytest.raises(UnicodeDecodeError, match=("'ascii' codec can't decode byte")):
        read_xml(filename, encoding="ascii", parser=parser)


@td.skip_if_no("lxml")
def test_parser_consistency_with_encoding(datapath):
    filename = datapath("io", "data", "xml", "baby_names.xml")
    df_lxml = read_xml(filename, parser="lxml", encoding="ISO-8859-1")
    df_etree = read_xml(filename, parser="etree", encoding="iso-8859-1")

    tm.assert_frame_equal(df_lxml, df_etree)


# PARSER


@td.skip_if_installed("lxml")
def test_default_parser_no_lxml(datapath):
    filename = datapath("io", "data", "xml", "books.xml")

    with pytest.raises(
        ImportError, match=("lxml not found, please install or use the etree parser.")
    ):
        read_xml(filename)


def test_wrong_parser(datapath):
    filename = datapath("io", "data", "xml", "books.xml")

    with pytest.raises(
        ValueError, match=("Values for parser can only be lxml or etree.")
    ):
        read_xml(filename, parser="bs4")


# STYLESHEET


@td.skip_if_no("lxml")
def test_stylesheet_file(datapath):
    kml = datapath("io", "data", "xml", "cta_rail_lines.kml")
    xsl = datapath("io", "data", "xml", "flatten_doc.xsl")

    df_style = read_xml(
        kml,
        xpath=".//k:Placemark",
        namespaces={"k": "http://www.opengis.net/kml/2.2"},
        stylesheet=xsl,
    )

    tm.assert_frame_equal(df_kml, df_style)


@td.skip_if_no("lxml")
def test_stylesheet_file_like(datapath, mode):
    kml = datapath("io", "data", "xml", "cta_rail_lines.kml")
    xsl = datapath("io", "data", "xml", "flatten_doc.xsl")

    with open(xsl, mode) as f:
        df_style = read_xml(
            kml,
            xpath=".//k:Placemark",
            namespaces={"k": "http://www.opengis.net/kml/2.2"},
            stylesheet=f,
        )

    tm.assert_frame_equal(df_kml, df_style)


@td.skip_if_no("lxml")
def test_stylesheet_io(datapath, mode):
    kml = datapath("io", "data", "xml", "cta_rail_lines.kml")
    xsl = datapath("io", "data", "xml", "flatten_doc.xsl")

    xsl_obj: BytesIO | StringIO

    with open(xsl, mode) as f:
        if mode == "rb":
            xsl_obj = BytesIO(f.read())
        else:
            xsl_obj = StringIO(f.read())

    df_style = read_xml(
        kml,
        xpath=".//k:Placemark",
        namespaces={"k": "http://www.opengis.net/kml/2.2"},
        stylesheet=xsl_obj,
    )

    tm.assert_frame_equal(df_kml, df_style)


@td.skip_if_no("lxml")
def test_stylesheet_buffered_reader(datapath, mode):
    kml = datapath("io", "data", "xml", "cta_rail_lines.kml")
    xsl = datapath("io", "data", "xml", "flatten_doc.xsl")

    with open(xsl, mode) as f:
        xsl_obj = f.read()

    df_style = read_xml(
        kml,
        xpath=".//k:Placemark",
        namespaces={"k": "http://www.opengis.net/kml/2.2"},
        stylesheet=xsl_obj,
    )

    tm.assert_frame_equal(df_kml, df_style)


@td.skip_if_no("lxml")
def test_not_stylesheet(datapath):
    from lxml.etree import XSLTParseError

    kml = datapath("io", "data", "xml", "cta_rail_lines.kml")
    xsl = datapath("io", "data", "xml", "books.xml")

    with pytest.raises(XSLTParseError, match=("document is not a stylesheet")):
        read_xml(kml, stylesheet=xsl)


@td.skip_if_no("lxml")
def test_incorrect_xsl_syntax(datapath):
    from lxml.etree import XMLSyntaxError

    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                              xmlns:k="http://www.opengis.net/kml/2.2"/>
    <xsl:output method="xml" omit-xml-declaration="yes"
                cdata-section-elements="k:description" indent="yes"/>
    <xsl:strip-space elements="*"/>

    <xsl:template match="node()|@*">
     <xsl:copy>
       <xsl:apply-templates select="node()|@*"/>
     </xsl:copy>
    </xsl:template>

    <xsl:template match="k:MultiGeometry|k:LineString">
        <xsl:apply-templates select='*'/>
    </xsl:template>

    <xsl:template match="k:description|k:Snippet|k:Style"/>
</xsl:stylesheet>"""

    kml = datapath("io", "data", "xml", "cta_rail_lines.kml")

    with pytest.raises(
        XMLSyntaxError, match=("Extra content at the end of the document")
    ):
        read_xml(kml, stylesheet=xsl)


@td.skip_if_no("lxml")
def test_incorrect_xsl_eval(datapath):
    from lxml.etree import XSLTParseError

    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                              xmlns:k="http://www.opengis.net/kml/2.2">
    <xsl:output method="xml" omit-xml-declaration="yes"
                cdata-section-elements="k:description" indent="yes"/>
    <xsl:strip-space elements="*"/>

    <xsl:template match="node(*)|@*">
     <xsl:copy>
       <xsl:apply-templates select="node()|@*"/>
     </xsl:copy>
    </xsl:template>

    <xsl:template match="k:MultiGeometry|k:LineString">
        <xsl:apply-templates select='*'/>
    </xsl:template>

    <xsl:template match="k:description|k:Snippet|k:Style"/>
</xsl:stylesheet>"""

    kml = datapath("io", "data", "xml", "cta_rail_lines.kml")

    with pytest.raises(XSLTParseError, match=("failed to compile")):
        read_xml(kml, stylesheet=xsl)


@td.skip_if_no("lxml")
def test_incorrect_xsl_apply(datapath):
    from lxml.etree import XSLTApplyError

    xsl = """\
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="xml" encoding="utf-8" indent="yes" />
    <xsl:strip-space elements="*"/>

    <xsl:template match="@*|node()">
        <xsl:copy>
            <xsl:copy-of select="document('non_existent.xml')/*"/>
        </xsl:copy>
    </xsl:template>
</xsl:stylesheet>"""

    kml = datapath("io", "data", "xml", "cta_rail_lines.kml")

    with pytest.raises(XSLTApplyError, match=("Cannot resolve URI")):
        read_xml(kml, stylesheet=xsl)


@td.skip_if_no("lxml")
def test_wrong_stylesheet():
    from lxml.etree import XMLSyntaxError

    kml = os.path.join("data", "xml", "cta_rail_lines.kml")
    xsl = os.path.join("data", "xml", "flatten.xsl")

    with pytest.raises(
        XMLSyntaxError,
        match=("Start tag expected, '<' not found"),
    ):
        read_xml(kml, stylesheet=xsl)


@td.skip_if_no("lxml")
def test_stylesheet_file_close(datapath, mode):
    kml = datapath("io", "data", "xml", "cta_rail_lines.kml")
    xsl = datapath("io", "data", "xml", "flatten_doc.xsl")

    xsl_obj: BytesIO | StringIO

    with open(xsl, mode) as f:
        if mode == "rb":
            xsl_obj = BytesIO(f.read())
        else:
            xsl_obj = StringIO(f.read())

        read_xml(kml, stylesheet=xsl_obj)

        assert not f.closed


@td.skip_if_no("lxml")
def test_stylesheet_with_etree(datapath):
    kml = os.path.join("data", "xml", "cta_rail_lines.kml")
    xsl = os.path.join("data", "xml", "flatten_doc.xsl")

    with pytest.raises(
        ValueError, match=("To use stylesheet, you need lxml installed")
    ):
        read_xml(kml, parser="etree", stylesheet=xsl)


@td.skip_if_no("lxml")
@pytest.mark.parametrize("val", ["", b""])
def test_empty_stylesheet(val):
    from lxml.etree import XMLSyntaxError

    kml = os.path.join("data", "xml", "cta_rail_lines.kml")

    with pytest.raises(
        XMLSyntaxError, match=("Document is empty|Start tag expected, '<' not found")
    ):
        read_xml(kml, stylesheet=val)


@tm.network
@td.skip_if_no("lxml")
def test_online_stylesheet():
    xml = "https://www.w3schools.com/xml/cdcatalog_with_xsl.xml"
    xsl = "https://www.w3schools.com/xml/cdcatalog.xsl"

    df_xsl = read_xml(
        xml,
        xpath=".//tr[td and position() <= 6]",
        names=["title", "artist"],
        stylesheet=xsl,
    )

    df_expected = DataFrame(
        {
            "title": {
                0: "Empire Burlesque",
                1: "Hide your heart",
                2: "Greatest Hits",
                3: "Still got the blues",
                4: "Eros",
            },
            "artist": {
                0: "Bob Dylan",
                1: "Bonnie Tyler",
                2: "Dolly Parton",
                3: "Gary Moore",
                4: "Eros Ramazzotti",
            },
        }
    )

    tm.assert_frame_equal(df_expected, df_xsl)


# COMPRESSION


@pytest.mark.parametrize("comp", ["bz2", "gzip", "xz", "zip"])
def test_compression_read(parser, comp):
    with tm.ensure_clean() as path:
        geom_df.to_xml(path, index=False, parser=parser, compression=comp)

        xml_df = read_xml(path, parser=parser, compression=comp)

    tm.assert_frame_equal(xml_df, geom_df)


@pytest.mark.parametrize("comp", ["gzip", "xz", "zip"])
def test_wrong_compression_bz2(parser, comp):
    with tm.ensure_clean() as path:
        geom_df.to_xml(path, parser=parser, compression=comp)

        with pytest.raises(OSError, match="Invalid data stream"):
            read_xml(path, parser=parser, compression="bz2")


@pytest.mark.parametrize("comp", ["bz2", "xz", "zip"])
def test_wrong_compression_gz(parser, comp):
    with tm.ensure_clean() as path:
        geom_df.to_xml(path, parser=parser, compression=comp)

        with pytest.raises(OSError, match="Not a gzipped file"):
            read_xml(path, parser=parser, compression="gzip")


@pytest.mark.parametrize("comp", ["bz2", "gzip", "zip"])
def test_wrong_compression_xz(parser, comp):
    from lzma import LZMAError

    with tm.ensure_clean() as path:
        geom_df.to_xml(path, parser=parser, compression=comp)

        with pytest.raises(LZMAError, match="Input format not supported by decoder"):
            read_xml(path, parser=parser, compression="xz")


@pytest.mark.parametrize("comp", ["bz2", "gzip", "xz"])
def test_wrong_compression_zip(parser, comp):
    from zipfile import BadZipFile

    with tm.ensure_clean() as path:
        geom_df.to_xml(path, parser=parser, compression=comp)

        with pytest.raises(BadZipFile, match="File is not a zip file"):
            read_xml(path, parser=parser, compression="zip")


def test_unsuported_compression(datapath, parser):
    with pytest.raises(ValueError, match="Unrecognized compression type"):
        with tm.ensure_clean() as path:
            read_xml(path, parser=parser, compression="7z")


# STORAGE OPTIONS


@tm.network
@td.skip_if_no("s3fs")
@td.skip_if_no("lxml")
def test_s3_parser_consistency():
    # Python Software Foundation (2019 IRS-990 RETURN)
    s3 = "s3://irs-form-990/201923199349319487_public.xml"

    df_lxml = read_xml(
        s3,
        xpath=".//irs:Form990PartVIISectionAGrp",
        namespaces={"irs": "http://www.irs.gov/efile"},
        parser="lxml",
        storage_options={"anon": True},
    )

    df_etree = read_xml(
        s3,
        xpath=".//irs:Form990PartVIISectionAGrp",
        namespaces={"irs": "http://www.irs.gov/efile"},
        parser="etree",
        storage_options={"anon": True},
    )

    tm.assert_frame_equal(df_lxml, df_etree)
