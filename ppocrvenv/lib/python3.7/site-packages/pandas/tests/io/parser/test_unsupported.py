"""
Tests that features that are currently unsupported in
either the Python or C parser are actually enforced
and are clearly communicated to the user.

Ultimately, the goal is to remove test cases from this
test suite as new feature support is added to the parsers.
"""
from io import StringIO

import pytest

from pandas.errors import ParserError

import pandas._testing as tm

from pandas.io.parsers import read_csv
import pandas.io.parsers.readers as parsers


@pytest.fixture(params=["python", "python-fwf"], ids=lambda val: val)
def python_engine(request):
    return request.param


class TestUnsupportedFeatures:
    def test_mangle_dupe_cols_false(self):
        # see gh-12935
        data = "a b c\n1 2 3"
        msg = "is not supported"

        for engine in ("c", "python"):
            with pytest.raises(ValueError, match=msg):
                read_csv(StringIO(data), engine=engine, mangle_dupe_cols=False)

    def test_c_engine(self):
        # see gh-6607
        data = "a b c\n1 2 3"
        msg = "does not support"

        # specify C engine with unsupported options (raise)
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine="c", sep=None, delim_whitespace=False)
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine="c", sep=r"\s")
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine="c", sep="\t", quotechar=chr(128))
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine="c", skipfooter=1)

        # specify C-unsupported options without python-unsupported options
        with tm.assert_produces_warning(parsers.ParserWarning):
            read_csv(StringIO(data), sep=None, delim_whitespace=False)
        with tm.assert_produces_warning(parsers.ParserWarning):
            read_csv(StringIO(data), sep=r"\s")
        with tm.assert_produces_warning(parsers.ParserWarning):
            read_csv(StringIO(data), sep="\t", quotechar=chr(128))
        with tm.assert_produces_warning(parsers.ParserWarning):
            read_csv(StringIO(data), skipfooter=1)

        text = """                      A       B       C       D        E
one two three   four
a   b   10.0032 5    -0.5109 -2.3358 -0.4645  0.05076  0.3640
a   q   20      4     0.4473  1.4152  0.2834  1.00661  0.1744
x   q   30      3    -0.6662 -0.5243 -0.3580  0.89145  2.5838"""
        msg = "Error tokenizing data"

        with pytest.raises(ParserError, match=msg):
            read_csv(StringIO(text), sep="\\s+")
        with pytest.raises(ParserError, match=msg):
            read_csv(StringIO(text), engine="c", sep="\\s+")

        msg = "Only length-1 thousands markers supported"
        data = """A|B|C
1|2,334|5
10|13|10.
"""
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), thousands=",,")
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), thousands="")

        msg = "Only length-1 line terminators supported"
        data = "a,b,c~~1,2,3~~4,5,6"
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), lineterminator="~~")

    def test_python_engine(self, python_engine):
        from pandas.io.parsers.readers import _python_unsupported as py_unsupported

        data = """1,2,3,,
1,2,3,4,
1,2,3,4,5
1,2,,,
1,2,3,4,"""

        for default in py_unsupported:
            msg = (
                f"The {repr(default)} option is not "
                f"supported with the {repr(python_engine)} engine"
            )

            kwargs = {default: object()}
            with pytest.raises(ValueError, match=msg):
                read_csv(StringIO(data), engine=python_engine, **kwargs)

    def test_python_engine_file_no_next(self, python_engine):
        # see gh-16530
        class NoNextBuffer:
            def __init__(self, csv_data):
                self.data = csv_data

            def __iter__(self):
                return self

            def read(self):
                return self.data

        data = "a\n1"
        msg = "The 'python' engine cannot iterate"

        with pytest.raises(ValueError, match=msg):
            read_csv(NoNextBuffer(data), engine=python_engine)
