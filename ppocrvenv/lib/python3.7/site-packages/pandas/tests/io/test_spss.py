from pathlib import Path

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

pyreadstat = pytest.importorskip("pyreadstat")


@pytest.mark.parametrize("path_klass", [lambda p: p, Path])
def test_spss_labelled_num(path_klass, datapath):
    # test file from the Haven project (https://haven.tidyverse.org/)
    fname = path_klass(datapath("io", "data", "spss", "labelled-num.sav"))

    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame({"VAR00002": "This is one"}, index=[0])
    expected["VAR00002"] = pd.Categorical(expected["VAR00002"])
    tm.assert_frame_equal(df, expected)

    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({"VAR00002": 1.0}, index=[0])
    tm.assert_frame_equal(df, expected)


def test_spss_labelled_num_na(datapath):
    # test file from the Haven project (https://haven.tidyverse.org/)
    fname = datapath("io", "data", "spss", "labelled-num-na.sav")

    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame({"VAR00002": ["This is one", None]})
    expected["VAR00002"] = pd.Categorical(expected["VAR00002"])
    tm.assert_frame_equal(df, expected)

    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({"VAR00002": [1.0, np.nan]})
    tm.assert_frame_equal(df, expected)


def test_spss_labelled_str(datapath):
    # test file from the Haven project (https://haven.tidyverse.org/)
    fname = datapath("io", "data", "spss", "labelled-str.sav")

    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame({"gender": ["Male", "Female"]})
    expected["gender"] = pd.Categorical(expected["gender"])
    tm.assert_frame_equal(df, expected)

    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({"gender": ["M", "F"]})
    tm.assert_frame_equal(df, expected)


def test_spss_umlauts(datapath):
    # test file from the Haven project (https://haven.tidyverse.org/)
    fname = datapath("io", "data", "spss", "umlauts.sav")

    df = pd.read_spss(fname, convert_categoricals=True)
    expected = pd.DataFrame(
        {"var1": ["the ä umlaut", "the ü umlaut", "the ä umlaut", "the ö umlaut"]}
    )
    expected["var1"] = pd.Categorical(expected["var1"])
    tm.assert_frame_equal(df, expected)

    df = pd.read_spss(fname, convert_categoricals=False)
    expected = pd.DataFrame({"var1": [1.0, 2.0, 1.0, 3.0]})
    tm.assert_frame_equal(df, expected)


def test_spss_usecols(datapath):
    # usecols must be list-like
    fname = datapath("io", "data", "spss", "labelled-num.sav")

    with pytest.raises(TypeError, match="usecols must be list-like."):
        pd.read_spss(fname, usecols="VAR00002")
