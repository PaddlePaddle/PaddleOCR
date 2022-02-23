"""
Testing that we work in the downstream packages
"""
import importlib
import subprocess
import sys

import numpy as np  # noqa
import pytest

import pandas.util._test_decorators as td

from pandas import DataFrame
import pandas._testing as tm


def import_module(name):
    # we *only* want to skip if the module is truly not available
    # and NOT just an actual import error because of pandas changes

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.skip(f"skipping as {name} not available")


@pytest.fixture
def df():
    return DataFrame({"A": [1, 2, 3]})


# TODO(ArrayManager) dask is still accessing the blocks
# https://github.com/dask/dask/pull/7318
@td.skip_array_manager_not_yet_implemented
def test_dask(df):

    toolz = import_module("toolz")  # noqa
    dask = import_module("dask")  # noqa

    import dask.dataframe as dd

    ddf = dd.from_pandas(df, npartitions=3)
    assert ddf.A is not None
    assert ddf.compute() is not None


def test_xarray(df):

    xarray = import_module("xarray")  # noqa

    assert df.to_xarray() is not None


@td.skip_if_no("cftime")
@td.skip_if_no("xarray", "0.10.4")
def test_xarray_cftimeindex_nearest():
    # https://github.com/pydata/xarray/issues/3751
    import cftime
    import xarray

    times = xarray.cftime_range("0001", periods=2)
    result = times.get_loc(cftime.DatetimeGregorian(2000, 1, 1), method="nearest")
    expected = 1
    assert result == expected


def test_oo_optimizable():
    # GH 21071
    subprocess.check_call([sys.executable, "-OO", "-c", "import pandas"])


def test_oo_optimized_datetime_index_unpickle():
    # GH 42866
    subprocess.check_call(
        [
            sys.executable,
            "-OO",
            "-c",
            (
                "import pandas as pd, pickle; "
                "pickle.loads(pickle.dumps(pd.date_range('2021-01-01', periods=1)))"
            ),
        ]
    )


@tm.network
# Cython import warning
@pytest.mark.filterwarnings("ignore:pandas.util.testing is deprecated")
@pytest.mark.filterwarnings("ignore:can't:ImportWarning")
@pytest.mark.filterwarnings(
    # patsy needs to update their imports
    "ignore:Using or importing the ABCs from 'collections:DeprecationWarning"
)
def test_statsmodels():

    statsmodels = import_module("statsmodels")  # noqa
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = sm.datasets.get_rdataset("Guerry", "HistData").data
    smf.ols("Lottery ~ Literacy + np.log(Pop1831)", data=df).fit()


# Cython import warning
@pytest.mark.filterwarnings("ignore:can't:ImportWarning")
def test_scikit_learn(df):

    sklearn = import_module("sklearn")  # noqa
    from sklearn import (
        datasets,
        svm,
    )

    digits = datasets.load_digits()
    clf = svm.SVC(gamma=0.001, C=100.0)
    clf.fit(digits.data[:-1], digits.target[:-1])
    clf.predict(digits.data[-1:])


# Cython import warning and traitlets
@tm.network
@pytest.mark.filterwarnings("ignore")
def test_seaborn():

    seaborn = import_module("seaborn")
    tips = seaborn.load_dataset("tips")
    seaborn.stripplot(x="day", y="total_bill", data=tips)


def test_pandas_gbq(df):

    pandas_gbq = import_module("pandas_gbq")  # noqa


@pytest.mark.xfail(reason="0.8.1 tries to import urlencode from pd.io.common")
@tm.network
def test_pandas_datareader():

    pandas_datareader = import_module("pandas_datareader")
    pandas_datareader.DataReader("F", "quandl", "2017-01-01", "2017-02-01")


# importing from pandas, Cython import warning
@pytest.mark.filterwarnings("ignore:can't resolve:ImportWarning")
def test_geopandas():

    geopandas = import_module("geopandas")
    fp = geopandas.datasets.get_path("naturalearth_lowres")
    assert geopandas.read_file(fp) is not None


# Cython import warning
@pytest.mark.filterwarnings("ignore:can't resolve:ImportWarning")
@pytest.mark.filterwarnings("ignore:RangeIndex.* is deprecated:DeprecationWarning")
def test_pyarrow(df):

    pyarrow = import_module("pyarrow")
    table = pyarrow.Table.from_pandas(df)
    result = table.to_pandas()
    tm.assert_frame_equal(result, df)


def test_missing_required_dependency():
    # GH 23868
    # To ensure proper isolation, we pass these flags
    # -S : disable site-packages
    # -s : disable user site-packages
    # -E : disable PYTHON* env vars, especially PYTHONPATH
    # https://github.com/MacPython/pandas-wheels/pull/50

    pyexe = sys.executable.replace("\\", "/")

    # We skip this test if pandas is installed as a site package. We first
    # import the package normally and check the path to the module before
    # executing the test which imports pandas with site packages disabled.
    call = [pyexe, "-c", "import pandas;print(pandas.__file__)"]
    output = subprocess.check_output(call).decode()
    if "site-packages" in output:
        pytest.skip("pandas installed as site package")

    # This test will fail if pandas is installed as a site package. The flags
    # prevent pandas being imported and the test will report Failed: DID NOT
    # RAISE <class 'subprocess.CalledProcessError'>
    call = [pyexe, "-sSE", "-c", "import pandas"]

    msg = (
        rf"Command '\['{pyexe}', '-sSE', '-c', 'import pandas'\]' "
        "returned non-zero exit status 1."
    )

    with pytest.raises(subprocess.CalledProcessError, match=msg) as exc:
        subprocess.check_output(call, stderr=subprocess.STDOUT)

    output = exc.value.stdout.decode()
    for name in ["numpy", "pytz", "dateutil"]:
        assert name in output
