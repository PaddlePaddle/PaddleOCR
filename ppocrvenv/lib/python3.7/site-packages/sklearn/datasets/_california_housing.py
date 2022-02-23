"""California housing dataset.

The original database is available from StatLib

    http://lib.stat.cmu.edu/datasets/

The data contains 20,640 observations on 9 variables.

This dataset contains the average house value as target variable
and the following input variables (features): average income,
housing average age, average rooms, average bedrooms, population,
average occupation, latitude, and longitude in that order.

References
----------

Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
Statistics and Probability Letters, 33 (1997) 291-297.

"""
# Authors: Peter Prettenhofer
# License: BSD 3 clause

from os.path import exists
from os import makedirs, remove
import tarfile

import numpy as np
import logging

import joblib

from . import get_data_home
from ._base import _convert_data_dataframe
from ._base import _fetch_remote
from ._base import _pkl_filepath
from ._base import RemoteFileMetadata
from ._base import load_descr
from ..utils import Bunch


# The original data can be found at:
# https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz
ARCHIVE = RemoteFileMetadata(
    filename="cal_housing.tgz",
    url="https://ndownloader.figshare.com/files/5976036",
    checksum="aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681",
)

logger = logging.getLogger(__name__)


def fetch_california_housing(
    *, data_home=None, download_if_missing=True, return_X_y=False, as_frame=False
):
    """Load the California housing dataset (regression).

    ==============   ==============
    Samples total             20640
    Dimensionality                8
    Features                   real
    Target           real 0.15 - 5.
    ==============   ==============

    Read more in the :ref:`User Guide <california_housing_dataset>`.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : bool, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.


    return_X_y : bool, default=False.
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

        .. versionadded:: 0.20

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical). The target is
        a pandas DataFrame or Series depending on the number of target_columns.

        .. versionadded:: 0.23

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray, shape (20640, 8)
            Each row corresponding to the 8 feature values in order.
            If ``as_frame`` is True, ``data`` is a pandas object.
        target : numpy array of shape (20640,)
            Each value corresponds to the average
            house value in units of 100,000.
            If ``as_frame`` is True, ``target`` is a pandas object.
        feature_names : list of length 8
            Array of ordered feature names used in the dataset.
        DESCR : str
            Description of the California housing dataset.
        frame : pandas DataFrame
            Only present when `as_frame=True`. DataFrame with ``data`` and
            ``target``.

            .. versionadded:: 0.23

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.20

    Notes
    -----

    This dataset consists of 20,640 samples and 9 features.
    """
    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)

    filepath = _pkl_filepath(data_home, "cal_housing.pkz")
    if not exists(filepath):
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        logger.info(
            "Downloading Cal. housing from {} to {}".format(ARCHIVE.url, data_home)
        )

        archive_path = _fetch_remote(ARCHIVE, dirname=data_home)

        with tarfile.open(mode="r:gz", name=archive_path) as f:
            cal_housing = np.loadtxt(
                f.extractfile("CaliforniaHousing/cal_housing.data"), delimiter=","
            )
            # Columns are not in the same order compared to the previous
            # URL resource on lib.stat.cmu.edu
            columns_index = [8, 7, 2, 3, 4, 5, 6, 1, 0]
            cal_housing = cal_housing[:, columns_index]

            joblib.dump(cal_housing, filepath, compress=6)
        remove(archive_path)

    else:
        cal_housing = joblib.load(filepath)

    feature_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    target, data = cal_housing[:, 0], cal_housing[:, 1:]

    # avg rooms = total rooms / households
    data[:, 2] /= data[:, 5]

    # avg bed rooms = total bed rooms / households
    data[:, 3] /= data[:, 5]

    # avg occupancy = population / households
    data[:, 5] = data[:, 4] / data[:, 5]

    # target in units of 100,000
    target = target / 100000.0

    descr = load_descr("california_housing.rst")

    X = data
    y = target

    frame = None
    target_names = [
        "MedHouseVal",
    ]
    if as_frame:
        frame, X, y = _convert_data_dataframe(
            "fetch_california_housing", data, target, feature_names, target_names
        )

    if return_X_y:
        return X, y

    return Bunch(
        data=X,
        target=y,
        frame=frame,
        target_names=target_names,
        feature_names=feature_names,
        DESCR=descr,
    )
