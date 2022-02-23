"""Forest covertype dataset.

A classic dataset for classification benchmarks, featuring categorical and
real-valued features.

The dataset page is available from UCI Machine Learning Repository

    https://archive.ics.uci.edu/ml/datasets/Covertype

Courtesy of Jock A. Blackard and Colorado State University.
"""

# Author: Lars Buitinck
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
# License: BSD 3 clause

from gzip import GzipFile
import logging
from os.path import exists, join
from os import remove, makedirs

import numpy as np
import joblib

from . import get_data_home
from ._base import _convert_data_dataframe
from ._base import _fetch_remote
from ._base import RemoteFileMetadata
from ._base import load_descr
from ..utils import Bunch
from ._base import _pkl_filepath
from ..utils import check_random_state


# The original data can be found in:
# https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
ARCHIVE = RemoteFileMetadata(
    filename="covtype.data.gz",
    url="https://ndownloader.figshare.com/files/5976039",
    checksum="614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771",
)

logger = logging.getLogger(__name__)

# Column names reference:
# https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info
FEATURE_NAMES = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
FEATURE_NAMES += [f"Wilderness_Area_{i}" for i in range(4)]
FEATURE_NAMES += [f"Soil_Type_{i}" for i in range(40)]
TARGET_NAMES = ["Cover_Type"]


def fetch_covtype(
    *,
    data_home=None,
    download_if_missing=True,
    random_state=None,
    shuffle=False,
    return_X_y=False,
    as_frame=False,
):
    """Load the covertype dataset (classification).

    Download it if necessary.

    =================   ============
    Classes                        7
    Samples total             581012
    Dimensionality                54
    Features                     int
    =================   ============

    Read more in the :ref:`User Guide <covtype_dataset>`.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : bool, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    shuffle : bool, default=False
        Whether to shuffle dataset.

    return_X_y : bool, default=False
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

        .. versionadded:: 0.20

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is a pandas DataFrame or
        Series depending on the number of target columns. If `return_X_y` is
        True, then (`data`, `target`) will be pandas DataFrames or Series as
        described below.

        .. versionadded:: 0.24

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray of shape (581012, 54)
            Each row corresponds to the 54 features in the dataset.
        target : ndarray of shape (581012,)
            Each value corresponds to one of
            the 7 forest covertypes with values
            ranging between 1 to 7.
        frame : dataframe of shape (581012, 55)
            Only present when `as_frame=True`. Contains `data` and `target`.
        DESCR : str
            Description of the forest covertype dataset.
        feature_names : list
            The names of the dataset columns.
        target_names: list
            The names of the target columns.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.20

    """

    data_home = get_data_home(data_home=data_home)
    covtype_dir = join(data_home, "covertype")
    samples_path = _pkl_filepath(covtype_dir, "samples")
    targets_path = _pkl_filepath(covtype_dir, "targets")
    available = exists(samples_path)

    if download_if_missing and not available:
        if not exists(covtype_dir):
            makedirs(covtype_dir)
        logger.info("Downloading %s" % ARCHIVE.url)

        archive_path = _fetch_remote(ARCHIVE, dirname=covtype_dir)
        Xy = np.genfromtxt(GzipFile(filename=archive_path), delimiter=",")
        # delete archive
        remove(archive_path)

        X = Xy[:, :-1]
        y = Xy[:, -1].astype(np.int32, copy=False)

        joblib.dump(X, samples_path, compress=9)
        joblib.dump(y, targets_path, compress=9)

    elif not available and not download_if_missing:
        raise IOError("Data not found and `download_if_missing` is False")
    try:
        X, y
    except NameError:
        X = joblib.load(samples_path)
        y = joblib.load(targets_path)

    if shuffle:
        ind = np.arange(X.shape[0])
        rng = check_random_state(random_state)
        rng.shuffle(ind)
        X = X[ind]
        y = y[ind]

    fdescr = load_descr("covtype.rst")

    frame = None
    if as_frame:
        frame, X, y = _convert_data_dataframe(
            caller_name="fetch_covtype",
            data=X,
            target=y,
            feature_names=FEATURE_NAMES,
            target_names=TARGET_NAMES,
        )
    if return_X_y:
        return X, y

    return Bunch(
        data=X,
        target=y,
        frame=frame,
        target_names=TARGET_NAMES,
        feature_names=FEATURE_NAMES,
        DESCR=fdescr,
    )
