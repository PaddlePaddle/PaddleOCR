"""
Base IO code for all datasets
"""

# Copyright (c) 2007 David Cournapeau <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
#               2010 Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause
import csv
import hashlib
import gzip
import shutil
from collections import namedtuple
from os import environ, listdir, makedirs
from os.path import expanduser, isdir, join, splitext
from importlib import resources

from ..utils import Bunch
from ..utils import check_random_state
from ..utils import check_pandas_support
from ..utils.deprecation import deprecated

import numpy as np

from urllib.request import urlretrieve

DATA_MODULE = "sklearn.datasets.data"
DESCR_MODULE = "sklearn.datasets.descr"
IMAGES_MODULE = "sklearn.datasets.images"

RemoteFileMetadata = namedtuple("RemoteFileMetadata", ["filename", "url", "checksum"])


def get_data_home(data_home=None) -> str:
    """Return the path of the scikit-learn data dir.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data dir is set to a folder named 'scikit_learn_data' in the
    user home folder.

    Alternatively, it can be set by the 'SCIKIT_LEARN_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str, default=None
        The path to scikit-learn data directory. If `None`, the default path
        is `~/sklearn_learn_data`.
    """
    if data_home is None:
        data_home = environ.get("SCIKIT_LEARN_DATA", join("~", "scikit_learn_data"))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.

    Parameters
    ----------
    data_home : str, default=None
        The path to scikit-learn data directory. If `None`, the default path
        is `~/sklearn_learn_data`.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def _convert_data_dataframe(
    caller_name, data, target, feature_names, target_names, sparse_data=False
):
    pd = check_pandas_support("{} with as_frame=True".format(caller_name))
    if not sparse_data:
        data_df = pd.DataFrame(data, columns=feature_names)
    else:
        data_df = pd.DataFrame.sparse.from_spmatrix(data, columns=feature_names)

    target_df = pd.DataFrame(target, columns=target_names)
    combined_df = pd.concat([data_df, target_df], axis=1)
    X = combined_df[feature_names]
    y = combined_df[target_names]
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    return combined_df, X, y


def load_files(
    container_path,
    *,
    description=None,
    categories=None,
    load_content=True,
    shuffle=True,
    encoding=None,
    decode_error="strict",
    random_state=0,
):
    """Load text files with categories as subfolder names.

    Individual samples are assumed to be files stored a two levels folder
    structure such as the following:

        container_folder/
            category_1_folder/
                file_1.txt
                file_2.txt
                ...
                file_42.txt
            category_2_folder/
                file_43.txt
                file_44.txt
                ...

    The folder names are used as supervised signal label names. The individual
    file names are not important.

    This function does not try to extract features into a numpy array or scipy
    sparse matrix. In addition, if load_content is false it does not try to
    load the files in memory.

    To use text files in a scikit-learn classification or clustering algorithm,
    you will need to use the :mod`~sklearn.feature_extraction.text` module to
    build a feature extraction transformer that suits your problem.

    If you set load_content=True, you should also specify the encoding of the
    text using the 'encoding' parameter. For many modern text files, 'utf-8'
    will be the correct encoding. If you leave encoding equal to None, then the
    content will be made of bytes instead of Unicode, and you will not be able
    to use most functions in :mod:`~sklearn.feature_extraction.text`.

    Similar feature extractors should be built for other kind of unstructured
    data input such as images, audio, video, ...

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    container_path : str
        Path to the main folder holding one subfolder per category.

    description : str, default=None
        A paragraph describing the characteristic of the dataset: its source,
        reference, etc.

    categories : list of str, default=None
        If None (default), load all the categories. If not None, list of
        category names to load (other categories ignored).

    load_content : bool, default=True
        Whether to load or not the content of the different files. If true a
        'data' attribute containing the text information is present in the data
        structure returned. If not, a filenames attribute gives the path to the
        files.

    shuffle : bool, default=True
        Whether or not to shuffle the data: might be important for models that
        make the assumption that the samples are independent and identically
        distributed (i.i.d.), such as stochastic gradient descent.

    encoding : str, default=None
        If None, do not try to decode the content of the files (e.g. for images
        or other non-text content). If not None, encoding to use to decode text
        files to Unicode if load_content is True.

    decode_error : {'strict', 'ignore', 'replace'}, default='strict'
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. Passed as keyword
        argument 'errors' to bytes.decode.

    random_state : int, RandomState instance or None, default=0
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : list of str
            Only present when `load_content=True`.
            The raw text data to learn.
        target : ndarray
            The target labels (integer index).
        target_names : list
            The names of target classes.
        DESCR : str
            The full description of the dataset.
        filenames: ndarray
            The filenames holding the dataset.
    """
    target = []
    target_names = []
    filenames = []

    folders = [
        f for f in sorted(listdir(container_path)) if isdir(join(container_path, f))
    ]

    if categories is not None:
        folders = [f for f in folders if f in categories]

    for label, folder in enumerate(folders):
        target_names.append(folder)
        folder_path = join(container_path, folder)
        documents = [join(folder_path, d) for d in sorted(listdir(folder_path))]
        target.extend(len(documents) * [label])
        filenames.extend(documents)

    # convert to array for fancy indexing
    filenames = np.array(filenames)
    target = np.array(target)

    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(filenames.shape[0])
        random_state.shuffle(indices)
        filenames = filenames[indices]
        target = target[indices]

    if load_content:
        data = []
        for filename in filenames:
            with open(filename, "rb") as f:
                data.append(f.read())
        if encoding is not None:
            data = [d.decode(encoding, decode_error) for d in data]
        return Bunch(
            data=data,
            filenames=filenames,
            target_names=target_names,
            target=target,
            DESCR=description,
        )

    return Bunch(
        filenames=filenames, target_names=target_names, target=target, DESCR=description
    )


def load_csv_data(
    data_file_name,
    *,
    data_module=DATA_MODULE,
    descr_file_name=None,
    descr_module=DESCR_MODULE,
):
    """Loads `data_file_name` from `data_module with `importlib.resources`.

    Parameters
    ----------
    data_file_name : str
        Name of csv file to be loaded from `data_module/data_file_name`.
        For example `'wine_data.csv'`.

    data_module : str or module, default='sklearn.datasets.data'
        Module where data lives. The default is `'sklearn.datasets.data'`.

    descr_file_name : str, default=None
        Name of rst file to be loaded from `descr_module/descr_file_name`.
        For example `'wine_data.rst'`. See also :func:`load_descr`.
        If not None, also returns the corresponding description of
        the dataset.

    descr_module : str or module, default='sklearn.datasets.descr'
        Module where `descr_file_name` lives. See also :func:`load_descr`.
        The default is `'sklearn.datasets.descr'`.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features)
        A 2D array with each row representing one sample and each column
        representing the features of a given sample.

    target : ndarry of shape (n_samples,)
        A 1D array holding target variables for all the samples in `data`.
        For example target[0] is the target variable for data[0].

    target_names : ndarry of shape (n_samples,)
        A 1D array containing the names of the classifications. For example
        target_names[0] is the name of the target[0] class.

    descr : str, optional
        Description of the dataset (the content of `descr_file_name`).
        Only returned if `descr_file_name` is not None.
    """
    with resources.open_text(data_module, data_file_name) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=int)

    if descr_file_name is None:
        return data, target, target_names
    else:
        assert descr_module is not None
        descr = load_descr(descr_module=descr_module, descr_file_name=descr_file_name)
        return data, target, target_names, descr


def load_gzip_compressed_csv_data(
    data_file_name,
    *,
    data_module=DATA_MODULE,
    descr_file_name=None,
    descr_module=DESCR_MODULE,
    encoding="utf-8",
    **kwargs,
):
    """Loads gzip-compressed `data_file_name` from `data_module` with `importlib.resources`.

    1) Open resource file with `importlib.resources.open_binary`
    2) Decompress file obj with `gzip.open`
    3) Load decompressed data with `np.loadtxt`

    Parameters
    ----------
    data_file_name : str
        Name of gzip-compressed csv file  (`'*.csv.gz'`) to be loaded from
        `data_module/data_file_name`. For example `'diabetes_data.csv.gz'`.

    data_module : str or module, default='sklearn.datasets.data'
        Module where data lives. The default is `'sklearn.datasets.data'`.

    descr_file_name : str, default=None
        Name of rst file to be loaded from `descr_module/descr_file_name`.
        For example `'wine_data.rst'`. See also :func:`load_descr`.
        If not None, also returns the corresponding description of
        the dataset.

    descr_module : str or module, default='sklearn.datasets.descr'
        Module where `descr_file_name` lives. See also :func:`load_descr`.
        The default  is `'sklearn.datasets.descr'`.

    encoding : str, default="utf-8"
        Name of the encoding that the gzip-decompressed file will be
        decoded with. The default is 'utf-8'.

    **kwargs : dict, optional
        Keyword arguments to be passed to `np.loadtxt`;
        e.g. delimiter=','.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features)
        A 2D array with each row representing one sample and each column
        representing the features and/or target of a given sample.

    descr : str, optional
        Description of the dataset (the content of `descr_file_name`).
        Only returned if `descr_file_name` is not None.
    """
    with resources.open_binary(data_module, data_file_name) as compressed_file:
        compressed_file = gzip.open(compressed_file, mode="rt", encoding=encoding)
        data = np.loadtxt(compressed_file, **kwargs)

    if descr_file_name is None:
        return data
    else:
        assert descr_module is not None
        descr = load_descr(descr_module=descr_module, descr_file_name=descr_file_name)
        return data, descr


def load_descr(descr_file_name, *, descr_module=DESCR_MODULE):
    """Load `descr_file_name` from `descr_module` with `importlib.resources`.

    Parameters
    ----------
    descr_file_name : str, default=None
        Name of rst file to be loaded from `descr_module/descr_file_name`.
        For example `'wine_data.rst'`. See also :func:`load_descr`.
        If not None, also returns the corresponding description of
        the dataset.

    descr_module : str or module, default='sklearn.datasets.descr'
        Module where `descr_file_name` lives. See also :func:`load_descr`.
        The default  is `'sklearn.datasets.descr'`.

    Returns
    -------
    fdescr : str
        Content of `descr_file_name`.
    """
    fdescr = resources.read_text(descr_module, descr_file_name)

    return fdescr


def load_wine(*, return_X_y=False, as_frame=False):
    """Load and return the wine dataset (classification).

    .. versionadded:: 0.18

    The wine dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class        [59,71,48]
    Samples total                  178
    Dimensionality                  13
    Features            real, positive
    =================   ==============

    Read more in the :ref:`User Guide <wine_dataset>`.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (178, 13)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, Series} of shape (178,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        frame: DataFrame of shape (178, 14)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        DESCR: str
            The full description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    The copy of UCI ML Wine Data Set dataset is downloaded and modified to fit
    standard format from:
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

    Examples
    --------
    Let's say you are interested in the samples 10, 80, and 140, and want to
    know their class name.

    >>> from sklearn.datasets import load_wine
    >>> data = load_wine()
    >>> data.target[[10, 80, 140]]
    array([0, 1, 2])
    >>> list(data.target_names)
    ['class_0', 'class_1', 'class_2']
    """

    data, target, target_names, fdescr = load_csv_data(
        data_file_name="wine_data.csv", descr_file_name="wine_data.rst"
    )

    feature_names = [
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols",
        "flavanoids",
        "nonflavanoid_phenols",
        "proanthocyanins",
        "color_intensity",
        "hue",
        "od280/od315_of_diluted_wines",
        "proline",
    ]

    frame = None
    target_columns = [
        "target",
    ]
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_wine", data, target, feature_names, target_columns
        )

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        DESCR=fdescr,
        feature_names=feature_names,
    )


def load_iris(*, return_X_y=False, as_frame=False):
    """Load and return the iris dataset (classification).

    The iris dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============

    Read more in the :ref:`User Guide <iris_dataset>`.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (150, 4)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, Series} of shape (150,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        frame: DataFrame of shape (150, 5)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.

            .. versionadded:: 0.20

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarray. The first containing a 2D array of shape
        (n_samples, n_features) with each row representing one sample and
        each column representing the features. The second ndarray of shape
        (n_samples,) containing the target samples.

        .. versionadded:: 0.18

    Notes
    -----
        .. versionchanged:: 0.20
            Fixed two wrong data points according to Fisher's paper.
            The new version is the same as in R, but not as in the UCI
            Machine Learning Repository.

    Examples
    --------
    Let's say you are interested in the samples 10, 25, and 50, and want to
    know their class name.

    >>> from sklearn.datasets import load_iris
    >>> data = load_iris()
    >>> data.target[[10, 25, 50]]
    array([0, 0, 1])
    >>> list(data.target_names)
    ['setosa', 'versicolor', 'virginica']
    """
    data_file_name = "iris.csv"
    data, target, target_names, fdescr = load_csv_data(
        data_file_name=data_file_name, descr_file_name="iris.rst"
    )

    feature_names = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    frame = None
    target_columns = [
        "target",
    ]
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_iris", data, target, feature_names, target_columns
        )

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file_name,
        data_module=DATA_MODULE,
    )


def load_breast_cancer(*, return_X_y=False, as_frame=False):
    """Load and return the breast cancer wisconsin dataset (classification).

    The breast cancer dataset is a classic and very easy binary classification
    dataset.

    =================   ==============
    Classes                          2
    Samples per class    212(M),357(B)
    Samples total                  569
    Dimensionality                  30
    Features            real, positive
    =================   ==============

    Read more in the :ref:`User Guide <breast_cancer_dataset>`.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (569, 30)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, Series} of shape (569,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        frame: DataFrame of shape (569, 31)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.

            .. versionadded:: 0.20

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18

    The copy of UCI ML Breast Cancer Wisconsin (Diagnostic) dataset is
    downloaded from:
    https://goo.gl/U2Uwz2

    Examples
    --------
    Let's say you are interested in the samples 10, 50, and 85, and want to
    know their class name.

    >>> from sklearn.datasets import load_breast_cancer
    >>> data = load_breast_cancer()
    >>> data.target[[10, 50, 85]]
    array([0, 1, 0])
    >>> list(data.target_names)
    ['malignant', 'benign']
    """
    data_file_name = "breast_cancer.csv"
    data, target, target_names, fdescr = load_csv_data(
        data_file_name=data_file_name, descr_file_name="breast_cancer.rst"
    )

    feature_names = np.array(
        [
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
            "mean smoothness",
            "mean compactness",
            "mean concavity",
            "mean concave points",
            "mean symmetry",
            "mean fractal dimension",
            "radius error",
            "texture error",
            "perimeter error",
            "area error",
            "smoothness error",
            "compactness error",
            "concavity error",
            "concave points error",
            "symmetry error",
            "fractal dimension error",
            "worst radius",
            "worst texture",
            "worst perimeter",
            "worst area",
            "worst smoothness",
            "worst compactness",
            "worst concavity",
            "worst concave points",
            "worst symmetry",
            "worst fractal dimension",
        ]
    )

    frame = None
    target_columns = [
        "target",
    ]
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_breast_cancer", data, target, feature_names, target_columns
        )

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file_name,
        data_module=DATA_MODULE,
    )


def load_digits(*, n_class=10, return_X_y=False, as_frame=False):
    """Load and return the digits dataset (classification).

    Each datapoint is a 8x8 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class             ~180
    Samples total                 1797
    Dimensionality                  64
    Features             integers 0-16
    =================   ==============

    Read more in the :ref:`User Guide <digits_dataset>`.

    Parameters
    ----------
    n_class : int, default=10
        The number of classes to return. Between 0 and 10.

    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (1797, 64)
            The flattened data matrix. If `as_frame=True`, `data` will be
            a pandas DataFrame.
        target: {ndarray, Series} of shape (1797,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.

            .. versionadded:: 0.20

        frame: DataFrame of shape (1797, 65)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        images: {ndarray} of shape (1797, 8, 8)
            The raw image data.
        DESCR: str
            The full description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18

    This is a copy of the test set of the UCI ML hand-written digits datasets
    https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

    Examples
    --------
    To load the data and visualize the images::

        >>> from sklearn.datasets import load_digits
        >>> digits = load_digits()
        >>> print(digits.data.shape)
        (1797, 64)
        >>> import matplotlib.pyplot as plt
        >>> plt.gray()
        >>> plt.matshow(digits.images[0])
        <...>
        >>> plt.show()
    """

    data, fdescr = load_gzip_compressed_csv_data(
        data_file_name="digits.csv.gz", descr_file_name="digits.rst", delimiter=","
    )

    target = data[:, -1].astype(int, copy=False)
    flat_data = data[:, :-1]
    images = flat_data.view()
    images.shape = (-1, 8, 8)

    if n_class < 10:
        idx = target < n_class
        flat_data, target = flat_data[idx], target[idx]
        images = images[idx]

    feature_names = [
        "pixel_{}_{}".format(row_idx, col_idx)
        for row_idx in range(8)
        for col_idx in range(8)
    ]

    frame = None
    target_columns = [
        "target",
    ]
    if as_frame:
        frame, flat_data, target = _convert_data_dataframe(
            "load_digits", flat_data, target, feature_names, target_columns
        )

    if return_X_y:
        return flat_data, target

    return Bunch(
        data=flat_data,
        target=target,
        frame=frame,
        feature_names=feature_names,
        target_names=np.arange(10),
        images=images,
        DESCR=fdescr,
    )


def load_diabetes(*, return_X_y=False, as_frame=False):
    """Load and return the diabetes dataset (regression).

    ==============   ==================
    Samples total    442
    Dimensionality   10
    Features         real, -.2 < x < .2
    Targets          integer 25 - 346
    ==============   ==================

    .. note::
       The meaning of each feature (i.e. `feature_names`) might be unclear
       (especially for `ltg`) as the documentation of the original dataset is
       not explicit. We provide information that seems correct in regard with
       the scientific literature in this field of research.

    Read more in the :ref:`User Guide <diabetes_dataset>`.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (442, 10)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, Series} of shape (442,)
            The regression target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        frame: DataFrame of shape (442, 11)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        DESCR: str
            The full description of the dataset.
        data_filename: str
            The path to the location of the data.
        target_filename: str
            The path to the location of the target.

    (data, target) : tuple if ``return_X_y`` is True
        Returns a tuple of two ndarray of shape (n_samples, n_features)
        A 2D array with each row representing one sample and each column
        representing the features and/or target of a given sample.
        .. versionadded:: 0.18
    """
    data_filename = "diabetes_data.csv.gz"
    target_filename = "diabetes_target.csv.gz"
    data = load_gzip_compressed_csv_data(data_filename)
    target = load_gzip_compressed_csv_data(target_filename)

    fdescr = load_descr("diabetes.rst")

    feature_names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

    frame = None
    target_columns = [
        "target",
    ]
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_diabetes", data, target, feature_names, target_columns
        )

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        frame=frame,
        DESCR=fdescr,
        feature_names=feature_names,
        data_filename=data_filename,
        target_filename=target_filename,
        data_module=DATA_MODULE,
    )


def load_linnerud(*, return_X_y=False, as_frame=False):
    """Load and return the physical exercise Linnerud dataset.

    This dataset is suitable for multi-ouput regression tasks.

    ==============   ============================
    Samples total    20
    Dimensionality   3 (for both data and target)
    Features         integer
    Targets          integer
    ==============   ============================

    Read more in the :ref:`User Guide <linnerrud_dataset>`.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (20, 3)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, dataframe} of shape (20, 3)
            The regression targets. If `as_frame=True`, `target` will be
            a pandas DataFrame.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of the target columns.
        frame: DataFrame of shape (20, 6)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        DESCR: str
            The full description of the dataset.
        data_filename: str
            The path to the location of the data.
        target_filename: str
            The path to the location of the target.

            .. versionadded:: 0.20

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18
    """
    data_filename = "linnerud_exercise.csv"
    target_filename = "linnerud_physiological.csv"

    # Read header and data
    with resources.open_text(DATA_MODULE, data_filename) as f:
        header_exercise = f.readline().split()
        f.seek(0)  # reset file obj
        data_exercise = np.loadtxt(f, skiprows=1)

    with resources.open_text(DATA_MODULE, target_filename) as f:
        header_physiological = f.readline().split()
        f.seek(0)  # reset file obj
        data_physiological = np.loadtxt(f, skiprows=1)

    fdescr = load_descr("linnerud.rst")

    frame = None
    if as_frame:
        (frame, data_exercise, data_physiological) = _convert_data_dataframe(
            "load_linnerud",
            data_exercise,
            data_physiological,
            header_exercise,
            header_physiological,
        )
    if return_X_y:
        return data_exercise, data_physiological

    return Bunch(
        data=data_exercise,
        feature_names=header_exercise,
        target=data_physiological,
        target_names=header_physiological,
        frame=frame,
        DESCR=fdescr,
        data_filename=data_filename,
        target_filename=target_filename,
        data_module=DATA_MODULE,
    )


@deprecated(
    r"""`load_boston` is deprecated in 1.0 and will be removed in 1.2.

    The Boston housing prices dataset has an ethical problem. You can refer to
    the documentation of this function for further details.

    The scikit-learn maintainers therefore strongly discourage the use of this
    dataset unless the purpose of the code is to study and educate about
    ethical issues in data science and machine learning.

    In this special case, you can fetch the dataset from the original
    source::

        import pandas as pd
        import numpy as np


        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]

    Alternative datasets include the California housing dataset (i.e.
    :func:`~sklearn.datasets.fetch_california_housing`) and the Ames housing
    dataset. You can load the datasets as follows::

        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()

    for the California housing dataset and::

        from sklearn.datasets import fetch_openml
        housing = fetch_openml(name="house_prices", as_frame=True)

    for the Ames housing dataset.
    """
)
def load_boston(*, return_X_y=False):
    r"""Load and return the boston house-prices dataset (regression).

    ==============   ==============
    Samples total               506
    Dimensionality               13
    Features         real, positive
    Targets           real 5. - 50.
    ==============   ==============

    Read more in the :ref:`User Guide <boston_dataset>`.

    .. deprecated:: 1.0
       This function is deprecated in 1.0 and will be removed in 1.2. See the
       warning message below for further details regarding the alternative
       datasets.

    .. warning::
        The Boston housing prices dataset has an ethical problem: as
        investigated in [1]_, the authors of this dataset engineered a
        non-invertible variable "B" assuming that racial self-segregation had a
        positive impact on house prices [2]_. Furthermore the goal of the
        research that led to the creation of this dataset was to study the
        impact of air quality but it did not give adequate demonstration of the
        validity of this assumption.

        The scikit-learn maintainers therefore strongly discourage the use of
        this dataset unless the purpose of the code is to study and educate
        about ethical issues in data science and machine learning.

        In this special case, you can fetch the dataset from the original
        source::

            import pandas as pd  # doctest: +SKIP
            import numpy as np


            data_url = "http://lib.stat.cmu.edu/datasets/boston"
            raw_df = pd.read_csv(data_url, sep="s+", skiprows=22, header=None)
            data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
            target = raw_df.values[1::2, 2]

        Alternative datasets include the California housing dataset [3]_
        (i.e. :func:`~sklearn.datasets.fetch_california_housing`) and Ames
        housing dataset [4]_. You can load the datasets as follows::

            from sklearn.datasets import fetch_california_housing
            housing = fetch_california_housing()

        for the California housing dataset and::

            from sklearn.datasets import fetch_openml
            housing = fetch_openml(name="house_prices", as_frame=True)  # noqa

        for the Ames housing dataset.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray of shape (506, 13)
            The data matrix.
        target : ndarray of shape (506,)
            The regression target.
        filename : str
            The physical location of boston csv dataset.

            .. versionadded:: 0.20

        DESCR : str
            The full description of the dataset.
        feature_names : ndarray
            The names of features

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18

    Notes
    -----
        .. versionchanged:: 0.20
            Fixed a wrong data point at [445, 0].

    References
    ----------
    .. [1] `Racist data destruction? M Carlisle,
            <https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8>`_
    .. [2] `Harrison Jr, David, and Daniel L. Rubinfeld.
           "Hedonic housing prices and the demand for clean air."
           Journal of environmental economics and management 5.1 (1978): 81-102.
           <https://www.researchgate.net/publication/4974606_Hedonic_housing_prices_and_the_demand_for_clean_air>`_
    .. [3] `California housing dataset
            <https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset>`_
    .. [4] `Ames housing dataset
            <https://www.openml.org/d/42165>`_

    Examples
    --------
    >>> import warnings
    >>> from sklearn.datasets import load_boston
    >>> with warnings.catch_warnings():
    ...     # You should probably not use this dataset.
    ...     warnings.filterwarnings("ignore")
    ...     X, y = load_boston(return_X_y=True)
    >>> print(X.shape)
    (506, 13)
    """
    # TODO: once the deprecation period is over, implement a module level
    # `__getattr__` function in`sklearn.datasets` to raise an exception with
    # an informative error message at import time instead of just removing
    # load_boston. The goal is to avoid having beginners that copy-paste code
    # from numerous books and tutorials that use this dataset loader get
    # a confusing ImportError when trying to learn scikit-learn.
    # See: https://www.python.org/dev/peps/pep-0562/

    descr_text = load_descr("boston_house_prices.rst")

    data_file_name = "boston_house_prices.csv"
    with resources.open_text(DATA_MODULE, data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,))
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.float64)

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        # last column is target value
        feature_names=feature_names[:-1],
        DESCR=descr_text,
        filename=data_file_name,
        data_module=DATA_MODULE,
    )


def load_sample_images():
    """Load sample images for image manipulation.

    Loads both, ``china`` and ``flower``.

    Read more in the :ref:`User Guide <sample_images>`.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        images : list of ndarray of shape (427, 640, 3)
            The two sample image.
        filenames : list
            The filenames for the images.
        DESCR : str
            The full description of the dataset.

    Examples
    --------
    To load the data and visualize the images:

    >>> from sklearn.datasets import load_sample_images
    >>> dataset = load_sample_images()     #doctest: +SKIP
    >>> len(dataset.images)                #doctest: +SKIP
    2
    >>> first_img_data = dataset.images[0] #doctest: +SKIP
    >>> first_img_data.shape               #doctest: +SKIP
    (427, 640, 3)
    >>> first_img_data.dtype               #doctest: +SKIP
    dtype('uint8')
    """
    # import PIL only when needed
    from ..externals._pilutil import imread

    descr = load_descr("README.txt", descr_module=IMAGES_MODULE)

    filenames, images = [], []
    for filename in sorted(resources.contents(IMAGES_MODULE)):
        if filename.endswith(".jpg"):
            filenames.append(filename)
            with resources.open_binary(IMAGES_MODULE, filename) as image_file:
                image = imread(image_file)
            images.append(image)

    return Bunch(images=images, filenames=filenames, DESCR=descr)


def load_sample_image(image_name):
    """Load the numpy array of a single sample image

    Read more in the :ref:`User Guide <sample_images>`.

    Parameters
    ----------
    image_name : {`china.jpg`, `flower.jpg`}
        The name of the sample image loaded

    Returns
    -------
    img : 3D array
        The image as a numpy array: height x width x color

    Examples
    --------

    >>> from sklearn.datasets import load_sample_image
    >>> china = load_sample_image('china.jpg')   # doctest: +SKIP
    >>> china.dtype                              # doctest: +SKIP
    dtype('uint8')
    >>> china.shape                              # doctest: +SKIP
    (427, 640, 3)
    >>> flower = load_sample_image('flower.jpg') # doctest: +SKIP
    >>> flower.dtype                             # doctest: +SKIP
    dtype('uint8')
    >>> flower.shape                             # doctest: +SKIP
    (427, 640, 3)
    """
    images = load_sample_images()
    index = None
    for i, filename in enumerate(images.filenames):
        if filename.endswith(image_name):
            index = i
            break
    if index is None:
        raise AttributeError("Cannot find sample image: %s" % image_name)
    return images.images[index]


def _pkl_filepath(*args, **kwargs):
    """Return filename for Python 3 pickles

    args[-1] is expected to be the ".pkl" filename. For compatibility with
    older scikit-learn versions, a suffix is inserted before the extension.

    _pkl_filepath('/path/to/folder', 'filename.pkl') returns
    '/path/to/folder/filename_py3.pkl'

    """
    py3_suffix = kwargs.get("py3_suffix", "_py3")
    basename, ext = splitext(args[-1])
    basename += py3_suffix
    new_args = args[:-1] + (basename + ext,)
    return join(*new_args)


def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()


def _fetch_remote(remote, dirname=None):
    """Helper function to download a remote dataset into path

    Fetch a dataset pointed by remote's url, save into path using remote's
    filename and ensure its integrity based on the SHA256 Checksum of the
    downloaded file.

    Parameters
    ----------
    remote : RemoteFileMetadata
        Named tuple containing remote dataset meta information: url, filename
        and checksum

    dirname : str
        Directory to save the file to.

    Returns
    -------
    file_path: str
        Full path of the created file.
    """

    file_path = remote.filename if dirname is None else join(dirname, remote.filename)
    urlretrieve(remote.url, file_path)
    checksum = _sha256(file_path)
    if remote.checksum != checksum:
        raise IOError(
            "{} has an SHA256 checksum ({}) "
            "differing from expected ({}), "
            "file may be corrupted.".format(file_path, checksum, remote.checksum)
        )
    return file_path
