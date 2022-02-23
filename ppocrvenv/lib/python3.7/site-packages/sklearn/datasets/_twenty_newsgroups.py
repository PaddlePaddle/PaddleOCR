"""Caching loader for the 20 newsgroups text classification dataset.


The description of the dataset is available on the official website at:

    http://people.csail.mit.edu/jrennie/20Newsgroups/

Quoting the introduction:

    The 20 Newsgroups data set is a collection of approximately 20,000
    newsgroup documents, partitioned (nearly) evenly across 20 different
    newsgroups. To the best of my knowledge, it was originally collected
    by Ken Lang, probably for his Newsweeder: Learning to filter netnews
    paper, though he does not explicitly mention this collection. The 20
    newsgroups collection has become a popular data set for experiments
    in text applications of machine learning techniques, such as text
    classification and text clustering.

This dataset loader will download the recommended "by date" variant of the
dataset and which features a point in time split between the train and
test sets. The compressed dataset size is around 14 Mb compressed. Once
uncompressed the train set is 52 MB and the test set is 34 MB.
"""
# Copyright (c) 2011 Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

import os
import logging
import tarfile
import pickle
import shutil
import re
import codecs

import numpy as np
import scipy.sparse as sp
import joblib

from . import get_data_home
from . import load_files
from ._base import _convert_data_dataframe
from ._base import _pkl_filepath
from ._base import _fetch_remote
from ._base import RemoteFileMetadata
from ._base import load_descr
from ..feature_extraction.text import CountVectorizer
from .. import preprocessing
from ..utils import check_random_state, Bunch

logger = logging.getLogger(__name__)

# The original data can be found at:
# https://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz
ARCHIVE = RemoteFileMetadata(
    filename="20news-bydate.tar.gz",
    url="https://ndownloader.figshare.com/files/5975967",
    checksum="8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610",
)

CACHE_NAME = "20news-bydate.pkz"
TRAIN_FOLDER = "20news-bydate-train"
TEST_FOLDER = "20news-bydate-test"


def _download_20newsgroups(target_dir, cache_path):
    """Download the 20 newsgroups data and stored it as a zipped pickle."""
    train_path = os.path.join(target_dir, TRAIN_FOLDER)
    test_path = os.path.join(target_dir, TEST_FOLDER)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    logger.info("Downloading dataset from %s (14 MB)", ARCHIVE.url)
    archive_path = _fetch_remote(ARCHIVE, dirname=target_dir)

    logger.debug("Decompressing %s", archive_path)
    tarfile.open(archive_path, "r:gz").extractall(path=target_dir)
    os.remove(archive_path)

    # Store a zipped pickle
    cache = dict(
        train=load_files(train_path, encoding="latin1"),
        test=load_files(test_path, encoding="latin1"),
    )
    compressed_content = codecs.encode(pickle.dumps(cache), "zlib_codec")
    with open(cache_path, "wb") as f:
        f.write(compressed_content)

    shutil.rmtree(target_dir)
    return cache


def strip_newsgroup_header(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.

    Parameters
    ----------
    text : str
        The text from which to remove the signature block.
    """
    _before, _blankline, after = text.partition("\n\n")
    return after


_QUOTE_RE = re.compile(
    r"(writes in|writes:|wrote:|says:|said:" r"|^In article|^Quoted from|^\||^>)"
)


def strip_newsgroup_quoting(text):
    """
    Given text in "news" format, strip lines beginning with the quote
    characters > or |, plus lines that often introduce a quoted section
    (for example, because they contain the string 'writes:'.)

    Parameters
    ----------
    text : str
        The text from which to remove the signature block.
    """
    good_lines = [line for line in text.split("\n") if not _QUOTE_RE.search(line)]
    return "\n".join(good_lines)


def strip_newsgroup_footer(text):
    """
    Given text in "news" format, attempt to remove a signature block.

    As a rough heuristic, we assume that signatures are set apart by either
    a blank line or a line made of hyphens, and that it is the last such line
    in the file (disregarding blank lines at the end).

    Parameters
    ----------
    text : str
        The text from which to remove the signature block.
    """
    lines = text.strip().split("\n")
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip("-") == "":
            break

    if line_num > 0:
        return "\n".join(lines[:line_num])
    else:
        return text


def fetch_20newsgroups(
    *,
    data_home=None,
    subset="train",
    categories=None,
    shuffle=True,
    random_state=42,
    remove=(),
    download_if_missing=True,
    return_X_y=False,
):
    """Load the filenames and data from the 20 newsgroups dataset \
(classification).

    Download it if necessary.

    =================   ==========
    Classes                     20
    Samples total            18846
    Dimensionality               1
    Features                  text
    =================   ==========

    Read more in the :ref:`User Guide <20newsgroups_dataset>`.

    Parameters
    ----------
    data_home : str, default=None
        Specify a download and cache folder for the datasets. If None,
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    subset : {'train', 'test', 'all'}, default='train'
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.

    categories : array-like, dtype=str, default=None
        If None (default), load all the categories.
        If not None, list of category names to load (other categories
        ignored).

    shuffle : bool, default=True
        Whether or not to shuffle the data: might be important for models that
        make the assumption that the samples are independent and identically
        distributed (i.i.d.), such as stochastic gradient descent.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    remove : tuple, default=()
        May contain any subset of ('headers', 'footers', 'quotes'). Each of
        these are kinds of text that will be detected and removed from the
        newsgroup posts, preventing classifiers from overfitting on
        metadata.

        'headers' removes newsgroup headers, 'footers' removes blocks at the
        ends of posts that look like signatures, and 'quotes' removes lines
        that appear to be quoting another post.

        'headers' follows an exact standard; the other filters are not always
        correct.

    download_if_missing : bool, default=True
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : bool, default=False
        If True, returns `(data.data, data.target)` instead of a Bunch
        object.

        .. versionadded:: 0.22

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : list of shape (n_samples,)
            The data list to learn.
        target: ndarray of shape (n_samples,)
            The target labels.
        filenames: list of shape (n_samples,)
            The path to the location of the data.
        DESCR: str
            The full description of the dataset.
        target_names: list of shape (n_classes,)
            The names of target classes.

    (data, target) : tuple if `return_X_y=True`
        .. versionadded:: 0.22
    """

    data_home = get_data_home(data_home=data_home)
    cache_path = _pkl_filepath(data_home, CACHE_NAME)
    twenty_home = os.path.join(data_home, "20news_home")
    cache = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, "zlib_codec")
            cache = pickle.loads(uncompressed_content)
        except Exception as e:
            print(80 * "_")
            print("Cache loading failed")
            print(80 * "_")
            print(e)

    if cache is None:
        if download_if_missing:
            logger.info("Downloading 20news dataset. This may take a few minutes.")
            cache = _download_20newsgroups(
                target_dir=twenty_home, cache_path=cache_path
            )
        else:
            raise IOError("20Newsgroups dataset not found")

    if subset in ("train", "test"):
        data = cache[subset]
    elif subset == "all":
        data_lst = list()
        target = list()
        filenames = list()
        for subset in ("train", "test"):
            data = cache[subset]
            data_lst.extend(data.data)
            target.extend(data.target)
            filenames.extend(data.filenames)

        data.data = data_lst
        data.target = np.array(target)
        data.filenames = np.array(filenames)
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset
        )

    fdescr = load_descr("twenty_newsgroups.rst")

    data.DESCR = fdescr

    if "headers" in remove:
        data.data = [strip_newsgroup_header(text) for text in data.data]
    if "footers" in remove:
        data.data = [strip_newsgroup_footer(text) for text in data.data]
    if "quotes" in remove:
        data.data = [strip_newsgroup_quoting(text) for text in data.data]

    if categories is not None:
        labels = [(data.target_names.index(cat), cat) for cat in categories]
        # Sort the categories to have the ordering of the labels
        labels.sort()
        labels, categories = zip(*labels)
        mask = np.in1d(data.target, labels)
        data.filenames = data.filenames[mask]
        data.target = data.target[mask]
        # searchsorted to have continuous labels
        data.target = np.searchsorted(labels, data.target)
        data.target_names = list(categories)
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[mask]
        data.data = data_lst.tolist()

    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(data.target.shape[0])
        random_state.shuffle(indices)
        data.filenames = data.filenames[indices]
        data.target = data.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[indices]
        data.data = data_lst.tolist()

    if return_X_y:
        return data.data, data.target

    return data


def fetch_20newsgroups_vectorized(
    *,
    subset="train",
    remove=(),
    data_home=None,
    download_if_missing=True,
    return_X_y=False,
    normalize=True,
    as_frame=False,
):
    """Load and vectorize the 20 newsgroups dataset (classification).

    Download it if necessary.

    This is a convenience function; the transformation is done using the
    default settings for
    :class:`~sklearn.feature_extraction.text.CountVectorizer`. For more
    advanced usage (stopword filtering, n-gram extraction, etc.), combine
    fetch_20newsgroups with a custom
    :class:`~sklearn.feature_extraction.text.CountVectorizer`,
    :class:`~sklearn.feature_extraction.text.HashingVectorizer`,
    :class:`~sklearn.feature_extraction.text.TfidfTransformer` or
    :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.

    The resulting counts are normalized using
    :func:`sklearn.preprocessing.normalize` unless normalize is set to False.

    =================   ==========
    Classes                     20
    Samples total            18846
    Dimensionality          130107
    Features                  real
    =================   ==========

    Read more in the :ref:`User Guide <20newsgroups_dataset>`.

    Parameters
    ----------
    subset : {'train', 'test', 'all'}, default='train'
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.

    remove : tuple, default=()
        May contain any subset of ('headers', 'footers', 'quotes'). Each of
        these are kinds of text that will be detected and removed from the
        newsgroup posts, preventing classifiers from overfitting on
        metadata.

        'headers' removes newsgroup headers, 'footers' removes blocks at the
        ends of posts that look like signatures, and 'quotes' removes lines
        that appear to be quoting another post.

    data_home : str, default=None
        Specify an download and cache folder for the datasets. If None,
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : bool, default=True
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : bool, default=False
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

        .. versionadded:: 0.20

    normalize : bool, default=True
        If True, normalizes each document's feature vector to unit norm using
        :func:`sklearn.preprocessing.normalize`.

        .. versionadded:: 0.22

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string, or categorical). The target is
        a pandas DataFrame or Series depending on the number of
        `target_columns`.

        .. versionadded:: 0.24

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data: {sparse matrix, dataframe} of shape (n_samples, n_features)
            The input data matrix. If ``as_frame`` is `True`, ``data`` is
            a pandas DataFrame with sparse columns.
        target: {ndarray, series} of shape (n_samples,)
            The target labels. If ``as_frame`` is `True`, ``target`` is a
            pandas Series.
        target_names: list of shape (n_classes,)
            The names of target classes.
        DESCR: str
            The full description of the dataset.
        frame: dataframe of shape (n_samples, n_features + 1)
            Only present when `as_frame=True`. Pandas DataFrame with ``data``
            and ``target``.

            .. versionadded:: 0.24

    (data, target) : tuple if ``return_X_y`` is True
        `data` and `target` would be of the format defined in the `Bunch`
        description above.

        .. versionadded:: 0.20
    """
    data_home = get_data_home(data_home=data_home)
    filebase = "20newsgroup_vectorized"
    if remove:
        filebase += "remove-" + "-".join(remove)
    target_file = _pkl_filepath(data_home, filebase + ".pkl")

    # we shuffle but use a fixed seed for the memoization
    data_train = fetch_20newsgroups(
        data_home=data_home,
        subset="train",
        categories=None,
        shuffle=True,
        random_state=12,
        remove=remove,
        download_if_missing=download_if_missing,
    )

    data_test = fetch_20newsgroups(
        data_home=data_home,
        subset="test",
        categories=None,
        shuffle=True,
        random_state=12,
        remove=remove,
        download_if_missing=download_if_missing,
    )

    if os.path.exists(target_file):
        try:
            X_train, X_test, feature_names = joblib.load(target_file)
        except ValueError as e:
            raise ValueError(
                f"The cached dataset located in {target_file} was fetched "
                "with an older scikit-learn version and it is not compatible "
                "with the scikit-learn version imported. You need to "
                f"manually delete the file: {target_file}."
            ) from e
    else:
        vectorizer = CountVectorizer(dtype=np.int16)
        X_train = vectorizer.fit_transform(data_train.data).tocsr()
        X_test = vectorizer.transform(data_test.data).tocsr()
        feature_names = vectorizer.get_feature_names_out()

        joblib.dump((X_train, X_test, feature_names), target_file, compress=9)

    # the data is stored as int16 for compactness
    # but normalize needs floats
    if normalize:
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        preprocessing.normalize(X_train, copy=False)
        preprocessing.normalize(X_test, copy=False)

    target_names = data_train.target_names

    if subset == "train":
        data = X_train
        target = data_train.target
    elif subset == "test":
        data = X_test
        target = data_test.target
    elif subset == "all":
        data = sp.vstack((X_train, X_test)).tocsr()
        target = np.concatenate((data_train.target, data_test.target))
    else:
        raise ValueError(
            "%r is not a valid subset: should be one of ['train', 'test', 'all']"
            % subset
        )

    fdescr = load_descr("twenty_newsgroups.rst")

    frame = None
    target_name = ["category_class"]

    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "fetch_20newsgroups_vectorized",
            data,
            target,
            feature_names,
            target_names=target_name,
            sparse_data=True,
        )

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        feature_names=feature_names,
        DESCR=fdescr,
    )
