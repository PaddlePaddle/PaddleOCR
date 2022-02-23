"""RCV1 dataset.

The dataset page is available at

    http://jmlr.csail.mit.edu/papers/volume5/lewis04a/
"""

# Author: Tom Dupre la Tour
# License: BSD 3 clause

import logging

from os import remove, makedirs
from os.path import exists, join
from gzip import GzipFile

import numpy as np
import scipy.sparse as sp
import joblib

from . import get_data_home
from ._base import _pkl_filepath
from ._base import _fetch_remote
from ._base import RemoteFileMetadata
from ._base import load_descr
from ._svmlight_format_io import load_svmlight_files
from ..utils import shuffle as shuffle_
from ..utils import Bunch


# The original vectorized data can be found at:
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt0.dat.gz
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt1.dat.gz
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt2.dat.gz
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt3.dat.gz
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_train.dat.gz
# while the original stemmed token files can be found
# in the README, section B.12.i.:
#    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm
XY_METADATA = (
    RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/5976069",
        checksum="ed40f7e418d10484091b059703eeb95ae3199fe042891dcec4be6696b9968374",
        filename="lyrl2004_vectors_test_pt0.dat.gz",
    ),
    RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/5976066",
        checksum="87700668ae45d45d5ca1ef6ae9bd81ab0f5ec88cc95dcef9ae7838f727a13aa6",
        filename="lyrl2004_vectors_test_pt1.dat.gz",
    ),
    RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/5976063",
        checksum="48143ac703cbe33299f7ae9f4995db49a258690f60e5debbff8995c34841c7f5",
        filename="lyrl2004_vectors_test_pt2.dat.gz",
    ),
    RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/5976060",
        checksum="dfcb0d658311481523c6e6ca0c3f5a3e1d3d12cde5d7a8ce629a9006ec7dbb39",
        filename="lyrl2004_vectors_test_pt3.dat.gz",
    ),
    RemoteFileMetadata(
        url="https://ndownloader.figshare.com/files/5976057",
        checksum="5468f656d0ba7a83afc7ad44841cf9a53048a5c083eedc005dcdb5cc768924ae",
        filename="lyrl2004_vectors_train.dat.gz",
    ),
)

# The original data can be found at:
# http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz
TOPICS_METADATA = RemoteFileMetadata(
    url="https://ndownloader.figshare.com/files/5976048",
    checksum="2a98e5e5d8b770bded93afc8930d88299474317fe14181aee1466cc754d0d1c1",
    filename="rcv1v2.topics.qrels.gz",
)

logger = logging.getLogger(__name__)


def fetch_rcv1(
    *,
    data_home=None,
    subset="all",
    download_if_missing=True,
    random_state=None,
    shuffle=False,
    return_X_y=False,
):
    """Load the RCV1 multilabel dataset (classification).

    Download it if necessary.

    Version: RCV1-v2, vectors, full sets, topics multilabels.

    =================   =====================
    Classes                               103
    Samples total                      804414
    Dimensionality                      47236
    Features            real, between 0 and 1
    =================   =====================

    Read more in the :ref:`User Guide <rcv1_dataset>`.

    .. versionadded:: 0.17

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    subset : {'train', 'test', 'all'}, default='all'
        Select the dataset to load: 'train' for the training set
        (23149 samples), 'test' for the test set (781265 samples),
        'all' for both, with the training samples first if shuffle is False.
        This follows the official LYRL2004 chronological split.

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
        If True, returns ``(dataset.data, dataset.target)`` instead of a Bunch
        object. See below for more information about the `dataset.data` and
        `dataset.target` object.

        .. versionadded:: 0.20

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : sparse matrix of shape (804414, 47236), dtype=np.float64
            The array has 0.16% of non zero values. Will be of CSR format.
        target : sparse matrix of shape (804414, 103), dtype=np.uint8
            Each sample has a value of 1 in its categories, and 0 in others.
            The array has 3.15% of non zero values. Will be of CSR format.
        sample_id : ndarray of shape (804414,), dtype=np.uint32,
            Identification number of each sample, as ordered in dataset.data.
        target_names : ndarray of shape (103,), dtype=object
            Names of each target (RCV1 topics), as ordered in dataset.target.
        DESCR : str
            Description of the RCV1 dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.20
    """
    N_SAMPLES = 804414
    N_FEATURES = 47236
    N_CATEGORIES = 103
    N_TRAIN = 23149

    data_home = get_data_home(data_home=data_home)
    rcv1_dir = join(data_home, "RCV1")
    if download_if_missing:
        if not exists(rcv1_dir):
            makedirs(rcv1_dir)

    samples_path = _pkl_filepath(rcv1_dir, "samples.pkl")
    sample_id_path = _pkl_filepath(rcv1_dir, "sample_id.pkl")
    sample_topics_path = _pkl_filepath(rcv1_dir, "sample_topics.pkl")
    topics_path = _pkl_filepath(rcv1_dir, "topics_names.pkl")

    # load data (X) and sample_id
    if download_if_missing and (not exists(samples_path) or not exists(sample_id_path)):
        files = []
        for each in XY_METADATA:
            logger.info("Downloading %s" % each.url)
            file_path = _fetch_remote(each, dirname=rcv1_dir)
            files.append(GzipFile(filename=file_path))

        Xy = load_svmlight_files(files, n_features=N_FEATURES)

        # Training data is before testing data
        X = sp.vstack([Xy[8], Xy[0], Xy[2], Xy[4], Xy[6]]).tocsr()
        sample_id = np.hstack((Xy[9], Xy[1], Xy[3], Xy[5], Xy[7]))
        sample_id = sample_id.astype(np.uint32, copy=False)

        joblib.dump(X, samples_path, compress=9)
        joblib.dump(sample_id, sample_id_path, compress=9)

        # delete archives
        for f in files:
            f.close()
            remove(f.name)
    else:
        X = joblib.load(samples_path)
        sample_id = joblib.load(sample_id_path)

    # load target (y), categories, and sample_id_bis
    if download_if_missing and (
        not exists(sample_topics_path) or not exists(topics_path)
    ):
        logger.info("Downloading %s" % TOPICS_METADATA.url)
        topics_archive_path = _fetch_remote(TOPICS_METADATA, dirname=rcv1_dir)

        # parse the target file
        n_cat = -1
        n_doc = -1
        doc_previous = -1
        y = np.zeros((N_SAMPLES, N_CATEGORIES), dtype=np.uint8)
        sample_id_bis = np.zeros(N_SAMPLES, dtype=np.int32)
        category_names = {}
        with GzipFile(filename=topics_archive_path, mode="rb") as f:
            for line in f:
                line_components = line.decode("ascii").split(" ")
                if len(line_components) == 3:
                    cat, doc, _ = line_components
                    if cat not in category_names:
                        n_cat += 1
                        category_names[cat] = n_cat

                    doc = int(doc)
                    if doc != doc_previous:
                        doc_previous = doc
                        n_doc += 1
                        sample_id_bis[n_doc] = doc
                    y[n_doc, category_names[cat]] = 1

        # delete archive
        remove(topics_archive_path)

        # Samples in X are ordered with sample_id,
        # whereas in y, they are ordered with sample_id_bis.
        permutation = _find_permutation(sample_id_bis, sample_id)
        y = y[permutation, :]

        # save category names in a list, with same order than y
        categories = np.empty(N_CATEGORIES, dtype=object)
        for k in category_names.keys():
            categories[category_names[k]] = k

        # reorder categories in lexicographic order
        order = np.argsort(categories)
        categories = categories[order]
        y = sp.csr_matrix(y[:, order])

        joblib.dump(y, sample_topics_path, compress=9)
        joblib.dump(categories, topics_path, compress=9)
    else:
        y = joblib.load(sample_topics_path)
        categories = joblib.load(topics_path)

    if subset == "all":
        pass
    elif subset == "train":
        X = X[:N_TRAIN, :]
        y = y[:N_TRAIN, :]
        sample_id = sample_id[:N_TRAIN]
    elif subset == "test":
        X = X[N_TRAIN:, :]
        y = y[N_TRAIN:, :]
        sample_id = sample_id[N_TRAIN:]
    else:
        raise ValueError(
            "Unknown subset parameter. Got '%s' instead of one"
            " of ('all', 'train', test')" % subset
        )

    if shuffle:
        X, y, sample_id = shuffle_(X, y, sample_id, random_state=random_state)

    fdescr = load_descr("rcv1.rst")

    if return_X_y:
        return X, y

    return Bunch(
        data=X, target=y, sample_id=sample_id, target_names=categories, DESCR=fdescr
    )


def _inverse_permutation(p):
    """Inverse permutation p."""
    n = p.size
    s = np.zeros(n, dtype=np.int32)
    i = np.arange(n, dtype=np.int32)
    np.put(s, p, i)  # s[p] = i
    return s


def _find_permutation(a, b):
    """Find the permutation from a to b."""
    t = np.argsort(a)
    u = np.argsort(b)
    u_ = _inverse_permutation(u)
    return t[u_]
