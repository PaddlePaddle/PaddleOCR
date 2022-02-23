"""Modified Olivetti faces dataset.

The original database was available from (now defunct)

    https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

The version retrieved here comes in MATLAB format from the personal
web page of Sam Roweis:

    https://cs.nyu.edu/~roweis/
"""

# Copyright (c) 2011 David Warde-Farley <wardefar at iro dot umontreal dot ca>
# License: BSD 3 clause

from os.path import exists
from os import makedirs, remove

import numpy as np
from scipy.io.matlab import loadmat
import joblib

from . import get_data_home
from ._base import _fetch_remote
from ._base import RemoteFileMetadata
from ._base import _pkl_filepath
from ._base import load_descr
from ..utils import check_random_state, Bunch

# The original data can be found at:
# https://cs.nyu.edu/~roweis/data/olivettifaces.mat
FACES = RemoteFileMetadata(
    filename="olivettifaces.mat",
    url="https://ndownloader.figshare.com/files/5976027",
    checksum="b612fb967f2dc77c9c62d3e1266e0c73d5fca46a4b8906c18e454d41af987794",
)


def fetch_olivetti_faces(
    *,
    data_home=None,
    shuffle=False,
    random_state=0,
    download_if_missing=True,
    return_X_y=False,
):
    """Load the Olivetti faces data-set from AT&T (classification).

    Download it if necessary.

    =================   =====================
    Classes                                40
    Samples total                         400
    Dimensionality                       4096
    Features            real, between 0 and 1
    =================   =====================

    Read more in the :ref:`User Guide <olivetti_faces_dataset>`.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    shuffle : bool, default=False
        If True the order of the dataset is shuffled to avoid having
        images of the same person grouped.

    random_state : int, RandomState instance or None, default=0
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    download_if_missing : bool, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : bool, default=False
        If True, returns `(data, target)` instead of a `Bunch` object. See
        below for more information about the `data` and `target` object.

        .. versionadded:: 0.22

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data: ndarray, shape (400, 4096)
            Each row corresponds to a ravelled
            face image of original size 64 x 64 pixels.
        images : ndarray, shape (400, 64, 64)
            Each row is a face image
            corresponding to one of the 40 subjects of the dataset.
        target : ndarray, shape (400,)
            Labels associated to each face image.
            Those labels are ranging from 0-39 and correspond to the
            Subject IDs.
        DESCR : str
            Description of the modified Olivetti Faces Dataset.

    (data, target) : tuple if `return_X_y=True`
        .. versionadded:: 0.22
    """
    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)
    filepath = _pkl_filepath(data_home, "olivetti.pkz")
    if not exists(filepath):
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        print("downloading Olivetti faces from %s to %s" % (FACES.url, data_home))
        mat_path = _fetch_remote(FACES, dirname=data_home)
        mfile = loadmat(file_name=mat_path)
        # delete raw .mat data
        remove(mat_path)

        faces = mfile["faces"].T.copy()
        joblib.dump(faces, filepath, compress=6)
        del mfile
    else:
        faces = joblib.load(filepath)

    # We want floating point data, but float32 is enough (there is only
    # one byte of precision in the original uint8s anyway)
    faces = np.float32(faces)
    faces = faces - faces.min()
    faces /= faces.max()
    faces = faces.reshape((400, 64, 64)).transpose(0, 2, 1)
    # 10 images per class, 400 images total, each class is contiguous.
    target = np.array([i // 10 for i in range(400)])
    if shuffle:
        random_state = check_random_state(random_state)
        order = random_state.permutation(len(faces))
        faces = faces[order]
        target = target[order]
    faces_vectorized = faces.reshape(len(faces), -1)

    fdescr = load_descr("olivetti_faces.rst")

    if return_X_y:
        return faces_vectorized, target

    return Bunch(data=faces_vectorized, images=faces, target=target, DESCR=fdescr)
