"""Labeled Faces in the Wild (LFW) dataset

This dataset is a collection of JPEG pictures of famous people collected
over the internet, all details are available on the official website:

    http://vis-www.cs.umass.edu/lfw/
"""
# Copyright (c) 2011 Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

from os import listdir, makedirs, remove
from os.path import join, exists, isdir

import logging

import numpy as np
import joblib
from joblib import Memory

from ._base import (
    get_data_home,
    _fetch_remote,
    RemoteFileMetadata,
    load_descr,
)
from ..utils import Bunch
from ..utils.fixes import parse_version

logger = logging.getLogger(__name__)

# The original data can be found in:
# http://vis-www.cs.umass.edu/lfw/lfw.tgz
ARCHIVE = RemoteFileMetadata(
    filename="lfw.tgz",
    url="https://ndownloader.figshare.com/files/5976018",
    checksum="055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0",
)

# The original funneled data can be found in:
# http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
FUNNELED_ARCHIVE = RemoteFileMetadata(
    filename="lfw-funneled.tgz",
    url="https://ndownloader.figshare.com/files/5976015",
    checksum="b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a",
)

# The original target data can be found in:
# http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt',
# http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt',
# http://vis-www.cs.umass.edu/lfw/pairs.txt',
TARGETS = (
    RemoteFileMetadata(
        filename="pairsDevTrain.txt",
        url="https://ndownloader.figshare.com/files/5976012",
        checksum="1d454dada7dfeca0e7eab6f65dc4e97a6312d44cf142207be28d688be92aabfa",
    ),
    RemoteFileMetadata(
        filename="pairsDevTest.txt",
        url="https://ndownloader.figshare.com/files/5976009",
        checksum="7cb06600ea8b2814ac26e946201cdb304296262aad67d046a16a7ec85d0ff87c",
    ),
    RemoteFileMetadata(
        filename="pairs.txt",
        url="https://ndownloader.figshare.com/files/5976006",
        checksum="ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592",
    ),
)


#
# Common private utilities for data fetching from the original LFW website
# local disk caching, and image decoding.
#


def _check_fetch_lfw(data_home=None, funneled=True, download_if_missing=True):
    """Helper function to download any missing LFW data"""

    data_home = get_data_home(data_home=data_home)
    lfw_home = join(data_home, "lfw_home")

    if not exists(lfw_home):
        makedirs(lfw_home)

    for target in TARGETS:
        target_filepath = join(lfw_home, target.filename)
        if not exists(target_filepath):
            if download_if_missing:
                logger.info("Downloading LFW metadata: %s", target.url)
                _fetch_remote(target, dirname=lfw_home)
            else:
                raise IOError("%s is missing" % target_filepath)

    if funneled:
        data_folder_path = join(lfw_home, "lfw_funneled")
        archive = FUNNELED_ARCHIVE
    else:
        data_folder_path = join(lfw_home, "lfw")
        archive = ARCHIVE

    if not exists(data_folder_path):
        archive_path = join(lfw_home, archive.filename)
        if not exists(archive_path):
            if download_if_missing:
                logger.info("Downloading LFW data (~200MB): %s", archive.url)
                _fetch_remote(archive, dirname=lfw_home)
            else:
                raise IOError("%s is missing" % archive_path)

        import tarfile

        logger.debug("Decompressing the data archive to %s", data_folder_path)
        tarfile.open(archive_path, "r:gz").extractall(path=lfw_home)
        remove(archive_path)

    return lfw_home, data_folder_path


def _load_imgs(file_paths, slice_, color, resize):
    """Internally used to load images"""
    # import PIL only when needed
    from ..externals._pilutil import imread, imresize

    # compute the portion of the images to load to respect the slice_ parameter
    # given by the caller
    default_slice = (slice(0, 250), slice(0, 250))
    if slice_ is None:
        slice_ = default_slice
    else:
        slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))

    h_slice, w_slice = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)

    if resize is not None:
        resize = float(resize)
        h = int(resize * h)
        w = int(resize * w)

    # allocate some contiguous memory to host the decoded image slices
    n_faces = len(file_paths)
    if not color:
        faces = np.zeros((n_faces, h, w), dtype=np.float32)
    else:
        faces = np.zeros((n_faces, h, w, 3), dtype=np.float32)

    # iterate over the collected file path to load the jpeg files as numpy
    # arrays
    for i, file_path in enumerate(file_paths):
        if i % 1000 == 0:
            logger.debug("Loading face #%05d / %05d", i + 1, n_faces)

        # Checks if jpeg reading worked. Refer to issue #3594 for more
        # details.
        img = imread(file_path)
        if img.ndim == 0:
            raise RuntimeError(
                "Failed to read the image file %s, "
                "Please make sure that libjpeg is installed" % file_path
            )

        face = np.asarray(img[slice_], dtype=np.float32)
        face /= 255.0  # scale uint8 coded colors to the [0.0, 1.0] floats
        if resize is not None:
            face = imresize(face, resize)
        if not color:
            # average the color channels to compute a gray levels
            # representation
            face = face.mean(axis=2)

        faces[i, ...] = face

    return faces


#
# Task #1:  Face Identification on picture with names
#


def _fetch_lfw_people(
    data_folder_path, slice_=None, color=False, resize=None, min_faces_per_person=0
):
    """Perform the actual data loading for the lfw people dataset

    This operation is meant to be cached by a joblib wrapper.
    """
    # scan the data folder content to retain people with more that
    # `min_faces_per_person` face pictures
    person_names, file_paths = [], []
    for person_name in sorted(listdir(data_folder_path)):
        folder_path = join(data_folder_path, person_name)
        if not isdir(folder_path):
            continue
        paths = [join(folder_path, f) for f in sorted(listdir(folder_path))]
        n_pictures = len(paths)
        if n_pictures >= min_faces_per_person:
            person_name = person_name.replace("_", " ")
            person_names.extend([person_name] * n_pictures)
            file_paths.extend(paths)

    n_faces = len(file_paths)
    if n_faces == 0:
        raise ValueError(
            "min_faces_per_person=%d is too restrictive" % min_faces_per_person
        )

    target_names = np.unique(person_names)
    target = np.searchsorted(target_names, person_names)

    faces = _load_imgs(file_paths, slice_, color, resize)

    # shuffle the faces with a deterministic RNG scheme to avoid having
    # all faces of the same person in a row, as it would break some
    # cross validation and learning algorithms such as SGD and online
    # k-means that make an IID assumption

    indices = np.arange(n_faces)
    np.random.RandomState(42).shuffle(indices)
    faces, target = faces[indices], target[indices]
    return faces, target, target_names


def fetch_lfw_people(
    *,
    data_home=None,
    funneled=True,
    resize=0.5,
    min_faces_per_person=0,
    color=False,
    slice_=(slice(70, 195), slice(78, 172)),
    download_if_missing=True,
    return_X_y=False,
):
    """Load the Labeled Faces in the Wild (LFW) people dataset \
(classification).

    Download it if necessary.

    =================   =======================
    Classes                                5749
    Samples total                         13233
    Dimensionality                         5828
    Features            real, between 0 and 255
    =================   =======================

    Read more in the :ref:`User Guide <labeled_faces_in_the_wild_dataset>`.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    funneled : bool, default=True
        Download and use the funneled variant of the dataset.

    resize : float, default=0.5
        Ratio used to resize the each face picture.

    min_faces_per_person : int, default=None
        The extracted dataset will only retain pictures of people that have at
        least `min_faces_per_person` different pictures.

    color : bool, default=False
        Keep the 3 RGB channels instead of averaging them to a single
        gray level channel. If color is True the shape of the data has
        one more dimension than the shape with color = False.

    slice_ : tuple of slice, default=(slice(70, 195), slice(78, 172))
        Provide a custom 2D slice (height, width) to extract the
        'interesting' part of the jpeg files and avoid use statistical
        correlation from the background

    download_if_missing : bool, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : bool, default=False
        If True, returns ``(dataset.data, dataset.target)`` instead of a Bunch
        object. See below for more information about the `dataset.data` and
        `dataset.target` object.

        .. versionadded:: 0.20

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : numpy array of shape (13233, 2914)
            Each row corresponds to a ravelled face image
            of original size 62 x 47 pixels.
            Changing the ``slice_`` or resize parameters will change the
            shape of the output.
        images : numpy array of shape (13233, 62, 47)
            Each row is a face image corresponding to one of the 5749 people in
            the dataset. Changing the ``slice_``
            or resize parameters will change the shape of the output.
        target : numpy array of shape (13233,)
            Labels associated to each face image.
            Those labels range from 0-5748 and correspond to the person IDs.
        DESCR : str
            Description of the Labeled Faces in the Wild (LFW) dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.20

    """
    lfw_home, data_folder_path = _check_fetch_lfw(
        data_home=data_home, funneled=funneled, download_if_missing=download_if_missing
    )
    logger.debug("Loading LFW people faces from %s", lfw_home)

    # wrap the loader in a memoizing function that will return memmaped data
    # arrays for optimal memory usage
    if parse_version(joblib.__version__) < parse_version("0.12"):
        # Deal with change of API in joblib
        m = Memory(cachedir=lfw_home, compress=6, verbose=0)
    else:
        m = Memory(location=lfw_home, compress=6, verbose=0)
    load_func = m.cache(_fetch_lfw_people)

    # load and memoize the pairs as np arrays
    faces, target, target_names = load_func(
        data_folder_path,
        resize=resize,
        min_faces_per_person=min_faces_per_person,
        color=color,
        slice_=slice_,
    )

    X = faces.reshape(len(faces), -1)

    fdescr = load_descr("lfw.rst")

    if return_X_y:
        return X, target

    # pack the results as a Bunch instance
    return Bunch(
        data=X, images=faces, target=target, target_names=target_names, DESCR=fdescr
    )


#
# Task #2:  Face Verification on pairs of face pictures
#


def _fetch_lfw_pairs(
    index_file_path, data_folder_path, slice_=None, color=False, resize=None
):
    """Perform the actual data loading for the LFW pairs dataset

    This operation is meant to be cached by a joblib wrapper.
    """
    # parse the index file to find the number of pairs to be able to allocate
    # the right amount of memory before starting to decode the jpeg files
    with open(index_file_path, "rb") as index_file:
        split_lines = [ln.decode().strip().split("\t") for ln in index_file]
    pair_specs = [sl for sl in split_lines if len(sl) > 2]
    n_pairs = len(pair_specs)

    # iterating over the metadata lines for each pair to find the filename to
    # decode and load in memory
    target = np.zeros(n_pairs, dtype=int)
    file_paths = list()
    for i, components in enumerate(pair_specs):
        if len(components) == 3:
            target[i] = 1
            pair = (
                (components[0], int(components[1]) - 1),
                (components[0], int(components[2]) - 1),
            )
        elif len(components) == 4:
            target[i] = 0
            pair = (
                (components[0], int(components[1]) - 1),
                (components[2], int(components[3]) - 1),
            )
        else:
            raise ValueError("invalid line %d: %r" % (i + 1, components))
        for j, (name, idx) in enumerate(pair):
            try:
                person_folder = join(data_folder_path, name)
            except TypeError:
                person_folder = join(data_folder_path, str(name, "UTF-8"))
            filenames = list(sorted(listdir(person_folder)))
            file_path = join(person_folder, filenames[idx])
            file_paths.append(file_path)

    pairs = _load_imgs(file_paths, slice_, color, resize)
    shape = list(pairs.shape)
    n_faces = shape.pop(0)
    shape.insert(0, 2)
    shape.insert(0, n_faces // 2)
    pairs.shape = shape

    return pairs, target, np.array(["Different persons", "Same person"])


def fetch_lfw_pairs(
    *,
    subset="train",
    data_home=None,
    funneled=True,
    resize=0.5,
    color=False,
    slice_=(slice(70, 195), slice(78, 172)),
    download_if_missing=True,
):
    """Load the Labeled Faces in the Wild (LFW) pairs dataset (classification).

    Download it if necessary.

    =================   =======================
    Classes                                   2
    Samples total                         13233
    Dimensionality                         5828
    Features            real, between 0 and 255
    =================   =======================

    In the official `README.txt`_ this task is described as the
    "Restricted" task.  As I am not sure as to implement the
    "Unrestricted" variant correctly, I left it as unsupported for now.

      .. _`README.txt`: http://vis-www.cs.umass.edu/lfw/README.txt

    The original images are 250 x 250 pixels, but the default slice and resize
    arguments reduce them to 62 x 47.

    Read more in the :ref:`User Guide <labeled_faces_in_the_wild_dataset>`.

    Parameters
    ----------
    subset : {'train', 'test', '10_folds'}, default='train'
        Select the dataset to load: 'train' for the development training
        set, 'test' for the development test set, and '10_folds' for the
        official evaluation set that is meant to be used with a 10-folds
        cross validation.

    data_home : str, default=None
        Specify another download and cache folder for the datasets. By
        default all scikit-learn data is stored in '~/scikit_learn_data'
        subfolders.

    funneled : bool, default=True
        Download and use the funneled variant of the dataset.

    resize : float, default=0.5
        Ratio used to resize the each face picture.

    color : bool, default=False
        Keep the 3 RGB channels instead of averaging them to a single
        gray level channel. If color is True the shape of the data has
        one more dimension than the shape with color = False.

    slice_ : tuple of slice, default=(slice(70, 195), slice(78, 172))
        Provide a custom 2D slice (height, width) to extract the
        'interesting' part of the jpeg files and avoid use statistical
        correlation from the background

    download_if_missing : bool, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray of shape (2200, 5828). Shape depends on ``subset``.
            Each row corresponds to 2 ravel'd face images
            of original size 62 x 47 pixels.
            Changing the ``slice_``, ``resize`` or ``subset`` parameters
            will change the shape of the output.
        pairs : ndarray of shape (2200, 2, 62, 47). Shape depends on ``subset``
            Each row has 2 face images corresponding
            to same or different person from the dataset
            containing 5749 people. Changing the ``slice_``,
            ``resize`` or ``subset`` parameters will change the shape of the
            output.
        target : numpy array of shape (2200,). Shape depends on ``subset``.
            Labels associated to each pair of images.
            The two label values being different persons or the same person.
        DESCR : str
            Description of the Labeled Faces in the Wild (LFW) dataset.

    """
    lfw_home, data_folder_path = _check_fetch_lfw(
        data_home=data_home, funneled=funneled, download_if_missing=download_if_missing
    )
    logger.debug("Loading %s LFW pairs from %s", subset, lfw_home)

    # wrap the loader in a memoizing function that will return memmaped data
    # arrays for optimal memory usage
    if parse_version(joblib.__version__) < parse_version("0.12"):
        # Deal with change of API in joblib
        m = Memory(cachedir=lfw_home, compress=6, verbose=0)
    else:
        m = Memory(location=lfw_home, compress=6, verbose=0)
    load_func = m.cache(_fetch_lfw_pairs)

    # select the right metadata file according to the requested subset
    label_filenames = {
        "train": "pairsDevTrain.txt",
        "test": "pairsDevTest.txt",
        "10_folds": "pairs.txt",
    }
    if subset not in label_filenames:
        raise ValueError(
            "subset='%s' is invalid: should be one of %r"
            % (subset, list(sorted(label_filenames.keys())))
        )
    index_file_path = join(lfw_home, label_filenames[subset])

    # load and memoize the pairs as np arrays
    pairs, target, target_names = load_func(
        index_file_path, data_folder_path, resize=resize, color=color, slice_=slice_
    )

    fdescr = load_descr("lfw.rst")

    # pack the results as a Bunch instance
    return Bunch(
        data=pairs.reshape(len(pairs), -1),
        pairs=pairs,
        target=target,
        target_names=target_names,
        DESCR=fdescr,
    )
