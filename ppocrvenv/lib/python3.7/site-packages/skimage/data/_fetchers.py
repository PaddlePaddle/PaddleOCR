"""Standard test images.

For more images, see

 - http://sipi.usc.edu/database/database.php

"""
import numpy as np
import shutil
from packaging import version

from ..util.dtype import img_as_bool
from ._binary_blobs import binary_blobs
from ._registry import registry, legacy_registry, registry_urls

from .. import __version__

import os.path as osp
import os

legacy_data_dir = osp.abspath(osp.dirname(__file__))
skimage_distribution_dir = osp.join(legacy_data_dir, '..')

try:
    from pooch import file_hash
except ModuleNotFoundError:
    # Function taken from
    # https://github.com/fatiando/pooch/blob/master/pooch/utils.py
    def file_hash(fname, alg="sha256"):
        """
        Calculate the hash of a given file.
        Useful for checking if a file has changed or been corrupted.
        Parameters
        ----------
        fname : str
            The name of the file.
        alg : str
            The type of the hashing algorithm
        Returns
        -------
        hash : str
            The hash of the file.
        Examples
        --------
        >>> fname = "test-file-for-hash.txt"
        >>> with open(fname, "w") as f:
        ...     __ = f.write("content of the file")
        >>> print(file_hash(fname))
        0fc74468e6a9a829f103d069aeb2bb4f8646bad58bf146bb0e3379b759ec4a00
        >>> import os
        >>> os.remove(fname)
        """
        import hashlib
        if alg not in hashlib.algorithms_available:
            raise ValueError(f'Algorithm \'{alg}\' not available in hashlib')
        # Calculate the hash in chunks to avoid overloading the memory
        chunksize = 65536
        hasher = hashlib.new(alg)
        with open(fname, "rb") as fin:
            buff = fin.read(chunksize)
            while buff:
                hasher.update(buff)
                buff = fin.read(chunksize)
        return hasher.hexdigest()


def _has_hash(path, expected_hash):
    """Check if the provided path has the expected hash."""
    if not osp.exists(path):
        return False
    return file_hash(path) == expected_hash


def create_image_fetcher():
    try:
        import pooch
        # older versions of Pooch don't have a __version__ attribute
        if not hasattr(pooch, '__version__'):
            retry = {}
        else:
            pooch_version = pooch.__version__.lstrip('v')
            retry = {'retry_if_failed': 3}
            # Keep version check in synch with
            # scikit-image/requirements/optional.txt
            if version.parse(pooch_version) < version.parse('1.3.0'):
                # we need a more recent version of pooch to retry
                retry = {}
    except ImportError:
        # Without pooch, fallback on the standard data directory
        # which for now, includes a few limited data samples
        return None, legacy_data_dir

    # Pooch expects a `+` to exist in development versions.
    # Since scikit-image doesn't follow that convention, we have to manually
    # remove `.dev` with a `+` if it exists.
    # This helps pooch understand that it should look in master
    # to find the required files
    if '+git' in __version__:
        skimage_version_for_pooch = __version__.replace('.dev0+git', '+git')
    else:
        skimage_version_for_pooch = __version__.replace('.dev', '+')

    if '+' in skimage_version_for_pooch:
        url = ("https://github.com/scikit-image/scikit-image/raw/"
               "{version}/skimage/")
    else:
        url = ("https://github.com/scikit-image/scikit-image/raw/"
               "v{version}/skimage/")

    # Create a new friend to manage your sample data storage
    image_fetcher = pooch.create(
        # Pooch uses appdirs to select an appropriate directory for the cache
        # on each platform.
        # https://github.com/ActiveState/appdirs
        # On linux this converges to
        # '$HOME/.cache/scikit-image'
        # With a version qualifier
        path=pooch.os_cache("scikit-image"),
        base_url=url,
        version=skimage_version_for_pooch,
        version_dev="v0.19.x",
        env="SKIMAGE_DATADIR",
        registry=registry,
        urls=registry_urls,
        # Note: this should read `retry_if_failed=3,`, but we generate that
        # dynamically at import time above, in case installed pooch is a less
        # recent version
        **retry,
    )

    data_dir = osp.join(str(image_fetcher.abspath), 'data')
    return image_fetcher, data_dir


image_fetcher, data_dir = create_image_fetcher()

if image_fetcher is None:
    has_pooch = False
else:
    has_pooch = True


def _skip_pytest_case_requiring_pooch(data_filename):
    """If a test case is calling pooch, skip it.

    This running the test suite in environments without internet
    access, skipping only the tests that try to fetch external data.
    """

    # Check if pytest is currently running.
    # Packagers might use pytest to run the tests suite, but may not
    # want to run it online with pooch as a dependency.
    # As such, we will avoid failing the test, and silently skipping it.
    if 'PYTEST_CURRENT_TEST' in os.environ:
        # https://docs.pytest.org/en/latest/example/simple.html#pytest-current-test-environment-variable  # noqa
        import pytest
        # Pytest skip raises an exception that allows the
        # tests to be skipped
        pytest.skip(f'Unable to download {data_filename}',
                    allow_module_level=True)


def _fetch(data_filename):
    """Fetch a given data file from either the local cache or the repository.

    This function provides the path location of the data file given
    its name in the scikit-image repository.

    Parameters
    ----------
    data_filename:
        Name of the file in the scikit-image repository. e.g.
        'restoration/tess/camera_rl.npy'.

    Returns
    -------
    Path of the local file as a python string.

    Raises
    ------
    KeyError:
        If the filename is not known to the scikit-image distribution.

    ModuleNotFoundError:
        If the filename is known to the scikit-image distribution but pooch
        is not installed.

    ConnectionError:
        If scikit-image is unable to connect to the internet but the
        dataset has not been downloaded yet.
    """
    resolved_path = osp.join(data_dir, '..', data_filename)
    expected_hash = registry[data_filename]

    # Case 1:
    # The file may already be in the data_dir.
    # We may have decided to ship it in the scikit-image distribution.
    if _has_hash(resolved_path, expected_hash):
        # Nothing to be done, file is where it is expected to be
        return resolved_path

    # Case 2:
    # The user is using a cloned version of the github repo, which
    # contains both the publicly shipped data, and test data.
    # In this case, the file would be located relative to the
    # skimage_distribution_dir
    gh_repository_path = osp.join(skimage_distribution_dir, data_filename)
    if _has_hash(gh_repository_path, expected_hash):
        parent = osp.dirname(resolved_path)
        os.makedirs(parent, exist_ok=True)
        shutil.copy2(gh_repository_path, resolved_path)
        return resolved_path

    # Case 3:
    # Pooch not found.
    if image_fetcher is None:
        _skip_pytest_case_requiring_pooch(data_filename)

        raise ModuleNotFoundError(
            "The requested file is part of the scikit-image distribution, "
            "but requires the installation of an optional dependency, pooch. "
            "To install pooch, use your preferred python package manager. "
            "Follow installation instruction found at "
            "https://scikit-image.org/docs/stable/install.html"
        )

    # Case 4:
    # Pooch needs to download the data. Let the image fetcher to search for
    # our data. A ConnectionError is raised if no internet connection is
    # available.
    try:
        resolved_path = image_fetcher.fetch(data_filename)
    except ConnectionError as err:
        _skip_pytest_case_requiring_pooch(data_filename)

        # If we decide in the future to suppress the underlying 'requests'
        # error, change this to `raise ... from None`. See PEP 3134.
        raise ConnectionError(
            'Tried to download a scikit-image dataset, but no internet '
            'connection is available. To avoid this message in the '
            'future, try `skimage.data.download_all()` when you are '
            'connected to the internet.'
        ) from err
    return resolved_path


def _init_pooch():
    os.makedirs(data_dir, exist_ok=True)

    # Copy in the README.txt if it doesn't already exist.
    # If the file was originally copied to the data cache directory read-only
    # then we cannot overwrite it, nor do we need to copy on every init.
    # In general, as the data cache directory contains the scikit-image version
    # it should not be necessary to overwrite this file as it should not
    # change.
    dest_path = osp.join(data_dir, 'README.txt')
    if not os.path.isfile(dest_path):
        shutil.copy2(osp.join(skimage_distribution_dir, 'data', 'README.txt'),
                     dest_path)

    # Fetch all legacy data so that it is available by default
    for filename in legacy_registry:
        _fetch(filename)


# This function creates directories, and has been the source of issues for
# downstream users, see
# https://github.com/scikit-image/scikit-image/issues/4660
# https://github.com/scikit-image/scikit-image/issues/4664
if has_pooch:
    _init_pooch()


def download_all(directory=None):
    """Download all datasets for use with scikit-image offline.

    Scikit-image datasets are no longer shipped with the library by default.
    This allows us to use higher quality datasets, while keeping the
    library download size small.

    This function requires the installation of an optional dependency, pooch,
    to download the full dataset. Follow installation instruction found at

        https://scikit-image.org/docs/stable/install.html

    Call this function to download all sample images making them available
    offline on your machine.

    Parameters
    ----------
    directory: path-like, optional
        The directory where the dataset should be stored.

    Raises
    ------
    ModuleNotFoundError:
        If pooch is not install, this error will be raised.

    Notes
    -----
    scikit-image will only search for images stored in the default directory.
    Only specify the directory if you wish to download the images to your own
    folder for a particular reason. You can access the location of the default
    data directory by inspecting the variable `skimage.data.data_dir`.
    """

    if image_fetcher is None:
        raise ModuleNotFoundError(
            "To download all package data, scikit-image needs an optional "
            "dependency, pooch."
            "To install pooch, follow our installation instructions found at "
            "https://scikit-image.org/docs/stable/install.html"
        )
    # Consider moving this kind of logic to Pooch
    old_dir = image_fetcher.path
    try:
        if directory is not None:
            image_fetcher.path = directory

        for filename in image_fetcher.registry:
            _fetch(filename)
    finally:
        image_fetcher.path = old_dir


def lbp_frontal_face_cascade_filename():
    """Return the path to the XML file containing the weak classifier cascade.

    These classifiers were trained using LBP features. The file is part
    of the OpenCV repository [1]_.

    References
    ----------
    .. [1] OpenCV lbpcascade trained files
           https://github.com/opencv/opencv/tree/master/data/lbpcascades
    """

    return _fetch('data/lbpcascade_frontalface_opencv.xml')


def _load(f, as_gray=False):
    """Load an image file located in the data directory.

    Parameters
    ----------
    f : string
        File name.
    as_gray : bool, optional
        Whether to convert the image to grayscale.

    Returns
    -------
    img : ndarray
        Image loaded from ``skimage.data_dir``.
    """
    # importing io is quite slow since it scans all the backends
    # we lazy import it here
    from ..io import imread
    return imread(_fetch(f), as_gray=as_gray)


def camera():
    """Gray-level "camera" image.

    Can be used for segmentation and denoising examples.

    Returns
    -------
    camera : (512, 512) uint8 ndarray
        Camera image.

    Notes
    -----
    No copyright restrictions. CC0 by the photographer (Lav Varshney).

    .. versionchanged:: 0.18
        This image was replaced due to copyright restrictions. For more
        information, please see [1]_.

    References
    ----------
    .. [1] https://github.com/scikit-image/scikit-image/issues/3927
    """
    return _load("data/camera.png")


def eagle():
    """A golden eagle.

    Suitable for examples on segmentation, Hough transforms, and corner
    detection.

    Notes
    -----
    No copyright restrictions. CC0 by the photographer (Dayane Machado).

    Returns
    -------
    eagle : (2019, 1826) uint8 ndarray
        Eagle image.
    """
    return _load("data/eagle.png")


def astronaut():
    """Color image of the astronaut Eileen Collins.

    Photograph of Eileen Collins, an American astronaut. She was selected
    as an astronaut in 1992 and first piloted the space shuttle STS-63 in
    1995. She retired in 2006 after spending a total of 38 days, 8 hours
    and 10 minutes in outer space.

    This image was downloaded from the NASA Great Images database
    <https://flic.kr/p/r9qvLn>`__.

    No known copyright restrictions, released into the public domain.

    Returns
    -------
    astronaut : (512, 512, 3) uint8 ndarray
        Astronaut image.
    """

    return _load("data/astronaut.png")


def brick():
    """Brick wall.

    Returns
    -------
    brick : (512, 512) uint8 image
        A small section of a brick wall.

    Notes
    -----
    The original image was downloaded from
    `CC0Textures <https://cc0textures.com/view.php?tex=Bricks25>`_ and licensed
    under the Creative Commons CC0 License.

    A perspective transform was then applied to the image, prior to
    rotating it by 90 degrees, cropping and scaling it to obtain the final
    image.
    """

    """
    The following code was used to obtain the final image.

    >>> import sys; print(sys.version)
    >>> import platform; print(platform.platform())
    >>> import skimage; print(f'scikit-image version: {skimage.__version__}')
    >>> import numpy; print(f'numpy version: {numpy.__version__}')
    >>> import imageio; print(f'imageio version {imageio.__version__}')
    3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21)
    [GCC 7.3.0]
    Linux-5.0.0-20-generic-x86_64-with-debian-buster-sid
    scikit-image version: 0.16.dev0
    numpy version: 1.16.4
    imageio version 2.4.1

    >>> import requests
    >>> import zipfile
    >>> url = 'https://cdn.struffelproductions.com/file/cc0textures/Bricks25/%5B2K%5DBricks25.zip'
    >>> r = requests.get(url)
    >>> with open('[2K]Bricks25.zip', 'bw') as f:
    ...     f.write(r.content)
    >>> with zipfile.ZipFile('[2K]Bricks25.zip') as z:
    ... z.extract('Bricks25_col.jpg')

    >>> from numpy.linalg import inv
    >>> from skimage.transform import rescale, warp, rotate
    >>> from skimage.color import rgb2gray
    >>> from imageio import imread, imwrite
    >>> from skimage import img_as_ubyte
    >>> import numpy as np


    >>> # Obtained playing around with GIMP 2.10 with their perspective tool
    >>> H = inv(np.asarray([[ 0.54764, -0.00219, 0],
    ...                     [-0.12822,  0.54688, 0],
    ...                     [-0.00022,        0, 1]]))


    >>> brick_orig = imread('Bricks25_col.jpg')
    >>> brick = warp(brick_orig, H)
    >>> brick = rescale(brick[:1024, :1024], (0.5, 0.5, 1))
    >>> brick = rotate(brick, -90)
    >>> imwrite('brick.png', img_as_ubyte(rgb2gray(brick)))
    """
    return _load("data/brick.png", as_gray=True)


def grass():
    """Grass.

    Returns
    -------
    grass : (512, 512) uint8 image
        Some grass.

    Notes
    -----
    The original image was downloaded from
    `DeviantArt <https://www.deviantart.com/linolafett/art/Grass-01-434853879>`__
    and licensed under the Creative Commons CC0 License.

    The downloaded image was cropped to include a region of ``(512, 512)``
    pixels around the top left corner, converted to grayscale, then to uint8
    prior to saving the result in PNG format.

    """

    """
    The following code was used to obtain the final image.

    >>> import sys; print(sys.version)
    >>> import platform; print(platform.platform())
    >>> import skimage; print(f'scikit-image version: {skimage.__version__}')
    >>> import numpy; print(f'numpy version: {numpy.__version__}')
    >>> import imageio; print(f'imageio version {imageio.__version__}')
    3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21)
    [GCC 7.3.0]
    Linux-5.0.0-20-generic-x86_64-with-debian-buster-sid
    scikit-image version: 0.16.dev0
    numpy version: 1.16.4
    imageio version 2.4.1

    >>> import requests
    >>> import zipfile
    >>> url = 'https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/a407467e-4ff0-49f1-923f-c9e388e84612/d76wfef-2878b78d-5dce-43f9-be36-26ec9bc0df3b.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2E0MDc0NjdlLTRmZjAtNDlmMS05MjNmLWM5ZTM4OGU4NDYxMlwvZDc2d2ZlZi0yODc4Yjc4ZC01ZGNlLTQzZjktYmUzNi0yNmVjOWJjMGRmM2IuanBnIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.98hIcOTCqXWQ67Ec5bM5eovKEn2p91mWB3uedH61ynI'
    >>> r = requests.get(url)
    >>> with open('grass_orig.jpg', 'bw') as f:
    ...     f.write(r.content)
    >>> grass_orig = imageio.imread('grass_orig.jpg')
    >>> grass = skimage.img_as_ubyte(skimage.color.rgb2gray(grass_orig[:512, :512]))
    >>> imageio.imwrite('grass.png', grass)
    """
    return _load("data/grass.png", as_gray=True)


def gravel():
    """Gravel

    Returns
    -------
    gravel : (512, 512) uint8 image
        Grayscale gravel sample.

    Notes
    -----
    The original image was downloaded from
    `CC0Textures <https://cc0textures.com/view.php?tex=Gravel04>`__ and
    licensed under the Creative Commons CC0 License.

    The downloaded image was then rescaled to ``(1024, 1024)``, then the
    top left ``(512, 512)`` pixel region  was cropped prior to converting the
    image to grayscale and uint8 data type. The result was saved using the
    PNG format.
    """

    """
    The following code was used to obtain the final image.

    >>> import sys; print(sys.version)
    >>> import platform; print(platform.platform())
    >>> import skimage; print(f'scikit-image version: {skimage.__version__}')
    >>> import numpy; print(f'numpy version: {numpy.__version__}')
    >>> import imageio; print(f'imageio version {imageio.__version__}')
    3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21)
    [GCC 7.3.0]
    Linux-5.0.0-20-generic-x86_64-with-debian-buster-sid
    scikit-image version: 0.16.dev0
    numpy version: 1.16.4
    imageio version 2.4.1

    >>> import requests
    >>> import zipfile

    >>> url = 'https://cdn.struffelproductions.com/file/cc0textures/Gravel04/%5B2K%5DGravel04.zip'
    >>> r = requests.get(url)
    >>> with open('[2K]Gravel04.zip', 'bw') as f:
    ...     f.write(r.content)

    >>> with zipfile.ZipFile('[2K]Gravel04.zip') as z:
    ...     z.extract('Gravel04_col.jpg')

    >>> from skimage.transform import resize
    >>> gravel_orig = imageio.imread('Gravel04_col.jpg')
    >>> gravel = resize(gravel_orig, (1024, 1024))
    >>> gravel = skimage.img_as_ubyte(skimage.color.rgb2gray(gravel[:512, :512]))
    >>> imageio.imwrite('gravel.png', gravel)
    """
    return _load("data/gravel.png", as_gray=True)


def text():
    """Gray-level "text" image used for corner detection.

    Notes
    -----
    This image was downloaded from Wikipedia
    <https://en.wikipedia.org/wiki/File:Corner.png>`__.

    No known copyright restrictions, released into the public domain.

    Returns
    -------
    text : (172, 448) uint8 ndarray
        Text image.
    """

    return _load("data/text.png")


def checkerboard():
    """Checkerboard image.

    Checkerboards are often used in image calibration, since the
    corner-points are easy to locate.  Because of the many parallel
    edges, they also visualise distortions particularly well.

    Returns
    -------
    checkerboard : (200, 200) uint8 ndarray
        Checkerboard image.
    """
    return _load("data/chessboard_GRAY.png")


def cells3d():
    """3D fluorescence microscopy image of cells.

    The returned data is a 3D multichannel array with dimensions provided in
    ``(z, c, y, x)`` order. Each voxel has a size of ``(0.29 0.26 0.26)``
    micrometer. Channel 0 contains cell membranes, channel 1 contains nuclei.

    Returns
    -------
    cells3d: (60, 2, 256, 256) uint16 ndarray
        The volumetric images of cells taken with an optical microscope.

    Notes
    -----
    The data for this was provided by the Allen Institute for Cell Science.

    It has been downsampled by a factor of 4 in the row and column dimensions
    to reduce computational time.

    The microscope reports the following voxel spacing in microns:

        * Original voxel size is ``(0.290, 0.065, 0.065)``.
        * Scaling factor is ``(1, 4, 4)`` in each dimension.
        * After rescaling the voxel size is ``(0.29 0.26 0.26)``.
    """

    return _load("data/cells3d.tif")


def human_mitosis():
    """Image of human cells undergoing mitosis.

    Returns
    -------
    human_mitosis: (512, 512) uint8 ndimage
        Data of human cells undergoing mitosis taken during the preperation
        of the manuscript in [1]_.

    Notes
    -----
    Copyright David Root. Licensed under CC-0 [2]_.

    References
    ----------
    .. [1] Moffat J, Grueneberg DA, Yang X, Kim SY, Kloepfer AM, Hinkle G,
           Piqani B, Eisenhaure TM, Luo B, Grenier JK, Carpenter AE, Foo SY,
           Stewart SA, Stockwell BR, Hacohen N, Hahn WC, Lander ES,
           Sabatini DM, Root DE (2006) A lentiviral RNAi library for human and
           mouse genes applied to an arrayed viral high-content screen. Cell,
           124(6):1283-98 / :DOI: `10.1016/j.cell.2006.01.040` PMID 16564017

    .. [2] GitHub licensing discussion
           https://github.com/CellProfiler/examples/issues/41

    """
    return _load('data/mitosis.tif')


def cell():
    """Cell floating in saline.

    This is a quantitative phase image retrieved from a digital hologram using
    the Python library ``qpformat``. The image shows a cell with high phase
    value, above the background phase.

    Because of a banding pattern artifact in the background, this image is a
    good test of thresholding algorithms. The pixel spacing is 0.107 µm.

    These data were part of a comparison between several refractive index
    retrieval techniques for spherical objects as part of [1]_.

    This image is CC0, dedicated to the public domain. You may copy, modify, or
    distribute it without asking permission.

    Returns
    -------
    cell : (660, 550) uint8 array
        Image of a cell.

    References
    ----------
    .. [1] Paul Müller, Mirjam Schürmann, Salvatore Girardo, Gheorghe Cojoc,
           and Jochen Guck. "Accurate evaluation of size and refractive index
           for spherical objects in quantitative phase imaging." Optics Express
           26(8): 10729-10743 (2018). :DOI:`10.1364/OE.26.010729`
    """
    return _load('data/cell.png')


def coins():
    """Greek coins from Pompeii.

    This image shows several coins outlined against a gray background.
    It is especially useful in, e.g. segmentation tests, where
    individual objects need to be identified against a background.
    The background shares enough grey levels with the coins that a
    simple segmentation is not sufficient.

    Notes
    -----
    This image was downloaded from the
    `Brooklyn Museum Collection
    <https://www.brooklynmuseum.org/opencollection/archives/image/51611>`__.

    No known copyright restrictions.

    Returns
    -------
    coins : (303, 384) uint8 ndarray
        Coins image.
    """
    return _load("data/coins.png")


def kidney():
    """Mouse kidney tissue.

    This biological tissue on a pre-prepared slide was imaged with confocal
    fluorescence microscopy (Nikon C1 inverted microscope).
    Image shape is (16, 512, 512, 3). That is 512x512 pixels in X-Y,
    16 image slices in Z, and 3 color channels
    (emission wavelengths 450nm, 515nm, and 605nm, respectively).
    Real-space voxel size is 1.24 microns in X-Y, and 1.25 microns in Z.
    Data type is unsigned 16-bit integers.

    Notes
    -----
    This image was acquired by Genevieve Buckley at Monasoh Micro Imaging in
    2018.
    License: CC0

    Returns
    -------
    kidney : (16, 512, 512, 3) uint16 ndarray
        Kidney 3D multichannel image.
    """
    return _load("data/kidney.tif")


def lily():
    """Lily of the valley plant stem.

    This plant stem on a pre-prepared slide was imaged with confocal
    fluorescence microscopy (Nikon C1 inverted microscope).
    Image shape is (922, 922, 4). That is 922x922 pixels in X-Y,
    with 4 color channels.
    Real-space voxel size is 1.24 microns in X-Y.
    Data type is unsigned 16-bit integers.

    Notes
    -----
    This image was acquired by Genevieve Buckley at Monasoh Micro Imaging in
    2018.
    License: CC0

    Returns
    -------
    lily : (922, 922, 4) uint16 ndarray
        Lily 2D multichannel image.
    """
    return _load("data/lily.tif")


def logo():
    """Scikit-image logo, a RGBA image.

    Returns
    -------
    logo : (500, 500, 4) uint8 ndarray
        Logo image.
    """
    return _load("data/logo.png")


def microaneurysms():
    """Gray-level "microaneurysms" image.

    Detail from an image of the retina (green channel).
    The image is a crop of image 07_dr.JPG from the
    High-Resolution Fundus (HRF) Image Database:
    https://www5.cs.fau.de/research/data/fundus-images/

    Notes
    -----
    No copyright restrictions. CC0 given by owner (Andreas Maier).

    Returns
    -------
    microaneurysms : (102, 102) uint8 ndarray
        Retina image with lesions.

    References
    ----------
    .. [1] Budai, A., Bock, R, Maier, A., Hornegger, J.,
           Michelson, G. (2013).  Robust Vessel Segmentation in Fundus
           Images. International Journal of Biomedical Imaging, vol. 2013,
           2013.
           :DOI:`10.1155/2013/154860`
    """
    return _load("data/microaneurysms.png")


def moon():
    """Surface of the moon.

    This low-contrast image of the surface of the moon is useful for
    illustrating histogram equalization and contrast stretching.

    Returns
    -------
    moon : (512, 512) uint8 ndarray
        Moon image.
    """
    return _load("data/moon.png")


def page():
    """Scanned page.

    This image of printed text is useful for demonstrations requiring uneven
    background illumination.

    Returns
    -------
    page : (191, 384) uint8 ndarray
        Page image.
    """
    return _load("data/page.png")


def horse():
    """Black and white silhouette of a horse.

    This image was downloaded from
    `openclipart <http://openclipart.org/detail/158377/horse-by-marauder>`

    No copyright restrictions. CC0 given by owner (Andreas Preuss (marauder)).

    Returns
    -------
    horse : (328, 400) bool ndarray
        Horse image.
    """
    return img_as_bool(_load("data/horse.png", as_gray=True))


def clock():
    """Motion blurred clock.

    This photograph of a wall clock was taken while moving the camera in an
    aproximately horizontal direction.  It may be used to illustrate
    inverse filters and deconvolution.

    Released into the public domain by the photographer (Stefan van der Walt).

    Returns
    -------
    clock : (300, 400) uint8 ndarray
        Clock image.
    """
    return _load("data/clock_motion.png")


def immunohistochemistry():
    """Immunohistochemical (IHC) staining with hematoxylin counterstaining.

    This picture shows colonic glands where the IHC expression of FHL2 protein
    is revealed with DAB. Hematoxylin counterstaining is applied to enhance the
    negative parts of the tissue.

    This image was acquired at the Center for Microscopy And Molecular Imaging
    (CMMI).

    No known copyright restrictions.

    Returns
    -------
    immunohistochemistry : (512, 512, 3) uint8 ndarray
        Immunohistochemistry image.
    """
    return _load("data/ihc.png")


def chelsea():
    """Chelsea the cat.

    An example with texture, prominent edges in horizontal and diagonal
    directions, as well as features of differing scales.

    Notes
    -----
    No copyright restrictions.  CC0 by the photographer (Stefan van der Walt).

    Returns
    -------
    chelsea : (300, 451, 3) uint8 ndarray
        Chelsea image.
    """
    return _load("data/chelsea.png")


# Define an alias for chelsea that is more descriptive.
cat = chelsea


def coffee():
    """Coffee cup.

    This photograph is courtesy of Pikolo Espresso Bar.
    It contains several elliptical shapes as well as varying texture (smooth
    porcelain to course wood grain).

    Notes
    -----
    No copyright restrictions.  CC0 by the photographer (Rachel Michetti).

    Returns
    -------
    coffee : (400, 600, 3) uint8 ndarray
        Coffee image.
    """
    return _load("data/coffee.png")


def hubble_deep_field():
    """Hubble eXtreme Deep Field.

    This photograph contains the Hubble Telescope's farthest ever view of
    the universe. It can be useful as an example for multi-scale
    detection.

    Notes
    -----
    This image was downloaded from
    `HubbleSite
    <http://hubblesite.org/newscenter/archive/releases/2012/37/image/a/>`__.

    The image was captured by NASA and `may be freely used in the public domain
    <http://www.nasa.gov/audience/formedia/features/MP_Photo_Guidelines.html>`_.

    Returns
    -------
    hubble_deep_field : (872, 1000, 3) uint8 ndarray
        Hubble deep field image.
    """
    return _load("data/hubble_deep_field.jpg")


def retina():
    """Human retina.

    This image of a retina is useful for demonstrations requiring circular
    images.

    Notes
    -----
    This image was downloaded from
    `wikimedia <https://commons.wikimedia.org/wiki/File:Fundus_photograph_of_normal_left_eye.jpg>`.
    This file is made available under the Creative Commons CC0 1.0 Universal
    Public Domain Dedication.

    References
    ----------
    .. [1] Häggström, Mikael (2014). "Medical gallery of Mikael Häggström 2014".
           WikiJournal of Medicine 1 (2). :DOI:`10.15347/wjm/2014.008`.
           ISSN 2002-4436. Public Domain

    Returns
    -------
    retina : (1411, 1411, 3) uint8 ndarray
        Retina image in RGB.
    """
    return _load("data/retina.jpg")


def shepp_logan_phantom():
    """Shepp Logan Phantom.

    References
    ----------
    .. [1] L. A. Shepp and B. F. Logan, "The Fourier reconstruction of a head
           section," in IEEE Transactions on Nuclear Science, vol. 21,
           no. 3, pp. 21-43, June 1974. :DOI:`10.1109/TNS.1974.6499235`

    Returns
    -------
    phantom : (400, 400) float64 image
        Image of the Shepp-Logan phantom in grayscale.
    """
    return _load("data/phantom.png", as_gray=True)


def colorwheel():
    """Color Wheel.

    Returns
    -------
    colorwheel : (370, 371, 3) uint8 image
        A colorwheel.
    """
    return _load("data/color.png")


def rocket():
    """Launch photo of DSCOVR on Falcon 9 by SpaceX.

    This is the launch photo of Falcon 9 carrying DSCOVR lifted off from
    SpaceX's Launch Complex 40 at Cape Canaveral Air Force Station, FL.

    Notes
    -----
    This image was downloaded from
    `SpaceX Photos
    <https://www.flickr.com/photos/spacexphotos/16511594820/in/photostream/>`__.

    The image was captured by SpaceX and `released in the public domain
    <http://arstechnica.com/tech-policy/2015/03/elon-musk-puts-spacex-photos-into-the-public-domain/>`_.

    Returns
    -------
    rocket : (427, 640, 3) uint8 ndarray
        Rocket image.
    """
    return _load("data/rocket.jpg")


def stereo_motorcycle():
    """Rectified stereo image pair with ground-truth disparities.

    The two images are rectified such that every pixel in the left image has
    its corresponding pixel on the same scanline in the right image. That means
    that both images are warped such that they have the same orientation but a
    horizontal spatial offset (baseline). The ground-truth pixel offset in
    column direction is specified by the included disparity map.

    The two images are part of the Middlebury 2014 stereo benchmark. The
    dataset was created by Nera Nesic, Porter Westling, Xi Wang, York Kitajima,
    Greg Krathwohl, and Daniel Scharstein at Middlebury College. A detailed
    description of the acquisition process can be found in [1]_.

    The images included here are down-sampled versions of the default exposure
    images in the benchmark. The images are down-sampled by a factor of 4 using
    the function `skimage.transform.downscale_local_mean`. The calibration data
    in the following and the included ground-truth disparity map are valid for
    the down-sampled images::

        Focal length:           994.978px
        Principal point x:      311.193px
        Principal point y:      254.877px
        Principal point dx:      31.086px
        Baseline:               193.001mm

    Returns
    -------
    img_left : (500, 741, 3) uint8 ndarray
        Left stereo image.
    img_right : (500, 741, 3) uint8 ndarray
        Right stereo image.
    disp : (500, 741, 3) float ndarray
        Ground-truth disparity map, where each value describes the offset in
        column direction between corresponding pixels in the left and the right
        stereo images. E.g. the corresponding pixel of
        ``img_left[10, 10 + disp[10, 10]]`` is ``img_right[10, 10]``.
        NaNs denote pixels in the left image that do not have ground-truth.

    Notes
    -----
    The original resolution images, images with different exposure and
    lighting, and ground-truth depth maps can be found at the Middlebury
    website [2]_.

    References
    ----------
    .. [1] D. Scharstein, H. Hirschmueller, Y. Kitajima, G. Krathwohl, N.
           Nesic, X. Wang, and P. Westling. High-resolution stereo datasets
           with subpixel-accurate ground truth. In German Conference on Pattern
           Recognition (GCPR 2014), Muenster, Germany, September 2014.
    .. [2] http://vision.middlebury.edu/stereo/data/scenes2014/

    """
    filename = _fetch("data/motorcycle_disp.npz")
    disp = np.load(filename)['arr_0']
    return (_load("data/motorcycle_left.png"),
            _load("data/motorcycle_right.png"),
            disp)


def lfw_subset():
    """Subset of data from the LFW dataset.

    This database is a subset of the LFW database containing:

    * 100 faces
    * 100 non-faces

    The full dataset is available at [2]_.

    Returns
    -------
    images : (200, 25, 25) uint8 ndarray
        100 first images are faces and subsequent 100 are non-faces.

    Notes
    -----
    The faces were randomly selected from the LFW dataset and the non-faces
    were extracted from the background of the same dataset. The cropped ROIs
    have been resized to a 25 x 25 pixels.

    References
    ----------
    .. [1] Huang, G., Mattar, M., Lee, H., & Learned-Miller, E. G. (2012).
           Learning to align from scratch. In Advances in Neural Information
           Processing Systems (pp. 764-772).
    .. [2] http://vis-www.cs.umass.edu/lfw/

    """
    return np.load(_fetch('data/lfw_subset.npy'))


def skin():
    """Microscopy image of dermis and epidermis (skin layers).

    Hematoxylin and eosin stained slide at 10x of normal epidermis and dermis
    with a benign intradermal nevus.

    Notes
    -----
    This image requires an Internet connection the first time it is called,
    and to have the ``pooch`` package installed, in order to fetch the image
    file from the scikit-image datasets repository.

    The source of this image is
    https://en.wikipedia.org/wiki/File:Normal_Epidermis_and_Dermis_with_Intradermal_Nevus_10x.JPG

    The image was released in the public domain by its author Kilbad.

    Returns
    -------
    skin : (960, 1280, 3) RGB image of uint8
    """
    return _load('data/skin.jpg')


def brain():
    """Subset of data from the University of North Carolina Volume Rendering
    Test Data Set.

    The full dataset is available at [1]_.

    Returns
    -------
    image : (10, 256, 256) uint16 ndarray

    Notes
    -----
    The 3D volume consists of 10 layers from the larger volume.

    References
    ----------
    .. [1] https://graphics.stanford.edu/data/voldata/

    """
    return _load("data/brain.tiff")


def vortex():
    """Case B1 image pair from the first PIV challenge.

    Returns
    -------
    image0, image1 : (512, 512) grayscale images
        A pair of images featuring synthetic moving particles.

    Notes
    -----
    This image was licensed as CC0 by its author, Prof. Koji Okamoto, with
    thanks to Prof. Jun Sakakibara, who maintains the PIV Challenge site.

    References
    ----------
    .. [1] Particle Image Velocimetry (PIV) Challenge site
           http://pivchallenge.org
    .. [2] 1st PIV challenge Case B: http://pivchallenge.org/pub/index.html#b
    """
    return (_load('data/pivchallenge-B-B001_1.tif'),
            _load('data/pivchallenge-B-B001_2.tif'))
