"""
=============================
Species distribution dataset
=============================

This dataset represents the geographic distribution of species.
The dataset is provided by Phillips et. al. (2006).

The two species are:

 - `"Bradypus variegatus"
   <http://www.iucnredlist.org/details/3038/0>`_ ,
   the Brown-throated Sloth.

 - `"Microryzomys minutus"
   <http://www.iucnredlist.org/details/13408/0>`_ ,
   also known as the Forest Small Rice Rat, a rodent that lives in Peru,
   Colombia, Ecuador, Peru, and Venezuela.

References
----------

`"Maximum entropy modeling of species geographic distributions"
<http://rob.schapire.net/papers/ecolmod.pdf>`_ S. J. Phillips,
R. P. Anderson, R. E. Schapire - Ecological Modelling, 190:231-259, 2006.

Notes
-----

For an example of using this dataset, see
:ref:`examples/applications/plot_species_distribution_modeling.py
<sphx_glr_auto_examples_applications_plot_species_distribution_modeling.py>`.
"""

# Authors: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Jake Vanderplas <vanderplas@astro.washington.edu>
#
# License: BSD 3 clause

from io import BytesIO
from os import makedirs, remove
from os.path import exists

import logging
import numpy as np

import joblib

from . import get_data_home
from ._base import _fetch_remote
from ._base import RemoteFileMetadata
from ..utils import Bunch
from ._base import _pkl_filepath

# The original data can be found at:
# https://biodiversityinformatics.amnh.org/open_source/maxent/samples.zip
SAMPLES = RemoteFileMetadata(
    filename="samples.zip",
    url="https://ndownloader.figshare.com/files/5976075",
    checksum="abb07ad284ac50d9e6d20f1c4211e0fd3c098f7f85955e89d321ee8efe37ac28",
)

# The original data can be found at:
# https://biodiversityinformatics.amnh.org/open_source/maxent/coverages.zip
COVERAGES = RemoteFileMetadata(
    filename="coverages.zip",
    url="https://ndownloader.figshare.com/files/5976078",
    checksum="4d862674d72e79d6cee77e63b98651ec7926043ba7d39dcb31329cf3f6073807",
)

DATA_ARCHIVE_NAME = "species_coverage.pkz"


logger = logging.getLogger(__name__)


def _load_coverage(F, header_length=6, dtype=np.int16):
    """Load a coverage file from an open file object.

    This will return a numpy array of the given dtype
    """
    header = [F.readline() for _ in range(header_length)]
    make_tuple = lambda t: (t.split()[0], float(t.split()[1]))
    header = dict([make_tuple(line) for line in header])

    M = np.loadtxt(F, dtype=dtype)
    nodata = int(header[b"NODATA_value"])
    if nodata != -9999:
        M[nodata] = -9999
    return M


def _load_csv(F):
    """Load csv file.

    Parameters
    ----------
    F : file object
        CSV file open in byte mode.

    Returns
    -------
    rec : np.ndarray
        record array representing the data
    """
    names = F.readline().decode("ascii").strip().split(",")

    rec = np.loadtxt(F, skiprows=0, delimiter=",", dtype="a22,f4,f4")
    rec.dtype.names = names
    return rec


def construct_grids(batch):
    """Construct the map grid from the batch object

    Parameters
    ----------
    batch : Batch object
        The object returned by :func:`fetch_species_distributions`

    Returns
    -------
    (xgrid, ygrid) : 1-D arrays
        The grid corresponding to the values in batch.coverages
    """
    # x,y coordinates for corner cells
    xmin = batch.x_left_lower_corner + batch.grid_size
    xmax = xmin + (batch.Nx * batch.grid_size)
    ymin = batch.y_left_lower_corner + batch.grid_size
    ymax = ymin + (batch.Ny * batch.grid_size)

    # x coordinates of the grid cells
    xgrid = np.arange(xmin, xmax, batch.grid_size)
    # y coordinates of the grid cells
    ygrid = np.arange(ymin, ymax, batch.grid_size)

    return (xgrid, ygrid)


def fetch_species_distributions(*, data_home=None, download_if_missing=True):
    """Loader for species distribution dataset from Phillips et. al. (2006)

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : bool, default=True
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        coverages : array, shape = [14, 1592, 1212]
            These represent the 14 features measured
            at each point of the map grid.
            The latitude/longitude values for the grid are discussed below.
            Missing data is represented by the value -9999.
        train : record array, shape = (1624,)
            The training points for the data.  Each point has three fields:

            - train['species'] is the species name
            - train['dd long'] is the longitude, in degrees
            - train['dd lat'] is the latitude, in degrees
        test : record array, shape = (620,)
            The test points for the data.  Same format as the training data.
        Nx, Ny : integers
            The number of longitudes (x) and latitudes (y) in the grid
        x_left_lower_corner, y_left_lower_corner : floats
            The (x,y) position of the lower-left corner, in degrees
        grid_size : float
            The spacing between points of the grid, in degrees

    References
    ----------

    * `"Maximum entropy modeling of species geographic distributions"
      <http://rob.schapire.net/papers/ecolmod.pdf>`_
      S. J. Phillips, R. P. Anderson, R. E. Schapire - Ecological Modelling,
      190:231-259, 2006.

    Notes
    -----

    This dataset represents the geographic distribution of species.
    The dataset is provided by Phillips et. al. (2006).

    The two species are:

    - `"Bradypus variegatus"
      <http://www.iucnredlist.org/details/3038/0>`_ ,
      the Brown-throated Sloth.

    - `"Microryzomys minutus"
      <http://www.iucnredlist.org/details/13408/0>`_ ,
      also known as the Forest Small Rice Rat, a rodent that lives in Peru,
      Colombia, Ecuador, Peru, and Venezuela.

    - For an example of using this dataset with scikit-learn, see
      :ref:`examples/applications/plot_species_distribution_modeling.py
      <sphx_glr_auto_examples_applications_plot_species_distribution_modeling.py>`.
    """
    data_home = get_data_home(data_home)
    if not exists(data_home):
        makedirs(data_home)

    # Define parameters for the data files.  These should not be changed
    # unless the data model changes.  They will be saved in the npz file
    # with the downloaded data.
    extra_params = dict(
        x_left_lower_corner=-94.8,
        Nx=1212,
        y_left_lower_corner=-56.05,
        Ny=1592,
        grid_size=0.05,
    )
    dtype = np.int16

    archive_path = _pkl_filepath(data_home, DATA_ARCHIVE_NAME)

    if not exists(archive_path):
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")
        logger.info("Downloading species data from %s to %s" % (SAMPLES.url, data_home))
        samples_path = _fetch_remote(SAMPLES, dirname=data_home)
        with np.load(samples_path) as X:  # samples.zip is a valid npz
            for f in X.files:
                fhandle = BytesIO(X[f])
                if "train" in f:
                    train = _load_csv(fhandle)
                if "test" in f:
                    test = _load_csv(fhandle)
        remove(samples_path)

        logger.info(
            "Downloading coverage data from %s to %s" % (COVERAGES.url, data_home)
        )
        coverages_path = _fetch_remote(COVERAGES, dirname=data_home)
        with np.load(coverages_path) as X:  # coverages.zip is a valid npz
            coverages = []
            for f in X.files:
                fhandle = BytesIO(X[f])
                logger.debug(" - converting {}".format(f))
                coverages.append(_load_coverage(fhandle))
            coverages = np.asarray(coverages, dtype=dtype)
        remove(coverages_path)

        bunch = Bunch(coverages=coverages, test=test, train=train, **extra_params)
        joblib.dump(bunch, archive_path, compress=9)
    else:
        bunch = joblib.load(archive_path)

    return bunch
