"""Image Processing for Python

``scikit-image`` (a.k.a. ``skimage``) is a collection of algorithms for image
processing and computer vision.

The main package of ``skimage`` only provides a few utilities for converting
between image data types; for most features, you need to import one of the
following subpackages:

Subpackages
-----------
color
    Color space conversion.
data
    Test images and example data.
draw
    Drawing primitives (lines, text, etc.) that operate on NumPy arrays.
exposure
    Image intensity adjustment, e.g., histogram equalization, etc.
feature
    Feature detection and extraction, e.g., texture analysis corners, etc.
filters
    Sharpening, edge finding, rank filters, thresholding, etc.
graph
    Graph-theoretic operations, e.g., shortest paths.
io
    Reading, saving, and displaying images and video.
measure
    Measurement of image properties, e.g., region properties and contours.
metrics
    Metrics corresponding to images, e.g. distance metrics, similarity, etc.
morphology
    Morphological operations, e.g., opening or skeletonization.
restoration
    Restoration algorithms, e.g., deconvolution algorithms, denoising, etc.
segmentation
    Partitioning an image into multiple regions.
transform
    Geometric and other transforms, e.g., rotation or the Radon transform.
util
    Generic utilities.
viewer
    A simple graphical user interface for visualizing results and exploring
    parameters.

Utility Functions
-----------------
img_as_float
    Convert an image to floating point format, with values in [0, 1].
    Is similar to `img_as_float64`, but will not convert lower-precision
    floating point arrays to `float64`.
img_as_float32
    Convert an image to single-precision (32-bit) floating point format,
    with values in [0, 1].
img_as_float64
    Convert an image to double-precision (64-bit) floating point format,
    with values in [0, 1].
img_as_uint
    Convert an image to unsigned integer format, with values in [0, 65535].
img_as_int
    Convert an image to signed integer format, with values in [-32768, 32767].
img_as_ubyte
    Convert an image to unsigned byte format, with values in [0, 255].
img_as_bool
    Convert an image to boolean format, with values either True or False.
dtype_limits
    Return intensity limits, i.e. (min, max) tuple, of the image's dtype.

"""

__version__ = '0.19.1'

submodules = [
    'color',
    'data',
    'draw',
    'exposure',
    'feature',
    'filters',
    'future',
    'graph',
    'io',
    'measure',
    'metrics',
    'morphology',
    'registration',
    'restoration',
    'segmentation',
    'transform',
    'util',
    'viewer'
]

from ._shared.version_requirements import ensure_python_version
ensure_python_version((3, 7))

from ._shared import lazy
__getattr__, __lazy_dir__, _ = lazy.attach(
    __name__,
    submodules,
    submod_attrs={'data': ['data_dir']}
)


def __dir__():
    return __lazy_dir__() + ['__version__']

# Logic for checking for improper install and importing while in the source
# tree when package has not been installed inplace.
# Code adapted from scikit-learn's __check_build module.
_INPLACE_MSG = """
It appears that you are importing a local scikit-image source tree. For
this, you need to have an inplace install. Maybe you are in the source
directory and you need to try from another location."""

_STANDARD_MSG = """
Your install of scikit-image appears to be broken.
Try re-installing the package following the instructions at:
https://scikit-image.org/docs/stable/install.html """


def _raise_build_error(e):
    # Raise a comprehensible error
    import os.path as osp
    local_dir = osp.split(__file__)[0]
    msg = _STANDARD_MSG
    if local_dir == "skimage":
        # Picking up the local install: this will work only if the
        # install is an 'inplace build'
        msg = _INPLACE_MSG
    raise ImportError("""%s
It seems that scikit-image has not been built correctly.
%s""" % (e, msg))


try:
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of skimage when
    # the binaries are not built
    __SKIMAGE_SETUP__
except NameError:
    __SKIMAGE_SETUP__ = False

if __SKIMAGE_SETUP__:
    import sys
    sys.stderr.write('Partial import of skimage during the build process.\n')
    # We are not importing the rest of the scikit during the build
    # process, as it may not be compiled yet
else:
    try:
        from ._shared import geometry
        del geometry
    except ImportError as e:
        _raise_build_error(e)

    # All skimage root imports go here
    from .util.dtype import (img_as_float32,
                             img_as_float64,
                             img_as_float,
                             img_as_int,
                             img_as_uint,
                             img_as_ubyte,
                             img_as_bool,
                             dtype_limits)
    from .util.lookfor import lookfor

if 'dev' in __version__:
    # Append last commit date and hash to dev version information, if available

    import subprocess
    import os.path

    try:
        p = subprocess.Popen(
            ['git', 'log', '-1', '--format="%h %aI"'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(__file__),
        )
    except FileNotFoundError:
        pass
    else:
        out, err = p.communicate()
        if p.returncode == 0:
            git_hash, git_date = (
                out.decode('utf-8')
                .strip()
                .replace('"', '')
                .split('T')[0]
                .replace('-', '')
                .split()
            )

            __version__ = '+'.join(
                [tag for tag in __version__.split('+')
                 if not tag.startswith('git')]
            )
            __version__ += f'+git{git_date}.{git_hash}'

from skimage._shared.tester import PytestTester  # noqa
test = PytestTester(__name__)
del PytestTester
