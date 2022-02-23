import warnings

from .footprints import (
    square, rectangle, diamond, disk, cube, octahedron, ball, octagon, star
)

warnings.warn(
    "The `skimage.morphology.selem` module is deprecated and will be removed "
    "in scikit-image 1.0 (`skimage.morphology.selem` has been moved to "
    "`skimage.morphology.footprints`).",
    FutureWarning, stacklevel=2
)
