import warnings

from .grayreconstruct import reconstruction

warnings.warn(
    "Importing from skimage.morphology.greyreconstruct is deprecated. "
    "Please import from skimage.morphology instead.",
    FutureWarning, stacklevel=2
)
