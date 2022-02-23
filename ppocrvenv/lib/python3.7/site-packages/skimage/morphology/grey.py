import warnings

from .gray import (black_tophat, closing, dilation, erosion, opening,
                   white_tophat)

__all__ = ['erosion', 'dilation', 'opening', 'closing', 'white_tophat',
           'black_tophat']


warnings.warn(
    "Importing from skimage.morphology.grey is deprecated. "
    "Please import from skimage.morphology instead.",
    FutureWarning, stacklevel=2
)
