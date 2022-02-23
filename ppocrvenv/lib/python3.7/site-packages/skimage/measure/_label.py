from scipy import ndimage
from ._ccomp import label_cython as clabel
from .._shared.utils import deprecate_kwarg


def _label_bool(image, background=None, return_num=False, connectivity=None):
    """Faster implementation of clabel for boolean input.

    See context: https://github.com/scikit-image/scikit-image/issues/4833
    """
    from ..morphology._util import _resolve_neighborhood
    if background == 1:
        image = ~image

    if connectivity is None:
        connectivity = image.ndim

    if not 1 <= connectivity <= image.ndim:
        raise ValueError(
            f'Connectivity for {image.ndim}D image should '
            f'be in [1, ..., {image.ndim}]. Got {connectivity}.'
        )

    footprint = _resolve_neighborhood(None, connectivity, image.ndim)
    result = ndimage.label(image, structure=footprint)

    if return_num:
        return result
    else:
        return result[0]


@deprecate_kwarg({'input': 'label_image'},
                 deprecated_version='0.19',
                 removed_version='1.0')
def label(label_image, background=None, return_num=False, connectivity=None):
    r"""Label connected regions of an integer array.

    Two pixels are connected when they are neighbors and have the same value.
    In 2D, they can be neighbors either in a 1- or 2-connected sense.
    The value refers to the maximum number of orthogonal hops to consider a
    pixel/voxel a neighbor::

      1-connectivity     2-connectivity     diagonal connection close-up

           [ ]           [ ]  [ ]  [ ]             [ ]
            |               \  |  /                 |  <- hop 2
      [ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
            |               /  |  \             hop 1
           [ ]           [ ]  [ ]  [ ]

    Parameters
    ----------
    label_image : ndarray of dtype int
        Image to label.
    background : int, optional
        Consider all pixels with this value as background pixels, and label
        them as 0. By default, 0-valued pixels are considered as background
        pixels.
    return_num : bool, optional
        Whether to return the number of assigned labels.
    connectivity : int, optional
        Maximum number of orthogonal hops to consider a pixel/voxel
        as a neighbor.
        Accepted values are ranging from  1 to input.ndim. If ``None``, a full
        connectivity of ``input.ndim`` is used.

    Returns
    -------
    labels : ndarray of dtype int
        Labeled array, where all connected regions are assigned the
        same integer value.
    num : int, optional
        Number of labels, which equals the maximum label index and is only
        returned if return_num is `True`.

    See Also
    --------
    regionprops
    regionprops_table

    References
    ----------
    .. [1] Christophe Fiorio and Jens Gustedt, "Two linear time Union-Find
           strategies for image processing", Theoretical Computer Science
           154 (1996), pp. 165-181.
    .. [2] Kensheng Wu, Ekow Otoo and Arie Shoshani, "Optimizing connected
           component labeling algorithms", Paper LBNL-56864, 2005,
           Lawrence Berkeley National Laboratory (University of California),
           http://repositories.cdlib.org/lbnl/LBNL-56864

    Examples
    --------
    >>> import numpy as np
    >>> x = np.eye(3).astype(int)
    >>> print(x)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    >>> print(label(x, connectivity=1))
    [[1 0 0]
     [0 2 0]
     [0 0 3]]
    >>> print(label(x, connectivity=2))
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    >>> print(label(x, background=-1))
    [[1 2 2]
     [2 1 2]
     [2 2 1]]
    >>> x = np.array([[1, 0, 0],
    ...               [1, 1, 5],
    ...               [0, 0, 0]])
    >>> print(label(x))
    [[1 0 0]
     [1 1 2]
     [0 0 0]]
    """
    if label_image.dtype == bool:
        return _label_bool(label_image, background=background,
                           return_num=return_num, connectivity=connectivity)
    else:
        return clabel(label_image, background, return_num, connectivity)
