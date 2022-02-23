import inspect
from functools import wraps
from math import atan2
from math import pi as PI
from math import sqrt
from warnings import warn

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist

from . import _moments
from ._find_contours import find_contours
from ._marching_cubes_lewiner import marching_cubes
from ._regionprops_utils import euler_number, perimeter, perimeter_crofton

__all__ = ['regionprops', 'euler_number', 'perimeter', 'perimeter_crofton']


# All values in this PROPS dict correspond to current scikit-image property
# names. The keys in this PROPS dict correspond to older names used in prior
# releases. For backwards compatibility, these older names will continue to
# work, but will not be documented.
PROPS = {
    'Area': 'area',
    'BoundingBox': 'bbox',
    'BoundingBoxArea': 'area_bbox',
    'bbox_area': 'area_bbox',
    'CentralMoments': 'moments_central',
    'Centroid': 'centroid',
    'ConvexArea': 'area_convex',
    'convex_area': 'area_convex',
    # 'ConvexHull',
    'ConvexImage': 'image_convex',
    'convex_image': 'image_convex',
    'Coordinates': 'coords',
    'Eccentricity': 'eccentricity',
    'EquivDiameter': 'equivalent_diameter_area',
    'equivalent_diameter': 'equivalent_diameter_area',
    'EulerNumber': 'euler_number',
    'Extent': 'extent',
    # 'Extrema',
    'FeretDiameter': 'feret_diameter_max',
    'FeretDiameterMax': 'feret_diameter_max',
    'FilledArea': 'area_filled',
    'filled_area': 'area_filled',
    'FilledImage': 'image_filled',
    'filled_image': 'image_filled',
    'HuMoments': 'moments_hu',
    'Image': 'image',
    'InertiaTensor': 'inertia_tensor',
    'InertiaTensorEigvals': 'inertia_tensor_eigvals',
    'IntensityImage': 'image_intensity',
    'intensity_image': 'image_intensity',
    'Label': 'label',
    'LocalCentroid': 'centroid_local',
    'local_centroid': 'centroid_local',
    'MajorAxisLength': 'axis_major_length',
    'major_axis_length': 'axis_major_length',
    'MaxIntensity': 'intensity_max',
    'max_intensity': 'intensity_max',
    'MeanIntensity': 'intensity_mean',
    'mean_intensity': 'intensity_mean',
    'MinIntensity': 'intensity_min',
    'min_intensity': 'intensity_min',
    'MinorAxisLength': 'axis_minor_length',
    'minor_axis_length': 'axis_minor_length',
    'Moments': 'moments',
    'NormalizedMoments': 'moments_normalized',
    'Orientation': 'orientation',
    'Perimeter': 'perimeter',
    'CroftonPerimeter': 'perimeter_crofton',
    # 'PixelIdxList',
    # 'PixelList',
    'Slice': 'slice',
    'Solidity': 'solidity',
    # 'SubarrayIdx'
    'WeightedCentralMoments': 'moments_weighted_central',
    'weighted_moments_central': 'moments_weighted_central',
    'WeightedCentroid': 'centroid_weighted',
    'weighted_centroid': 'centroid_weighted',
    'WeightedHuMoments': 'moments_weighted_hu',
    'weighted_moments_hu': 'moments_weighted_hu',
    'WeightedLocalCentroid': 'centroid_weighted_local',
    'weighted_local_centroid': 'centroid_weighted_local',
    'WeightedMoments': 'moments_weighted',
    'weighted_moments': 'moments_weighted',
    'WeightedNormalizedMoments': 'moments_weighted_normalized',
    'weighted_moments_normalized': 'moments_weighted_normalized',
}

OBJECT_COLUMNS = {
    'image', 'coords', 'image_convex', 'slice',
    'image_filled', 'image_intensity'
}

COL_DTYPES = {
    'area': int,
    'area_bbox': int,
    'area_convex': int,
    'area_filled': int,
    'axis_major_length': float,
    'axis_minor_length': float,
    'bbox': int,
    'centroid': float,
    'centroid_local': float,
    'centroid_weighted': float,
    'centroid_weighted_local': float,
    'coords': object,
    'eccentricity': float,
    'equivalent_diameter_area': float,
    'euler_number': int,
    'extent': float,
    'feret_diameter_max': float,
    'image': object,
    'image_convex': object,
    'image_filled': object,
    'image_intensity': object,
    'inertia_tensor': float,
    'inertia_tensor_eigvals': float,
    'intensity_max': float,
    'intensity_mean': float,
    'intensity_min': float,
    'label': int,
    'moments': float,
    'moments_central': float,
    'moments_hu': float,
    'moments_normalized': float,
    'moments_weighted': float,
    'moments_weighted_central': float,
    'moments_weighted_hu': float,
    'moments_weighted_normalized': float,
    'orientation': float,
    'perimeter': float,
    'perimeter_crofton': float,
    'slice': object,
    'solidity': float,
}

PROP_VALS = set(PROPS.values())


def _infer_number_of_required_args(func):
    """Infer the number of required arguments for a function

    Parameters
    ----------
    func : callable
        The function that is being inspected.

    Returns
    -------
    n_args : int
        The number of required arguments of func.
    """
    argspec = inspect.getfullargspec(func)
    n_args = len(argspec.args)
    if argspec.defaults is not None:
        n_args -= len(argspec.defaults)
    return n_args


def _infer_regionprop_dtype(func, *, intensity, ndim):
    """Infer the dtype of a region property calculated by func.

    If a region property function always returns the same shape and type of
    output regardless of input size, then the dtype is the dtype of the
    returned array. Otherwise, the property has object dtype.

    Parameters
    ----------
    func : callable
        Function to be tested. The signature should be array[bool] -> Any if
        intensity is False, or *(array[bool], array[float]) -> Any otherwise.
    intensity : bool
        Whether the regionprop is calculated on an intensity image.
    ndim : int
        The number of dimensions for which to check func.

    Returns
    -------
    dtype : NumPy data type
        The data type of the returned property.
    """
    labels = [1, 2]
    sample = np.zeros((3,) * ndim, dtype=np.intp)
    sample[(0,) * ndim] = labels[0]
    sample[(slice(1, None),) * ndim] = labels[1]
    propmasks = [(sample == n) for n in labels]

    rng = np.random.default_rng()

    if intensity and _infer_number_of_required_args(func) == 2:
        def _func(mask):
            return func(mask, rng.random(sample.shape))
    else:
        _func = func
    props1, props2 = map(_func, propmasks)
    if (np.isscalar(props1) and np.isscalar(props2)
            or np.array(props1).shape == np.array(props2).shape):
        dtype = np.array(props1).dtype.type
    else:
        dtype = np.object_
    return dtype


def _cached(f):
    @wraps(f)
    def wrapper(obj):
        cache = obj._cache
        prop = f.__name__

        if not ((prop in cache) and obj._cache_active):
            cache[prop] = f(obj)

        return cache[prop]

    return wrapper


def only2d(method):
    @wraps(method)
    def func2d(self, *args, **kwargs):
        if self._ndim > 2:
            raise NotImplementedError('Property %s is not implemented for '
                                      '3D images' % method.__name__)
        return method(self, *args, **kwargs)
    return func2d


def _inertia_eigvals_to_axes_lengths_3D(inertia_tensor_eigvals):
    """Compute ellipsoid axis lengths from inertia tensor eigenvalues.

    Paramters
    ---------
    inertia_tensor_eigvals : seqeunce of float
        A sequence of 3 floating point eigenvalues, sorted in descending order.

    Returns
    -------
    axis_lengths : list of float
        The ellipsoid axis lengths sorted in descending order.

    Notes
    -----
    Let a >= b >= c be the ellipsoid semi-axes and s1 >= s2 >= s3 be the
    inertia tensor eigenvalues.

    The inertia tensor eigenvalues are given for a solid ellipsoid in [1]_.
    s1 = 1 / 5 * (a**2 + b**2)
    s2 = 1 / 5 * (a**2 + c**2)
    s3 = 1 / 5 * (b**2 + c**2)

    Rearranging to solve for a, b, c in terms of s1, s2, s3 gives
    a = math.sqrt(5 / 2 * ( s1 + s2 - s3))
    b = math.sqrt(5 / 2 * ( s1 - s2 + s3))
    c = math.sqrt(5 / 2 * (-s1 + s2 + s3))

    We can then simply replace sqrt(5/2) by sqrt(10) to get the full axes
    lengths rather than the semi-axes lengths.

    References
    ----------
    ..[1] https://en.wikipedia.org/wiki/List_of_moments_of_inertia#List_of_3D_inertia_tensors  # noqa
    """
    axis_lengths = []
    for ax in range(2, -1, -1):
        w = sum(v * -1 if i == ax else v
                for i, v in enumerate(inertia_tensor_eigvals))
        axis_lengths.append(sqrt(10 * w))
    return axis_lengths


class RegionProperties:
    """Please refer to `skimage.measure.regionprops` for more information
    on the available region properties.
    """

    def __init__(self, slice, label, label_image, intensity_image,
                 cache_active, *, extra_properties=None):

        if intensity_image is not None:
            ndim = label_image.ndim
            if not (
                    intensity_image.shape[:ndim] == label_image.shape
                    and intensity_image.ndim in [ndim, ndim + 1]
                    ):
                raise ValueError('Label and intensity image shapes must match,'
                                 ' except for channel (last) axis.')
            multichannel = label_image.shape < intensity_image.shape
        else:
            multichannel = False

        self.label = label

        self._slice = slice
        self.slice = slice
        self._label_image = label_image
        self._intensity_image = intensity_image

        self._cache_active = cache_active
        self._cache = {}
        self._ndim = label_image.ndim
        self._multichannel = multichannel
        self._spatial_axes = tuple(range(self._ndim))

        self._extra_properties = {}
        if extra_properties is not None:
            for func in extra_properties:
                name = func.__name__
                if hasattr(self, name):
                    msg = (
                        f"Extra property '{name}' is shadowed by existing "
                        f"property and will be inaccessible. Consider "
                        f"renaming it."
                    )
                    warn(msg)
            self._extra_properties = {
                func.__name__: func for func in extra_properties
            }

    def __getattr__(self, attr):
        if attr in self._extra_properties:
            func = self._extra_properties[attr]
            n_args = _infer_number_of_required_args(func)
            # determine whether func requires intensity image
            if n_args == 2:
                if self._intensity_image is not None:
                    if self._multichannel:
                        multichannel_list = [func(self.image,
                                                  self.image_intensity[..., i])
                                             for i in range(
                            self.image_intensity.shape[-1])]
                        return np.stack(multichannel_list, axis=-1)
                    else:
                        return func(self.image, self.image_intensity)
                else:
                    raise AttributeError(
                        f'intensity image required to calculate {attr}'
                    )
            elif n_args == 1:
                return func(self.image)
            else:
                raise AttributeError(
                    f'Custom regionprop function\'s number of arguments must '
                    f'be 1 or 2, but {attr} takes {n_args} arguments.'
                )
        elif attr in PROPS and attr.lower() == attr:
            # retrieve deprecated property (excluding old CamelCase ones)
            return getattr(self, PROPS[attr])
        else:
            raise AttributeError(
                f"'{type(self)}' object has no attribute '{attr}'"
            )

    def __setattr__(self, name, value):
        if name in PROPS:
            super().__setattr__(PROPS[name], value)
        else:
            super().__setattr__(name, value)

    @property
    @_cached
    def area(self):
        return np.sum(self.image)

    @property
    def bbox(self):
        """
        Returns
        -------
        A tuple of the bounding box's start coordinates for each dimension,
        followed by the end coordinates for each dimension
        """
        return tuple([self.slice[i].start for i in range(self._ndim)] +
                     [self.slice[i].stop for i in range(self._ndim)])

    @property
    def area_bbox(self):
        return self.image.size

    @property
    def centroid(self):
        return tuple(self.coords.mean(axis=0))

    @property
    @_cached
    def area_convex(self):
        return np.sum(self.image_convex)

    @property
    @_cached
    def image_convex(self):
        from ..morphology.convex_hull import convex_hull_image
        return convex_hull_image(self.image)

    @property
    def coords(self):
        indices = np.nonzero(self.image)
        return np.vstack([indices[i] + self.slice[i].start
                          for i in range(self._ndim)]).T

    @property
    @only2d
    def eccentricity(self):
        l1, l2 = self.inertia_tensor_eigvals
        if l1 == 0:
            return 0
        return sqrt(1 - l2 / l1)

    @property
    def equivalent_diameter_area(self):
        return (2 * self._ndim * self.area / PI) ** (1 / self._ndim)

    @property
    def euler_number(self):
        if self._ndim not in [2, 3]:
            raise NotImplementedError('Euler number is implemented for '
                                      '2D or 3D images only')
        return euler_number(self.image, self._ndim)

    @property
    def extent(self):
        return self.area / self.image.size

    @property
    def feret_diameter_max(self):
        identity_convex_hull = np.pad(self.image_convex,
                                      2, mode='constant', constant_values=0)
        if self._ndim == 2:
            coordinates = np.vstack(find_contours(identity_convex_hull, .5,
                                                  fully_connected='high'))
        elif self._ndim == 3:
            coordinates, _, _, _ = marching_cubes(identity_convex_hull,
                                                  level=.5)
        distances = pdist(coordinates, 'sqeuclidean')
        return sqrt(np.max(distances))

    @property
    def area_filled(self):
        return np.sum(self.image_filled)

    @property
    @_cached
    def image_filled(self):
        structure = np.ones((3,) * self._ndim)
        return ndi.binary_fill_holes(self.image, structure)

    @property
    @_cached
    def image(self):
        return self._label_image[self.slice] == self.label

    @property
    @_cached
    def inertia_tensor(self):
        mu = self.moments_central
        return _moments.inertia_tensor(self.image, mu)

    @property
    @_cached
    def inertia_tensor_eigvals(self):
        return _moments.inertia_tensor_eigvals(self.image,
                                               T=self.inertia_tensor)

    @property
    @_cached
    def image_intensity(self):
        if self._intensity_image is None:
            raise AttributeError('No intensity image specified.')
        image = (
            self.image
            if not self._multichannel
            else np.expand_dims(self.image, self._ndim)
        )
        return self._intensity_image[self.slice] * image

    def _image_intensity_double(self):
        return self.image_intensity.astype(np.double, copy=False)

    @property
    def centroid_local(self):
        M = self.moments
        M0 = M[(0,) * self._ndim]

        def _get_element(axis):
            return (0,) * axis + (1,) + (0,) * (self._ndim - 1 - axis)

        return np.asarray(
            tuple(M[_get_element(axis)] / M0 for axis in range(self._ndim)))

    @property
    def intensity_max(self):
        vals = self.image_intensity[self.image]
        return np.max(vals, axis=0).astype(np.double, copy=False)

    @property
    def intensity_mean(self):
        return np.mean(self.image_intensity[self.image], axis=0)

    @property
    def intensity_min(self):
        vals = self.image_intensity[self.image]
        return np.min(vals, axis=0).astype(np.double, copy=False)

    @property
    def axis_major_length(self):
        if self._ndim == 2:
            l1 = self.inertia_tensor_eigvals[0]
            return 4 * sqrt(l1)
        elif self._ndim == 3:
            # equivalent to _inertia_eigvals_to_axes_lengths_3D(ev)[0]
            ev = self.inertia_tensor_eigvals
            return sqrt(10 * (ev[0] + ev[1] - ev[2]))
        else:
            raise ValueError("axis_major_length only available in 2D and 3D")

    @property
    def axis_minor_length(self):
        if self._ndim == 2:
            l2 = self.inertia_tensor_eigvals[-1]
            return 4 * sqrt(l2)
        elif self._ndim == 3:
            # equivalent to _inertia_eigvals_to_axes_lengths_3D(ev)[-1]
            ev = self.inertia_tensor_eigvals
            return sqrt(10 * (-ev[0] + ev[1] + ev[2]))
        else:
            raise ValueError("axis_minor_length only available in 2D and 3D")

    @property
    @_cached
    def moments(self):
        M = _moments.moments(self.image.astype(np.uint8), 3)
        return M

    @property
    @_cached
    def moments_central(self):
        mu = _moments.moments_central(self.image.astype(np.uint8),
                                      self.centroid_local, order=3)
        return mu

    @property
    @only2d
    def moments_hu(self):
        return _moments.moments_hu(self.moments_normalized)

    @property
    @_cached
    def moments_normalized(self):
        return _moments.moments_normalized(self.moments_central, 3)

    @property
    @only2d
    def orientation(self):
        a, b, b, c = self.inertia_tensor.flat
        if a - c == 0:
            if b < 0:
                return -PI / 4.
            else:
                return PI / 4.
        else:
            return 0.5 * atan2(-2 * b, c - a)

    @property
    @only2d
    def perimeter(self):
        return perimeter(self.image, 4)

    @property
    @only2d
    def perimeter_crofton(self):
        return perimeter_crofton(self.image, 4)

    @property
    def solidity(self):
        return self.area / self.area_convex

    @property
    def centroid_weighted(self):
        ctr = self.centroid_weighted_local
        return tuple(idx + slc.start
                     for idx, slc in zip(ctr, self.slice))

    @property
    def centroid_weighted_local(self):
        M = self.moments_weighted
        M0 = M[(0,) * self._ndim]

        def _get_element(axis):
            return (0,) * axis + (1,) + (0,) * (self._ndim - 1 - axis)

        return np.asarray(
            tuple(M[_get_element(axis)] / M0 for axis in range(self._ndim)))

    @property
    @_cached
    def moments_weighted(self):
        image = self._image_intensity_double()
        if self._multichannel:
            moments = np.stack(
                    [_moments.moments(image[..., i], order=3)
                        for i in range(image.shape[-1])],
                    axis=-1,
                    )
        else:
            moments = _moments.moments(image, order=3)
        return moments

    @property
    @_cached
    def moments_weighted_central(self):
        ctr = self.centroid_weighted_local
        image = self._image_intensity_double()
        if self._multichannel:
            moments_list = [
                _moments.moments_central(
                    image[..., i], center=ctr[..., i], order=3
                )
                for i in range(image.shape[-1])
            ]
            moments = np.stack(moments_list, axis=-1)
        else:
            moments = _moments.moments_central(image, ctr, order=3)
        return moments

    @property
    @only2d
    def moments_weighted_hu(self):
        nu = self.moments_weighted_normalized
        if self._multichannel:
            nchannels = self._intensity_image.shape[-1]
            return np.stack(
                [_moments.moments_hu(nu[..., i]) for i in range(nchannels)],
                axis=-1,
            )
        else:
            return _moments.moments_hu(nu)

    @property
    @_cached
    def moments_weighted_normalized(self):
        mu = self.moments_weighted_central
        if self._multichannel:
            nchannels = self._intensity_image.shape[-1]
            return np.stack(
                [_moments.moments_normalized(mu[..., i], order=3)
                 for i in range(nchannels)],
                axis=-1,
            )
        else:
            return _moments.moments_normalized(mu, order=3)
        return _moments.moments_normalized(self.moments_weighted_central, 3)

    def __iter__(self):
        props = PROP_VALS

        if self._intensity_image is None:
            unavailable_props = ('image_intensity',
                                 'intensity_max',
                                 'intensity_mean',
                                 'intensity_min',
                                 'moments_weighted',
                                 'moments_weighted_central',
                                 'centroid_weighted',
                                 'centroid_weighted_local',
                                 'moments_weighted_hu',
                                 'moments_weighted_normalized')

            props = props.difference(unavailable_props)

        return iter(sorted(props))

    def __getitem__(self, key):
        value = getattr(self, key, None)
        if value is not None:
            return value
        else:  # backwards compatibility
            return getattr(self, PROPS[key])

    def __eq__(self, other):
        if not isinstance(other, RegionProperties):
            return False

        for key in PROP_VALS:
            try:
                # so that NaNs are equal
                np.testing.assert_equal(getattr(self, key, None),
                                        getattr(other, key, None))
            except AssertionError:
                return False

        return True


# For compatibility with code written prior to 0.16
_RegionProperties = RegionProperties


def _props_to_dict(regions, properties=('label', 'bbox'), separator='-'):
    """Convert image region properties list into a column dictionary.

    Parameters
    ----------
    regions : (N,) list
        List of RegionProperties objects as returned by :func:`regionprops`.
    properties : tuple or list of str, optional
        Properties that will be included in the resulting dictionary
        For a list of available properties, please see :func:`regionprops`.
        Users should remember to add "label" to keep track of region
        identities.
    separator : str, optional
        For non-scalar properties not listed in OBJECT_COLUMNS, each element
        will appear in its own column, with the index of that element separated
        from the property name by this separator. For example, the inertia
        tensor of a 2D region will appear in four columns:
        ``inertia_tensor-0-0``, ``inertia_tensor-0-1``, ``inertia_tensor-1-0``,
        and ``inertia_tensor-1-1`` (where the separator is ``-``).

        Object columns are those that cannot be split in this way because the
        number of columns would change depending on the object. For example,
        ``image`` and ``coords``.

    Returns
    -------
    out_dict : dict
        Dictionary mapping property names to an array of values of that
        property, one value per region. This dictionary can be used as input to
        pandas ``DataFrame`` to map property names to columns in the frame and
        regions to rows.

    Notes
    -----
    Each column contains either a scalar property, an object property, or an
    element in a multidimensional array.

    Properties with scalar values for each region, such as "eccentricity", will
    appear as a float or int array with that property name as key.

    Multidimensional properties *of fixed size* for a given image dimension,
    such as "centroid" (every centroid will have three elements in a 3D image,
    no matter the region size), will be split into that many columns, with the
    name {property_name}{separator}{element_num} (for 1D properties),
    {property_name}{separator}{elem_num0}{separator}{elem_num1} (for 2D
    properties), and so on.

    For multidimensional properties that don't have a fixed size, such as
    "image" (the image of a region varies in size depending on the region
    size), an object array will be used, with the corresponding property name
    as the key.

    Examples
    --------
    >>> from skimage import data, util, measure
    >>> image = data.coins()
    >>> label_image = measure.label(image > 110, connectivity=image.ndim)
    >>> proplist = regionprops(label_image, image)
    >>> props = _props_to_dict(proplist, properties=['label', 'inertia_tensor',
    ...                                              'inertia_tensor_eigvals'])
    >>> props  # doctest: +ELLIPSIS +SKIP
    {'label': array([ 1,  2, ...]), ...
     'inertia_tensor-0-0': array([  4.012...e+03,   8.51..., ...]), ...
     ...,
     'inertia_tensor_eigvals-1': array([  2.67...e+02,   2.83..., ...])}

    The resulting dictionary can be directly passed to pandas, if installed, to
    obtain a clean DataFrame:

    >>> import pandas as pd  # doctest: +SKIP
    >>> data = pd.DataFrame(props)  # doctest: +SKIP
    >>> data.head()  # doctest: +SKIP
       label  inertia_tensor-0-0  ...  inertia_tensor_eigvals-1
    0      1         4012.909888  ...                267.065503
    1      2            8.514739  ...                  2.834806
    2      3            0.666667  ...                  0.000000
    3      4            0.000000  ...                  0.000000
    4      5            0.222222  ...                  0.111111

    """

    out = {}
    n = len(regions)
    for prop in properties:
        r = regions[0]
        # Copy the original property name so the output will have the
        # user-provided property name in the case of deprecated names.
        orig_prop = prop
        # determine the current property name for any deprecated property.
        prop = PROPS.get(prop, prop)
        rp = getattr(r, prop)
        if prop in COL_DTYPES:
            dtype = COL_DTYPES[prop]
        else:
            func = r._extra_properties[prop]
            dtype = _infer_regionprop_dtype(
                func,
                intensity=r._intensity_image is not None,
                ndim=r.image.ndim,
            )

        # scalars and objects are dedicated one column per prop
        # array properties are raveled into multiple columns
        # for more info, refer to notes 1
        if np.isscalar(rp) or prop in OBJECT_COLUMNS or dtype is np.object_:
            column_buffer = np.empty(n, dtype=dtype)
            for i in range(n):
                column_buffer[i] = regions[i][prop]
            out[orig_prop] = np.copy(column_buffer)
        else:
            if isinstance(rp, np.ndarray):
                shape = rp.shape
            else:
                shape = (len(rp),)

            # precompute property column names and locations
            modified_props = []
            locs = []
            for ind in np.ndindex(shape):
                modified_props.append(
                    separator.join(map(str, (orig_prop,) + ind))
                )
                locs.append(ind if len(ind) > 1 else ind[0])

            # fill temporary column data_array
            n_columns = len(locs)
            column_data = np.empty((n, n_columns), dtype=dtype)
            for k in range(n):
                rp = regions[k][prop]
                for i, loc in enumerate(locs):
                    column_data[k, i] = rp[loc]

            # add the columns to the output dictionary
            for i, modified_prop in enumerate(modified_props):
                out[modified_prop] = column_data[:, i]
    return out


def regionprops_table(label_image, intensity_image=None,
                      properties=('label', 'bbox'),
                      *,
                      cache=True, separator='-', extra_properties=None):
    """Compute image properties and return them as a pandas-compatible table.

    The table is a dictionary mapping column names to value arrays. See Notes
    section below for details.

    .. versionadded:: 0.16

    Parameters
    ----------
    label_image : (N, M[, P]) ndarray
        Labeled input image. Labels with value 0 are ignored.
    intensity_image : (M, N[, P][, C]) ndarray, optional
        Intensity (i.e., input) image with same size as labeled image, plus
        optionally an extra dimension for multichannel data. Currently,
        this extra channel dimension, if present, must be the last axis.
        Default is None.

        .. versionchanged:: 0.18.0
            The ability to provide an extra dimension for channels was added.
    properties : tuple or list of str, optional
        Properties that will be included in the resulting dictionary
        For a list of available properties, please see :func:`regionprops`.
        Users should remember to add "label" to keep track of region
        identities.
    cache : bool, optional
        Determine whether to cache calculated properties. The computation is
        much faster for cached properties, whereas the memory consumption
        increases.
    separator : str, optional
        For non-scalar properties not listed in OBJECT_COLUMNS, each element
        will appear in its own column, with the index of that element separated
        from the property name by this separator. For example, the inertia
        tensor of a 2D region will appear in four columns:
        ``inertia_tensor-0-0``, ``inertia_tensor-0-1``, ``inertia_tensor-1-0``,
        and ``inertia_tensor-1-1`` (where the separator is ``-``).

        Object columns are those that cannot be split in this way because the
        number of columns would change depending on the object. For example,
        ``image`` and ``coords``.
    extra_properties : Iterable of callables
        Add extra property computation functions that are not included with
        skimage. The name of the property is derived from the function name,
        the dtype is inferred by calling the function on a small sample.
        If the name of an extra property clashes with the name of an existing
        property the extra property wil not be visible and a UserWarning is
        issued. A property computation function must take a region mask as its
        first argument. If the property requires an intensity image, it must
        accept the intensity image as the second argument.

    Returns
    -------
    out_dict : dict
        Dictionary mapping property names to an array of values of that
        property, one value per region. This dictionary can be used as input to
        pandas ``DataFrame`` to map property names to columns in the frame and
        regions to rows. If the image has no regions,
        the arrays will have length 0, but the correct type.

    Notes
    -----
    Each column contains either a scalar property, an object property, or an
    element in a multidimensional array.

    Properties with scalar values for each region, such as "eccentricity", will
    appear as a float or int array with that property name as key.

    Multidimensional properties *of fixed size* for a given image dimension,
    such as "centroid" (every centroid will have three elements in a 3D image,
    no matter the region size), will be split into that many columns, with the
    name {property_name}{separator}{element_num} (for 1D properties),
    {property_name}{separator}{elem_num0}{separator}{elem_num1} (for 2D
    properties), and so on.

    For multidimensional properties that don't have a fixed size, such as
    "image" (the image of a region varies in size depending on the region
    size), an object array will be used, with the corresponding property name
    as the key.

    Examples
    --------
    >>> from skimage import data, util, measure
    >>> image = data.coins()
    >>> label_image = measure.label(image > 110, connectivity=image.ndim)
    >>> props = measure.regionprops_table(label_image, image,
    ...                           properties=['label', 'inertia_tensor',
    ...                                       'inertia_tensor_eigvals'])
    >>> props  # doctest: +ELLIPSIS +SKIP
    {'label': array([ 1,  2, ...]), ...
     'inertia_tensor-0-0': array([  4.012...e+03,   8.51..., ...]), ...
     ...,
     'inertia_tensor_eigvals-1': array([  2.67...e+02,   2.83..., ...])}

    The resulting dictionary can be directly passed to pandas, if installed, to
    obtain a clean DataFrame:

    >>> import pandas as pd  # doctest: +SKIP
    >>> data = pd.DataFrame(props)  # doctest: +SKIP
    >>> data.head()  # doctest: +SKIP
       label  inertia_tensor-0-0  ...  inertia_tensor_eigvals-1
    0      1         4012.909888  ...                267.065503
    1      2            8.514739  ...                  2.834806
    2      3            0.666667  ...                  0.000000
    3      4            0.000000  ...                  0.000000
    4      5            0.222222  ...                  0.111111

    [5 rows x 7 columns]

    If we want to measure a feature that does not come as a built-in
    property, we can define custom functions and pass them as
    ``extra_properties``. For example, we can create a custom function
    that measures the intensity quartiles in a region:

    >>> from skimage import data, util, measure
    >>> import numpy as np
    >>> def quartiles(regionmask, intensity):
    ...     return np.percentile(intensity[regionmask], q=(25, 50, 75))
    >>>
    >>> image = data.coins()
    >>> label_image = measure.label(image > 110, connectivity=image.ndim)
    >>> props = measure.regionprops_table(label_image, intensity_image=image,
    ...                                   properties=('label',),
    ...                                   extra_properties=(quartiles,))
    >>> import pandas as pd # doctest: +SKIP
    >>> pd.DataFrame(props).head() # doctest: +SKIP
           label  quartiles-0  quartiles-1  quartiles-2
    0      1       117.00        123.0        130.0
    1      2       111.25        112.0        114.0
    2      3       111.00        111.0        111.0
    3      4       111.00        111.5        112.5
    4      5       112.50        113.0        114.0

    """
    regions = regionprops(label_image, intensity_image=intensity_image,
                          cache=cache, extra_properties=extra_properties)
    if extra_properties is not None:
        properties = (
            list(properties) + [prop.__name__ for prop in extra_properties]
        )
    if len(regions) == 0:
        ndim = label_image.ndim
        label_image = np.zeros((3,) * ndim, dtype=int)
        label_image[(1,) * ndim] = 1
        if intensity_image is not None:
            intensity_image = np.zeros(
                    label_image.shape + intensity_image.shape[ndim:],
                    dtype=intensity_image.dtype
                    )
        regions = regionprops(label_image, intensity_image=intensity_image,
                              cache=cache, extra_properties=extra_properties)

        out_d = _props_to_dict(regions, properties=properties,
                               separator=separator)
        return {k: v[:0] for k, v in out_d.items()}

    return _props_to_dict(
        regions, properties=properties, separator=separator
    )


def regionprops(label_image, intensity_image=None, cache=True,
                coordinates=None, *, extra_properties=None):
    r"""Measure properties of labeled image regions.

    Parameters
    ----------
    label_image : (M, N[, P]) ndarray
        Labeled input image. Labels with value 0 are ignored.

        .. versionchanged:: 0.14.1
            Previously, ``label_image`` was processed by ``numpy.squeeze`` and
            so any number of singleton dimensions was allowed. This resulted in
            inconsistent handling of images with singleton dimensions. To
            recover the old behaviour, use
            ``regionprops(np.squeeze(label_image), ...)``.
    intensity_image : (M, N[, P][, C]) ndarray, optional
        Intensity (i.e., input) image with same size as labeled image, plus
        optionally an extra dimension for multichannel data. Currently,
        this extra channel dimension, if present, must be the last axis.
        Default is None.

        .. versionchanged:: 0.18.0
            The ability to provide an extra dimension for channels was added.
    cache : bool, optional
        Determine whether to cache calculated properties. The computation is
        much faster for cached properties, whereas the memory consumption
        increases.
    coordinates : DEPRECATED
        This argument is deprecated and will be removed in a future version
        of scikit-image.

        See :ref:`Coordinate conventions <numpy-images-coordinate-conventions>`
        for more details.

        .. deprecated:: 0.16.0
            Use "rc" coordinates everywhere. It may be sufficient to call
            ``numpy.transpose`` on your label image to get the same values as
            0.15 and earlier. However, for some properties, the transformation
            will be less trivial. For example, the new orientation is
            :math:`\frac{\pi}{2}` plus the old orientation.
    extra_properties : Iterable of callables
        Add extra property computation functions that are not included with
        skimage. The name of the property is derived from the function name,
        the dtype is inferred by calling the function on a small sample.
        If the name of an extra property clashes with the name of an existing
        property the extra property wil not be visible and a UserWarning is
        issued. A property computation function must take a region mask as its
        first argument. If the property requires an intensity image, it must
        accept the intensity image as the second argument.

    Returns
    -------
    properties : list of RegionProperties
        Each item describes one labeled region, and can be accessed using the
        attributes listed below.

    Notes
    -----
    The following properties can be accessed as attributes or keys:

    **area** : int
        Number of pixels of the region.
    **area_bbox** : int
        Number of pixels of bounding box.
    **area_convex** : int
        Number of pixels of convex hull image, which is the smallest convex
        polygon that encloses the region.
    **area_filled** : int
        Number of pixels of the region will all the holes filled in. Describes
        the area of the image_filled.
    **axis_major_length** : float
        The length of the major axis of the ellipse that has the same
        normalized second central moments as the region.
    **axis_minor_length** : float
        The length of the minor axis of the ellipse that has the same
        normalized second central moments as the region.
    **bbox** : tuple
        Bounding box ``(min_row, min_col, max_row, max_col)``.
        Pixels belonging to the bounding box are in the half-open interval
        ``[min_row; max_row)`` and ``[min_col; max_col)``.
    **centroid** : array
        Centroid coordinate tuple ``(row, col)``.
    **centroid_local** : array
        Centroid coordinate tuple ``(row, col)``, relative to region bounding
        box.
    **centroid_weighted** : array
        Centroid coordinate tuple ``(row, col)`` weighted with intensity
        image.
    **centroid_weighted_local** : array
        Centroid coordinate tuple ``(row, col)``, relative to region bounding
        box, weighted with intensity image.
    **coords** : (N, 2) ndarray
        Coordinate list ``(row, col)`` of the region.
    **eccentricity** : float
        Eccentricity of the ellipse that has the same second-moments as the
        region. The eccentricity is the ratio of the focal distance
        (distance between focal points) over the major axis length.
        The value is in the interval [0, 1).
        When it is 0, the ellipse becomes a circle.
    **equivalent_diameter_area** : float
        The diameter of a circle with the same area as the region.
    **euler_number** : int
        Euler characteristic of the set of non-zero pixels.
        Computed as number of connected components subtracted by number of
        holes (input.ndim connectivity). In 3D, number of connected
        components plus number of holes subtracted by number of tunnels.
    **extent** : float
        Ratio of pixels in the region to pixels in the total bounding box.
        Computed as ``area / (rows * cols)``
    **feret_diameter_max** : float
        Maximum Feret's diameter computed as the longest distance between
        points around a region's convex hull contour as determined by
        ``find_contours``. [5]_
    **image** : (H, J) ndarray
        Sliced binary region image which has the same size as bounding box.
    **image_convex** : (H, J) ndarray
        Binary convex hull image which has the same size as bounding box.
    **image_filled** : (H, J) ndarray
        Binary region image with filled holes which has the same size as
        bounding box.
    **image_intensity** : ndarray
        Image inside region bounding box.
    **inertia_tensor** : ndarray
        Inertia tensor of the region for the rotation around its mass.
    **inertia_tensor_eigvals** : tuple
        The eigenvalues of the inertia tensor in decreasing order.
    **intensity_max** : float
        Value with the greatest intensity in the region.
    **intensity_mean** : float
        Value with the mean intensity in the region.
    **intensity_min** : float
        Value with the least intensity in the region.
    **label** : int
        The label in the labeled input image.
    **moments** : (3, 3) ndarray
        Spatial moments up to 3rd order::

            m_ij = sum{ array(row, col) * row^i * col^j }

        where the sum is over the `row`, `col` coordinates of the region.
    **moments_central** : (3, 3) ndarray
        Central moments (translation invariant) up to 3rd order::

            mu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }

        where the sum is over the `row`, `col` coordinates of the region,
        and `row_c` and `col_c` are the coordinates of the region's centroid.
    **moments_hu** : tuple
        Hu moments (translation, scale and rotation invariant).
    **moments_normalized** : (3, 3) ndarray
        Normalized moments (translation and scale invariant) up to 3rd order::

            nu_ij = mu_ij / m_00^[(i+j)/2 + 1]

        where `m_00` is the zeroth spatial moment.
    **moments_weighted** : (3, 3) ndarray
        Spatial moments of intensity image up to 3rd order::

            wm_ij = sum{ array(row, col) * row^i * col^j }

        where the sum is over the `row`, `col` coordinates of the region.
    **moments_weighted_central** : (3, 3) ndarray
        Central moments (translation invariant) of intensity image up to
        3rd order::

            wmu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }

        where the sum is over the `row`, `col` coordinates of the region,
        and `row_c` and `col_c` are the coordinates of the region's weighted
        centroid.
    **moments_weighted_hu** : tuple
        Hu moments (translation, scale and rotation invariant) of intensity
        image.
    **moments_weighted_normalized** : (3, 3) ndarray
        Normalized moments (translation and scale invariant) of intensity
        image up to 3rd order::

            wnu_ij = wmu_ij / wm_00^[(i+j)/2 + 1]

        where ``wm_00`` is the zeroth spatial moment (intensity-weighted area).
    **orientation** : float
        Angle between the 0th axis (rows) and the major
        axis of the ellipse that has the same second moments as the region,
        ranging from `-pi/2` to `pi/2` counter-clockwise.
    **perimeter** : float
        Perimeter of object which approximates the contour as a line
        through the centers of border pixels using a 4-connectivity.
    **perimeter_crofton** : float
        Perimeter of object approximated by the Crofton formula in 4
        directions.
    **slice** : tuple of slices
        A slice to extract the object from the source image.
    **solidity** : float
        Ratio of pixels in the region to pixels of the convex hull image.

    Each region also supports iteration, so that you can do::

      for prop in region:
          print(prop, region[prop])

    See Also
    --------
    label

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] https://en.wikipedia.org/wiki/Image_moment
    .. [5] W. Pabst, E. Gregorová. Characterization of particles and particle
           systems, pp. 27-28. ICT Prague, 2007.
           https://old.vscht.cz/sil/keramika/Characterization_of_particles/CPPS%20_English%20version_.pdf

    Examples
    --------
    >>> from skimage import data, util
    >>> from skimage.measure import label, regionprops
    >>> img = util.img_as_ubyte(data.coins()) > 110
    >>> label_img = label(img, connectivity=img.ndim)
    >>> props = regionprops(label_img)
    >>> # centroid of first labeled object
    >>> props[0].centroid
    (22.72987986048314, 81.91228523446583)
    >>> # centroid of first labeled object
    >>> props[0]['centroid']
    (22.72987986048314, 81.91228523446583)

    Add custom measurements by passing functions as ``extra_properties``

    >>> from skimage import data, util
    >>> from skimage.measure import label, regionprops
    >>> import numpy as np
    >>> img = util.img_as_ubyte(data.coins()) > 110
    >>> label_img = label(img, connectivity=img.ndim)
    >>> def pixelcount(regionmask):
    ...     return np.sum(regionmask)
    >>> props = regionprops(label_img, extra_properties=(pixelcount,))
    >>> props[0].pixelcount
    7741
    >>> props[1]['pixelcount']
    42

    """

    if label_image.ndim not in (2, 3):
        raise TypeError('Only 2-D and 3-D images supported.')

    if not np.issubdtype(label_image.dtype, np.integer):
        if np.issubdtype(label_image.dtype, bool):
            raise TypeError(
                    'Non-integer image types are ambiguous: '
                    'use skimage.measure.label to label the connected'
                    'components of label_image,'
                    'or label_image.astype(np.uint8) to interpret'
                    'the True values as a single label.')
        else:
            raise TypeError(
                    'Non-integer label_image types are ambiguous')

    if coordinates is not None:
        if coordinates == 'rc':
            msg = ('The coordinates keyword argument to skimage.measure.'
                   'regionprops is deprecated. All features are now computed '
                   'in rc (row-column) coordinates. Please remove '
                   '`coordinates="rc"` from all calls to regionprops before '
                   'updating scikit-image.')
            warn(msg, stacklevel=2, category=FutureWarning)
        else:
            msg = ('Values other than "rc" for the "coordinates" argument '
                   'to skimage.measure.regionprops are no longer supported. '
                   'You should update your code to use "rc" coordinates and '
                   'stop using the "coordinates" argument, or use skimage '
                   'version 0.15.x or earlier.')
            raise ValueError(msg)

    regions = []

    objects = ndi.find_objects(label_image)
    for i, sl in enumerate(objects):
        if sl is None:
            continue

        label = i + 1

        props = RegionProperties(sl, label, label_image, intensity_image,
                                 cache, extra_properties=extra_properties)
        regions.append(props)

    return regions


def _parse_docs():
    import re
    import textwrap

    doc = regionprops.__doc__ or ''
    matches = re.finditer(r'\*\*(\w+)\*\* \:.*?\n(.*?)(?=\n    [\*\S]+)',
                          doc, flags=re.DOTALL)
    prop_doc = {m.group(1): textwrap.dedent(m.group(2)) for m in matches}

    return prop_doc


def _install_properties_docs():
    prop_doc = _parse_docs()

    for p in [member for member in dir(RegionProperties)
              if not member.startswith('_')]:
        getattr(RegionProperties, p).__doc__ = prop_doc[p]


if __debug__:
    # don't install docstrings when in optimized/non-debug mode
    _install_properties_docs()
