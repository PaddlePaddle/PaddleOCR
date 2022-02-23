"""Classes to represent keypoints, i.e. points given as xy-coordinates."""
from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.spatial.distance
import six.moves as sm

from .. import imgaug as ia
from .base import IAugmentable
from .utils import (normalize_shape, project_coords,
                    _remove_out_of_image_fraction_)


def compute_geometric_median(points=None, eps=1e-5, X=None):
    """Estimate the geometric median of points in 2D.

    Code from https://stackoverflow.com/a/30305181

    Parameters
    ----------
    points : (N,2) ndarray
        Points in 2D. Second axis must be given in xy-form.

    eps : float, optional
        Distance threshold when to return the median.

    X : None or (N,2) ndarray, optional
        Deprecated.

    Returns
    -------
    (2,) ndarray
        Geometric median as xy-coordinate.

    """
    # pylint: disable=invalid-name
    if X is not None:
        assert points is None
        ia.warn_deprecated("Using 'X' is deprecated, use 'points' instead.")
        points = X

    y = np.mean(points, 0)

    while True:
        dist = scipy.spatial.distance.cdist(points, [y])
        nonzeros = (dist != 0)[:, 0]

        dist_inv = 1 / dist[nonzeros]
        dist_inv_sum = np.sum(dist_inv)
        dist_inv_norm = dist_inv / dist_inv_sum
        T = np.sum(dist_inv_norm * points[nonzeros], 0)

        num_zeros = len(points) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(points):
            return y
        else:
            R = (T - y) * dist_inv_sum
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if scipy.spatial.distance.euclidean(y, y1) < eps:
            return y1

        y = y1


class Keypoint(object):
    """A single keypoint (aka landmark) on an image.

    Parameters
    ----------
    x : number
        Coordinate of the keypoint on the x axis.

    y : number
        Coordinate of the keypoint on the y axis.

    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def coords(self):
        """Get the xy-coordinates as an ``(N,2)`` ndarray.

        Added in 0.4.0.

        Returns
        -------
        ndarray
            An ``(N, 2)`` ``float32`` ndarray with ``N=1`` containing the
            coordinates of this keypoints.

        """
        arr = np.empty((1, 2), dtype=np.float32)
        arr[0, :] = [self.x, self.y]
        return arr

    @property
    def x_int(self):
        """Get the keypoint's x-coordinate, rounded to the closest integer.

        Returns
        -------
        result : int
            Keypoint's x-coordinate, rounded to the closest integer.

        """
        return int(np.round(self.x))

    @property
    def y_int(self):
        """Get the keypoint's y-coordinate, rounded to the closest integer.

        Returns
        -------
        result : int
            Keypoint's y-coordinate, rounded to the closest integer.

        """
        return int(np.round(self.y))

    @property
    def xy(self):
        """Get the keypoint's x- and y-coordinate as a single array.

        Added in 0.4.0.

        Returns
        -------
        ndarray
            A ``(2,)`` ``ndarray`` denoting the xy-coordinate pair.

        """
        return self.coords[0, :]

    @property
    def xy_int(self):
        """Get the keypoint's xy-coord, rounded to closest integer.

        Added in 0.4.0.

        Returns
        -------
        ndarray
            A ``(2,)`` ``ndarray`` denoting the xy-coordinate pair.

        """
        return np.round(self.xy).astype(np.int32)

    def project_(self, from_shape, to_shape):
        """Project in-place the keypoint onto a new position on a new image.

        E.g. if the keypoint is on its original image
        at ``x=(10 of 100 pixels)`` and ``y=(20 of 100 pixels)`` and is
        projected onto a new image with size ``(width=200, height=200)``, its
        new position will be ``(20, 40)``.

        This is intended for cases where the original image is resized.
        It cannot be used for more complex changes (e.g. padding, cropping).

        Added in 0.4.0.

        Parameters
        ----------
        from_shape : tuple of int
            Shape of the original image. (Before resize.)

        to_shape : tuple of int
            Shape of the new image. (After resize.)

        Returns
        -------
        imgaug.augmentables.kps.Keypoint
            Keypoint object with new coordinates.
            The instance of the keypoint may have been modified in-place.

        """
        xy_proj = project_coords([(self.x, self.y)], from_shape, to_shape)
        self.x, self.y = xy_proj[0]
        return self

    def project(self, from_shape, to_shape):
        """Project the keypoint onto a new position on a new image.

        E.g. if the keypoint is on its original image
        at ``x=(10 of 100 pixels)`` and ``y=(20 of 100 pixels)`` and is
        projected onto a new image with size ``(width=200, height=200)``, its
        new position will be ``(20, 40)``.

        This is intended for cases where the original image is resized.
        It cannot be used for more complex changes (e.g. padding, cropping).

        Parameters
        ----------
        from_shape : tuple of int
            Shape of the original image. (Before resize.)

        to_shape : tuple of int
            Shape of the new image. (After resize.)

        Returns
        -------
        imgaug.augmentables.kps.Keypoint
            Keypoint object with new coordinates.

        """
        return self.deepcopy().project_(from_shape, to_shape)

    def is_out_of_image(self, image):
        """Estimate whether this point is outside of the given image plane.

        Added in 0.4.0.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape
            and must contain at least two integers.

        Returns
        -------
        bool
            ``True`` is the point is inside the image plane, ``False``
            otherwise.

        """
        shape = normalize_shape(image)
        height, width = shape[0:2]
        y_inside = (0 <= self.y < height)
        x_inside = (0 <= self.x < width)
        return not y_inside or not x_inside

    def compute_out_of_image_fraction(self, image):
        """Compute fraction of the keypoint that is out of the image plane.

        The fraction is always either ``1.0`` (point is outside of the image
        plane) or ``0.0`` (point is inside the image plane). This method
        exists for consistency with other augmentables, e.g. bounding boxes.

        Added in 0.4.0.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape
            and must contain at least two integers.

        Returns
        -------
        float
            Either ``1.0`` (point is outside of the image plane) or
            ``0.0`` (point is inside of it).

        """
        return float(self.is_out_of_image(image))

    def shift_(self, x=0, y=0):
        """Move the keypoint around on an image in-place.

        Added in 0.4.0.

        Parameters
        ----------
        x : number, optional
            Move by this value on the x axis.

        y : number, optional
            Move by this value on the y axis.

        Returns
        -------
        imgaug.augmentables.kps.Keypoint
            Keypoint object with new coordinates.
            The instance of the keypoint may have been modified in-place.

        """
        self.x += x
        self.y += y
        return self

    def shift(self, x=0, y=0):
        """Move the keypoint around on an image.

        Parameters
        ----------
        x : number, optional
            Move by this value on the x axis.

        y : number, optional
            Move by this value on the y axis.

        Returns
        -------
        imgaug.augmentables.kps.Keypoint
            Keypoint object with new coordinates.

        """
        return self.deepcopy().shift_(x, y)

    def draw_on_image(self, image, color=(0, 255, 0), alpha=1.0, size=3,
                      copy=True, raise_if_out_of_image=False):
        """Draw the keypoint onto a given image.

        The keypoint is drawn as a square.

        Parameters
        ----------
        image : (H,W,3) ndarray
            The image onto which to draw the keypoint.

        color : int or list of int or tuple of int or (3,) ndarray, optional
            The RGB color of the keypoint.
            If a single ``int`` ``C``, then that is equivalent to ``(C,C,C)``.

        alpha : float, optional
            The opacity of the drawn keypoint, where ``1.0`` denotes a fully
            visible keypoint and ``0.0`` an invisible one.

        size : int, optional
            The size of the keypoint. If set to ``S``, each square will have
            size ``S x S``.

        copy : bool, optional
            Whether to copy the image before drawing the keypoint.

        raise_if_out_of_image : bool, optional
            Whether to raise an exception if the keypoint is outside of the
            image.

        Returns
        -------
        image : (H,W,3) ndarray
            Image with drawn keypoint.

        """
        # pylint: disable=redefined-outer-name
        if copy:
            image = np.copy(image)

        if image.ndim == 2:
            assert ia.is_single_number(color), (
                "Got a 2D image. Expected then 'color' to be a single number, "
                "but got %s." % (str(color),))
        elif image.ndim == 3 and ia.is_single_number(color):
            color = [color] * image.shape[-1]

        input_dtype = image.dtype
        alpha_color = color
        if alpha < 0.01:
            # keypoint invisible, nothing to do
            return image

        if alpha > 0.99:
            alpha = 1
        else:
            image = image.astype(np.float32, copy=False)
            alpha_color = alpha * np.array(color)

        height, width = image.shape[0:2]

        y, x = self.y_int, self.x_int

        x1 = max(x - size//2, 0)
        x2 = min(x + 1 + size//2, width)
        y1 = max(y - size//2, 0)
        y2 = min(y + 1 + size//2, height)

        x1_clipped, x2_clipped = np.clip([x1, x2], 0, width)
        y1_clipped, y2_clipped = np.clip([y1, y2], 0, height)

        x1_clipped_ooi = (x1_clipped < 0 or x1_clipped >= width)
        x2_clipped_ooi = (x2_clipped < 0 or x2_clipped >= width+1)
        y1_clipped_ooi = (y1_clipped < 0 or y1_clipped >= height)
        y2_clipped_ooi = (y2_clipped < 0 or y2_clipped >= height+1)
        x_ooi = (x1_clipped_ooi and x2_clipped_ooi)
        y_ooi = (y1_clipped_ooi and y2_clipped_ooi)
        x_zero_size = (x2_clipped - x1_clipped) < 1  # min size is 1px
        y_zero_size = (y2_clipped - y1_clipped) < 1
        if not x_ooi and not y_ooi and not x_zero_size and not y_zero_size:
            if alpha == 1:
                image[y1_clipped:y2_clipped, x1_clipped:x2_clipped] = color
            else:
                image[y1_clipped:y2_clipped, x1_clipped:x2_clipped] = (
                    (1 - alpha)
                    * image[y1_clipped:y2_clipped, x1_clipped:x2_clipped]
                    + alpha_color
                )
        else:
            if raise_if_out_of_image:
                raise Exception(
                    "Cannot draw keypoint x=%.8f, y=%.8f on image with "
                    "shape %s." % (y, x, image.shape))

        if image.dtype.name != input_dtype.name:
            if input_dtype.name == "uint8":
                image = np.clip(image, 0, 255, out=image)
            image = image.astype(input_dtype, copy=False)
        return image

    def generate_similar_points_manhattan(self, nb_steps, step_size,
                                          return_array=False):
        """Generate nearby points based on manhattan distance.

        To generate the first neighbouring points, a distance of ``S`` (step
        size) is moved from the center point (this keypoint) to the top,
        right, bottom and left, resulting in four new points. From these new
        points, the pattern is repeated. Overlapping points are ignored.

        The resulting points have a shape similar to a square rotated
        by ``45`` degrees.

        Parameters
        ----------
        nb_steps : int
            The number of steps to move from the center point.
            ``nb_steps=1`` results in a total of ``5`` output points (one
            center point + four neighbours).

        step_size : number
            The step size to move from every point to its neighbours.

        return_array : bool, optional
            Whether to return the generated points as a list of
            :class:`Keypoint` or an array of shape ``(N,2)``, where ``N`` is
            the number of generated points and the second axis contains the
            x-/y-coordinates.

        Returns
        -------
        list of imgaug.augmentables.kps.Keypoint or (N,2) ndarray
            If `return_array` was ``False``, then a list of :class:`Keypoint`.
            Otherwise a numpy array of shape ``(N,2)``, where ``N`` is the
            number of generated points and the second axis contains
            the x-/y-coordinates. The center keypoint (the one on which this
            function was called) is always included.

        """
        # TODO add test
        # Points generates in manhattan style with S steps have a shape
        # similar to a 45deg rotated square. The center line with the origin
        # point has S+1+S = 1+2*S points (S to the left, S to the right).
        # The lines above contain (S+1+S)-2 + (S+1+S)-2-2 + ... + 1 points.
        # E.g. for S=2 it would be 3+1=4 and for S=3 it would be 5+3+1=9.
        # Same for the lines below the center. Hence the total number of
        # points is S+1+S + 2*(S^2).
        nb_points = nb_steps + 1 + nb_steps + 2*(nb_steps**2)
        points = np.zeros((nb_points, 2), dtype=np.float32)

        # we start at the bottom-most line and move towards the top-most line
        yy = np.linspace(
            self.y - nb_steps * step_size,
            self.y + nb_steps * step_size,
            nb_steps + 1 + nb_steps)

        # bottom-most line contains only one point
        width = 1

        nth_point = 0
        for i_y, y in enumerate(yy):
            if width == 1:
                xx = [self.x]
            else:
                xx = np.linspace(
                    self.x - (width-1)//2 * step_size,
                    self.x + (width-1)//2 * step_size,
                    width)
            for x in xx:
                points[nth_point] = [x, y]
                nth_point += 1
            if i_y < nb_steps:
                width += 2
            else:
                width -= 2

        if return_array:
            return points
        return [self.deepcopy(x=point[0], y=point[1]) for point in points]

    def coords_almost_equals(self, other, max_distance=1e-4):
        """Estimate if this and another KP have almost identical coordinates.

        Added in 0.4.0.

        Parameters
        ----------
        other : imgaug.augmentables.kps.Keypoint or iterable
            The other keypoint with which to compare this one.
            If this is an ``iterable``, it is assumed to contain the
            xy-coordinates of a keypoint.

        max_distance : number, optional
            The maximum euclidean distance between a this keypoint and the
            other one. If the distance is exceeded, the two keypoints are not
            viewed as equal.

        Returns
        -------
        bool
            Whether the two keypoints have almost identical coordinates.

        """
        if ia.is_np_array(other):
            # we use flat here in case other is (N,2) instead of (4,)
            coords_b = other.flat
        elif ia.is_iterable(other):
            coords_b = list(ia.flatten(other))
        else:
            assert isinstance(other, Keypoint), (
                "Expected 'other' to be an iterable containing one "
                "(x,y)-coordinate pair or a Keypoint. "
                "Got type %s." % (type(other),))
            coords_b = other.coords.flat

        coords_a = self.coords

        return np.allclose(coords_a.flat, coords_b, atol=max_distance, rtol=0)

    def almost_equals(self, other, max_distance=1e-4):
        """Compare this and another KP's coordinates.

        .. note::

            This method is currently identical to ``coords_almost_equals``.
            It exists for consistency with ``BoundingBox`` and ``Polygons``.

        Added in 0.4.0.

        Parameters
        ----------
        other : imgaug.augmentables.kps.Keypoint or iterable
            The other object to compare against. Expected to be a
            ``Keypoint``.

        max_distance : number, optional
            See
            :func:`~imgaug.augmentables.kps.Keypoint.coords_almost_equals`.

        Returns
        -------
        bool
            ``True`` if the coordinates are almost equal. Otherwise ``False``.

        """
        return self.coords_almost_equals(other, max_distance=max_distance)

    def copy(self, x=None, y=None):
        """Create a shallow copy of the keypoint instance.

        Parameters
        ----------
        x : None or number, optional
            Coordinate of the keypoint on the x axis.
            If ``None``, the instance's value will be copied.

        y : None or number, optional
            Coordinate of the keypoint on the y axis.
            If ``None``, the instance's value will be copied.

        Returns
        -------
        imgaug.augmentables.kps.Keypoint
            Shallow copy.

        """
        return self.deepcopy(x=x, y=y)

    def deepcopy(self, x=None, y=None):
        """Create a deep copy of the keypoint instance.

        Parameters
        ----------
        x : None or number, optional
            Coordinate of the keypoint on the x axis.
            If ``None``, the instance's value will be copied.

        y : None or number, optional
            Coordinate of the keypoint on the y axis.
            If ``None``, the instance's value will be copied.

        Returns
        -------
        imgaug.augmentables.kps.Keypoint
            Deep copy.

        """
        x = self.x if x is None else x
        y = self.y if y is None else y
        return Keypoint(x=x, y=y)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Keypoint(x=%.8f, y=%.8f)" % (self.x, self.y)


class KeypointsOnImage(IAugmentable):
    """Container for all keypoints on a single image.

    Parameters
    ----------
    keypoints : list of imgaug.augmentables.kps.Keypoint
        List of keypoints on the image.

    shape : tuple of int or ndarray
        The shape of the image on which the objects are placed.
        Either an image with shape ``(H,W,[C])`` or a ``tuple`` denoting
        such an image shape.

    Examples
    --------
    >>> import numpy as np
    >>> from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
    >>>
    >>> image = np.zeros((70, 70))
    >>> kps = [Keypoint(x=10, y=20), Keypoint(x=34, y=60)]
    >>> kps_oi = KeypointsOnImage(kps, shape=image.shape)

    """

    def __init__(self, keypoints, shape):
        self.keypoints = keypoints
        self.shape = normalize_shape(shape)

    @property
    def items(self):
        """Get the keypoints in this container.

        Added in 0.4.0.

        Returns
        -------
        list of Keypoint
            Keypoints within this container.

        """
        return self.keypoints

    @items.setter
    def items(self, value):
        """Set the keypoints in this container.

        Added in 0.4.0.

        Parameters
        ----------
        value : list of Keypoint
            Keypoints within this container.

        """
        self.keypoints = value

    @property
    def height(self):
        """Get the image height.

        Returns
        -------
        int
            Image height.

        """
        return self.shape[0]

    @property
    def width(self):
        """Get the image width.

        Returns
        -------
        int
            Image width.

        """
        return self.shape[1]

    @property
    def empty(self):
        """Determine whether this object contains zero keypoints.

        Returns
        -------
        bool
            ``True`` if this object contains zero keypoints.

        """
        return len(self.keypoints) == 0

    def on_(self, image):
        """Project all keypoints from one image shape to a new one in-place.

        Added in 0.4.0.

        Parameters
        ----------
        image : ndarray or tuple of int
            New image onto which the keypoints are to be projected.
            May also simply be that new image's shape tuple.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            Object containing all projected keypoints.
            The object may have been modified in-place.

        """
        # pylint: disable=invalid-name
        on_shape = normalize_shape(image)
        if on_shape[0:2] == self.shape[0:2]:
            self.shape = on_shape  # channels may differ
            return self

        for i, kp in enumerate(self.keypoints):
            self.keypoints[i] = kp.project_(self.shape, on_shape)
        self.shape = on_shape
        return self

    def on(self, image):
        """Project all keypoints from one image shape to a new one.

        Parameters
        ----------
        image : ndarray or tuple of int
            New image onto which the keypoints are to be projected.
            May also simply be that new image's shape tuple.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            Object containing all projected keypoints.

        """
        # pylint: disable=invalid-name
        return self.deepcopy().on_(image)

    def draw_on_image(self, image, color=(0, 255, 0), alpha=1.0, size=3,
                      copy=True, raise_if_out_of_image=False):
        """Draw all keypoints onto a given image.

        Each keypoint is drawn as a square of provided color and size.

        Parameters
        ----------
        image : (H,W,3) ndarray
            The image onto which to draw the keypoints.
            This image should usually have the same shape as
            set in ``KeypointsOnImage.shape``.

        color : int or list of int or tuple of int or (3,) ndarray, optional
            The RGB color of all keypoints.
            If a single ``int`` ``C``, then that is equivalent to ``(C,C,C)``.

        alpha : float, optional
            The opacity of the drawn keypoint, where ``1.0`` denotes a fully
            visible keypoint and ``0.0`` an invisible one.

        size : int, optional
            The size of each point. If set to ``C``, each square will have
            size ``C x C``.

        copy : bool, optional
            Whether to copy the image before drawing the points.

        raise_if_out_of_image : bool, optional
            Whether to raise an exception if any keypoint is outside of the
            image.

        Returns
        -------
        (H,W,3) ndarray
            Image with drawn keypoints.

        """
        # pylint: disable=redefined-outer-name
        image = np.copy(image) if copy else image
        for keypoint in self.keypoints:
            image = keypoint.draw_on_image(
                image, color=color, alpha=alpha, size=size, copy=False,
                raise_if_out_of_image=raise_if_out_of_image)
        return image

    def remove_out_of_image_fraction_(self, fraction):
        """Remove all KPs with an OOI fraction of at least `fraction` in-place.

        'OOI' is the abbreviation for 'out of image'.

        This method exists for consistency with other augmentables, e.g.
        bounding boxes.

        Added in 0.4.0.

        Parameters
        ----------
        fraction : number
            Minimum out of image fraction that a keypoint has to have in
            order to be removed. Note that any keypoint can only have a
            fraction of either ``1.0`` (is outside) or ``0.0`` (is inside).
            Set this to ``0.0+eps`` to remove all points that are outside of
            the image. Setting this to ``0.0`` will remove all points.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            Reduced set of keypoints, with those thathad an out of image
            fraction greater or equal the given one removed.
            The object may have been modified in-place.

        """
        return _remove_out_of_image_fraction_(self, fraction)

    def remove_out_of_image_fraction(self, fraction):
        """Remove all KPs with an out of image fraction of at least `fraction`.

        This method exists for consistency with other augmentables, e.g.
        bounding boxes.

        Added in 0.4.0.

        Parameters
        ----------
        fraction : number
            Minimum out of image fraction that a keypoint has to have in
            order to be removed. Note that any keypoint can only have a
            fraction of either ``1.0`` (is outside) or ``0.0`` (is inside).
            Set this to ``0.0+eps`` to remove all points that are outside of
            the image. Setting this to ``0.0`` will remove all points.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            Reduced set of keypoints, with those thathad an out of image
            fraction greater or equal the given one removed.

        """
        return self.deepcopy().remove_out_of_image_fraction_(fraction)

    def clip_out_of_image_(self):
        """Remove all KPs that are outside of the image plane.

        This method exists for consistency with other augmentables, e.g.
        bounding boxes.

        Added in 0.4.0.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            Keypoints that are inside the image plane.
            The object may have been modified in-place.

        """
        # we could use anything >0 here as the fraction
        return self.remove_out_of_image_fraction_(0.5)

    def clip_out_of_image(self):
        """Remove all KPs that are outside of the image plane.

        This method exists for consistency with other augmentables, e.g.
        bounding boxes.

        Added in 0.4.0.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            Keypoints that are inside the image plane.

        """
        return self.deepcopy().clip_out_of_image_()

    def shift_(self, x=0, y=0):
        """Move the keypoints on the x/y-axis in-place.

        Added in 0.4.0.

        Parameters
        ----------
        x : number, optional
            Move each keypoint by this value on the x axis.

        y : number, optional
            Move each keypoint by this value on the y axis.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            Keypoints after moving them.
            The object and its items may have been modified in-place.

        """
        for i, keypoint in enumerate(self.keypoints):
            self.keypoints[i] = keypoint.shift_(x=x, y=y)
        return self

    def shift(self, x=0, y=0):
        """Move the keypoints on the x/y-axis.

        Parameters
        ----------
        x : number, optional
            Move each keypoint by this value on the x axis.

        y : number, optional
            Move each keypoint by this value on the y axis.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            Keypoints after moving them.

        """
        return self.deepcopy().shift_(x=x, y=y)

    @ia.deprecated(alt_func="KeypointsOnImage.to_xy_array()")
    def get_coords_array(self):
        """Convert all keypoint coordinates to an array of shape ``(N,2)``.

        Returns
        -------
        (N, 2) ndarray
            Array containing the coordinates of all keypoints.
            ``N`` denotes the number of keypoints. The second axis denotes
            the x/y-coordinates.

        """
        return self.to_xy_array()

    def to_xy_array(self):
        """Convert all keypoint coordinates to an array of shape ``(N,2)``.

        Returns
        -------
        (N, 2) ndarray
            Array containing the coordinates of all keypoints.
            ``N`` denotes the number of keypoints. The second axis denotes
            the x/y-coordinates.

        """
        result = np.zeros((len(self.keypoints), 2), dtype=np.float32)
        for i, keypoint in enumerate(self.keypoints):
            result[i, 0] = keypoint.x
            result[i, 1] = keypoint.y
        return result

    @staticmethod
    @ia.deprecated(alt_func="KeypointsOnImage.from_xy_array()")
    def from_coords_array(coords, shape):
        """Convert an ``(N,2)`` array to a ``KeypointsOnImage`` object.

        Parameters
        ----------
        coords : (N, 2) ndarray
            Coordinates of ``N`` keypoints on an image, given as a ``(N,2)``
            array of xy-coordinates.

        shape : tuple
            The shape of the image on which the keypoints are placed.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            :class:`KeypointsOnImage` object containing the array's keypoints.

        """
        return KeypointsOnImage.from_xy_array(coords, shape)

    @classmethod
    def from_xy_array(cls, xy, shape):
        """Convert an ``(N,2)`` array to a ``KeypointsOnImage`` object.

        Parameters
        ----------
        xy : (N, 2) ndarray or iterable of iterable of number
            Coordinates of ``N`` keypoints on an image, given as a ``(N,2)``
            array of xy-coordinates.

        shape : tuple of int or ndarray
            The shape of the image on which the keypoints are placed.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            :class:`KeypointsOnImage` object containing the array's keypoints.

        """
        xy = np.array(xy, dtype=np.float32)

        # note that np.array([]) is (0,), not (0, 2)
        if xy.shape[0] == 0:  # pylint: disable=unsubscriptable-object
            return KeypointsOnImage([], shape)

        assert xy.ndim == 2 and xy.shape[-1] == 2, (  # pylint: disable=unsubscriptable-object
            "Expected input array to have shape (N,2), "
            "got shape %s." % (xy.shape,))
        keypoints = [Keypoint(x=coord[0], y=coord[1]) for coord in xy]
        return KeypointsOnImage(keypoints, shape)

    def fill_from_xy_array_(self, xy):
        """Modify the keypoint coordinates of this instance in-place.

        .. note::

            This currently expects that `xy` contains exactly as many
            coordinates as there are keypoints in this instance. Otherwise,
            an ``AssertionError`` will be raised.

        Added in 0.4.0.

        Parameters
        ----------
        xy : (N, 2) ndarray or iterable of iterable of number
            Coordinates of ``N`` keypoints on an image, given as a ``(N,2)``
            array of xy-coordinates. ``N`` must match the number of keypoints
            in this instance.

        Returns
        -------
        KeypointsOnImage
            This instance itself, with updated keypoint coordinates.
            Note that the instance was modified in-place.

        """
        xy = np.array(xy, dtype=np.float32)

        # note that np.array([]) is (0,), not (0, 2)
        assert xy.shape[0] == 0 or (xy.ndim == 2 and xy.shape[-1] == 2), (  # pylint: disable=unsubscriptable-object
            "Expected input array to have shape (N,2), "
            "got shape %s." % (xy.shape,))

        assert len(xy) == len(self.keypoints), (
            "Expected to receive as many keypoint coordinates as there are "
            "currently keypoints in this instance. Got %d, expected %d." % (
                len(xy), len(self.keypoints)))

        for kp, (x, y) in zip(self.keypoints, xy):
            kp.x = x
            kp.y = y

        return self

    # TODO add to_gaussian_heatmaps(), from_gaussian_heatmaps()
    def to_keypoint_image(self, size=1):
        """Create an ``(H,W,N)`` image with keypoint coordinates set to ``255``.

        This method generates a new ``uint8`` array of shape ``(H,W,N)``,
        where ``H`` is the ``.shape`` height, ``W`` the ``.shape`` width and
        ``N`` is the number of keypoints. The array is filled with zeros.
        The coordinate of the ``n``-th keypoint is set to ``255`` in the
        ``n``-th channel.

        This function can be used as a helper when augmenting keypoints with
        a method that only supports the augmentation of images.

        Parameters
        -------
        size : int
            Size of each (squared) point.

        Returns
        -------
        (H,W,N) ndarray
            Image in which the keypoints are marked. ``H`` is the height,
            defined in ``KeypointsOnImage.shape[0]`` (analogous ``W``).
            ``N`` is the number of keypoints.

        """
        height, width = self.shape[0:2]
        image = np.zeros((height, width, len(self.keypoints)), dtype=np.uint8)
        assert size % 2 != 0, (
            "Expected 'size' to have an odd value, got %d instead." % (size,))
        sizeh = max(0, (size-1)//2)
        for i, keypoint in enumerate(self.keypoints):
            # TODO for float values spread activation over several cells
            # here and do voting at the end
            y = keypoint.y_int
            x = keypoint.x_int

            x1 = np.clip(x - sizeh, 0, width-1)
            x2 = np.clip(x + sizeh + 1, 0, width)
            y1 = np.clip(y - sizeh, 0, height-1)
            y2 = np.clip(y + sizeh + 1, 0, height)

            if x1 < x2 and y1 < y2:
                image[y1:y2, x1:x2, i] = 128
            if 0 <= y < height and 0 <= x < width:
                image[y, x, i] = 255
        return image

    @staticmethod
    def from_keypoint_image(image, if_not_found_coords={"x": -1, "y": -1},
                            threshold=1, nb_channels=None):
        """Convert ``to_keypoint_image()`` outputs to ``KeypointsOnImage``.

        This is the inverse of :func:`KeypointsOnImage.to_keypoint_image`.

        Parameters
        ----------
        image : (H,W,N) ndarray
            The keypoints image. N is the number of keypoints.

        if_not_found_coords : tuple or list or dict or None, optional
            Coordinates to use for keypoints that cannot be found in `image`.

            * If this is a ``list``/``tuple``, it must contain two ``int``
              values.
            * If it is a ``dict``, it must contain the keys ``x`` and
              ``y`` with each containing one ``int`` value.
            * If this is ``None``, then the keypoint will not be added to the
              final :class:`KeypointsOnImage` object.

        threshold : int, optional
            The search for keypoints works by searching for the argmax in
            each channel. This parameters contains the minimum value that
            the max must have in order to be viewed as a keypoint.

        nb_channels : None or int, optional
            Number of channels of the image on which the keypoints are placed.
            Some keypoint augmenters require that information.
            If set to ``None``, the keypoint's shape will be set
            to ``(height, width)``, otherwise ``(height, width, nb_channels)``.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            The extracted keypoints.

        """
        # pylint: disable=dangerous-default-value
        assert image.ndim == 3, (
            "Expected 'image' to have three dimensions, "
            "got %d with shape %s instead." % (image.ndim, image.shape))
        height, width, nb_keypoints = image.shape

        drop_if_not_found = False
        if if_not_found_coords is None:
            drop_if_not_found = True
            if_not_found_x = -1
            if_not_found_y = -1
        elif isinstance(if_not_found_coords, (tuple, list)):
            assert len(if_not_found_coords) == 2, (
                "Expected tuple 'if_not_found_coords' to contain exactly two "
                "values, got %d values." % (len(if_not_found_coords),))
            if_not_found_x = if_not_found_coords[0]
            if_not_found_y = if_not_found_coords[1]
        elif isinstance(if_not_found_coords, dict):
            if_not_found_x = if_not_found_coords["x"]
            if_not_found_y = if_not_found_coords["y"]
        else:
            raise Exception(
                "Expected if_not_found_coords to be None or tuple or list "
                "or dict, got %s." % (type(if_not_found_coords),))

        keypoints = []
        for i in sm.xrange(nb_keypoints):
            maxidx_flat = np.argmax(image[..., i])
            maxidx_ndim = np.unravel_index(maxidx_flat, (height, width))

            found = (image[maxidx_ndim[0], maxidx_ndim[1], i] >= threshold)
            if found:
                x = maxidx_ndim[1] + 0.5
                y = maxidx_ndim[0] + 0.5
                keypoints.append(Keypoint(x=x, y=y))
            else:
                if drop_if_not_found:
                    # dont add the keypoint to the result list, i.e. drop it
                    pass
                else:
                    keypoints.append(Keypoint(x=if_not_found_x,
                                              y=if_not_found_y))

        out_shape = (height, width)
        if nb_channels is not None:
            out_shape += (nb_channels,)
        return KeypointsOnImage(keypoints, shape=out_shape)

    def to_distance_maps(self, inverted=False):
        """Generate a ``(H,W,N)`` array of distance maps for ``N`` keypoints.

        The ``n``-th distance map contains at every location ``(y, x)`` the
        euclidean distance to the ``n``-th keypoint.

        This function can be used as a helper when augmenting keypoints with a
        method that only supports the augmentation of images.

        Parameters
        -------
        inverted : bool, optional
            If ``True``, inverted distance maps are returned where each
            distance value d is replaced by ``d/(d+1)``, i.e. the distance
            maps have values in the range ``(0.0, 1.0]`` with ``1.0`` denoting
            exactly the position of the respective keypoint.

        Returns
        -------
        (H,W,N) ndarray
            A ``float32`` array containing ``N`` distance maps for ``N``
            keypoints. Each location ``(y, x, n)`` in the array denotes the
            euclidean distance at ``(y, x)`` to the ``n``-th keypoint.
            If `inverted` is ``True``, the distance ``d`` is replaced
            by ``d/(d+1)``. The height and width of the array match the
            height and width in ``KeypointsOnImage.shape``.

        """
        height, width = self.shape[0:2]
        distance_maps = np.zeros((height, width, len(self.keypoints)),
                                 dtype=np.float32)

        yy = np.arange(0, height)
        xx = np.arange(0, width)
        grid_xx, grid_yy = np.meshgrid(xx, yy)

        for i, keypoint in enumerate(self.keypoints):
            y, x = keypoint.y, keypoint.x
            distance_maps[:, :, i] = (grid_xx - x) ** 2 + (grid_yy - y) ** 2
        distance_maps = np.sqrt(distance_maps)
        if inverted:
            return 1/(distance_maps+1)
        return distance_maps

    # TODO add option to if_not_found_coords to reuse old keypoint coords
    @staticmethod
    def from_distance_maps(distance_maps, inverted=False,
                           if_not_found_coords={"x": -1, "y": -1},
                           threshold=None, nb_channels=None):
        """Convert outputs of ``to_distance_maps()`` to ``KeypointsOnImage``.

        This is the inverse of :func:`KeypointsOnImage.to_distance_maps`.

        Parameters
        ----------
        distance_maps : (H,W,N) ndarray
            The distance maps. ``N`` is the number of keypoints.

        inverted : bool, optional
            Whether the given distance maps were generated in inverted mode
            (i.e. :func:`KeypointsOnImage.to_distance_maps` was called with
            ``inverted=True``) or in non-inverted mode.

        if_not_found_coords : tuple or list or dict or None, optional
            Coordinates to use for keypoints that cannot be found
            in `distance_maps`.

            * If this is a ``list``/``tuple``, it must contain two ``int``
              values.
            * If it is a ``dict``, it must contain the keys ``x`` and
              ``y`` with each containing one ``int`` value.
            * If this is ``None``, then the keypoint will not be added to the
              final :class:`KeypointsOnImage` object.

        threshold : float, optional
            The search for keypoints works by searching for the
            argmin (non-inverted) or argmax (inverted) in each channel. This
            parameters contains the maximum (non-inverted) or
            minimum (inverted) value to accept in order to view a hit as a
            keypoint. Use ``None`` to use no min/max.

        nb_channels : None or int, optional
            Number of channels of the image on which the keypoints are placed.
            Some keypoint augmenters require that information.
            If set to ``None``, the keypoint's shape will be set
            to ``(height, width)``, otherwise ``(height, width, nb_channels)``.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            The extracted keypoints.

        """
        # pylint: disable=dangerous-default-value
        assert distance_maps.ndim == 3, (
            "Expected three-dimensional input, got %d dimensions and "
            "shape %s." % (distance_maps.ndim, distance_maps.shape))
        height, width, nb_keypoints = distance_maps.shape

        drop_if_not_found = False
        if if_not_found_coords is None:
            drop_if_not_found = True
            if_not_found_x = -1
            if_not_found_y = -1
        elif isinstance(if_not_found_coords, (tuple, list)):
            assert len(if_not_found_coords) == 2, (
                "Expected tuple/list 'if_not_found_coords' to contain "
                "exactly two entries, got %d." % (len(if_not_found_coords),))
            if_not_found_x = if_not_found_coords[0]
            if_not_found_y = if_not_found_coords[1]
        elif isinstance(if_not_found_coords, dict):
            if_not_found_x = if_not_found_coords["x"]
            if_not_found_y = if_not_found_coords["y"]
        else:
            raise Exception(
                "Expected if_not_found_coords to be None or tuple or list or "
                "dict, got %s." % (type(if_not_found_coords),))

        keypoints = []
        for i in sm.xrange(nb_keypoints):
            # TODO introduce voting here among all distance values that have
            #      min/max values
            if inverted:
                hitidx_flat = np.argmax(distance_maps[..., i])
            else:
                hitidx_flat = np.argmin(distance_maps[..., i])
            hitidx_ndim = np.unravel_index(hitidx_flat, (height, width))
            if not inverted and threshold is not None:
                found = (distance_maps[hitidx_ndim[0], hitidx_ndim[1], i]
                         < threshold)
            elif inverted and threshold is not None:
                found = (distance_maps[hitidx_ndim[0], hitidx_ndim[1], i]
                         >= threshold)
            else:
                found = True
            if found:
                keypoints.append(Keypoint(x=hitidx_ndim[1], y=hitidx_ndim[0]))
            else:
                if drop_if_not_found:
                    # dont add the keypoint to the result list, i.e. drop it
                    pass
                else:
                    keypoints.append(Keypoint(x=if_not_found_x,
                                              y=if_not_found_y))

        out_shape = (height, width)
        if nb_channels is not None:
            out_shape += (nb_channels,)
        return KeypointsOnImage(keypoints, shape=out_shape)

    # TODO add to_keypoints_on_image_() and call that wherever possible
    def to_keypoints_on_image(self):
        """Convert the keypoints to one ``KeypointsOnImage`` instance.

        This method exists for consistency with ``BoundingBoxesOnImage``,
        ``PolygonsOnImage`` and ``LineStringsOnImage``.

        Added in 0.4.0.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            Copy of this keypoints instance.

        """
        return self.deepcopy()

    def invert_to_keypoints_on_image_(self, kpsoi):
        """Invert the output of ``to_keypoints_on_image()`` in-place.

        This function writes in-place into this ``KeypointsOnImage``
        instance.

        Added in 0.4.0.

        Parameters
        ----------
        kpsoi : imgaug.augmentables.kps.KeypointsOnImages
            Keypoints to copy data from, i.e. the outputs of
            ``to_keypoints_on_image()``.

        Returns
        -------
        KeypointsOnImage
            Keypoints container with updated coordinates.
            Note that the instance is also updated in-place.

        """
        nb_points_exp = len(self.keypoints)
        assert len(kpsoi.keypoints) == nb_points_exp, (
            "Expected %d coordinates, got %d." % (
                nb_points_exp, len(kpsoi.keypoints)))

        for kp_target, kp_source in zip(self.keypoints, kpsoi.keypoints):
            kp_target.x = kp_source.x
            kp_target.y = kp_source.y

        self.shape = kpsoi.shape
        return self

    def copy(self, keypoints=None, shape=None):
        """Create a shallow copy of the ``KeypointsOnImage`` object.

        Parameters
        ----------
        keypoints : None or list of imgaug.Keypoint, optional
            List of keypoints on the image.
            If ``None``, the instance's keypoints will be copied.

        shape : tuple of int, optional
            The shape of the image on which the keypoints are placed.
            If ``None``, the instance's shape will be copied.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            Shallow copy.

        """
        if keypoints is None:
            keypoints = self.keypoints[:]
        if shape is None:
            # use tuple() here in case the shape was provided as a list
            shape = tuple(self.shape)

        return KeypointsOnImage(keypoints, shape)

    def deepcopy(self, keypoints=None, shape=None):
        """Create a deep copy of the ``KeypointsOnImage`` object.

        Parameters
        ----------
        keypoints : None or list of imgaug.Keypoint, optional
            List of keypoints on the image.
            If ``None``, the instance's keypoints will be copied.

        shape : tuple of int, optional
            The shape of the image on which the keypoints are placed.
            If ``None``, the instance's shape will be copied.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            Deep copy.

        """
        # Manual copy is far faster than deepcopy, so use manual copy here.
        if keypoints is None:
            keypoints = [kp.deepcopy() for kp in self.keypoints]
        if shape is None:
            # use tuple() here in case the shape was provided as a list
            shape = tuple(self.shape)

        return KeypointsOnImage(keypoints, shape)

    def __getitem__(self, indices):
        """Get the keypoint(s) with given indices.

        Added in 0.4.0.

        Returns
        -------
        list of imgaug.augmentables.kps.Keypoint
            Keypoint(s) with given indices.

        """
        return self.keypoints[indices]

    def __iter__(self):
        """Iterate over the keypoints in this container.

        Added in 0.4.0.

        Yields
        ------
        Keypoint
            A keypoint in this container.
            The order is identical to the order in the keypoint list
            provided upon class initialization.

        """
        return iter(self.items)

    def __len__(self):
        """Get the number of items in this instance.

        Added in 0.4.0.

        Returns
        -------
        int
            Number of items in this instance.

        """
        return len(self.items)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "KeypointsOnImage(%s, shape=%s)" % (
            str(self.keypoints), self.shape)
