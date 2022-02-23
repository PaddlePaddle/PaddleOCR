"""Classes dealing with polygons."""
from __future__ import print_function, division, absolute_import

import traceback
import collections

import numpy as np
import scipy.spatial.distance
import six.moves as sm
import skimage.draw
import skimage.measure

from .. import imgaug as ia
from .. import random as iarandom
from .base import IAugmentable
from .utils import (normalize_shape,
                    interpolate_points,
                    _remove_out_of_image_fraction_,
                    project_coords_,
                    _normalize_shift_args)


def recover_psois_(psois, psois_orig, recoverer, random_state):
    """Apply a polygon recoverer to input polygons in-place.

    Parameters
    ----------
    psois : list of imgaug.augmentables.polys.PolygonsOnImage or imgaug.augmentables.polys.PolygonsOnImage
        The possibly broken polygons, e.g. after augmentation.
        The `recoverer` is applied to them.

    psois_orig : list of imgaug.augmentables.polys.PolygonsOnImage or imgaug.augmentables.polys.PolygonsOnImage
        Original polygons that were later changed to `psois`.
        They are an extra input to `recoverer`.

    recoverer : imgaug.augmentables.polys._ConcavePolygonRecoverer
        The polygon recoverer used to repair broken input polygons.

    random_state : None or int or RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState
        An RNG to use during the polygon recovery.

    Returns
    -------
    list of imgaug.augmentables.polys.PolygonsOnImage or imgaug.augmentables.polys.PolygonsOnImage
        List of repaired polygons. Note that this is `psois`, which was
        changed in-place.

    """
    input_was_list = True
    if not isinstance(psois, list):
        input_was_list = False
        psois = [psois]
        psois_orig = [psois_orig]

    for i, psoi in enumerate(psois):
        for j, polygon in enumerate(psoi.polygons):
            poly_rec = recoverer.recover_from(
                polygon.exterior, psois_orig[i].polygons[j],
                random_state)

            # Don't write into `polygon.exterior[...] = ...` because the
            # shapes might have changed. We could also first check if the
            # shapes are identical and only then write in-place, but as the
            # array for `poly_rec.exterior` was already created, that would
            # not provide any benefits.
            polygon.exterior = poly_rec.exterior

    if not input_was_list:
        return psois[0]
    return psois


# TODO somehow merge with BoundingBox
# TODO add functions: simplify() (eg via shapely.ops.simplify()),
# extend(all_sides=0, top=0, right=0, bottom=0, left=0),
# intersection(other, default=None), union(other), iou(other), to_heatmap, to_mask
class Polygon(object):
    """Class representing polygons.

    Each polygon is parameterized by its corner points, given as absolute
    x- and y-coordinates with sub-pixel accuracy.

    Parameters
    ----------
    exterior : list of imgaug.augmentables.kps.Keypoint or list of tuple of float or (N,2) ndarray
        List of points defining the polygon. May be either a ``list`` of
        :class:`~imgaug.augmentables.kps.Keypoint` objects or a ``list`` of
        ``tuple`` s in xy-form or a numpy array of shape (N,2) for ``N``
        points in xy-form.
        All coordinates are expected to be the absolute subpixel-coordinates
        on the image, given as ``float`` s, e.g. ``x=10.7`` and ``y=3.4`` for a
        point at coordinates ``(10.7, 3.4)``. Their order is expected to be
        clock-wise. They are expected to not be closed (i.e. first and last
        coordinate differ).

    label : None or str, optional
        Label of the polygon, e.g. a string representing the class.

    """

    def __init__(self, exterior, label=None):
        """Create a new Polygon instance."""
        # TODO get rid of this deferred import
        from imgaug.augmentables.kps import Keypoint

        if isinstance(exterior, list):
            if not exterior:
                # for empty lists, make sure that the shape is (0, 2) and
                # not (0,) as that is also expected when the input is a numpy
                # array
                self.exterior = np.zeros((0, 2), dtype=np.float32)
            elif isinstance(exterior[0], Keypoint):
                # list of Keypoint
                self.exterior = np.float32([[point.x, point.y]
                                            for point in exterior])
            else:
                # list of tuples (x, y)
                # TODO just np.float32(exterior) here?
                self.exterior = np.float32([[point[0], point[1]]
                                            for point in exterior])
        else:
            assert ia.is_np_array(exterior), (
                "Expected exterior to be a list of tuples (x, y) or "
                "an (N, 2) array, got type %s" % (exterior,))
            assert exterior.ndim == 2 and exterior.shape[1] == 2, (
                "Expected exterior to be a list of tuples (x, y) or "
                "an (N, 2) array, got an array of shape %s" % (
                    exterior.shape,))
            # TODO deal with int inputs here?
            self.exterior = np.float32(exterior)

        # Remove last point if it is essentially the same as the first
        # point (polygons are always assumed to be closed anyways). This also
        # prevents problems with shapely, which seems to add the last point
        # automatically.
        is_closed = (
            len(self.exterior) >= 2
            and np.allclose(self.exterior[0, :], self.exterior[-1, :]))
        if is_closed:
            self.exterior = self.exterior[:-1]

        self.label = label

    @property
    def coords(self):
        """Alias for attribute ``exterior``.

        Added in 0.4.0.

        Returns
        -------
        ndarray
            An ``(N, 2)`` ``float32`` ndarray containing the coordinates of
            this polygon. This identical to the attribute ``exterior``.

        """
        return self.exterior

    @property
    def xx(self):
        """Get the x-coordinates of all points on the exterior.

        Returns
        -------
        (N,2) ndarray
            ``float32`` x-coordinates array of all points on the exterior.

        """
        return self.exterior[:, 0]

    @property
    def yy(self):
        """Get the y-coordinates of all points on the exterior.

        Returns
        -------
        (N,2) ndarray
            ``float32`` y-coordinates array of all points on the exterior.

        """
        return self.exterior[:, 1]

    @property
    def xx_int(self):
        """Get the discretized x-coordinates of all points on the exterior.

        The conversion from ``float32`` coordinates to ``int32`` is done
        by first rounding the coordinates to the closest integer and then
        removing everything after the decimal point.

        Returns
        -------
        (N,2) ndarray
            ``int32`` x-coordinates of all points on the exterior.

        """
        return np.int32(np.round(self.xx))

    @property
    def yy_int(self):
        """Get the discretized y-coordinates of all points on the exterior.

        The conversion from ``float32`` coordinates to ``int32`` is done
        by first rounding the coordinates to the closest integer and then
        removing everything after the decimal point.

        Returns
        -------
        (N,2) ndarray
            ``int32`` y-coordinates of all points on the exterior.

        """
        return np.int32(np.round(self.yy))

    @property
    def is_valid(self):
        """Estimate whether the polygon has a valid geometry.

        To to be considered valid, the polygon must be made up of at
        least ``3`` points and have a concave shape, i.e. line segments may
        not intersect or overlap. Multiple consecutive points are allowed to
        have the same coordinates.

        Returns
        -------
        bool
            ``True`` if polygon has at least ``3`` points and is concave,
            otherwise ``False``.

        """
        if len(self.exterior) < 3:
            return False
        return self.to_shapely_polygon().is_valid

    @property
    def area(self):
        """Compute the area of the polygon.

        Returns
        -------
        number
            Area of the polygon.

        """
        if len(self.exterior) < 3:
            return 0.0
        poly = self.to_shapely_polygon()
        return poly.area

    @property
    def height(self):
        """Compute the height of a bounding box encapsulating the polygon.

        The height is computed based on the two exterior coordinates with
        lowest and largest x-coordinates.

        Returns
        -------
        number
            Height of the polygon.

        """
        yy = self.yy
        return max(yy) - min(yy)

    @property
    def width(self):
        """Compute the width of a bounding box encapsulating the polygon.

        The width is computed based on the two exterior coordinates with
        lowest and largest x-coordinates.

        Returns
        -------
        number
            Width of the polygon.

        """
        xx = self.xx
        return max(xx) - min(xx)

    def project_(self, from_shape, to_shape):
        """Project the polygon onto an image with different shape in-place.

        The relative coordinates of all points remain the same.
        E.g. a point at ``(x=20, y=20)`` on an image
        ``(width=100, height=200)`` will be projected on a new
        image ``(width=200, height=100)`` to ``(x=40, y=10)``.

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
        imgaug.augmentables.polys.Polygon
            Polygon object with new coordinates.
            The object may have been modified in-place.

        """
        self.exterior = project_coords_(self.coords, from_shape, to_shape)
        return self

    def project(self, from_shape, to_shape):
        """Project the polygon onto an image with different shape.

        The relative coordinates of all points remain the same.
        E.g. a point at ``(x=20, y=20)`` on an image
        ``(width=100, height=200)`` will be projected on a new
        image ``(width=200, height=100)`` to ``(x=40, y=10)``.

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
        imgaug.augmentables.polys.Polygon
            Polygon object with new coordinates.

        """
        return self.deepcopy().project_(from_shape, to_shape)

    def find_closest_point_index(self, x, y, return_distance=False):
        """Find the index of the exterior point closest to given coordinates.

        "Closeness" is here defined based on euclidean distance.
        This method will raise an ``AssertionError`` if the exterior contains
        no points.

        Parameters
        ----------
        x : number
            X-coordinate around which to search for close points.

        y : number
            Y-coordinate around which to search for close points.

        return_distance : bool, optional
            Whether to also return the distance of the closest point.

        Returns
        -------
        int
            Index of the closest point.

        number
            Euclidean distance to the closest point.
            This value is only returned if `return_distance` was set
            to ``True``.

        """
        assert len(self.exterior) > 0, (
            "Cannot find the closest point on a polygon which's exterior "
            "contains no points.")
        distances = []
        for x2, y2 in self.exterior:
            dist = (x2 - x) ** 2 + (y2 - y) ** 2
            distances.append(dist)
        distances = np.sqrt(distances)
        closest_idx = np.argmin(distances)
        if return_distance:
            return closest_idx, distances[closest_idx]
        return closest_idx

    def compute_out_of_image_area(self, image):
        """Compute the area of the BB that is outside of the image plane.

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
            Total area of the bounding box that is outside of the image plane.
            Can be ``0.0``.

        """
        polys_clipped = self.clip_out_of_image(image)
        if len(polys_clipped) == 0:
            return self.area
        return self.area - sum([poly.area for poly in polys_clipped])

    def compute_out_of_image_fraction(self, image):
        """Compute fraction of polygon area outside of the image plane.

        This estimates ``f = A_ooi / A``, where ``A_ooi`` is the area of the
        polygon that is outside of the image plane, while ``A`` is the
        total area of the bounding box.

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
            Fraction of the polygon area that is outside of the image
            plane. Returns ``0.0`` if the polygon is fully inside of
            the image plane or has zero points. If the polygon has an area
            of zero, the polygon is treated similarly to a :class:`LineString`,
            i.e. the fraction of the line that is outside the image plane is
            returned.

        """
        area = self.area
        if area == 0:
            return self.to_line_string().compute_out_of_image_fraction(image)
        return self.compute_out_of_image_area(image) / area

    # TODO keep this method? it is almost an alias for is_out_of_image()
    def is_fully_within_image(self, image):
        """Estimate whether the polygon is fully inside an image plane.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape and
            must contain at least two ``int`` s.

        Returns
        -------
        bool
            ``True`` if the polygon is fully inside the image area.
            ``False`` otherwise.

        """
        return not self.is_out_of_image(image, fully=True, partly=True)

    # TODO keep this method? it is almost an alias for is_out_of_image()
    def is_partly_within_image(self, image):
        """Estimate whether the polygon is at least partially inside an image.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape and
            must contain at least two ``int`` s.

        Returns
        -------
        bool
            ``True`` if the polygon is at least partially inside the image area.
            ``False`` otherwise.

        """
        return not self.is_out_of_image(image, fully=True, partly=False)

    def is_out_of_image(self, image, fully=True, partly=False):
        """Estimate whether the polygon is partially/fully outside of an image.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape and
            must contain at least two ``int`` s.

        fully : bool, optional
            Whether to return ``True`` if the polygon is fully outside of the
            image area.

        partly : bool, optional
            Whether to return ``True`` if the polygon is at least partially
            outside fo the image area.

        Returns
        -------
        bool
            ``True`` if the polygon is partially/fully outside of the image
            area, depending on defined parameters.
            ``False`` otherwise.

        """
        # TODO this is inconsistent with line strings, which return a default
        #      value in these cases
        if len(self.exterior) == 0:
            raise Exception("Cannot determine whether the polygon is inside "
                            "the image, because it contains no points.")

        # The line string is identical to the edge of the polygon.
        # If the edge is fully inside the image, we know that the polygon must
        # be fully inside the image.
        # If the edge is partially outside of the image, we know that the
        # polygon is partially outside of the image.
        # Only if the edge is fully outside of the image we cannot be sure if
        # the polygon's inner area overlaps with the image (e.g. if the
        # polygon contains the whole image in it).
        ls = self.to_line_string()
        if ls.is_fully_within_image(image):
            return False
        if ls.is_out_of_image(image, fully=False, partly=True):
            return partly

        # LS is fully outside of the image. Estimate whether there is any
        # intersection with the image plane. If so, we know that there is
        # partial overlap (full overlap would mean that the LS was fully inside
        # the image).
        polys = self.clip_out_of_image(image)
        if len(polys) > 0:
            return partly
        return fully

    @ia.deprecated(alt_func="Polygon.clip_out_of_image()",
                   comment="clip_out_of_image() has the exactly same "
                           "interface.")
    def cut_out_of_image(self, image):
        """Cut off all parts of the polygon that are outside of an image."""
        return self.clip_out_of_image(image)

    # TODO this currently can mess up the order of points - change somehow to
    #      keep the order
    def clip_out_of_image(self, image):
        """Cut off all parts of the polygon that are outside of an image.

        This operation may lead to new points being created.
        As a single polygon may be split into multiple new polygons, the result
        is always a list, which may contain more than one output polygon.

        This operation will return an empty list if the polygon is completely
        outside of the image plane.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use for the clipping of the polygon.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape and must
            contain at least two ``int`` s.

        Returns
        -------
        list of imgaug.augmentables.polys.Polygon
            Polygon, clipped to fall within the image dimensions.
            Returned as a ``list``, because the clipping can split the polygon
            into multiple parts. The list may also be empty, if the polygon was
            fully outside of the image plane.

        """
        # load shapely lazily, which makes the dependency more optional
        import shapely.geometry

        # Shapely polygon conversion requires at least 3 coordinates
        if len(self.exterior) == 0:
            return []
        if len(self.exterior) in [1, 2]:
            ls = self.to_line_string(closed=False)
            ls_clipped = ls.clip_out_of_image(image)
            assert len(ls_clipped) <= 1
            if len(ls_clipped) == 0:
                return []
            return [self.deepcopy(exterior=ls_clipped[0].coords)]

        h, w = image.shape[0:2] if ia.is_np_array(image) else image[0:2]
        poly_shapely = self.to_shapely_polygon()
        poly_image = shapely.geometry.Polygon([(0, 0), (w, 0), (w, h), (0, h)])
        multipoly_inter_shapely = poly_shapely.intersection(poly_image)
        ignore_types = (shapely.geometry.LineString,
                        shapely.geometry.MultiLineString,
                        shapely.geometry.point.Point,
                        shapely.geometry.MultiPoint)
        if isinstance(multipoly_inter_shapely, shapely.geometry.Polygon):
            multipoly_inter_shapely = shapely.geometry.MultiPolygon(
                [multipoly_inter_shapely])
        elif isinstance(multipoly_inter_shapely,
                        shapely.geometry.MultiPolygon):
            # we got a multipolygon from shapely, no need to change anything
            # anymore
            pass
        elif isinstance(multipoly_inter_shapely, ignore_types):
            # polygons that become (one or more) lines/points after clipping
            # are here ignored
            multipoly_inter_shapely = shapely.geometry.MultiPolygon([])
        elif isinstance(multipoly_inter_shapely,
                        shapely.geometry.GeometryCollection):
            # Shapely returns GEOMETRYCOLLECTION EMPTY if there is nothing
            # remaining after the clip.
            assert multipoly_inter_shapely.is_empty
            return []
        else:
            print(multipoly_inter_shapely, image, self.exterior)
            raise Exception(
                "Got an unexpected result of type %s from Shapely for "
                "image (%d, %d) and polygon %s. This is an internal error. "
                "Please report." % (
                    type(multipoly_inter_shapely), h, w, self.exterior)
            )

        polygons = []
        for poly_inter_shapely in multipoly_inter_shapely.geoms:
            polygons.append(Polygon.from_shapely(poly_inter_shapely,
                                                 label=self.label))

        # Shapely changes the order of points, we try here to preserve it as
        # much as possible.
        # Note here, that all points of the new polygon might have high
        # distance to the points on the old polygon. This can happen if the
        # polygon overlaps with the image plane, but all of its points are
        # outside of the image plane. The new polygon will not be made up of
        # any of the old points.
        polygons_reordered = []
        for polygon in polygons:
            best_idx = None
            best_dist = None
            for x, y in self.exterior:
                point_idx, dist = polygon.find_closest_point_index(
                    x=x, y=y, return_distance=True)
                if best_idx is None or dist < best_dist:
                    best_idx = point_idx
                    best_dist = dist
            if best_idx is not None:
                polygon_reordered = \
                    polygon.change_first_point_by_index(best_idx)
                polygons_reordered.append(polygon_reordered)

        return polygons_reordered

    def shift_(self, x=0, y=0):
        """Move this polygon along the x/y-axis in-place.

        The origin ``(0, 0)`` is at the top left of the image.

        Added in 0.4.0.

        Parameters
        ----------
        x : number, optional
            Value to be added to all x-coordinates. Positive values shift
            towards the right images.

        y : number, optional
            Value to be added to all y-coordinates. Positive values shift
            towards the bottom images.

        Returns
        -------
        imgaug.augmentables.polys.Polygon
            Shifted polygon.
            The object may have been modified in-place.

        """
        self.exterior[:, 0] += x
        self.exterior[:, 1] += y
        return self

    def shift(self, x=0, y=0, top=None, right=None, bottom=None, left=None):
        """Move this polygon along the x/y-axis.

        The origin ``(0, 0)`` is at the top left of the image.

        Parameters
        ----------
        x : number, optional
            Value to be added to all x-coordinates. Positive values shift
            towards the right images.

        y : number, optional
            Value to be added to all y-coordinates. Positive values shift
            towards the bottom images.

        top : None or int, optional
            Deprecated since 0.4.0.
            Amount of pixels by which to shift this object *from* the
            top (towards the bottom).

        right : None or int, optional
            Deprecated since 0.4.0.
            Amount of pixels by which to shift this object *from* the
            right (towards the left).

        bottom : None or int, optional
            Deprecated since 0.4.0.
            Amount of pixels by which to shift this object *from* the
            bottom (towards the top).

        left : None or int, optional
            Deprecated since 0.4.0.
            Amount of pixels by which to shift this object *from* the
            left (towards the right).

        Returns
        -------
        imgaug.augmentables.polys.Polygon
            Shifted polygon.

        """
        x, y = _normalize_shift_args(
            x, y, top=top, right=right, bottom=bottom, left=left)
        return self.deepcopy().shift_(x=x, y=y)

    # TODO separate this into draw_face_on_image() and draw_border_on_image()
    # TODO add tests for line thickness
    def draw_on_image(self,
                      image,
                      color=(0, 255, 0), color_face=None,
                      color_lines=None, color_points=None,
                      alpha=1.0, alpha_face=None,
                      alpha_lines=None, alpha_points=None,
                      size=1, size_lines=None, size_points=None,
                      raise_if_out_of_image=False):
        """Draw the polygon on an image.

        Parameters
        ----------
        image : (H,W,C) ndarray
            The image onto which to draw the polygon. Usually expected to be
            of dtype ``uint8``, though other dtypes are also handled.

        color : iterable of int, optional
            The color to use for the whole polygon.
            Must correspond to the channel layout of the image. Usually RGB.
            The values for `color_face`, `color_lines` and `color_points`
            will be derived from this color if they are set to ``None``.
            This argument has no effect if `color_face`, `color_lines`
            and `color_points` are all set anything other than ``None``.

        color_face : None or iterable of int, optional
            The color to use for the inner polygon area (excluding perimeter).
            Must correspond to the channel layout of the image. Usually RGB.
            If this is ``None``, it will be derived from ``color * 1.0``.

        color_lines : None or iterable of int, optional
            The color to use for the line (aka perimeter/border) of the
            polygon.
            Must correspond to the channel layout of the image. Usually RGB.
            If this is ``None``, it will be derived from ``color * 0.5``.

        color_points : None or iterable of int, optional
            The color to use for the corner points of the polygon.
            Must correspond to the channel layout of the image. Usually RGB.
            If this is ``None``, it will be derived from ``color * 0.5``.

        alpha : float, optional
            The opacity of the whole polygon, where ``1.0`` denotes a
            completely visible polygon and ``0.0`` an invisible one.
            The values for `alpha_face`, `alpha_lines` and `alpha_points`
            will be derived from this alpha value if they are set to ``None``.
            This argument has no effect if `alpha_face`, `alpha_lines`
            and `alpha_points` are all set anything other than ``None``.

        alpha_face : None or number, optional
            The opacity of the polygon's inner area (excluding the perimeter),
            where ``1.0`` denotes a completely visible inner area and ``0.0``
            an invisible one.
            If this is ``None``, it will be derived from ``alpha * 0.5``.

        alpha_lines : None or number, optional
            The opacity of the polygon's line (aka perimeter/border),
            where ``1.0`` denotes a completely visible line and ``0.0`` an
            invisible one.
            If this is ``None``, it will be derived from ``alpha * 1.0``.

        alpha_points : None or number, optional
            The opacity of the polygon's corner points, where ``1.0`` denotes
            completely visible corners and ``0.0`` invisible ones.
            If this is ``None``, it will be derived from ``alpha * 1.0``.

        size : int, optional
            Size of the polygon.
            The sizes of the line and points are derived from this value,
            unless they are set.

        size_lines : None or int, optional
            Thickness of the polygon's line (aka perimeter/border).
            If ``None``, this value is derived from `size`.

        size_points : int, optional
            Size of the points in pixels.
            If ``None``, this value is derived from ``3 * size``.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the polygon is fully
            outside of the image. If set to ``False``, no error will be
            raised and only the parts inside the image will be drawn.

        Returns
        -------
        (H,W,C) ndarray
            Image with the polygon drawn on it. Result dtype is the same as the
            input dtype.

        """
        # pylint: disable=invalid-name
        def _assert_not_none(arg_name, arg_value):
            assert arg_value is not None, (
                "Expected '%s' to not be None, got type %s." % (
                    arg_name, type(arg_value),))

        def _default_to(var, default):
            if var is None:
                return default
            return var

        _assert_not_none("color", color)
        _assert_not_none("alpha", alpha)
        _assert_not_none("size", size)

        # FIXME due to the np.array(.) and the assert at ndim==2 below, this
        #       will always fail on 2D images?
        color_face = _default_to(color_face, np.array(color))
        color_lines = _default_to(color_lines, np.array(color) * 0.5)
        color_points = _default_to(color_points, np.array(color) * 0.5)

        alpha_face = _default_to(alpha_face, alpha * 0.5)
        alpha_lines = _default_to(alpha_lines, alpha)
        alpha_points = _default_to(alpha_points, alpha)

        size_lines = _default_to(size_lines, size)
        size_points = _default_to(size_points, size * 3)

        if image.ndim == 2:
            assert ia.is_single_number(color_face), (
                "Got a 2D image. Expected then 'color_face' to be a single "
                "number, but got %s." % (str(color_face),))
            color_face = [color_face]
        elif image.ndim == 3 and ia.is_single_number(color_face):
            color_face = [color_face] * image.shape[-1]

        if alpha_face < 0.01:
            alpha_face = 0
        elif alpha_face > 0.99:
            alpha_face = 1

        if raise_if_out_of_image and self.is_out_of_image(image):
            raise Exception("Cannot draw polygon %s on image with "
                            "shape %s." % (str(self), image.shape))

        # TODO np.clip to image plane if is_fully_within_image(), similar to
        #      how it is done for bounding boxes

        # TODO improve efficiency by only drawing in rectangle that covers
        #      poly instead of drawing in the whole image
        # TODO for a rectangular polygon, the face coordinates include the
        #      top/left boundary but not the right/bottom boundary. This may
        #      be unintuitive when not drawing the boundary. Maybe somehow
        #      remove the boundary coordinates from the face coordinates after
        #      generating both?
        input_dtype = image.dtype
        result = image.astype(np.float32)
        rr, cc = skimage.draw.polygon(
            self.yy_int, self.xx_int, shape=image.shape)
        if len(rr) > 0:
            if alpha_face == 1:
                result[rr, cc] = np.float32(color_face)
            elif alpha_face == 0:
                pass
            else:
                result[rr, cc] = (
                    (1 - alpha_face) * result[rr, cc, :]
                    + alpha_face * np.float32(color_face)
                )

        ls_open = self.to_line_string(closed=False)
        ls_closed = self.to_line_string(closed=True)
        result = ls_closed.draw_lines_on_image(
            result, color=color_lines, alpha=alpha_lines,
            size=size_lines, raise_if_out_of_image=raise_if_out_of_image)
        result = ls_open.draw_points_on_image(
            result, color=color_points, alpha=alpha_points,
            size=size_points, raise_if_out_of_image=raise_if_out_of_image)

        if input_dtype.type == np.uint8:
            # TODO make clipping more flexible
            result = np.clip(np.round(result), 0, 255).astype(input_dtype)
        else:
            result = result.astype(input_dtype)

        return result

    # TODO add pad, similar to LineStrings
    # TODO add pad_max, similar to LineStrings
    # TODO add prevent_zero_size, similar to LineStrings
    def extract_from_image(self, image):
        """Extract all image pixels within the polygon area.

        This method returns a rectangular image array. All pixels within
        that rectangle that do not belong to the polygon area will be filled
        with zeros (i.e. they will be black).
        The method will also zero-pad the image if the polygon is
        partially/fully outside of the image.

        Parameters
        ----------
        image : (H,W) ndarray or (H,W,C) ndarray
            The image from which to extract the pixels within the polygon.

        Returns
        -------
        (H',W') ndarray or (H',W',C) ndarray
            Pixels within the polygon. Zero-padded if the polygon is
            partially/fully outside of the image.

        """
        assert image.ndim in [2, 3], (
            "Expected image of shape (H,W,[C]), got shape %s." % (
                image.shape,))

        if len(self.exterior) <= 2:
            raise Exception("Polygon must be made up of at least 3 points to "
                            "extract its area from an image.")

        bb = self.to_bounding_box()
        bb_area = bb.extract_from_image(image)
        if self.is_out_of_image(image, fully=True, partly=False):
            return bb_area

        xx = self.xx_int
        yy = self.yy_int
        xx_mask = xx - np.min(xx)
        yy_mask = yy - np.min(yy)
        height_mask = np.max(yy_mask)
        width_mask = np.max(xx_mask)

        rr_face, cc_face = skimage.draw.polygon(
            yy_mask, xx_mask, shape=(height_mask, width_mask))

        mask = np.zeros((height_mask, width_mask), dtype=np.bool)
        mask[rr_face, cc_face] = True

        if image.ndim == 3:
            mask = np.tile(mask[:, :, np.newaxis], (1, 1, image.shape[2]))

        return bb_area * mask

    def change_first_point_by_coords(self, x, y, max_distance=1e-4,
                                     raise_if_too_far_away=True):
        """
        Reorder exterior points so that the point closest to given x/y is first.

        This method takes a given ``(x,y)`` coordinate, finds the closest
        corner point on the exterior and reorders all exterior corner points
        so that the found point becomes the first one in the array.

        If no matching points are found, an exception is raised.

        Parameters
        ----------
        x : number
            X-coordinate of the point.

        y : number
            Y-coordinate of the point.

        max_distance : None or number, optional
            Maximum distance past which possible matches are ignored.
            If ``None`` the distance limit is deactivated.

        raise_if_too_far_away : bool, optional
            Whether to raise an exception if the closest found point is too
            far away (``True``) or simply return an unchanged copy if this
            object (``False``).

        Returns
        -------
        imgaug.augmentables.polys.Polygon
            Copy of this polygon with the new point order.

        """
        if len(self.exterior) == 0:
            raise Exception("Cannot reorder polygon points, because it "
                            "contains no points.")

        closest_idx, closest_dist = self.find_closest_point_index(
            x=x, y=y, return_distance=True)
        if max_distance is not None and closest_dist > max_distance:
            if not raise_if_too_far_away:
                return self.deepcopy()

            closest_point = self.exterior[closest_idx, :]
            raise Exception(
                "Closest found point (%.9f, %.9f) exceeds max_distance of "
                "%.9f exceeded" % (
                    closest_point[0], closest_point[1], closest_dist))
        return self.change_first_point_by_index(closest_idx)

    def change_first_point_by_index(self, point_idx):
        """
        Reorder exterior points so that the point with given index is first.

        This method takes a given index and reorders all exterior corner points
        so that the point with that index becomes the first one in the array.

        An ``AssertionError`` will be raised if the index does not match
        any exterior point's index or the exterior does not contain any points.

        Parameters
        ----------
        point_idx : int
            Index of the desired starting point.

        Returns
        -------
        imgaug.augmentables.polys.Polygon
            Copy of this polygon with the new point order.

        """
        assert 0 <= point_idx < len(self.exterior), (
            "Expected index of new first point to be in the discrete interval "
            "[0..%d). Got index %d." % (len(self.exterior), point_idx))

        if point_idx == 0:
            return self.deepcopy()
        exterior = np.concatenate(
            (self.exterior[point_idx:, :], self.exterior[:point_idx, :]),
            axis=0
        )
        return self.deepcopy(exterior=exterior)

    def subdivide_(self, points_per_edge):
        """Derive a new poly with ``N`` interpolated points per edge in-place.

        See :func:`~imgaug.augmentables.lines.LineString.subdivide` for details.

        Added in 0.4.0.

        Parameters
        ----------
        points_per_edge : int
            Number of points to interpolate on each edge.

        Returns
        -------
        imgaug.augmentables.polys.Polygon
            Polygon with subdivided edges.
            The object may have been modified in-place.

        """
        if len(self.exterior) == 1:
            return self
        ls = self.to_line_string(closed=True)
        ls_sub = ls.subdivide(points_per_edge)
        # [:-1] even works if the polygon contains zero points
        exterior_subdivided = ls_sub.coords[:-1]
        self.exterior = exterior_subdivided
        return self

    def subdivide(self, points_per_edge):
        """Derive a new polygon with ``N`` interpolated points per edge.

        See :func:`~imgaug.augmentables.lines.LineString.subdivide` for details.

        Added in 0.4.0.

        Parameters
        ----------
        points_per_edge : int
            Number of points to interpolate on each edge.

        Returns
        -------
        imgaug.augmentables.polys.Polygon
            Polygon with subdivided edges.

        """
        return self.deepcopy().subdivide_(points_per_edge)

    def to_shapely_polygon(self):
        """Convert this polygon to a ``Shapely`` ``Polygon``.

        Returns
        -------
        shapely.geometry.Polygon
            The ``Shapely`` ``Polygon`` matching this polygon's exterior.

        """
        # load shapely lazily, which makes the dependency more optional
        import shapely.geometry

        return shapely.geometry.Polygon(
            [(point[0], point[1]) for point in self.exterior])

    def to_shapely_line_string(self, closed=False, interpolate=0):
        """Convert this polygon to a ``Shapely`` ``LineString`` object.

        Parameters
        ----------
        closed : bool, optional
            Whether to return the line string with the last point being
            identical to the first point.

        interpolate : int, optional
            Number of points to interpolate between any pair of two
            consecutive points. These points are added to the final line string.

        Returns
        -------
        shapely.geometry.LineString
            The ``Shapely`` ``LineString`` matching the polygon's exterior.

        """
        return _convert_points_to_shapely_line_string(
            self.exterior, closed=closed, interpolate=interpolate)

    def to_bounding_box(self):
        """Convert this polygon to a bounding box containing the polygon.

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBox
            Bounding box that tightly encapsulates the polygon.

        """
        # TODO get rid of this deferred import
        from imgaug.augmentables.bbs import BoundingBox

        xx = self.xx
        yy = self.yy
        return BoundingBox(x1=min(xx), x2=max(xx),
                           y1=min(yy), y2=max(yy),
                           label=self.label)

    def to_keypoints(self):
        """Convert this polygon's exterior to ``Keypoint`` instances.

        Returns
        -------
        list of imgaug.augmentables.kps.Keypoint
            Exterior vertices as :class:`~imgaug.augmentables.kps.Keypoint`
            instances.

        """
        # TODO get rid of this deferred import
        from imgaug.augmentables.kps import Keypoint

        return [Keypoint(x=point[0], y=point[1]) for point in self.exterior]

    def to_line_string(self, closed=True):
        """Convert this polygon's exterior to a ``LineString`` instance.

        Parameters
        ----------
        closed : bool, optional
            Whether to close the line string, i.e. to add the first point of
            the `exterior` also as the last point at the end of the line string.
            This has no effect if the polygon has a single point or zero
            points.

        Returns
        -------
        imgaug.augmentables.lines.LineString
            Exterior of the polygon as a line string.

        """
        from imgaug.augmentables.lines import LineString
        if not closed or len(self.exterior) <= 1:
            return LineString(self.exterior, label=self.label)
        return LineString(
            np.concatenate([self.exterior, self.exterior[0:1, :]], axis=0),
            label=self.label)

    @staticmethod
    def from_shapely(polygon_shapely, label=None):
        """Create a polygon from a ``Shapely`` ``Polygon``.

        .. note::

            This will remove any holes in the shapely polygon.

        Parameters
        ----------
        polygon_shapely : shapely.geometry.Polygon
             The shapely polygon.

        label : None or str, optional
            The label of the new polygon.

        Returns
        -------
        imgaug.augmentables.polys.Polygon
            A polygon with the same exterior as the ``Shapely`` ``Polygon``.

        """
        # load shapely lazily, which makes the dependency more optional
        import shapely.geometry

        assert isinstance(polygon_shapely, shapely.geometry.Polygon), (
            "Expected the input to be a shapely.geometry.Polgon instance. "
            "Got %s." % (type(polygon_shapely),))
        # polygon_shapely.exterior can be None if the polygon was
        # instantiated without points
        has_no_exterior = (
            polygon_shapely.exterior is None
            or len(polygon_shapely.exterior.coords) == 0)
        if has_no_exterior:
            return Polygon([], label=label)
        exterior = np.float32([[x, y]
                               for (x, y)
                               in polygon_shapely.exterior.coords])
        return Polygon(exterior, label=label)

    def coords_almost_equals(self, other, max_distance=1e-4,
                             points_per_edge=8):
        """Alias for :func:`Polygon.exterior_almost_equals`.

        Parameters
        ----------
        other : imgaug.augmentables.polys.Polygon or (N,2) ndarray or list of tuple
            See
            :func:`~imgaug.augmentables.polys.Polygon.exterior_almost_equals`.

        max_distance : number, optional
            See
            :func:`~imgaug.augmentables.polys.Polygon.exterior_almost_equals`.

        points_per_edge : int, optional
            See
            :func:`~imgaug.augmentables.polys.Polygon.exterior_almost_equals`.

        Returns
        -------
        bool
            Whether the two polygon's exteriors can be viewed as equal
            (approximate test).

        """
        return self.exterior_almost_equals(
            other, max_distance=max_distance, points_per_edge=points_per_edge)

    def exterior_almost_equals(self, other, max_distance=1e-4,
                               points_per_edge=8):
        """Estimate if this and another polygon's exterior are almost identical.

        The two exteriors can have different numbers of points, but any point
        randomly sampled on the exterior of one polygon should be close to the
        closest point on the exterior of the other polygon.

        .. note::

            This method works in an approximative way. One can come up with
            polygons with fairly different shapes that will still be estimated
            as equal by this method. In practice however this should be
            unlikely to be the case. The probability for something like that
            goes down as the interpolation parameter is increased.

        Parameters
        ----------
        other : imgaug.augmentables.polys.Polygon or (N,2) ndarray or list of tuple
            The other polygon with which to compare the exterior.
            If this is an ``ndarray``, it is assumed to represent an exterior.
            It must then have dtype ``float32`` and shape ``(N,2)`` with the
            second dimension denoting xy-coordinates.
            If this is a ``list`` of ``tuple`` s, it is assumed to represent
            an exterior. Each tuple then must contain exactly two ``number`` s,
            denoting xy-coordinates.

        max_distance : number, optional
            The maximum euclidean distance between a point on one polygon and
            the closest point on the other polygon. If the distance is exceeded
            for any such pair, the two exteriors are not viewed as equal. The
            points are either the points contained in the polygon's exterior
            ndarray or interpolated points between these.

        points_per_edge : int, optional
            How many points to interpolate on each edge.

        Returns
        -------
        bool
            Whether the two polygon's exteriors can be viewed as equal
            (approximate test).

        """
        if isinstance(other, list):
            other = Polygon(np.float32(other))
        elif ia.is_np_array(other):
            other = Polygon(other)
        else:
            assert isinstance(other, Polygon), (
                "Expected 'other' to be a list of coordinates, a coordinate "
                "array or a single Polygon. Got type %s." % (type(other),))

        return self.to_line_string(closed=True).coords_almost_equals(
            other.to_line_string(closed=True),
            max_distance=max_distance,
            points_per_edge=points_per_edge
        )

    def almost_equals(self, other, max_distance=1e-4, points_per_edge=8):
        """
        Estimate if this polygon's and another's geometry/labels are similar.

        This is the same as
        :func:`~imgaug.augmentables.polys.Polygon.exterior_almost_equals` but
        additionally compares the labels.

        Parameters
        ----------
        other : imgaug.augmentables.polys.Polygon
            The other object to compare against. Expected to be a ``Polygon``.

        max_distance : float, optional
            See
            :func:`~imgaug.augmentables.polys.Polygon.exterior_almost_equals`.

        points_per_edge : int, optional
            See
            :func:`~imgaug.augmentables.polys.Polygon.exterior_almost_equals`.

        Returns
        -------
        bool
            ``True`` if the coordinates are almost equal and additionally
            the labels are equal. Otherwise ``False``.

        """
        if self.label != other.label:
            return False
        return self.exterior_almost_equals(
            other, max_distance=max_distance, points_per_edge=points_per_edge)

    def copy(self, exterior=None, label=None):
        """Create a shallow copy of this object.

        Parameters
        ----------
        exterior : list of imgaug.augmentables.kps.Keypoint or list of tuple or (N,2) ndarray, optional
            List of points defining the polygon. See
            :func:`~imgaug.augmentables.polys.Polygon.__init__` for details.

        label : None or str, optional
            If not ``None``, the ``label`` of the copied object will be set
            to this value.

        Returns
        -------
        imgaug.augmentables.polys.Polygon
            Shallow copy.

        """
        return self.deepcopy(exterior=exterior, label=label)

    def deepcopy(self, exterior=None, label=None):
        """Create a deep copy of this object.

        Parameters
        ----------
        exterior : list of Keypoint or list of tuple or (N,2) ndarray, optional
            List of points defining the polygon. See
            `imgaug.augmentables.polys.Polygon.__init__` for details.

        label : None or str
            If not ``None``, the ``label`` of the copied object will be set
            to this value.

        Returns
        -------
        imgaug.augmentables.polys.Polygon
            Deep copy.

        """
        return Polygon(
            exterior=np.copy(self.exterior) if exterior is None else exterior,
            label=self.label if label is None else label)

    def __getitem__(self, indices):
        """Get the coordinate(s) with given indices.

        Added in 0.4.0.

        Returns
        -------
        ndarray
            xy-coordinate(s) as ``ndarray``.

        """
        return self.exterior[indices]

    def __iter__(self):
        """Iterate over the coordinates of this instance.

        Added in 0.4.0.

        Yields
        ------
        ndarray
            An ``(2,)`` ``ndarray`` denoting an xy-coordinate pair.

        """
        return iter(self.exterior)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        points_str = ", ".join([
            "(x=%.3f, y=%.3f)" % (point[0], point[1])
            for point
            in self.exterior])
        return "Polygon([%s] (%d points), label=%s)" % (
            points_str, len(self.exterior), self.label)


# TODO add tests for this
class PolygonsOnImage(IAugmentable):
    """Container for all polygons on a single image.

    Parameters
    ----------
    polygons : list of imgaug.augmentables.polys.Polygon
        List of polygons on the image.

    shape : tuple of int or ndarray
        The shape of the image on which the objects are placed.
        Either an image with shape ``(H,W,[C])`` or a ``tuple`` denoting
        such an image shape.

    Examples
    --------
    >>> import numpy as np
    >>> from imgaug.augmentables.polys import Polygon, PolygonsOnImage
    >>> image = np.zeros((100, 100))
    >>> polys = [
    >>>     Polygon([(0.5, 0.5), (100.5, 0.5), (100.5, 100.5), (0.5, 100.5)]),
    >>>     Polygon([(50.5, 0.5), (100.5, 50.5), (50.5, 100.5), (0.5, 50.5)])
    >>> ]
    >>> polys_oi = PolygonsOnImage(polys, shape=image.shape)

    """

    def __init__(self, polygons, shape):
        self.polygons = polygons
        self.shape = normalize_shape(shape)

    @property
    def items(self):
        """Get the polygons in this container.

        Added in 0.4.0.

        Returns
        -------
        list of Polygon
            Polygons within this container.

        """
        return self.polygons

    @items.setter
    def items(self, value):
        """Set the polygons in this container.

        Added in 0.4.0.

        Parameters
        ----------
        value : list of Polygon
            Polygons within this container.

        """
        self.polygons = value

    @property
    def empty(self):
        """Estimate whether this object contains zero polygons.

        Returns
        -------
        bool
            ``True`` if this object contains zero polygons.

        """
        return len(self.polygons) == 0

    def on_(self, image):
        """Project all polygons from one image shape to a new one in-place.

        Added in 0.4.0.

        Parameters
        ----------
        image : ndarray or tuple of int
            New image onto which the polygons are to be projected.
            May also simply be that new image's shape ``tuple``.

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage
            Object containing all projected polygons.
            The object and its items may have been modified in-place.

        """
        # pylint: disable=invalid-name
        on_shape = normalize_shape(image)
        if on_shape[0:2] == self.shape[0:2]:
            self.shape = on_shape  # channels may differ
            return self

        for i, item in enumerate(self.items):
            self.polygons[i] = item.project_(self.shape, on_shape)
        self.shape = on_shape
        return self

    def on(self, image):
        """Project all polygons from one image shape to a new one.

        Parameters
        ----------
        image : ndarray or tuple of int
            New image onto which the polygons are to be projected.
            May also simply be that new image's shape ``tuple``.

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage
            Object containing all projected polygons.

        """
        # pylint: disable=invalid-name
        return self.deepcopy().on_(image)

    def draw_on_image(self,
                      image,
                      color=(0, 255, 0), color_face=None,
                      color_lines=None, color_points=None,
                      alpha=1.0, alpha_face=None,
                      alpha_lines=None, alpha_points=None,
                      size=1, size_lines=None, size_points=None,
                      raise_if_out_of_image=False):
        """Draw all polygons onto a given image.

        Parameters
        ----------
        image : (H,W,C) ndarray
            The image onto which to draw the bounding boxes.
            This image should usually have the same shape as set in
            ``PolygonsOnImage.shape``.

        color : iterable of int, optional
            The color to use for the whole polygons.
            Must correspond to the channel layout of the image. Usually RGB.
            The values for `color_face`, `color_lines` and `color_points`
            will be derived from this color if they are set to ``None``.
            This argument has no effect if `color_face`, `color_lines`
            and `color_points` are all set anything other than ``None``.

        color_face : None or iterable of int, optional
            The color to use for the inner polygon areas (excluding perimeters).
            Must correspond to the channel layout of the image. Usually RGB.
            If this is ``None``, it will be derived from ``color * 1.0``.

        color_lines : None or iterable of int, optional
            The color to use for the lines (aka perimeters/borders) of the
            polygons. Must correspond to the channel layout of the image.
            Usually RGB. If this is ``None``, it will be derived
            from ``color * 0.5``.

        color_points : None or iterable of int, optional
            The color to use for the corner points of the polygons.
            Must correspond to the channel layout of the image. Usually RGB.
            If this is ``None``, it will be derived from ``color * 0.5``.

        alpha : float, optional
            The opacity of the whole polygons, where ``1.0`` denotes
            completely visible polygons and ``0.0`` invisible ones.
            The values for `alpha_face`, `alpha_lines` and `alpha_points`
            will be derived from this alpha value if they are set to ``None``.
            This argument has no effect if `alpha_face`, `alpha_lines`
            and `alpha_points` are all set anything other than ``None``.

        alpha_face : None or number, optional
            The opacity of the polygon's inner areas (excluding the perimeters),
            where ``1.0`` denotes completely visible inner areas and ``0.0``
            invisible ones.
            If this is ``None``, it will be derived from ``alpha * 0.5``.

        alpha_lines : None or number, optional
            The opacity of the polygon's lines (aka perimeters/borders),
            where ``1.0`` denotes completely visible perimeters and ``0.0``
            invisible ones.
            If this is ``None``, it will be derived from ``alpha * 1.0``.

        alpha_points : None or number, optional
            The opacity of the polygon's corner points, where ``1.0`` denotes
            completely visible corners and ``0.0`` invisible ones.
            Currently this is an on/off choice, i.e. only ``0.0`` or ``1.0``
            are allowed.
            If this is ``None``, it will be derived from ``alpha * 1.0``.

        size : int, optional
            Size of the polygons.
            The sizes of the line and points are derived from this value,
            unless they are set.

        size_lines : None or int, optional
            Thickness of the polygon lines (aka perimeter/border).
            If ``None``, this value is derived from `size`.

        size_points : int, optional
            The size of all corner points. If set to ``C``, each corner point
            will be drawn as a square of size ``C x C``.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if any polygon is fully
            outside of the image. If set to False, no error will be raised and
            only the parts inside the image will be drawn.

        Returns
        -------
        (H,W,C) ndarray
            Image with drawn polygons.

        """
        for poly in self.polygons:
            image = poly.draw_on_image(
                image,
                color=color,
                color_face=color_face,
                color_lines=color_lines,
                color_points=color_points,
                alpha=alpha,
                alpha_face=alpha_face,
                alpha_lines=alpha_lines,
                alpha_points=alpha_points,
                size=size,
                size_lines=size_lines,
                size_points=size_points,
                raise_if_out_of_image=raise_if_out_of_image
            )
        return image

    def remove_out_of_image_(self, fully=True, partly=False):
        """Remove all polygons that are fully/partially OOI in-place.

        'OOI' is the abbreviation for 'out of image'.

        Added in 0.4.0.

        Parameters
        ----------
        fully : bool, optional
            Whether to remove polygons that are fully outside of the image.

        partly : bool, optional
            Whether to remove polygons that are partially outside of the image.

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage
            Reduced set of polygons. Those that are fully/partially
            outside of the given image plane are removed.
            The object and its items may have been modified in-place.

        """
        self.polygons = [
            poly for poly in self.polygons
            if not poly.is_out_of_image(self.shape, fully=fully, partly=partly)
        ]
        return self

    def remove_out_of_image(self, fully=True, partly=False):
        """Remove all polygons that are fully/partially outside of an image.

        Parameters
        ----------
        fully : bool, optional
            Whether to remove polygons that are fully outside of the image.

        partly : bool, optional
            Whether to remove polygons that are partially outside of the image.

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage
            Reduced set of polygons. Those that are fully/partially
            outside of the given image plane are removed.

        """
        return self.deepcopy().remove_out_of_image_(fully, partly)

    def remove_out_of_image_fraction_(self, fraction):
        """Remove all Polys with an OOI fraction of ``>=fraction`` in-place.

        Added in 0.4.0.

        Parameters
        ----------
        fraction : number
            Minimum out of image fraction that a polygon has to have in
            order to be removed. A fraction of ``1.0`` removes only polygons
            that are ``100%`` outside of the image. A fraction of ``0.0``
            removes all polygons.

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage
            Reduced set of polygons, with those that had an out of image
            fraction greater or equal the given one removed.
            The object and its items may have been modified in-place.

        """
        return _remove_out_of_image_fraction_(self, fraction)

    def remove_out_of_image_fraction(self, fraction):
        """Remove all Polys with an out of image fraction of ``>=fraction``.

        Added in 0.4.0.

        Parameters
        ----------
        fraction : number
            Minimum out of image fraction that a polygon has to have in
            order to be removed. A fraction of ``1.0`` removes only polygons
            that are ``100%`` outside of the image. A fraction of ``0.0``
            removes all polygons.

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage
            Reduced set of polygons, with those that had an out of image
            fraction greater or equal the given one removed.

        """
        return self.copy().remove_out_of_image_fraction_(fraction)

    def clip_out_of_image_(self):
        """Clip off all parts from all polygons that are OOI in-place.

        'OOI' is the abbreviation for 'out of image'.

        .. note::

            The result can contain fewer polygons than the input did. That
            happens when a polygon is fully outside of the image plane.

        .. note::

            The result can also contain *more* polygons than the input
            did. That happens when distinct parts of a polygon are only
            connected by areas that are outside of the image plane and hence
            will be clipped off, resulting in two or more unconnected polygon
            parts that are left in the image plane.

        Added in 0.4.0.

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage
            Polygons, clipped to fall within the image dimensions.
            The count of output polygons may differ from the input count.
            The object and its items may have been modified in-place.

        """
        self.polygons = [
            poly_clipped
            for poly in self.polygons
            for poly_clipped in poly.clip_out_of_image(self.shape)]
        return self

    def clip_out_of_image(self):
        """Clip off all parts from all polygons that are outside of an image.

        .. note::

            The result can contain fewer polygons than the input did. That
            happens when a polygon is fully outside of the image plane.

        .. note::

            The result can also contain *more* polygons than the input
            did. That happens when distinct parts of a polygon are only
            connected by areas that are outside of the image plane and hence
            will be clipped off, resulting in two or more unconnected polygon
            parts that are left in the image plane.

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage
            Polygons, clipped to fall within the image dimensions.
            The count of output polygons may differ from the input count.

        """
        return self.copy().clip_out_of_image_()

    def shift_(self, x=0, y=0):
        """Move the polygons along the x/y-axis in-place.

        The origin ``(0, 0)`` is at the top left of the image.

        Added in 0.4.0.

        Parameters
        ----------
        x : number, optional
            Value to be added to all x-coordinates. Positive values shift
            towards the right images.

        y : number, optional
            Value to be added to all y-coordinates. Positive values shift
            towards the bottom images.

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage
            Shifted polygons.

        """
        for i, poly in enumerate(self.polygons):
            self.polygons[i] = poly.shift_(x=x, y=y)
        return self

    def shift(self, x=0, y=0, top=None, right=None, bottom=None, left=None):
        """Move the polygons along the x/y-axis.

        The origin ``(0, 0)`` is at the top left of the image.

        Parameters
        ----------
        x : number, optional
            Value to be added to all x-coordinates. Positive values shift
            towards the right images.

        y : number, optional
            Value to be added to all y-coordinates. Positive values shift
            towards the bottom images.

        top : None or int, optional
            Deprecated since 0.4.0.
            Amount of pixels by which to shift all objects *from* the
            top (towards the bottom).

        right : None or int, optional
            Deprecated since 0.4.0.
            Amount of pixels by which to shift all objects *from* the
            right (towads the left).

        bottom : None or int, optional
            Deprecated since 0.4.0.
            Amount of pixels by which to shift all objects *from* the
            bottom (towards the top).

        left : None or int, optional
            Deprecated since 0.4.0.
            Amount of pixels by which to shift all objects *from* the
            left (towards the right).

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage
            Shifted polygons.

        """
        x, y = _normalize_shift_args(
            x, y, top=top, right=right, bottom=bottom, left=left)
        return self.deepcopy().shift_(x=x, y=y)

    def subdivide_(self, points_per_edge):
        """Interpolate ``N`` points on each polygon.

        Added in 0.4.0.

        Parameters
        ----------
        points_per_edge : int
            Number of points to interpolate on each edge.

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage
            Subdivided polygons.

        """
        for i, poly in enumerate(self.polygons):
            self.polygons[i] = poly.subdivide_(points_per_edge)
        return self

    def subdivide(self, points_per_edge):
        """Interpolate ``N`` points on each polygon.

        Added in 0.4.0.

        Parameters
        ----------
        points_per_edge : int
            Number of points to interpolate on each edge.

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage
            Subdivided polygons.

        """
        return self.deepcopy().subdivide_(points_per_edge)

    def to_xy_array(self):
        """Convert all polygon coordinates to one array of shape ``(N,2)``.

        Added in 0.4.0.

        Returns
        -------
        (N, 2) ndarray
            Array containing all xy-coordinates of all polygons within this
            instance.

        """
        if self.empty:
            return np.zeros((0, 2), dtype=np.float32)
        return np.concatenate([poly.exterior for poly in self.polygons])

    def fill_from_xy_array_(self, xy):
        """Modify the corner coordinates of all polygons in-place.

        .. note::

            This currently expects that `xy` contains exactly as many
            coordinates as the polygons within this instance have corner
            points. Otherwise, an ``AssertionError`` will be raised.

        .. warning::

            This does not validate the new coordinates or repair the resulting
            polygons. If bad coordinates are provided, the result will be
            invalid polygons (e.g. self-intersections).

        Added in 0.4.0.

        Parameters
        ----------
        xy : (N, 2) ndarray or iterable of iterable of number
            XY-Coordinates of ``N`` corner points. ``N`` must match the
            number of corner points in all polygons within this instance.

        Returns
        -------
        PolygonsOnImage
            This instance itself, with updated coordinates.
            Note that the instance was modified in-place.

        """
        xy = np.array(xy, dtype=np.float32)

        # note that np.array([]) is (0,), not (0, 2)
        assert xy.shape[0] == 0 or (xy.ndim == 2 and xy.shape[-1] == 2), (  # pylint: disable=unsubscriptable-object
            "Expected input array to have shape (N,2), "
            "got shape %s." % (xy.shape,))

        counter = 0
        for poly in self.polygons:
            nb_points = len(poly.exterior)
            assert counter + nb_points <= len(xy), (
                "Received fewer points than there are corner points in the "
                "exteriors of all polygons. Got %d points, expected %d." % (
                    len(xy), sum([len(p.exterior) for p in self.polygons])))

            poly.exterior[:, ...] = xy[counter:counter+nb_points]
            counter += nb_points

        assert counter == len(xy), (
            "Expected to get exactly as many xy-coordinates as there are "
            "points in the exteriors of all polygons within this instance. "
            "Got %d points, could only assign %d points." % (
                len(xy), counter,))

        return self

    def to_keypoints_on_image(self):
        """Convert the polygons to one ``KeypointsOnImage`` instance.

        Added in 0.4.0.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            A keypoints instance containing ``N`` coordinates for a total
            of ``N`` points in all exteriors of the polygons within this
            container. Order matches the order in ``polygons``.

        """
        from . import KeypointsOnImage
        if self.empty:
            return KeypointsOnImage([], shape=self.shape)
        exteriors = np.concatenate(
            [poly.exterior for poly in self.polygons],
            axis=0)
        return KeypointsOnImage.from_xy_array(exteriors, shape=self.shape)

    def invert_to_keypoints_on_image_(self, kpsoi):
        """Invert the output of ``to_keypoints_on_image()`` in-place.

        This function writes in-place into this ``PolygonsOnImage``
        instance.

        Added in 0.4.0.

        Parameters
        ----------
        kpsoi : imgaug.augmentables.kps.KeypointsOnImages
            Keypoints to convert back to polygons, i.e. the outputs
            of ``to_keypoints_on_image()``.

        Returns
        -------
        PolygonsOnImage
            Polygons container with updated coordinates.
            Note that the instance is also updated in-place.

        """
        polys = self.polygons
        exteriors = [poly.exterior for poly in polys]
        nb_points_exp = sum([len(exterior) for exterior in exteriors])
        assert len(kpsoi.keypoints) == nb_points_exp, (
            "Expected %d coordinates, got %d." % (
                nb_points_exp, len(kpsoi.keypoints)))

        xy_arr = kpsoi.to_xy_array()

        counter = 0
        for poly in polys:
            exterior = poly.exterior
            exterior[:, :] = xy_arr[counter:counter+len(exterior), :]
            counter += len(exterior)
        self.shape = kpsoi.shape
        return self

    def copy(self, polygons=None, shape=None):
        """Create a shallow copy of this object.

        Parameters
        ----------
        polygons : None or list of imgaug.augmentables.polys.Polygons, optional
            List of polygons on the image.
            If not ``None``, then the ``polygons`` attribute of the copied
            object will be set to this value.

        shape : None or tuple of int or ndarray, optional
            The shape of the image on which the objects are placed.
            Either an image with shape ``(H,W,[C])`` or a tuple denoting
            such an image shape.
            If not ``None``, then the ``shape`` attribute of the copied object
            will be set to this value.

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage
            Shallow copy.

        """
        if polygons is None:
            polygons = self.polygons[:]
        if shape is None:
            # use tuple() here in case the shape was provided as a list
            shape = tuple(self.shape)

        return PolygonsOnImage(polygons, shape)

    def deepcopy(self, polygons=None, shape=None):
        """Create a deep copy of this object.

        Parameters
        ----------
        polygons : None or list of imgaug.augmentables.polys.Polygons, optional
            List of polygons on the image.
            If not ``None``, then the ``polygons`` attribute of the copied
            object will be set to this value.

        shape : None or tuple of int or ndarray, optional
            The shape of the image on which the objects are placed.
            Either an image with shape ``(H,W,[C])`` or a tuple denoting
            such an image shape.
            If not ``None``, then the ``shape`` attribute of the copied object
            will be set to this value.

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage
            Deep copy.

        """
        # Manual copy is far faster than deepcopy, so use manual copy here.
        if polygons is None:
            polygons = [poly.deepcopy() for poly in self.polygons]
        if shape is None:
            # use tuple() here in case the shape was provided as a list
            shape = tuple(self.shape)

        return PolygonsOnImage(polygons, shape)

    def __getitem__(self, indices):
        """Get the polygon(s) with given indices.

        Added in 0.4.0.

        Returns
        -------
        list of imgaug.augmentables.polys.Polygon
            Polygon(s) with given indices.

        """
        return self.polygons[indices]

    def __iter__(self):
        """Iterate over the polygons in this container.

        Added in 0.4.0.

        Yields
        ------
        Polygon
            A polygon in this container.
            The order is identical to the order in the polygon list
            provided upon class initialization.

        """
        return iter(self.polygons)

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
        return "PolygonsOnImage(%s, shape=%s)" % (
            str(self.polygons), self.shape)


def _convert_points_to_shapely_line_string(points, closed=False,
                                           interpolate=0):
    # load shapely lazily, which makes the dependency more optional
    import shapely.geometry

    if len(points) <= 1:
        raise Exception(
            "Conversion to shapely line string requires at least two points, "
            "but points input contains only %d points." % (len(points),))

    points_tuples = [(point[0], point[1]) for point in points]

    # interpolate points between each consecutive pair of points
    if interpolate > 0:
        points_tuples = interpolate_points(points_tuples, interpolate)

    # close if requested and not yet closed
    # used here intentionally `points` instead of `points_tuples`
    if closed and len(points) > 1:
        points_tuples.append(points_tuples[0])

    return shapely.geometry.LineString(points_tuples)


class _ConcavePolygonRecoverer(object):
    def __init__(self, threshold_duplicate_points=1e-4, noise_strength=1e-4,
                 oversampling=0.01, max_segment_difference=1e-4):
        self.threshold_duplicate_points = threshold_duplicate_points
        self.noise_strength = noise_strength
        self.oversampling = oversampling
        self.max_segment_difference = max_segment_difference

        # this limits the maximum amount of points after oversampling, i.e.
        # if N points are input into oversampling, then M oversampled points
        # are generated such that N+M <= this value
        self.oversample_up_to_n_points_max = 75

        # ----
        # parameters for _fit_best_valid_polygon()
        # ----
        # how many changes may be done max to the initial (convex hull) polygon
        # before simply returning the result
        self.fit_n_changes_max = 100
        # for how many iterations the optimization loop may run max
        # before simply returning the result
        self.fit_n_iters_max = 3
        # how far (wrt. to their position in the input list) two points may be
        # apart max to consider adding an edge between them (in the first loop
        # iteration and the ones after that)
        self.fit_max_dist_first_iter = 1
        self.fit_max_dist_other_iters = 2
        # The fit loop first generates candidate edges and then modifies the
        # polygon based on these candidates. This limits the maximum amount
        # of considered candidates. If the number is less than the possible
        # number of candidates, they are randomly subsampled. Values beyond
        # 100 significantly increase runtime (for polygons that reach that
        # number).
        self.fit_n_candidates_before_sort_max = 100

        # If abs(x) or abs(y) of any coordinate of a polygon is beyond this
        # value, no intersection points will be computed anymore. That is done,
        # because the underlying library to find these points uses float
        # values as keys and may therefore start to encounter inaccuracies
        # leading to exceptions within that library.
        self.limit_coords_values_for_inter_search = 50000

        # Rounding of coordinates to use before feeding them into the
        # library to search for intersection points. Note that the library
        # was set to also use a corresponding eps of 1e-4.
        self.decimals = 4

    def recover_from(self, new_exterior, old_polygon, random_state=0):
        assert isinstance(new_exterior, list) or (
            ia.is_np_array(new_exterior)
            and new_exterior.ndim == 2
            and new_exterior.shape[1] == 2), (
                "Expected exterior as list or (N,2) ndarray, got type %s." % (
                    type(new_exterior),))
        assert len(new_exterior) >= 3, \
            "Cannot recover a concave polygon from less than three points."

        # create Polygon instance, if it is already valid then just return
        # immediately
        polygon = old_polygon.deepcopy(exterior=new_exterior)
        if polygon.is_valid:
            return polygon

        random_state = iarandom.RNG(random_state)
        rss = random_state.duplicate(3)

        # remove consecutive duplicate points
        new_exterior = self._remove_consecutive_duplicate_points(new_exterior)

        # check that points are not all identical or on a line
        new_exterior = self._fix_polygon_is_line(new_exterior, rss[0])

        # jitter duplicate points
        new_exterior = self._jitter_duplicate_points(new_exterior, rss[1])

        # generate intersection points
        segment_add_points = self._generate_intersection_points(
            new_exterior, decimals=self.decimals)

        # oversample points around intersections
        if self.oversampling is not None and self.oversampling > 0:
            segment_add_points = self._oversample_intersection_points(
                new_exterior, segment_add_points)

        # integrate new points into exterior
        new_exterior_inter = self._insert_intersection_points(
            new_exterior, segment_add_points)

        # find best fit polygon, starting from convext polygon
        new_exterior_concave_ids = self._fit_best_valid_polygon(
            new_exterior_inter, rss[2])
        new_exterior_concave = [
            new_exterior_inter[idx] for idx in new_exterior_concave_ids]

        # TODO return new_exterior_concave here instead of polygon, leave it to
        #      caller to decide what to do with it
        return old_polygon.deepcopy(exterior=new_exterior_concave)

    def _remove_consecutive_duplicate_points(self, points):
        result = []
        for point in points:
            if result:
                dist = np.linalg.norm(
                    np.float32(point) - np.float32(result[-1]))
                is_same = (dist < self.threshold_duplicate_points)
                if not is_same:
                    result.append(point)
            else:
                result.append(point)
        if len(result) >= 2:
            dist = np.linalg.norm(
                np.float32(result[0]) - np.float32(result[-1]))
            is_same = (dist < self.threshold_duplicate_points)
            result = result[0:-1] if is_same else result
        return result

    # fix polygons for which all points are on a line
    def _fix_polygon_is_line(self, exterior, random_state):
        assert len(exterior) >= 3, (
            "Can only fix line-like polygons with an exterior containing at "
            "least 3 points. Got one with %d points." % (len(exterior),))
        noise_strength = self.noise_strength
        while self._is_polygon_line(exterior):
            noise = random_state.uniform(
                -noise_strength, noise_strength, size=(len(exterior), 2)
            ).astype(np.float32)
            exterior = [(point[0] + noise_i[0], point[1] + noise_i[1])
                        for point, noise_i in zip(exterior, noise)]
            noise_strength = noise_strength * 10
            assert noise_strength > 0, (
                "Expected noise strength to be >0, got %.4f." % (
                    noise_strength,))
        return exterior

    @classmethod
    def _is_polygon_line(cls, exterior):
        vec_down = np.float32([0, 1])
        point1 = exterior[0]
        angles = set()
        for point2 in exterior[1:]:
            vec = np.float32(point2) - np.float32(point1)
            angle = ia.angle_between_vectors(vec_down, vec)
            angles.add(int(angle * 1000))
        return len(angles) <= 1

    def _jitter_duplicate_points(self, exterior, random_state):
        def _find_duplicates(exterior_with_duplicates):
            points_map = collections.defaultdict(list)

            for i, point in enumerate(exterior_with_duplicates):
                # we use 10/x here to be a bit more lenient, the precise
                # distance test is further below
                x = int(np.round(point[0]
                                 * ((1/10) / self.threshold_duplicate_points)))
                y = int(np.round(point[1]
                                 * ((1/10) / self.threshold_duplicate_points)))
                for direction0 in [-1, 0, 1]:
                    for direction1 in [-1, 0, 1]:
                        points_map[(x+direction0, y+direction1)].append(i)

            duplicates = [False] * len(exterior_with_duplicates)
            for key in points_map:
                candidates = points_map[key]
                for i, p0_idx in enumerate(candidates):
                    p0_idx = candidates[i]
                    point0 = exterior_with_duplicates[p0_idx]
                    if duplicates[p0_idx]:
                        continue

                    for j in range(i+1, len(candidates)):
                        p1_idx = candidates[j]
                        point1 = exterior_with_duplicates[p1_idx]
                        if duplicates[p1_idx]:
                            continue

                        dist = np.sqrt(
                            (point0[0] - point1[0])**2
                            + (point0[1] - point1[1])**2)
                        if dist < self.threshold_duplicate_points:
                            duplicates[p1_idx] = True

            return duplicates

        noise_strength = self.noise_strength
        assert noise_strength > 0, (
            "Expected noise strength to be >0, got %.4f." % (noise_strength,))
        exterior = exterior[:]
        converged = False
        while not converged:
            duplicates = _find_duplicates(exterior)
            if any(duplicates):
                noise = random_state.uniform(
                    -self.noise_strength,
                    self.noise_strength,
                    size=(len(exterior), 2)
                ).astype(np.float32)

                for i, is_duplicate in enumerate(duplicates):
                    if is_duplicate:
                        exterior[i] = (
                            exterior[i][0] + noise[i][0],
                            exterior[i][1] + noise[i][1])

                noise_strength *= 10
            else:
                converged = True

        return exterior

    # TODO remove?
    @classmethod
    def _calculate_circumference(cls, points):
        assert len(points) >= 3, (
            "Need at least 3 points on the exterior to compute the "
            "circumference. Got %d." % (len(points),))
        points = np.array(points, dtype=np.float32)
        points_matrix = np.zeros((len(points), 4), dtype=np.float32)
        points_matrix[:, 0:2] = points
        points_matrix[0:-1, 2:4] = points_matrix[1:, 0:2]
        points_matrix[-1, 2:4] = points_matrix[0, 0:2]
        distances = np.linalg.norm(
            points_matrix[:, 0:2] - points_matrix[:, 2:4], axis=1)
        return np.sum(distances)

    def _generate_intersection_points(self, exterior,
                                      one_point_per_intersection=True,
                                      decimals=4):
        # pylint: disable=broad-except
        largest_value = np.max(np.abs(np.array(exterior, dtype=np.float32)))
        too_large_values = (
            largest_value > self.limit_coords_values_for_inter_search)
        if too_large_values:
            ia.warn(
                "Encountered during polygon repair a polygon with extremely "
                "large coordinate values beyond %d. Will skip intersection "
                "point computation for that polygon. This avoids exceptions "
                "and is -- due to the extreme distortion -- likely pointless "
                "anyways (i.e. the polygon is already broken beyond repair). "
                "Try using weaker augmentation parameters to avoid such "
                "large coordinate values." % (
                    self.limit_coords_values_for_inter_search,)
            )
            return [[] for _ in range(len(exterior))]

        if ia.is_np_array(exterior):
            exterior = list(exterior)
        assert isinstance(exterior, list), (
            "Expected 'exterior' to be a list or a ndarray. "
            "Got type %s." % (type(exterior),))
        assert all([len(point) == 2 for point in exterior]), (
            "Expected 'exterior' to contain (x,y) coordinate pairs. "
            "Got lengths %s." % (
                ", ".join([str(len(point)) for point in exterior])))
        if len(exterior) <= 0:
            return []

        # use (*[i][0], *[i][1]) formulation here instead of just *[i],
        # because this way we convert numpy arrays to tuples of floats, which
        # is required by isect_segments_include_segments
        segments = [
            (
                (
                    np.round(float(exterior[i][0]), decimals),
                    np.round(float(exterior[i][1]), decimals)
                ),
                (
                    np.round(float(exterior[(i + 1) % len(exterior)][0]),
                             decimals),
                    np.round(float(exterior[(i + 1) % len(exterior)][1]),
                             decimals)
                )
            )
            for i in range(len(exterior))
        ]

        # returns [(point, [(segment_p0, segment_p1), ..]), ...]
        from imgaug.external.poly_point_isect_py2py3 import (
            isect_segments_include_segments)

        try:
            intersections = isect_segments_include_segments(segments)
        except Exception as exc:
            # Exceptions in the segment intersection search can at least
            # happen due to large float coords (the library uses
            # floats as indices, which is bound to cause inaccuracies).
            # Usually such exceptions should not appear, as too large
            # coordinate values are already caught at the start of this
            # function. For the case that there are more errors, this block
            # will prevent a full crash.
            ia.warn(
                "Encountered exception %s during polygon repair in segment "
                "intersection computation. Will skip that step." % (
                    str(exc),))
            traceback.print_exc()
            return [[] for _ in range(len(exterior))]

        # estimate to which segment the found intersection points belong
        segments_add_points = [[] for _ in range(len(segments))]
        for point, associated_segments in intersections:
            # the intersection point may be associated with multiple segments,
            # but we only want to add it once, so pick the first segment
            if one_point_per_intersection:
                associated_segments = [associated_segments[0]]

            for seg_inter_p0, seg_inter_p1 in associated_segments:
                diffs = []
                dists = []
                for seg_p0, seg_p1 in segments:
                    dist_p0p0 = np.linalg.norm(seg_p0 - np.array(seg_inter_p0))
                    dist_p1p1 = np.linalg.norm(seg_p1 - np.array(seg_inter_p1))
                    dist_p0p1 = np.linalg.norm(seg_p0 - np.array(seg_inter_p1))
                    dist_p1p0 = np.linalg.norm(seg_p1 - np.array(seg_inter_p0))
                    diff = min(dist_p0p0 + dist_p1p1, dist_p0p1 + dist_p1p0)
                    diffs.append(diff)
                    dists.append(np.linalg.norm(
                        (seg_p0[0] - point[0], seg_p0[1] - point[1])
                    ))

                min_diff = np.min(diffs)
                if min_diff < self.max_segment_difference:
                    idx = int(np.argmin(diffs))
                    segments_add_points[idx].append((point, dists[idx]))
                else:
                    ia.warn(
                        "Couldn't find fitting segment in "
                        "_generate_intersection_points(). Ignoring "
                        "intersection point.")

        # sort intersection points by their distance to point 0 in each segment
        # (clockwise ordering, this does something only for segments with
        # >=2 intersection points)
        segment_add_points_sorted = []
        for idx in range(len(segments_add_points)):
            points = [t[0] for t in segments_add_points[idx]]
            dists = [t[1] for t in segments_add_points[idx]]
            if len(points) < 2:
                segment_add_points_sorted.append(points)
            else:
                both = sorted(zip(points, dists), key=lambda t: t[1])
                # keep points, drop distances
                segment_add_points_sorted.append([a for a, _b in both])
        return segment_add_points_sorted

    def _oversample_intersection_points(self, exterior, segment_add_points):
        # segment_add_points must be sorted

        if self.oversampling is None or self.oversampling <= 0:
            return segment_add_points

        segment_add_points_sorted_overs = [
            [] for _ in range(len(segment_add_points))]

        n_points = len(exterior)
        for i, last in enumerate(exterior):
            for j, p_inter in enumerate(segment_add_points[i]):
                direction = (p_inter[0] - last[0], p_inter[1] - last[1])

                if j == 0:
                    # previous point was non-intersection, place 1 new point
                    oversample = [1.0 - self.oversampling]
                else:
                    # previous point was intersection, place 2 new points
                    oversample = [self.oversampling, 1.0 - self.oversampling]

                for dist in oversample:
                    point_over = (last[0] + dist * direction[0],
                                  last[1] + dist * direction[1])
                    segment_add_points_sorted_overs[i].append(point_over)
                segment_add_points_sorted_overs[i].append(p_inter)
                last = p_inter

                is_last_in_group = (j == len(segment_add_points[i]) - 1)
                if is_last_in_group:
                    # previous point was oversampled, next point is
                    # non-intersection, place 1 new point between the two
                    exterior_point = exterior[(i + 1) % len(exterior)]
                    direction = (exterior_point[0] - last[0],
                                 exterior_point[1] - last[1])
                    segment_add_points_sorted_overs[i].append(
                        (last[0] + self.oversampling * direction[0],
                         last[1] + self.oversampling * direction[1])
                    )
                    last = segment_add_points_sorted_overs[i][-1]

                n_points += len(segment_add_points_sorted_overs[i])
                if n_points > self.oversample_up_to_n_points_max:
                    return segment_add_points_sorted_overs

        return segment_add_points_sorted_overs

    @classmethod
    def _insert_intersection_points(cls, exterior, segment_add_points):
        # segment_add_points must be sorted

        assert len(exterior) == len(segment_add_points), (
            "Expected one entry in 'segment_add_points' for every point in "
            "the exterior. Got %d (segment_add_points) and %d (exterior) "
            "entries instead." % (len(segment_add_points), len(exterior)))
        exterior_interp = []
        for i, point0 in enumerate(exterior):
            point0 = exterior[i]
            exterior_interp.append(point0)
            for p_inter in segment_add_points[i]:
                exterior_interp.append(p_inter)
        return exterior_interp

    def _fit_best_valid_polygon(self, points, random_state):
        if len(points) < 2:
            return None

        def _compute_distance_point_to_line(point, line_start, line_end):
            x_diff = line_end[0] - line_start[0]
            y_diff = line_end[1] - line_start[1]
            num = abs(
                y_diff*point[0] - x_diff*point[1]
                + line_end[0]*line_start[1] - line_end[1]*line_start[0]
            )
            den = np.sqrt(y_diff**2 + x_diff**2)
            if den == 0:
                return np.sqrt(
                    (point[0] - line_start[0])**2
                    + (point[1] - line_start[1])**2)
            return num / den

        poly = Polygon(points)
        if poly.is_valid:
            return sm.xrange(len(points))

        hull = scipy.spatial.ConvexHull(points)
        points_kept = list(hull.vertices)
        points_left = [i for i in range(len(points)) if i not in points_kept]

        iteration = 0
        n_changes = 0
        converged = False
        while not converged:
            candidates = []

            # estimate distance metrics for points-segment pairs:
            #  (1) distance (in vertices) between point and segment-start-point
            #      in original input point chain
            #  (2) euclidean distance between point and segment/line
            # TODO this can be done more efficiently by caching the values and
            #      only computing distances to segments that have changed in
            #      the last iteration
            # TODO these distances are not really the best metrics here.
            #      Something like IoU between new and old (invalid) polygon
            #      would be better, but can probably only be computed for
            #      pairs of valid polygons. Maybe something based on pointwise
            #      distances, where the points are sampled on the edges (not
            #      edge vertices themselves). Maybe something based on drawing
            #      the perimeter on images or based on distance maps.
            point_kept_idx_to_pos = {
                point_idx: i for i, point_idx in enumerate(points_kept)}

            # generate all possible combinations from <points_kept> and
            # <points_left>
            combos = np.transpose([
                np.tile(
                    np.int32(points_left), len(np.int32(points_kept))
                ),
                np.repeat(
                    np.int32(points_kept), len(np.int32(points_left))
                )
            ])
            combos = np.concatenate(
                (combos, np.zeros((combos.shape[0], 3), dtype=np.int32)),
                axis=1)

            # copy columns 0, 1 into 2, 3 so that 2 is always the lower value
            mask = combos[:, 0] < combos[:, 1]
            combos[:, 2:4] = combos[:, 0:2]
            combos[mask, 2] = combos[mask, 1]
            combos[mask, 3] = combos[mask, 0]

            # distance (in indices) between each pair of <point_kept> and
            # <point_left>
            combos[:, 4] = np.minimum(
                combos[:, 3] - combos[:, 2],
                len(points) - combos[:, 3] + combos[:, 2]
            )

            # limit candidates
            max_dist = self.fit_max_dist_other_iters
            if iteration > 0:
                max_dist = self.fit_max_dist_first_iter
            candidate_rows = combos[combos[:, 4] <= max_dist]
            do_limit = (
                self.fit_n_candidates_before_sort_max is not None
                and len(candidate_rows) > self.fit_n_candidates_before_sort_max)
            if do_limit:
                random_state.shuffle(candidate_rows)
                candidate_rows = candidate_rows[
                    0:self.fit_n_candidates_before_sort_max]

            for row in candidate_rows:
                point_left_idx = row[0]
                point_kept_idx = row[1]
                in_points_kept_pos = point_kept_idx_to_pos[point_kept_idx]
                segment_start_idx = point_kept_idx
                segment_end_idx = points_kept[
                    (in_points_kept_pos+1) % len(points_kept)]
                segment_start = points[segment_start_idx]
                segment_end = points[segment_end_idx]
                if iteration == 0:
                    dist_eucl = 0
                else:
                    dist_eucl = _compute_distance_point_to_line(
                        points[point_left_idx], segment_start, segment_end)
                candidates.append(
                    (point_left_idx, point_kept_idx, row[4], dist_eucl))

            # Sort computed distances first by minimal vertex-distance (see
            # above, metric 1) (ASC), then by euclidean distance
            # (metric 2) (ASC).
            candidate_ids = np.arange(len(candidates))
            candidate_ids = sorted(
                candidate_ids,
                key=lambda idx: (candidates[idx][2], candidates[idx][3]))
            if self.fit_n_changes_max is not None:
                candidate_ids = candidate_ids[:self.fit_n_changes_max]

            # Iterate over point-segment pairs in sorted order. For each such
            # candidate: Add the point to the already collected points,
            # create a polygon from that and check if the polygon is valid.
            # If it is, add the point to the output list and recalculate
            # distance metrics. If it isn't valid, proceed with the next
            # candidate until no more candidates are left.
            #
            # small change: this now no longer breaks upon the first found
            # point that leads to a valid polygon, but checks all candidates
            # instead
            is_valid = False
            done = set()
            for candidate_idx in candidate_ids:
                point_left_idx = candidates[candidate_idx][0]
                point_kept_idx = candidates[candidate_idx][1]
                if (point_left_idx, point_kept_idx) not in done:
                    in_points_kept_idx = [
                        i
                        for i, point_idx
                        in enumerate(points_kept)
                        if point_idx == point_kept_idx
                    ][0]
                    points_kept_hypothesis = points_kept[:]
                    points_kept_hypothesis.insert(
                        in_points_kept_idx+1,
                        point_left_idx)
                    poly_hypothesis = Polygon([
                        points[idx] for idx in points_kept_hypothesis])
                    if poly_hypothesis.is_valid:
                        is_valid = True
                        points_kept = points_kept_hypothesis
                        points_left = [point_idx
                                       for point_idx
                                       in points_left
                                       if point_idx != point_left_idx]
                        n_changes += 1
                        if n_changes >= self.fit_n_changes_max:
                            return points_kept
                    done.add((point_left_idx, point_kept_idx))
                    done.add((point_kept_idx, point_left_idx))

            # none of the left points could be used to create a valid polygon?
            # (this automatically covers the case of no points being left)
            if not is_valid and iteration > 0:
                converged = True

            iteration += 1
            has_reached_iters_max = (
                self.fit_n_iters_max is not None
                and iteration > self.fit_n_iters_max)
            if has_reached_iters_max:
                break

        return points_kept


# TODO remove this? was previously only used by Polygon.clip_*(), but that
#      doesn't use it anymore
class MultiPolygon(object):
    """
    Class that represents several polygons.

    Parameters
    ----------
    geoms : list of imgaug.augmentables.polys.Polygon
        List of the polygons.

    """
    def __init__(self, geoms):
        """Create a new MultiPolygon instance."""
        assert (
            len(geoms) == 0
            or all([isinstance(el, Polygon) for el in geoms])), (
                "Expected 'geoms' to a list of Polygon instances. "
                "Got types %s." % (", ".join([str(el) for el in geoms])))
        self.geoms = geoms

    @staticmethod
    def from_shapely(geometry, label=None):
        """Create a MultiPolygon from a shapely object.

        This also creates all necessary ``Polygon`` s contained in this
        ``MultiPolygon``.

        Parameters
        ----------
        geometry : shapely.geometry.MultiPolygon or shapely.geometry.Polygon or shapely.geometry.collection.GeometryCollection
            The object to convert to a MultiPolygon.

        label : None or str, optional
            A label assigned to all Polygons within the MultiPolygon.

        Returns
        -------
        imgaug.augmentables.polys.MultiPolygon
            The derived MultiPolygon.

        """
        # load shapely lazily, which makes the dependency more optional
        import shapely.geometry

        if isinstance(geometry, shapely.geometry.MultiPolygon):
            return MultiPolygon([
                Polygon.from_shapely(poly, label=label)
                for poly
                in geometry.geoms])
        if isinstance(geometry, shapely.geometry.Polygon):
            return MultiPolygon([Polygon.from_shapely(geometry, label=label)])
        if isinstance(geometry,
                      shapely.geometry.collection.GeometryCollection):
            assert all([
                isinstance(poly, shapely.geometry.Polygon)
                for poly
                in geometry.geoms]), (
                    "Expected the geometry collection to only contain shapely "
                    "polygons. Got types %s." % (
                        ", ".join([str(type(v)) for v in geometry.geoms])))
            return MultiPolygon([
                Polygon.from_shapely(poly, label=label)
                for poly
                in geometry.geoms])

        raise Exception(
            "Unknown datatype '%s'. Expected shapely.geometry.Polygon or "
            "shapely.geometry.MultiPolygon or "
            "shapely.geometry.collections.GeometryCollection." % (
                type(geometry),))
