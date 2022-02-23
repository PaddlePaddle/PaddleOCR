from typing import List, Union, Dict, Dict, Any, Optional
from abc import ABC, abstractmethod
from collections.abc import Iterable, MutableSequence
from copy import copy, deepcopy
from inspect import getmembers, isfunction
import warnings
import functools

import numpy as np
import pandas as pd
from PIL import Image
from cv2 import getPerspectiveTransform as _getPerspectiveTransform
from cv2 import warpPerspective as _warpPerspective

__all__ = ["Interval", "Rectangle", "Quadrilateral", "TextBlock", "Layout"]


def _cvt_coordinates_to_points(coords):

    x_1, y_1, x_2, y_2 = coords
    return np.array(
        [
            [x_1, y_1],  # Top Left
            [x_2, y_1],  # Top Right
            [x_2, y_2],  # Bottom Right
            [x_1, y_2],  # Bottom Left
        ]
    )


def _cvt_points_to_coordinates(points):
    x_1 = points[:, 0].min()
    y_1 = points[:, 1].min()
    x_2 = points[:, 0].max()
    y_2 = points[:, 1].max()
    return (x_1, y_1, x_2, y_2)


def _perspective_transformation(M, points, is_inv=False):

    if is_inv:
        M = np.linalg.inv(M)

    src_mid = np.hstack([points, np.ones((points.shape[0], 1))]).T  # 3x4
    dst_mid = np.matmul(M, src_mid)

    dst = (dst_mid / dst_mid[-1]).T[:, :2]  # 4x2

    return dst


def _vertice_in_polygon(vertice, polygon_points):
    # The polygon_points are ordered clockwise

    # The implementation is based on the algorithm from
    # https://demonstrations.wolfram.com/AnEfficientTestForAPointToBeInAConvexPolygon/

    points = polygon_points - vertice  # shift the coordinates origin to the vertice
    edges = np.append(points, points[0:1, :], axis=0)
    return all([np.linalg.det([e1, e2]) >= 0 for e1, e2 in zip(edges, edges[1:])])
    # If the points are ordered clockwise, the det should <=0


def _polygon_area(xs, ys):
    """Calculate the area of polygons using
    `Shoelace Formula <https://en.wikipedia.org/wiki/Shoelace_formula>`_.

    Args:
        xs (`np.ndarray`): The x coordinates of the points
        ys (`np.ndarray`): The y coordinates of the points
    """

    # Refer to: https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    # The formula is equivalent to the original one indicated in the wikipedia
    # page.

    return 0.5 * np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))


def mixin_textblock_meta(func):
    @functools.wraps(func)
    def wrap(self, *args, **kwargs):
        out = func(self, *args, **kwargs)
        if isinstance(out, BaseCoordElement):
            self = copy(self)
            self.block = out
            return self

    return wrap


def inherit_docstrings(cls=None, *, base_class=None):

    # Refer to https://stackoverflow.com/a/17393254
    if cls is None:
        return functools.partial(inherit_docstrings, base_class=base_class)

    for name, func in getmembers(cls, isfunction):
        if func.__doc__:
            continue
        if base_class == None:
            for parent in cls.__mro__[1:]:
                if hasattr(parent, name):
                    func.__doc__ = getattr(parent, name).__doc__
                    break
        else:
            if hasattr(base_class, name):
                func.__doc__ = getattr(base_class, name).__doc__

    return cls


def support_textblock(func):
    @functools.wraps(func)
    def wrap(self, other, *args, **kwargs):
        if isinstance(other, TextBlock):
            other = other.block
        out = func(self, other, *args, **kwargs)
        return out

    return wrap


class NotSupportedShapeError(Exception):
    """For now (v0.2), if the created shape might be a polygon (shapes with more than 4 vertices),
    layoutparser will raise NotSupportedShapeError. It is expected to be fixed in the future versions.
    See
    :ref:`shape_operations:problems-related-to-the-quadrilateral-class`.
    """


class InvalidShapeError(Exception):
    """For shape operations like intersection of union, lp will raise the InvalidShapeError when
    invalid shapes are created (e.g., intersecting a rectangle and an interval).
    """


class BaseLayoutElement:
    def set(self, inplace=False, **kwargs):

        obj = self if inplace else copy(self)
        var_dict = vars(obj)
        for key, val in kwargs.items():
            if key in var_dict:
                var_dict[key] = val
            elif f"_{key}" in var_dict:
                var_dict[f"_{key}"] = val
            else:
                raise ValueError(f"Unknown attribute name: {key}")

        return obj

    def __repr__(self):

        info_str = ", ".join([f"{key}={val}" for key, val in vars(self).items()])
        return f"{self.__class__.__name__}({info_str})"

    def __eq__(self, other):

        if other.__class__ is not self.__class__:
            return False

        return vars(self) == vars(other)


class BaseCoordElement(ABC, BaseLayoutElement):
    @property
    @abstractmethod
    def _name(self) -> str:
        """The name of the class"""
        pass

    @property
    @abstractmethod
    def _features(self) -> List[str]:
        """A list of features names used for initializing the class object"""
        pass

    #######################################################################
    #########################  Layout Properties  #########################
    #######################################################################

    @property
    @abstractmethod
    def width(self):
        pass

    @property
    @abstractmethod
    def height(self):
        pass

    @property
    @abstractmethod
    def coordinates(self):
        pass

    @property
    @abstractmethod
    def points(self):
        pass

    @property
    @abstractmethod
    def area(self):
        pass

    #######################################################################
    ###   Geometric Relations (relative to, condition on, and is in)    ###
    #######################################################################

    @abstractmethod
    def condition_on(self, other):
        """
        Given the current element in relative coordinates to another element which is in absolute coordinates,
        generate a new element of the current element in absolute coordinates.

        Args:
            other (:obj:`BaseCoordElement`):
                The other layout element involved in the geometric operations.

        Raises:
            Exception: Raise error when the input type of the other element is invalid.

        Returns:
            :obj:`BaseCoordElement`:
                The BaseCoordElement object of the original element in the absolute coordinate system.
        """

        pass

    @abstractmethod
    def relative_to(self, other):
        """
        Given the current element and another element both in absolute coordinates,
        generate a new element of the current element in relative coordinates to the other element.

        Args:
            other (:obj:`BaseCoordElement`): The other layout element involved in the geometric operations.

        Raises:
            Exception: Raise error when the input type of the other element is invalid.

        Returns:
            :obj:`BaseCoordElement`:
                The BaseCoordElement object of the original element in the relative coordinate system.
        """

        pass

    @abstractmethod
    def is_in(self, other, soft_margin={}, center=False):
        """
        Identify whether the current element is within another element.

        Args:
            other (:obj:`BaseCoordElement`):
                The other layout element involved in the geometric operations.
            soft_margin (:obj:`dict`, `optional`, defaults to `{}`):
                Enlarge the other element with wider margins to relax the restrictions.
            center (:obj:`bool`, `optional`, defaults to `False`):
                The toggle to determine whether the center (instead of the four corners)
                of the current element is in the other element.

        Returns:
            :obj:`bool`: Returns `True` if the current element is in the other element and `False` if not.
        """

        pass

    #######################################################################
    ################# Shape Operations (intersect, union)  ################
    #######################################################################

    @abstractmethod
    def intersect(self, other: "BaseCoordElement", strict: bool = True):
        """Intersect the current shape with the other object, with operations defined in
        :doc:`../notes/shape_operations`.
        """

    @abstractmethod
    def union(self, other: "BaseCoordElement", strict: bool = True):
        """Union the current shape with the other object, with operations defined in
        :doc:`../notes/shape_operations`.
        """

    #######################################################################
    ############### Geometric Operations (pad, shift, scale) ##############
    #######################################################################

    @abstractmethod
    def pad(self, left=0, right=0, top=0, bottom=0, safe_mode=True):
        """Pad the layout element on the four sides of the polygon with the user-defined pixels. If
        safe_mode is set to True, the function will cut off the excess padding that falls on the negative
        side of the coordinates.

        Args:
            left (:obj:`int`, `optional`, defaults to 0): The number of pixels to pad on the upper side of the polygon.
            right (:obj:`int`, `optional`, defaults to 0): The number of pixels to pad on the lower side of the polygon.
            top (:obj:`int`, `optional`, defaults to 0): The number of pixels to pad on the left side of the polygon.
            bottom (:obj:`int`, `optional`, defaults to 0): The number of pixels to pad on the right side of the polygon.
            safe_mode (:obj:`bool`, `optional`, defaults to True): A bool value to toggle the safe_mode.

        Returns:
            :obj:`BaseCoordElement`: The padded BaseCoordElement object.
        """

        pass

    @abstractmethod
    def shift(self, shift_distance=0):
        """
        Shift the layout element by user specified amounts on x and y axis respectively. If shift_distance is one
        numeric value, the element will by shifted by the same specified amount on both x and y axis.

        Args:
            shift_distance (:obj:`numeric` or :obj:`Tuple(numeric)` or :obj:`List[numeric]`):
                The number of pixels used to shift the element.

        Returns:
            :obj:`BaseCoordElement`: The shifted BaseCoordElement of the same shape-specific class.
        """

        pass

    @abstractmethod
    def scale(self, scale_factor=1):
        """
        Scale the layout element by a user specified amount on x and y axis respectively. If scale_factor is one
        numeric value, the element will by scaled by the same specified amount on both x and y axis.

        Args:
            scale_factor (:obj:`numeric` or :obj:`Tuple(numeric)` or :obj:`List[numeric]`): The amount for downscaling or upscaling the element.

        Returns:
            :obj:`BaseCoordElement`: The scaled BaseCoordElement of the same shape-specific class.
        """

        pass

    #######################################################################
    ################################# MISC ################################
    #######################################################################

    @abstractmethod
    def crop_image(self, image):
        """
        Crop the input image according to the coordinates of the element.

        Args:
            image (:obj:`Numpy array`): The array of the input image.

        Returns:
            :obj:`Numpy array`: The array of the cropped image.
        """

        pass

    #######################################################################
    ########################## Import and Export ##########################
    #######################################################################

    def to_dict(self) -> Dict[str, Any]:
        """
        Generate a dictionary representation of the current object:
            {
                "block_type": <"interval", "rectangle", "quadrilateral"> ,
                "non_empty_block_attr1": value1,
                ...
            }
        """

        data = {
            key: getattr(self, key)
            for key in self._features
            if getattr(self, key) is not None
        }
        data["block_type"] = self._name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseCoordElement":
        """Initialize an instance based on the dictionary representation

        Args:
            data (:obj:`dict`): The dictionary representation of the object
        """

        assert (
            cls._name == data["block_type"]
        ), f"Incompatible block types {data['block_type']}"

        return cls(**{f: data[f] for f in cls._features})


@inherit_docstrings
class Interval(BaseCoordElement):
    """
    This class describes the coordinate system of an interval, a block defined by a pair of start and end point
    on the designated axis and same length as the base canvas on the other axis.

    Args:
        start (:obj:`numeric`):
            The coordinate of the start point on the designated axis.
        end (:obj:`numeric`):
            The end coordinate on the same axis as start.
        axis (:obj:`str`):
            The designated axis that the end points belong to.
        canvas_height (:obj:`numeric`, `optional`, defaults to 0):
            The height of the canvas that the interval is on.
        canvas_width (:obj:`numeric`, `optional`, defaults to 0):
            The width of the canvas that the interval is on.
    """

    _name = "interval"
    _features = ["start", "end", "axis", "canvas_height", "canvas_width"]

    def __init__(self, start, end, axis, canvas_height=None, canvas_width=None):

        assert start <= end, f"Invalid input for start and end. Start must <= end."
        self.start = start
        self.end = end

        assert axis in ["x", "y"], f"Invalid axis {axis}. Axis must be in 'x' or 'y'"
        self.axis = axis

        self.canvas_height = canvas_height or 0
        self.canvas_width = canvas_width or 0

    @property
    def height(self):
        """
        Calculate the height of the interval. If the interval is along the x-axis, the height will be the
        height of the canvas, otherwise, it will be the difference between the start and end point.

        Returns:
            :obj:`numeric`: Output the numeric value of the height.
        """

        if self.axis == "x":
            return self.canvas_height
        else:
            return self.end - self.start

    @property
    def width(self):
        """
        Calculate the width of the interval. If the interval is along the y-axis, the width will be the
        width of the canvas, otherwise, it will be the difference between the start and end point.

        Returns:
            :obj:`numeric`: Output the numeric value of the width.
        """

        if self.axis == "y":
            return self.canvas_width
        else:
            return self.end - self.start

    @property
    def coordinates(self):
        """
        This method considers an interval as a rectangle and calculates the coordinates of the upper left
        and lower right corners to define the interval.

        Returns:
            :obj:`Tuple(numeric)`:
                Output the numeric values of the coordinates in a Tuple of size four.
        """

        if self.axis == "x":
            coords = (self.start, 0, self.end, self.canvas_height)
        else:
            coords = (0, self.start, self.canvas_width, self.end)

        return coords

    @property
    def points(self):
        """
        Return the coordinates of all four corners of the interval in a clockwise fashion
        starting from the upper left.

        Returns:
            :obj:`Numpy array`: A Numpy array of shape 4x2 containing the coordinates.
        """

        return _cvt_coordinates_to_points(self.coordinates)

    @property
    def center(self):
        """
        Calculate the mid-point between the start and end point.

        Returns:
            :obj:`Tuple(numeric)`: Returns of coordinate of the center.
        """

        return (self.start + self.end) / 2.0

    @property
    def area(self):
        """Return the area of the covered region of the interval.
        The area is bounded to the canvas. If the interval is put
        on a canvas, the area equals to interval width * canvas height
        (axis='x') or interval height * canvas width (axis='y').
        Otherwise, the area is zero.
        """
        return self.height * self.width

    def put_on_canvas(self, canvas):
        """
        Set the height and the width of the canvas that the interval is on.

        Args:
            canvas (:obj:`Numpy array` or :obj:`BaseCoordElement` or :obj:`PIL.Image.Image`):
                The base element that the interval is on. The numpy array should be the
                format of `[height, width]`.

        Returns:
            :obj:`Interval`:
                A copy of the current Interval with its canvas height and width set to
                those of the input canvas.
        """

        if isinstance(canvas, np.ndarray):
            h, w = canvas.shape[:2]
        elif isinstance(canvas, BaseCoordElement):
            h, w = canvas.height, canvas.width
        elif isinstance(canvas, Image.Image):
            w, h = canvas.size
        else:
            raise NotImplementedError

        return self.set(canvas_height=h, canvas_width=w)

    @support_textblock
    def condition_on(self, other):

        if isinstance(other, Interval):
            if other.axis == self.axis:
                d = other.start
                # Reset the canvas size in the absolute coordinates
                return self.__class__(self.start + d, self.end + d, self.axis)
            else:
                return copy(self)

        elif isinstance(other, Rectangle):

            return self.put_on_canvas(other).to_rectangle().condition_on(other)

        elif isinstance(other, Quadrilateral):

            return self.put_on_canvas(other).to_quadrilateral().condition_on(other)

        else:
            raise Exception(f"Invalid input type {other.__class__} for other")

    @support_textblock
    def relative_to(self, other):

        if isinstance(other, Interval):
            if other.axis == self.axis:
                d = other.start
                # Reset the canvas size in the absolute coordinates
                return self.__class__(self.start - d, self.end - d, self.axis)
            else:
                return copy(self)

        elif isinstance(other, Rectangle):

            return self.put_on_canvas(other).to_rectangle().relative_to(other)

        elif isinstance(other, Quadrilateral):

            return self.put_on_canvas(other).to_quadrilateral().relative_to(other)

        else:
            raise Exception(f"Invalid input type {other.__class__} for other")

    @support_textblock
    def is_in(self, other, soft_margin={}, center=False):

        other = other.pad(**soft_margin)

        if isinstance(other, Interval):
            if self.axis != other.axis:
                return False
            else:
                if not center:
                    return other.start <= self.start <= self.end <= other.end
                else:
                    return other.start <= self.center <= other.end

        elif isinstance(other, Rectangle) or isinstance(other, Quadrilateral):
            x_1, y_1, x_2, y_2 = other.coordinates

            if center:
                if self.axis == "x":
                    return x_1 <= self.center <= x_2
                else:
                    return y_1 <= self.center <= y_2
            else:
                if self.axis == "x":
                    return x_1 <= self.start <= self.end <= x_2
                else:
                    return y_1 <= self.start <= self.end <= y_2

        else:
            raise Exception(f"Invalid input type {other.__class__} for other")

    @support_textblock
    def intersect(self, other: BaseCoordElement, strict: bool = True):
        """"""

        if isinstance(other, Interval):
            if self.axis != other.axis:
                if self.axis == "x" and other.axis == "y":
                    return Rectangle(self.start, other.start, self.end, other.end)
                else:
                    return Rectangle(other.start, self.start, other.end, self.end)
            else:
                return self.__class__(
                    max(self.start, other.start),
                    min(self.end, other.end),
                    self.axis,
                    self.canvas_height,
                    self.canvas_width,
                )

        elif isinstance(other, Rectangle):
            x_1, y_1, x_2, y_2 = other.coordinates
            if self.axis == "x":
                return Rectangle(max(x_1, self.start), y_1, min(x_2, self.end), y_2)
            elif self.axis == "y":
                return Rectangle(x_1, max(y_1, self.start), x_2, min(y_2, self.end))

        elif isinstance(other, Quadrilateral):
            if strict:
                raise NotSupportedShapeError(
                    "The intersection between an Interval and a Quadrilateral might generate Polygon shapes that are not supported in the current version of layoutparser. You can pass `strict=False` in the input that converts the Quadrilateral to Rectangle to avoid this Exception."
                )
            else:
                warnings.warn(
                    f"With `strict=False`, the other of shape {other.__class__} will be converted to {Rectangle} for obtaining the intersection"
                )
                return self.intersect(other.to_rectangle())

        else:
            raise Exception(f"Invalid input type {other.__class__} for other")

    @support_textblock
    def union(self, other: BaseCoordElement, strict: bool = True):
        """"""
        if isinstance(other, Interval):
            if self.axis != other.axis:
                raise InvalidShapeError(
                    f"Unioning two intervals of different axes is not allowed."
                )
            else:
                return self.__class__(
                    min(self.start, other.start),
                    max(self.end, other.end),
                    self.axis,
                    self.canvas_height,
                    self.canvas_width,
                )

        elif isinstance(other, Rectangle):
            x_1, y_1, x_2, y_2 = other.coordinates
            if self.axis == "x":
                return Rectangle(min(x_1, self.start), y_1, max(x_2, self.end), y_2)
            elif self.axis == "y":
                return Rectangle(x_1, min(y_1, self.start), x_2, max(y_2, self.end))

        elif isinstance(other, Quadrilateral):
            if strict:
                raise NotSupportedShapeError(
                    "The intersection between an Interval and a Quadrilateral might generate Polygon shapes that are not supported in the current version of layoutparser. You can pass `strict=False` in the input that converts the Quadrilateral to Rectangle to avoid this Exception."
                )
            else:
                warnings.warn(
                    f"With `strict=False`, the other of shape {other.__class__} will be converted to {Rectangle} for obtaining the intersection"
                )
                return self.union(other.to_rectangle())

        else:
            raise Exception(f"Invalid input type {other.__class__} for other")

    def pad(self, left=0, right=0, top=0, bottom=0, safe_mode=True):

        if self.axis == "x":
            start = self.start - left
            end = self.end + right
            if top or bottom:
                warnings.warn(
                    f"Invalid padding top/bottom for an x axis {self.__class__.__name__}"
                )
        else:
            start = self.start - top
            end = self.end + bottom
            if left or right:
                warnings.warn(
                    f"Invalid padding right/left for a y axis {self.__class__.__name__}"
                )

        if safe_mode:
            start = max(0, start)

        return self.set(start=start, end=end)

    def shift(self, shift_distance):
        """
        Shift the interval by a user specified amount along the same axis that the interval is defined on.

        Args:
            shift_distance (:obj:`numeric`): The number of pixels used to shift the interval.

        Returns:
            :obj:`BaseCoordElement`: The shifted Interval object.
        """

        if isinstance(shift_distance, Iterable):
            shift_distance = (
                shift_distance[0] if self.axis == "x" else shift_distance[1]
            )
            warnings.warn(
                f"Input shift for multiple axes. Only use the distance for the {self.axis} axis"
            )

        start = self.start + shift_distance
        end = self.end + shift_distance
        return self.set(start=start, end=end)

    def scale(self, scale_factor):
        """
        Scale the layout element by a user specified amount the same axis that the interval is defined on.

        Args:
            scale_factor (:obj:`numeric`): The amount for downscaling or upscaling the element.

        Returns:
            :obj:`BaseCoordElement`: The scaled Interval object.
        """

        if isinstance(scale_factor, Iterable):
            scale_factor = scale_factor[0] if self.axis == "x" else scale_factor[1]
            warnings.warn(
                f"Input scale for multiple axes. Only use the factor for the {self.axis} axis"
            )

        start = self.start * scale_factor
        end = self.end * scale_factor
        return self.set(start=start, end=end)

    def crop_image(self, image):
        x_1, y_1, x_2, y_2 = self.put_on_canvas(image).coordinates
        return image[int(y_1) : int(y_2), int(x_1) : int(x_2)]

    def to_rectangle(self):
        """
        Convert the Interval to a Rectangle element.

        Returns:
            :obj:`Rectangle`: The converted Rectangle object.
        """
        return Rectangle(*self.coordinates)

    def to_quadrilateral(self):
        """
        Convert the Interval to a Quadrilateral element.

        Returns:
            :obj:`Quadrilateral`: The converted Quadrilateral object.
        """
        return Quadrilateral(self.points)

    @classmethod
    def from_series(cls, series):
        series = series.dropna()
        if series.get("x_1") and series.get("x_2"):
            axis = "x"
            start, end = series.get("x_1"), series.get("x_2")
        else:
            axis = "y"
            start, end = series.get("y_1"), series.get("y_2")

        return cls(
            start,
            end,
            axis=axis,
            canvas_height=series.get("height") or 0,
            canvas_width=series.get("width") or 0,
        )


@inherit_docstrings
class Rectangle(BaseCoordElement):
    """
    This class describes the coordinate system of an axial rectangle box using two points as indicated below::

            (x_1, y_1) ----
            |             |
            |             |
            |             |
            ---- (x_2, y_2)

    Args:
        x_1 (:obj:`numeric`):
            x coordinate on the horizontal axis of the upper left corner of the rectangle.
        y_1 (:obj:`numeric`):
            y coordinate on the vertical axis of the upper left corner of the rectangle.
        x_2 (:obj:`numeric`):
            x coordinate on the horizontal axis of the lower right corner of the rectangle.
        y_2 (:obj:`numeric`):
            y coordinate on the vertical axis of the lower right corner of the rectangle.
    """

    _name = "rectangle"
    _features = ["x_1", "y_1", "x_2", "y_2"]

    def __init__(self, x_1, y_1, x_2, y_2):

        self.x_1 = x_1
        self.y_1 = y_1
        self.x_2 = x_2
        self.y_2 = y_2

    @property
    def height(self):
        """
        Calculate the height of the rectangle.

        Returns:
            :obj:`numeric`: Output the numeric value of the height.
        """

        return self.y_2 - self.y_1

    @property
    def width(self):
        """
        Calculate the width of the rectangle.

        Returns:
            :obj:`numeric`: Output the numeric value of the width.
        """

        return self.x_2 - self.x_1

    @property
    def coordinates(self):
        """
        Return the coordinates of the two points that define the rectangle.

        Returns:
            :obj:`Tuple(numeric)`: Output the numeric values of the coordinates in a Tuple of size four.
        """

        return (self.x_1, self.y_1, self.x_2, self.y_2)

    @property
    def points(self):
        """
        Return the coordinates of all four corners of the rectangle in a clockwise fashion
        starting from the upper left.

        Returns:
            :obj:`Numpy array`: A Numpy array of shape 4x2 containing the coordinates.
        """

        return _cvt_coordinates_to_points(self.coordinates)

    @property
    def center(self):
        """
        Calculate the center of the rectangle.

        Returns:
            :obj:`Tuple(numeric)`: Returns of coordinate of the center.
        """

        return (self.x_1 + self.x_2) / 2.0, (self.y_1 + self.y_2) / 2.0

    @property
    def area(self):
        """
        Return the area of the rectangle.
        """
        return self.width * self.height

    @support_textblock
    def condition_on(self, other):

        if isinstance(other, Interval):
            if other.axis == "x":
                dx, dy = other.start, 0
            else:
                dx, dy = 0, other.start

            return self.__class__(
                self.x_1 + dx, self.y_1 + dy, self.x_2 + dx, self.y_2 + dy
            )

        elif isinstance(other, Rectangle):
            dx, dy, _, _ = other.coordinates

            return self.__class__(
                self.x_1 + dx, self.y_1 + dy, self.x_2 + dx, self.y_2 + dy
            )

        elif isinstance(other, Quadrilateral):
            transformed_points = _perspective_transformation(
                other.perspective_matrix, self.points, is_inv=True
            )

            return other.__class__(transformed_points, self.height, self.width)

        else:
            raise Exception(f"Invalid input type {other.__class__} for other")

    @support_textblock
    def relative_to(self, other):
        if isinstance(other, Interval):
            if other.axis == "x":
                dx, dy = other.start, 0
            else:
                dx, dy = 0, other.start

            return self.__class__(
                self.x_1 - dx, self.y_1 - dy, self.x_2 - dx, self.y_2 - dy
            )

        elif isinstance(other, Rectangle):
            dx, dy, _, _ = other.coordinates

            return self.__class__(
                self.x_1 - dx, self.y_1 - dy, self.x_2 - dx, self.y_2 - dy
            )

        elif isinstance(other, Quadrilateral):
            transformed_points = _perspective_transformation(
                other.perspective_matrix, self.points, is_inv=False
            )

            return other.__class__(transformed_points, self.height, self.width)

        else:
            raise Exception(f"Invalid input type {other.__class__} for other")

    @support_textblock
    def is_in(self, other, soft_margin={}, center=False):

        other = other.pad(**soft_margin)

        if isinstance(other, Interval):
            if not center:
                if other.axis == "x":
                    start, end = self.x_1, self.x_2
                else:
                    start, end = self.y_1, self.y_2
                return other.start <= start <= end <= other.end
            else:
                c = self.center[0] if other.axis == "x" else self.center[1]
                return other.start <= c <= other.end

        elif isinstance(other, Rectangle):
            x_interval = other.to_interval(axis="x")
            y_interval = other.to_interval(axis="y")
            return self.is_in(x_interval, center=center) and self.is_in(
                y_interval, center=center
            )

        elif isinstance(other, Quadrilateral):

            if not center:
                # This is equivalent to determine all the points of the
                # rectangle is in the quadrilateral.
                is_vertice_in = [
                    _vertice_in_polygon(vertice, other.points)
                    for vertice in self.points
                ]
                return all(is_vertice_in)
            else:
                center = np.array(self.center)
                return _vertice_in_polygon(center, other.points)

        else:
            raise Exception(f"Invalid input type {other.__class__} for other")

    @support_textblock
    def intersect(self, other: BaseCoordElement, strict: bool = True):
        """"""

        if isinstance(other, Interval):
            return other.intersect(self)

        elif isinstance(other, Rectangle):

            return self.__class__(
                max(self.x_1, other.x_1),
                max(self.y_1, other.y_1),
                min(self.x_2, other.x_2),
                min(self.y_2, other.y_2),
            )

        elif isinstance(other, Quadrilateral):
            if strict:
                raise NotSupportedShapeError(
                    "The intersection between a Rectangle and a Quadrilateral might generate Polygon shapes that are not supported in the current version of layoutparser. You can pass `strict=False` in the input that converts the Quadrilateral to Rectangle to avoid this Exception."
                )
            else:
                warnings.warn(
                    f"With `strict=False`, the other of shape {other.__class__} will be converted to {Rectangle} for obtaining the intersection"
                )
                return self.intersect(other.to_rectangle())

        else:
            raise Exception(f"Invalid input type {other.__class__} for other")

    @support_textblock
    def union(self, other: BaseCoordElement, strict: bool = True):
        """"""
        if isinstance(other, Interval):
            return other.intersect(self)

        elif isinstance(other, Rectangle):
            return self.__class__(
                min(self.x_1, other.x_1),
                min(self.y_1, other.y_1),
                max(self.x_2, other.x_2),
                max(self.y_2, other.y_2),
            )

        elif isinstance(other, Quadrilateral):
            if strict:
                raise NotSupportedShapeError(
                    "The intersection between an Interval and a Quadrilateral might generate Polygon shapes that are not supported in the current version of layoutparser. You can pass `strict=False` in the input that converts the Quadrilateral to Rectangle to avoid this Exception."
                )
            else:
                warnings.warn(
                    f"With `strict=False`, the other of shape {other.__class__} will be converted to {Rectangle} for obtaining the intersection"
                )
                return self.union(other.to_rectangle())

        else:
            raise Exception(f"Invalid input type {other.__class__} for other")

    def pad(self, left=0, right=0, top=0, bottom=0, safe_mode=True):

        x_1 = self.x_1 - left
        y_1 = self.y_1 - top
        x_2 = self.x_2 + right
        y_2 = self.y_2 + bottom

        if safe_mode:
            x_1 = max(0, x_1)
            y_1 = max(0, y_1)

        return self.__class__(x_1, y_1, x_2, y_2)

    def shift(self, shift_distance=0):

        if not isinstance(shift_distance, Iterable):
            shift_x = shift_distance
            shift_y = shift_distance
        else:
            assert (
                len(shift_distance) == 2
            ), "shift_distance should have 2 elements, one for x dimension and one for y dimension"
            shift_x, shift_y = shift_distance

        x_1 = self.x_1 + shift_x
        y_1 = self.y_1 + shift_y
        x_2 = self.x_2 + shift_x
        y_2 = self.y_2 + shift_y
        return self.__class__(x_1, y_1, x_2, y_2)

    def scale(self, scale_factor=1):

        if not isinstance(scale_factor, Iterable):
            scale_x = scale_factor
            scale_y = scale_factor
        else:
            assert (
                len(scale_factor) == 2
            ), "scale_factor should have 2 elements, one for x dimension and one for y dimension"
            scale_x, scale_y = scale_factor

        x_1 = self.x_1 * scale_x
        y_1 = self.y_1 * scale_y
        x_2 = self.x_2 * scale_x
        y_2 = self.y_2 * scale_y
        return self.__class__(x_1, y_1, x_2, y_2)

    def crop_image(self, image):
        x_1, y_1, x_2, y_2 = self.coordinates
        return image[int(y_1) : int(y_2), int(x_1) : int(x_2)]

    def to_interval(self, axis, **kwargs):
        if axis == "x":
            start, end = self.x_1, self.x_2
        else:
            start, end = self.y_1, self.y_2

        return Interval(start, end, axis=axis, **kwargs)

    def to_quadrilateral(self):
        return Quadrilateral(self.points)

    @classmethod
    def from_series(cls, series):
        series = series.dropna()
        return cls(*[series[fname] for fname in cls.feature_names])


@inherit_docstrings
class Quadrilateral(BaseCoordElement):
    """
    This class describes the coodinate system of a four-sided polygon. A quadrilateral is defined by
    the coordinates of its 4 corners in a clockwise order starting with the upper left corner (as shown below)::

        points[0] -...- points[1]
        |                      |
        .                      .
        .                      .
        .                      .
        |                      |
        points[3] -...- points[2]

    Args:
        points (:obj:`Numpy array` or `list`):
            A `np.ndarray` of shape 4x2  for four corner coordinates
            or a list of length 8 for in the format of
            `[p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y]`
            or a list of length 4 in the format of
            `[[p0_x, p0_y], [p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y]]`.
        height (:obj:`numeric`, `optional`, defaults to `None`):
            The height of the quadrilateral. This is to better support the perspective
            transformation from the OpenCV library.
        width (:obj:`numeric`, `optional`, defaults to `None`):
            The width of the quadrilateral. Similarly as height, this is to better support the perspective
            transformation from the OpenCV library.
    """

    _name = "quadrilateral"
    _features = ["points", "height", "width"]

    def __init__(
        self, points: Union[np.ndarray, List, List[List]], height=None, width=None
    ):

        if isinstance(points, np.ndarray):
            if points.shape != (4, 2):
                raise ValueError(f"Invalid points shape: {points.shape}.")
        elif isinstance(points, list):
            if len(points) == 8:
                points = np.array(points).reshape(4, 2)
            elif len(points) == 4 and isinstance(points[0], list):
                points = np.array(points)
            else:
                raise ValueError(
                    f"Invalid number of points element {len(points)}. Should be 8."
                )
        else:
            raise ValueError(
                f"Invalid input type for points {type(points)}."
                "Please make sure it is a list of np.ndarray."
            )

        self._points = points
        self._width = width
        self._height = height

    @property
    def height(self):
        """
        Return the user defined height, otherwise the height of its circumscribed rectangle.

        Returns:
            :obj:`numeric`: Output the numeric value of the height.
        """

        if self._height is not None:
            return self._height
        return self.points[:, 1].max() - self.points[:, 1].min()

    @property
    def width(self):
        """
        Return the user defined width, otherwise the width of its circumscribed rectangle.

        Returns:
            :obj:`numeric`: Output the numeric value of the width.
        """

        if self._width is not None:
            return self._width
        return self.points[:, 0].max() - self.points[:, 0].min()

    @property
    def coordinates(self):
        """
        Return the coordinates of the upper left and lower right corners points that
        define the circumscribed rectangle.

        Returns
            :obj:`Tuple(numeric)`: Output the numeric values of the coordinates in a Tuple of size four.
        """

        return _cvt_points_to_coordinates(self.points)

    @property
    def points(self):
        """
        Return the coordinates of all four corners of the quadrilateral in a clockwise fashion
        starting from the upper left.

        Returns:
            :obj:`Numpy array`: A Numpy array of shape 4x2 containing the coordinates.
        """

        return self._points

    @property
    def center(self):
        """
        Calculate the center of the quadrilateral.

        Returns:
            :obj:`Tuple(numeric)`: Returns of coordinate of the center.
        """

        return tuple(self.points.mean(axis=0).tolist())

    @property
    def area(self):
        """
        Return the area of the quadrilateral.
        """
        return _polygon_area(self.points[:, 0], self.points[:, 1])

    @property
    def mapped_rectangle_points(self):

        x_map = {0: 0, 1: 0, 2: self.width, 3: self.width}
        y_map = {0: 0, 1: 0, 2: self.height, 3: self.height}

        return self.map_to_points_ordering(x_map, y_map)

    @property
    def perspective_matrix(self):
        return _getPerspectiveTransform(
            self.points.astype("float32"),
            self.mapped_rectangle_points.astype("float32"),
        )

    def map_to_points_ordering(self, x_map, y_map):

        points_ordering = self.points.argsort(axis=0).argsort(axis=0)
        # Ref: https://github.com/numpy/numpy/issues/8757#issuecomment-355126992

        return np.vstack(
            [
                np.vectorize(x_map.get)(points_ordering[:, 0]),
                np.vectorize(y_map.get)(points_ordering[:, 1]),
            ]
        ).T

    @support_textblock
    def condition_on(self, other):

        if isinstance(other, Interval):

            if other.axis == "x":
                return self.shift([other.start, 0])
            else:
                return self.shift([0, other.start])

        elif isinstance(other, Rectangle):

            return self.shift([other.x_1, other.y_1])

        elif isinstance(other, Quadrilateral):

            transformed_points = _perspective_transformation(
                other.perspective_matrix, self.points, is_inv=True
            )
            return self.__class__(transformed_points, self.height, self.width)

        else:
            raise Exception(f"Invalid input type {other.__class__} for other")

    @support_textblock
    def relative_to(self, other):

        if isinstance(other, Interval):

            if other.axis == "x":
                return self.shift([-other.start, 0])
            else:
                return self.shift([0, -other.start])

        elif isinstance(other, Rectangle):

            return self.shift([-other.x_1, -other.y_1])

        elif isinstance(other, Quadrilateral):

            transformed_points = _perspective_transformation(
                other.perspective_matrix, self.points, is_inv=False
            )
            return self.__class__(transformed_points, self.height, self.width)

        else:
            raise Exception(f"Invalid input type {other.__class__} for other")

    @support_textblock
    def is_in(self, other, soft_margin={}, center=False):

        other = other.pad(**soft_margin)

        if isinstance(other, Interval):
            if not center:
                if other.axis == "x":
                    start, end = self.coordinates[0], self.coordinates[2]
                else:
                    start, end = self.coordinates[1], self.coordinates[3]
                return other.start <= start <= end <= other.end
            else:
                c = self.center[0] if other.axis == "x" else self.center[1]
                return other.start <= c <= other.end

        elif isinstance(other, Rectangle):
            x_interval = other.to_interval(axis="x")
            y_interval = other.to_interval(axis="y")
            return self.is_in(x_interval, center=center) and self.is_in(
                y_interval, center=center
            )

        elif isinstance(other, Quadrilateral):

            if not center:
                # This is equivalent to determine all the points of the
                # rectangle is in the quadrilateral.
                is_vertice_in = [
                    _vertice_in_polygon(vertice, other.points)
                    for vertice in self.points
                ]
                return all(is_vertice_in)
            else:
                center = np.array(self.center)
                return _vertice_in_polygon(center, other.points)

        else:
            raise Exception(f"Invalid input type {other.__class__} for other")

    @support_textblock
    def intersect(self, other: BaseCoordElement, strict: bool = True):
        """"""

        if strict:
            raise NotSupportedShapeError(
                "The intersection between a Quadrilateral and other objects might generate Polygon shapes that are not supported in the current version of layoutparser. You can pass `strict=False` in the input that converts the Quadrilateral to Rectangle to avoid this Exception."
            )
        else:
            if isinstance(other, Interval) or isinstance(other, Rectangle):
                warnings.warn(
                    f"With `strict=False`, the current Quadrilateral object will be converted to {Rectangle} for obtaining the intersection"
                )
                return other.intersect(self.to_rectangle())
            elif isinstance(other, Quadrilateral):
                warnings.warn(
                    f"With `strict=False`, both input Quadrilateral objects will be converted to {Rectangle} for obtaining the intersection"
                )
                return self.to_rectangle().intersect(other.to_rectangle())
            else:
                raise Exception(f"Invalid input type {other.__class__} for other")

    @support_textblock
    def union(self, other: BaseCoordElement, strict: bool = True):
        """"""
        if strict:
            raise NotSupportedShapeError(
                "The intersection between a Quadrilateral and other objects might generate Polygon shapes that are not supported in the current version of layoutparser. You can pass `strict=False` in the input that converts the Quadrilateral to Rectangle to avoid this Exception."
            )
        else:
            if isinstance(other, Interval) or isinstance(other, Rectangle):
                warnings.warn(
                    f"With `strict=False`, the current Quadrilateral object will be converted to {Rectangle} for obtaining the intersection"
                )
                return other.union(self.to_rectangle())
            elif isinstance(other, Quadrilateral):
                warnings.warn(
                    f"With `strict=False`, both input Quadrilateral objects will be converted to {Rectangle} for obtaining the intersection"
                )
                return self.to_rectangle().union(other.to_rectangle())
            else:
                raise Exception(f"Invalid input type {other.__class__} for other")

    def pad(self, left=0, right=0, top=0, bottom=0, safe_mode=True):

        x_map = {0: -left, 1: -left, 2: right, 3: right}
        y_map = {0: -top, 1: -top, 2: bottom, 3: bottom}

        padding_mat = self.map_to_points_ordering(x_map, y_map)

        points = self.points + padding_mat
        if safe_mode:
            points = np.maximum(points, 0)

        return self.set(points=points)

    def shift(self, shift_distance=0):

        if not isinstance(shift_distance, Iterable):
            shift_mat = [shift_distance, shift_distance]
        else:
            assert (
                len(shift_distance) == 2
            ), "shift_distance should have 2 elements, one for x dimension and one for y dimension"
            shift_mat = shift_distance

        points = self.points + np.array(shift_mat)

        return self.set(points=points)

    def scale(self, scale_factor=1):

        if not isinstance(scale_factor, Iterable):
            scale_mat = [scale_factor, scale_factor]
        else:
            assert (
                len(scale_factor) == 2
            ), "scale_factor should have 2 elements, one for x dimension and one for y dimension"
            scale_mat = scale_factor

        points = self.points * np.array(scale_mat)

        return self.set(points=points)

    def crop_image(self, image):
        """
        Crop the input image using the points of the quadrilateral instance.

        Args:
            image (:obj:`Numpy array`): The array of the input image.

        Returns:
            :obj:`Numpy array`: The array of the cropped image.
        """

        return _warpPerspective(
            image, self.perspective_matrix, (int(self.width), int(self.height))
        )

    def to_interval(self, axis, **kwargs):

        x_1, y_1, x_2, y_2 = self.coordinates
        if axis == "x":
            start, end = x_1, x_2
        else:
            start, end = y_1, y_2

        return Interval(start, end, axis=axis, **kwargs)

    def to_rectangle(self):
        return Rectangle(*self.coordinates)

    @classmethod
    def from_series(cls, series):
        series = series.dropna()

        points = pd.to_numeric(series[cls.feature_names[:8]]).values.reshape(4, -2)

        return cls(
            points=points, height=series.get("height"), width=series.get("width")
        )

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return False
        return np.isclose(self.points, other.points).all()

    def __repr__(self):
        keys = ["points", "width", "height"]
        info_str = ", ".join([f"{key}={getattr(self, key)}" for key in keys])
        return f"{self.__class__.__name__}({info_str})"

    def to_dict(self) -> Dict[str, Any]:

        """
        Generate a dictionary representation of the current object::

            {
                "block_type": "quadrilateral",
                "points": [
                    p[0,0], p[0,1],
                    p[1,0], p[1,1],
                    p[2,0], p[2,1],
                    p[3,0], p[3,1]
                ],
                "height": value,
                "width": value
            }
        """
        data = super().to_dict()
        data["points"] = data["points"].reshape(-1).tolist()
        return data


ALL_BASECOORD_ELEMENTS = [Interval, Rectangle, Quadrilateral]

BASECOORD_ELEMENT_NAMEMAP = {ele._name: ele for ele in ALL_BASECOORD_ELEMENTS}
BASECOORD_ELEMENT_INDEXMAP = {
    ele._name: idx for idx, ele in enumerate(ALL_BASECOORD_ELEMENTS)
}


@inherit_docstrings(base_class=BaseCoordElement)
class TextBlock(BaseLayoutElement):
    """
    This class constructs content-related information of a layout element in addition to its coordinate definitions
    (i.e. Interval, Rectangle or Quadrilateral).

    Args:
        block (:obj:`BaseCoordElement`):
            The shape-specific coordinate systems that the text block belongs to.
        text (:obj:`str`, `optional`, defaults to None):
            The ocr'ed text results within the boundaries of the text block.
        id (:obj:`int`, `optional`, defaults to `None`):
            The id of the text block.
        type (:obj:`int`, `optional`, defaults to `None`):
            The type of the text block.
        parent (:obj:`int`, `optional`, defaults to `None`):
            The id of the parent object.
        next (:obj:`int`, `optional`, defaults to `None`):
            The id of the next block.
        score (:obj:`numeric`, defaults to `None`):
            The prediction confidence of the block
    """

    _name = "textblock"
    _features = ["text", "id", "type", "parent", "next", "score"]

    def __init__(
        self, block, text=None, id=None, type=None, parent=None, next=None, score=None
    ):

        assert isinstance(block, BaseCoordElement)
        self.block = block

        self.text = text
        self.id = id
        self.type = type
        self.parent = parent
        self.next = next
        self.score = score

    @property
    def height(self):
        """
        Return the height of the shape-specific block.

        Returns:
            :obj:`numeric`: Output the numeric value of the height.
        """

        return self.block.height

    @property
    def width(self):
        """
        Return the width of the shape-specific block.

        Returns:
            :obj:`numeric`: Output the numeric value of the width.
        """

        return self.block.width

    @property
    def coordinates(self):
        """
        Return the coordinates of the two corner points that define the shape-specific block.

        Returns:
            :obj:`Tuple(numeric)`: Output the numeric values of the coordinates in a Tuple of size four.
        """

        return self.block.coordinates

    @property
    def points(self):
        """
        Return the coordinates of all four corners of the shape-specific block in a clockwise fashion
        starting from the upper left.

        Returns:
            :obj:`Numpy array`: A Numpy array of shape 4x2 containing the coordinates.
        """

        return self.block.points

    @property
    def area(self):
        """
        Return the area of associated block.
        """
        return self.block.area

    @mixin_textblock_meta
    def condition_on(self, other):
        return self.block.condition_on(other)

    @mixin_textblock_meta
    def relative_to(self, other):
        return self.block.relative_to(other)

    def is_in(self, other, soft_margin={}, center=False):
        return self.block.is_in(other, soft_margin, center)

    @mixin_textblock_meta
    def union(self, other: BaseCoordElement, strict: bool = True):
        return self.block.union(other, strict=strict)

    @mixin_textblock_meta
    def intersect(self, other: BaseCoordElement, strict: bool = True):
        return self.block.intersect(other, strict=strict)

    @mixin_textblock_meta
    def shift(self, shift_distance):
        return self.block.shift(shift_distance)

    @mixin_textblock_meta
    def pad(self, left=0, right=0, top=0, bottom=0, safe_mode=True):
        return self.block.pad(left, right, top, bottom, safe_mode)

    @mixin_textblock_meta
    def scale(self, scale_factor):
        return self.block.scale(scale_factor)

    def crop_image(self, image):
        return self.block.crop_image(image)

    def to_interval(self, axis: Optional[str] = None, **kwargs):
        if isinstance(self.block, Interval):
            return self
        else:
            if not axis:
                raise ValueError(
                    f"Please provide valid `axis` values {'x' or 'y'} as the input"
                )
            return self.set(block=self.block.to_interval(axis=axis, **kwargs))

    def to_rectangle(self):
        if isinstance(self.block, Rectangle):
            return self
        else:
            return self.set(block=self.block.to_rectangle())

    def to_quadrilateral(self):
        if isinstance(self.block, Quadrilateral):
            return self
        else:
            return self.set(block=self.block.to_quadrilateral())

    @classmethod
    def from_series(cls, series):

        features = {fname: series.get(fname) for fname in cls.feature_names}
        series = series.dropna()
        if set(Quadrilateral.feature_names[:8]).issubset(series.index):
            target_type = Quadrilateral
        elif set(Interval.feature_names).issubset(series.index):
            target_type = Interval
        elif set(Rectangle.feature_names).issubset(series.index):
            target_type = Rectangle
        else:
            target_type = Interval

        return cls(block=target_type.from_series(series), **features)

    def to_dict(self) -> Dict[str, Any]:
        """
        Generate a dictionary representation of the current textblock of the format::

            {
                "block_type": <name of self.block>,
                <attributes of self.block combined with
                    non-empty self._features>
            }
        """
        base_dict = self.block.to_dict()
        for f in self._features:
            val = getattr(self, f)
            if val is not None:
                base_dict[f] = getattr(self, f)
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextBlock":
        """Initialize the textblock based on the dictionary representation.
        It generate the block based on the `block_type` and `block_attr`,
        and loads the textblock specific features from the dict.

        Args:
            data (:obj:`dict`): The dictionary representation of the object
        """
        assert (
            data["block_type"] in BASECOORD_ELEMENT_NAMEMAP
        ), f"Invalid block_type {data['block_type']}"

        block = BASECOORD_ELEMENT_NAMEMAP[data["block_type"]].from_dict(data)

        return cls(block, **{f: data.get(f, None) for f in cls._features})


class Layout(MutableSequence):
    """
    The :obj:`Layout` class id designed for processing a list of layout elements
    on a page. It stores the layout elements in a list and the related `page_data`,
    and provides handy APIs for processing all the layout elements in batch. `

    Args:
        blocks (:obj:`list`):
            A list of layout element blocks
        page_data (Dict, optional):
            A dictionary storing the page (canvas) related information
            like `height`, `width`, etc. It should be passed in as a
            keyword argument to avoid any confusion.
            Defaults to None.
    """

    def __init__(self, blocks: Optional[List] = None, *, page_data: Dict = None):
        self._blocks = blocks if blocks is not None else []
        self.page_data = page_data or {}

    def __getitem__(self, key):
        blocks = self._blocks[key]
        if isinstance(key, slice):
            return self.__class__(self._blocks[key], page_data=self.page_data)
        else:
            return blocks

    def __setitem__(self, key, newvalue):
        self._blocks[key] = newvalue

    def __delitem__(self, key):
        del self._blocks[key]

    def __len__(self):
        return len(self._blocks)

    def __iter__(self):
        for ele in self._blocks:
            yield ele

    def __repr__(self):
        info_str = ", ".join([f"{key}={val}" for key, val in vars(self).items()])
        return f"{self.__class__.__name__}({info_str})"

    def __eq__(self, other):
        if isinstance(other, Layout):
            return (
                all((a, b) for a, b in zip(self, other))
                and self.page_data == other.page_data
            )
        else:
            return False

    def __add__(self, other):
        if isinstance(other, Layout):
            if self.page_data == other.page_data:
                return self.__class__(
                    self._blocks + other._blocks, page_data=self.page_data
                )
            elif self.page_data == {} or other.page_data == {}:
                return self.__class__(
                    self._blocks + other._blocks,
                    page_data=self.page_data or other.page_data,
                )
            else:
                raise ValueError(
                    f"Incompatible page_data for two innputs: {self.page_data} vs {other.page_data}."
                )
        elif isinstance(other, list):
            return self.__class__(self._blocks + other, page_data=self.page_data)
        else:
            raise ValueError(
                f"Invalid input type for other {other.__class__.__name__}."
            )

    def insert(self, key, value):
        self._blocks.insert(key, value)

    def copy(self):
        return self.__class__(copy(self._blocks), page_data=self.page_data)

    def relative_to(self, other):
        return self.__class__(
            [ele.relative_to(other) for ele in self], page_data=self.page_data
        )

    def condition_on(self, other):
        return self.__class__(
            [ele.condition_on(other) for ele in self], page_data=self.page_data
        )

    def is_in(self, other, soft_margin={}, center=False):
        return self.__class__(
            [ele.is_in(other, soft_margin, center) for ele in self],
            page_data=self.page_data,
        )

    def sort(self, key=None, reverse=False, inplace=False) -> Optional["Layout"]:
        """Sort the list of blocks based on the given 

        Args:
            key ([type], optional): key specifies a function of one argument that 
            is used to extract a comparison key from each list element. 
            Defaults to None.
            reverse (bool, optional): reverse is a boolean value. If set to True, 
            then the list elements are sorted as if each comparison were reversed. 
            Defaults to False.
            inplace (bool, optional): whether to perform the sort inplace. If set 
            to False, it will return another object instance with _block sorted in
            the order. Defaults to False.

        Examples::
            >>> import layoutparser as lp
            >>> i = lp.Interval(4, 5, axis="y")
            >>> l = lp.Layout([i, i.shift(2)])
            >>> l.sort(key=lambda x: x.coordinates[1], reverse=True)

        """
        if not inplace:
            return self.__class__(sorted(self._blocks, key=key, reverse=reverse), page_data=self.page_data)
        else:
            self._blocks.sort(key=key, reverse=reverse)

    def filter_by(self, other, soft_margin={}, center=False):
        """
        Return a `Layout` object containing the elements that are in the `other` object.

        Args:
            other (:obj:`BaseCoordElement`):
                The block to filter the current elements.

        Returns:
            :obj:`Layout`:
                A new layout object after filtering.
        """
        return self.__class__(
            [ele for ele in self if ele.is_in(other, soft_margin, center)],
            page_data=self.page_data,
        )

    def shift(self, shift_distance):
        """
        Shift all layout elements by user specified amounts on x and y axis respectively. If shift_distance is one
        numeric value, the element will by shifted by the same specified amount on both x and y axis.

        Args:
            shift_distance (:obj:`numeric` or :obj:`Tuple(numeric)` or :obj:`List[numeric]`):
                The number of pixels used to shift the element.

        Returns:
            :obj:`Layout`:
                A new layout object with all the elements shifted in the specified values.
        """
        return self.__class__(
            [ele.shift(shift_distance) for ele in self], page_data=self.page_data
        )

    def pad(self, left=0, right=0, top=0, bottom=0, safe_mode=True):
        """Pad all layout elements on the four sides of the polygon with the user-defined pixels. If
        safe_mode is set to True, the function will cut off the excess padding that falls on the negative
        side of the coordinates.

        Args:
            left (:obj:`int`, `optional`, defaults to 0): The number of pixels to pad on the upper side of the polygon.
            right (:obj:`int`, `optional`, defaults to 0): The number of pixels to pad on the lower side of the polygon.
            top (:obj:`int`, `optional`, defaults to 0): The number of pixels to pad on the left side of the polygon.
            bottom (:obj:`int`, `optional`, defaults to 0): The number of pixels to pad on the right side of the polygon.
            safe_mode (:obj:`bool`, `optional`, defaults to True): A bool value to toggle the safe_mode.

        Returns:
            :obj:`Layout`:
                A new layout object with all the elements padded in the specified values.
        """
        return self.__class__(
            [ele.pad(left, right, top, bottom, safe_mode) for ele in self],
            page_data=self.page_data,
        )

    def scale(self, scale_factor):
        """
        Scale all layout element by a user specified amount on x and y axis respectively. If scale_factor is one
        numeric value, the element will by scaled by the same specified amount on both x and y axis.

        Args:
            scale_factor (:obj:`numeric` or :obj:`Tuple(numeric)` or :obj:`List[numeric]`): The amount for downscaling or upscaling the element.

        Returns:
            :obj:`Layout`:
                A new layout object with all the elements scaled in the specified values.
        """
        return self.__class__(
            [ele.scale(scale_factor) for ele in self], page_data=self.page_data
        )

    def crop_image(self, image):
        return [ele.crop_image(image) for ele in self]

    def get_texts(self):
        """
        Iterate through all the text blocks in the list and append their ocr'ed text results.

        Returns:
            :obj:`List[str]`: A list of text strings of the text blocks in the list of layout elements.
        """

        return [ele.text for ele in self if hasattr(ele, "text")]

    def get_info(self, attr_name):
        """Given user-provided attribute name, check all the elements in the list and return the corresponding
        attribute values.

        Args:
            attr_name (:obj:`str`): The text string of certain attribute name.

        Returns:
            :obj:`List`:
                The list of the corresponding attribute value (if exist) of each element in the list.
        """
        return [getattr(ele, attr_name) for ele in self if hasattr(ele, attr_name)]

    def to_dict(self) -> Dict[str, Any]:
        """Generate a dict representation of the layout object with
        the page_data and all the blocks in its dict representation.

        Returns:
            :obj:`Dict`:
                The dictionary representation of the layout object.
        """
        return {"page_data": self.page_data, "blocks": [ele.to_dict() for ele in self]}

    def get_homogeneous_blocks(self) -> List[BaseLayoutElement]:
        """Convert all elements into blocks of the same type based
        on the type casting rule::

            Interval < Rectangle < Quadrilateral < TextBlock

        Returns:
            List[BaseLayoutElement]:
                A list of base layout elements of the maximal compatible
                type
        """

        # Detect the maximal compatible type
        has_textblock = False
        max_coord_level = -1
        for ele in self:

            if isinstance(ele, TextBlock):
                has_textblock = True
                block = ele.block
            else:
                block = ele

            max_coord_level = max(
                max_coord_level, BASECOORD_ELEMENT_INDEXMAP[block._name]
            )
        target_coord_name = ALL_BASECOORD_ELEMENTS[max_coord_level]._name

        if has_textblock:
            new_blocks = []
            for ele in self:
                if isinstance(ele, TextBlock):
                    ele = copy(ele)
                    if ele.block._name != target_coord_name:
                        ele.block = getattr(ele.block, f"to_{target_coord_name}")()
                else:
                    if ele._name != target_coord_name:
                        ele = getattr(ele, f"to_{target_coord_name}")()
                    ele = TextBlock(block)
                new_blocks.append(ele)
        else:
            new_blocks = [
                getattr(ele, f"to_{target_coord_name}")()
                if ele._name != target_coord_name
                else ele
                for ele in self
            ]

        return new_blocks

    def to_dataframe(self, enforce_same_type=False) -> pd.DataFrame:
        """Convert the layout object into the dataframe.
        Warning: the page data won't be exported.

        Args:
            enforce_same_type (:obj:`bool`, optional):
                If true, it will convert all the contained blocks to
                the maximal compatible data type.
                Defaults to False.

        Returns:
            pd.DataFrame:
                The dataframe representation of layout object
        """
        if enforce_same_type:
            blocks = self.get_homogeneous_blocks()
        else:
            blocks = self

        df = pd.DataFrame([ele.to_dict() for ele in blocks])

        return df
