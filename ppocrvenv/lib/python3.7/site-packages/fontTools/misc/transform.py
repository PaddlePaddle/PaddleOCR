"""Affine 2D transformation matrix class.

The Transform class implements various transformation matrix operations,
both on the matrix itself, as well as on 2D coordinates.

Transform instances are effectively immutable: all methods that operate on the
transformation itself always return a new instance. This has as the
interesting side effect that Transform instances are hashable, ie. they can be
used as dictionary keys.

This module exports the following symbols:

Transform
	this is the main class
Identity
	Transform instance set to the identity transformation
Offset
	Convenience function that returns a translating transformation
Scale
	Convenience function that returns a scaling transformation

:Example:

	>>> t = Transform(2, 0, 0, 3, 0, 0)
	>>> t.transformPoint((100, 100))
	(200, 300)
	>>> t = Scale(2, 3)
	>>> t.transformPoint((100, 100))
	(200, 300)
	>>> t.transformPoint((0, 0))
	(0, 0)
	>>> t = Offset(2, 3)
	>>> t.transformPoint((100, 100))
	(102, 103)
	>>> t.transformPoint((0, 0))
	(2, 3)
	>>> t2 = t.scale(0.5)
	>>> t2.transformPoint((100, 100))
	(52.0, 53.0)
	>>> import math
	>>> t3 = t2.rotate(math.pi / 2)
	>>> t3.transformPoint((0, 0))
	(2.0, 3.0)
	>>> t3.transformPoint((100, 100))
	(-48.0, 53.0)
	>>> t = Identity.scale(0.5).translate(100, 200).skew(0.1, 0.2)
	>>> t.transformPoints([(0, 0), (1, 1), (100, 100)])
	[(50.0, 100.0), (50.550167336042726, 100.60135501775433), (105.01673360427253, 160.13550177543362)]
	>>>
"""

from typing import NamedTuple


__all__ = ["Transform", "Identity", "Offset", "Scale"]


_EPSILON = 1e-15
_ONE_EPSILON = 1 - _EPSILON
_MINUS_ONE_EPSILON = -1 + _EPSILON


def _normSinCos(v):
	if abs(v) < _EPSILON:
		v = 0
	elif v > _ONE_EPSILON:
		v = 1
	elif v < _MINUS_ONE_EPSILON:
		v = -1
	return v


class Transform(NamedTuple):

	"""2x2 transformation matrix plus offset, a.k.a. Affine transform.
	Transform instances are immutable: all transforming methods, eg.
	rotate(), return a new Transform instance.

	:Example:

		>>> t = Transform()
		>>> t
		<Transform [1 0 0 1 0 0]>
		>>> t.scale(2)
		<Transform [2 0 0 2 0 0]>
		>>> t.scale(2.5, 5.5)
		<Transform [2.5 0 0 5.5 0 0]>
		>>>
		>>> t.scale(2, 3).transformPoint((100, 100))
		(200, 300)

	Transform's constructor takes six arguments, all of which are
	optional, and can be used as keyword arguments::

		>>> Transform(12)
		<Transform [12 0 0 1 0 0]>
		>>> Transform(dx=12)
		<Transform [1 0 0 1 12 0]>
		>>> Transform(yx=12)
		<Transform [1 0 12 1 0 0]>

	Transform instances also behave like sequences of length 6::

		>>> len(Identity)
		6
		>>> list(Identity)
		[1, 0, 0, 1, 0, 0]
		>>> tuple(Identity)
		(1, 0, 0, 1, 0, 0)

	Transform instances are comparable::

		>>> t1 = Identity.scale(2, 3).translate(4, 6)
		>>> t2 = Identity.translate(8, 18).scale(2, 3)
		>>> t1 == t2
		1

	But beware of floating point rounding errors::

		>>> t1 = Identity.scale(0.2, 0.3).translate(0.4, 0.6)
		>>> t2 = Identity.translate(0.08, 0.18).scale(0.2, 0.3)
		>>> t1
		<Transform [0.2 0 0 0.3 0.08 0.18]>
		>>> t2
		<Transform [0.2 0 0 0.3 0.08 0.18]>
		>>> t1 == t2
		0

	Transform instances are hashable, meaning you can use them as
	keys in dictionaries::

		>>> d = {Scale(12, 13): None}
		>>> d
		{<Transform [12 0 0 13 0 0]>: None}

	But again, beware of floating point rounding errors::

		>>> t1 = Identity.scale(0.2, 0.3).translate(0.4, 0.6)
		>>> t2 = Identity.translate(0.08, 0.18).scale(0.2, 0.3)
		>>> t1
		<Transform [0.2 0 0 0.3 0.08 0.18]>
		>>> t2
		<Transform [0.2 0 0 0.3 0.08 0.18]>
		>>> d = {t1: None}
		>>> d
		{<Transform [0.2 0 0 0.3 0.08 0.18]>: None}
		>>> d[t2]
		Traceback (most recent call last):
		  File "<stdin>", line 1, in ?
		KeyError: <Transform [0.2 0 0 0.3 0.08 0.18]>
	"""

	xx: float = 1
	xy: float = 0
	yx: float = 0
	yy: float = 1
	dx: float = 0
	dy: float = 0

	def transformPoint(self, p):
		"""Transform a point.

		:Example:

			>>> t = Transform()
			>>> t = t.scale(2.5, 5.5)
			>>> t.transformPoint((100, 100))
			(250.0, 550.0)
		"""
		(x, y) = p
		xx, xy, yx, yy, dx, dy = self
		return (xx*x + yx*y + dx, xy*x + yy*y + dy)

	def transformPoints(self, points):
		"""Transform a list of points.

		:Example:

			>>> t = Scale(2, 3)
			>>> t.transformPoints([(0, 0), (0, 100), (100, 100), (100, 0)])
			[(0, 0), (0, 300), (200, 300), (200, 0)]
			>>>
		"""
		xx, xy, yx, yy, dx, dy = self
		return [(xx*x + yx*y + dx, xy*x + yy*y + dy) for x, y in points]

	def transformVector(self, v):
		"""Transform an (dx, dy) vector, treating translation as zero.

		:Example:

			>>> t = Transform(2, 0, 0, 2, 10, 20)
			>>> t.transformVector((3, -4))
			(6, -8)
			>>>
		"""
		(dx, dy) = v
		xx, xy, yx, yy = self[:4]
		return (xx*dx + yx*dy, xy*dx + yy*dy)

	def transformVectors(self, vectors):
		"""Transform a list of (dx, dy) vector, treating translation as zero.

		:Example:
			>>> t = Transform(2, 0, 0, 2, 10, 20)
			>>> t.transformVectors([(3, -4), (5, -6)])
			[(6, -8), (10, -12)]
			>>>
		"""
		xx, xy, yx, yy = self[:4]
		return [(xx*dx + yx*dy, xy*dx + yy*dy) for dx, dy in vectors]

	def translate(self, x=0, y=0):
		"""Return a new transformation, translated (offset) by x, y.

		:Example:
			>>> t = Transform()
			>>> t.translate(20, 30)
			<Transform [1 0 0 1 20 30]>
			>>>
		"""
		return self.transform((1, 0, 0, 1, x, y))

	def scale(self, x=1, y=None):
		"""Return a new transformation, scaled by x, y. The 'y' argument
		may be None, which implies to use the x value for y as well.

		:Example:
			>>> t = Transform()
			>>> t.scale(5)
			<Transform [5 0 0 5 0 0]>
			>>> t.scale(5, 6)
			<Transform [5 0 0 6 0 0]>
			>>>
		"""
		if y is None:
			y = x
		return self.transform((x, 0, 0, y, 0, 0))

	def rotate(self, angle):
		"""Return a new transformation, rotated by 'angle' (radians).

		:Example:
			>>> import math
			>>> t = Transform()
			>>> t.rotate(math.pi / 2)
			<Transform [0 1 -1 0 0 0]>
			>>>
		"""
		import math
		c = _normSinCos(math.cos(angle))
		s = _normSinCos(math.sin(angle))
		return self.transform((c, s, -s, c, 0, 0))

	def skew(self, x=0, y=0):
		"""Return a new transformation, skewed by x and y.

		:Example:
			>>> import math
			>>> t = Transform()
			>>> t.skew(math.pi / 4)
			<Transform [1 0 1 1 0 0]>
			>>>
		"""
		import math
		return self.transform((1, math.tan(y), math.tan(x), 1, 0, 0))

	def transform(self, other):
		"""Return a new transformation, transformed by another
		transformation.

		:Example:
			>>> t = Transform(2, 0, 0, 3, 1, 6)
			>>> t.transform((4, 3, 2, 1, 5, 6))
			<Transform [8 9 4 3 11 24]>
			>>>
		"""
		xx1, xy1, yx1, yy1, dx1, dy1 = other
		xx2, xy2, yx2, yy2, dx2, dy2 = self
		return self.__class__(
				xx1*xx2 + xy1*yx2,
				xx1*xy2 + xy1*yy2,
				yx1*xx2 + yy1*yx2,
				yx1*xy2 + yy1*yy2,
				xx2*dx1 + yx2*dy1 + dx2,
				xy2*dx1 + yy2*dy1 + dy2)

	def reverseTransform(self, other):
		"""Return a new transformation, which is the other transformation
		transformed by self. self.reverseTransform(other) is equivalent to
		other.transform(self).

		:Example:
			>>> t = Transform(2, 0, 0, 3, 1, 6)
			>>> t.reverseTransform((4, 3, 2, 1, 5, 6))
			<Transform [8 6 6 3 21 15]>
			>>> Transform(4, 3, 2, 1, 5, 6).transform((2, 0, 0, 3, 1, 6))
			<Transform [8 6 6 3 21 15]>
			>>>
		"""
		xx1, xy1, yx1, yy1, dx1, dy1 = self
		xx2, xy2, yx2, yy2, dx2, dy2 = other
		return self.__class__(
				xx1*xx2 + xy1*yx2,
				xx1*xy2 + xy1*yy2,
				yx1*xx2 + yy1*yx2,
				yx1*xy2 + yy1*yy2,
				xx2*dx1 + yx2*dy1 + dx2,
				xy2*dx1 + yy2*dy1 + dy2)

	def inverse(self):
		"""Return the inverse transformation.

		:Example:
			>>> t = Identity.translate(2, 3).scale(4, 5)
			>>> t.transformPoint((10, 20))
			(42, 103)
			>>> it = t.inverse()
			>>> it.transformPoint((42, 103))
			(10.0, 20.0)
			>>>
		"""
		if self == Identity:
			return self
		xx, xy, yx, yy, dx, dy = self
		det = xx*yy - yx*xy
		xx, xy, yx, yy = yy/det, -xy/det, -yx/det, xx/det
		dx, dy = -xx*dx - yx*dy, -xy*dx - yy*dy
		return self.__class__(xx, xy, yx, yy, dx, dy)

	def toPS(self):
		"""Return a PostScript representation

		:Example:

			>>> t = Identity.scale(2, 3).translate(4, 5)
			>>> t.toPS()
			'[2 0 0 3 8 15]'
			>>>
		"""
		return "[%s %s %s %s %s %s]" % self

	def __bool__(self):
		"""Returns True if transform is not identity, False otherwise.

		:Example:

			>>> bool(Identity)
			False
			>>> bool(Transform())
			False
			>>> bool(Scale(1.))
			False
			>>> bool(Scale(2))
			True
			>>> bool(Offset())
			False
			>>> bool(Offset(0))
			False
			>>> bool(Offset(2))
			True
		"""
		return self != Identity

	def __repr__(self):
		return "<%s [%g %g %g %g %g %g]>" % ((self.__class__.__name__,) + self)


Identity = Transform()

def Offset(x=0, y=0):
	"""Return the identity transformation offset by x, y.

	:Example:
		>>> Offset(2, 3)
		<Transform [1 0 0 1 2 3]>
		>>>
	"""
	return Transform(1, 0, 0, 1, x, y)

def Scale(x, y=None):
	"""Return the identity transformation scaled by x, y. The 'y' argument
	may be None, which implies to use the x value for y as well.

	:Example:
		>>> Scale(2, 3)
		<Transform [2 0 0 3 0 0]>
		>>>
	"""
	if y is None:
		y = x
	return Transform(x, 0, 0, y, 0, 0)


if __name__ == "__main__":
	import sys
	import doctest
	sys.exit(doctest.testmod().failed)
