from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import AbstractPointPen
from fontTools.pens.recordingPen import RecordingPen


class _PassThruComponentsMixin(object):

    def addComponent(self, glyphName, transformation, **kwargs):
        self._outPen.addComponent(glyphName, transformation, **kwargs)


class FilterPen(_PassThruComponentsMixin, AbstractPen):

    """ Base class for pens that apply some transformation to the coordinates
    they receive and pass them to another pen.

    You can override any of its methods. The default implementation does
    nothing, but passes the commands unmodified to the other pen.

    >>> from fontTools.pens.recordingPen import RecordingPen
    >>> rec = RecordingPen()
    >>> pen = FilterPen(rec)
    >>> v = iter(rec.value)

    >>> pen.moveTo((0, 0))
    >>> next(v)
    ('moveTo', ((0, 0),))

    >>> pen.lineTo((1, 1))
    >>> next(v)
    ('lineTo', ((1, 1),))

    >>> pen.curveTo((2, 2), (3, 3), (4, 4))
    >>> next(v)
    ('curveTo', ((2, 2), (3, 3), (4, 4)))

    >>> pen.qCurveTo((5, 5), (6, 6), (7, 7), (8, 8))
    >>> next(v)
    ('qCurveTo', ((5, 5), (6, 6), (7, 7), (8, 8)))

    >>> pen.closePath()
    >>> next(v)
    ('closePath', ())

    >>> pen.moveTo((9, 9))
    >>> next(v)
    ('moveTo', ((9, 9),))

    >>> pen.endPath()
    >>> next(v)
    ('endPath', ())

    >>> pen.addComponent('foo', (1, 0, 0, 1, 0, 0))
    >>> next(v)
    ('addComponent', ('foo', (1, 0, 0, 1, 0, 0)))
    """

    def __init__(self, outPen):
        self._outPen = outPen

    def moveTo(self, pt):
        self._outPen.moveTo(pt)

    def lineTo(self, pt):
        self._outPen.lineTo(pt)

    def curveTo(self, *points):
        self._outPen.curveTo(*points)

    def qCurveTo(self, *points):
        self._outPen.qCurveTo(*points)

    def closePath(self):
        self._outPen.closePath()

    def endPath(self):
        self._outPen.endPath()


class ContourFilterPen(_PassThruComponentsMixin, RecordingPen):
    """A "buffered" filter pen that accumulates contour data, passes
    it through a ``filterContour`` method when the contour is closed or ended,
    and finally draws the result with the output pen.

    Components are passed through unchanged.
    """

    def __init__(self, outPen):
        super(ContourFilterPen, self).__init__()
        self._outPen = outPen

    def closePath(self):
        super(ContourFilterPen, self).closePath()
        self._flushContour()

    def endPath(self):
        super(ContourFilterPen, self).endPath()
        self._flushContour()

    def _flushContour(self):
        result = self.filterContour(self.value)
        if result is not None:
            self.value = result
        self.replay(self._outPen)
        self.value = []

    def filterContour(self, contour):
        """Subclasses must override this to perform the filtering.

        The contour is a list of pen (operator, operands) tuples.
        Operators are strings corresponding to the AbstractPen methods:
        "moveTo", "lineTo", "curveTo", "qCurveTo", "closePath" and
        "endPath". The operands are the positional arguments that are
        passed to each method.

        If the method doesn't return a value (i.e. returns None), it's
        assumed that the argument was modified in-place.
        Otherwise, the return value is drawn with the output pen.
        """
        return  # or return contour


class FilterPointPen(_PassThruComponentsMixin, AbstractPointPen):
    """ Baseclass for point pens that apply some transformation to the
    coordinates they receive and pass them to another point pen.

    You can override any of its methods. The default implementation does
    nothing, but passes the commands unmodified to the other pen.

    >>> from fontTools.pens.recordingPen import RecordingPointPen
    >>> rec = RecordingPointPen()
    >>> pen = FilterPointPen(rec)
    >>> v = iter(rec.value)
    >>> pen.beginPath(identifier="abc")
    >>> next(v)
    ('beginPath', (), {'identifier': 'abc'})
    >>> pen.addPoint((1, 2), "line", False)
    >>> next(v)
    ('addPoint', ((1, 2), 'line', False, None), {})
    >>> pen.addComponent("a", (2, 0, 0, 2, 10, -10), identifier="0001")
    >>> next(v)
    ('addComponent', ('a', (2, 0, 0, 2, 10, -10)), {'identifier': '0001'})
    >>> pen.endPath()
    >>> next(v)
    ('endPath', (), {})
    """

    def __init__(self, outPointPen):
        self._outPen = outPointPen

    def beginPath(self, **kwargs):
        self._outPen.beginPath(**kwargs)

    def endPath(self):
        self._outPen.endPath()

    def addPoint(self, pt, segmentType=None, smooth=False, name=None, **kwargs):
        self._outPen.addPoint(pt, segmentType, smooth, name, **kwargs)
