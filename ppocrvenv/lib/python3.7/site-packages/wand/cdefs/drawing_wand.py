""":mod:`wand.cdefs.drawing_wand` --- Drawing-Wand definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 0.5.0
"""
from ctypes import (POINTER, c_void_p, c_char_p, c_double, c_int, c_uint,
                    c_size_t, c_ubyte, c_ulong)
from wand.cdefs.wandtypes import c_ssize_t
from wand.cdefs.structures import PointInfo

__all__ = ('load',)


def load(lib, IM_VERSION):
    """Define Drawing Wand methods. The ImageMagick version is given as a
    second argument for comparison. This will quick to determine which methods
    are available from the library, and can be implemented as::

        if IM_VERSION < 0x700:
            # ... do ImageMagick-6 methods ...
        else
            # ... do ImageMagick-7 methods ...

    .. seealso::

        #include "wand/drawing-wand.h"
        // Or
        #include "MagickWand/drawing-wand.h"

    :param lib: the loaded ``MagickWand`` library
    :type lib: :class:`ctypes.CDLL`
    :param IM_VERSION: the ImageMagick version number (i.e. 0x0689)
    :type IM_VERSION: :class:`numbers.Integral`

    .. versionadded:: 0.5.0

    """
    is_im_6 = IM_VERSION < 0x700
    is_im_7 = IM_VERSION >= 0x700
    lib.NewDrawingWand.restype = c_void_p
    lib.CloneDrawingWand.argtypes = [c_void_p]
    lib.CloneDrawingWand.restype = c_void_p
    lib.DestroyDrawingWand.argtypes = [c_void_p]
    lib.DestroyDrawingWand.restype = c_void_p
    lib.IsDrawingWand.argtypes = [c_void_p]
    lib.IsDrawingWand.restype = c_int
    lib.DrawGetException.argtypes = [c_void_p, POINTER(c_int)]
    lib.DrawGetException.restype = c_void_p
    lib.DrawClearException.argtypes = [c_void_p]
    lib.DrawClearException.restype = c_int
    lib.DrawAffine.argtypes = [c_void_p, c_void_p]
    if is_im_7:
        lib.DrawAlpha.argtypes = [c_void_p, c_double, c_double, c_int]
    lib.DrawComment.argtypes = [c_void_p, c_char_p]
    lib.DrawComposite.argtypes = [
        c_void_p, c_int, c_double, c_double, c_double, c_double, c_void_p
    ]
    lib.DrawComposite.restype = c_uint
    lib.DrawSetBorderColor.argtypes = [c_void_p, c_void_p]
    lib.DrawSetClipPath.argtypes = [c_void_p, c_char_p]
    lib.DrawSetClipPath.restype = c_int
    lib.DrawSetClipRule.argtypes = [c_void_p, c_uint]
    lib.DrawSetClipUnits.argtypes = [c_void_p, c_uint]
    lib.DrawSetFont.argtypes = [c_void_p, c_char_p]
    lib.DrawSetFontFamily.argtypes = [c_void_p, c_char_p]
    lib.DrawSetFontFamily.restype = c_uint
    lib.DrawSetFontResolution.argtypes = [c_void_p, c_double, c_double]
    lib.DrawSetFontResolution.restype = c_uint
    lib.DrawSetFontSize.argtypes = [c_void_p, c_double]
    lib.DrawSetFontStretch.argtypes = [c_void_p, c_int]
    lib.DrawSetFontStyle.argtypes = [c_void_p, c_int]
    lib.DrawSetFontWeight.argtypes = [c_void_p, c_size_t]
    lib.DrawSetFillColor.argtypes = [c_void_p, c_void_p]
    lib.DrawSetFillOpacity.argtypes = [c_void_p, c_double]
    lib.DrawSetFillPatternURL.argtypes = [c_void_p, c_char_p]
    lib.DrawSetFillPatternURL.restype = c_uint
    lib.DrawSetFillRule.argtypes = [c_void_p, c_uint]
    lib.DrawSetOpacity.argtypes = [c_void_p, c_double]
    lib.DrawSetStrokeAntialias.argtypes = [c_void_p, c_int]
    lib.DrawSetStrokeColor.argtypes = [c_void_p, c_void_p]
    lib.DrawSetStrokeDashArray.argtypes = [
        c_void_p, c_size_t, POINTER(c_double)
    ]
    lib.DrawSetStrokeDashOffset.argtypes = [c_void_p, c_double]
    lib.DrawSetStrokeLineCap.argtypes = [c_void_p, c_int]
    lib.DrawSetStrokeLineJoin.argtypes = [c_void_p, c_int]
    lib.DrawSetStrokeMiterLimit.argtypes = [c_void_p, c_size_t]
    lib.DrawSetStrokeOpacity.argtypes = [c_void_p, c_double]
    lib.DrawSetStrokePatternURL.argtypes = [c_void_p, c_char_p]
    lib.DrawSetStrokePatternURL.restype = c_uint
    lib.DrawSetStrokeWidth.argtypes = [c_void_p, c_double]
    lib.DrawSetTextAlignment.argtypes = [c_void_p, c_int]
    lib.DrawSetTextAntialias.argtypes = [c_void_p, c_int]
    lib.DrawSetTextDecoration.argtypes = [c_void_p, c_int]
    try:
        lib.DrawSetTextDirection.argtypes = [c_void_p, c_int]
    except AttributeError:
        lib.DrawSetTextDirection = None
    lib.DrawSetTextEncoding.argtypes = [c_void_p, c_char_p]
    try:
        lib.DrawSetTextInterlineSpacing.argtypes = [c_void_p, c_double]
    except AttributeError:
        lib.DrawSetTextInterlineSpacing = None
    lib.DrawSetTextInterwordSpacing.argtypes = [c_void_p, c_double]
    lib.DrawSetTextKerning.argtypes = [c_void_p, c_double]
    lib.DrawSetTextUnderColor.argtypes = [c_void_p, c_void_p]
    lib.DrawSetVectorGraphics.argtypes = [c_void_p, c_char_p]
    lib.DrawSetVectorGraphics.restype = c_int
    lib.DrawResetVectorGraphics.argtypes = [c_void_p]
    lib.DrawSetViewbox.argtypes = [
        c_void_p, c_ssize_t, c_ssize_t, c_ssize_t, c_ssize_t
    ]
    lib.DrawGetBorderColor.argtypes = [c_void_p, c_void_p]
    lib.DrawGetClipPath.argtypes = [c_void_p]
    lib.DrawGetClipPath.restype = c_void_p
    lib.DrawGetClipRule.argtypes = [c_void_p]
    lib.DrawGetClipRule.restype = c_uint
    lib.DrawGetClipUnits.argtypes = [c_void_p]
    lib.DrawGetClipUnits.restype = c_uint
    lib.DrawGetFillColor.argtypes = [c_void_p, c_void_p]
    lib.DrawGetFillOpacity.argtypes = [c_void_p]
    lib.DrawGetFillOpacity.restype = c_double
    lib.DrawGetFillRule.argtypes = [c_void_p]
    lib.DrawGetFillRule.restype = c_uint
    lib.DrawGetOpacity.argtypes = [c_void_p]
    lib.DrawGetOpacity.restype = c_double
    lib.DrawGetStrokeAntialias.argtypes = [c_void_p]
    lib.DrawGetStrokeAntialias.restype = c_int
    lib.DrawGetStrokeColor.argtypes = [c_void_p, c_void_p]
    lib.DrawGetStrokeDashArray.argtypes = [c_void_p, POINTER(c_size_t)]
    lib.DrawGetStrokeDashArray.restype = POINTER(c_double)
    lib.DrawGetStrokeDashOffset.argtypes = [c_void_p]
    lib.DrawGetStrokeDashOffset.restype = c_double
    lib.DrawGetStrokeLineCap.argtypes = [c_void_p]
    lib.DrawGetStrokeLineCap.restype = c_int
    lib.DrawGetStrokeLineJoin.argtypes = [c_void_p]
    lib.DrawGetStrokeLineJoin.restype = c_int
    lib.DrawGetStrokeMiterLimit.argtypes = [c_void_p]
    lib.DrawGetStrokeMiterLimit.restype = c_size_t
    lib.DrawGetStrokeOpacity.argtypes = [c_void_p]
    lib.DrawGetStrokeOpacity.restype = c_double
    lib.DrawGetStrokeWidth.argtypes = [c_void_p]
    lib.DrawGetStrokeWidth.restype = c_double
    lib.DrawGetFont.argtypes = [c_void_p]
    lib.DrawGetFont.restype = c_void_p
    lib.DrawGetFontFamily.argtypes = [c_void_p]
    lib.DrawGetFontFamily.restype = c_void_p
    lib.DrawGetFontResolution.argtypes = [
        c_void_p, POINTER(c_double), POINTER(c_double)
    ]
    lib.DrawGetFontResolution.restype = c_uint
    lib.DrawGetFontSize.argtypes = [c_void_p]
    lib.DrawGetFontSize.restype = c_double
    lib.DrawGetFontStyle.argtypes = [c_void_p]
    lib.DrawGetFontStyle.restype = c_int
    lib.DrawGetFontWeight.argtypes = [c_void_p]
    lib.DrawGetFontWeight.restype = c_size_t
    lib.DrawGetFontStretch.argtypes = [c_void_p]
    lib.DrawGetFontStretch.restype = c_int
    lib.DrawGetTextAlignment.argtypes = [c_void_p]
    lib.DrawGetTextAlignment.restype = c_int
    lib.DrawGetTextAntialias.argtypes = [c_void_p]
    lib.DrawGetTextAntialias.restype = c_int
    lib.DrawGetTextDecoration.argtypes = [c_void_p]
    lib.DrawGetTextDecoration.restype = c_int
    try:
        lib.DrawGetTextDirection.argtypes = [c_void_p]
        lib.DrawGetTextDirection.restype = c_int
    except AttributeError:
        lib.DrawGetTextDirection = None
    lib.DrawGetTextEncoding.argtypes = [c_void_p]
    lib.DrawGetTextEncoding.restype = c_void_p
    try:
        lib.DrawGetTextInterlineSpacing.argtypes = [c_void_p]
        lib.DrawGetTextInterlineSpacing.restype = c_double
    except AttributeError:
        lib.DrawGetTextInterlineSpacing = None
    lib.DrawGetTextInterwordSpacing.argtypes = [c_void_p]
    lib.DrawGetTextInterwordSpacing.restype = c_double
    lib.DrawGetTextKerning.argtypes = [c_void_p]
    lib.DrawGetTextKerning.restype = c_double
    lib.DrawGetTextUnderColor.argtypes = [c_void_p, c_void_p]
    lib.DrawGetVectorGraphics.argtypes = [c_void_p]
    lib.DrawGetVectorGraphics.restype = c_void_p
    lib.DrawSetGravity.argtypes = [c_void_p, c_int]
    lib.DrawGetGravity.argtypes = [c_void_p]
    lib.DrawGetGravity.restype = c_int
    lib.ClearDrawingWand.argtypes = [c_void_p]
    lib.DrawAnnotation.argtypes = [
        c_void_p, c_double, c_double, POINTER(c_ubyte)
    ]
    lib.DrawArc.argtypes = [
        c_void_p, c_double, c_double, c_double, c_double, c_double, c_double
    ]
    lib.DrawBezier.argtypes = [c_void_p, c_ulong, POINTER(PointInfo)]
    lib.DrawCircle.argtypes = [
        c_void_p, c_double, c_double, c_double, c_double
    ]
    lib.DrawColor.argtypes = [c_void_p, c_double, c_double, c_uint]
    lib.DrawEllipse.argtypes = [
        c_void_p, c_double, c_double, c_double, c_double, c_double, c_double
    ]
    lib.DrawLine.argtypes = [c_void_p, c_double, c_double, c_double, c_double]
    if is_im_6:
        lib.DrawMatte.argtypes = [c_void_p, c_double, c_double, c_int]
    else:
        lib.DrawMatte = None
    lib.DrawPathClose.argtypes = [c_void_p]
    lib.DrawPathCurveToAbsolute.argtypes = [
        c_void_p, c_double, c_double, c_double, c_double, c_double, c_double
    ]
    lib.DrawPathCurveToRelative.argtypes = [
        c_void_p, c_double, c_double, c_double, c_double, c_double, c_double
    ]
    lib.DrawPathCurveToQuadraticBezierAbsolute.argtypes = [
        c_void_p, c_double, c_double, c_double, c_double
    ]
    lib.DrawPathCurveToQuadraticBezierRelative.argtypes = [
        c_void_p, c_double, c_double, c_double, c_double
    ]
    lib.DrawPathCurveToQuadraticBezierSmoothAbsolute.argtypes = [
        c_void_p, c_double, c_double
    ]
    lib.DrawPathCurveToQuadraticBezierSmoothRelative.argtypes = [
        c_void_p, c_double, c_double
    ]
    lib.DrawPathCurveToSmoothAbsolute.argtypes = [
        c_void_p, c_double, c_double, c_double, c_double
    ]
    lib.DrawPathCurveToSmoothRelative.argtypes = [
        c_void_p, c_double, c_double, c_double, c_double
    ]
    lib.DrawPathEllipticArcAbsolute.argtypes = [
        c_void_p, c_double, c_double, c_double, c_uint, c_uint, c_double,
        c_double
    ]
    lib.DrawPathEllipticArcRelative.argtypes = [
        c_void_p, c_double, c_double, c_double, c_uint, c_uint, c_double,
        c_double
    ]
    lib.DrawPathFinish.argtypes = [c_void_p]
    lib.DrawPathLineToAbsolute.argtypes = [c_void_p, c_double, c_double]
    lib.DrawPathLineToRelative.argtypes = [c_void_p, c_double, c_double]
    lib.DrawPathLineToHorizontalAbsolute.argtypes = [c_void_p, c_double]
    lib.DrawPathLineToHorizontalRelative.argtypes = [c_void_p, c_double]
    lib.DrawPathLineToVerticalAbsolute.argtypes = [c_void_p, c_double]
    lib.DrawPathLineToVerticalRelative.argtypes = [c_void_p, c_double]
    lib.DrawPathMoveToAbsolute.argtypes = [c_void_p, c_double, c_double]
    lib.DrawPathMoveToRelative.argtypes = [c_void_p, c_double, c_double]
    lib.DrawPathStart.argtypes = [c_void_p]
    lib.DrawPoint.argtypes = [c_void_p, c_double, c_double]
    lib.DrawPolygon.argtypes = [c_void_p, c_ulong, POINTER(PointInfo)]
    lib.DrawPolyline.argtypes = [c_void_p, c_ulong, POINTER(PointInfo)]
    lib.DrawRotate.argtypes = [c_void_p, c_double]
    lib.DrawRectangle.argtypes = [
        c_void_p, c_double, c_double, c_double, c_double
    ]
    lib.DrawRoundRectangle.argtypes = [
        c_void_p, c_double, c_double, c_double, c_double, c_double, c_double
    ]
    lib.DrawScale.argtypes = [c_void_p, c_double, c_double]
    lib.DrawSkewX.argtypes = [c_void_p, c_double]
    lib.DrawSkewY.argtypes = [c_void_p, c_double]
    lib.DrawTranslate.argtypes = [c_void_p, c_double, c_double]
    lib.PushDrawingWand.argtypes = [c_void_p]
    lib.PushDrawingWand.restype = c_uint
    lib.DrawPushClipPath.argtypes = [c_void_p, c_char_p]
    lib.DrawPushDefs.argtypes = [c_void_p]
    lib.DrawPushPattern.argtypes = [
        c_void_p, c_char_p, c_double, c_double, c_double, c_double
    ]
    lib.DrawPushClipPath.restype = c_uint
    lib.PopDrawingWand.argtypes = [c_void_p]
    lib.PopDrawingWand.restype = c_uint
    lib.DrawPopClipPath.argtypes = [c_void_p]
    lib.DrawPopDefs.argtypes = [c_void_p]
    lib.DrawPopPattern.argtypes = [c_void_p]
