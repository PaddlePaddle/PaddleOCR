""":mod:`wand.cdefs.magick_image` --- Magick-Image definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 0.5.0
"""
from ctypes import (CFUNCTYPE, POINTER, c_void_p, c_int, c_size_t, c_double,
                    c_char_p, c_ubyte, c_bool)
from wand.cdefs.wandtypes import c_ssize_t

__all__ = ('MagickProgressMonitor', 'load')


#: (:class:`ctypes.CFUNCTYPE`) a function type to allow ImageMagick's progress
#: monitoring to call a python function. For example::
#:
#:     def myCallBack(filename, offset, size, user_data):
#:         print(filename, offset, '/', size)
#:         return True
#:     iMyCallBack = MagickProgressMonitor(myCallBack)
#:     library.MagickSetImageProgressMonitor(wand_instance,
#:                                           iMyCallBack,
#:                                           None)
#:
#: .. note::
#:
#:     TODO - Move to isolated module. This shouldn't be defined at time of
#:     mload. It might be wiser to create a method to allow the user to ask for
#:     C-function-type.
MagickProgressMonitor = CFUNCTYPE(c_bool,
                                  c_char_p,
                                  c_ssize_t,
                                  c_size_t,
                                  c_void_p)


def load(lib, IM_VERSION):
    """Define Magick Image methods. The ImageMagick version is given as a
    second argument for comparison. This will quick to determine which methods
    are available from the library, and can be implemented as::

        if IM_VERSION < 0x700:
            # ... do ImageMagick-6 methods ...
        else
            # ... do ImageMagick-7 methods ...

    .. seealso::

        #include "wand/magick-image.h"
        // Or
        #include "MagickWand/magick-image.h"

    :param lib: the loaded ``MagickWand`` library
    :type lib: :class:`ctypes.CDLL`
    :param IM_VERSION: the ImageMagick version number (i.e. 0x0689)
    :type IM_VERSION: :class:`numbers.Integral`

    .. versionadded:: 0.5.0

    """
    is_im_6 = IM_VERSION < 0x700
    is_im_7 = IM_VERSION >= 0x700
    lib.GetImageFromMagickWand.argtypes = [c_void_p]
    lib.GetImageFromMagickWand.restype = c_void_p
    lib.MagickAdaptiveBlurImage.argtypes = [c_void_p, c_double, c_double]
    lib.MagickAdaptiveBlurImage.restype = c_bool
    if is_im_6:
        lib.MagickAdaptiveBlurImageChannel.argtypes = [
            c_void_p, c_int, c_double, c_double
        ]
        lib.MagickAdaptiveBlurImageChannel.restype = c_bool
    lib.MagickAdaptiveResizeImage.argtypes = [c_void_p, c_size_t, c_size_t]
    lib.MagickAdaptiveResizeImage.restype = c_bool
    lib.MagickAdaptiveSharpenImage.argtypes = [c_void_p, c_double, c_double]
    lib.MagickAdaptiveSharpenImage.restype = c_bool
    if is_im_6:
        lib.MagickAdaptiveSharpenImageChannel.argtypes = [
            c_void_p, c_int, c_double, c_double
        ]
        lib.MagickAdaptiveSharpenImageChannel.restype = c_bool
    if is_im_6:
        lib.MagickAdaptiveThresholdImage.argtypes = [
            c_void_p, c_size_t, c_size_t, c_ssize_t
        ]
        lib.MagickAdaptiveThresholdImage.restype = c_bool
    else:
        lib.MagickAdaptiveThresholdImage.argtypes = [
            c_void_p, c_size_t, c_size_t, c_double
        ]
        lib.MagickAdaptiveThresholdImage.restype = c_bool
    lib.MagickAddImage.argtypes = [c_void_p, c_void_p]
    lib.MagickAddImage.restype = c_bool
    if is_im_6:
        lib.MagickAddNoiseImage.argtypes = [c_void_p, c_int]
        lib.MagickAddNoiseImage.restype = c_bool
    else:
        lib.MagickAddNoiseImage.argtypes = [c_void_p, c_int, c_double]
        lib.MagickAddNoiseImage.restype = c_bool
    if is_im_6:
        lib.MagickAddNoiseImageChannel.argtypes = [c_void_p, c_int, c_int]
        lib.MagickAddNoiseImageChannel.restype = c_bool
    lib.MagickAffineTransformImage.argtypes = [c_void_p, c_void_p]
    lib.MagickAffineTransformImage.restype = c_bool
    lib.MagickAnnotateImage.argtypes = [
        c_void_p, c_void_p, c_double, c_double, c_double, c_char_p
    ]
    lib.MagickAnnotateImage.restype = c_int
    lib.MagickAnimateImages.argtypes = [c_void_p, c_char_p]
    lib.MagickAnimateImages.restype = c_bool
    lib.MagickAppendImages.argtypes = [c_void_p, c_int]
    lib.MagickAppendImages.restype = c_void_p
    lib.MagickAutoGammaImage.argtypes = [c_void_p]
    lib.MagickAutoGammaImage.restype = c_bool
    if is_im_6:
        lib.MagickAutoGammaImageChannel.argtypes = [c_void_p, c_int]
        lib.MagickAutoGammaImageChannel.restype = c_bool
    lib.MagickAutoLevelImage.argtypes = [c_void_p]
    lib.MagickAutoLevelImage.restype = c_bool
    if is_im_6:
        lib.MagickAutoLevelImageChannel.argtypes = [c_void_p, c_int]
        lib.MagickAutoLevelImageChannel.restype = c_bool
    try:
        lib.MagickAutoOrientImage.argtypes = [c_void_p]
    except AttributeError:
        # MagickAutoOrientImage was added in 6.8.9+, we have a fallback
        # function, so we pass silently if we cannot import it.
        pass
    if IM_VERSION >= 0x708:
        try:
            lib.MagickAutoThresholdImage.argtypes = [c_void_p, c_int]
            lib.MagickAutoThresholdImage.restype = c_bool
        except AttributeError:
            lib.MagickAutoThresholdImage = None
    else:
        lib.MagickAutoThresholdImage = None
    lib.MagickBlackThresholdImage.argtypes = [c_void_p, c_void_p]
    lib.MagickBlackThresholdImage.restype = c_bool
    lib.MagickBlueShiftImage.argtypes = [c_void_p, c_double]
    lib.MagickBlueShiftImage.restype = c_bool
    lib.MagickBlurImage.argtypes = [c_void_p, c_double, c_double]
    lib.MagickBlurImage.restype = c_bool
    if is_im_6:
        lib.MagickBlurImageChannel.argtypes = [c_void_p, c_int, c_double,
                                               c_double]
        lib.MagickBlurImageChannel.restype = c_bool
    border_image_args = [c_void_p, c_void_p, c_size_t, c_size_t]
    if is_im_7:
        border_image_args.append(c_int)
    lib.MagickBorderImage.argtypes = border_image_args
    lib.MagickBorderImage.restype = c_bool
    lib.MagickBrightnessContrastImage.argtypes = [c_void_p, c_double, c_double]
    lib.MagickBrightnessContrastImage.restype = c_bool
    if is_im_6:
        lib.MagickBrightnessContrastImageChannel.argtypes = [
            c_void_p, c_int, c_double, c_double
        ]
        lib.MagickBrightnessContrastImageChannel.restype = c_bool
    if IM_VERSION >= 0x708:
        try:
            lib.MagickCannyEdgeImage.argtypes = [c_void_p, c_double, c_double,
                                                 c_double, c_double]
            lib.MagickCannyEdgeImage.restype = c_bool
        except AttributeError:
            lib.MagickCannyEdgeImage = None
    else:
        lib.MagickCannyEdgeImage = None
    lib.MagickCharcoalImage.argtypes = [c_void_p, c_double, c_double]
    lib.MagickCharcoalImage.restype = c_bool
    lib.MagickChopImage.argtypes = [
        c_void_p, c_size_t, c_size_t, c_ssize_t, c_ssize_t
    ]
    lib.MagickChopImage.restype = c_bool
    if is_im_7:
        try:
            lib.MagickCLAHEImage.argtypes = [c_void_p, c_size_t, c_size_t,
                                             c_double, c_double]
            lib.MagickCLAHEImage.restype = c_bool
        except AttributeError:
            lib.MagickCLAHEImage = None
    else:
        lib.MagickCLAHEImage = None
    lib.MagickClampImage.argtypes = [c_void_p]
    lib.MagickClampImage.restype = c_bool
    if is_im_6:
        lib.MagickClampImageChannel.argtypes = [c_void_p, c_int]
        lib.MagickClampImageChannel.restype = c_bool
    lib.MagickClipImage.argtypes = [c_void_p]
    lib.MagickClipImage.restype = c_bool
    lib.MagickClipImagePath.argtypes = [c_void_p, c_char_p, c_bool]
    lib.MagickClipImagePath.restype = c_bool
    if is_im_7:
        lib.MagickClutImage.argtypes = [c_void_p, c_void_p, c_int]
        lib.MagickClutImage.restype = c_bool
    else:
        lib.MagickClutImage.argtypes = [c_void_p, c_void_p]
        lib.MagickClutImage.restype = c_bool
    if is_im_6:
        lib.MagickClutImageChannel.argtypes = [c_void_p, c_int, c_void_p]
        lib.MagickClutImageChannel.restype = c_bool
    lib.MagickCoalesceImages.argtypes = [c_void_p]
    lib.MagickCoalesceImages.restype = c_void_p
    lib.MagickColorDecisionListImage.argtypes = [c_void_p, c_char_p]
    lib.MagickColorDecisionListImage.restype = c_bool
    lib.MagickColorizeImage.argtypes = [c_void_p, c_void_p, c_void_p]
    lib.MagickColorizeImage.restype = c_bool
    lib.MagickColorMatrixImage.argtypes = [c_void_p, c_void_p]
    lib.MagickColorMatrixImage.restype = c_bool
    if IM_VERSION > 0x709:
        lib.MagickColorThresholdImage.argtypes = [c_void_p, c_void_p, c_void_p]
        lib.MagickColorThresholdImage.restype = c_bool
    else:
        lib.MagickColorThresholdImage = None
    lib.MagickCommentImage.argtypes = [c_void_p, c_char_p]
    lib.MagickCommentImage.restype = c_bool
    lib.MagickCombineImages.argtypes = [c_void_p, c_int]
    lib.MagickCombineImages.restype = c_void_p
    if is_im_6:
        lib.MagickCompareImageChannels.argtypes = [
            c_void_p, c_void_p, c_int, c_double
        ]
        lib.MagickCompareImageChannels.restype = c_void_p
        lib.MagickCompareImageLayers.argtypes = [c_void_p, c_int]
        lib.MagickCompareImageLayers.restype = c_void_p
    lib.MagickCompareImages.argtypes = [
        c_void_p, c_void_p, c_int, POINTER(c_double)
    ]
    lib.MagickCompareImages.restype = c_void_p
    if is_im_6:
        try:
            lib.MagickCompareImageLayers.argtypes = [c_void_p, c_int]
            lib.MagickCompareImageLayers.restype = c_void_p
        except AttributeError:
            lib.MagickCompareImageLayers = None
    else:
        try:
            lib.MagickCompareImagesLayers.argtypes = [c_void_p, c_int]
            lib.MagickCompareImagesLayers.restype = c_void_p
        except AttributeError:
            lib.MagickCompareImagesLayers = None
    if IM_VERSION >= 0x708:
        try:
            lib.MagickComplexImages.argtypes = [c_void_p, c_int]
            lib.MagickComplexImages.restype = c_void_p
        except AttributeError:
            lib.MagickComplexImages = None
    else:
        lib.MagickComplexImages = None
    if is_im_6:
        lib.MagickCompositeImage.argtypes = [
            c_void_p, c_void_p, c_int, c_ssize_t, c_ssize_t
        ]
    else:
        lib.MagickCompositeImage.argtypes = [
            c_void_p, c_void_p, c_int, c_bool, c_ssize_t, c_ssize_t
        ]
    lib.MagickCompositeImage.restype = c_bool
    try:
        lib.MagickCompositeLayers.argtypes = [
            c_void_p, c_void_p, c_int, c_ssize_t, c_ssize_t
        ]
        lib.MagickCompositeLayers.restype = c_bool
    except AttributeError:
        lib.MagickCompositeLayers = None
    if is_im_6:
        lib.MagickCompositeImageChannel.argtypes = [
            c_void_p, c_int, c_void_p, c_int, c_ssize_t, c_ssize_t
        ]
        lib.MagickCompositeImageChannel.restype = c_bool
    else:
        lib.MagickCompositeImageChannel = None
    if IM_VERSION >= 0x708:
        try:
            lib.MagickConnectedComponentsImage.argtypes = [
                c_void_p, c_size_t, POINTER(c_void_p)
            ]
            lib.MagickConnectedComponentsImage.restype = c_bool
        except AttributeError:
            lib.MagickConnectedComponentsImage = None
    else:
        lib.MagickConnectedComponentsImage = None
    lib.MagickConstituteImage.argtypes = [
        c_void_p, c_size_t, c_size_t, c_char_p, c_int, c_void_p
    ]
    lib.MagickContrastImage.argtypes = [c_void_p, c_bool]
    lib.MagickContrastImage.restype = c_bool
    lib.MagickContrastStretchImage.argtypes = [c_void_p, c_double, c_double]
    if is_im_6:
        lib.MagickContrastStretchImageChannel.argtypes = [
            c_void_p, c_int, c_double, c_double
        ]
    else:
        lib.MagickContrastStretchImageChannel = None
    lib.MagickConvolveImage.argtypes = [c_void_p, c_size_t, c_double]
    lib.MagickConvolveImage.restype = c_bool
    if is_im_6:
        lib.MagickConvolveImageChannel.argtypes = [
            c_void_p, c_int, c_size_t, c_double
        ]
        lib.MagickConvolveImageChannel.restype = c_bool
    lib.MagickCropImage.argtypes = [
        c_void_p, c_size_t, c_size_t, c_ssize_t, c_ssize_t
    ]
    lib.MagickCropImage.restype = c_bool
    lib.MagickCycleColormapImage.argtypes = [c_void_p, c_ssize_t]
    lib.MagickCycleColormapImage.restype = c_bool
    lib.MagickDecipherImage.argtypes = [c_void_p, c_char_p]
    lib.MagickDecipherImage.restype = c_bool
    lib.MagickDeconstructImages.argtypes = [c_void_p]
    lib.MagickDeconstructImages.restype = c_void_p
    lib.MagickDeskewImage.argtypes = [c_void_p, c_double]
    lib.MagickDeskewImage.restype = c_bool
    lib.MagickDespeckleImage.argtypes = [c_void_p]
    lib.MagickDespeckleImage.restype = c_bool
    lib.MagickDestroyImage.argtypes = [c_void_p]
    lib.MagickDestroyImage.restype = c_void_p
    lib.MagickDisplayImage.argtypes = [c_void_p, c_char_p]
    lib.MagickDisplayImage.restype = c_bool
    lib.MagickDisplayImages.argtypes = [c_void_p, c_char_p]
    lib.MagickDisplayImages.restype = c_bool
    lib.MagickDistortImage.argtypes = [
        c_void_p, c_int, c_size_t, POINTER(c_double), c_int
    ]
    lib.MagickDistortImage.restype = c_int
    lib.MagickDrawImage.argtypes = [c_void_p, c_void_p]
    lib.MagickDrawImage.restype = c_int
    lib.MagickEdgeImage.argtypes = [c_void_p, c_double]
    lib.MagickEdgeImage.restype = c_bool
    lib.MagickEmbossImage.argtypes = [c_void_p, c_double, c_double]
    lib.MagickEmbossImage.restype = c_bool
    lib.MagickEncipherImage.argtypes = [c_void_p, c_char_p]
    lib.MagickEncipherImage.restype = c_bool
    lib.MagickEnhanceImage.argtypes = [c_void_p]
    lib.MagickEnhanceImage.restype = c_bool
    lib.MagickEqualizeImage.argtypes = [c_void_p]
    lib.MagickEqualizeImage.restype = c_bool
    if is_im_6:
        lib.MagickEqualizeImageChannel.argtypes = [c_void_p, c_int]
        lib.MagickEqualizeImageChannel.restype = c_bool
    lib.MagickEvaluateImage.argtypes = [c_void_p, c_int, c_double]
    if is_im_6:
        lib.MagickEvaluateImageChannel.argtypes = [
            c_void_p, c_int, c_int, c_double
        ]
    else:
        lib.MagickEvaluateImageChannel = None
    lib.MagickEvaluateImages.argtypes = [c_void_p, c_int]
    lib.MagickEvaluateImages.restype = c_void_p
    lib.MagickExportImagePixels.argtypes = [
        c_void_p, c_ssize_t, c_ssize_t, c_size_t, c_size_t, c_char_p, c_int,
        c_void_p
    ]
    lib.MagickExportImagePixels.restype = c_bool
    lib.MagickExtentImage.argtypes = [
        c_void_p, c_size_t, c_size_t, c_ssize_t, c_ssize_t
    ]
    lib.MagickExtentImage.restype = c_bool
    if is_im_6:
        lib.MagickFilterImage.argtypes = [c_void_p, c_void_p]
        lib.MagickFilterImage.restype = c_bool
        lib.MagickFilterImageChannel.argtypes = [c_void_p, c_int, c_void_p]
        lib.MagickFilterImageChannel.restype = c_bool
    lib.MagickFlipImage.argtypes = [c_void_p]
    lib.MagickFlipImage.restype = c_bool
    lib.MagickFloodfillPaintImage.argtypes = [
        c_void_p, c_int, c_void_p, c_double, c_void_p, c_ssize_t, c_ssize_t,
        c_bool
    ]
    lib.MagickFloodfillPaintImage.restype = c_bool
    lib.MagickFlopImage.argtypes = [c_void_p]
    lib.MagickFlopImage.restype = c_bool
    lib.MagickForwardFourierTransformImage.argtypes = [c_void_p, c_bool]
    lib.MagickForwardFourierTransformImage.restype = c_bool
    if is_im_6:
        lib.MagickFrameImage.argtypes = [
            c_void_p, c_void_p, c_size_t, c_size_t, c_ssize_t, c_ssize_t
        ]
    else:
        lib.MagickFrameImage.argtypes = [
            c_void_p, c_void_p, c_size_t, c_size_t, c_ssize_t, c_ssize_t,
            c_int
        ]
    lib.MagickFrameImage.restype = c_bool
    lib.MagickFunctionImage.argtypes = [
        c_void_p, c_int, c_size_t, POINTER(c_double)
    ]
    lib.MagickFunctionImage.restype = c_bool
    if is_im_6:
        lib.MagickFunctionImageChannel.argtypes = [
            c_void_p, c_int, c_int, c_size_t, POINTER(c_double)
        ]
        lib.MagickFunctionImageChannel.restype = c_bool
    else:
        lib.MagickFunctionImageChannel = None
    lib.MagickFxImage.argtypes = [c_void_p, c_char_p]
    lib.MagickFxImage.restype = c_void_p
    if is_im_6:
        lib.MagickFxImageChannel.argtypes = [c_void_p, c_int, c_char_p]
        lib.MagickFxImageChannel.restype = c_void_p
    else:
        lib.MagickFxImageChannel = None
    lib.MagickGammaImage.argtypes = [c_void_p, c_double]
    lib.MagickGammaImage.restype = c_bool
    if is_im_6:
        lib.MagickGammaImageChannel.argtypes = [c_void_p, c_int, c_double]
        lib.MagickGammaImageChannel.restype = c_bool
    else:
        lib.MagickGammaImageChannel = None
    lib.MagickGaussianBlurImage.argtypes = [c_void_p, c_double, c_double]
    lib.MagickGaussianBlurImage.restype = c_bool
    if is_im_6:
        lib.MagickGaussianBlurImageChannel.argtypes = [
            c_void_p, c_int, c_double, c_double
        ]
        lib.MagickGaussianBlurImageChannel.restype = c_bool
    else:
        lib.MagickGaussianBlurImageChannel = None
    lib.MagickGetImage.argtypes = [c_void_p]
    lib.MagickGetImage.restype = c_void_p
    lib.MagickGetImageAlphaChannel.argtypes = [c_void_p]
    lib.MagickGetImageAlphaChannel.restype = c_bool
    lib.MagickGetImageBackgroundColor.argtypes = [c_void_p, c_void_p]
    lib.MagickGetImageBackgroundColor.restype = c_bool
    lib.MagickGetImageBlob.argtypes = [c_void_p, POINTER(c_size_t)]
    lib.MagickGetImageBlob.restype = POINTER(c_ubyte)
    if is_im_6:
        lib.MagickGetImageBluePrimary.argtypes = [
            c_void_p, POINTER(c_double), POINTER(c_double)
        ]
    else:
        lib.MagickGetImageBluePrimary.argtypes = [
            c_void_p, POINTER(c_double), POINTER(c_double), POINTER(c_double)
        ]
    lib.MagickGetImageBluePrimary.restype = c_bool
    lib.MagickGetImageBorderColor.argtypes = [c_void_p, c_void_p]
    lib.MagickGetImageBorderColor.restype = c_bool
    if is_im_6:
        lib.MagickGetImageChannelDepth.argtypes = [c_void_p, c_int]
        lib.MagickGetImageChannelDepth.restype = c_size_t
        lib.MagickGetImageChannelFeatures.argtypes = [c_void_p, c_size_t]
        lib.MagickGetImageChannelFeatures.restype = c_void_p
        lib.MagickGetImageChannelKurtosis.argtypes = [
            c_void_p, c_int, POINTER(c_double), POINTER(c_double)
        ]
        lib.MagickGetImageChannelKurtosis.restype = c_bool
        lib.MagickGetImageChannelMean.argtypes = [
            c_void_p, c_int, POINTER(c_double), POINTER(c_double)
        ]
        lib.MagickGetImageChannelMean.restype = c_bool
        lib.MagickGetImageChannelRange.argtypes = [
            c_void_p, c_int, POINTER(c_double), POINTER(c_double)
        ]
        lib.MagickGetImageChannelRange.restype = c_bool
        lib.MagickGetImageChannelStatistics.argtypes = [c_void_p]
        lib.MagickGetImageChannelStatistics.restype = c_void_p
        lib.MagickGetImageClipMask.argtypes = [c_void_p]
        lib.MagickGetImageClipMask.restype = c_void_p
    else:
        lib.MagickGetImageChannelDepth = None
    lib.MagickGetImageColormapColor.argtypes = [c_void_p, c_size_t, c_void_p]
    lib.MagickGetImageColormapColor.restype = c_bool
    lib.MagickGetImageColors.argtypes = [c_void_p]
    lib.MagickGetImageColors.restype = c_size_t
    lib.MagickGetImageColorspace.argtypes = [c_void_p]
    lib.MagickGetImageColorspace.restype = c_int
    lib.MagickGetImageCompose.argtypes = [c_void_p]
    lib.MagickGetImageCompose.restype = c_int
    lib.MagickGetImageCompression.argtypes = [c_void_p]
    lib.MagickGetImageCompression.restype = c_int
    lib.MagickGetImageCompressionQuality.argtypes = [c_void_p]
    lib.MagickGetImageCompressionQuality.restype = c_ssize_t
    try:
        lib.MagickGetImageEndian.argtypes = [c_void_p]
        lib.MagickGetImageEndian.restype = c_int
    except AttributeError:
        lib.MagickGetImageEndian = None
    lib.MagickGetImageDelay.argtypes = [c_void_p]
    lib.MagickGetImageDelay.restype = c_size_t
    lib.MagickGetImageDepth.argtypes = [c_void_p]
    lib.MagickGetImageDepth.restype = c_size_t
    lib.MagickGetImageDispose.argtypes = [c_void_p]
    lib.MagickGetImageDispose.restype = c_int
    lib.MagickGetImageDistortion.argtypes = [
        c_void_p, c_void_p, c_int, POINTER(c_double)
    ]
    lib.MagickGetImageDistortion.restype = c_bool
    if is_im_7:
        lib.MagickGetImageFeatures.argtypes = [c_void_p, c_size_t]
        lib.MagickGetImageFeatures.restype = c_void_p
    lib.MagickGetImageFilename.argtypes = [c_void_p]
    lib.MagickGetImageFilename.restype = c_void_p
    lib.MagickGetImageFormat.argtypes = [c_void_p]
    lib.MagickGetImageFormat.restype = c_void_p
    lib.MagickGetImageFuzz.argtypes = [c_void_p]
    lib.MagickGetImageFuzz.restype = c_double
    lib.MagickGetImageGamma.argtypes = [c_void_p]
    lib.MagickGetImageGamma.restype = c_double
    lib.MagickGetImageGravity.argtypes = [c_void_p]
    lib.MagickGetImageGravity.restype = c_int
    if is_im_6:
        lib.MagickGetImageGreenPrimary.argtypes = [
            c_void_p, POINTER(c_double), POINTER(c_double)
        ]
    else:
        lib.MagickGetImageGreenPrimary.argtypes = [
            c_void_p, POINTER(c_double), POINTER(c_double), POINTER(c_double)
        ]
    lib.MagickGetImageGreenPrimary.restype = c_bool
    lib.MagickGetImageHeight.argtypes = [c_void_p]
    lib.MagickGetImageHeight.restype = c_size_t
    lib.MagickGetImageHistogram.argtypes = [c_void_p, POINTER(c_size_t)]
    lib.MagickGetImageHistogram.restype = POINTER(c_void_p)
    lib.MagickGetImageInterlaceScheme.argtypes = [c_void_p]
    lib.MagickGetImageInterlaceScheme.restype = c_int
    lib.MagickGetImageIterations.argtypes = [c_void_p]
    lib.MagickGetImageIterations.restype = c_size_t
    lib.MagickGetImageInterpolateMethod.argtypes = [c_void_p]
    lib.MagickGetImageInterpolateMethod.restype = c_int
    if is_im_7:
        lib.MagickGetImageKurtosis.argtypes = [
            c_void_p, POINTER(c_double), POINTER(c_double)
        ]
        lib.MagickGetImageKurtosis.restype = c_bool
    lib.MagickGetImageLength.argtypes = [c_void_p, POINTER(c_size_t)]
    lib.MagickGetImageLength.restype = c_bool
    if is_im_7:
        lib.MagickGetImageMask.argtypes = [c_void_p, c_int]
        lib.MagickGetImageMask.restype = c_void_p
    else:
        lib.MagickGetImageMask = None
    lib.MagickGetImageMatteColor.argtypes = [c_void_p, c_void_p]
    lib.MagickGetImageMatteColor.restype = c_bool
    if is_im_7:
        lib.MagickGetImageMean.argtypes = [
            c_void_p, POINTER(c_double), POINTER(c_double)
        ]
        lib.MagickGetImageMean.restype = c_bool
    lib.MagickGetImageOrientation.argtypes = [c_void_p]
    lib.MagickGetImageOrientation.restype = c_int
    lib.MagickGetImagePage.argtypes = [
        c_void_p, POINTER(c_size_t), POINTER(c_size_t), POINTER(c_ssize_t),
        POINTER(c_ssize_t)
    ]
    lib.MagickGetImagePage.restype = c_bool
    lib.MagickGetImagePixelColor.argtypes = [
        c_void_p, c_ssize_t, c_ssize_t, c_void_p
    ]
    lib.MagickGetImagePixelColor.restype = c_bool
    lib.MagickGetImageRange.argtypes = [
        c_void_p, POINTER(c_double), POINTER(c_double)
    ]
    lib.MagickGetImageRange.restype = c_bool
    if is_im_6:
        lib.MagickGetImageRedPrimary.argtypes = [
            c_void_p, POINTER(c_double), POINTER(c_double)
        ]
    else:
        lib.MagickGetImageRedPrimary.argtypes = [
            c_void_p, POINTER(c_double), POINTER(c_double), POINTER(c_double)
        ]
    lib.MagickGetImageRedPrimary.restype = c_bool
    lib.MagickGetImageRegion.argtypes = [
        c_void_p, c_size_t, c_size_t, c_ssize_t, c_ssize_t
    ]
    lib.MagickGetImageRegion.restype = c_void_p
    lib.MagickGetImageResolution.argtypes = [
        c_void_p, POINTER(c_double), POINTER(c_double)
    ]
    lib.MagickGetImageRenderingIntent.argtypes = [c_void_p]
    lib.MagickGetImageRenderingIntent.restype = c_int
    lib.MagickGetImageResolution.argtypes = [
        c_void_p, POINTER(c_double), POINTER(c_double)
    ]
    lib.MagickGetImageResolution.restype = c_bool
    lib.MagickGetImagesBlob.argtypes = [c_void_p, POINTER(c_size_t)]
    lib.MagickGetImagesBlob.restype = POINTER(c_ubyte)
    lib.MagickGetImageScene.argtypes = [c_void_p]
    lib.MagickGetImageScene.restype = c_size_t
    lib.MagickGetImageSignature.argtypes = [c_void_p]
    lib.MagickGetImageSignature.restype = c_void_p
    lib.MagickGetImageTicksPerSecond.argtypes = [c_void_p]
    lib.MagickGetImageTicksPerSecond.restype = c_size_t
    lib.MagickGetImageTotalInkDensity.argtypes = [c_void_p]
    lib.MagickGetImageTotalInkDensity.restype = c_double
    lib.MagickGetImageType.argtypes = [c_void_p]
    lib.MagickGetImageType.restype = c_int
    lib.MagickGetImageUnits.argtypes = [c_void_p]
    lib.MagickGetImageVirtualPixelMethod.argtypes = [c_void_p]
    if is_im_6:
        lib.MagickGetImageWhitePoint.argtypes = [
            c_void_p, POINTER(c_double), POINTER(c_double)
        ]
    else:
        lib.MagickGetImageWhitePoint.argtypes = [
            c_void_p, POINTER(c_double), POINTER(c_double), POINTER(c_double)
        ]
    lib.MagickGetImageWhitePoint.restype = c_bool
    lib.MagickGetImageWidth.argtypes = [c_void_p]
    lib.MagickGetImageWidth.restype = c_size_t
    lib.MagickGetNumberImages.argtypes = [c_void_p]
    lib.MagickGetNumberImages.restype = c_size_t
    lib.MagickHaldClutImage.argtypes = [c_void_p, c_void_p]
    lib.MagickHaldClutImage.restype = c_bool
    if is_im_6:
        lib.MagickHaldClutImageChannel.argtypes = [c_void_p, c_int, c_void_p]
        lib.MagickHaldClutImageChannel.restype = c_bool
    else:
        lib.MagickHaldClutImageChannel = None
    lib.MagickHasNextImage.argtypes = [c_void_p]
    lib.MagickHasNextImage.restype = c_bool
    lib.MagickHasPreviousImage.argtypes = [c_void_p]
    lib.MagickHasPreviousImage.restype = c_bool
    if IM_VERSION >= 0x708:
        try:
            lib.MagickHoughLineImage.argtypes = [c_void_p, c_size_t, c_size_t,
                                                 c_size_t]
            lib.MagickHoughLineImage.restype = c_bool
        except AttributeError:
            lib.MagickHoughLineImage = None
    else:
        lib.MagickHoughLineImage = None
    lib.MagickIdentifyImage.argtypes = [c_void_p]
    lib.MagickIdentifyImage.restype = c_void_p
    if is_im_6:
        lib.MagickImplodeImage.argtypes = [c_void_p, c_double]
    else:
        lib.MagickImplodeImage.argtypes = [c_void_p, c_double, c_int]
    lib.MagickImplodeImage.restype = c_bool
    lib.MagickImportImagePixels.argtypes = [
        c_void_p, c_ssize_t, c_ssize_t, c_size_t, c_size_t, c_char_p, c_int,
        c_void_p
    ]
    lib.MagickImportImagePixels.restype = c_bool
    lib.MagickInverseFourierTransformImage.argtypes = [
        c_void_p, c_void_p, c_double
    ]
    lib.MagickInverseFourierTransformImage.restype = c_bool
    if IM_VERSION > 0x709:
        try:
            lib.MagickKmeansImage.argtypes = [
                c_void_p, c_size_t, c_size_t, c_double
            ]
            lib.MagickKmeansImage.restype = c_bool
        except AttributeError:
            lib.MagickKmeansImage = None
    else:
        lib.MagickKmeansImage = None
    if IM_VERSION >= 0x708:
        try:
            lib.MagickKuwaharaImage.argtypes = [c_void_p, c_double, c_double]
            lib.MagickKuwaharaImage.restype = c_bool
        except AttributeError:
            lib.MagickKuwaharaImage = None
    else:
        lib.MagickKuwaharaImage = None
    lib.MagickLabelImage.argtypes = [c_void_p, c_char_p]
    lib.MagickLabelImage.restype = c_bool
    lib.MagickLevelImage.argtypes = [c_void_p, c_double, c_double, c_double]
    lib.MagickLevelImage.restype = c_bool
    if is_im_6:
        lib.MagickLevelImageChannel.argtypes = [
            c_void_p, c_int, c_double, c_double, c_double
        ]
        lib.MagickLevelImageChannel.restype = c_bool
    else:
        lib.MagickLevelImageChannel = None
    if IM_VERSION >= 0x708:
        try:
            lib.MagickLevelImageColors.argtypes = [
                c_void_p, c_void_p, c_void_p, c_bool
            ]
            lib.MagickLevelImageColors.restype = c_bool
        except AttributeError:
            lib.MagickLevelImageColors = None
        try:
            lib.MagickLevelizeImage.argtypes = [c_void_p, c_double, c_double,
                                                c_double]
            lib.MagickLevelizeImage.restype = c_bool
        except AttributeError:
            lib.MagickLevelizeImage = None
    else:
        lib.MagickLevelImageColors = None
        lib.MagickLevelizeImage = None
    lib.MagickLinearStretchImage.argtypes = [c_void_p, c_double, c_double]
    lib.MagickLinearStretchImage.restype = c_bool
    lib.MagickLiquidRescaleImage.argtypes = [
        c_void_p, c_size_t, c_size_t, c_double, c_double
    ]
    lib.MagickLiquidRescaleImage.restype = c_bool
    try:
        lib.MagickLocalContrastImage.argtypes = [c_void_p, c_double, c_double]
        lib.MagickLocalContrastImage.restype = c_bool
    except AttributeError:
        lib.MagickLocalContrastImage = None
    lib.MagickMagnifyImage.argtypes = [c_void_p]
    lib.MagickMagnifyImage.restype = c_bool
    if IM_VERSION >= 0x708:
        try:
            lib.MagickMeanShiftImage.argtypes = [c_void_p, c_size_t, c_size_t,
                                                 c_double]
            lib.MagickMeanShiftImage.restype = c_bool
        except AttributeError:
            lib.MagickMeanShiftImage = None
    else:
        lib.MagickMeanShiftImage = None
    if is_im_6:
        lib.MagickMedianFilterImage.argtypes = [c_void_p, c_double]
        lib.MagickMedianFilterImage.restype = c_bool
    lib.MagickMergeImageLayers.argtypes = [c_void_p, c_int]
    lib.MagickMergeImageLayers.restype = c_void_p
    lib.MagickMinifyImage.argtypes = [c_void_p]
    lib.MagickMinifyImage.restype = c_bool
    try:
        lib.MagickModeImage.argtypes = [c_void_p, c_double]
        lib.MagickModeImage.restype = c_bool
    except AttributeError:
        pass
    lib.MagickModulateImage.argtypes = [c_void_p, c_double, c_double, c_double]
    lib.MagickModulateImage.restype = c_bool
    lib.MagickMontageImage.argtypes = [
        c_void_p, c_void_p, c_char_p, c_char_p, c_int, c_char_p
    ]
    lib.MagickMontageImage.restype = c_void_p
    lib.MagickMorphImages.argtypes = [c_void_p, c_size_t]
    lib.MagickMorphImages.restype = c_void_p
    lib.MagickMorphologyImage.argtypes = [c_void_p, c_int, c_ssize_t, c_void_p]
    lib.MagickMorphologyImage.restype = c_bool
    if is_im_6:
        lib.MagickMorphologyImageChannel.argtypes = [
            c_void_p, c_int, c_int, c_ssize_t, c_void_p
        ]
        lib.MagickMorphologyImageChannel.restype = c_bool
    else:
        lib.MagickMorphologyImageChannel = None
    lib.MagickMotionBlurImage.argtypes = [
        c_void_p, c_double, c_double, c_double
    ]
    lib.MagickMotionBlurImage.restype = c_bool
    if is_im_6:
        lib.MagickMotionBlurImageChannel.argtypes = [
            c_void_p, c_int, c_double, c_double, c_double
        ]
        lib.MagickMotionBlurImageChannel.restype = c_bool
    else:
        lib.MagickMotionBlurImageChannel = None
    lib.MagickNegateImage.argtypes = [c_void_p, c_int]
    lib.MagickNegateImage.restype = c_bool
    if is_im_6:
        lib.MagickNegateImageChannel.argtypes = [c_void_p, c_int, c_int]
        lib.MagickNegateImageChannel.restype = c_bool
    else:
        lib.MagickNegateImageChannel = None
    lib.MagickNewImage.argtypes = [c_void_p, c_int, c_int, c_void_p]
    lib.MagickNewImage.restype = c_bool
    lib.MagickNextImage.argtypes = [c_void_p]
    lib.MagickNextImage.restype = c_bool
    lib.MagickNormalizeImage.argtypes = [c_void_p]
    lib.MagickNormalizeImage.restype = c_bool
    if is_im_6:
        lib.MagickNormalizeImageChannel.argtypes = [c_void_p, c_int]
        lib.MagickNormalizeImageChannel.restype = c_bool
    else:
        lib.MagickNormalizeImageChannel = None
    if is_im_6:
        lib.MagickOilPaintImage.argtypes = [c_void_p, c_double]
        lib.MagickOilPaintImage.restype = c_bool
    else:
        lib.MagickOilPaintImage.argtypes = [c_void_p, c_double, c_double]
        lib.MagickOilPaintImage.restype = c_bool
    lib.MagickOpaquePaintImage.argtypes = [
        c_void_p, c_void_p, c_void_p, c_double, c_bool
    ]
    lib.MagickOpaquePaintImage.restype = c_bool
    if is_im_6:
        lib.MagickOpaquePaintImageChannel.argtypes = [
            c_void_p, c_int, c_void_p, c_void_p, c_double, c_bool
        ]
        lib.MagickOpaquePaintImageChannel.restype = c_bool
    lib.MagickOptimizeImageLayers.argtypes = [c_void_p]
    lib.MagickOptimizeImageLayers.restype = c_void_p
    try:
        lib.MagickOptimizeImageTransparency.argtypes = [c_void_p]
        lib.MagickOptimizeImageTransparency.restype = c_bool
    except AttributeError:
        lib.MagickOptimizeImageTransparency = None
    if is_im_7:
        lib.MagickOrderedDitherImage.argtypes = [c_void_p, c_char_p]
        lib.MagickOrderedDitherImage.restype = c_bool
    if is_im_6:
        lib.MagickOrderedPosterizeImage.argtypes = [c_void_p, c_char_p]
        lib.MagickOrderedPosterizeImage.restype = c_bool
        lib.MagickOrderedPosterizeImageChannel.argtypes = [
            c_void_p, c_int, c_char_p
        ]
        lib.MagickOrderedPosterizeImageChannel.restype = c_bool
    lib.MagickPingImage.argtypes = [c_void_p, c_char_p]
    lib.MagickPingImage.restype = c_bool
    lib.MagickPingImageBlob.argtypes = [c_void_p, c_void_p, c_size_t]
    lib.MagickPingImageBlob.restype = c_bool
    lib.MagickPingImageFile.argtypes = [c_void_p, c_void_p]
    lib.MagickPingImageFile.restype = c_bool
    if is_im_6:
        lib.MagickPolaroidImage.argtypes = [c_void_p, c_void_p, c_double]
        lib.MagickPolaroidImage.restype = c_bool
    else:
        lib.MagickPolaroidImage.argtypes = [
            c_void_p, c_void_p, c_char_p, c_double, c_int
        ]
        lib.MagickPolaroidImage.restype = c_bool
    if IM_VERSION >= 0x708:
        try:
            lib.MagickPolynomialImage.argtypes = [c_void_p, c_size_t,
                                                  POINTER(c_double)]
            lib.MagickPolynomialImage.restype = c_bool
        except AttributeError:
            lib.MagickPolynomialImage = None
    else:
        lib.MagickPolynomialImage = None
    lib.MagickPosterizeImage.argtypes = [c_void_p, c_size_t, c_bool]
    lib.MagickPosterizeImage.restype = c_bool
    lib.MagickPreviewImages.argtypes = [c_void_p, c_int]
    lib.MagickPreviewImages.restype = c_void_p
    lib.MagickPreviousImage.argtypes = [c_void_p]
    lib.MagickPreviousImage.restype = c_bool
    if IM_VERSION < 0x700:
        lib.MagickQuantizeImage.argtypes = [
            c_void_p, c_size_t, c_int, c_size_t, c_bool, c_bool
        ]
        lib.MagickQuantizeImage.restypes = c_bool
        lib.MagickQuantizeImages.argtypes = [
            c_void_p, c_size_t, c_int, c_size_t, c_bool, c_bool
        ]
        lib.MagickQuantizeImages.restype = c_bool
    else:
        lib.MagickQuantizeImage.argtypes = [
            c_void_p, c_size_t, c_int, c_size_t, c_int, c_bool
        ]
        lib.MagickQuantizeImage.restypes = c_bool
        lib.MagickQuantizeImages.argtypes = [
            c_void_p, c_size_t, c_int, c_size_t, c_int, c_bool
        ]
        lib.MagickQuantizeImages.restype = c_bool
    lib.MagickRaiseImage.argtypes = [
        c_void_p, c_size_t, c_size_t, c_ssize_t, c_ssize_t, c_bool
    ]
    lib.MagickRandomThresholdImage.argtypes = [c_void_p, c_double, c_double]
    lib.MagickRandomThresholdImage.restype = c_bool
    if is_im_6:
        lib.MagickRandomThresholdImageChannel.argtypes = [
            c_void_p, c_int, c_double, c_double
        ]
        lib.MagickRandomThresholdImageChannel.restype = c_bool
    if IM_VERSION >= 0x708:
        try:
            lib.MagickRangeThresholdImage.argtypes = [c_void_p, c_double,
                                                      c_double, c_double,
                                                      c_double]
            lib.MagickRangeThresholdImage.restype = c_bool
        except AttributeError:
            lib.MagickRangeThresholdImage = None
    else:
        lib.MagickRangeThresholdImage = None
    lib.MagickReadImage.argtypes = [c_void_p, c_char_p]
    lib.MagickReadImage.restype = c_bool
    lib.MagickReadImageBlob.argtypes = [c_void_p, c_void_p, c_size_t]
    lib.MagickReadImageBlob.restype = c_bool
    lib.MagickReadImageFile.argtypes = [c_void_p, c_void_p]
    lib.MagickReadImageFile.restype = c_bool
    try:
        lib.MagickReduceNoiseImage.argtypes = [c_void_p, c_double]
        lib.MagickReduceNoiseImage.restype = c_bool
    except AttributeError:
        pass
    lib.MagickRemapImage.argtypes = [c_void_p, c_void_p, c_int]
    lib.MagickRemapImage.restype = c_bool
    lib.MagickRemoveImage.argtypes = [c_void_p]
    lib.MagickRemoveImage.restype = c_bool
    lib.MagickResampleImage.argtypes = [
        c_void_p, c_double, c_double, c_int, c_double
    ]
    lib.MagickResampleImage.restype = c_bool
    lib.MagickResetImagePage.argtypes = [c_void_p, c_char_p]
    lib.MagickResetImagePage.restype = c_bool
    lib.MagickResizeImage.argtypes = [
        c_void_p, c_size_t, c_size_t, c_int, c_double
    ]
    lib.MagickResizeImage.restype = c_bool
    lib.MagickRollImage.argtypes = [c_void_p, c_ssize_t, c_ssize_t]
    lib.MagickRollImage.restype = c_bool
    lib.MagickRotateImage.argtypes = [c_void_p, c_void_p, c_double]
    lib.MagickRotateImage.restype = c_bool
    try:
        lib.MagickRotationalBlurImage.argtypes = [c_void_p, c_double]
        lib.MagickRotationalBlurImage.restype = c_bool
        if is_im_6:
            lib.MagickRotationalBlurImageChannel.argtypes = [
                c_void_p, c_int, c_double
            ]
            lib.MagickRotationalBlurImageChannel.restype = c_bool
    except AttributeError:
        lib.MagickRotationalBlurImage = None
        pass
    lib.MagickSampleImage.argtypes = [c_void_p, c_size_t, c_size_t]
    lib.MagickSampleImage.restype = c_bool
    lib.MagickScaleImage.argtypes = [c_void_p, c_size_t, c_size_t]
    lib.MagickScaleImage.restype = c_bool
    lib.MagickSegmentImage.argtypes = [
        c_void_p, c_int, c_bool, c_double, c_double
    ]
    lib.MagickSegmentImage.restype = c_bool
    lib.MagickSelectiveBlurImage.argtypes = [
        c_void_p, c_double, c_double, c_double
    ]
    lib.MagickSelectiveBlurImage.restype = c_bool
    if is_im_6:
        lib.MagickSelectiveBlurImageChannel.argtypes = [
            c_void_p, c_int, c_double, c_double, c_double
        ]
        lib.MagickSelectiveBlurImageChannel.restype = c_bool
    lib.MagickSepiaToneImage.argtypes = [c_void_p, c_double]
    lib.MagickSepiaToneImage.restype = c_bool
    if is_im_6:
        lib.MagickSeparateImage = None
        lib.MagickSeparateImageChannel.argtypes = [c_void_p, c_int]
        lib.MagickSeparateImageChannel.restype = c_bool
    else:
        lib.MagickSeparateImage.argtypes = [c_void_p, c_int]
        lib.MagickSeparateImage.restype = c_bool
        lib.MagickSeparateImageChannel = None
    lib.MagickSetImage.argtypes = [c_void_p, c_void_p]
    lib.MagickSetImage.restype = c_bool
    lib.MagickSetImageAlphaChannel.argtypes = [c_void_p, c_int]
    lib.MagickSetImageAlphaChannel.restype = c_bool
    lib.MagickSetImageBackgroundColor.argtypes = [c_void_p, c_void_p]
    lib.MagickSetImageBackgroundColor.restype = c_bool
    if is_im_6:
        lib.MagickSetImageBias.argtypes = [c_void_p, c_double]
        lib.MagickSetImageBias.restype = c_bool
    else:
        lib.MagickSetImageBias = None
    if is_im_6:
        lib.MagickSetImageBluePrimary.argtypes = [
            c_void_p, c_double, c_double
        ]
    else:
        lib.MagickSetImageBluePrimary.argtypes = [
            c_void_p, c_double, c_double, c_double
        ]
    lib.MagickSetImageBluePrimary.restype = c_bool
    lib.MagickSetImageBorderColor.argtypes = [c_void_p, c_void_p]
    lib.MagickSetImageBorderColor.restype = c_bool
    if is_im_6:
        lib.MagickSetImageChannelDepth.argtypes = [c_void_p, c_int, c_size_t]
        lib.MagickSetImageClipMask.argtypes = [c_void_p, c_void_p]
        lib.MagickSetImageClipMask.restype = c_bool
    else:
        lib.MagickSetImageChannelDepth = None
        lib.MagickSetImageClipMask = None
    if is_im_7:
        lib.MagickSetImageChannelMask.argtypes = [c_void_p, c_int]
        lib.MagickSetImageChannelMask.restype = c_int
    lib.MagickSetImageColor.argtypes = [c_void_p, c_void_p]
    lib.MagickSetImageColor.restype = c_bool
    lib.MagickSetImageColormapColor.argtypes = [c_void_p, c_size_t, c_void_p]
    lib.MagickSetImageColormapColor.restype = c_bool
    lib.MagickSetImageColorspace.argtypes = [c_void_p, c_int]
    lib.MagickSetImageColorspace.restype = c_bool
    lib.MagickSetImageCompose.argtypes = [c_void_p, c_int]
    lib.MagickSetImageCompose.restype = c_bool
    lib.MagickSetImageCompression.argtypes = [c_void_p, c_int]
    lib.MagickSetImageCompression.restype = c_bool
    lib.MagickSetImageCompressionQuality.argtypes = [c_void_p, c_ssize_t]
    lib.MagickSetImageCompressionQuality.restype = c_bool
    lib.MagickSetImageDelay.argtypes = [c_void_p, c_ssize_t]
    lib.MagickSetImageDelay.restype = c_bool
    lib.MagickSetImageDepth.argtypes = [c_void_p]
    lib.MagickSetImageDepth.restype = c_bool
    lib.MagickSetImageDispose.argtypes = [c_void_p, c_int]
    lib.MagickSetImageDispose.restype = c_bool
    try:
        lib.MagickSetImageEndian.argtypes = [c_void_p, c_int]
        lib.MagickSetImageEndian.restype = c_bool
    except AttributeError:
        lib.MagickSetImageEndian = None
    lib.MagickSetImageExtent.argtypes = [c_void_p, c_size_t, c_size_t]
    lib.MagickSetImageExtent.restype = c_bool
    lib.MagickSetImageFilename.argtypes = [c_void_p, c_char_p]
    lib.MagickSetImageFilename.restype = c_bool
    lib.MagickSetImageFormat.argtypes = [c_void_p, c_char_p]
    lib.MagickSetImageFormat.restype = c_bool
    lib.MagickSetImageFuzz.argtypes = [c_void_p, c_double]
    lib.MagickSetImageFuzz.restype = c_bool
    lib.MagickSetImageGamma.argtypes = [c_void_p, c_double]
    lib.MagickSetImageGamma.restype = c_bool
    lib.MagickSetImageGravity.argtypes = [c_void_p, c_int]
    lib.MagickSetImageGravity.restype = c_bool
    if is_im_6:
        lib.MagickSetImageGreenPrimary.argtypes = [
            c_void_p, c_double, c_double
        ]
    else:
        lib.MagickSetImageGreenPrimary.argtypes = [
            c_void_p, c_double, c_double, c_double
        ]
    lib.MagickSetImageGreenPrimary.restype = c_bool
    lib.MagickSetImageInterlaceScheme.argtypes = [c_void_p, c_int]
    lib.MagickSetImageInterlaceScheme.restype = c_bool
    lib.MagickSetImageInterpolateMethod.argtypes = [c_void_p, c_int]
    lib.MagickSetImageInterpolateMethod.restype = c_bool
    lib.MagickSetImageIterations.argtypes = [c_void_p, c_size_t]
    lib.MagickSetImageIterations.restype = c_bool
    if is_im_7:
        lib.MagickSetImageMask.argtypes = [c_void_p, c_int, c_void_p]
        lib.MagickSetImageMask.restype = c_bool
    else:
        lib.MagickSetImageMask = None
    lib.MagickSetImageMatte.argtypes = [c_void_p, c_bool]
    lib.MagickSetImageMatteColor.argtypes = [c_void_p, c_void_p]
    if is_im_6:
        lib.MagickSetImageOpacity.argtypes = [c_void_p, c_double]
        lib.MagickSetImageOpacity.restype = c_bool
    else:
        lib.MagickSetImageOpacity = None
    lib.MagickSetImageOrientation.argtypes = [c_void_p, c_int]
    lib.MagickSetImagePage.argtypes = [
        c_void_p, c_size_t, c_size_t, c_ssize_t, c_ssize_t
    ]
    lib.MagickSetImagePage.restype = c_int
    lib.MagickSetImageProgressMonitor.argtypes = [
        c_void_p, MagickProgressMonitor, c_void_p
    ]
    if is_im_6:
        lib.MagickSetImageRedPrimary.argtypes = [
            c_void_p, c_double, c_double
        ]
    else:
        lib.MagickSetImageRedPrimary.argtypes = [
            c_void_p, c_double, c_double, c_double
        ]
    lib.MagickSetImageRedPrimary.restype = c_bool
    lib.MagickSetImageRenderingIntent.argtypes = [c_void_p, c_int]
    lib.MagickSetImageRenderingIntent.restype = c_bool
    lib.MagickSetImageResolution.argtypes = [c_void_p, c_double, c_double]
    lib.MagickSetImageScene.argtypes = [c_void_p, c_size_t]
    lib.MagickSetImageScene.restype = c_bool
    lib.MagickSetImageTicksPerSecond.argtypes = [c_void_p, c_ssize_t]
    lib.MagickSetImageTicksPerSecond.restype = c_bool
    lib.MagickSetImageType.argtypes = [c_void_p, c_int]
    lib.MagickSetImageUnits.argtypes = [c_void_p, c_int]
    lib.MagickSetImageVirtualPixelMethod.argtypes = [c_void_p, c_int]
    if is_im_6:
        lib.MagickSetImageWhitePoint.argtypes = [
            c_void_p, c_double, c_double
        ]
    else:
        lib.MagickSetImageWhitePoint.argtypes = [
            c_void_p, c_double, c_double, c_double
        ]
    lib.MagickSetImageWhitePoint.restype = c_bool
    lib.MagickShadeImage.argtypes = [c_void_p, c_bool, c_double, c_double]
    lib.MagickShadeImage.restype = c_bool
    lib.MagickShadowImage.argtypes = [
        c_void_p, c_double, c_double, c_ssize_t, c_ssize_t
    ]
    lib.MagickShadowImage.restype = c_bool
    lib.MagickSharpenImage.argtypes = [c_void_p, c_double, c_double]
    lib.MagickSharpenImage.restype = c_bool
    if is_im_6:
        lib.MagickSharpenImageChannel.argtypes = [
            c_void_p, c_int, c_double, c_double
        ]
        lib.MagickSharpenImageChannel.restype = c_bool
    else:
        lib.MagickSharpenImageChannel = None
    lib.MagickShaveImage.argtypes = [c_void_p, c_size_t, c_size_t]
    lib.MagickShaveImage.restype = c_void_p
    lib.MagickShearImage.argtypes = [c_void_p, c_void_p, c_double, c_double]
    lib.MagickShearImage.restype = c_bool
    lib.MagickSigmoidalContrastImage.argtypes = [
        c_void_p, c_bool, c_double, c_double
    ]
    lib.MagickSigmoidalContrastImage.restype = c_bool
    if is_im_6:
        lib.MagickSigmoidalContrastImageChannel.argtypes = [
            c_void_p, c_int, c_bool, c_double, c_double
        ]
        lib.MagickSigmoidalContrastImageChannel.restype = c_bool
    else:
        lib.MagickSigmoidalContrastImageChannel = None
    if is_im_6:
        lib.MagickSimilarityImage.argtypes = [
            c_void_p, c_void_p, c_void_p, POINTER(c_double)
        ]
        lib.MagickSimilarityImage.restype = c_void_p
    else:
        lib.MagickSimilarityImage.argtypes = [
            c_void_p, c_void_p, c_int, c_double, c_void_p, POINTER(c_double)
        ]
        lib.MagickSimilarityImage.restype = c_void_p
    lib.MagickSketchImage.argtypes = [c_void_p, c_double, c_double, c_double]
    lib.MagickSketchImage.restype = c_bool
    lib.MagickSmushImages.argtypes = [c_void_p, c_bool, c_ssize_t]
    lib.MagickSmushImages.restype = c_void_p
    lib.MagickSolarizeImage.argtypes = [c_void_p, c_double]
    lib.MagickSolarizeImage.restype = c_bool
    try:
        lib.MagickSolarizeImageChannel.argtypes = [c_void_p, c_int, c_double]
        lib.MagickSolarizeImageChannel.restype = c_bool
    except AttributeError:
        lib.MagickSolarizeImageChannel = None
    if is_im_6:
        lib.MagickSparseColorImage.argtypes = [
            c_void_p, c_int, c_int, c_size_t, POINTER(c_double)
        ]
    else:
        lib.MagickSparseColorImage.argtypes = [
            c_void_p, c_int, c_size_t, POINTER(c_double)
        ]
    lib.MagickSparseColorImage.restype = c_bool
    lib.MagickSpliceImage.argtypes = [
        c_void_p, c_size_t, c_size_t, c_ssize_t, c_ssize_t
    ]
    lib.MagickSpliceImage.restype = c_bool
    if is_im_6:
        lib.MagickSpreadImage.argtypes = [c_void_p, c_double]
        lib.MagickSpreadImage.restype = c_bool
    else:
        lib.MagickSpreadImage.argtypes = [c_void_p, c_int, c_double]
        lib.MagickSpreadImage.restype = c_bool
    lib.MagickStatisticImage.argtypes = [c_void_p, c_int, c_size_t, c_size_t]
    lib.MagickStatisticImage.restype = c_bool
    if is_im_6:
        try:
            # TODO - Arguments for MagickStatisticImageChannel changed
            # around commit 2d8a006b @ Feb 9 13:02:53 2013. Use IM_VERSION
            # to determine correct method signature/arguments.
            lib.MagickStatisticImageChannel.argtypes = [
                c_void_p, c_int, c_int, c_size_t, c_size_t
            ]
            lib.MagickStatisticImageChannel.restype = c_bool
        except AttributeError:
            lib.MagickStatisticImageChannel = None
    else:
        lib.MagickStatisticImageChannel = None
    lib.MagickSteganoImage.argtypes = [c_void_p, c_void_p, c_ssize_t]
    lib.MagickSteganoImage.restype = c_void_p
    lib.MagickStereoImage.argtypes = [c_void_p, c_void_p]
    lib.MagickStereoImage.restype = c_void_p
    lib.MagickStripImage.argtypes = [c_void_p]
    lib.MagickStripImage.restype = c_bool
    if is_im_6:
        lib.MagickSwirlImage.argtypes = [c_void_p, c_double]
        lib.MagickSwirlImage.restype = c_bool
    else:
        lib.MagickSwirlImage.argtypes = [c_void_p, c_double, c_int]
        lib.MagickSwirlImage.restype = c_bool
    lib.MagickTextureImage.argtypes = [c_void_p, c_void_p]
    lib.MagickTextureImage.restype = c_void_p
    lib.MagickTintImage.argtypes = [c_void_p, c_void_p, c_void_p]
    lib.MagickTintImage.restype = c_bool
    lib.MagickThresholdImage.argtypes = [c_void_p, c_double]
    lib.MagickThresholdImage.restype = c_bool
    lib.MagickThresholdImageChannel.argtypes = [c_void_p, c_int, c_double]
    lib.MagickThresholdImageChannel.restype = c_bool
    lib.MagickThumbnailImage.argtypes = [c_void_p, c_size_t, c_size_t]
    lib.MagickThumbnailImage.restype = c_bool
    if is_im_6:
        lib.MagickTransformImage.argtypes = [c_void_p, c_char_p, c_char_p]
        lib.MagickTransformImage.restype = c_void_p
    else:
        lib.MagickTransformImage = None
    lib.MagickTransformImageColorspace.argtypes = [c_void_p, c_int]
    lib.MagickTransformImageColorspace.restype = c_bool
    lib.MagickTransparentPaintImage.argtypes = [
        c_void_p, c_void_p, c_double, c_double, c_int
    ]
    lib.MagickTransparentPaintImage.restype = c_bool
    lib.MagickTransposeImage.argtypes = [c_void_p]
    lib.MagickTransposeImage.restype = c_bool
    lib.MagickTransverseImage.argtypes = [c_void_p]
    lib.MagickTransverseImage.restype = c_bool
    lib.MagickTrimImage.argtypes = [c_void_p, c_double]
    lib.MagickTrimImage.restype = c_bool
    lib.MagickUniqueImageColors.argtypes = [c_void_p]
    lib.MagickUniqueImageColors.restype = c_bool
    lib.MagickUnsharpMaskImage.argtypes = [
        c_void_p, c_double, c_double, c_double, c_double
    ]
    lib.MagickUnsharpMaskImage.restype = c_bool
    if is_im_6:
        lib.MagickUnsharpMaskImageChannel.argtypes = [
            c_void_p, c_int, c_double, c_double, c_double, c_double
        ]
        lib.MagickUnsharpMaskImageChannel.restype = c_bool
    else:
        lib.MagickUnsharpMaskImageChannel = None
    lib.MagickVignetteImage.argtypes = [
        c_void_p, c_double, c_double, c_ssize_t, c_ssize_t
    ]
    lib.MagickVignetteImage.restype = c_bool
    if is_im_6:
        lib.MagickWaveImage.argtypes = [c_void_p, c_double, c_double]
        lib.MagickWaveImage.restype = c_bool
    else:
        lib.MagickWaveImage.argtypes = [c_void_p, c_double, c_double, c_int]
        lib.MagickWaveImage.restype = c_bool
    if IM_VERSION >= 0x708:
        try:
            lib.MagickWaveletDenoiseImage.argtypes = [c_void_p, c_double,
                                                      c_double]
            lib.MagickWaveletDenoiseImage.restype = c_bool
        except AttributeError:
            lib.MagickWaveletDenoiseImage = None
    else:
        lib.MagickWaveletDenoiseImage = None
    if IM_VERSION > 0x709:
        try:
            lib.MagickWhiteBalanceImage.argtypes = [c_void_p]
            lib.MagickWhiteBalanceImage.restype = c_bool
        except AttributeError:
            lib.MagickWhiteBalanceImage = None
    else:
        lib.MagickWhiteBalanceImage = None
    lib.MagickWhiteThresholdImage.argtypes = [c_void_p, c_void_p]
    lib.MagickWhiteThresholdImage.restype = c_bool
    lib.MagickWriteImage.argtypes = [c_void_p, c_char_p]
    lib.MagickWriteImage.restype = c_bool
    lib.MagickWriteImageFile.argtypes = [c_void_p, c_void_p]
    lib.MagickWriteImageFile.restype = c_bool
    lib.MagickWriteImages.argtypes = [c_void_p, c_char_p, c_int]
    lib.MagickWriteImages.restype = c_bool
    lib.MagickWriteImagesFile.argtypes = [c_void_p, c_void_p]
    lib.MagickWriteImagesFile.restype = c_bool
