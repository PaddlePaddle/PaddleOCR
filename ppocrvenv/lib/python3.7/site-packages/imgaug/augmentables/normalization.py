"""Functions dealing with normalization of user input data to imgaug classes."""
from __future__ import print_function, division, absolute_import
import functools

import numpy as np

from .. import imgaug as ia
from .. import dtypes as iadt
from .base import IAugmentable


def _preprocess_shapes(shapes):
    if shapes is None:
        return None
    if ia.is_np_array(shapes):
        assert shapes.ndim in [3, 4], (
            "Expected array 'shapes' to be 3- or 4-dimensional, got %d "
            "dimensions and shape %s instead." % (shapes.ndim, shapes.shape))
        return [image.shape for image in shapes]

    assert isinstance(shapes, list), (
        "Expected 'shapes' to be None or ndarray or list, got type %s "
        "instead." % (type(shapes),))
    result = []
    for shape_i in shapes:
        if isinstance(shape_i, tuple):
            result.append(shape_i)
        else:
            assert ia.is_np_array(shape_i), (
                "Expected each entry in list 'shapes' to be either a "
                "tuple or an ndarray, got type %s." % (type(shape_i),))
            result.append(shape_i.shape)
    return result


def _assert_exactly_n_shapes(shapes, n, from_ntype, to_ntype):
    if shapes is None:
        raise ValueError(
            "Tried to convert data of form '%s' to '%s'. This required %d "
            "corresponding image shapes, but argument 'shapes' was set to "
            "None. This can happen e.g. if no images were provided in a "
            "Batch, as these would usually be used to automatically derive "
            "image shapes." % (from_ntype, to_ntype, n))

    if len(shapes) != n:
        raise ValueError(
            "Tried to convert data of form '%s' to '%s'. This required "
            "exactly %d corresponding image shapes, but instead %d were "
            "provided. This can happen e.g. if more images were provided "
            "than corresponding augmentables, e.g. 10 images but only 5 "
            "segmentation maps. It can also happen if there was a "
            "misunderstanding about how an augmentable input would be "
            "parsed. E.g. if a list of N (x,y)-tuples was provided as "
            "keypoints and the expectation was that this would be parsed "
            "as one keypoint per image for N images, but instead it was "
            "parsed as N keypoints on 1 image (i.e. 'shapes' would have to "
            "contain 1 shape, but N would be provided). To avoid this, it "
            "is recommended to provide imgaug standard classes, e.g. "
            "KeypointsOnImage for keypoints instead of lists of "
            "tuples." % (from_ntype, to_ntype, n, len(shapes)))


def _assert_single_array_ndim(arr, ndim, shape_str, to_ntype):
    if arr.ndim != ndim:
        raise ValueError(
            "Tried to convert an array to list of %s. Expected "
            "that array to be of shape %s, i.e. %d-dimensional, but "
            "got %d dimensions instead." % (
                to_ntype, shape_str, ndim, arr.ndim,))


def _assert_many_arrays_ndim(arrs, ndim, shape_str, to_ntype):
    # For polygons, this can be a list of lists of arrays, hence we must
    # flatten the lists here.
    # itertools.chain.from_iterable() seems to flatten the arrays too, so it
    # cannot be used here.
    iterable_type_str = "iterable"
    if len(arrs) == 0:
        arrs_flat = []
    elif ia.is_np_array(arrs[0]):
        arrs_flat = arrs
    else:
        iterable_type_str = "iterable of iterable"
        arrs_flat = [arr for arrs_sublist in arrs for arr in arrs_sublist]

    if any([arr.ndim != ndim for arr in arrs_flat]):
        raise ValueError(
            "Tried to convert an %s of arrays to a list of "
            "%s. Expected each array to be of shape %s, "
            "i.e. to be %d-dimensional, but got dimensions %s "
            "instead (array shapes: %s)." % (
                iterable_type_str, to_ntype, shape_str, ndim,
                ", ".join([str(arr.ndim) for arr in arrs_flat]),
                ", ".join([str(arr.shape) for arr in arrs_flat])))


def _assert_single_array_last_dim_exactly(arr, size, to_ntype):
    if arr.shape[-1] != size:
        raise ValueError(
            "Tried to convert an array to a list of %s. Expected the array's "
            "last dimension to have size %d, but got %d instead (array "
            "shape: %s)." % (
                to_ntype, size, arr.shape[-1], str(arr.shape)))


def _assert_many_arrays_last_dim_exactly(arrs, size, to_ntype):
    # For polygons, this can be a list of lists of arrays, hence we must
    # flatten the lists here.
    # itertools.chain.from_iterable() seems to flatten the arrays too, so it
    # cannot be used here.
    iterable_type_str = "iterable"
    if len(arrs) == 0:
        arrs_flat = []
    elif ia.is_np_array(arrs[0]):
        arrs_flat = arrs
    else:
        iterable_type_str = "iterable of iterable"
        arrs_flat = [arr for arrs_sublist in arrs for arr in arrs_sublist]

    if any([arr.shape[-1] != size for arr in arrs_flat]):
        raise ValueError(
            "Tried to convert an %s of array to a list of %s. Expected the "
            "arrays' last dimensions to have size %d, but got %s instead "
            "(array shapes: %s)." % (
                iterable_type_str, to_ntype, size,
                ", ".join([str(arr.shape[-1]) for arr in arrs_flat]),
                ", ".join([str(arr.shape) for arr in arrs_flat])))


def normalize_images(images):
    if images is None:
        return None
    if ia.is_np_array(images):
        if images.ndim == 2:
            return images[np.newaxis, ..., np.newaxis]
        if images.ndim == 3:
            return images[..., np.newaxis]
        return images
    if ia.is_iterable(images):
        result = []
        for image in images:
            assert image.ndim in [2, 3], (
                "Got a list of arrays as argument 'images'. Expected each "
                "array in that list to have 2 or 3 dimensions, i.e. shape "
                "(H,W) or (H,W,C). Got %d dimensions "
                "instead." % (image.ndim,))

            if image.ndim == 2:
                result.append(image[..., np.newaxis])
            else:
                result.append(image)
        return result
    raise ValueError(
        "Expected argument 'images' to be any of the following: "
        "None or array or iterable of array. Got type: %s." % (
            type(images),))


def normalize_heatmaps(inputs, shapes=None):
    # TODO get rid of this deferred import
    from imgaug.augmentables.heatmaps import HeatmapsOnImage

    shapes = _preprocess_shapes(shapes)
    ntype = estimate_heatmaps_norm_type(inputs)
    _assert_exactly_n_shapes_partial = functools.partial(
        _assert_exactly_n_shapes,
        from_ntype=ntype, to_ntype="List[HeatmapsOnImage]", shapes=shapes)

    if ntype == "None":
        return None
    if ntype == "array[float]":
        _assert_single_array_ndim(inputs, 4, "(N,H,W,C)", "HeatmapsOnImage")
        _assert_exactly_n_shapes_partial(n=len(inputs))
        return [HeatmapsOnImage(attr_i, shape=shape_i)
                for attr_i, shape_i in zip(inputs, shapes)]
    if ntype == "HeatmapsOnImage":
        return [inputs]
    if ntype == "iterable[empty]":
        return None
    if ntype == "iterable-array[float]":
        _assert_many_arrays_ndim(inputs, 3, "(H,W,C)", "HeatmapsOnImage")
        _assert_exactly_n_shapes_partial(n=len(inputs))
        return [HeatmapsOnImage(attr_i, shape=shape_i)
                for attr_i, shape_i in zip(inputs, shapes)]
    assert ntype == "iterable-HeatmapsOnImage", (
        "Got unknown normalization type '%s'." % (ntype,))
    return inputs  # len allowed to differ from len of images


def normalize_segmentation_maps(inputs, shapes=None):
    # TODO get rid of this deferred import
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage

    shapes = _preprocess_shapes(shapes)
    ntype = estimate_segmaps_norm_type(inputs)
    _assert_exactly_n_shapes_partial = functools.partial(
        _assert_exactly_n_shapes,
        from_ntype=ntype, to_ntype="List[SegmentationMapsOnImage]",
        shapes=shapes)

    if ntype == "None":
        return None
    if ntype in ["array[int]", "array[uint]", "array[bool]"]:
        _assert_single_array_ndim(inputs, 4, "(N,H,W,#SegmapsPerImage)",
                                  "SegmentationMapsOnImage")
        _assert_exactly_n_shapes_partial(n=len(inputs))
        if ntype == "array[bool]":
            return [SegmentationMapsOnImage(attr_i, shape=shape)
                    for attr_i, shape in zip(inputs, shapes)]
        return [SegmentationMapsOnImage(attr_i, shape=shape)
                for attr_i, shape in zip(inputs, shapes)]
    if ntype == "SegmentationMapsOnImage":
        return [inputs]
    if ntype == "iterable[empty]":
        return None
    if ntype in ["iterable-array[int]",
                 "iterable-array[uint]",
                 "iterable-array[bool]"]:
        _assert_many_arrays_ndim(inputs, 3, "(H,W,#SegmapsPerImage)",
                                 "SegmentationMapsOnImage")
        _assert_exactly_n_shapes_partial(n=len(inputs))
        if ntype == "iterable-array[bool]":
            return [SegmentationMapsOnImage(attr_i, shape=shape)
                    for attr_i, shape in zip(inputs, shapes)]
        return [SegmentationMapsOnImage(attr_i, shape=shape)
                for attr_i, shape in zip(inputs, shapes)]

    assert ntype == "iterable-SegmentationMapsOnImage", (
        "Got unknown normalization type '%s'." % (ntype,))
    return inputs  # len allowed to differ from len of images


def normalize_keypoints(inputs, shapes=None):
    # TODO get rid of this deferred import
    from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

    shapes = _preprocess_shapes(shapes)
    ntype = estimate_keypoints_norm_type(inputs)
    _assert_exactly_n_shapes_partial = functools.partial(
        _assert_exactly_n_shapes,
        from_ntype=ntype, to_ntype="List[KeypointsOnImage]",
        shapes=shapes)

    if ntype == "None":
        return inputs
    if ntype in ["array[float]", "array[int]", "array[uint]"]:
        _assert_single_array_ndim(inputs, 3, "(N,K,2)", "KeypointsOnImage")
        _assert_single_array_last_dim_exactly(inputs, 2, "KeypointsOnImage")
        _assert_exactly_n_shapes_partial(n=len(inputs))
        return [
            KeypointsOnImage.from_xy_array(attr_i, shape=shape)
            for attr_i, shape
            in zip(inputs, shapes)
        ]
    if ntype == "tuple[number,size=2]":
        _assert_exactly_n_shapes_partial(n=1)
        return [KeypointsOnImage([Keypoint(x=inputs[0], y=inputs[1])],
                                 shape=shapes[0])]
    if ntype == "Keypoint":
        _assert_exactly_n_shapes_partial(n=1)
        return [KeypointsOnImage([inputs], shape=shapes[0])]
    if ntype == "KeypointsOnImage":
        return [inputs]
    if ntype == "iterable[empty]":
        return None
    if ntype in ["iterable-array[float]",
                 "iterable-array[int]",
                 "iterable-array[uint]"]:
        _assert_many_arrays_ndim(inputs, 2, "(K,2)", "KeypointsOnImage")
        _assert_many_arrays_last_dim_exactly(inputs, 2, "KeypointsOnImage")
        _assert_exactly_n_shapes_partial(n=len(inputs))
        return [
            KeypointsOnImage.from_xy_array(attr_i, shape=shape)
            for attr_i, shape
            in zip(inputs, shapes)
        ]
    if ntype == "iterable-tuple[number,size=2]":
        _assert_exactly_n_shapes_partial(n=1)
        return [KeypointsOnImage([Keypoint(x=x, y=y) for x, y in inputs],
                                 shape=shapes[0])]
    if ntype == "iterable-Keypoint":
        _assert_exactly_n_shapes_partial(n=1)
        return [KeypointsOnImage(inputs, shape=shapes[0])]
    if ntype == "iterable-KeypointsOnImage":
        return inputs
    if ntype == "iterable-iterable[empty]":
        return None
    if ntype == "iterable-iterable-tuple[number,size=2]":
        _assert_exactly_n_shapes_partial(n=len(inputs))
        return [
            KeypointsOnImage.from_xy_array(
                np.array(attr_i, dtype=np.float32),
                shape=shape)
            for attr_i, shape
            in zip(inputs, shapes)
        ]

    assert ntype == "iterable-iterable-Keypoint", (
        "Got unknown normalization type '%s'." % (ntype,))
    _assert_exactly_n_shapes_partial(n=len(inputs))
    return [KeypointsOnImage(attr_i, shape=shape)
            for attr_i, shape
            in zip(inputs, shapes)]


def normalize_bounding_boxes(inputs, shapes=None):
    # TODO get rid of this deferred import
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

    shapes = _preprocess_shapes(shapes)
    ntype = estimate_bounding_boxes_norm_type(inputs)
    _assert_exactly_n_shapes_partial = functools.partial(
        _assert_exactly_n_shapes,
        from_ntype=ntype, to_ntype="List[BoundingBoxesOnImage]",
        shapes=shapes)

    if ntype == "None":
        return None
    if ntype in ["array[float]", "array[int]", "array[uint]"]:
        _assert_single_array_ndim(inputs, 3, "(N,B,4)", "BoundingBoxesOnImage")
        _assert_single_array_last_dim_exactly(
            inputs, 4, "BoundingBoxesOnImage")
        _assert_exactly_n_shapes_partial(n=len(inputs))
        return [
            BoundingBoxesOnImage.from_xyxy_array(attr_i, shape=shape)
            for attr_i, shape
            in zip(inputs, shapes)
        ]
    if ntype == "tuple[number,size=4]":
        _assert_exactly_n_shapes_partial(n=1)
        return [
            BoundingBoxesOnImage(
                [BoundingBox(
                    x1=inputs[0], y1=inputs[1],
                    x2=inputs[2], y2=inputs[3])],
                shape=shapes[0])
        ]
    if ntype == "BoundingBox":
        _assert_exactly_n_shapes_partial(n=1)
        return [BoundingBoxesOnImage([inputs], shape=shapes[0])]
    if ntype == "BoundingBoxesOnImage":
        return [inputs]
    if ntype == "iterable[empty]":
        return None
    if ntype in ["iterable-array[float]",
                 "iterable-array[int]",
                 "iterable-array[uint]"]:
        _assert_many_arrays_ndim(inputs, 2, "(B,4)", "BoundingBoxesOnImage")
        _assert_many_arrays_last_dim_exactly(inputs, 4, "BoundingBoxesOnImage")
        _assert_exactly_n_shapes_partial(n=len(inputs))
        return [
            BoundingBoxesOnImage.from_xyxy_array(attr_i, shape=shape)
            for attr_i, shape
            in zip(inputs, shapes)
        ]
    if ntype == "iterable-tuple[number,size=4]":
        _assert_exactly_n_shapes_partial(n=1)
        return [
            BoundingBoxesOnImage(
                [BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                 for x1, y1, x2, y2 in inputs],
                shape=shapes[0])
        ]
    if ntype == "iterable-BoundingBox":
        _assert_exactly_n_shapes_partial(n=1)
        return [BoundingBoxesOnImage(inputs, shape=shapes[0])]
    if ntype == "iterable-BoundingBoxesOnImage":
        return inputs
    if ntype == "iterable-iterable[empty]":
        return None
    if ntype == "iterable-iterable-tuple[number,size=4]":
        _assert_exactly_n_shapes_partial(n=len(inputs))
        return [
            BoundingBoxesOnImage.from_xyxy_array(
                np.array(attr_i, dtype=np.float32),
                shape=shape)
            for attr_i, shape
            in zip(inputs, shapes)
        ]

    assert ntype == "iterable-iterable-BoundingBox", (
        "Got unknown normalization type '%s'." % (ntype,))
    _assert_exactly_n_shapes_partial(n=len(inputs))
    return [BoundingBoxesOnImage(attr_i, shape=shape)
            for attr_i, shape
            in zip(inputs, shapes)]


def normalize_polygons(inputs, shapes=None):
    # TODO get rid of this deferred import
    from imgaug.augmentables.polys import Polygon, PolygonsOnImage

    return _normalize_polygons_and_line_strings(
        cls_single=Polygon,
        cls_oi=PolygonsOnImage,
        axis_names=["#polys", "#points"],
        estimate_ntype_func=estimate_polygons_norm_type,
        inputs=inputs, shapes=shapes
    )


def normalize_line_strings(inputs, shapes=None):
    # TODO get rid of this deferred import
    from imgaug.augmentables.lines import LineString, LineStringsOnImage

    return _normalize_polygons_and_line_strings(
        cls_single=LineString,
        cls_oi=LineStringsOnImage,
        axis_names=["#lines", "#points"],
        estimate_ntype_func=estimate_line_strings_norm_type,
        inputs=inputs, shapes=shapes
    )


def _normalize_polygons_and_line_strings(cls_single, cls_oi, axis_names,
                                         estimate_ntype_func,
                                         inputs, shapes=None):
    cls_single_name = cls_single.__name__
    cls_oi_name = cls_oi.__name__
    axis_names_4_str = "(N,%s,%s,2)" % (axis_names[0], axis_names[1])
    axis_names_3_str = "(%s,%s,2)" % (axis_names[0], axis_names[1])
    axis_names_2_str = "(%s,2)" % (axis_names[1],)

    shapes = _preprocess_shapes(shapes)
    ntype = estimate_ntype_func(inputs)
    _assert_exactly_n_shapes_partial = functools.partial(
        _assert_exactly_n_shapes,
        from_ntype=ntype, to_ntype=("List[%s]" % (cls_oi_name,)),
        shapes=shapes)

    if ntype == "None":
        return None
    if ntype in ["array[float]", "array[int]", "array[uint]"]:
        _assert_single_array_ndim(inputs, 4, axis_names_4_str,
                                  cls_oi_name)
        _assert_single_array_last_dim_exactly(inputs, 2, cls_oi_name)
        _assert_exactly_n_shapes_partial(n=len(inputs))
        return [
            cls_oi(
                [cls_single(points) for points in attr_i],
                shape=shape)
            for attr_i, shape
            in zip(inputs, shapes)
        ]
    if ntype == cls_single_name:
        _assert_exactly_n_shapes_partial(n=1)
        return [cls_oi([inputs], shape=shapes[0])]
    if ntype == cls_oi_name:
        return [inputs]
    if ntype == "iterable[empty]":
        return None
    if ntype in ["iterable-array[float]",
                 "iterable-array[int]",
                 "iterable-array[uint]"]:
        _assert_many_arrays_ndim(inputs, 3, axis_names_3_str,
                                 cls_oi_name)
        _assert_many_arrays_last_dim_exactly(inputs, 2, cls_oi_name)
        _assert_exactly_n_shapes_partial(n=len(inputs))
        return [
            cls_oi([cls_single(points) for points in attr_i], shape=shape)
            for attr_i, shape
            in zip(inputs, shapes)
        ]
    if ntype == "iterable-tuple[number,size=2]":
        _assert_exactly_n_shapes_partial(n=1)
        return [cls_oi([cls_single(inputs)], shape=shapes[0])]
    if ntype == "iterable-Keypoint":
        _assert_exactly_n_shapes_partial(n=1)
        return [cls_oi([cls_single(inputs)], shape=shapes[0])]
    if ntype == ("iterable-%s" % (cls_single_name,)):
        _assert_exactly_n_shapes_partial(n=1)
        return [cls_oi(inputs, shape=shapes[0])]
    if ntype == ("iterable-%s" % (cls_oi_name,)):
        return inputs
    if ntype == "iterable-iterable[empty]":
        return None
    if ntype in ["iterable-iterable-array[float]",
                 "iterable-iterable-array[int]",
                 "iterable-iterable-array[uint]"]:
        _assert_many_arrays_ndim(inputs, 2, axis_names_2_str, cls_oi_name)
        _assert_many_arrays_last_dim_exactly(inputs, 2, cls_oi_name)
        _assert_exactly_n_shapes_partial(n=len(inputs))
        return [
            cls_oi(
                [cls_single(points) for points in attr_i],
                shape=shape)
            for attr_i, shape
            in zip(inputs, shapes)
        ]
    if ntype == "iterable-iterable-tuple[number,size=2]":
        _assert_exactly_n_shapes_partial(n=1)
        return [
            cls_oi([cls_single(attr_i) for attr_i in inputs],
                   shape=shapes[0])
        ]
    if ntype == "iterable-iterable-Keypoint":
        _assert_exactly_n_shapes_partial(n=1)
        return [
            cls_oi([cls_single(attr_i) for attr_i in inputs],
                   shape=shapes[0])
        ]
    if ntype == ("iterable-iterable-%s" % (cls_single_name,)):
        _assert_exactly_n_shapes_partial(n=len(inputs))
        return [
            cls_oi(attr_i, shape=shape)
            for attr_i, shape
            in zip(inputs, shapes)
        ]
    if ntype == "iterable-iterable-iterable[empty]":
        return None

    assert ntype in ["iterable-iterable-iterable-tuple[number,size=2]",
                     "iterable-iterable-iterable-Keypoint"], (
                         "Got unknown normalization type '%s'." % (ntype,))
    _assert_exactly_n_shapes_partial(n=len(inputs))
    return [
        cls_oi(
            [cls_single(points) for points in attr_i],
            shape=shape)
        for attr_i, shape
        in zip(inputs, shapes)
    ]


def invert_normalize_images(images, images_old):
    if images_old is None:
        assert images is None, (
            "Expected (normalized) 'images' to be None due to (unnormalized) "
            "'images_old' being None. Got type %s instead." % (type(images),))
        return None
    if ia.is_np_array(images_old):
        if not ia.is_np_array(images):
            # Images were turned from array to list during augmentation.
            # This can happen for e.g. crop operations.
            # We will proceed as if the old images were a list.
            # One could also generate an array-output if all shapes and dtypes
            # in `images` are the same. This was not done here, because
            # (a) that would incur a performance penalty and (b) it would
            # lead to less consistent outputs.
            if images_old.ndim == 2:
                # dont interpret first axis as N if `images_old` was a single
                # image
                return invert_normalize_images(images, [images_old])
            return invert_normalize_images(images, list(images_old))

        if images_old.ndim == 2:
            assert images.shape[0] == 1, (
                "Expected normalized images of shape (N,H,W,C) to have "
                "N=1 due to the unnormalized images being a single 2D "
                "image. Got instead N=%d and shape %s." % (
                    images.shape[0], images.shape))
            assert images.shape[3] == 1, (
                "Expected normalized images of shape (N,H,W,C) to have "
                "C=1 due to the unnormalized images being a single 2D "
                "image. Got instead C=%d and shape %s." % (
                    images.shape[3], images.shape))
            return images[0, ..., 0]
        if images_old.ndim == 3:
            assert images.shape[3] == 1, (
                "Expected normalized images of shape (N,H,W,C) to have "
                "C=1 due to unnormalized images being a single 3D image. "
                "Got instead C=%d and shape %s" % (
                    images.shape[3], images.shape))
            return images[..., 0]
        return images
    if ia.is_iterable(images_old):
        result = []
        for image, image_old in zip(images, images_old):
            if image_old.ndim == 2:
                assert image.shape[2] == 1, (
                    "Expected each image of shape (H,W,C) to have C=1 due to "
                    "the corresponding unnormalized image being a 2D image. "
                    "Got instead C=%d and shape %s." % (
                        image.shape[2], image.shape))
                result.append(image[:, :, 0])
            else:
                assert image_old.ndim == 3, (
                    "Expected 'image_old' to be three-dimensional, got %d "
                    "dimensions and shape %s." % (
                        image_old.ndim, image_old.shape))
                result.append(image)
        return result
    raise ValueError(
        "Expected argument 'images_old' to be any of the following: "
        "None or array or iterable of array. Got type: %s." % (
            type(images_old),))


def invert_normalize_heatmaps(heatmaps, heatmaps_old):
    ntype = estimate_heatmaps_norm_type(heatmaps_old)
    if ntype == "None":
        assert heatmaps is None, (
            "Expected (normalized) 'heatmaps' to be None due (unnormalized) "
            "'heatmaps_old' being None. Got type %s instead." % (
                type(heatmaps),))
        return heatmaps
    if ntype == "array[float]":
        assert len(heatmaps) == heatmaps_old.shape[0], (
            "Expected as many heatmaps after normalization as before "
            "normalization. Got %d (after) and %d (before)." % (
                len(heatmaps), heatmaps_old.shape[0]))
        input_dtype = heatmaps_old.dtype
        return restore_dtype_and_merge(
            [hm_i.arr_0to1 for hm_i in heatmaps],
            input_dtype)
    if ntype == "HeatmapsOnImage":
        assert len(heatmaps) == 1, (
            "Expected as many heatmaps after normalization as before "
            "normalization. Got %d (after) and %d (before)." % (
                len(heatmaps), 1))
        return heatmaps[0]
    if ntype == "iterable[empty]":
        assert heatmaps is None, (
            "Expected heatmaps after normalization to be None, due to the "
            "heatmaps before normalization being an empty iterable. "
            "Got type %s instead." % (type(heatmaps),))
        return []
    if ntype == "iterable-array[float]":
        nonempty, _, _ = find_first_nonempty(heatmaps_old)
        input_dtype = nonempty.dtype
        return [restore_dtype_and_merge(hm_i.arr_0to1, input_dtype)
                for hm_i in heatmaps]

    assert ntype == "iterable-HeatmapsOnImage", (
        "Got unknown normalization type '%s'." % (ntype,))
    return heatmaps


def invert_normalize_segmentation_maps(segmentation_maps,
                                       segmentation_maps_old):
    ntype = estimate_segmaps_norm_type(segmentation_maps_old)
    if ntype == "None":
        assert segmentation_maps is None, (
            "Expected (normalized) 'segmentation_maps' to be None due "
            "(unnormalized) 'segmentation_maps_old' being None. Got type %s "
            "instead." % (type(segmentation_maps),))
        return segmentation_maps
    if ntype in ["array[int]", "array[uint]", "array[bool]"]:
        assert len(segmentation_maps) == segmentation_maps_old.shape[0], (
            "Expected as many segmentation maps after normalization as before "
            "normalization. Got %d (after) and %d (before)." % (
                len(segmentation_maps), segmentation_maps_old.shape[0]))
        input_dtype = segmentation_maps_old.dtype
        return restore_dtype_and_merge(
            [segmap_i.get_arr() for segmap_i in segmentation_maps],
            input_dtype)
    if ntype == "SegmentationMapsOnImage":
        assert len(segmentation_maps) == 1, (
            "Expected as many segmentation maps after normalization as before "
            "normalization. Got %d (after) and %d (before)." % (
                len(segmentation_maps), 1))
        return segmentation_maps[0]
    if ntype == "iterable[empty]":
        assert segmentation_maps is None, (
            "Expected segmentation maps after normalization to be None, due "
            "to the segmentation maps before normalization being an empty "
            "iterable. Got type %s instead." % (type(segmentation_maps),))
        return []
    if ntype in ["iterable-array[int]",
                 "iterable-array[uint]",
                 "iterable-array[bool]"]:
        nonempty, _, _ = find_first_nonempty(segmentation_maps_old)
        input_dtype = nonempty.dtype
        return [restore_dtype_and_merge(segmap_i.get_arr(), input_dtype)
                for segmap_i in segmentation_maps]

    assert ntype == "iterable-SegmentationMapsOnImage", (
        "Got unknown normalization type '%s'." % (ntype,))
    return segmentation_maps


def invert_normalize_keypoints(keypoints, keypoints_old):
    ntype = estimate_keypoints_norm_type(keypoints_old)
    if ntype == "None":
        assert keypoints is None, (
            "Expected (normalized) 'keypoints' to be None due (unnormalized) "
            "'keypoints_old' being None. Got type %s instead." % (
                type(keypoints),))
        return keypoints
    if ntype in ["array[float]", "array[int]", "array[uint]"]:
        assert len(keypoints) == 1, (
            "Expected a single KeypointsOnImage instance after normalization "
            "due to getting a single ndarray before normalization. "
            "Got %d instances instead." % (len(keypoints),))
        input_dtype = keypoints_old.dtype
        return restore_dtype_and_merge(
            [kpsoi.to_xy_array() for kpsoi in keypoints],
            input_dtype)
    if ntype == "tuple[number,size=2]":
        assert len(keypoints) == 1, (
            "Expected a single KeypointsOnImage instance after normalization "
            "due to getting a single (x,y) tuple before normalization. "
            "Got %d instances instead." % (len(keypoints),))
        assert len(keypoints[0].keypoints) == 1, (
            "Expected a KeypointsOnImage instance containing a single "
            "Keypoint after normalization due to getting a single (x,y) tuple "
            "before normalization. Got %d keypoints instead." % (
                len(keypoints[0].keypoints)
            ))
        return (keypoints[0].keypoints[0].x,
                keypoints[0].keypoints[0].y)
    if ntype == "Keypoint":
        assert len(keypoints) == 1, (
            "Expected a single KeypointsOnImage instance after normalization "
            "due to getting a single Keypoint before normalization. "
            "Got %d instances instead." % (len(keypoints),))
        assert len(keypoints[0].keypoints) == 1, (
            "Expected a KeypointsOnImage instance containing a single "
            "Keypoint after normalization due to getting a single Keypoint "
            "before normalization. Got %d keypoints instead." % (
                len(keypoints[0].keypoints)
            ))
        return keypoints[0].keypoints[0]
    if ntype == "KeypointsOnImage":
        assert len(keypoints) == 1, (
            "Expected a single KeypointsOnImage instance after normalization "
            "due to getting a single KeypointsOnImage before normalization. "
            "Got %d instances instead." % (len(keypoints),))
        return keypoints[0]
    if ntype == "iterable[empty]":
        assert keypoints is None, (
            "Expected keypoints after normalization to be None, due "
            "to the keypoints before normalization being an empty "
            "iterable. Got type %s instead." % (type(keypoints),))
        return []
    if ntype in ["iterable-array[float]",
                 "iterable-array[int]",
                 "iterable-array[uint]"]:
        nonempty, _, _ = find_first_nonempty(keypoints_old)
        input_dtype = nonempty.dtype
        return [
            restore_dtype_and_merge(kps_i.to_xy_array(), input_dtype)
            for kps_i in keypoints]
    if ntype == "iterable-tuple[number,size=2]":
        assert len(keypoints) == 1, (
            "Expected a single KeypointsOnImage instance after normalization "
            "due to getting an iterable of (x,y) tuples before "
            "normalization. Got %d instances instead." % (len(keypoints),))
        return [
            (kp.x, kp.y) for kp in keypoints[0].keypoints]
    if ntype == "iterable-Keypoint":
        assert len(keypoints) == 1, (
            "Expected a single KeypointsOnImage instance after normalization "
            "due to getting an iterable of Keypoint before "
            "normalization. Got %d instances instead." % (len(keypoints),))
        return keypoints[0].keypoints
    if ntype == "iterable-KeypointsOnImage":
        return keypoints
    if ntype == "iterable-iterable[empty]":
        assert keypoints is None, (
            "Expected keypoints after normalization to be None, due "
            "to the keypoints before normalization being an empty "
            "iterable of iterables. Got type %s instead." % (type(keypoints),))
        return keypoints_old[:]
    if ntype == "iterable-iterable-tuple[number,size=2]":
        return [
            [(kp.x, kp.y) for kp in kpsoi.keypoints]
            for kpsoi in keypoints]

    assert ntype == "iterable-iterable-Keypoint", (
        "Got unknown normalization type '%s'." % (ntype,))
    return [kpsoi.keypoints[:] for kpsoi in keypoints]


def invert_normalize_bounding_boxes(bounding_boxes, bounding_boxes_old):
    ntype = estimate_normalization_type(bounding_boxes_old)
    if ntype == "None":
        assert bounding_boxes is None, (
            "Expected (normalized) 'bounding_boxes' to be None due "
            "(unnormalized) 'bounding_boxes_old' being None. Got type %s "
            "instead." % (type(bounding_boxes),))
        return bounding_boxes
    if ntype in ["array[float]", "array[int]", "array[uint]"]:
        assert len(bounding_boxes) == 1, (
            "Expected a single BoundingBoxesOnImage instance after "
            "normalization due to getting a single ndarray before "
            "normalization. Got %d instances instead." % (
                len(bounding_boxes),))
        input_dtype = bounding_boxes_old.dtype
        return restore_dtype_and_merge([
            bbsoi.to_xyxy_array() for bbsoi in bounding_boxes
        ], input_dtype)
    if ntype == "tuple[number,size=4]":
        assert len(bounding_boxes) == 1, (
            "Expected a single BoundingBoxesOnImage instance after "
            "normalization due to getting a single (x1,y1,x2,y2) tuple before "
            "normalization. Got %d instances instead." % (
                len(bounding_boxes),))
        assert len(bounding_boxes[0].bounding_boxes) == 1, (
            "Expected a BoundingBoxesOnImage instance containing a single "
            "BoundingBox after normalization due to getting a single "
            "(x1,y1,x2,y2) tuple before normalization. Got %d bounding boxes "
            "instead." % (len(bounding_boxes[0].bounding_boxes)))
        bb = bounding_boxes[0].bounding_boxes[0]
        return bb.x1, bb.y1, bb.x2, bb.y2
    if ntype == "BoundingBox":
        assert len(bounding_boxes) == 1, (
            "Expected a single BoundingBoxesOnImage instance after "
            "normalization due to getting a single BoundingBox before "
            "normalization. Got %d instances instead." % (
                len(bounding_boxes),))
        assert len(bounding_boxes[0].bounding_boxes) == 1, (
            "Expected a BoundingBoxesOnImage instance containing a single "
            "BoundingBox after normalization due to getting a single "
            "BoundingBox before normalization. Got %d bounding boxes "
            "instead." % (len(bounding_boxes[0].bounding_boxes)))
        return bounding_boxes[0].bounding_boxes[0]
    if ntype == "BoundingBoxesOnImage":
        assert len(bounding_boxes) == 1, (
            "Expected a single BoundingBoxesOnImage instance after "
            "normalization due to getting a single BoundingBoxesOnImage "
            "before normalization. Got %d instances instead." % (
                len(bounding_boxes),))
        return bounding_boxes[0]
    if ntype == "iterable[empty]":
        assert bounding_boxes is None, (
            "Expected bounding boxes after normalization to be None, due "
            "to the bounding boxes before normalization being an empty "
            "iterable. Got type %s instead." % (type(bounding_boxes),))
        return []
    if ntype in ["iterable-array[float]",
                 "iterable-array[int]",
                 "iterable-array[uint]"]:
        nonempty, _, _ = find_first_nonempty(bounding_boxes_old)
        input_dtype = nonempty.dtype
        return [
            restore_dtype_and_merge(bbsoi.to_xyxy_array(), input_dtype)
            for bbsoi in bounding_boxes]
    if ntype == "iterable-tuple[number,size=4]":
        assert len(bounding_boxes) == 1, (
            "Expected a single BoundingBoxesOnImage instance after "
            "normalization due to getting a an iterable of (x1,y1,x2,y2) "
            "tuples before normalization. Got %d instances instead." % (
                len(bounding_boxes),))
        return [
            (bb.x1, bb.y1, bb.x2, bb.y2)
            for bb in bounding_boxes[0].bounding_boxes]
    if ntype == "iterable-BoundingBox":
        assert len(bounding_boxes) == 1, (
            "Expected a single BoundingBoxesOnImage instance after "
            "normalization due to getting an iterable of BoundingBox before "
            "normalization. Got %d instances instead." % (
                len(bounding_boxes),))
        return bounding_boxes[0].bounding_boxes
    if ntype == "iterable-BoundingBoxesOnImage":
        return bounding_boxes
    if ntype == "iterable-iterable[empty]":
        assert bounding_boxes is None, (
            "Expected bounding boxes after normalization to be None, due "
            "to the bounding boxes before normalization being an empty "
            "iterable of iterables. Got type %s instead." % (
                type(bounding_boxes),))
        return bounding_boxes_old[:]
    if ntype == "iterable-iterable-tuple[number,size=4]":
        return [
            [(bb.x1, bb.y1, bb.x2, bb.y2) for bb in bbsoi.bounding_boxes]
            for bbsoi in bounding_boxes]

    assert ntype == "iterable-iterable-BoundingBox", (
        "Got unknown normalization type '%s'." % (ntype,))
    return [bbsoi.bounding_boxes[:] for bbsoi in bounding_boxes]


def invert_normalize_polygons(polygons, polygons_old):
    return _invert_normalize_polygons_and_line_strings(
        polygons, polygons_old, estimate_polygons_norm_type,
        "Polygon",
        "PolygonsOnImage",
        lambda psoi: psoi.polygons,
        lambda poly: poly.exterior)


def invert_normalize_line_strings(line_strings, line_strings_old):
    return _invert_normalize_polygons_and_line_strings(
        line_strings, line_strings_old, estimate_line_strings_norm_type,
        "LineString",
        "LineStringsOnImage",
        lambda lsoi: lsoi.line_strings,
        lambda ls: ls.coords)


def _invert_normalize_polygons_and_line_strings(inputs, inputs_old,
                                                estimate_ntype_func,
                                                cls_single_name,
                                                cls_oi_name,
                                                get_entities_func,
                                                get_points_func):
    # TODO get rid of this deferred import
    from imgaug.augmentables.kps import Keypoint

    ntype = estimate_ntype_func(inputs_old)
    if ntype == "None":
        assert inputs is None, (
            "Expected (normalized) polygons/line strings to be None due "
            "(unnormalized) polygons/line strings being None. Got type %s "
            "instead." % (type(inputs),))
        return inputs
    if ntype in ["array[float]", "array[int]", "array[uint]"]:
        input_dtype = inputs_old.dtype
        return restore_dtype_and_merge([
            [get_points_func(entity) for entity in get_entities_func(oi)]
            for oi in inputs
        ], input_dtype)
    if ntype == cls_single_name:
        assert len(inputs) == 1, (
            "Expected a single %s instance after normalization "
            "due to getting a single %s before normalization. "
            "Got %d instances instead." % (
                cls_oi_name, cls_single_name, len(inputs),))
        assert len(get_entities_func(inputs[0])) == 1, (
            "Expected a %s instance containing a single "
            "%s after normalization due to getting a single %s "
            "before normalization. Got %d instances instead." % (
                cls_oi_name, cls_single_name, cls_single_name,
                len(get_entities_func(inputs[0]))))
        return get_entities_func(inputs[0])[0]
    if ntype == cls_oi_name:
        assert len(inputs) == 1, (
            "Expected a single %s instance after normalization "
            "due to getting a single %s before normalization. "
            "Got %d instances instead." % (
                cls_oi_name, cls_oi_name, len(inputs),))
        return inputs[0]
    if ntype == "iterable[empty]":
        assert inputs is None, (
            "Expected polygons/line strings after normalization to be None, "
            "due to the polygons/line strings before normalization being an "
            "empty iterable. Got type %s instead." % (type(inputs),))
        return []
    if ntype in ["iterable-array[float]",
                 "iterable-array[int]",
                 "iterable-array[uint]"]:
        nonempty, _, _ = find_first_nonempty(inputs_old)
        input_dtype = nonempty.dtype
        return [
            restore_dtype_and_merge(
                [get_points_func(entity) for entity in get_entities_func(oi)],
                input_dtype)
            for oi in inputs
        ]
    if ntype == "iterable-tuple[number,size=2]":
        assert len(inputs) == 1, (
            "Expected a single %s instance after normalization "
            "due to getting an iterable of (x,y) tuples before "
            "normalization. Got %d instances instead." % (
                cls_oi_name, len(inputs),))
        assert len(get_entities_func(inputs[0])) == 1, (
            "Expected a %s instance after normalization "
            "containing a single %s instance due to getting an iterable "
            "of (x,y) tuples before normalization. "
            "Got a %s with %d %s instances instead." % (
                cls_oi_name, cls_single_name, cls_oi_name, cls_single_name,
                len(inputs),))
        return [(point[0], point[1])
                for point in get_points_func(get_entities_func(inputs[0])[0])]
    if ntype == "iterable-Keypoint":
        assert len(inputs) == 1, (
            "Expected a single %s instance after normalization "
            "due to getting an iterable of Keypoint before "
            "normalization. Got %d instances instead." % (
                cls_oi_name, len(inputs),))
        assert len(get_entities_func(inputs[0])) == 1, (
            "Expected a %s instance after normalization "
            "containing a single %s instance due to getting an iterable "
            "of Keypoint before normalization. "
            "Got a %s with %d %s instances instead." % (
                cls_oi_name, cls_single_name, cls_oi_name, cls_single_name,
                len(inputs),))
        return [Keypoint(x=point[0], y=point[1])
                for point in get_points_func(get_entities_func(inputs[0])[0])]
    if ntype == ("iterable-%s" % (cls_single_name,)):
        assert len(inputs) == 1, (
            "Expected a single %s instance after normalization "
            "due to getting an iterable of %s before "
            "normalization. Got %d instances instead." % (
                cls_oi_name, cls_single_name, len(inputs),))
        assert len(get_entities_func(inputs[0])) == len(inputs_old), (
            "Expected a %s instance after normalization "
            "containing a single %s instance due to getting an iterable "
            "of %s before normalization. "
            "Got a %s with %d %s instances instead." % (
                cls_oi_name, cls_single_name, cls_single_name, cls_oi_name,
                cls_single_name, len(inputs),))
        return get_entities_func(inputs[0])
    if ntype == ("iterable-%s" % (cls_oi_name,)):
        return inputs
    if ntype == "iterable-iterable[empty]":
        assert inputs is None, (
            "Expected polygons/line strings after normalization to be None, "
            "due to the polygons/line strings before normalization being an "
            "empty iterable of iterables. Got type %s instead." % (
                type(inputs),))
        return inputs_old[:]
    if ntype in ["iterable-iterable-array[float]",
                 "iterable-iterable-array[int]",
                 "iterable-iterable-array[uint]"]:
        nonempty, _, _ = find_first_nonempty(inputs_old)
        input_dtype = nonempty.dtype
        return [
            [restore_dtype_and_merge(get_points_func(entity), input_dtype)
             for entity in get_entities_func(oi)]
            for oi in inputs
        ]
    if ntype == "iterable-iterable-tuple[number,size=2]":
        assert len(inputs) == 1, (
            "Expected a single %s instance after normalization "
            "due to getting an iterable of iterables of (x,y) tuples before "
            "normalization. Got %d instances instead." % (
                cls_oi_name, len(inputs),))
        return [
            [(point[0], point[1]) for point in get_points_func(entity)]
            for entity in get_entities_func(inputs[0])]
    if ntype == "iterable-iterable-Keypoint":
        assert len(inputs) == 1, (
            "Expected a single %s instance after normalization "
            "due to getting an iterable of iterables of Keypoint before "
            "normalization. Got %d instances instead." % (
                cls_oi_name, len(inputs),))
        return [
            [Keypoint(x=point[0], y=point[1])
             for point in get_points_func(entity)]
            for entity in get_entities_func(inputs[0])]
    if ntype == ("iterable-iterable-%s" % (cls_single_name,)):
        return [get_entities_func(oi) for oi in inputs]
    if ntype == "iterable-iterable-iterable[empty]":
        return inputs_old[:]
    if ntype == "iterable-iterable-iterable-tuple[number,size=2]":
        return [
            [
                [
                    (point[0], point[1])
                    for point in get_points_func(entity)
                ]
                for entity in get_entities_func(oi)
            ]
            for oi in inputs]

    assert ntype == "iterable-iterable-iterable-Keypoint", (
        "Got unknown normalization type '%s'." % (ntype,))
    return [
        [
            [
                Keypoint(x=point[0], y=point[1])
                for point in get_points_func(entity)
            ]
            for entity in get_entities_func(oi)
        ]
        for oi in inputs]


def _assert_is_of_norm_type(type_str, valid_type_strs, arg_name):
    assert type_str in valid_type_strs, (
        "Got an unknown datatype for argument '%s'. "
        "Expected datatypes were: %s. Got: %s." % (
            arg_name, ", ".join(valid_type_strs), type_str))


def estimate_heatmaps_norm_type(heatmaps):
    type_str = estimate_normalization_type(heatmaps)
    valid_type_strs = [
        "None",
        "array[float]",
        "HeatmapsOnImage",
        "iterable[empty]",
        "iterable-array[float]",
        "iterable-HeatmapsOnImage"
    ]
    _assert_is_of_norm_type(type_str, valid_type_strs, "heatmaps")
    return type_str


def estimate_segmaps_norm_type(segmentation_maps):
    type_str = estimate_normalization_type(segmentation_maps)
    valid_type_strs = [
        "None",
        "array[int]",
        "array[uint]",
        "array[bool]",
        "SegmentationMapsOnImage",
        "iterable[empty]",
        "iterable-array[int]",
        "iterable-array[uint]",
        "iterable-array[bool]",
        "iterable-SegmentationMapsOnImage"
    ]
    _assert_is_of_norm_type(
        type_str, valid_type_strs, "segmentation_maps")
    return type_str


def estimate_keypoints_norm_type(keypoints):
    type_str = estimate_normalization_type(keypoints)
    valid_type_strs = [
        "None",
        "array[float]",
        "array[int]",
        "array[uint]",
        "tuple[number,size=2]",
        "Keypoint",
        "KeypointsOnImage",
        "iterable[empty]",
        "iterable-array[float]",
        "iterable-array[int]",
        "iterable-array[uint]",
        "iterable-tuple[number,size=2]",
        "iterable-Keypoint",
        "iterable-KeypointsOnImage",
        "iterable-iterable[empty]",
        "iterable-iterable-tuple[number,size=2]",
        "iterable-iterable-Keypoint"
    ]
    _assert_is_of_norm_type(type_str, valid_type_strs, "keypoints")
    return type_str


def estimate_bounding_boxes_norm_type(bounding_boxes):
    type_str = estimate_normalization_type(bounding_boxes)
    valid_type_strs = [
        "None",
        "array[float]",
        "array[int]",
        "array[uint]",
        "tuple[number,size=4]",
        "BoundingBox",
        "BoundingBoxesOnImage",
        "iterable[empty]",
        "iterable-array[float]",
        "iterable-array[int]",
        "iterable-array[uint]",
        "iterable-tuple[number,size=4]",
        "iterable-BoundingBox",
        "iterable-BoundingBoxesOnImage",
        "iterable-iterable[empty]",
        "iterable-iterable-tuple[number,size=4]",
        "iterable-iterable-BoundingBox"
    ]
    _assert_is_of_norm_type(
        type_str, valid_type_strs, "bounding_boxes")
    return type_str


def estimate_polygons_norm_type(polygons):
    return _estimate_polygons_and_line_segments_norm_type(
        polygons, "Polygon", "PolygonsOnImage", "polygons")


def estimate_line_strings_norm_type(line_strings):
    return _estimate_polygons_and_line_segments_norm_type(
        line_strings, "LineString", "LineStringsOnImage", "line_strings")


def _estimate_polygons_and_line_segments_norm_type(inputs, cls_single_name,
                                                   cls_oi_name,
                                                   augmentable_name):
    type_str = estimate_normalization_type(inputs)
    valid_type_strs = [
        "None",
        "array[float]",
        "array[int]",
        "array[uint]",
        cls_single_name,
        cls_oi_name,
        "iterable[empty]",
        "iterable-array[float]",
        "iterable-array[int]",
        "iterable-array[uint]",
        "iterable-tuple[number,size=2]",
        "iterable-Keypoint",
        "iterable-%s" % (cls_single_name,),
        "iterable-%s" % (cls_oi_name,),
        "iterable-iterable[empty]",
        "iterable-iterable-array[float]",
        "iterable-iterable-array[int]",
        "iterable-iterable-array[uint]",
        "iterable-iterable-tuple[number,size=2]",
        "iterable-iterable-Keypoint",
        "iterable-iterable-%s" % (cls_single_name,),
        "iterable-iterable-iterable[empty]",
        "iterable-iterable-iterable-tuple[number,size=2]",
        "iterable-iterable-iterable-Keypoint"
    ]
    _assert_is_of_norm_type(type_str, valid_type_strs, augmentable_name)
    return type_str


def estimate_normalization_type(inputs):
    nonempty, success, parents = find_first_nonempty(inputs)
    type_str = _nonempty_info_to_type_str(nonempty, success, parents)
    return type_str


def restore_dtype_and_merge(arr, input_dtype):
    if isinstance(arr, list):
        arr = [restore_dtype_and_merge(arr_i, input_dtype)
               for arr_i in arr]
        shapes = [arr_i.shape for arr_i in arr]
        if len(set(shapes)) == 1:
            arr = np.array(arr)

    if ia.is_np_array(arr):
        arr = iadt.restore_dtypes_(arr, input_dtype)
    return arr


def _is_iterable(obj):
    return (
        ia.is_iterable(obj)
        and not isinstance(obj, IAugmentable)  # not e.g. KeypointsOnImage
        and not hasattr(obj, "coords")  # not BBs, Polys, LS
        and not ia.is_string(obj)
    )


def find_first_nonempty(attr, parents=None):
    if parents is None:
        parents = []

    if attr is None or ia.is_np_array(attr):
        return attr, True, parents
    # we exclude strings here, as otherwise we would get the first
    # character, while we want to get the whole string
    if _is_iterable(attr):
        if len(attr) == 0:
            return None, False, parents

        # this prevents the loop below from becoming infinite if the
        # element in the iterable is identical with the iterable,
        # as is the case for e.g. strings
        if attr[0] is attr:
            return attr, True, parents

        # Usually in case of empty lists, all lists should have similar
        # depth. We are a bit more tolerant here and pick the deepest one.
        # Only parents would really need to be tracked here, we could
        # ignore nonempty and success as they will always have the same
        # values (if only empty lists exist).
        nonempty_deepest = None
        success_deepest = False
        parents_deepest = parents
        for attr_i in attr:
            nonempty, success, parents_found = find_first_nonempty(
                attr_i, parents=parents+[attr])
            if success:
                # on any nonempty hit we return immediately as we assume
                # that the datatypes do not change between child branches
                return nonempty, success, parents_found
            if len(parents_found) > len(parents_deepest):
                nonempty_deepest = nonempty
                success_deepest = success
                parents_deepest = parents_found

        return nonempty_deepest, success_deepest, parents_deepest

    return attr, True, parents


def _nonempty_info_to_type_str(nonempty, success, parents):
    assert len(parents) <= 4, "Expected 'parents' to be <=4, got %d." % (
        len(parents),)
    parent_iters = ""
    if len(parents) > 0:
        parent_iters = "%s-" % ("-".join(["iterable"] * len(parents)),)

    if not success:
        return "%siterable[empty]" % (parent_iters,)

    is_parent_tuple = (
        len(parents) >= 1
        and isinstance(parents[-1], tuple)
    )

    if is_parent_tuple:
        is_only_numbers_in_tuple = (
            len(parents[-1]) > 0
            and all([ia.is_single_number(val) for val in parents[-1]])
        )

        if is_only_numbers_in_tuple:
            parent_iters = "-".join(["iterable"] * (len(parents)-1))
            tpl_name = "tuple[number,size=%d]" % (len(parents[-1]),)
            return "-".join([parent_iters, tpl_name]).lstrip("-")

    if nonempty is None:
        return "None"
    if ia.is_np_array(nonempty):
        kind = nonempty.dtype.kind
        kind_map = {"f": "float", "u": "uint", "i": "int", "b": "bool"}
        return "%sarray[%s]" % (
            parent_iters, kind_map[kind] if kind in kind_map else kind)

    # even int, str etc. are objects in python, so anything left should
    # offer a __class__ attribute
    assert isinstance(nonempty, object), (
        "Expected 'nonempty' to be an object, got type %s." % (
            type(nonempty),))
    return "%s%s" % (parent_iters, nonempty.__class__.__name__)
