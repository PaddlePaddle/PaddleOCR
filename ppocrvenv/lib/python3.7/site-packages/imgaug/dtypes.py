"""Functions to interact/analyze with numpy dtypes."""
from __future__ import print_function, division

import numpy as np
import six.moves as sm

import imgaug as ia

KIND_TO_DTYPES = {
    "i": ["int8", "int16", "int32", "int64"],
    "u": ["uint8", "uint16", "uint32", "uint64"],
    "b": ["bool"],
    "f": ["float16", "float32", "float64", "float128"]
}


def normalize_dtypes(dtypes):
    if not isinstance(dtypes, list):
        return [normalize_dtype(dtypes)]
    return [normalize_dtype(dtype) for dtype in dtypes]


def normalize_dtype(dtype):
    assert not isinstance(dtype, list), (
        "Expected a single dtype-like, got a list instead.")
    return (
        dtype.dtype
        if ia.is_np_array(dtype) or ia.is_np_scalar(dtype)
        else np.dtype(dtype)
    )


def change_dtype_(arr, dtype, clip=True, round=True):
    # pylint: disable=redefined-builtin
    assert ia.is_np_array(arr), (
        "Expected array as input, got type %s." % (type(arr),))
    dtype = normalize_dtype(dtype)

    if arr.dtype.name == dtype.name:
        return arr

    if round and arr.dtype.kind == "f" and dtype.kind in ["u", "i", "b"]:
        arr = np.round(arr)

    if clip:
        min_value, _, max_value = get_value_range_of_dtype(dtype)
        arr = clip_(arr, min_value, max_value)

    return arr.astype(dtype, copy=False)


def change_dtypes_(images, dtypes, clip=True, round=True):
    # pylint: disable=redefined-builtin
    if ia.is_np_array(images):
        if ia.is_iterable(dtypes):
            dtypes = normalize_dtypes(dtypes)
            n_distinct_dtypes = len({dt.name for dt in dtypes})
            assert len(dtypes) == len(images), (
                "If an iterable of dtypes is provided to "
                "change_dtypes_(), it must contain as many dtypes as "
                "there are images. Got %d dtypes and %d images." % (
                    len(dtypes), len(images))
            )

            assert n_distinct_dtypes == 1, (
                "If an image array is provided to change_dtypes_(), the "
                "provided 'dtypes' argument must either be a single dtype "
                "or an iterable of N times the *same* dtype for N images. "
                "Got %d distinct dtypes." % (n_distinct_dtypes,)
            )

            dtype = dtypes[0]
        else:
            dtype = normalize_dtype(dtypes)

        result = change_dtype_(images, dtype, clip=clip, round=round)
    elif ia.is_iterable(images):
        dtypes = (
            [normalize_dtype(dtypes)] * len(images)
            if not isinstance(dtypes, list)
            else normalize_dtypes(dtypes)
        )
        assert len(images) == len(dtypes), (
            "Expected the provided images and dtypes to match, but got "
            "iterables of size %d (images) %d (dtypes)." % (
                len(images), len(dtypes)))

        result = images
        for i, (image, dtype) in enumerate(zip(images, dtypes)):
            assert ia.is_np_array(image), (
                "Expected each image to be an ndarray, got type %s "
                "instead." % (type(image),))
            result[i] = change_dtype_(image, dtype, clip=clip, round=round)
    else:
        raise Exception("Expected numpy array or iterable of numpy arrays, "
                        "got type '%s'." % (type(images),))
    return result


# TODO replace this everywhere in the library with change_dtypes_
# TODO mark as deprecated
def restore_dtypes_(images, dtypes, clip=True, round=True):
    # pylint: disable=redefined-builtin
    return change_dtypes_(images, dtypes, clip=clip, round=round)


def copy_dtypes_for_restore(images, force_list=False):
    if ia.is_np_array(images):
        if force_list:
            return [images.dtype for _ in sm.xrange(len(images))]
        return images.dtype
    return [image.dtype for image in images]


def increase_itemsize_of_dtype(dtype, factor):
    dtype = normalize_dtype(dtype)

    assert ia.is_single_integer(factor), (
        "Expected 'factor' to be an integer, got type %s instead." % (
            type(factor),))
    # int8 -> int64 = factor 8
    # uint8 -> uint64 = factor 8
    # float16 -> float128 = factor 8
    assert factor in [1, 2, 4, 8], (
        "The itemsize may only be increased any of the following factors: "
        "1, 2, 4 or 8. Got factor %d." % (factor,))
    assert dtype.kind != "b", "Cannot increase the itemsize of boolean."

    dt_high_name = "%s%d" % (dtype.kind, dtype.itemsize * factor)

    try:
        dt_high = np.dtype(dt_high_name)
        return dt_high
    except TypeError:
        raise TypeError(
            "Unable to create a numpy dtype matching the name '%s'. "
            "This error was caused when trying to find a dtype "
            "that increases the itemsize of dtype '%s' by a factor of %d."
            "This error can be avoided by choosing arrays with lower "
            "resolution dtypes as inputs, e.g. by reducing "
            "float32 to float16." % (
                dt_high_name,
                dtype.name,
                factor
            )
        )


def get_minimal_dtype(arrays, increase_itemsize_factor=1):
    assert isinstance(arrays, list), (
        "Expected a list of arrays or dtypes, got type %s." % (type(arrays),))
    assert len(arrays) > 0, (
        "Cannot estimate minimal dtype of an empty iterable.")

    input_dts = normalize_dtypes(arrays)

    # This loop construct handles (1) list of a single dtype, (2) list of two
    # dtypes and (3) list of 3+ dtypes. Note that promote_dtypes() always
    # expects exactly two dtypes.
    promoted_dt = input_dts[0]
    input_dts = input_dts[1:]
    while len(input_dts) >= 1:
        promoted_dt = np.promote_types(promoted_dt, input_dts[0])
        input_dts = input_dts[1:]

    if increase_itemsize_factor > 1:
        assert isinstance(promoted_dt, np.dtype), (
            "Expected numpy.dtype output from numpy.promote_dtypes, got type "
            "%s." % (type(promoted_dt),))
        return increase_itemsize_of_dtype(promoted_dt,
                                          increase_itemsize_factor)
    return promoted_dt


# TODO rename to: promote_arrays_to_minimal_dtype_
def promote_array_dtypes_(arrays, dtypes=None, increase_itemsize_factor=1):
    if dtypes is None:
        dtypes = normalize_dtypes(arrays)
    elif not isinstance(dtypes, list):
        dtypes = [dtypes]
    dtype = get_minimal_dtype(dtypes,
                              increase_itemsize_factor=increase_itemsize_factor)
    return change_dtypes_(arrays, dtype, clip=False, round=False)


def increase_array_resolutions_(arrays, factor):
    dts = normalize_dtypes(arrays)
    dts = [increase_itemsize_of_dtype(dt, factor) for dt in dts]
    return change_dtypes_(arrays, dts, round=False, clip=False)


def get_value_range_of_dtype(dtype):
    dtype = normalize_dtype(dtype)

    if dtype.kind == "f":
        finfo = np.finfo(dtype)
        return finfo.min, 0.0, finfo.max
    if dtype.kind == "u":
        iinfo = np.iinfo(dtype)
        return iinfo.min, iinfo.min + 0.5 * iinfo.max, iinfo.max
    if dtype.kind == "i":
        iinfo = np.iinfo(dtype)
        return iinfo.min, -0.5, iinfo.max
    if dtype.kind == "b":
        return 0, None, 1

    raise Exception("Cannot estimate value range of dtype '%s' "
                    "(type: %s)" % (str(dtype), type(dtype)))


# TODO call this function wherever data is clipped
def clip_(array, min_value, max_value):
    # uint64 is disallowed, because numpy's clip seems to convert it to float64
    # int64 is disallowed, because numpy's clip converts it to float64 since
    # 1.17
    # TODO find the cause for that
    gate_dtypes(array,
                allowed=["bool",
                         "uint8", "uint16", "uint32",
                         "int8", "int16", "int32",
                         "float16", "float32", "float64", "float128"],
                disallowed=["uint64", "int64"],
                augmenter=None)

    # If the min of the input value range is above the allowed min, we do not
    # have to clip to the allowed min as we cannot exceed it anyways.
    # Analogous for max. In fact, we must not clip then to min/max as that can
    # lead to errors in numpy's clip. E.g.
    #     >>> arr = np.zeros((1,), dtype=np.int32)
    #     >>> np.clip(arr, 0, np.iinfo(np.dtype("uint32")).max)
    # will return
    #     array([-1], dtype=int32)
    # (observed on numpy version 1.15.2).
    min_value_arrdt, _, max_value_arrdt = get_value_range_of_dtype(array.dtype)
    if min_value is not None and min_value < min_value_arrdt:
        min_value = None
    if max_value is not None and max_value_arrdt < max_value:
        max_value = None

    if min_value is not None or max_value is not None:
        # for scalar arrays, i.e. with shape = (), "out" is not a valid
        # argument
        if len(array.shape) == 0:
            array = np.clip(array, min_value, max_value)
        elif array.dtype.name == "int32":
            # Since 1.17 (before maybe too?), numpy.clip() turns int32
            # to float64. float64 should cover the whole value range of int32,
            # so the dtype is not rejected here.
            # TODO Verify this. Is rounding needed before conversion?
            array = np.clip(array, min_value, max_value).astype(array.dtype)
        else:
            array = np.clip(array, min_value, max_value, out=array)
    return array


def clip_to_dtype_value_range_(array, dtype, validate=True,
                               validate_values=None):
    dtype = normalize_dtype(dtype)
    min_value, _, max_value = get_value_range_of_dtype(dtype)
    if validate:
        array_val = array
        if ia.is_single_integer(validate):
            assert validate >= 1, (
                "If 'validate' is an integer, it must have a value >=1, "
                "got %d instead." % (validate,))
            assert validate_values is None, (
                "If 'validate' is an integer, 'validate_values' must be "
                "None. Got type %s instead." % (type(validate_values),))
            array_val = array.flat[0:validate]
        if validate_values is not None:
            min_value_found, max_value_found = validate_values
        else:
            min_value_found = np.min(array_val)
            max_value_found = np.max(array_val)
        assert min_value <= min_value_found <= max_value, (
            "Minimum value of array is outside of allowed value range (%.4f "
            "vs %.4f to %.4f)." % (min_value_found, min_value, max_value))
        assert min_value <= max_value_found <= max_value, (
            "Maximum value of array is outside of allowed value range (%.4f "
            "vs %.4f to %.4f)." % (max_value_found, min_value, max_value))

    return clip_(array, min_value, max_value)


def gate_dtypes(dtypes, allowed, disallowed, augmenter=None):
    # assume that at least one allowed dtype string is given
    assert len(allowed) > 0, (
        "Expected at least one dtype to be allowed, but got an empty list.")
    # check only first dtype for performance
    assert ia.is_string(allowed[0]), (
        "Expected only strings as dtypes, but got type %s." % (
            type(allowed[0]),))

    if len(disallowed) > 0:
        # check only first disallowed dtype for performance
        assert ia.is_string(disallowed[0]), (
            "Expected only strings as dtypes, but got type %s." % (
                type(disallowed[0]),))

    # verify that "allowed" and "disallowed" do not contain overlapping
    # dtypes
    inters = set(allowed).intersection(set(disallowed))
    nb_overlapping = len(inters)
    assert nb_overlapping == 0, (
        "Expected 'allowed' and 'disallowed' to not contain the same dtypes, "
        "but %d appeared in both arguments. Got allowed: %s, "
        "disallowed: %s, intersection: %s" % (
            nb_overlapping,
            ", ".join(allowed),
            ", ".join(disallowed),
            ", ".join(inters))
    )

    dtypes = normalize_dtypes(dtypes)

    for dtype in dtypes:
        if dtype.name in allowed:
            pass
        elif dtype.name in disallowed:
            if augmenter is None:
                raise ValueError(
                    "Got dtype '%s', which is a forbidden dtype (%s)." % (
                        dtype.name, ", ".join(disallowed)
                    ))

            raise ValueError(
                "Got dtype '%s' in augmenter '%s' (class '%s'), which "
                "is a forbidden dtype (%s)." % (
                    dtype.name,
                    augmenter.name,
                    augmenter.__class__.__name__,
                    ", ".join(disallowed)
                ))
        else:
            if augmenter is None:
                ia.warn(
                    "Got dtype '%s', which was neither explicitly allowed "
                    "(%s), nor explicitly disallowed (%s). Generated "
                    "outputs may contain errors." % (
                        dtype.name,
                        ", ".join(allowed),
                        ", ".join(disallowed)
                    ))
            else:
                ia.warn(
                    "Got dtype '%s' in augmenter '%s' (class '%s'), which was "
                    "neither explicitly allowed (%s), nor explicitly "
                    "disallowed (%s). Generated outputs may contain "
                    "errors." % (
                        dtype.name,
                        augmenter.name,
                        augmenter.__class__.__name__,
                        ", ".join(allowed),
                        ", ".join(disallowed)
                    ))
