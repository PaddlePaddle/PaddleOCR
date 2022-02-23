import numpy as np
from skimage.segmentation import join_segmentations, relabel_sequential

from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
import pytest


def test_join_segmentations():
    s1 = np.array([[0, 0, 1, 1],
                   [0, 2, 1, 1],
                   [2, 2, 2, 1]])
    s2 = np.array([[0, 1, 1, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 1]])

    # test correct join
    # NOTE: technically, equality to j_ref is not required, only that there
    # is a one-to-one mapping between j and j_ref. I don't know of an easy way
    # to check this (i.e. not as error-prone as the function being tested)
    j = join_segmentations(s1, s2)
    j_ref = np.array([[0, 1, 3, 2],
                      [0, 5, 3, 2],
                      [4, 5, 5, 3]])
    assert_array_equal(j, j_ref)

    # test correct exception when arrays are different shapes
    s3 = np.array([[0, 0, 1, 1], [0, 2, 2, 1]])
    with testing.raises(ValueError):
        join_segmentations(s1, s3)


def _check_maps(ar, ar_relab, fw, inv):
    assert_array_equal(fw[ar], ar_relab)
    assert_array_equal(inv[ar_relab], ar)


def test_relabel_sequential_offset1():
    ar = np.array([1, 1, 5, 5, 8, 99, 42])
    ar_relab, fw, inv = relabel_sequential(ar)
    _check_maps(ar, ar_relab, fw, inv)
    ar_relab_ref = np.array([1, 1, 2, 2, 3, 5, 4])
    assert_array_equal(ar_relab, ar_relab_ref)
    fw_ref = np.zeros(100, int)
    fw_ref[1] = 1
    fw_ref[5] = 2
    fw_ref[8] = 3
    fw_ref[42] = 4
    fw_ref[99] = 5
    assert_array_equal(fw, fw_ref)
    inv_ref = np.array([0,  1,  5,  8, 42, 99])
    assert_array_equal(inv, inv_ref)


def test_relabel_sequential_offset5():
    ar = np.array([1, 1, 5, 5, 8, 99, 42])
    ar_relab, fw, inv = relabel_sequential(ar, offset=5)
    _check_maps(ar, ar_relab, fw, inv)
    ar_relab_ref = np.array([5, 5, 6, 6, 7, 9, 8])
    assert_array_equal(ar_relab, ar_relab_ref)
    fw_ref = np.zeros(100, int)
    fw_ref[1] = 5
    fw_ref[5] = 6
    fw_ref[8] = 7
    fw_ref[42] = 8
    fw_ref[99] = 9
    assert_array_equal(fw, fw_ref)
    inv_ref = np.array([0, 0, 0, 0, 0, 1,  5,  8, 42, 99])
    assert_array_equal(inv, inv_ref)


def test_relabel_sequential_offset5_with0():
    ar = np.array([1, 1, 5, 5, 8, 99, 42, 0])
    ar_relab, fw, inv = relabel_sequential(ar, offset=5)
    _check_maps(ar, ar_relab, fw, inv)
    ar_relab_ref = np.array([5, 5, 6, 6, 7, 9, 8, 0])
    assert_array_equal(ar_relab, ar_relab_ref)
    fw_ref = np.zeros(100, int)
    fw_ref[1] = 5
    fw_ref[5] = 6
    fw_ref[8] = 7
    fw_ref[42] = 8
    fw_ref[99] = 9
    assert_array_equal(fw, fw_ref)
    inv_ref = np.array([0, 0, 0, 0, 0, 1,  5,  8, 42, 99])
    assert_array_equal(inv, inv_ref)


def test_relabel_sequential_dtype():
    ar = np.array([1, 1, 5, 5, 8, 99, 42, 0], dtype=np.uint8)
    ar_relab, fw, inv = relabel_sequential(ar, offset=5)
    _check_maps(ar.astype(int), ar_relab, fw, inv)
    ar_relab_ref = np.array([5, 5, 6, 6, 7, 9, 8, 0])
    assert_array_equal(ar_relab, ar_relab_ref)
    fw_ref = np.zeros(100, int)
    fw_ref[1] = 5
    fw_ref[5] = 6
    fw_ref[8] = 7
    fw_ref[42] = 8
    fw_ref[99] = 9
    assert_array_equal(fw, fw_ref)
    inv_ref = np.array([0, 0, 0, 0, 0, 1,  5,  8, 42, 99])
    assert_array_equal(inv, inv_ref)


def test_relabel_sequential_signed_overflow():
    imax = np.iinfo(np.int32).max
    labels = np.array([0, 1, 99, 42, 42], dtype=np.int32)
    output, fw, inv = relabel_sequential(labels, offset=imax)
    reference = np.array([0, imax, imax + 2, imax + 1, imax + 1],
                         dtype=np.uint32)
    assert_array_equal(output, reference)
    assert output.dtype == reference.dtype


def test_very_large_labels():
    imax = np.iinfo(np.int64).max
    labels = np.array([0, 1, imax, 42, 42], dtype=np.int64)
    output, fw, inv = relabel_sequential(labels, offset=imax)
    assert np.max(output) == imax + 2


@pytest.mark.parametrize('dtype', (np.byte, np.short, np.intc, int,
                                   np.longlong, np.ubyte, np.ushort,
                                   np.uintc, np.uint, np.ulonglong))
@pytest.mark.parametrize('data_already_sequential', (False, True))
def test_relabel_sequential_int_dtype_stability(data_already_sequential,
                                                dtype):
    if data_already_sequential:
        ar = np.array([1, 3, 0, 2, 5, 4], dtype=dtype)
    else:
        ar = np.array([1, 1, 5, 5, 8, 99, 42, 0], dtype=dtype)
    assert all(a.dtype == dtype for a in relabel_sequential(ar))


def test_relabel_sequential_int_dtype_overflow():
    ar = np.array([1, 3, 0, 2, 5, 4], dtype=np.uint8)
    offset = 254
    ar_relab, fw, inv = relabel_sequential(ar, offset=offset)
    _check_maps(ar, ar_relab, fw, inv)
    assert all(a.dtype == np.uint16 for a in (ar_relab, fw))
    assert inv.dtype == ar.dtype
    ar_relab_ref = np.where(ar > 0, ar.astype(int) + offset - 1, 0)
    assert_array_equal(ar_relab, ar_relab_ref)


def test_relabel_sequential_negative_values():
    ar = np.array([1, 1, 5, -5, 8, 99, 42, 0])
    with pytest.raises(ValueError):
        relabel_sequential(ar)


@pytest.mark.parametrize('offset', (0, -3))
@pytest.mark.parametrize('data_already_sequential', (False, True))
def test_relabel_sequential_nonpositive_offset(data_already_sequential,
                                               offset):
    if data_already_sequential:
        ar = np.array([1, 3, 0, 2, 5, 4])
    else:
        ar = np.array([1, 1, 5, 5, 8, 99, 42, 0])
    with pytest.raises(ValueError):
        relabel_sequential(ar, offset=offset)


@pytest.mark.parametrize('offset', (1, 5))
@pytest.mark.parametrize('with0', (False, True))
@pytest.mark.parametrize('input_starts_at_offset', (False, True))
def test_relabel_sequential_already_sequential(offset, with0,
                                               input_starts_at_offset):
    if with0:
        ar = np.array([1, 3, 0, 2, 5, 4])
    else:
        ar = np.array([1, 3, 2, 5, 4])
    if input_starts_at_offset:
        ar[ar > 0] += offset - 1
    ar_relab, fw, inv = relabel_sequential(ar, offset=offset)
    _check_maps(ar, ar_relab, fw, inv)
    if input_starts_at_offset:
        ar_relab_ref = ar
    else:
        ar_relab_ref = np.where(ar > 0, ar + offset - 1, 0)
    assert_array_equal(ar_relab, ar_relab_ref)


def test_incorrect_input_dtype():
    labels = np.array([0, 2, 2, 1, 1, 8], dtype=float)
    with testing.raises(TypeError):
        _ = relabel_sequential(labels)


def test_arraymap_call():
    ar = np.array([1, 1, 5, 5, 8, 99, 42, 0], dtype=np.intp)
    relabeled, fw, inv = relabel_sequential(ar)
    testing.assert_array_equal(relabeled, fw(ar))
    testing.assert_array_equal(ar, inv(relabeled))


def test_arraymap_len():
    ar = np.array([1, 1, 5, 5, 8, 99, 42, 0], dtype=np.intp)
    relabeled, fw, inv = relabel_sequential(ar)
    assert len(fw) == 100
    assert len(fw) == len(np.array(fw))
    assert len(inv) == 6
    assert len(inv) == len(np.array(inv))


def test_arraymap_set():
    ar = np.array([1, 1, 5, 5, 8, 99, 42, 0], dtype=np.intp)
    relabeled, fw, inv = relabel_sequential(ar)
    fw[72] = 6
    assert fw[72] == 6
