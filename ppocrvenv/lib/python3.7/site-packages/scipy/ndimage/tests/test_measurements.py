import os.path

import numpy as np
from numpy.testing import (assert_, assert_array_almost_equal, assert_equal,
                           assert_almost_equal, assert_array_equal,
                           suppress_warnings)
from pytest import raises as assert_raises

import scipy.ndimage as ndimage


from . import types


class Test_measurements_stats:
    """ndimage.measurements._stats() is a utility used by other functions."""

    def test_a(self):
        x = [0, 1, 2, 6]
        labels = [0, 0, 1, 1]
        index = [0, 1]
        for shp in [(4,), (2, 2)]:
            x = np.array(x).reshape(shp)
            labels = np.array(labels).reshape(shp)
            counts, sums = ndimage.measurements._stats(
                x, labels=labels, index=index)
            assert_array_equal(counts, [2, 2])
            assert_array_equal(sums, [1.0, 8.0])

    def test_b(self):
        # Same data as test_a, but different labels.  The label 9 exceeds the
        # length of 'labels', so this test will follow a different code path.
        x = [0, 1, 2, 6]
        labels = [0, 0, 9, 9]
        index = [0, 9]
        for shp in [(4,), (2, 2)]:
            x = np.array(x).reshape(shp)
            labels = np.array(labels).reshape(shp)
            counts, sums = ndimage.measurements._stats(
                x, labels=labels, index=index)
            assert_array_equal(counts, [2, 2])
            assert_array_equal(sums, [1.0, 8.0])

    def test_a_centered(self):
        x = [0, 1, 2, 6]
        labels = [0, 0, 1, 1]
        index = [0, 1]
        for shp in [(4,), (2, 2)]:
            x = np.array(x).reshape(shp)
            labels = np.array(labels).reshape(shp)
            counts, sums, centers = ndimage.measurements._stats(
                x, labels=labels, index=index, centered=True)
            assert_array_equal(counts, [2, 2])
            assert_array_equal(sums, [1.0, 8.0])
            assert_array_equal(centers, [0.5, 8.0])

    def test_b_centered(self):
        x = [0, 1, 2, 6]
        labels = [0, 0, 9, 9]
        index = [0, 9]
        for shp in [(4,), (2, 2)]:
            x = np.array(x).reshape(shp)
            labels = np.array(labels).reshape(shp)
            counts, sums, centers = ndimage.measurements._stats(
                x, labels=labels, index=index, centered=True)
            assert_array_equal(counts, [2, 2])
            assert_array_equal(sums, [1.0, 8.0])
            assert_array_equal(centers, [0.5, 8.0])

    def test_nonint_labels(self):
        x = [0, 1, 2, 6]
        labels = [0.0, 0.0, 9.0, 9.0]
        index = [0.0, 9.0]
        for shp in [(4,), (2, 2)]:
            x = np.array(x).reshape(shp)
            labels = np.array(labels).reshape(shp)
            counts, sums, centers = ndimage.measurements._stats(
                x, labels=labels, index=index, centered=True)
            assert_array_equal(counts, [2, 2])
            assert_array_equal(sums, [1.0, 8.0])
            assert_array_equal(centers, [0.5, 8.0])


class Test_measurements_select:
    """ndimage.measurements._select() is a utility used by other functions."""

    def test_basic(self):
        x = [0, 1, 6, 2]
        cases = [
            ([0, 0, 1, 1], [0, 1]),           # "Small" integer labels
            ([0, 0, 9, 9], [0, 9]),           # A label larger than len(labels)
            ([0.0, 0.0, 7.0, 7.0], [0.0, 7.0]),   # Non-integer labels
        ]
        for labels, index in cases:
            result = ndimage.measurements._select(
                x, labels=labels, index=index)
            assert_(len(result) == 0)
            result = ndimage.measurements._select(
                x, labels=labels, index=index, find_max=True)
            assert_(len(result) == 1)
            assert_array_equal(result[0], [1, 6])
            result = ndimage.measurements._select(
                x, labels=labels, index=index, find_min=True)
            assert_(len(result) == 1)
            assert_array_equal(result[0], [0, 2])
            result = ndimage.measurements._select(
                x, labels=labels, index=index, find_min=True,
                find_min_positions=True)
            assert_(len(result) == 2)
            assert_array_equal(result[0], [0, 2])
            assert_array_equal(result[1], [0, 3])
            assert_equal(result[1].dtype.kind, 'i')
            result = ndimage.measurements._select(
                x, labels=labels, index=index, find_max=True,
                find_max_positions=True)
            assert_(len(result) == 2)
            assert_array_equal(result[0], [1, 6])
            assert_array_equal(result[1], [1, 2])
            assert_equal(result[1].dtype.kind, 'i')


def test_label01():
    data = np.ones([])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, 1)
    assert_equal(n, 1)


def test_label02():
    data = np.zeros([])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, 0)
    assert_equal(n, 0)


def test_label03():
    data = np.ones([1])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, [1])
    assert_equal(n, 1)


def test_label04():
    data = np.zeros([1])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, [0])
    assert_equal(n, 0)


def test_label05():
    data = np.ones([5])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, [1, 1, 1, 1, 1])
    assert_equal(n, 1)


def test_label06():
    data = np.array([1, 0, 1, 1, 0, 1])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, [1, 0, 2, 2, 0, 3])
    assert_equal(n, 3)


def test_label07():
    data = np.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, [[0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0]])
    assert_equal(n, 0)


def test_label08():
    data = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [1, 1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0]])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, [[1, 0, 0, 0, 0, 0],
                                    [0, 0, 2, 2, 0, 0],
                                    [0, 0, 2, 2, 2, 0],
                                    [3, 3, 0, 0, 0, 0],
                                    [3, 3, 0, 0, 0, 0],
                                    [0, 0, 0, 4, 4, 0]])
    assert_equal(n, 4)


def test_label09():
    data = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [1, 1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0]])
    struct = ndimage.generate_binary_structure(2, 2)
    out, n = ndimage.label(data, struct)
    assert_array_almost_equal(out, [[1, 0, 0, 0, 0, 0],
                                    [0, 0, 2, 2, 0, 0],
                                    [0, 0, 2, 2, 2, 0],
                                    [2, 2, 0, 0, 0, 0],
                                    [2, 2, 0, 0, 0, 0],
                                    [0, 0, 0, 3, 3, 0]])
    assert_equal(n, 3)


def test_label10():
    data = np.array([[0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 1, 0],
                     [0, 1, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0, 0]])
    struct = ndimage.generate_binary_structure(2, 2)
    out, n = ndimage.label(data, struct)
    assert_array_almost_equal(out, [[0, 0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 1, 0],
                                    [0, 1, 1, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0]])
    assert_equal(n, 1)


def test_label11():
    for type in types:
        data = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 0],
                         [1, 1, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 0]], type)
        out, n = ndimage.label(data)
        expected = [[1, 0, 0, 0, 0, 0],
                    [0, 0, 2, 2, 0, 0],
                    [0, 0, 2, 2, 2, 0],
                    [3, 3, 0, 0, 0, 0],
                    [3, 3, 0, 0, 0, 0],
                    [0, 0, 0, 4, 4, 0]]
        assert_array_almost_equal(out, expected)
        assert_equal(n, 4)


def test_label11_inplace():
    for type in types:
        data = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 0],
                         [1, 1, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 0]], type)
        n = ndimage.label(data, output=data)
        expected = [[1, 0, 0, 0, 0, 0],
                    [0, 0, 2, 2, 0, 0],
                    [0, 0, 2, 2, 2, 0],
                    [3, 3, 0, 0, 0, 0],
                    [3, 3, 0, 0, 0, 0],
                    [0, 0, 0, 4, 4, 0]]
        assert_array_almost_equal(data, expected)
        assert_equal(n, 4)


def test_label12():
    for type in types:
        data = np.array([[0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 1],
                         [0, 0, 1, 0, 1, 1],
                         [0, 0, 1, 1, 1, 1],
                         [0, 0, 0, 1, 1, 0]], type)
        out, n = ndimage.label(data)
        expected = [[0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 1, 0, 1, 1],
                    [0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 0]]
        assert_array_almost_equal(out, expected)
        assert_equal(n, 1)


def test_label13():
    for type in types:
        data = np.array([[1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                         [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                        type)
        out, n = ndimage.label(data)
        expected = [[1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                    [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        assert_array_almost_equal(out, expected)
        assert_equal(n, 1)


def test_label_output_typed():
    data = np.ones([5])
    for t in types:
        output = np.zeros([5], dtype=t)
        n = ndimage.label(data, output=output)
        assert_array_almost_equal(output, 1)
        assert_equal(n, 1)


def test_label_output_dtype():
    data = np.ones([5])
    for t in types:
        output, n = ndimage.label(data, output=t)
        assert_array_almost_equal(output, 1)
        assert output.dtype == t


def test_label_output_wrong_size():
    data = np.ones([5])
    for t in types:
        output = np.zeros([10], t)
        assert_raises((RuntimeError, ValueError),
                      ndimage.label, data, output=output)


def test_label_structuring_elements():
    data = np.loadtxt(os.path.join(os.path.dirname(
        __file__), "data", "label_inputs.txt"))
    strels = np.loadtxt(os.path.join(
        os.path.dirname(__file__), "data", "label_strels.txt"))
    results = np.loadtxt(os.path.join(
        os.path.dirname(__file__), "data", "label_results.txt"))
    data = data.reshape((-1, 7, 7))
    strels = strels.reshape((-1, 3, 3))
    results = results.reshape((-1, 7, 7))
    r = 0
    for i in range(data.shape[0]):
        d = data[i, :, :]
        for j in range(strels.shape[0]):
            s = strels[j, :, :]
            assert_equal(ndimage.label(d, s)[0], results[r, :, :])
            r += 1


def test_ticket_742():
    def SE(img, thresh=.7, size=4):
        mask = img > thresh
        rank = len(mask.shape)
        la, co = ndimage.label(mask,
                               ndimage.generate_binary_structure(rank, rank))
        _ = ndimage.find_objects(la)

    if np.dtype(np.intp) != np.dtype('i'):
        shape = (3, 1240, 1240)
        a = np.random.rand(np.prod(shape)).reshape(shape)
        # shouldn't crash
        SE(a)


def test_gh_issue_3025():
    """Github issue #3025 - improper merging of labels"""
    d = np.zeros((60, 320))
    d[:, :257] = 1
    d[:, 260:] = 1
    d[36, 257] = 1
    d[35, 258] = 1
    d[35, 259] = 1
    assert ndimage.label(d, np.ones((3, 3)))[1] == 1


def test_label_default_dtype():
    test_array = np.random.rand(10, 10)
    label, no_features = ndimage.label(test_array > 0.5)
    assert_(label.dtype in (np.int32, np.int64))
    # Shouldn't raise an exception
    ndimage.find_objects(label)


def test_find_objects01():
    data = np.ones([], dtype=int)
    out = ndimage.find_objects(data)
    assert_(out == [()])


def test_find_objects02():
    data = np.zeros([], dtype=int)
    out = ndimage.find_objects(data)
    assert_(out == [])


def test_find_objects03():
    data = np.ones([1], dtype=int)
    out = ndimage.find_objects(data)
    assert_equal(out, [(slice(0, 1, None),)])


def test_find_objects04():
    data = np.zeros([1], dtype=int)
    out = ndimage.find_objects(data)
    assert_equal(out, [])


def test_find_objects05():
    data = np.ones([5], dtype=int)
    out = ndimage.find_objects(data)
    assert_equal(out, [(slice(0, 5, None),)])


def test_find_objects06():
    data = np.array([1, 0, 2, 2, 0, 3])
    out = ndimage.find_objects(data)
    assert_equal(out, [(slice(0, 1, None),),
                       (slice(2, 4, None),),
                       (slice(5, 6, None),)])


def test_find_objects07():
    data = np.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]])
    out = ndimage.find_objects(data)
    assert_equal(out, [])


def test_find_objects08():
    data = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 2, 2, 0, 0],
                     [0, 0, 2, 2, 2, 0],
                     [3, 3, 0, 0, 0, 0],
                     [3, 3, 0, 0, 0, 0],
                     [0, 0, 0, 4, 4, 0]])
    out = ndimage.find_objects(data)
    assert_equal(out, [(slice(0, 1, None), slice(0, 1, None)),
                       (slice(1, 3, None), slice(2, 5, None)),
                       (slice(3, 5, None), slice(0, 2, None)),
                       (slice(5, 6, None), slice(3, 5, None))])


def test_find_objects09():
    data = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 2, 2, 0, 0],
                     [0, 0, 2, 2, 2, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 4, 4, 0]])
    out = ndimage.find_objects(data)
    assert_equal(out, [(slice(0, 1, None), slice(0, 1, None)),
                       (slice(1, 3, None), slice(2, 5, None)),
                       None,
                       (slice(5, 6, None), slice(3, 5, None))])


def test_sum01():
    for type in types:
        input = np.array([], type)
        output = ndimage.sum(input)
        assert_equal(output, 0.0)


def test_sum02():
    for type in types:
        input = np.zeros([0, 4], type)
        output = ndimage.sum(input)
        assert_equal(output, 0.0)


def test_sum03():
    for type in types:
        input = np.ones([], type)
        output = ndimage.sum(input)
        assert_almost_equal(output, 1.0)


def test_sum04():
    for type in types:
        input = np.array([1, 2], type)
        output = ndimage.sum(input)
        assert_almost_equal(output, 3.0)


def test_sum05():
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.sum(input)
        assert_almost_equal(output, 10.0)


def test_sum06():
    labels = np.array([], bool)
    for type in types:
        input = np.array([], type)
        output = ndimage.sum(input, labels=labels)
        assert_equal(output, 0.0)


def test_sum07():
    labels = np.ones([0, 4], bool)
    for type in types:
        input = np.zeros([0, 4], type)
        output = ndimage.sum(input, labels=labels)
        assert_equal(output, 0.0)


def test_sum08():
    labels = np.array([1, 0], bool)
    for type in types:
        input = np.array([1, 2], type)
        output = ndimage.sum(input, labels=labels)
        assert_equal(output, 1.0)


def test_sum09():
    labels = np.array([1, 0], bool)
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.sum(input, labels=labels)
        assert_almost_equal(output, 4.0)


def test_sum10():
    labels = np.array([1, 0], bool)
    input = np.array([[1, 2], [3, 4]], bool)
    output = ndimage.sum(input, labels=labels)
    assert_almost_equal(output, 2.0)


def test_sum11():
    labels = np.array([1, 2], np.int8)
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.sum(input, labels=labels,
                             index=2)
        assert_almost_equal(output, 6.0)


def test_sum12():
    labels = np.array([[1, 2], [2, 4]], np.int8)
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.sum(input, labels=labels, index=[4, 8, 2])
        assert_array_almost_equal(output, [4.0, 0.0, 5.0])


def test_sum_labels():
    labels = np.array([[1, 2], [2, 4]], np.int8)
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output_sum = ndimage.sum(input, labels=labels, index=[4, 8, 2])
        output_labels = ndimage.sum_labels(
            input, labels=labels, index=[4, 8, 2])

        assert (output_sum == output_labels).all()
        assert_array_almost_equal(output_labels, [4.0, 0.0, 5.0])


def test_mean01():
    labels = np.array([1, 0], bool)
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.mean(input, labels=labels)
        assert_almost_equal(output, 2.0)


def test_mean02():
    labels = np.array([1, 0], bool)
    input = np.array([[1, 2], [3, 4]], bool)
    output = ndimage.mean(input, labels=labels)
    assert_almost_equal(output, 1.0)


def test_mean03():
    labels = np.array([1, 2])
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.mean(input, labels=labels,
                              index=2)
        assert_almost_equal(output, 3.0)


def test_mean04():
    labels = np.array([[1, 2], [2, 4]], np.int8)
    with np.errstate(all='ignore'):
        for type in types:
            input = np.array([[1, 2], [3, 4]], type)
            output = ndimage.mean(input, labels=labels,
                                  index=[4, 8, 2])
            assert_array_almost_equal(output[[0, 2]], [4.0, 2.5])
            assert_(np.isnan(output[1]))


def test_minimum01():
    labels = np.array([1, 0], bool)
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.minimum(input, labels=labels)
        assert_almost_equal(output, 1.0)


def test_minimum02():
    labels = np.array([1, 0], bool)
    input = np.array([[2, 2], [2, 4]], bool)
    output = ndimage.minimum(input, labels=labels)
    assert_almost_equal(output, 1.0)


def test_minimum03():
    labels = np.array([1, 2])
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.minimum(input, labels=labels,
                                 index=2)
        assert_almost_equal(output, 2.0)


def test_minimum04():
    labels = np.array([[1, 2], [2, 3]])
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.minimum(input, labels=labels,
                                 index=[2, 3, 8])
        assert_array_almost_equal(output, [2.0, 4.0, 0.0])


def test_maximum01():
    labels = np.array([1, 0], bool)
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.maximum(input, labels=labels)
        assert_almost_equal(output, 3.0)


def test_maximum02():
    labels = np.array([1, 0], bool)
    input = np.array([[2, 2], [2, 4]], bool)
    output = ndimage.maximum(input, labels=labels)
    assert_almost_equal(output, 1.0)


def test_maximum03():
    labels = np.array([1, 2])
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.maximum(input, labels=labels,
                                 index=2)
        assert_almost_equal(output, 4.0)


def test_maximum04():
    labels = np.array([[1, 2], [2, 3]])
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.maximum(input, labels=labels,
                                 index=[2, 3, 8])
        assert_array_almost_equal(output, [3.0, 4.0, 0.0])


def test_maximum05():
    # Regression test for ticket #501 (Trac)
    x = np.array([-3, -2, -1])
    assert_equal(ndimage.maximum(x), -1)


def test_median01():
    a = np.array([[1, 2, 0, 1],
                  [5, 3, 0, 4],
                  [0, 0, 0, 7],
                  [9, 3, 0, 0]])
    labels = np.array([[1, 1, 0, 2],
                       [1, 1, 0, 2],
                       [0, 0, 0, 2],
                       [3, 3, 0, 0]])
    output = ndimage.median(a, labels=labels, index=[1, 2, 3])
    assert_array_almost_equal(output, [2.5, 4.0, 6.0])


def test_median02():
    a = np.array([[1, 2, 0, 1],
                  [5, 3, 0, 4],
                  [0, 0, 0, 7],
                  [9, 3, 0, 0]])
    output = ndimage.median(a)
    assert_almost_equal(output, 1.0)


def test_median03():
    a = np.array([[1, 2, 0, 1],
                  [5, 3, 0, 4],
                  [0, 0, 0, 7],
                  [9, 3, 0, 0]])
    labels = np.array([[1, 1, 0, 2],
                       [1, 1, 0, 2],
                       [0, 0, 0, 2],
                       [3, 3, 0, 0]])
    output = ndimage.median(a, labels=labels)
    assert_almost_equal(output, 3.0)


def test_median_gh12836_bool():
    # test boolean addition fix on example from gh-12836
    a = np.asarray([1, 1], dtype=bool)
    output = ndimage.median(a, labels=np.ones((2,)), index=[1])
    assert_array_almost_equal(output, [1.0])


def test_median_no_int_overflow():
    # test integer overflow fix on example from gh-12836
    a = np.asarray([65, 70], dtype=np.int8)
    output = ndimage.median(a, labels=np.ones((2,)), index=[1])
    assert_array_almost_equal(output, [67.5])


def test_variance01():
    with np.errstate(all='ignore'):
        for type in types:
            input = np.array([], type)
            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, "Mean of empty slice")
                output = ndimage.variance(input)
            assert_(np.isnan(output))


def test_variance02():
    for type in types:
        input = np.array([1], type)
        output = ndimage.variance(input)
        assert_almost_equal(output, 0.0)


def test_variance03():
    for type in types:
        input = np.array([1, 3], type)
        output = ndimage.variance(input)
        assert_almost_equal(output, 1.0)


def test_variance04():
    input = np.array([1, 0], bool)
    output = ndimage.variance(input)
    assert_almost_equal(output, 0.25)


def test_variance05():
    labels = [2, 2, 3]
    for type in types:
        input = np.array([1, 3, 8], type)
        output = ndimage.variance(input, labels, 2)
        assert_almost_equal(output, 1.0)


def test_variance06():
    labels = [2, 2, 3, 3, 4]
    with np.errstate(all='ignore'):
        for type in types:
            input = np.array([1, 3, 8, 10, 8], type)
            output = ndimage.variance(input, labels, [2, 3, 4])
            assert_array_almost_equal(output, [1.0, 1.0, 0.0])


def test_standard_deviation01():
    with np.errstate(all='ignore'):
        for type in types:
            input = np.array([], type)
            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, "Mean of empty slice")
                output = ndimage.standard_deviation(input)
            assert_(np.isnan(output))


def test_standard_deviation02():
    for type in types:
        input = np.array([1], type)
        output = ndimage.standard_deviation(input)
        assert_almost_equal(output, 0.0)


def test_standard_deviation03():
    for type in types:
        input = np.array([1, 3], type)
        output = ndimage.standard_deviation(input)
        assert_almost_equal(output, np.sqrt(1.0))


def test_standard_deviation04():
    input = np.array([1, 0], bool)
    output = ndimage.standard_deviation(input)
    assert_almost_equal(output, 0.5)


def test_standard_deviation05():
    labels = [2, 2, 3]
    for type in types:
        input = np.array([1, 3, 8], type)
        output = ndimage.standard_deviation(input, labels, 2)
        assert_almost_equal(output, 1.0)


def test_standard_deviation06():
    labels = [2, 2, 3, 3, 4]
    with np.errstate(all='ignore'):
        for type in types:
            input = np.array([1, 3, 8, 10, 8], type)
            output = ndimage.standard_deviation(input, labels, [2, 3, 4])
            assert_array_almost_equal(output, [1.0, 1.0, 0.0])


def test_standard_deviation07():
    labels = [1]
    with np.errstate(all='ignore'):
        for type in types:
            input = np.array([-0.00619519], type)
            output = ndimage.standard_deviation(input, labels, [1])
            assert_array_almost_equal(output, [0])


def test_minimum_position01():
    labels = np.array([1, 0], bool)
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.minimum_position(input, labels=labels)
        assert_equal(output, (0, 0))


def test_minimum_position02():
    for type in types:
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 0, 2],
                          [1, 5, 1, 1]], type)
        output = ndimage.minimum_position(input)
        assert_equal(output, (1, 2))


def test_minimum_position03():
    input = np.array([[5, 4, 2, 5],
                      [3, 7, 0, 2],
                      [1, 5, 1, 1]], bool)
    output = ndimage.minimum_position(input)
    assert_equal(output, (1, 2))


def test_minimum_position04():
    input = np.array([[5, 4, 2, 5],
                      [3, 7, 1, 2],
                      [1, 5, 1, 1]], bool)
    output = ndimage.minimum_position(input)
    assert_equal(output, (0, 0))


def test_minimum_position05():
    labels = [1, 2, 0, 4]
    for type in types:
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 0, 2],
                          [1, 5, 2, 3]], type)
        output = ndimage.minimum_position(input, labels)
        assert_equal(output, (2, 0))


def test_minimum_position06():
    labels = [1, 2, 3, 4]
    for type in types:
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 0, 2],
                          [1, 5, 1, 1]], type)
        output = ndimage.minimum_position(input, labels, 2)
        assert_equal(output, (0, 1))


def test_minimum_position07():
    labels = [1, 2, 3, 4]
    for type in types:
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 0, 2],
                          [1, 5, 1, 1]], type)
        output = ndimage.minimum_position(input, labels,
                                          [2, 3])
        assert_equal(output[0], (0, 1))
        assert_equal(output[1], (1, 2))


def test_maximum_position01():
    labels = np.array([1, 0], bool)
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.maximum_position(input,
                                          labels=labels)
        assert_equal(output, (1, 0))


def test_maximum_position02():
    for type in types:
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 8, 2],
                          [1, 5, 1, 1]], type)
        output = ndimage.maximum_position(input)
        assert_equal(output, (1, 2))


def test_maximum_position03():
    input = np.array([[5, 4, 2, 5],
                      [3, 7, 8, 2],
                      [1, 5, 1, 1]], bool)
    output = ndimage.maximum_position(input)
    assert_equal(output, (0, 0))


def test_maximum_position04():
    labels = [1, 2, 0, 4]
    for type in types:
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 8, 2],
                          [1, 5, 1, 1]], type)
        output = ndimage.maximum_position(input, labels)
        assert_equal(output, (1, 1))


def test_maximum_position05():
    labels = [1, 2, 0, 4]
    for type in types:
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 8, 2],
                          [1, 5, 1, 1]], type)
        output = ndimage.maximum_position(input, labels, 1)
        assert_equal(output, (0, 0))


def test_maximum_position06():
    labels = [1, 2, 0, 4]
    for type in types:
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 8, 2],
                          [1, 5, 1, 1]], type)
        output = ndimage.maximum_position(input, labels,
                                          [1, 2])
        assert_equal(output[0], (0, 0))
        assert_equal(output[1], (1, 1))


def test_maximum_position07():
    # Test float labels
    labels = np.array([1.0, 2.5, 0.0, 4.5])
    for type in types:
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 8, 2],
                          [1, 5, 1, 1]], type)
        output = ndimage.maximum_position(input, labels,
                                          [1.0, 4.5])
        assert_equal(output[0], (0, 0))
        assert_equal(output[1], (0, 3))


def test_extrema01():
    labels = np.array([1, 0], bool)
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output1 = ndimage.extrema(input, labels=labels)
        output2 = ndimage.minimum(input, labels=labels)
        output3 = ndimage.maximum(input, labels=labels)
        output4 = ndimage.minimum_position(input,
                                           labels=labels)
        output5 = ndimage.maximum_position(input,
                                           labels=labels)
        assert_equal(output1, (output2, output3, output4, output5))


def test_extrema02():
    labels = np.array([1, 2])
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output1 = ndimage.extrema(input, labels=labels,
                                  index=2)
        output2 = ndimage.minimum(input, labels=labels,
                                  index=2)
        output3 = ndimage.maximum(input, labels=labels,
                                  index=2)
        output4 = ndimage.minimum_position(input,
                                           labels=labels, index=2)
        output5 = ndimage.maximum_position(input,
                                           labels=labels, index=2)
        assert_equal(output1, (output2, output3, output4, output5))


def test_extrema03():
    labels = np.array([[1, 2], [2, 3]])
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output1 = ndimage.extrema(input, labels=labels,
                                  index=[2, 3, 8])
        output2 = ndimage.minimum(input, labels=labels,
                                  index=[2, 3, 8])
        output3 = ndimage.maximum(input, labels=labels,
                                  index=[2, 3, 8])
        output4 = ndimage.minimum_position(input,
                                           labels=labels, index=[2, 3, 8])
        output5 = ndimage.maximum_position(input,
                                           labels=labels, index=[2, 3, 8])
        assert_array_almost_equal(output1[0], output2)
        assert_array_almost_equal(output1[1], output3)
        assert_array_almost_equal(output1[2], output4)
        assert_array_almost_equal(output1[3], output5)


def test_extrema04():
    labels = [1, 2, 0, 4]
    for type in types:
        input = np.array([[5, 4, 2, 5],
                          [3, 7, 8, 2],
                          [1, 5, 1, 1]], type)
        output1 = ndimage.extrema(input, labels, [1, 2])
        output2 = ndimage.minimum(input, labels, [1, 2])
        output3 = ndimage.maximum(input, labels, [1, 2])
        output4 = ndimage.minimum_position(input, labels,
                                           [1, 2])
        output5 = ndimage.maximum_position(input, labels,
                                           [1, 2])
        assert_array_almost_equal(output1[0], output2)
        assert_array_almost_equal(output1[1], output3)
        assert_array_almost_equal(output1[2], output4)
        assert_array_almost_equal(output1[3], output5)


def test_center_of_mass01():
    expected = [0.0, 0.0]
    for type in types:
        input = np.array([[1, 0], [0, 0]], type)
        output = ndimage.center_of_mass(input)
        assert_array_almost_equal(output, expected)


def test_center_of_mass02():
    expected = [1, 0]
    for type in types:
        input = np.array([[0, 0], [1, 0]], type)
        output = ndimage.center_of_mass(input)
        assert_array_almost_equal(output, expected)


def test_center_of_mass03():
    expected = [0, 1]
    for type in types:
        input = np.array([[0, 1], [0, 0]], type)
        output = ndimage.center_of_mass(input)
        assert_array_almost_equal(output, expected)


def test_center_of_mass04():
    expected = [1, 1]
    for type in types:
        input = np.array([[0, 0], [0, 1]], type)
        output = ndimage.center_of_mass(input)
        assert_array_almost_equal(output, expected)


def test_center_of_mass05():
    expected = [0.5, 0.5]
    for type in types:
        input = np.array([[1, 1], [1, 1]], type)
        output = ndimage.center_of_mass(input)
        assert_array_almost_equal(output, expected)


def test_center_of_mass06():
    expected = [0.5, 0.5]
    input = np.array([[1, 2], [3, 1]], bool)
    output = ndimage.center_of_mass(input)
    assert_array_almost_equal(output, expected)


def test_center_of_mass07():
    labels = [1, 0]
    expected = [0.5, 0.0]
    input = np.array([[1, 2], [3, 1]], bool)
    output = ndimage.center_of_mass(input, labels)
    assert_array_almost_equal(output, expected)


def test_center_of_mass08():
    labels = [1, 2]
    expected = [0.5, 1.0]
    input = np.array([[5, 2], [3, 1]], bool)
    output = ndimage.center_of_mass(input, labels, 2)
    assert_array_almost_equal(output, expected)


def test_center_of_mass09():
    labels = [1, 2]
    expected = [(0.5, 0.0), (0.5, 1.0)]
    input = np.array([[1, 2], [1, 1]], bool)
    output = ndimage.center_of_mass(input, labels, [1, 2])
    assert_array_almost_equal(output, expected)


def test_histogram01():
    expected = np.ones(10)
    input = np.arange(10)
    output = ndimage.histogram(input, 0, 10, 10)
    assert_array_almost_equal(output, expected)


def test_histogram02():
    labels = [1, 1, 1, 1, 2, 2, 2, 2]
    expected = [0, 2, 0, 1, 1]
    input = np.array([1, 1, 3, 4, 3, 3, 3, 3])
    output = ndimage.histogram(input, 0, 4, 5, labels, 1)
    assert_array_almost_equal(output, expected)


def test_histogram03():
    labels = [1, 0, 1, 1, 2, 2, 2, 2]
    expected1 = [0, 1, 0, 1, 1]
    expected2 = [0, 0, 0, 3, 0]
    input = np.array([1, 1, 3, 4, 3, 5, 3, 3])
    output = ndimage.histogram(input, 0, 4, 5, labels, (1, 2))

    assert_array_almost_equal(output[0], expected1)
    assert_array_almost_equal(output[1], expected2)


def test_stat_funcs_2d():
    a = np.array([[5, 6, 0, 0, 0], [8, 9, 0, 0, 0], [0, 0, 0, 3, 5]])
    lbl = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 2, 2]])

    mean = ndimage.mean(a, labels=lbl, index=[1, 2])
    assert_array_equal(mean, [7.0, 4.0])

    var = ndimage.variance(a, labels=lbl, index=[1, 2])
    assert_array_equal(var, [2.5, 1.0])

    std = ndimage.standard_deviation(a, labels=lbl, index=[1, 2])
    assert_array_almost_equal(std, np.sqrt([2.5, 1.0]))

    med = ndimage.median(a, labels=lbl, index=[1, 2])
    assert_array_equal(med, [7.0, 4.0])

    min = ndimage.minimum(a, labels=lbl, index=[1, 2])
    assert_array_equal(min, [5, 3])

    max = ndimage.maximum(a, labels=lbl, index=[1, 2])
    assert_array_equal(max, [9, 5])


class TestWatershedIft:

    def test_watershed_ift01(self):
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[-1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], np.int8)
        out = ndimage.watershed_ift(data, markers, structure=[[1, 1, 1],
                                                              [1, 1, 1],
                                                              [1, 1, 1]])
        expected = [[-1, -1, -1, -1, -1, -1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1]]
        assert_array_almost_equal(out, expected)

    def test_watershed_ift02(self):
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[-1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], np.int8)
        out = ndimage.watershed_ift(data, markers)
        expected = [[-1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, 1, 1, 1, -1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, -1, 1, 1, 1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1]]
        assert_array_almost_equal(out, expected)

    def test_watershed_ift03(self):
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 3, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, -1]], np.int8)
        out = ndimage.watershed_ift(data, markers)
        expected = [[-1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, 2, -1, 3, -1, -1],
                    [-1, 2, 2, 3, 3, 3, -1],
                    [-1, 2, 2, 3, 3, 3, -1],
                    [-1, 2, 2, 3, 3, 3, -1],
                    [-1, -1, 2, -1, 3, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1]]
        assert_array_almost_equal(out, expected)

    def test_watershed_ift04(self):
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 3, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, -1]],
                           np.int8)
        out = ndimage.watershed_ift(data, markers,
                                    structure=[[1, 1, 1],
                                               [1, 1, 1],
                                               [1, 1, 1]])
        expected = [[-1, -1, -1, -1, -1, -1, -1],
                    [-1, 2, 2, 3, 3, 3, -1],
                    [-1, 2, 2, 3, 3, 3, -1],
                    [-1, 2, 2, 3, 3, 3, -1],
                    [-1, 2, 2, 3, 3, 3, -1],
                    [-1, 2, 2, 3, 3, 3, -1],
                    [-1, -1, -1, -1, -1, -1, -1]]
        assert_array_almost_equal(out, expected)

    def test_watershed_ift05(self):
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 3, 0, 2, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, -1]],
                           np.int8)
        out = ndimage.watershed_ift(data, markers,
                                    structure=[[1, 1, 1],
                                               [1, 1, 1],
                                               [1, 1, 1]])
        expected = [[-1, -1, -1, -1, -1, -1, -1],
                    [-1, 3, 3, 2, 2, 2, -1],
                    [-1, 3, 3, 2, 2, 2, -1],
                    [-1, 3, 3, 2, 2, 2, -1],
                    [-1, 3, 3, 2, 2, 2, -1],
                    [-1, 3, 3, 2, 2, 2, -1],
                    [-1, -1, -1, -1, -1, -1, -1]]
        assert_array_almost_equal(out, expected)

    def test_watershed_ift06(self):
        data = np.array([[0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[-1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], np.int8)
        out = ndimage.watershed_ift(data, markers,
                                    structure=[[1, 1, 1],
                                               [1, 1, 1],
                                               [1, 1, 1]])
        expected = [[-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1]]
        assert_array_almost_equal(out, expected)

    def test_watershed_ift07(self):
        shape = (7, 6)
        data = np.zeros(shape, dtype=np.uint8)
        data = data.transpose()
        data[...] = np.array([[0, 1, 0, 0, 0, 1, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 1, 1, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[-1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], np.int8)
        out = np.zeros(shape, dtype=np.int16)
        out = out.transpose()
        ndimage.watershed_ift(data, markers,
                              structure=[[1, 1, 1],
                                         [1, 1, 1],
                                         [1, 1, 1]],
                              output=out)
        expected = [[-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, 1, 1, 1, 1, 1, -1],
                    [-1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1]]
        assert_array_almost_equal(out, expected)

    def test_watershed_ift08(self):
        # Test cost larger than uint8. See gh-10069.
        shape = (2, 2)
        data = np.array([[256, 0],
                         [0, 0]], np.uint16)
        markers = np.array([[1, 0],
                            [0, 0]], np.int8)
        out = ndimage.watershed_ift(data, markers)
        expected = [[1, 1],
                    [1, 1]]
        assert_array_almost_equal(out, expected)
