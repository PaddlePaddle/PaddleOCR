import numpy as np

import pytest

from scipy.sparse import issparse
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix

from sklearn.utils.multiclass import type_of_target

from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import ignore_warnings
from sklearn.utils import _to_object_array

from sklearn.preprocessing._label import LabelBinarizer
from sklearn.preprocessing._label import MultiLabelBinarizer
from sklearn.preprocessing._label import LabelEncoder
from sklearn.preprocessing._label import label_binarize

from sklearn.preprocessing._label import _inverse_binarize_thresholding
from sklearn.preprocessing._label import _inverse_binarize_multiclass

from sklearn import datasets

iris = datasets.load_iris()


def toarray(a):
    if hasattr(a, "toarray"):
        a = a.toarray()
    return a


def test_label_binarizer():
    # one-class case defaults to negative label
    # For dense case:
    inp = ["pos", "pos", "pos", "pos"]
    lb = LabelBinarizer(sparse_output=False)
    expected = np.array([[0, 0, 0, 0]]).T
    got = lb.fit_transform(inp)
    assert_array_equal(lb.classes_, ["pos"])
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)

    # For sparse case:
    lb = LabelBinarizer(sparse_output=True)
    got = lb.fit_transform(inp)
    assert issparse(got)
    assert_array_equal(lb.classes_, ["pos"])
    assert_array_equal(expected, got.toarray())
    assert_array_equal(lb.inverse_transform(got.toarray()), inp)

    lb = LabelBinarizer(sparse_output=False)
    # two-class case
    inp = ["neg", "pos", "pos", "neg"]
    expected = np.array([[0, 1, 1, 0]]).T
    got = lb.fit_transform(inp)
    assert_array_equal(lb.classes_, ["neg", "pos"])
    assert_array_equal(expected, got)

    to_invert = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    assert_array_equal(lb.inverse_transform(to_invert), inp)

    # multi-class case
    inp = ["spam", "ham", "eggs", "ham", "0"]
    expected = np.array(
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    )
    got = lb.fit_transform(inp)
    assert_array_equal(lb.classes_, ["0", "eggs", "ham", "spam"])
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)


def test_label_binarizer_unseen_labels():
    lb = LabelBinarizer()

    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    got = lb.fit_transform(["b", "d", "e"])
    assert_array_equal(expected, got)

    expected = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    )
    got = lb.transform(["a", "b", "c", "d", "e", "f"])
    assert_array_equal(expected, got)


def test_label_binarizer_set_label_encoding():
    lb = LabelBinarizer(neg_label=-2, pos_label=0)

    # two-class case with pos_label=0
    inp = np.array([0, 1, 1, 0])
    expected = np.array([[-2, 0, 0, -2]]).T
    got = lb.fit_transform(inp)
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)

    lb = LabelBinarizer(neg_label=-2, pos_label=2)

    # multi-class case
    inp = np.array([3, 2, 1, 2, 0])
    expected = np.array(
        [
            [-2, -2, -2, +2],
            [-2, -2, +2, -2],
            [-2, +2, -2, -2],
            [-2, -2, +2, -2],
            [+2, -2, -2, -2],
        ]
    )
    got = lb.fit_transform(inp)
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)


@ignore_warnings
def test_label_binarizer_errors():
    # Check that invalid arguments yield ValueError
    one_class = np.array([0, 0, 0, 0])
    lb = LabelBinarizer().fit(one_class)

    multi_label = [(2, 3), (0,), (0, 2)]
    with pytest.raises(ValueError):
        lb.transform(multi_label)

    lb = LabelBinarizer()
    with pytest.raises(ValueError):
        lb.transform([])
    with pytest.raises(ValueError):
        lb.inverse_transform([])

    with pytest.raises(ValueError):
        LabelBinarizer(neg_label=2, pos_label=1)
    with pytest.raises(ValueError):
        LabelBinarizer(neg_label=2, pos_label=2)

    with pytest.raises(ValueError):
        LabelBinarizer(neg_label=1, pos_label=2, sparse_output=True)

    # Fail on y_type
    with pytest.raises(ValueError):
        _inverse_binarize_thresholding(
            y=csr_matrix([[1, 2], [2, 1]]),
            output_type="foo",
            classes=[1, 2],
            threshold=0,
        )

    # Sequence of seq type should raise ValueError
    y_seq_of_seqs = [[], [1, 2], [3], [0, 1, 3], [2]]
    with pytest.raises(ValueError):
        LabelBinarizer().fit_transform(y_seq_of_seqs)

    # Fail on the number of classes
    with pytest.raises(ValueError):
        _inverse_binarize_thresholding(
            y=csr_matrix([[1, 2], [2, 1]]),
            output_type="foo",
            classes=[1, 2, 3],
            threshold=0,
        )

    # Fail on the dimension of 'binary'
    with pytest.raises(ValueError):
        _inverse_binarize_thresholding(
            y=np.array([[1, 2, 3], [2, 1, 3]]),
            output_type="binary",
            classes=[1, 2, 3],
            threshold=0,
        )

    # Fail on multioutput data
    with pytest.raises(ValueError):
        LabelBinarizer().fit(np.array([[1, 3], [2, 1]]))
    with pytest.raises(ValueError):
        label_binarize(np.array([[1, 3], [2, 1]]), classes=[1, 2, 3])


@pytest.mark.parametrize(
    "values, classes, unknown",
    [
        (
            np.array([2, 1, 3, 1, 3], dtype="int64"),
            np.array([1, 2, 3], dtype="int64"),
            np.array([4], dtype="int64"),
        ),
        (
            np.array(["b", "a", "c", "a", "c"], dtype=object),
            np.array(["a", "b", "c"], dtype=object),
            np.array(["d"], dtype=object),
        ),
        (
            np.array(["b", "a", "c", "a", "c"]),
            np.array(["a", "b", "c"]),
            np.array(["d"]),
        ),
    ],
    ids=["int64", "object", "str"],
)
def test_label_encoder(values, classes, unknown):
    # Test LabelEncoder's transform, fit_transform and
    # inverse_transform methods
    le = LabelEncoder()
    le.fit(values)
    assert_array_equal(le.classes_, classes)
    assert_array_equal(le.transform(values), [1, 0, 2, 0, 2])
    assert_array_equal(le.inverse_transform([1, 0, 2, 0, 2]), values)
    le = LabelEncoder()
    ret = le.fit_transform(values)
    assert_array_equal(ret, [1, 0, 2, 0, 2])

    with pytest.raises(ValueError, match="unseen labels"):
        le.transform(unknown)


def test_label_encoder_negative_ints():
    le = LabelEncoder()
    le.fit([1, 1, 4, 5, -1, 0])
    assert_array_equal(le.classes_, [-1, 0, 1, 4, 5])
    assert_array_equal(le.transform([0, 1, 4, 4, 5, -1, -1]), [1, 2, 3, 3, 4, 0, 0])
    assert_array_equal(
        le.inverse_transform([1, 2, 3, 3, 4, 0, 0]), [0, 1, 4, 4, 5, -1, -1]
    )
    with pytest.raises(ValueError):
        le.transform([0, 6])


@pytest.mark.parametrize("dtype", ["str", "object"])
def test_label_encoder_str_bad_shape(dtype):
    le = LabelEncoder()
    le.fit(np.array(["apple", "orange"], dtype=dtype))
    msg = "should be a 1d array"
    with pytest.raises(ValueError, match=msg):
        le.transform("apple")


def test_label_encoder_errors():
    # Check that invalid arguments yield ValueError
    le = LabelEncoder()
    with pytest.raises(ValueError):
        le.transform([])
    with pytest.raises(ValueError):
        le.inverse_transform([])

    # Fail on unseen labels
    le = LabelEncoder()
    le.fit([1, 2, 3, -1, 1])
    msg = "contains previously unseen labels"
    with pytest.raises(ValueError, match=msg):
        le.inverse_transform([-2])
    with pytest.raises(ValueError, match=msg):
        le.inverse_transform([-2, -3, -4])

    # Fail on inverse_transform("")
    msg = r"should be a 1d array.+shape \(\)"
    with pytest.raises(ValueError, match=msg):
        le.inverse_transform("")


@pytest.mark.parametrize(
    "values",
    [
        np.array([2, 1, 3, 1, 3], dtype="int64"),
        np.array(["b", "a", "c", "a", "c"], dtype=object),
        np.array(["b", "a", "c", "a", "c"]),
    ],
    ids=["int64", "object", "str"],
)
def test_label_encoder_empty_array(values):
    le = LabelEncoder()
    le.fit(values)
    # test empty transform
    transformed = le.transform([])
    assert_array_equal(np.array([]), transformed)
    # test empty inverse transform
    inverse_transformed = le.inverse_transform([])
    assert_array_equal(np.array([]), inverse_transformed)


def test_sparse_output_multilabel_binarizer():
    # test input as iterable of iterables
    inputs = [
        lambda: [(2, 3), (1,), (1, 2)],
        lambda: ({2, 3}, {1}, {1, 2}),
        lambda: iter([iter((2, 3)), iter((1,)), {1, 2}]),
    ]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])

    inverse = inputs[0]()
    for sparse_output in [True, False]:
        for inp in inputs:
            # With fit_transform
            mlb = MultiLabelBinarizer(sparse_output=sparse_output)
            got = mlb.fit_transform(inp())
            assert issparse(got) == sparse_output
            if sparse_output:
                # verify CSR assumption that indices and indptr have same dtype
                assert got.indices.dtype == got.indptr.dtype
                got = got.toarray()
            assert_array_equal(indicator_mat, got)
            assert_array_equal([1, 2, 3], mlb.classes_)
            assert mlb.inverse_transform(got) == inverse

            # With fit
            mlb = MultiLabelBinarizer(sparse_output=sparse_output)
            got = mlb.fit(inp()).transform(inp())
            assert issparse(got) == sparse_output
            if sparse_output:
                # verify CSR assumption that indices and indptr have same dtype
                assert got.indices.dtype == got.indptr.dtype
                got = got.toarray()
            assert_array_equal(indicator_mat, got)
            assert_array_equal([1, 2, 3], mlb.classes_)
            assert mlb.inverse_transform(got) == inverse

    with pytest.raises(ValueError):
        mlb.inverse_transform(csr_matrix(np.array([[0, 1, 1], [2, 0, 0], [1, 1, 0]])))


def test_multilabel_binarizer():
    # test input as iterable of iterables
    inputs = [
        lambda: [(2, 3), (1,), (1, 2)],
        lambda: ({2, 3}, {1}, {1, 2}),
        lambda: iter([iter((2, 3)), iter((1,)), {1, 2}]),
    ]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])
    inverse = inputs[0]()
    for inp in inputs:
        # With fit_transform
        mlb = MultiLabelBinarizer()
        got = mlb.fit_transform(inp())
        assert_array_equal(indicator_mat, got)
        assert_array_equal([1, 2, 3], mlb.classes_)
        assert mlb.inverse_transform(got) == inverse

        # With fit
        mlb = MultiLabelBinarizer()
        got = mlb.fit(inp()).transform(inp())
        assert_array_equal(indicator_mat, got)
        assert_array_equal([1, 2, 3], mlb.classes_)
        assert mlb.inverse_transform(got) == inverse


def test_multilabel_binarizer_empty_sample():
    mlb = MultiLabelBinarizer()
    y = [[1, 2], [1], []]
    Y = np.array([[1, 1], [1, 0], [0, 0]])
    assert_array_equal(mlb.fit_transform(y), Y)


def test_multilabel_binarizer_unknown_class():
    mlb = MultiLabelBinarizer()
    y = [[1, 2]]
    Y = np.array([[1, 0], [0, 1]])
    warning_message = "unknown class.* will be ignored"
    with pytest.warns(UserWarning, match=warning_message):
        matrix = mlb.fit(y).transform([[4, 1], [2, 0]])

    Y = np.array([[1, 0, 0], [0, 1, 0]])
    mlb = MultiLabelBinarizer(classes=[1, 2, 3])
    with pytest.warns(UserWarning, match=warning_message):
        matrix = mlb.fit(y).transform([[4, 1], [2, 0]])
    assert_array_equal(matrix, Y)


def test_multilabel_binarizer_given_classes():
    inp = [(2, 3), (1,), (1, 2)]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
    # fit_transform()
    mlb = MultiLabelBinarizer(classes=[1, 3, 2])
    assert_array_equal(mlb.fit_transform(inp), indicator_mat)
    assert_array_equal(mlb.classes_, [1, 3, 2])

    # fit().transform()
    mlb = MultiLabelBinarizer(classes=[1, 3, 2])
    assert_array_equal(mlb.fit(inp).transform(inp), indicator_mat)
    assert_array_equal(mlb.classes_, [1, 3, 2])

    # ensure works with extra class
    mlb = MultiLabelBinarizer(classes=[4, 1, 3, 2])
    assert_array_equal(
        mlb.fit_transform(inp), np.hstack(([[0], [0], [0]], indicator_mat))
    )
    assert_array_equal(mlb.classes_, [4, 1, 3, 2])

    # ensure fit is no-op as iterable is not consumed
    inp = iter(inp)
    mlb = MultiLabelBinarizer(classes=[1, 3, 2])
    assert_array_equal(mlb.fit(inp).transform(inp), indicator_mat)

    # ensure a ValueError is thrown if given duplicate classes
    err_msg = (
        "The classes argument contains duplicate classes. Remove "
        "these duplicates before passing them to MultiLabelBinarizer."
    )
    mlb = MultiLabelBinarizer(classes=[1, 3, 2, 3])
    with pytest.raises(ValueError, match=err_msg):
        mlb.fit(inp)


def test_multilabel_binarizer_multiple_calls():
    inp = [(2, 3), (1,), (1, 2)]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])

    indicator_mat2 = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])

    # first call
    mlb = MultiLabelBinarizer(classes=[1, 3, 2])
    assert_array_equal(mlb.fit_transform(inp), indicator_mat)
    # second call change class
    mlb.classes = [1, 2, 3]
    assert_array_equal(mlb.fit_transform(inp), indicator_mat2)


def test_multilabel_binarizer_same_length_sequence():
    # Ensure sequences of the same length are not interpreted as a 2-d array
    inp = [[1], [0], [2]]
    indicator_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    # fit_transform()
    mlb = MultiLabelBinarizer()
    assert_array_equal(mlb.fit_transform(inp), indicator_mat)
    assert_array_equal(mlb.inverse_transform(indicator_mat), inp)

    # fit().transform()
    mlb = MultiLabelBinarizer()
    assert_array_equal(mlb.fit(inp).transform(inp), indicator_mat)
    assert_array_equal(mlb.inverse_transform(indicator_mat), inp)


def test_multilabel_binarizer_non_integer_labels():
    tuple_classes = _to_object_array([(1,), (2,), (3,)])
    inputs = [
        ([("2", "3"), ("1",), ("1", "2")], ["1", "2", "3"]),
        ([("b", "c"), ("a",), ("a", "b")], ["a", "b", "c"]),
        ([((2,), (3,)), ((1,),), ((1,), (2,))], tuple_classes),
    ]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])
    for inp, classes in inputs:
        # fit_transform()
        mlb = MultiLabelBinarizer()
        inp = np.array(inp, dtype=object)
        assert_array_equal(mlb.fit_transform(inp), indicator_mat)
        assert_array_equal(mlb.classes_, classes)
        indicator_mat_inv = np.array(mlb.inverse_transform(indicator_mat), dtype=object)
        assert_array_equal(indicator_mat_inv, inp)

        # fit().transform()
        mlb = MultiLabelBinarizer()
        assert_array_equal(mlb.fit(inp).transform(inp), indicator_mat)
        assert_array_equal(mlb.classes_, classes)
        indicator_mat_inv = np.array(mlb.inverse_transform(indicator_mat), dtype=object)
        assert_array_equal(indicator_mat_inv, inp)

    mlb = MultiLabelBinarizer()
    with pytest.raises(TypeError):
        mlb.fit_transform([({}), ({}, {"a": "b"})])


def test_multilabel_binarizer_non_unique():
    inp = [(1, 1, 1, 0)]
    indicator_mat = np.array([[1, 1]])
    mlb = MultiLabelBinarizer()
    assert_array_equal(mlb.fit_transform(inp), indicator_mat)


def test_multilabel_binarizer_inverse_validation():
    inp = [(1, 1, 1, 0)]
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(inp)
    # Not binary
    with pytest.raises(ValueError):
        mlb.inverse_transform(np.array([[1, 3]]))
    # The following binary cases are fine, however
    mlb.inverse_transform(np.array([[0, 0]]))
    mlb.inverse_transform(np.array([[1, 1]]))
    mlb.inverse_transform(np.array([[1, 0]]))

    # Wrong shape
    with pytest.raises(ValueError):
        mlb.inverse_transform(np.array([[1]]))
    with pytest.raises(ValueError):
        mlb.inverse_transform(np.array([[1, 1, 1]]))


def test_label_binarize_with_class_order():
    out = label_binarize([1, 6], classes=[1, 2, 4, 6])
    expected = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
    assert_array_equal(out, expected)

    # Modified class order
    out = label_binarize([1, 6], classes=[1, 6, 4, 2])
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    assert_array_equal(out, expected)

    out = label_binarize([0, 1, 2, 3], classes=[3, 2, 0, 1])
    expected = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]])
    assert_array_equal(out, expected)


def check_binarized_results(y, classes, pos_label, neg_label, expected):
    for sparse_output in [True, False]:
        if (pos_label == 0 or neg_label != 0) and sparse_output:
            with pytest.raises(ValueError):
                label_binarize(
                    y,
                    classes=classes,
                    neg_label=neg_label,
                    pos_label=pos_label,
                    sparse_output=sparse_output,
                )
            continue

        # check label_binarize
        binarized = label_binarize(
            y,
            classes=classes,
            neg_label=neg_label,
            pos_label=pos_label,
            sparse_output=sparse_output,
        )
        assert_array_equal(toarray(binarized), expected)
        assert issparse(binarized) == sparse_output

        # check inverse
        y_type = type_of_target(y)
        if y_type == "multiclass":
            inversed = _inverse_binarize_multiclass(binarized, classes=classes)

        else:
            inversed = _inverse_binarize_thresholding(
                binarized,
                output_type=y_type,
                classes=classes,
                threshold=((neg_label + pos_label) / 2.0),
            )

        assert_array_equal(toarray(inversed), toarray(y))

        # Check label binarizer
        lb = LabelBinarizer(
            neg_label=neg_label, pos_label=pos_label, sparse_output=sparse_output
        )
        binarized = lb.fit_transform(y)
        assert_array_equal(toarray(binarized), expected)
        assert issparse(binarized) == sparse_output
        inverse_output = lb.inverse_transform(binarized)
        assert_array_equal(toarray(inverse_output), toarray(y))
        assert issparse(inverse_output) == issparse(y)


def test_label_binarize_binary():
    y = [0, 1, 0]
    classes = [0, 1]
    pos_label = 2
    neg_label = -1
    expected = np.array([[2, -1], [-1, 2], [2, -1]])[:, 1].reshape((-1, 1))

    check_binarized_results(y, classes, pos_label, neg_label, expected)

    # Binary case where sparse_output = True will not result in a ValueError
    y = [0, 1, 0]
    classes = [0, 1]
    pos_label = 3
    neg_label = 0
    expected = np.array([[3, 0], [0, 3], [3, 0]])[:, 1].reshape((-1, 1))

    check_binarized_results(y, classes, pos_label, neg_label, expected)


def test_label_binarize_multiclass():
    y = [0, 1, 2]
    classes = [0, 1, 2]
    pos_label = 2
    neg_label = 0
    expected = 2 * np.eye(3)

    check_binarized_results(y, classes, pos_label, neg_label, expected)

    with pytest.raises(ValueError):
        label_binarize(
            y, classes=classes, neg_label=-1, pos_label=pos_label, sparse_output=True
        )


def test_label_binarize_multilabel():
    y_ind = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]])
    classes = [0, 1, 2]
    pos_label = 2
    neg_label = 0
    expected = pos_label * y_ind
    y_sparse = [
        sparse_matrix(y_ind)
        for sparse_matrix in [
            coo_matrix,
            csc_matrix,
            csr_matrix,
            dok_matrix,
            lil_matrix,
        ]
    ]

    for y in [y_ind] + y_sparse:
        check_binarized_results(y, classes, pos_label, neg_label, expected)

    with pytest.raises(ValueError):
        label_binarize(
            y, classes=classes, neg_label=-1, pos_label=pos_label, sparse_output=True
        )


def test_invalid_input_label_binarize():
    with pytest.raises(ValueError):
        label_binarize([0, 2], classes=[0, 2], pos_label=0, neg_label=1)
    with pytest.raises(ValueError, match="continuous target data is not "):
        label_binarize([1.2, 2.7], classes=[0, 1])
    with pytest.raises(ValueError, match="mismatch with the labels"):
        label_binarize([[1, 3]], classes=[1, 2, 3])


def test_inverse_binarize_multiclass():
    got = _inverse_binarize_multiclass(
        csr_matrix([[0, 1, 0], [-1, 0, -1], [0, 0, 0]]), np.arange(3)
    )
    assert_array_equal(got, np.array([1, 1, 0]))
