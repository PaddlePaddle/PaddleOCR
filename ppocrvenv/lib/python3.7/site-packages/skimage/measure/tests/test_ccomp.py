import numpy as np

from skimage.measure import label
import skimage.measure._ccomp as ccomp

from skimage._shared import testing
from skimage._shared.testing import assert_array_equal

BG = 0  # background value


class TestConnectedComponents:
    def setup(self):
        self.x = np.array([
            [0, 0, 3, 2, 1, 9],
            [0, 1, 1, 9, 2, 9],
            [0, 0, 1, 9, 9, 9],
            [3, 1, 1, 5, 3, 0]])

        self.labels = np.array([
            [0, 0, 1, 2, 3, 4],
            [0, 5, 5, 4, 2, 4],
            [0, 0, 5, 4, 4, 4],
            [6, 5, 5, 7, 8, 0]])

        # No background - there is no label 0, instead, labelling starts with 1
        # and all labels are incremented by 1.
        self.labels_nobg = self.labels + 1
        # The 0 at lower right corner is isolated, so it should get a new label
        self.labels_nobg[-1, -1] = 10

        # We say that background value is 9 (and bg label is 0)
        self.labels_bg_9 = self.labels_nobg.copy()
        self.labels_bg_9[self.x == 9] = 0
        # Then, where there was the label 5, we now expect 4 etc.
        # (we assume that the label of value 9 would normally be 5)
        self.labels_bg_9[self.labels_bg_9 > 5] -= 1

    def test_basic(self):
        assert_array_equal(label(self.x), self.labels)

        # Make sure data wasn't modified
        assert self.x[0, 2] == 3

        # Check that everything works if there is no background
        assert_array_equal(label(self.x, background=99), self.labels_nobg)
        # Check that everything works if background value != 0
        assert_array_equal(label(self.x, background=9), self.labels_bg_9)

    def test_random(self):
        x = (np.random.rand(20, 30) * 5).astype(int)
        labels = label(x)

        n = labels.max()
        for i in range(n):
            values = x[labels == i]
            assert np.all(values == values[0])

    def test_diag(self):
        x = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0]])
        assert_array_equal(label(x), x)

    def test_4_vs_8(self):
        x = np.array([[0, 1],
                      [1, 0]], dtype=int)

        assert_array_equal(label(x, connectivity=1),
                           [[0, 1],
                            [2, 0]])
        assert_array_equal(label(x, connectivity=2),
                           [[0, 1],
                            [1, 0]])

    def test_background(self):
        x = np.array([[1, 0, 0],
                      [1, 1, 5],
                      [0, 0, 0]])

        assert_array_equal(label(x), [[1, 0, 0],
                                      [1, 1, 2],
                                      [0, 0, 0]])

        assert_array_equal(label(x, background=0),
                           [[1, 0, 0],
                            [1, 1, 2],
                            [0, 0, 0]])

    def test_background_two_regions(self):
        x = np.array([[0, 0, 6],
                      [0, 0, 6],
                      [5, 5, 5]])

        res = label(x, background=0)
        assert_array_equal(res,
                           [[0, 0, 1],
                            [0, 0, 1],
                            [2, 2, 2]])

    def test_background_one_region_center(self):
        x = np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]])

        assert_array_equal(label(x, connectivity=1, background=0),
                           [[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])


    def test_return_num(self):
        x = np.array([[1, 0, 6],
                      [0, 0, 6],
                      [5, 5, 5]])

        assert_array_equal(label(x, return_num=True)[1], 3)

        assert_array_equal(label(x, background=-1, return_num=True)[1], 4)


class TestConnectedComponents3d:
    def setup(self):
        self.x = np.zeros((3, 4, 5), int)
        self.x[0] = np.array([[0, 3, 2, 1, 9],
                              [0, 1, 9, 2, 9],
                              [0, 1, 9, 9, 9],
                              [3, 1, 5, 3, 0]])

        self.x[1] = np.array([[3, 3, 2, 1, 9],
                              [0, 3, 9, 2, 1],
                              [0, 3, 3, 1, 1],
                              [3, 1, 3, 3, 0]])

        self.x[2] = np.array([[3, 3, 8, 8, 0],
                              [2, 3, 9, 8, 8],
                              [2, 3, 0, 8, 0],
                              [2, 1, 0, 0, 0]])

        self.labels = np.zeros((3, 4, 5), int)

        self.labels[0] = np.array([[0, 1, 2, 3, 4],
                                   [0, 5, 4, 2, 4],
                                   [0, 5, 4, 4, 4],
                                   [1, 5, 6, 1, 0]])

        self.labels[1] = np.array([[1, 1, 2, 3, 4],
                                   [0, 1, 4, 2, 3],
                                   [0, 1, 1, 3, 3],
                                   [1, 5, 1, 1, 0]])

        self.labels[2] = np.array([[1, 1, 7, 7, 0],
                                   [8, 1, 4, 7, 7],
                                   [8, 1, 0, 7, 0],
                                   [8, 5, 0, 0, 0]])

    def test_basic(self):
        labels = label(self.x)
        assert_array_equal(labels, self.labels)

        assert self.x[0, 0, 2] == 2, \
            "Data was modified!"

    def test_random(self):
        x = (np.random.rand(20, 30) * 5).astype(int)
        labels = label(x)

        n = labels.max()
        for i in range(n):
            values = x[labels == i]
            assert np.all(values == values[0])

    def test_diag(self):
        x = np.zeros((3, 3, 3), int)
        x[0, 2, 2] = 1
        x[1, 1, 1] = 1
        x[2, 0, 0] = 1
        assert_array_equal(label(x), x)

    def test_4_vs_8(self):
        x = np.zeros((2, 2, 2), int)
        x[0, 1, 1] = 1
        x[1, 0, 0] = 1
        label4 = x.copy()
        label4[1, 0, 0] = 2
        assert_array_equal(label(x, connectivity=1), label4)
        assert_array_equal(label(x, connectivity=3), x)

    def test_connectivity_1_vs_2(self):
        x = np.zeros((2, 2, 2), int)
        x[0, 1, 1] = 1
        x[1, 0, 0] = 1
        label1 = x.copy()
        label1[1, 0, 0] = 2
        assert_array_equal(label(x, connectivity=1), label1)
        assert_array_equal(label(x, connectivity=3), x)

    def test_background(self):
        x = np.zeros((2, 3, 3), int)
        x[0] = np.array([[1, 0, 0],
                         [1, 0, 0],
                         [0, 0, 0]])
        x[1] = np.array([[0, 0, 0],
                         [0, 1, 5],
                         [0, 0, 0]])

        lnb = x.copy()
        lnb[0] = np.array([[1, 2, 2],
                           [1, 2, 2],
                           [2, 2, 2]])
        lnb[1] = np.array([[2, 2, 2],
                           [2, 1, 3],
                           [2, 2, 2]])
        lb = x.copy()
        lb[0] = np.array([[1,  BG, BG],
                          [1,  BG, BG],
                          [BG, BG, BG]])
        lb[1] = np.array([[BG, BG, BG],
                          [BG, 1,   2],
                          [BG, BG, BG]])

        assert_array_equal(label(x), lb)
        assert_array_equal(label(x, background=-1), lnb)

    def test_background_two_regions(self):
        x = np.zeros((2, 3, 3), int)
        x[0] = np.array([[0, 0, 6],
                         [0, 0, 6],
                         [5, 5, 5]])
        x[1] = np.array([[6, 6, 0],
                         [5, 0, 0],
                         [0, 0, 0]])
        lb = x.copy()
        lb[0] = np.array([[BG, BG, 1],
                          [BG, BG, 1],
                          [2,   2, 2]])
        lb[1] = np.array([[1,  1,  BG],
                          [2,  BG, BG],
                          [BG, BG, BG]])

        res = label(x, background=0)
        assert_array_equal(res, lb)

    def test_background_one_region_center(self):
        x = np.zeros((3, 3, 3), int)
        x[1, 1, 1] = 1

        lb = np.ones_like(x) * BG
        lb[1, 1, 1] = 1

        assert_array_equal(label(x, connectivity=1, background=0), lb)

    def test_return_num(self):
        x = np.array([[1, 0, 6],
                      [0, 0, 6],
                      [5, 5, 5]])

        assert_array_equal(label(x, return_num=True)[1], 3)
        assert_array_equal(label(x, background=-1, return_num=True)[1], 4)

    def test_1D(self):
        x = np.array((0, 1, 2, 2, 1, 1, 0, 0))
        xlen = len(x)
        y = np.array((0, 1, 2, 2, 3, 3, 0, 0))
        reshapes = ((xlen,),
                    (1, xlen), (xlen, 1),
                    (1, xlen, 1), (xlen, 1, 1), (1, 1, xlen))
        for reshape in reshapes:
            x2 = x.reshape(reshape)
            labelled = label(x2)
            assert_array_equal(y, labelled.flatten())

    def test_nd(self):
        x = np.ones((1, 2, 3, 4))
        with testing.raises(NotImplementedError):
            label(x)


class TestSupport:
    def test_reshape(self):
        shapes_in = ((3, 1, 2), (1, 4, 5), (3, 1, 1), (2, 1), (1,))
        for shape in shapes_in:
            shape = np.array(shape)
            numones = sum(shape == 1)
            inp = np.random.random(shape)

            fixed, swaps = ccomp.reshape_array(inp)
            shape2 = fixed.shape
            # now check that all ones are at the beginning
            for i in range(numones):
                assert shape2[i] == 1

            back = ccomp.undo_reshape_array(fixed, swaps)
            # check that the undo works as expected
            assert_array_equal(inp, back)
