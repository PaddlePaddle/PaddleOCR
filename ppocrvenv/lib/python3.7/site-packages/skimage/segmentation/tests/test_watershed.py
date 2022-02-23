"""test_watershed.py - tests the watershed function

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Original author: Lee Kamentsky
"""
#Portions of this test were taken from scipy's watershed test in test_ndimage.py
#
# Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import math
import unittest

import numpy as np
import pytest
from scipy import ndimage as ndi

from skimage._shared.filters import gaussian
from skimage.measure import label

from .._watershed import watershed

eps = 1e-12
blob = np.array([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                 [255, 255, 255, 255, 255, 204, 204, 204, 204, 204, 204, 255, 255, 255, 255, 255],
                 [255, 255, 255, 204, 204, 183, 153, 153, 153, 153, 183, 204, 204, 255, 255, 255],
                 [255, 255, 204, 183, 153, 141, 111, 103, 103, 111, 141, 153, 183, 204, 255, 255],
                 [255, 255, 204, 153, 111,  94,  72,  52,  52,  72,  94, 111, 153, 204, 255, 255],
                 [255, 255, 204, 153, 111,  72,  39,   1,   1,  39,  72, 111, 153, 204, 255, 255],
                 [255, 255, 204, 183, 141, 111,  72,  39,  39,  72, 111, 141, 183, 204, 255, 255],
                 [255, 255, 255, 204, 183, 141, 111,  72,  72, 111, 141, 183, 204, 255, 255, 255],
                 [255, 255, 255, 255, 204, 183, 141,  94,  94, 141, 183, 204, 255, 255, 255, 255],
                 [255, 255, 255, 255, 255, 204, 153, 103, 103, 153, 204, 255, 255, 255, 255, 255],
                 [255, 255, 255, 255, 204, 183, 141,  94,  94, 141, 183, 204, 255, 255, 255, 255],
                 [255, 255, 255, 204, 183, 141, 111,  72,  72, 111, 141, 183, 204, 255, 255, 255],
                 [255, 255, 204, 183, 141, 111,  72,  39,  39,  72, 111, 141, 183, 204, 255, 255],
                 [255, 255, 204, 153, 111,  72,  39,   1,   1,  39,  72, 111, 153, 204, 255, 255],
                 [255, 255, 204, 153, 111,  94,  72,  52,  52,  72,  94, 111, 153, 204, 255, 255],
                 [255, 255, 204, 183, 153, 141, 111, 103, 103, 111, 141, 153, 183, 204, 255, 255],
                 [255, 255, 255, 204, 204, 183, 153, 153, 153, 153, 183, 204, 204, 255, 255, 255],
                 [255, 255, 255, 255, 255, 204, 204, 204, 204, 204, 204, 255, 255, 255, 255, 255],
                 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                 [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]])


def diff(a, b):
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    if not isinstance(b, np.ndarray):
        b = np.asarray(b)
    if (0 in a.shape) and (0 in b.shape):
        return 0.0
    b[a == 0] = 0
    if (a.dtype in [np.complex64, np.complex128] or
        b.dtype in [np.complex64, np.complex128]):
        a = np.asarray(a, np.complex128)
        b = np.asarray(b, np.complex128)
        t = ((a.real - b.real)**2).sum() + ((a.imag - b.imag)**2).sum()
    else:
        a = np.asarray(a)
        a = a.astype(np.float64)
        b = np.asarray(b)
        b = b.astype(np.float64)
        t = ((a - b)**2).sum()
    return math.sqrt(t)


class TestWatershed(unittest.TestCase):
    eight = np.ones((3, 3), bool)

    def test_watershed01(self):
        "watershed 1"
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[ -1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 1, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0]],
                                 np.int8)
        out = watershed(data, markers, self.eight)
        expected = np.array([[-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]])
        error = diff(expected, out)
        assert error < eps

    def test_watershed02(self):
        "watershed 2"
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
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
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], np.int8)
        out = watershed(data, markers)
        error = diff([[-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1,  1,  1,  1, -1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1, -1,  1,  1,  1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]], out)
        self.assertTrue(error < eps)

    def test_watershed03(self):
        "watershed 3"
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 3, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, -1]], np.int8)
        out = watershed(data, markers)
        error = diff([[-1, -1, -1, -1, -1, -1, -1],
                      [-1,  0,  2,  0,  3,  0, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  0,  2,  0,  3,  0, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]], out)
        self.assertTrue(error < eps)

    def test_watershed04(self):
        "watershed 4"
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 3, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, -1]], np.int8)
        out = watershed(data, markers, self.eight)
        error = diff([[-1, -1, -1, -1, -1, -1, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]], out)
        self.assertTrue(error < eps)

    def test_watershed05(self):
        "watershed 5"
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 0, 1, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 3, 0, 2, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, -1]], np.int8)
        out = watershed(data, markers, self.eight)
        error = diff([[-1, -1, -1, -1, -1, -1, -1],
                      [-1,  3,  3,  0,  2,  2, -1],
                      [-1,  3,  3,  0,  2,  2, -1],
                      [-1,  3,  3,  0,  2,  2, -1],
                      [-1,  3,  3,  0,  2,  2, -1],
                      [-1,  3,  3,  0,  2,  2, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]], out)
        self.assertTrue(error < eps)

    def test_watershed06(self):
        "watershed 6"
        data = np.array([[0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [-1, 0, 0, 0, 0, 0, 0]], np.int8)
        out = watershed(data, markers, self.eight)
        error = diff([[-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]], out)
        self.assertTrue(error < eps)

    def test_watershed07(self):
        "A regression test of a competitive case that failed"
        data = blob
        mask = (data != 255)
        markers = np.zeros(data.shape, int)
        markers[6, 7] = 1
        markers[14, 7] = 2
        out = watershed(data, markers, self.eight, mask=mask)
        #
        # The two objects should be the same size, except possibly for the
        # border region
        #
        size1 = np.sum(out == 1)
        size2 = np.sum(out == 2)
        self.assertTrue(abs(size1 - size2) <= 6)

    def test_watershed08(self):
        "The border pixels + an edge are all the same value"
        data = blob.copy()
        data[10, 7:9] = 141
        mask = (data != 255)
        markers = np.zeros(data.shape, int)
        markers[6, 7] = 1
        markers[14, 7] = 2
        out = watershed(data, markers, self.eight, mask=mask)
        #
        # The two objects should be the same size, except possibly for the
        # border region
        #
        size1 = np.sum(out == 1)
        size2 = np.sum(out == 2)
        self.assertTrue(abs(size1 - size2) <= 6)

    def test_watershed09(self):
        """Test on an image of reasonable size

        This is here both for timing (does it take forever?) and to
        ensure that the memory constraints are reasonable
        """
        image = np.zeros((1000, 1000))
        coords = np.random.uniform(0, 1000, (100, 2)).astype(int)
        markers = np.zeros((1000, 1000), int)
        idx = 1
        for x, y in coords:
            image[x, y] = 1
            markers[x, y] = idx
            idx += 1

        image = gaussian(image, 4, mode='reflect')
        watershed(image, markers, self.eight)
        ndi.watershed_ift(image.astype(np.uint16), markers, self.eight)

    def test_watershed10(self):
        "watershed 10"
        data = np.array([[1, 1, 1, 1],
                         [1, 1, 1, 1],
                         [1, 1, 1, 1],
                         [1, 1, 1, 1]], np.uint8)
        markers = np.array([[1, 0, 0, 2],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [3, 0, 0, 4]], np.int8)
        out = watershed(data, markers, self.eight)
        error = diff([[1, 1, 2, 2],
                      [1, 1, 2, 2],
                      [3, 3, 4, 4],
                      [3, 3, 4, 4]], out)
        self.assertTrue(error < eps)

    def test_watershed11(self):
        '''Make sure that all points on this plateau are assigned to closest seed'''
        # https://github.com/scikit-image/scikit-image/issues/803
        #
        # Make sure that no point in a level image is farther away
        # from its seed than any other
        #
        image = np.zeros((21, 21))
        markers = np.zeros((21, 21), int)
        markers[5, 5] = 1
        markers[5, 10] = 2
        markers[10, 5] = 3
        markers[10, 10] = 4

        structure = np.array([[False, True, False],
                              [True, True, True],
                              [False, True, False]])
        out = watershed(image, markers, structure)
        i, j = np.mgrid[0:21, 0:21]
        d = np.dstack(
            [np.sqrt((i.astype(float)-i0)**2, (j.astype(float)-j0)**2)
             for i0, j0 in ((5, 5), (5, 10), (10, 5), (10, 10))])
        dmin = np.min(d, 2)
        self.assertTrue(np.all(d[i, j, out[i, j]-1] == dmin))


    def test_watershed12(self):
        "The watershed line"
        data = np.array([[203, 255, 203, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153, 153],
                         [203, 255, 203, 153, 153, 153, 102, 102, 102, 102, 102, 102, 153, 153, 153, 153],
                         [203, 255, 203, 203, 153, 153, 102, 102,  77,   0, 102, 102, 153, 153, 203, 203],
                         [203, 255, 255, 203, 153, 153, 153, 102, 102, 102, 102, 153, 153, 203, 203, 255],
                         [203, 203, 255, 203, 203, 203, 153, 153, 153, 153, 153, 153, 203, 203, 255, 255],
                         [153, 203, 255, 255, 255, 203, 203, 203, 203, 203, 203, 203, 203, 255, 255, 203],
                         [153, 203, 203, 203, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 203, 203],
                         [153, 153, 153, 203, 203, 203, 203, 203, 255, 203, 203, 203, 203, 203, 203, 153],
                         [102, 102, 153, 153, 153, 153, 203, 203, 255, 203, 203, 255, 203, 153, 153, 153],
                         [102, 102, 102, 102, 102, 153, 203, 255, 255, 203, 203, 203, 203, 153, 102, 153],
                         [102,  51,  51, 102, 102, 153, 203, 255, 203, 203, 153, 153, 153, 153, 102, 153],
                         [ 77,  51,  51, 102, 153, 153, 203, 255, 203, 203, 203, 153, 102, 102, 102, 153],
                         [ 77,   0,  51, 102, 153, 203, 203, 255, 203, 255, 203, 153, 102,  51, 102, 153],
                         [ 77,   0,  51, 102, 153, 203, 255, 255, 203, 203, 203, 153, 102,   0, 102, 153],
                         [102,   0,  51, 102, 153, 203, 255, 203, 203, 153, 153, 153, 102, 102, 102, 153],
                         [102, 102, 102, 102, 153, 203, 255, 203, 153, 153, 153, 153, 153, 153, 153, 153]])
        markerbin = (data==0)
        marker = label(markerbin)
        ws = watershed(data, marker, connectivity=2, watershed_line=True)
        for lab, area in zip(range(4), [34,74,74,74]):
            self.assertTrue(np.sum(ws == lab) == area)



def test_compact_watershed():
    image = np.zeros((5, 6))
    image[:, 3:] = 1
    seeds = np.zeros((5, 6), dtype=int)
    seeds[2, 0] = 1
    seeds[2, 3] = 2
    compact = watershed(image, seeds, compactness=0.01)
    expected = np.array([[1, 1, 1, 2, 2, 2],
                         [1, 1, 1, 2, 2, 2],
                         [1, 1, 1, 2, 2, 2],
                         [1, 1, 1, 2, 2, 2],
                         [1, 1, 1, 2, 2, 2]], dtype=int)
    np.testing.assert_equal(compact, expected)
    normal = watershed(image, seeds)
    expected = np.ones(image.shape, dtype=int)
    expected[2, 3:] = 2
    np.testing.assert_equal(normal, expected)


def test_numeric_seed_watershed():
    """Test that passing just the number of seeds to watershed works."""
    image = np.zeros((5, 6))
    image[:, 3:] = 1
    compact = watershed(image, 2, compactness=0.01)
    expected = np.array([[1, 1, 1, 1, 2, 2],
                         [1, 1, 1, 1, 2, 2],
                         [1, 1, 1, 1, 2, 2],
                         [1, 1, 1, 1, 2, 2],
                         [1, 1, 1, 1, 2, 2]], dtype=np.int32)
    np.testing.assert_equal(compact, expected)


def test_incorrect_markers_shape():
    image = np.ones((5, 6))
    markers = np.ones((5, 7))
    with pytest.raises(ValueError):
        watershed(image, markers)


def test_incorrect_mask_shape():
    image = np.ones((5, 6))
    mask = np.ones((5, 7))
    with pytest.raises(ValueError):
        watershed(image, markers=4, mask=mask)


def test_markers_in_mask():
    data = blob
    mask = (data != 255)
    out = watershed(data, 25, connectivity=2, mask=mask)
    # There should be no markers where the mask is false
    assert np.all(out[~mask] == 0)


def test_no_markers():
    data = blob
    mask = (data != 255)
    out = watershed(data, mask=mask)
    assert np.max(out) == 2


def test_connectivity():
    """
    Watershed segmentation should output different result for 
    different connectivity
    when markers are calculated where None is supplied.
    Issue = 5084
    """
    # Generate a dummy BrightnessTemperature image
    x, y = np.indices((406, 270))
    x1, y1, x2, y2, x3, y3, x4, y4 = 200, 208, 300, 120, 100, 100, 340, 208
    r1, r2, r3, r4 = 100, 50, 40, 80
    mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
    mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
    mask_circle3 = (x - x3)**2 + (y - y3)**2 < r3**2
    mask_circle4 = (x - x4)**2 + (y - y4)**2 < r4**2
    image = np.logical_or(mask_circle1, mask_circle2)
    image = np.logical_or(image, mask_circle3)
    image = np.logical_or(image, mask_circle4)

    # calcuate distance in discrete increase
    DummyBT = ndi.distance_transform_edt(image)
    DummyBT_dis = np.around(DummyBT / 12, decimals = 0)*12
    # calculate the mask
    Img_mask = np.where(DummyBT_dis == 0, 0, 1)

    # segments for connectivity 1 and 2
    labels_c1 = watershed(200 - DummyBT_dis, mask=Img_mask, connectivity=1,
                          compactness=0.01)
    labels_c2 = watershed(200 - DummyBT_dis, mask=Img_mask, connectivity=2,
                          compactness=0.01)

    # assertions
    assert np.unique(labels_c1).shape[0] == 6
    assert np.unique(labels_c2).shape[0] == 5

    # checking via area of each individual segment.
    for lab, area in zip(range(6), [61824, 3653, 20467, 11097, 1301, 11278]):
        assert np.sum(labels_c1 == lab) == area

    for lab, area in zip(range(5), [61824, 3653, 20466, 12386, 11291]):
        assert np.sum(labels_c2 == lab) == area
