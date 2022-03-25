# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Tests IO functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pandas as pd

import motmetrics as mm

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')


def test_load_vatic():
    """Tests VATIC_TXT format."""
    df = mm.io.loadtxt(os.path.join(DATA_DIR, 'iotest/vatic.txt'), fmt=mm.io.Format.VATIC_TXT)

    expected = pd.DataFrame([
        # F,ID,Y,W,H,L,O,G,F,A1,A2,A3,A4
        (0, 0, 412, 0, 430, 124, 0, 0, 0, 'worker', 0, 0, 0, 0),
        (1, 0, 412, 10, 430, 114, 0, 0, 1, 'pc', 1, 0, 1, 0),
        (1, 1, 412, 0, 430, 124, 0, 0, 1, 'pc', 0, 1, 0, 0),
        (2, 2, 412, 0, 430, 124, 0, 0, 1, 'worker', 1, 1, 0, 1)
    ])

    assert (df.reset_index().values == expected.values).all()


def test_load_motchallenge():
    """Tests MOT15_2D format."""
    df = mm.io.loadtxt(os.path.join(DATA_DIR, 'iotest/motchallenge.txt'), fmt=mm.io.Format.MOT15_2D)

    expected = pd.DataFrame([
        (1, 1, 398, 181, 121, 229, 1, -1, -1),  # Note -1 on x and y for correcting matlab
        (1, 2, 281, 200, 92, 184, 1, -1, -1),
        (2, 2, 268, 201, 87, 182, 1, -1, -1),
        (2, 3, 70, 150, 100, 284, 1, -1, -1),
        (2, 4, 199, 205, 55, 137, 1, -1, -1),
    ])

    assert (df.reset_index().values == expected.values).all()


def test_load_detrac_mat():
    """Tests DETRAC_MAT format."""
    df = mm.io.loadtxt(os.path.join(DATA_DIR, 'iotest/detrac.mat'), fmt=mm.io.Format.DETRAC_MAT)

    expected = pd.DataFrame([
        (1., 1., 745., 356., 148., 115., 1., -1., -1.),
        (2., 1., 738., 350., 145., 111., 1., -1., -1.),
        (3., 1., 732., 343., 142., 107., 1., -1., -1.),
        (4., 1., 725., 336., 139., 104., 1., -1., -1.)
    ])

    assert (df.reset_index().values == expected.values).all()


def test_load_detrac_xml():
    """Tests DETRAC_XML format."""
    df = mm.io.loadtxt(os.path.join(DATA_DIR, 'iotest/detrac.xml'), fmt=mm.io.Format.DETRAC_XML)

    expected = pd.DataFrame([
        (1., 1., 744.6, 356.33, 148.2, 115.14, 1., -1., -1.),
        (2., 1., 738.2, 349.51, 145.21, 111.29, 1., -1., -1.),
        (3., 1., 731.8, 342.68, 142.23, 107.45, 1., -1., -1.),
        (4., 1., 725.4, 335.85, 139.24, 103.62, 1., -1., -1.)
    ])

    assert (df.reset_index().values == expected.values).all()
