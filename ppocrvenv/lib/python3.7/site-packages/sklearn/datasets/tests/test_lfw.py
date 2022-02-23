"""This test for the LFW require medium-size data downloading and processing

If the data has not been already downloaded by running the examples,
the tests won't run (skipped).

If the test are run, the first execution will be long (typically a bit
more than a couple of minutes) but as the dataset loader is leveraging
joblib, successive runs will be fast (less than 200ms).
"""

import random
import os
import shutil
import tempfile
import numpy as np
import pytest
from functools import partial
from sklearn.externals._pilutil import pillow_installed, imsave
from sklearn.datasets import fetch_lfw_pairs
from sklearn.datasets import fetch_lfw_people

from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import SkipTest
from sklearn.datasets.tests.test_common import check_return_X_y


SCIKIT_LEARN_DATA = None
SCIKIT_LEARN_EMPTY_DATA = None
LFW_HOME = None

FAKE_NAMES = [
    "Abdelatif_Smith",
    "Abhati_Kepler",
    "Camara_Alvaro",
    "Chen_Dupont",
    "John_Lee",
    "Lin_Bauman",
    "Onur_Lopez",
]


def setup_module():
    """Test fixture run once and common to all tests of this module"""
    if not pillow_installed:
        raise SkipTest("PIL not installed.")

    global SCIKIT_LEARN_DATA, SCIKIT_LEARN_EMPTY_DATA, LFW_HOME

    SCIKIT_LEARN_DATA = tempfile.mkdtemp(prefix="scikit_learn_lfw_test_")
    LFW_HOME = os.path.join(SCIKIT_LEARN_DATA, "lfw_home")

    SCIKIT_LEARN_EMPTY_DATA = tempfile.mkdtemp(prefix="scikit_learn_empty_test_")

    if not os.path.exists(LFW_HOME):
        os.makedirs(LFW_HOME)

    random_state = random.Random(42)
    np_rng = np.random.RandomState(42)

    # generate some random jpeg files for each person
    counts = {}
    for name in FAKE_NAMES:
        folder_name = os.path.join(LFW_HOME, "lfw_funneled", name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        n_faces = np_rng.randint(1, 5)
        counts[name] = n_faces
        for i in range(n_faces):
            file_path = os.path.join(folder_name, name + "_%04d.jpg" % i)
            uniface = np_rng.randint(0, 255, size=(250, 250, 3))
            try:
                imsave(file_path, uniface)
            except ImportError:
                raise SkipTest("PIL not installed")

    # add some random file pollution to test robustness
    with open(os.path.join(LFW_HOME, "lfw_funneled", ".test.swp"), "wb") as f:
        f.write(b"Text file to be ignored by the dataset loader.")

    # generate some pairing metadata files using the same format as LFW
    with open(os.path.join(LFW_HOME, "pairsDevTrain.txt"), "wb") as f:
        f.write(b"10\n")
        more_than_two = [name for name, count in counts.items() if count >= 2]
        for i in range(5):
            name = random_state.choice(more_than_two)
            first, second = random_state.sample(range(counts[name]), 2)
            f.write(("%s\t%d\t%d\n" % (name, first, second)).encode())

        for i in range(5):
            first_name, second_name = random_state.sample(FAKE_NAMES, 2)
            first_index = random_state.choice(np.arange(counts[first_name]))
            second_index = random_state.choice(np.arange(counts[second_name]))
            f.write(
                (
                    "%s\t%d\t%s\t%d\n"
                    % (first_name, first_index, second_name, second_index)
                ).encode()
            )

    with open(os.path.join(LFW_HOME, "pairsDevTest.txt"), "wb") as f:
        f.write(b"Fake place holder that won't be tested")

    with open(os.path.join(LFW_HOME, "pairs.txt"), "wb") as f:
        f.write(b"Fake place holder that won't be tested")


def teardown_module():
    """Test fixture (clean up) run once after all tests of this module"""
    if os.path.isdir(SCIKIT_LEARN_DATA):
        shutil.rmtree(SCIKIT_LEARN_DATA)
    if os.path.isdir(SCIKIT_LEARN_EMPTY_DATA):
        shutil.rmtree(SCIKIT_LEARN_EMPTY_DATA)


def test_load_empty_lfw_people():
    with pytest.raises(IOError):
        fetch_lfw_people(data_home=SCIKIT_LEARN_EMPTY_DATA, download_if_missing=False)


def test_load_fake_lfw_people():
    lfw_people = fetch_lfw_people(
        data_home=SCIKIT_LEARN_DATA, min_faces_per_person=3, download_if_missing=False
    )

    # The data is croped around the center as a rectangular bounding box
    # around the face. Colors are converted to gray levels:
    assert lfw_people.images.shape == (10, 62, 47)
    assert lfw_people.data.shape == (10, 2914)

    # the target is array of person integer ids
    assert_array_equal(lfw_people.target, [2, 0, 1, 0, 2, 0, 2, 1, 1, 2])

    # names of the persons can be found using the target_names array
    expected_classes = ["Abdelatif Smith", "Abhati Kepler", "Onur Lopez"]
    assert_array_equal(lfw_people.target_names, expected_classes)

    # It is possible to ask for the original data without any croping or color
    # conversion and not limit on the number of picture per person
    lfw_people = fetch_lfw_people(
        data_home=SCIKIT_LEARN_DATA,
        resize=None,
        slice_=None,
        color=True,
        download_if_missing=False,
    )
    assert lfw_people.images.shape == (17, 250, 250, 3)
    assert lfw_people.DESCR.startswith(".. _labeled_faces_in_the_wild_dataset:")

    # the ids and class names are the same as previously
    assert_array_equal(
        lfw_people.target, [0, 0, 1, 6, 5, 6, 3, 6, 0, 3, 6, 1, 2, 4, 5, 1, 2]
    )
    assert_array_equal(
        lfw_people.target_names,
        [
            "Abdelatif Smith",
            "Abhati Kepler",
            "Camara Alvaro",
            "Chen Dupont",
            "John Lee",
            "Lin Bauman",
            "Onur Lopez",
        ],
    )

    # test return_X_y option
    fetch_func = partial(
        fetch_lfw_people,
        data_home=SCIKIT_LEARN_DATA,
        resize=None,
        slice_=None,
        color=True,
        download_if_missing=False,
    )
    check_return_X_y(lfw_people, fetch_func)


def test_load_fake_lfw_people_too_restrictive():
    with pytest.raises(ValueError):
        fetch_lfw_people(
            data_home=SCIKIT_LEARN_DATA,
            min_faces_per_person=100,
            download_if_missing=False,
        )


def test_load_empty_lfw_pairs():
    with pytest.raises(IOError):
        fetch_lfw_pairs(data_home=SCIKIT_LEARN_EMPTY_DATA, download_if_missing=False)


def test_load_fake_lfw_pairs():
    lfw_pairs_train = fetch_lfw_pairs(
        data_home=SCIKIT_LEARN_DATA, download_if_missing=False
    )

    # The data is croped around the center as a rectangular bounding box
    # around the face. Colors are converted to gray levels:
    assert lfw_pairs_train.pairs.shape == (10, 2, 62, 47)

    # the target is whether the person is the same or not
    assert_array_equal(lfw_pairs_train.target, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    # names of the persons can be found using the target_names array
    expected_classes = ["Different persons", "Same person"]
    assert_array_equal(lfw_pairs_train.target_names, expected_classes)

    # It is possible to ask for the original data without any croping or color
    # conversion
    lfw_pairs_train = fetch_lfw_pairs(
        data_home=SCIKIT_LEARN_DATA,
        resize=None,
        slice_=None,
        color=True,
        download_if_missing=False,
    )
    assert lfw_pairs_train.pairs.shape == (10, 2, 250, 250, 3)

    # the ids and class names are the same as previously
    assert_array_equal(lfw_pairs_train.target, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert_array_equal(lfw_pairs_train.target_names, expected_classes)

    assert lfw_pairs_train.DESCR.startswith(".. _labeled_faces_in_the_wild_dataset:")
