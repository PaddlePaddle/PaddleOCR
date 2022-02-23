from scipy import ndimage as ndi
from skimage import data

import numpy as np

from skimage import measure
from skimage.segmentation._expand_labels import expand_labels

from skimage._shared import testing
from skimage._shared.testing import assert_array_equal

SAMPLE1D = np.array([0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
SAMPLE1D_EXPANDED_3 = np.array([4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])

# Some pixels are important edge cases with undefined behaviour:
# these are the pixels that are at the same distance from
# multiple labels. Ideally the label would be chosen at random
# to avoid bias, but as we are relying on the index map returned
# by the scipy.ndimage distance transform, what actually happens
# is determined by the upstream implementation of the distance
# tansform, thus we don't give any guarantees for the edge case pixels.
#  
# Regardless, it seems prudent to have a test including an edge case
# so we can detect whether future upstream changes in scipy.ndimage 
# modify the behaviour.

EDGECASE1D = np.array([0, 0, 4, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
EDGECASE1D_EXPANDED_3 = np.array([4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])

SAMPLE2D = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)

SAMPLE2D_EXPANDED_3 = np.array(
      [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 0],
       [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
       [1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2],
       [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
       [1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2],
       [1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
       [0, 0, 1, 0, 0, 0, 0, 2, 2, 2, 2],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0]]
       )

# non-integer expansion
SAMPLE2D_EXPANDED_1_5 = np.array(
      [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2],
       [1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2],
       [0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2],
       [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


EDGECASE2D = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
     [0, 0, 1, 1, 0, 2, 2, 0, 0, 0, 0],
     [0, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
)

EDGECASE2D_EXPANDED_4 = np.array(
      [[1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0],
       [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
       [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
       [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
       [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0]])

SAMPLE3D = np.array(
      [[[0, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]],

       [[0, 0, 0, 0],
        [0, 3, 3, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]],

       [[0, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 5, 0]],

       [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 5, 0]]])

SAMPLE3D_EXPANDED_2 =np.array(
      [[[3, 3, 3, 3],
        [3, 3, 3, 3],
        [3, 3, 3, 3],
        [0, 3, 5, 0]],

       [[3, 3, 3, 3],
        [3, 3, 3, 3],
        [3, 3, 3, 3],
        [0, 5, 5, 5]],

       [[3, 3, 3, 3],
        [3, 3, 3, 3],
        [3, 3, 5, 5],
        [5, 5, 5, 5]],

       [[3, 3, 3, 0],
        [3, 3, 3, 0],
        [3, 3, 5, 5],
        [5, 5, 5, 5]]])

SAMPLE_EDGECASE_BEHAVIOUR = np.array([[0, 1, 0, 0], [2, 0, 0, 0], [0, 3, 0, 0]])

@testing.parametrize(
    "input_array, expected_output, expand_distance", 
    [
    (SAMPLE1D, SAMPLE1D_EXPANDED_3, 3),
    (SAMPLE2D, SAMPLE2D_EXPANDED_3, 3),
    (SAMPLE2D, SAMPLE2D_EXPANDED_1_5, 1.5),
    (EDGECASE1D, EDGECASE1D_EXPANDED_3, 3),
    (EDGECASE2D, EDGECASE2D_EXPANDED_4, 4),
    (SAMPLE3D, SAMPLE3D_EXPANDED_2, 2)
    ]
)
def test_expand_labels(input_array, expected_output, expand_distance):
    expanded = expand_labels(input_array, expand_distance)
    assert_array_equal(expanded, expected_output)


@testing.parametrize('ndim', [2, 3])
@testing.parametrize('distance', range(6))
def test_binary_blobs(ndim, distance):
    """Check some invariants with label expansion.

    - New labels array should exactly contain the original labels array.
    - Distance to old labels array within new labels should never exceed input
      distance.
    - Distance beyond the expanded labels should always exceed the input
      distance.
    """
    img = data.binary_blobs(length=64, blob_size_fraction=0.05, n_dim=ndim)
    labels = measure.label(img)
    expanded = expand_labels(labels, distance=distance)
    original_mask = labels != 0
    assert_array_equal(labels[original_mask], expanded[original_mask])
    expanded_only_mask = (expanded - labels).astype(bool)
    distance_map = ndi.distance_transform_edt(~original_mask)
    expanded_distances = distance_map[expanded_only_mask]
    if expanded_distances.size > 0:
        assert np.all(expanded_distances <= distance)
    beyond_expanded_distances = distance_map[~expanded.astype(bool)]
    if beyond_expanded_distances.size > 0:
        assert np.all(beyond_expanded_distances > distance)


def test_edge_case_behaviour():
    """ Check edge case behavior to detect upstream changes

    For edge cases where a pixel has the same distance to several regions,
    lexicographical order seems to determine which region gets to expand
    into this pixel given the current upstream behaviour in
    scipy.ndimage.distance_map_edt.

    As a result, we expect different results when transposing the array.
    If this test fails, something has changed upstream.
    """
    expanded = expand_labels(SAMPLE_EDGECASE_BEHAVIOUR, 1)
    expanded_transpose = expand_labels(SAMPLE_EDGECASE_BEHAVIOUR.T, 1)
    assert not np.all(expanded == expanded_transpose.T)
    
