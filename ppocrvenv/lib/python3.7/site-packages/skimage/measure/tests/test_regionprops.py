import math

import numpy as np
import pytest
import scipy.ndimage as ndi
from numpy.testing import (assert_allclose, assert_almost_equal,
                           assert_array_almost_equal, assert_array_equal,
                           assert_equal)

from skimage import data, draw, transform
from skimage._shared._warnings import expected_warnings
from skimage.measure._regionprops import (COL_DTYPES, OBJECT_COLUMNS, PROPS,
                                          _inertia_eigvals_to_axes_lengths_3D,
                                          _parse_docs, _props_to_dict,
                                          euler_number, perimeter,
                                          perimeter_crofton, regionprops,
                                          regionprops_table)
from skimage.segmentation import slic

SAMPLE = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
     [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
     [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
)
INTENSITY_SAMPLE = SAMPLE.copy()
INTENSITY_SAMPLE[1, 9:11] = 2
INTENSITY_FLOAT_SAMPLE = INTENSITY_SAMPLE.copy().astype(np.float64) / 10.0

SAMPLE_MULTIPLE = np.eye(10, dtype=np.int32)
SAMPLE_MULTIPLE[3:5, 7:8] = 2
INTENSITY_SAMPLE_MULTIPLE = SAMPLE_MULTIPLE.copy() * 2.0

SAMPLE_3D = np.zeros((6, 6, 6), dtype=np.uint8)
SAMPLE_3D[1:3, 1:3, 1:3] = 1
SAMPLE_3D[3, 2, 2] = 1
INTENSITY_SAMPLE_3D = SAMPLE_3D.copy()


def test_all_props():
    region = regionprops(SAMPLE, INTENSITY_SAMPLE)[0]
    for prop in PROPS:
        try:
            # access legacy name via dict
            assert_almost_equal(region[prop], getattr(region, PROPS[prop]))

            # skip property access tests for old CamelCase names
            # (we intentionally do not provide properties for these)
            if prop.lower() == prop:
                # access legacy name via attribute
                assert_almost_equal(getattr(region, prop),
                                    getattr(region, PROPS[prop]))

        except TypeError:  # the `slice` property causes this
            pass


def test_all_props_3d():
    region = regionprops(SAMPLE_3D, INTENSITY_SAMPLE_3D)[0]
    for prop in PROPS:
        try:
            assert_almost_equal(region[prop], getattr(region, PROPS[prop]))

            # skip property access tests for old CamelCase names
            # (we intentionally do not provide properties for these)
            if prop.lower() == prop:
                assert_almost_equal(getattr(region, prop),
                                    getattr(region, PROPS[prop]))

        except (NotImplementedError, TypeError):
            pass


def test_dtype():
    regionprops(np.zeros((10, 10), dtype=int))
    regionprops(np.zeros((10, 10), dtype=np.uint))
    with pytest.raises(TypeError):
        regionprops(np.zeros((10, 10), dtype=float))
    with pytest.raises(TypeError):
        regionprops(np.zeros((10, 10), dtype=np.double))
    with pytest.raises(TypeError):
        regionprops(np.zeros((10, 10), dtype=bool))


def test_ndim():
    regionprops(np.zeros((10, 10), dtype=int))
    regionprops(np.zeros((10, 10, 1), dtype=int))
    regionprops(np.zeros((10, 10, 10), dtype=int))
    regionprops(np.zeros((1, 1), dtype=int))
    regionprops(np.zeros((1, 1, 1), dtype=int))
    with pytest.raises(TypeError):
        regionprops(np.zeros((10, 10, 10, 2), dtype=int))


def test_feret_diameter_max():
    # comparator result is based on SAMPLE from manually-inspected computations
    comparator_result = 18
    test_result = regionprops(SAMPLE)[0].feret_diameter_max
    assert np.abs(test_result - comparator_result) < 1
    # square, test that maximum Feret diameter is sqrt(2) * square side
    img = np.zeros((20, 20), dtype=np.uint8)
    img[2:-2, 2:-2] = 1
    feret_diameter_max = regionprops(img)[0].feret_diameter_max
    assert np.abs(feret_diameter_max - 16 * np.sqrt(2)) < 1


def test_feret_diameter_max_3d():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[2:-2, 2:-2] = 1
    img_3d = np.dstack((img,) * 3)
    feret_diameter_max = regionprops(img_3d)[0].feret_diameter_max
    assert np.abs(feret_diameter_max - 16 * np.sqrt(2)) < 1


def test_area():
    area = regionprops(SAMPLE)[0].area
    assert area == np.sum(SAMPLE)
    area = regionprops(SAMPLE_3D)[0].area
    assert area == np.sum(SAMPLE_3D)


def test_bbox():
    bbox = regionprops(SAMPLE)[0].bbox
    assert_array_almost_equal(bbox, (0, 0, SAMPLE.shape[0], SAMPLE.shape[1]))

    SAMPLE_mod = SAMPLE.copy()
    SAMPLE_mod[:, -1] = 0
    bbox = regionprops(SAMPLE_mod)[0].bbox
    assert_array_almost_equal(bbox, (0, 0, SAMPLE.shape[0], SAMPLE.shape[1]-1))

    bbox = regionprops(SAMPLE_3D)[0].bbox
    assert_array_almost_equal(bbox, (1, 1, 1, 4, 3, 3))


def test_area_bbox():
    padded = np.pad(SAMPLE, 5, mode='constant')
    bbox_area = regionprops(padded)[0].area_bbox
    assert_array_almost_equal(bbox_area, SAMPLE.size)


def test_moments_central():
    mu = regionprops(SAMPLE)[0].moments_central
    # determined with OpenCV
    assert_almost_equal(mu[2, 0], 436.00000000000045)
    # different from OpenCV results, bug in OpenCV
    assert_almost_equal(mu[3, 0], -737.333333333333)
    assert_almost_equal(mu[1, 1], -87.33333333333303)
    assert_almost_equal(mu[2, 1], -127.5555555555593)
    assert_almost_equal(mu[0, 2], 1259.7777777777774)
    assert_almost_equal(mu[1, 2], 2000.296296296291)
    assert_almost_equal(mu[0, 3], -760.0246913580195)


def test_centroid():
    centroid = regionprops(SAMPLE)[0].centroid
    # determined with MATLAB
    assert_array_almost_equal(centroid, (5.66666666666666, 9.444444444444444))


def test_centroid_3d():
    centroid = regionprops(SAMPLE_3D)[0].centroid
    # determined by mean along axis 1 of SAMPLE_3D.nonzero()
    assert_array_almost_equal(centroid, (1.66666667, 1.55555556, 1.55555556))


def test_area_convex():
    area = regionprops(SAMPLE)[0].area_convex
    assert area == 125


def test_image_convex():
    img = regionprops(SAMPLE)[0].image_convex
    ref = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    )
    assert_array_equal(img, ref)


def test_coordinates():
    sample = np.zeros((10, 10), dtype=np.int8)
    coords = np.array([[3, 2], [3, 3], [3, 4]])
    sample[coords[:, 0], coords[:, 1]] = 1
    prop_coords = regionprops(sample)[0].coords
    assert_array_equal(prop_coords, coords)

    sample = np.zeros((6, 6, 6), dtype=np.int8)
    coords = np.array([[1, 1, 1], [1, 2, 1], [1, 3, 1]])
    sample[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    prop_coords = regionprops(sample)[0].coords
    assert_array_equal(prop_coords, coords)


def test_slice():
    padded = np.pad(SAMPLE, ((2, 4), (5, 2)), mode='constant')
    nrow, ncol = SAMPLE.shape
    result = regionprops(padded)[0].slice
    expected = (slice(2, 2+nrow), slice(5, 5+ncol))
    assert_equal(result, expected)


def test_eccentricity():
    eps = regionprops(SAMPLE)[0].eccentricity
    assert_almost_equal(eps, 0.814629313427)

    img = np.zeros((5, 5), dtype=int)
    img[2, 2] = 1
    eps = regionprops(img)[0].eccentricity
    assert_almost_equal(eps, 0)


def test_equivalent_diameter_area():
    diameter = regionprops(SAMPLE)[0].equivalent_diameter_area
    # determined with MATLAB
    assert_almost_equal(diameter, 9.57461472963)


def test_euler_number():
    en = regionprops(SAMPLE)[0].euler_number
    assert en == 0

    SAMPLE_mod = SAMPLE.copy()
    SAMPLE_mod[7, -3] = 0
    en = regionprops(SAMPLE_mod)[0].euler_number
    assert en == -1

    en = euler_number(SAMPLE, 1)
    assert en == 2

    en = euler_number(SAMPLE_mod, 1)
    assert en == 1

    en = euler_number(SAMPLE_3D, 1)
    assert en == 1

    en = euler_number(SAMPLE_3D, 3)
    assert en == 1

    # for convex body, Euler number is 1
    SAMPLE_3D_2 = np.zeros((100, 100, 100))
    SAMPLE_3D_2[40:60, 40:60, 40:60] = 1
    en = euler_number(SAMPLE_3D_2, 3)
    assert en == 1

    SAMPLE_3D_2[45:55, 45:55, 45:55] = 0
    en = euler_number(SAMPLE_3D_2, 3)
    assert en == 2


def test_extent():
    extent = regionprops(SAMPLE)[0].extent
    assert_almost_equal(extent, 0.4)


def test_moments_hu():
    hu = regionprops(SAMPLE)[0].moments_hu
    ref = np.array([
        3.27117627e-01,
        2.63869194e-02,
        2.35390060e-02,
        1.23151193e-03,
        1.38882330e-06,
        -2.72586158e-05,
        -6.48350653e-06
    ])
    # bug in OpenCV caused in Central Moments calculation?
    assert_array_almost_equal(hu, ref)


def test_image():
    img = regionprops(SAMPLE)[0].image
    assert_array_equal(img, SAMPLE)

    img = regionprops(SAMPLE_3D)[0].image
    assert_array_equal(img, SAMPLE_3D[1:4, 1:3, 1:3])


def test_label():
    label = regionprops(SAMPLE)[0].label
    assert_array_equal(label, 1)

    label = regionprops(SAMPLE_3D)[0].label
    assert_array_equal(label, 1)


def test_area_filled():
    area = regionprops(SAMPLE)[0].area_filled
    assert area == np.sum(SAMPLE)

    SAMPLE_mod = SAMPLE.copy()
    SAMPLE_mod[7, -3] = 0
    area = regionprops(SAMPLE_mod)[0].area_filled
    assert area == np.sum(SAMPLE)


def test_image_filled():
    img = regionprops(SAMPLE)[0].image_filled
    assert_array_equal(img, SAMPLE)


def test_axis_major_length():
    length = regionprops(SAMPLE)[0].axis_major_length
    # MATLAB has different interpretation of ellipse than found in literature,
    # here implemented as found in literature
    assert_almost_equal(length, 16.7924234999)


def test_intensity_max():
    intensity = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE
                            )[0].intensity_max
    assert_almost_equal(intensity, 2)


def test_intensity_mean():
    intensity = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE
                            )[0].intensity_mean
    assert_almost_equal(intensity, 1.02777777777777)


def test_intensity_min():
    intensity = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE
                            )[0].intensity_min
    assert_almost_equal(intensity, 1)


def test_axis_minor_length():
    length = regionprops(SAMPLE)[0].axis_minor_length
    # MATLAB has different interpretation of ellipse than found in literature,
    # here implemented as found in literature
    assert_almost_equal(length, 9.739302807263)


def test_moments():
    m = regionprops(SAMPLE)[0].moments
    # determined with OpenCV
    assert_almost_equal(m[0, 0], 72.0)
    assert_almost_equal(m[0, 1], 680.0)
    assert_almost_equal(m[0, 2], 7682.0)
    assert_almost_equal(m[0, 3], 95588.0)
    assert_almost_equal(m[1, 0], 408.0)
    assert_almost_equal(m[1, 1], 3766.0)
    assert_almost_equal(m[1, 2], 43882.0)
    assert_almost_equal(m[2, 0], 2748.0)
    assert_almost_equal(m[2, 1], 24836.0)
    assert_almost_equal(m[3, 0], 19776.0)


def test_moments_normalized():
    nu = regionprops(SAMPLE)[0].moments_normalized

    # determined with OpenCV
    assert_almost_equal(nu[0, 2], 0.24301268861454037)
    assert_almost_equal(nu[0, 3], -0.017278118992041805)
    assert_almost_equal(nu[1, 1], -0.016846707818929982)
    assert_almost_equal(nu[1, 2], 0.045473992910668816)
    assert_almost_equal(nu[2, 0], 0.08410493827160502)
    assert_almost_equal(nu[2, 1], -0.002899800614433943)


def test_orientation():
    orient = regionprops(SAMPLE)[0].orientation
    # determined with MATLAB
    assert_almost_equal(orient, -1.4663278802756865)
    # test diagonal regions
    diag = np.eye(10, dtype=int)
    orient_diag = regionprops(diag)[0].orientation
    assert_almost_equal(orient_diag, -math.pi / 4)
    orient_diag = regionprops(np.flipud(diag))[0].orientation
    assert_almost_equal(orient_diag, math.pi / 4)
    orient_diag = regionprops(np.fliplr(diag))[0].orientation
    assert_almost_equal(orient_diag, math.pi / 4)
    orient_diag = regionprops(np.fliplr(np.flipud(diag)))[0].orientation
    assert_almost_equal(orient_diag, -math.pi / 4)


def test_perimeter():
    per = regionprops(SAMPLE)[0].perimeter
    assert_almost_equal(per, 55.2487373415)

    per = perimeter(SAMPLE.astype('double'), neighbourhood=8)
    assert_almost_equal(per, 46.8284271247)


def test_perimeter_crofton():
    per = regionprops(SAMPLE)[0].perimeter_crofton
    assert_almost_equal(per, 61.0800637973)

    per = perimeter_crofton(SAMPLE.astype('double'), directions=2)
    assert_almost_equal(per, 64.4026493985)


def test_solidity():
    solidity = regionprops(SAMPLE)[0].solidity
    assert_almost_equal(solidity, 0.576)


def test_moments_weighted_central():
    wmu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE
                      )[0].moments_weighted_central
    ref = np.array(
        [[7.4000000000e+01, 3.7303493627e-14, 1.2602837838e+03,
          -7.6561796932e+02],
         [-2.1316282073e-13, -8.7837837838e+01, 2.1571526662e+03,
          -4.2385971907e+03],
         [4.7837837838e+02, -1.4801314828e+02, 6.6989799420e+03,
          -9.9501164076e+03],
         [-7.5943608473e+02, -1.2714707125e+03, 1.5304076361e+04,
          -3.3156729271e+04]])

    np.set_printoptions(precision=10)
    assert_array_almost_equal(wmu, ref)


def test_centroid_weighted():
    centroid = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE
                           )[0].centroid_weighted
    assert_array_almost_equal(centroid, (5.540540540540, 9.445945945945))


def test_moments_weighted_hu():
    whu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE
                      )[0].moments_weighted_hu
    ref = np.array([
        3.1750587329e-01,
        2.1417517159e-02,
        2.3609322038e-02,
        1.2565683360e-03,
        8.3014209421e-07,
        -3.5073773473e-05,
        -6.7936409056e-06
    ])
    assert_array_almost_equal(whu, ref)


def test_moments_weighted():
    wm = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE
                     )[0].moments_weighted
    ref = np.array(
        [[7.4000000e+01, 6.9900000e+02, 7.8630000e+03, 9.7317000e+04],
         [4.1000000e+02, 3.7850000e+03, 4.4063000e+04, 5.7256700e+05],
         [2.7500000e+03, 2.4855000e+04, 2.9347700e+05, 3.9007170e+06],
         [1.9778000e+04, 1.7500100e+05, 2.0810510e+06, 2.8078871e+07]]
    )
    assert_array_almost_equal(wm, ref)


def test_moments_weighted_normalized():
    wnu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE
                      )[0].moments_weighted_normalized
    ref = np.array(
        [[np.nan,        np.nan, 0.2301467830, -0.0162529732],
         [np.nan, -0.0160405109, 0.0457932622, -0.0104598869],
         [0.0873590903, -0.0031421072, 0.0165315478, -0.0028544152],
         [-0.0161217406, -0.0031376984, 0.0043903193, -0.0011057191]]
    )
    assert_array_almost_equal(wnu, ref)


def test_label_sequence():
    a = np.empty((2, 2), dtype=int)
    a[:, :] = 2
    ps = regionprops(a)
    assert len(ps) == 1
    assert ps[0].label == 2


def test_pure_background():
    a = np.zeros((2, 2), dtype=int)
    ps = regionprops(a)
    assert len(ps) == 0


def test_invalid():
    ps = regionprops(SAMPLE)

    def get_intensity_image():
        ps[0].image_intensity

    with pytest.raises(AttributeError):
        get_intensity_image()


def test_invalid_size():
    wrong_intensity_sample = np.array([[1], [1]])
    with pytest.raises(ValueError):
        regionprops(SAMPLE, wrong_intensity_sample)


def test_equals():
    arr = np.zeros((100, 100), dtype=int)
    arr[0:25, 0:25] = 1
    arr[50:99, 50:99] = 2

    regions = regionprops(arr)
    r1 = regions[0]

    regions = regionprops(arr)
    r2 = regions[0]
    r3 = regions[1]

    assert_equal(r1 == r2, True, "Same regionprops are not equal")
    assert_equal(r1 != r3, True, "Different regionprops are equal")


def test_iterate_all_props():
    region = regionprops(SAMPLE)[0]
    p0 = {p: region[p] for p in region}

    region = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0]
    p1 = {p: region[p] for p in region}

    assert len(p0) < len(p1)


def test_cache():
    SAMPLE_mod = SAMPLE.copy()
    region = regionprops(SAMPLE_mod)[0]
    f0 = region.image_filled
    region._label_image[:10] = 1
    f1 = region.image_filled

    # Changed underlying image, but cache keeps result the same
    assert_array_equal(f0, f1)

    # Now invalidate cache
    region._cache_active = False
    f1 = region.image_filled

    assert np.any(f0 != f1)


def test_docstrings_and_props():
    def foo():
        """foo"""

    has_docstrings = bool(foo.__doc__)

    region = regionprops(SAMPLE)[0]

    docs = _parse_docs()
    props = [m for m in dir(region) if not m.startswith('_')]

    nr_docs_parsed = len(docs)
    nr_props = len(props)
    if has_docstrings:
        assert_equal(nr_docs_parsed, nr_props)
        ds = docs['moments_weighted_normalized']
        assert 'iteration' not in ds
        assert len(ds.split('\n')) > 3
    else:
        assert_equal(nr_docs_parsed, 0)


def test_props_to_dict():
    regions = regionprops(SAMPLE)
    out = _props_to_dict(regions)
    assert out == {'label': np.array([1]),
                   'bbox-0': np.array([0]), 'bbox-1': np.array([0]),
                   'bbox-2': np.array([10]), 'bbox-3': np.array([18])}

    regions = regionprops(SAMPLE)
    out = _props_to_dict(regions, properties=('label', 'area', 'bbox'),
                         separator='+')
    assert out == {'label': np.array([1]), 'area': np.array([72]),
                   'bbox+0': np.array([0]), 'bbox+1': np.array([0]),
                   'bbox+2': np.array([10]), 'bbox+3': np.array([18])}


def test_regionprops_table():
    out = regionprops_table(SAMPLE)
    assert out == {'label': np.array([1]),
                   'bbox-0': np.array([0]), 'bbox-1': np.array([0]),
                   'bbox-2': np.array([10]), 'bbox-3': np.array([18])}

    out = regionprops_table(SAMPLE, properties=('label', 'area', 'bbox'),
                            separator='+')
    assert out == {'label': np.array([1]), 'area': np.array([72]),
                   'bbox+0': np.array([0]), 'bbox+1': np.array([0]),
                   'bbox+2': np.array([10]), 'bbox+3': np.array([18])}


def test_regionprops_table_deprecated_vector_property():
    out = regionprops_table(SAMPLE, properties=('local_centroid',))
    for key in out.keys():
        # key reflects the deprecated name, not its new (centroid_local) value
        assert key.startswith('local_centroid')


def test_regionprops_table_deprecated_scalar_property():
    out = regionprops_table(SAMPLE, properties=('bbox_area',))
    assert list(out.keys()) == ['bbox_area']


def test_regionprops_table_equal_to_original():
    regions = regionprops(SAMPLE, INTENSITY_FLOAT_SAMPLE)
    out_table = regionprops_table(SAMPLE, INTENSITY_FLOAT_SAMPLE,
                                  properties=COL_DTYPES.keys())

    for prop, dtype in COL_DTYPES.items():
        for i, reg in enumerate(regions):
            rp = reg[prop]
            if np.isscalar(rp) or \
                    prop in OBJECT_COLUMNS or \
                    dtype is np.object_:
                assert_array_equal(rp, out_table[prop][i])
            else:
                shape = rp.shape if isinstance(rp, np.ndarray) else (len(rp),)
                for ind in np.ndindex(shape):
                    modified_prop = "-".join(map(str, (prop,) + ind))
                    loc = ind if len(ind) > 1 else ind[0]
                    assert_equal(rp[loc], out_table[modified_prop][i])


def test_regionprops_table_no_regions():
    out = regionprops_table(np.zeros((2, 2), dtype=int),
                            properties=('label', 'area', 'bbox'),
                            separator='+')
    assert len(out) == 6
    assert len(out['label']) == 0
    assert len(out['area']) == 0
    assert len(out['bbox+0']) == 0
    assert len(out['bbox+1']) == 0
    assert len(out['bbox+2']) == 0
    assert len(out['bbox+3']) == 0


def test_column_dtypes_complete():
    assert set(COL_DTYPES.keys()).union(OBJECT_COLUMNS) == set(PROPS.values())


def test_column_dtypes_correct():
    msg = 'mismatch with expected type,'
    region = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0]
    for col in COL_DTYPES:
        r = region[col]

        if col in OBJECT_COLUMNS:
            assert COL_DTYPES[col] == object
            continue

        t = type(np.ravel(r)[0])

        if np.issubdtype(t, np.floating):
            assert COL_DTYPES[col] == float, (
                f'{col} dtype {t} {msg} {COL_DTYPES[col]}'
            )
        elif np.issubdtype(t, np.integer):
            assert COL_DTYPES[col] == int, (
                f'{col} dtype {t} {msg} {COL_DTYPES[col]}'
            )
        else:
            assert False, (
                f'{col} dtype {t} {msg} {COL_DTYPES[col]}'
            )


def test_deprecated_coords_argument():
    with expected_warnings(['coordinates keyword argument']):
        regionprops(SAMPLE, coordinates='rc')
    with pytest.raises(ValueError):
        regionprops(SAMPLE, coordinates='xy')


def pixelcount(regionmask):
    """a short test for an extra property"""
    return np.sum(regionmask)


def intensity_median(regionmask, image_intensity):
    return np.median(image_intensity[regionmask])


def too_many_args(regionmask, image_intensity, superfluous):
    return 1


def too_few_args():
    return 1


def test_extra_properties():
    region = regionprops(SAMPLE, extra_properties=(pixelcount,))[0]
    assert region.pixelcount == np.sum(SAMPLE == 1)


def test_extra_properties_intensity():
    region = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE,
                         extra_properties=(intensity_median,)
                         )[0]
    assert region.intensity_median == np.median(INTENSITY_SAMPLE[SAMPLE == 1])


def test_extra_properties_no_intensity_provided():
    with pytest.raises(AttributeError):
        region = regionprops(SAMPLE, extra_properties=(intensity_median,))[0]
        _ = region.intensity_median


def test_extra_properties_nr_args():
    with pytest.raises(AttributeError):
        region = regionprops(SAMPLE, extra_properties=(too_few_args,))[0]
        _ = region.too_few_args
    with pytest.raises(AttributeError):
        region = regionprops(SAMPLE, extra_properties=(too_many_args,))[0]
        _ = region.too_many_args


def test_extra_properties_mixed():
    # mixed properties, with and without intensity
    region = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE,
                         extra_properties=(intensity_median, pixelcount)
                         )[0]
    assert region.intensity_median == np.median(INTENSITY_SAMPLE[SAMPLE == 1])
    assert region.pixelcount == np.sum(SAMPLE == 1)


def test_extra_properties_table():
    out = regionprops_table(SAMPLE_MULTIPLE,
                            intensity_image=INTENSITY_SAMPLE_MULTIPLE,
                            properties=('label',),
                            extra_properties=(intensity_median, pixelcount)
                            )
    assert_array_almost_equal(out['intensity_median'], np.array([2., 4.]))
    assert_array_equal(out['pixelcount'], np.array([10, 2]))


def test_multichannel():
    """Test that computing multichannel properties works."""
    astro = data.astronaut()[::4, ::4]
    astro_green = astro[..., 1]
    labels = slic(astro.astype(float), start_label=1)

    segment_idx = np.max(labels) // 2
    region = regionprops(labels,
                         astro_green,
                         extra_properties=[intensity_median]
                         )[segment_idx]
    region_multi = regionprops(labels,
                               astro,
                               extra_properties=[intensity_median]
                               )[segment_idx]

    for prop in list(PROPS.keys()) + ["intensity_median"]:
        p = region[prop]
        p_multi = region_multi[prop]
        if np.shape(p) == np.shape(p_multi):
            # property does not depend on multiple channels
            assert_array_equal(p, p_multi)
        else:
            # property uses multiple channels, returns props stacked along
            # final axis
            assert_allclose(p, np.asarray(p_multi)[..., 1], rtol=1e-12,
                            atol=1e-12)


def test_3d_ellipsoid_axis_lengths():
    """Verify that estimated axis lengths are correct.

    Uses an ellipsoid at an arbitrary position and orientation.
    """
    # generate a centered ellipsoid with non-uniform half-lengths (radii)
    half_lengths = (20, 10, 50)
    e = draw.ellipsoid(*half_lengths).astype(int)

    # Pad by asymmetric amounts so the ellipse isn't centered. Also, pad enough
    # that the rotated ellipse will still be within the original volume.
    e = np.pad(e, pad_width=[(30, 18), (30, 12), (40, 20)], mode='constant')

    # apply rotations to the ellipsoid
    R = transform.EuclideanTransform(rotation=[0.2, 0.3, 0.4],
                                     dimensionality=3)
    e = ndi.affine_transform(e, R.params)

    # Compute regionprops
    rp = regionprops(e)[0]

    # estimate principal axis lengths via the inertia tensor eigenvalues
    evs = rp.inertia_tensor_eigvals
    axis_lengths = _inertia_eigvals_to_axes_lengths_3D(evs)
    expected_lengths = sorted([2 * h for h in half_lengths], reverse=True)
    for ax_len_expected, ax_len in zip(expected_lengths, axis_lengths):
        # verify accuracy to within 1%
        assert abs(ax_len - ax_len_expected) < 0.01 * ax_len_expected

    # verify that the axis length regionprops also agree
    assert abs(rp.axis_major_length - axis_lengths[0]) < 1e-7
    assert abs(rp.axis_minor_length - axis_lengths[-1]) < 1e-7
