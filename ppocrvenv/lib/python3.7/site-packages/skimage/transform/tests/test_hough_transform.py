import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal

from skimage._shared.testing import test_parallel
from skimage import data
from skimage import transform
from skimage.draw import line, circle_perimeter, ellipse_perimeter

@test_parallel()
def test_hough_line():
    # Generate a test image
    img = np.zeros((100, 150), dtype=int)
    rr, cc = line(60, 130, 80, 10)
    img[rr, cc] = 1

    out, angles, d = transform.hough_line(img)

    y, x = np.where(out == out.max())
    dist = d[y[0]]
    theta = angles[x[0]]

    assert_almost_equal(dist, 80.0, 1)
    assert_almost_equal(theta, 1.41, 1)


def test_hough_line_angles():
    img = np.zeros((10, 10))
    img[0, 0] = 1

    out, angles, d = transform.hough_line(img, np.linspace(0, 360, 10))

    assert_equal(len(angles), 10)


def test_hough_line_bad_input():
    img = np.zeros(100)
    img[10] = 1

    # Expected error, img must be 2D
    with pytest.raises(ValueError):
        transform.hough_line(img, np.linspace(0, 360, 10))


def test_probabilistic_hough():
    # Generate a test image
    img = np.zeros((100, 100), dtype=int)
    for i in range(25, 75):
        img[100 - i, i] = 100
        img[i, i] = 100

    # decrease default theta sampling because similar orientations may confuse
    # as mentioned in article of Galambos et al
    theta = np.linspace(0, np.pi, 45)
    lines = transform.probabilistic_hough_line(
        img, threshold=10, line_length=10, line_gap=1, theta=theta)
    # sort the lines according to the x-axis
    sorted_lines = []
    for line in lines:
        line = list(line)
        line.sort(key=lambda x: x[0])
        sorted_lines.append(line)

    assert([(25, 75), (74, 26)] in sorted_lines)
    assert([(25, 25), (74, 74)] in sorted_lines)

    # Execute with default theta
    transform.probabilistic_hough_line(img, line_length=10, line_gap=3)


def test_probabilistic_hough_seed():
    # Load image that is likely to give a randomly varying number of lines
    image = data.checkerboard()

    # Use constant seed to ensure a deterministic output
    lines = transform.probabilistic_hough_line(image, threshold=50,
                                               line_length=50, line_gap=1,
                                               seed=41537233)
    assert len(lines) == 56


def test_probabilistic_hough_bad_input():
    img = np.zeros(100)
    img[10] = 1

    # Expected error, img must be 2D
    with pytest.raises(ValueError):
        transform.probabilistic_hough_line(img)


def test_hough_line_peaks():
    img = np.zeros((100, 150), dtype=int)
    rr, cc = line(60, 130, 80, 10)
    img[rr, cc] = 1

    out, angles, d = transform.hough_line(img)

    out, theta, dist = transform.hough_line_peaks(out, angles, d)

    assert_equal(len(dist), 1)
    assert_almost_equal(dist[0], 81.0, 1)
    assert_almost_equal(theta[0], 1.41, 1)


def test_hough_line_peaks_ordered():
    # Regression test per PR #1421
    testim = np.zeros((256, 64), dtype=bool)

    testim[50:100, 20] = True
    testim[20:225, 25] = True
    testim[15:35, 50] = True
    testim[1:-1, 58] = True

    hough_space, angles, dists = transform.hough_line(testim)

    hspace, _, _ = transform.hough_line_peaks(hough_space, angles, dists)
    assert hspace[0] > hspace[1]


def test_hough_line_peaks_dist():
    img = np.zeros((100, 100), dtype=bool)
    img[:, 30] = True
    img[:, 40] = True
    hspace, angles, dists = transform.hough_line(img)
    assert len(transform.hough_line_peaks(hspace, angles, dists,
                                          min_distance=5)[0]) == 2
    assert len(transform.hough_line_peaks(hspace, angles, dists,
                                          min_distance=15)[0]) == 1


def test_hough_line_peaks_angle():
    check_hough_line_peaks_angle()


def check_hough_line_peaks_angle():
    img = np.zeros((100, 100), dtype=bool)
    img[:, 0] = True
    img[0, :] = True

    hspace, angles, dists = transform.hough_line(img)
    assert len(transform.hough_line_peaks(hspace, angles, dists,
                                          min_angle=45)[0]) == 2
    assert len(transform.hough_line_peaks(hspace, angles, dists,
                                          min_angle=90)[0]) == 1

    theta = np.linspace(0, np.pi, 100)
    hspace, angles, dists = transform.hough_line(img, theta)
    assert len(transform.hough_line_peaks(hspace, angles, dists,
                                          min_angle=45)[0]) == 2
    assert len(transform.hough_line_peaks(hspace, angles, dists,
                                          min_angle=90)[0]) == 1

    theta = np.linspace(np.pi / 3, 4. / 3 * np.pi, 100)
    hspace, angles, dists = transform.hough_line(img, theta)
    assert len(transform.hough_line_peaks(hspace, angles, dists,
                                          min_angle=45)[0]) == 2
    assert len(transform.hough_line_peaks(hspace, angles, dists,
                                          min_angle=90)[0]) == 1


def test_hough_line_peaks_num():
    img = np.zeros((100, 100), dtype=bool)
    img[:, 30] = True
    img[:, 40] = True
    hspace, angles, dists = transform.hough_line(img)
    assert len(transform.hough_line_peaks(hspace, angles, dists,
                                          min_distance=0, min_angle=0,
                                          num_peaks=1)[0]) == 1


def test_hough_line_peaks_zero_input():
    # Test to make sure empty input doesn't cause a failure
    img = np.zeros((100, 100), dtype='uint8')
    theta = np.linspace(0, np.pi, 100)
    hspace, angles, dists = transform.hough_line(img, theta)
    h, a, d = transform.hough_line_peaks(hspace, angles, dists)
    assert_equal(a, np.array([]))


def test_hough_line_peaks_single_angle():
    # Regression test for gh-4814
    # This code snippet used to raise an IndexError
    img = np.random.random((100, 100))
    tested_angles = np.array([np.pi / 2])
    h, theta, d = transform.hough_line(img, theta=tested_angles)
    accum, angles, dists = transform.hough_line_peaks(h, theta, d, threshold=2)


@test_parallel()
def test_hough_circle():
    # Prepare picture
    img = np.zeros((120, 100), dtype=int)
    radius = 20
    x_0, y_0 = (99, 50)
    y, x = circle_perimeter(y_0, x_0, radius)
    img[x, y] = 1

    out1 = transform.hough_circle(img, radius)
    out2 = transform.hough_circle(img, [radius])
    assert_equal(out1, out2)
    out = transform.hough_circle(img, np.array([radius], dtype=np.intp))
    assert_equal(out, out1)
    x, y = np.where(out[0] == out[0].max())
    assert_equal(x[0], x_0)
    assert_equal(y[0], y_0)


def test_hough_circle_extended():
    # Prepare picture
    # The circle center is outside the image
    img = np.zeros((100, 100), dtype=int)
    radius = 20
    x_0, y_0 = (-5, 50)
    y, x = circle_perimeter(y_0, x_0, radius)
    img[x[np.where(x > 0)], y[np.where(x > 0)]] = 1

    out = transform.hough_circle(img, np.array([radius], dtype=np.intp),
                                 full_output=True)

    x, y = np.where(out[0] == out[0].max())
    # Offset for x_0, y_0
    assert_equal(x[0], x_0 + radius)
    assert_equal(y[0], y_0 + radius)


def test_hough_circle_peaks():
    x_0, y_0, rad_0 = (99, 50, 20)
    img = np.zeros((120, 100), dtype=int)
    y, x = circle_perimeter(y_0, x_0, rad_0)
    img[x, y] = 1

    x_1, y_1, rad_1 = (49, 60, 30)
    y, x = circle_perimeter(y_1, x_1, rad_1)
    img[x, y] = 1

    radii = [rad_0, rad_1]
    hspaces = transform.hough_circle(img, radii)
    out = transform.hough_circle_peaks(hspaces, radii, min_xdistance=1,
                                       min_ydistance=1, threshold=None,
                                       num_peaks=np.inf,
                                       total_num_peaks=np.inf)
    s = np.argsort(out[3])  # sort by radii
    assert_equal(out[1][s], np.array([y_0, y_1]))
    assert_equal(out[2][s], np.array([x_0, x_1]))
    assert_equal(out[3][s], np.array([rad_0, rad_1]))


def test_hough_circle_peaks_total_peak():
    img = np.zeros((120, 100), dtype=int)

    x_0, y_0, rad_0 = (99, 50, 20)
    y, x = circle_perimeter(y_0, x_0, rad_0)
    img[x, y] = 1

    x_1, y_1, rad_1 = (49, 60, 30)
    y, x = circle_perimeter(y_1, x_1, rad_1)
    img[x, y] = 1

    radii = [rad_0, rad_1]
    hspaces = transform.hough_circle(img, radii)
    out = transform.hough_circle_peaks(hspaces, radii, min_xdistance=1,
                                       min_ydistance=1, threshold=None,
                                       num_peaks=np.inf, total_num_peaks=1)
    assert_equal(out[1][0], np.array([y_1, ]))
    assert_equal(out[2][0], np.array([x_1, ]))
    assert_equal(out[3][0], np.array([rad_1, ]))


def test_hough_circle_peaks_min_distance():
    x_0, y_0, rad_0 = (50, 50, 20)
    img = np.zeros((120, 100), dtype=int)
    y, x = circle_perimeter(y_0, x_0, rad_0)
    img[x, y] = 1

    x_1, y_1, rad_1 = (60, 60, 30)
    y, x = circle_perimeter(y_1, x_1, rad_1)
    # Add noise and create an imperfect circle to lower the peak in Hough space
    y[::2] += 1
    x[::2] += 1
    img[x, y] = 1

    x_2, y_2, rad_2 = (70, 70, 20)
    y, x = circle_perimeter(y_2, x_2, rad_2)
    # Add noise and create an imperfect circle to lower the peak in Hough space
    y[::2] += 1
    x[::2] += 1
    img[x, y] = 1

    radii = [rad_0, rad_1, rad_2]
    hspaces = transform.hough_circle(img, radii)
    out = transform.hough_circle_peaks(hspaces, radii, min_xdistance=15,
                                       min_ydistance=15, threshold=None,
                                       num_peaks=np.inf,
                                       total_num_peaks=np.inf,
                                       normalize=True)

    # The second circle is too close to the first one
    # and has a weaker peak in Hough space due to imperfectness.
    # Therefore it got removed.
    assert_equal(out[1], np.array([y_0, y_2]))
    assert_equal(out[2], np.array([x_0, x_2]))
    assert_equal(out[3], np.array([rad_0, rad_2]))


def test_hough_circle_peaks_total_peak_and_min_distance():
    img = np.zeros((120, 120), dtype=int)
    cx = cy = [40, 50, 60, 70, 80]
    radii = range(20, 30, 2)
    for i in range(len(cx)):
        y, x = circle_perimeter(cy[i], cx[i], radii[i])
        img[x, y] = 1

    hspaces = transform.hough_circle(img, radii)
    out = transform.hough_circle_peaks(hspaces, radii, min_xdistance=15,
                                       min_ydistance=15, threshold=None,
                                       num_peaks=np.inf,
                                       total_num_peaks=2,
                                       normalize=True)

    # 2nd (4th) circle is removed as it is close to 1st (3rd) oneself.
    # 5th is removed as total_num_peaks = 2
    assert_equal(out[1], np.array(cy[:4:2]))
    assert_equal(out[2], np.array(cx[:4:2]))
    assert_equal(out[3], np.array(radii[:4:2]))


def test_hough_circle_peaks_normalize():
    x_0, y_0, rad_0 = (50, 50, 20)
    img = np.zeros((120, 100), dtype=int)
    y, x = circle_perimeter(y_0, x_0, rad_0)
    img[x, y] = 1

    x_1, y_1, rad_1 = (60, 60, 30)
    y, x = circle_perimeter(y_1, x_1, rad_1)
    img[x, y] = 1

    radii = [rad_0, rad_1]
    hspaces = transform.hough_circle(img, radii)
    out = transform.hough_circle_peaks(hspaces, radii, min_xdistance=15,
                                       min_ydistance=15, threshold=None,
                                       num_peaks=np.inf,
                                       total_num_peaks=np.inf,
                                       normalize=False)

    # Two perfect circles are close but the second one is bigger.
    # Therefore, it is picked due to its high peak.
    assert_equal(out[1], np.array([y_1]))
    assert_equal(out[2], np.array([x_1]))
    assert_equal(out[3], np.array([rad_1]))


def test_hough_ellipse_zero_angle():
    img = np.zeros((25, 25), dtype=int)
    rx = 6
    ry = 8
    x0 = 12
    y0 = 15
    angle = 0
    rr, cc = ellipse_perimeter(y0, x0, ry, rx)
    img[rr, cc] = 1
    result = transform.hough_ellipse(img, threshold=9)
    best = result[-1]
    assert_equal(best[1], y0)
    assert_equal(best[2], x0)
    assert_almost_equal(best[3], ry, decimal=1)
    assert_almost_equal(best[4], rx, decimal=1)
    assert_equal(best[5], angle)
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_non_zero_posangle1():
    # ry > rx, angle in [0:pi/2]
    img = np.zeros((30, 24), dtype=int)
    rx = 6
    ry = 12
    x0 = 10
    y0 = 15
    angle = np.pi / 1.35
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = transform.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    assert_almost_equal(best[1] / 100., y0 / 100., decimal=1)
    assert_almost_equal(best[2] / 100., x0 / 100., decimal=1)
    assert_almost_equal(best[3] / 10., ry / 10., decimal=1)
    assert_almost_equal(best[4] / 100., rx / 100., decimal=1)
    assert_almost_equal(best[5], angle, decimal=1)
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_non_zero_posangle2():
    # ry < rx, angle in [0:pi/2]
    img = np.zeros((30, 24), dtype=int)
    rx = 12
    ry = 6
    x0 = 10
    y0 = 15
    angle = np.pi / 1.35
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = transform.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    assert_almost_equal(best[1] / 100., y0 / 100., decimal=1)
    assert_almost_equal(best[2] / 100., x0 / 100., decimal=1)
    assert_almost_equal(best[3] / 10., ry / 10., decimal=1)
    assert_almost_equal(best[4] / 100., rx / 100., decimal=1)
    assert_almost_equal(best[5], angle, decimal=1)
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_non_zero_posangle3():
    # ry < rx, angle in [pi/2:pi]
    img = np.zeros((30, 24), dtype=int)
    rx = 12
    ry = 6
    x0 = 10
    y0 = 15
    angle = np.pi / 1.35 + np.pi / 2.
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = transform.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_non_zero_posangle4():
    # ry < rx, angle in [pi:3pi/4]
    img = np.zeros((30, 24), dtype=int)
    rx = 12
    ry = 6
    x0 = 10
    y0 = 15
    angle = np.pi / 1.35 + np.pi
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = transform.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_non_zero_negangle1():
    # ry > rx, angle in [0:-pi/2]
    img = np.zeros((30, 24), dtype=int)
    rx = 6
    ry = 12
    x0 = 10
    y0 = 15
    angle = - np.pi / 1.35
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = transform.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_non_zero_negangle2():
    # ry < rx, angle in [0:-pi/2]
    img = np.zeros((30, 24), dtype=int)
    rx = 12
    ry = 6
    x0 = 10
    y0 = 15
    angle = - np.pi / 1.35
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = transform.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_non_zero_negangle3():
    # ry < rx, angle in [-pi/2:-pi]
    img = np.zeros((30, 24), dtype=int)
    rx = 12
    ry = 6
    x0 = 10
    y0 = 15
    angle = - np.pi / 1.35 - np.pi / 2.
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = transform.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_non_zero_negangle4():
    # ry < rx, angle in [-pi:-3pi/4]
    img = np.zeros((30, 24), dtype=int)
    rx = 12
    ry = 6
    x0 = 10
    y0 = 15
    angle = - np.pi / 1.35 - np.pi
    rr, cc = ellipse_perimeter(y0, x0, ry, rx, orientation=angle)
    img[rr, cc] = 1
    result = transform.hough_ellipse(img, threshold=15, accuracy=3)
    result.sort(order='accumulator')
    best = result[-1]
    # Check if I re-draw the ellipse, points are the same!
    # ie check API compatibility between hough_ellipse and ellipse_perimeter
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]),
                                 orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)


def test_hough_ellipse_all_black_img():
    assert(transform.hough_ellipse(np.zeros((100, 100))).shape == (0, 6))
