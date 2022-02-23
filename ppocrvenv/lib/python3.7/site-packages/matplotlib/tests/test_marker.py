import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.path import Path
from matplotlib.testing.decorators import check_figures_equal

import pytest


def test_marker_fillstyle():
    marker_style = markers.MarkerStyle(marker='o', fillstyle='none')
    assert marker_style.get_fillstyle() == 'none'
    assert not marker_style.is_filled()


@pytest.mark.parametrize('marker', [
    'o',
    'x',
    '',
    'None',
    None,
    r'$\frac{1}{2}$',
    "$\u266B$",
    1,
    markers.TICKLEFT,
    [[-1, 0], [1, 0]],
    np.array([[-1, 0], [1, 0]]),
    Path([[0, 0], [1, 0]], [Path.MOVETO, Path.LINETO]),
    (5, 0),  # a pentagon
    (7, 1),  # a 7-pointed star
    (5, 2),  # asterisk
    (5, 0, 10),  # a pentagon, rotated by 10 degrees
    (7, 1, 10),  # a 7-pointed star, rotated by 10 degrees
    (5, 2, 10),  # asterisk, rotated by 10 degrees
    markers.MarkerStyle(),
    markers.MarkerStyle('o'),
])
def test_markers_valid(marker):
    # Checking this doesn't fail.
    markers.MarkerStyle(marker)


@pytest.mark.parametrize('marker', [
    'square',  # arbitrary string
    np.array([[-0.5, 0, 1, 2, 3]]),  # 1D array
    (1,),
    (5, 3),  # second parameter of tuple must be 0, 1, or 2
    (1, 2, 3, 4),
])
def test_markers_invalid(marker):
    with pytest.raises(ValueError):
        markers.MarkerStyle(marker)


class UnsnappedMarkerStyle(markers.MarkerStyle):
    """
    A MarkerStyle where the snap threshold is force-disabled.

    This is used to compare to polygon/star/asterisk markers which do not have
    any snap threshold set.
    """
    def _recache(self):
        super()._recache()
        self._snap_threshold = None


@check_figures_equal()
def test_poly_marker(fig_test, fig_ref):
    ax_test = fig_test.add_subplot()
    ax_ref = fig_ref.add_subplot()

    # Note, some reference sizes must be different because they have unit
    # *length*, while polygon markers are inscribed in a circle of unit
    # *radius*. This introduces a factor of np.sqrt(2), but since size is
    # squared, that becomes 2.
    size = 20**2

    # Squares
    ax_test.scatter([0], [0], marker=(4, 0, 45), s=size)
    ax_ref.scatter([0], [0], marker='s', s=size/2)

    # Diamonds, with and without rotation argument
    ax_test.scatter([1], [1], marker=(4, 0), s=size)
    ax_ref.scatter([1], [1], marker=UnsnappedMarkerStyle('D'), s=size/2)
    ax_test.scatter([1], [1.5], marker=(4, 0, 0), s=size)
    ax_ref.scatter([1], [1.5], marker=UnsnappedMarkerStyle('D'), s=size/2)

    # Pentagon, with and without rotation argument
    ax_test.scatter([2], [2], marker=(5, 0), s=size)
    ax_ref.scatter([2], [2], marker=UnsnappedMarkerStyle('p'), s=size)
    ax_test.scatter([2], [2.5], marker=(5, 0, 0), s=size)
    ax_ref.scatter([2], [2.5], marker=UnsnappedMarkerStyle('p'), s=size)

    # Hexagon, with and without rotation argument
    ax_test.scatter([3], [3], marker=(6, 0), s=size)
    ax_ref.scatter([3], [3], marker='h', s=size)
    ax_test.scatter([3], [3.5], marker=(6, 0, 0), s=size)
    ax_ref.scatter([3], [3.5], marker='h', s=size)

    # Rotated hexagon
    ax_test.scatter([4], [4], marker=(6, 0, 30), s=size)
    ax_ref.scatter([4], [4], marker='H', s=size)

    # Octagons
    ax_test.scatter([5], [5], marker=(8, 0, 22.5), s=size)
    ax_ref.scatter([5], [5], marker=UnsnappedMarkerStyle('8'), s=size)

    ax_test.set(xlim=(-0.5, 5.5), ylim=(-0.5, 5.5))
    ax_ref.set(xlim=(-0.5, 5.5), ylim=(-0.5, 5.5))


def test_star_marker():
    # We don't really have a strict equivalent to this marker, so we'll just do
    # a smoke test.
    size = 20**2

    fig, ax = plt.subplots()
    ax.scatter([0], [0], marker=(5, 1), s=size)
    ax.scatter([1], [1], marker=(5, 1, 0), s=size)
    ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 1.5))


# The asterisk marker is really a star with 0-size inner circle, so the ends
# are corners and get a slight bevel. The reference markers are just singular
# lines without corners, so they have no bevel, and we need to add a slight
# tolerance.
@check_figures_equal(tol=1.45)
def test_asterisk_marker(fig_test, fig_ref, request):
    ax_test = fig_test.add_subplot()
    ax_ref = fig_ref.add_subplot()

    # Note, some reference sizes must be different because they have unit
    # *length*, while asterisk markers are inscribed in a circle of unit
    # *radius*. This introduces a factor of np.sqrt(2), but since size is
    # squared, that becomes 2.
    size = 20**2

    def draw_ref_marker(y, style, size):
        # As noted above, every line is doubled. Due to antialiasing, these
        # doubled lines make a slight difference in the .png results.
        ax_ref.scatter([y], [y], marker=UnsnappedMarkerStyle(style), s=size)
        if request.getfixturevalue('ext') == 'png':
            ax_ref.scatter([y], [y], marker=UnsnappedMarkerStyle(style),
                           s=size)

    # Plus
    ax_test.scatter([0], [0], marker=(4, 2), s=size)
    draw_ref_marker(0, '+', size)
    ax_test.scatter([0.5], [0.5], marker=(4, 2, 0), s=size)
    draw_ref_marker(0.5, '+', size)

    # Cross
    ax_test.scatter([1], [1], marker=(4, 2, 45), s=size)
    draw_ref_marker(1, 'x', size/2)

    ax_test.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))
    ax_ref.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))


@check_figures_equal()
def test_marker_clipping(fig_ref, fig_test):
    # Plotting multiple markers can trigger different optimized paths in
    # backends, so compare single markers vs multiple to ensure they are
    # clipped correctly.
    marker_count = len(markers.MarkerStyle.markers)
    marker_size = 50
    ncol = 7
    nrow = marker_count // ncol + 1

    width = 2 * marker_size * ncol
    height = 2 * marker_size * nrow * 2
    fig_ref.set_size_inches((width / fig_ref.dpi, height / fig_ref.dpi))
    ax_ref = fig_ref.add_axes([0, 0, 1, 1])
    fig_test.set_size_inches((width / fig_test.dpi, height / fig_ref.dpi))
    ax_test = fig_test.add_axes([0, 0, 1, 1])

    for i, marker in enumerate(markers.MarkerStyle.markers):
        x = i % ncol
        y = i // ncol * 2

        # Singular markers per call.
        ax_ref.plot([x, x], [y, y + 1], c='k', linestyle='-', lw=3)
        ax_ref.plot(x, y, c='k',
                    marker=marker, markersize=marker_size, markeredgewidth=10,
                    fillstyle='full', markerfacecolor='white')
        ax_ref.plot(x, y + 1, c='k',
                    marker=marker, markersize=marker_size, markeredgewidth=10,
                    fillstyle='full', markerfacecolor='white')

        # Multiple markers in a single call.
        ax_test.plot([x, x], [y, y + 1], c='k', linestyle='-', lw=3,
                     marker=marker, markersize=marker_size, markeredgewidth=10,
                     fillstyle='full', markerfacecolor='white')

    ax_ref.set(xlim=(-0.5, ncol), ylim=(-0.5, 2 * nrow))
    ax_test.set(xlim=(-0.5, ncol), ylim=(-0.5, 2 * nrow))
    ax_ref.axis('off')
    ax_test.axis('off')
