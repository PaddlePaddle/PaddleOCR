import numpy as np

import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


# The original version of this test relied on mpl_toolkits's slightly different
# colorbar implementation; moving to matplotlib's own colorbar implementation
# caused the small image comparison error.
@image_comparison(['imagegrid_cbar_mode.png'],
                  remove_text=True, style='mpl20', tol=0.3)
def test_imagegrid_cbar_mode_edge():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    X, Y = np.meshgrid(np.linspace(0, 6, 30), np.linspace(0, 6, 30))
    arr = np.sin(X) * np.cos(Y) + 1j*(np.sin(3*Y) * np.cos(Y/2.))

    fig = plt.figure(figsize=(18, 9))

    positions = (241, 242, 243, 244, 245, 246, 247, 248)
    directions = ['row']*4 + ['column']*4
    cbar_locations = ['left', 'right', 'top', 'bottom']*2

    for position, direction, location in zip(
            positions, directions, cbar_locations):
        grid = ImageGrid(fig, position,
                         nrows_ncols=(2, 2),
                         direction=direction,
                         cbar_location=location,
                         cbar_size='20%',
                         cbar_mode='edge')
        ax1, ax2, ax3, ax4, = grid

        ax1.imshow(arr.real, cmap='nipy_spectral')
        ax2.imshow(arr.imag, cmap='hot')
        ax3.imshow(np.abs(arr), cmap='jet')
        ax4.imshow(np.arctan2(arr.imag, arr.real), cmap='hsv')

        # In each row/column, the "first" colorbars must be overwritten by the
        # "second" ones.  To achieve this, clear out the axes first.
        for ax in grid:
            ax.cax.cla()
            cb = ax.cax.colorbar(ax.images[0])


def test_imagegrid():
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 1))
    ax = grid[0]
    im = ax.imshow([[1, 2]], norm=mpl.colors.LogNorm())
    cb = ax.cax.colorbar(im)
    assert isinstance(cb.locator, mticker.LogLocator)
