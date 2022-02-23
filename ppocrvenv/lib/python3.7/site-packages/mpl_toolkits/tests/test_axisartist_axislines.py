import numpy as np
from matplotlib import _api
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from matplotlib.transforms import IdentityTransform

from mpl_toolkits.axisartist.axislines import SubplotZero, Subplot
from mpl_toolkits.axisartist import (
    Axes, SubplotHost, ParasiteAxes, ParasiteAxesAuxTrans)

import pytest


@image_comparison(['SubplotZero.png'], style='default')
def test_SubplotZero():
    # Remove this line when this test image is regenerated.
    plt.rcParams['text.kerning_factor'] = 6

    fig = plt.figure()

    ax = SubplotZero(fig, 1, 1, 1)
    fig.add_subplot(ax)

    ax.axis["xzero"].set_visible(True)
    ax.axis["xzero"].label.set_text("Axis Zero")

    for n in ["top", "right"]:
        ax.axis[n].set_visible(False)

    xx = np.arange(0, 2 * np.pi, 0.01)
    ax.plot(xx, np.sin(xx))
    ax.set_ylabel("Test")


@image_comparison(['Subplot.png'], style='default')
def test_Subplot():
    # Remove this line when this test image is regenerated.
    plt.rcParams['text.kerning_factor'] = 6

    fig = plt.figure()

    ax = Subplot(fig, 1, 1, 1)
    fig.add_subplot(ax)

    xx = np.arange(0, 2 * np.pi, 0.01)
    ax.plot(xx, np.sin(xx))
    ax.set_ylabel("Test")

    ax.axis["top"].major_ticks.set_tick_out(True)
    ax.axis["bottom"].major_ticks.set_tick_out(True)

    ax.axis["bottom"].set_label("Tk0")


def test_Axes():
    fig = plt.figure()
    ax = Axes(fig, [0.15, 0.1, 0.65, 0.8])
    fig.add_axes(ax)
    ax.plot([1, 2, 3], [0, 1, 2])
    ax.set_xscale('log')
    fig.canvas.draw()


@pytest.mark.parametrize('parasite_cls', [ParasiteAxes, ParasiteAxesAuxTrans])
@image_comparison(['ParasiteAxesAuxTrans_meshplot.png'],
                  remove_text=True, style='default', tol=0.075)
def test_ParasiteAxesAuxTrans(parasite_cls):
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    data = np.ones((6, 6))
    data[2, 2] = 2
    data[0, :] = 0
    data[-2, :] = 0
    data[:, 0] = 0
    data[:, -2] = 0
    x = np.arange(6)
    y = np.arange(6)
    xx, yy = np.meshgrid(x, y)

    funcnames = ['pcolor', 'pcolormesh', 'contourf']

    fig = plt.figure()
    for i, name in enumerate(funcnames):

        ax1 = SubplotHost(fig, 1, 3, i+1)
        fig.add_subplot(ax1)

        with _api.suppress_matplotlib_deprecation_warning():
            ax2 = parasite_cls(ax1, IdentityTransform())
        ax1.parasites.append(ax2)
        if name.startswith('pcolor'):
            getattr(ax2, name)(xx, yy, data[:-1, :-1])
        else:
            getattr(ax2, name)(xx, yy, data)
        ax1.set_xlim((0, 5))
        ax1.set_ylim((0, 5))

    ax2.contour(xx, yy, data, colors='k')
