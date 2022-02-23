from io import BytesIO
import pickle

import numpy as np
import pytest

from matplotlib import cm
from matplotlib.testing.decorators import image_comparison
from matplotlib.dates import rrulewrapper
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.figure as mfigure
from mpl_toolkits.axes_grid1 import parasite_axes


def test_simple():
    fig = plt.figure()
    pickle.dump(fig, BytesIO(), pickle.HIGHEST_PROTOCOL)

    ax = plt.subplot(121)
    pickle.dump(ax, BytesIO(), pickle.HIGHEST_PROTOCOL)

    ax = plt.axes(projection='polar')
    plt.plot(np.arange(10), label='foobar')
    plt.legend()

    pickle.dump(ax, BytesIO(), pickle.HIGHEST_PROTOCOL)

#    ax = plt.subplot(121, projection='hammer')
#    pickle.dump(ax, BytesIO(), pickle.HIGHEST_PROTOCOL)

    plt.figure()
    plt.bar(x=np.arange(10), height=np.arange(10))
    pickle.dump(plt.gca(), BytesIO(), pickle.HIGHEST_PROTOCOL)

    fig = plt.figure()
    ax = plt.axes()
    plt.plot(np.arange(10))
    ax.set_yscale('log')
    pickle.dump(fig, BytesIO(), pickle.HIGHEST_PROTOCOL)


@image_comparison(
    ['multi_pickle.png'], remove_text=True, style='mpl20', tol=0.082)
def test_complete():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    fig = plt.figure('Figure with a label?', figsize=(10, 6))

    plt.suptitle('Can you fit any more in a figure?')

    # make some arbitrary data
    x, y = np.arange(8), np.arange(10)
    data = u = v = np.linspace(0, 10, 80).reshape(10, 8)
    v = np.sin(v * -0.6)

    # Ensure lists also pickle correctly.
    plt.subplot(3, 3, 1)
    plt.plot(list(range(10)))

    plt.subplot(3, 3, 2)
    plt.contourf(data, hatches=['//', 'ooo'])
    plt.colorbar()

    plt.subplot(3, 3, 3)
    plt.pcolormesh(data)

    plt.subplot(3, 3, 4)
    plt.imshow(data)

    plt.subplot(3, 3, 5)
    plt.pcolor(data)

    ax = plt.subplot(3, 3, 6)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 9)
    plt.streamplot(x, y, u, v)

    ax = plt.subplot(3, 3, 7)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 9)
    plt.quiver(x, y, u, v)

    plt.subplot(3, 3, 8)
    plt.scatter(x, x**2, label='$x^2$')
    plt.legend(loc='upper left')

    plt.subplot(3, 3, 9)
    plt.errorbar(x, x * -0.5, xerr=0.2, yerr=0.4)

    #
    # plotting is done, now test its pickle-ability
    #
    result_fh = BytesIO()
    pickle.dump(fig, result_fh, pickle.HIGHEST_PROTOCOL)

    plt.close('all')

    # make doubly sure that there are no figures left
    assert plt._pylab_helpers.Gcf.figs == {}

    # wind back the fh and load in the figure
    result_fh.seek(0)
    fig = pickle.load(result_fh)

    # make sure there is now a figure manager
    assert plt._pylab_helpers.Gcf.figs != {}

    assert fig.get_label() == 'Figure with a label?'


def test_no_pyplot():
    # tests pickle-ability of a figure not created with pyplot
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    fig = mfigure.Figure()
    _ = FigureCanvasPdf(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([1, 2, 3], [1, 2, 3])
    pickle.dump(fig, BytesIO(), pickle.HIGHEST_PROTOCOL)


def test_renderer():
    from matplotlib.backends.backend_agg import RendererAgg
    renderer = RendererAgg(10, 20, 30)
    pickle.dump(renderer, BytesIO())


def test_image():
    # Prior to v1.4.0 the Image would cache data which was not picklable
    # once it had been drawn.
    from matplotlib.backends.backend_agg import new_figure_manager
    manager = new_figure_manager(1000)
    fig = manager.canvas.figure
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.arange(12).reshape(3, 4))
    manager.canvas.draw()
    pickle.dump(fig, BytesIO())


def test_polar():
    plt.subplot(polar=True)
    fig = plt.gcf()
    pf = pickle.dumps(fig)
    pickle.loads(pf)
    plt.draw()


class TransformBlob:
    def __init__(self):
        self.identity = mtransforms.IdentityTransform()
        self.identity2 = mtransforms.IdentityTransform()
        # Force use of the more complex composition.
        self.composite = mtransforms.CompositeGenericTransform(
            self.identity,
            self.identity2)
        # Check parent -> child links of TransformWrapper.
        self.wrapper = mtransforms.TransformWrapper(self.composite)
        # Check child -> parent links of TransformWrapper.
        self.composite2 = mtransforms.CompositeGenericTransform(
            self.wrapper,
            self.identity)


def test_transform():
    obj = TransformBlob()
    pf = pickle.dumps(obj)
    del obj

    obj = pickle.loads(pf)
    # Check parent -> child links of TransformWrapper.
    assert obj.wrapper._child == obj.composite
    # Check child -> parent links of TransformWrapper.
    assert [v() for v in obj.wrapper._parents.values()] == [obj.composite2]
    # Check input and output dimensions are set as expected.
    assert obj.wrapper.input_dims == obj.composite.input_dims
    assert obj.wrapper.output_dims == obj.composite.output_dims


def test_rrulewrapper():
    r = rrulewrapper(2)
    try:
        pickle.loads(pickle.dumps(r))
    except RecursionError:
        print('rrulewrapper pickling test failed')
        raise


def test_shared():
    fig, axs = plt.subplots(2, sharex=True)
    fig = pickle.loads(pickle.dumps(fig))
    fig.axes[0].set_xlim(10, 20)
    assert fig.axes[1].get_xlim() == (10, 20)


def test_inset_and_secondary():
    fig, ax = plt.subplots()
    ax.inset_axes([.1, .1, .3, .3])
    ax.secondary_xaxis("top", functions=(np.square, np.sqrt))
    pickle.loads(pickle.dumps(fig))


@pytest.mark.parametrize("cmap", cm._cmap_registry.values())
def test_cmap(cmap):
    pickle.dumps(cmap)


def test_unpickle_canvas():
    fig = mfigure.Figure()
    assert fig.canvas is not None
    out = BytesIO()
    pickle.dump(fig, out)
    out.seek(0)
    fig2 = pickle.load(out)
    assert fig2.canvas is not None


def test_mpl_toolkits():
    ax = parasite_axes.host_axes([0, 0, 1, 1])
    assert type(pickle.loads(pickle.dumps(ax))) == parasite_axes.HostAxes
