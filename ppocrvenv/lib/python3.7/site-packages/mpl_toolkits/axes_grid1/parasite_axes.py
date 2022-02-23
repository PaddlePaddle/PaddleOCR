import functools

from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.image as mimage
import matplotlib.transforms as mtransforms
from matplotlib.axes import subplot_class_factory
from matplotlib.transforms import Bbox
from .mpl_axes import Axes


class ParasiteAxesBase:

    def __init__(self, parent_axes, aux_transform=None,
                 *, viewlim_mode=None, **kwargs):
        self._parent_axes = parent_axes
        self.transAux = aux_transform
        self.set_viewlim_mode(viewlim_mode)
        kwargs["frameon"] = False
        super().__init__(parent_axes.figure, parent_axes._position, **kwargs)

    def cla(self):
        super().cla()
        martist.setp(self.get_children(), visible=False)
        self._get_lines = self._parent_axes._get_lines

    @_api.deprecated("3.5")
    def get_images_artists(self):
        artists = []
        images = []

        for a in self.get_children():
            if not a.get_visible():
                continue
            if isinstance(a, mimage.AxesImage):
                images.append(a)
            else:
                artists.append(a)

        return images, artists

    def pick(self, mouseevent):
        # This most likely goes to Artist.pick (depending on axes_class given
        # to the factory), which only handles pick events registered on the
        # axes associated with each child:
        super().pick(mouseevent)
        # But parasite axes are additionally given pick events from their host
        # axes (cf. HostAxesBase.pick), which we handle here:
        for a in self.get_children():
            if (hasattr(mouseevent.inaxes, "parasites")
                    and self in mouseevent.inaxes.parasites):
                a.pick(mouseevent)

    # aux_transform support

    def _set_lim_and_transforms(self):
        if self.transAux is not None:
            self.transAxes = self._parent_axes.transAxes
            self.transData = self.transAux + self._parent_axes.transData
            self._xaxis_transform = mtransforms.blended_transform_factory(
                self.transData, self.transAxes)
            self._yaxis_transform = mtransforms.blended_transform_factory(
                self.transAxes, self.transData)
        else:
            super()._set_lim_and_transforms()

    def set_viewlim_mode(self, mode):
        _api.check_in_list([None, "equal", "transform"], mode=mode)
        self._viewlim_mode = mode

    def get_viewlim_mode(self):
        return self._viewlim_mode

    @_api.deprecated("3.4", alternative="apply_aspect")
    def update_viewlim(self):
        return self._update_viewlim

    def _update_viewlim(self):  # Inline after deprecation elapses.
        viewlim = self._parent_axes.viewLim.frozen()
        mode = self.get_viewlim_mode()
        if mode is None:
            pass
        elif mode == "equal":
            self.axes.viewLim.set(viewlim)
        elif mode == "transform":
            self.axes.viewLim.set(
                viewlim.transformed(self.transAux.inverted()))
        else:
            _api.check_in_list([None, "equal", "transform"], mode=mode)

    def apply_aspect(self, position=None):
        self._update_viewlim()
        super().apply_aspect()

    # end of aux_transform support


parasite_axes_class_factory = cbook._make_class_factory(
    ParasiteAxesBase, "{}Parasite")
ParasiteAxes = parasite_axes_class_factory(Axes)


@_api.deprecated("3.4", alternative="ParasiteAxesBase")
class ParasiteAxesAuxTransBase:
    def __init__(self, parent_axes, aux_transform, viewlim_mode=None,
                 **kwargs):
        # Explicit wrapper for deprecation to work.
        super().__init__(parent_axes, aux_transform,
                         viewlim_mode=viewlim_mode, **kwargs)

    def _set_lim_and_transforms(self):
        self.transAxes = self._parent_axes.transAxes
        self.transData = self.transAux + self._parent_axes.transData
        self._xaxis_transform = mtransforms.blended_transform_factory(
            self.transData, self.transAxes)
        self._yaxis_transform = mtransforms.blended_transform_factory(
            self.transAxes, self.transData)

    def set_viewlim_mode(self, mode):
        _api.check_in_list([None, "equal", "transform"], mode=mode)
        self._viewlim_mode = mode

    def get_viewlim_mode(self):
        return self._viewlim_mode

    @_api.deprecated("3.4", alternative="apply_aspect")
    def update_viewlim(self):
        return self._update_viewlim()

    def _update_viewlim(self):  # Inline after deprecation elapses.
        viewlim = self._parent_axes.viewLim.frozen()
        mode = self.get_viewlim_mode()
        if mode is None:
            pass
        elif mode == "equal":
            self.axes.viewLim.set(viewlim)
        elif mode == "transform":
            self.axes.viewLim.set(
                viewlim.transformed(self.transAux.inverted()))
        else:
            _api.check_in_list([None, "equal", "transform"], mode=mode)

    def apply_aspect(self, position=None):
        self._update_viewlim()
        super().apply_aspect()


@_api.deprecated("3.4", alternative="parasite_axes_class_factory")
@functools.lru_cache(None)
def parasite_axes_auxtrans_class_factory(axes_class):
    if not issubclass(axes_class, ParasiteAxesBase):
        parasite_axes_class = parasite_axes_class_factory(axes_class)
    else:
        parasite_axes_class = axes_class
    return type("%sParasiteAuxTrans" % parasite_axes_class.__name__,
                (ParasiteAxesAuxTransBase, parasite_axes_class),
                {'name': 'parasite_axes'})


# Also deprecated.
with _api.suppress_matplotlib_deprecation_warning():
    ParasiteAxesAuxTrans = parasite_axes_auxtrans_class_factory(ParasiteAxes)


class HostAxesBase:
    def __init__(self, *args, **kwargs):
        self.parasites = []
        super().__init__(*args, **kwargs)

    def get_aux_axes(self, tr=None, viewlim_mode="equal", axes_class=Axes):
        """
        Add a parasite axes to this host.

        Despite this method's name, this should actually be thought of as an
        ``add_parasite_axes`` method.

        *tr* may be `.Transform`, in which case the following relation will
        hold: ``parasite.transData = tr + host.transData``.  Alternatively, it
        may be None (the default), no special relationship will hold between
        the parasite's and the host's ``transData``.
        """
        parasite_axes_class = parasite_axes_class_factory(axes_class)
        ax2 = parasite_axes_class(self, tr, viewlim_mode=viewlim_mode)
        # note that ax2.transData == tr + ax1.transData
        # Anything you draw in ax2 will match the ticks and grids of ax1.
        self.parasites.append(ax2)
        ax2._remove_method = self.parasites.remove
        return ax2

    def _get_legend_handles(self, legend_handler_map=None):
        all_handles = super()._get_legend_handles()
        for ax in self.parasites:
            all_handles.extend(ax._get_legend_handles(legend_handler_map))
        return all_handles

    def draw(self, renderer):
        orig_children_len = len(self._children)

        locator = self.get_axes_locator()
        if locator:
            pos = locator(self, renderer)
            self.set_position(pos, which="active")
            self.apply_aspect(pos)
        else:
            self.apply_aspect()

        rect = self.get_position()
        for ax in self.parasites:
            ax.apply_aspect(rect)
            self._children.extend(ax.get_children())

        super().draw(renderer)
        self._children = self._children[:orig_children_len]

    def cla(self):
        for ax in self.parasites:
            ax.cla()
        super().cla()

    def pick(self, mouseevent):
        super().pick(mouseevent)
        # Also pass pick events on to parasite axes and, in turn, their
        # children (cf. ParasiteAxesBase.pick)
        for a in self.parasites:
            a.pick(mouseevent)

    def twinx(self, axes_class=None):
        """
        Create a twin of Axes with a shared x-axis but independent y-axis.

        The y-axis of self will have ticks on the left and the returned axes
        will have ticks on the right.
        """
        ax = self._add_twin_axes(axes_class, sharex=self)
        self.axis["right"].set_visible(False)
        ax.axis["right"].set_visible(True)
        ax.axis["left", "top", "bottom"].set_visible(False)
        return ax

    def twiny(self, axes_class=None):
        """
        Create a twin of Axes with a shared y-axis but independent x-axis.

        The x-axis of self will have ticks on the bottom and the returned axes
        will have ticks on the top.
        """
        ax = self._add_twin_axes(axes_class, sharey=self)
        self.axis["top"].set_visible(False)
        ax.axis["top"].set_visible(True)
        ax.axis["left", "right", "bottom"].set_visible(False)
        return ax

    def twin(self, aux_trans=None, axes_class=None):
        """
        Create a twin of Axes with no shared axis.

        While self will have ticks on the left and bottom axis, the returned
        axes will have ticks on the top and right axis.
        """
        if aux_trans is None:
            aux_trans = mtransforms.IdentityTransform()
        ax = self._add_twin_axes(
            axes_class, aux_transform=aux_trans, viewlim_mode="transform")
        self.axis["top", "right"].set_visible(False)
        ax.axis["top", "right"].set_visible(True)
        ax.axis["left", "bottom"].set_visible(False)
        return ax

    def _add_twin_axes(self, axes_class, **kwargs):
        """
        Helper for `.twinx`/`.twiny`/`.twin`.

        *kwargs* are forwarded to the parasite axes constructor.
        """
        if axes_class is None:
            axes_class = self._base_axes_class
        ax = parasite_axes_class_factory(axes_class)(self, **kwargs)
        self.parasites.append(ax)
        ax._remove_method = self._remove_any_twin
        return ax

    def _remove_any_twin(self, ax):
        self.parasites.remove(ax)
        restore = ["top", "right"]
        if ax._sharex:
            restore.remove("top")
        if ax._sharey:
            restore.remove("right")
        self.axis[tuple(restore)].set_visible(True)
        self.axis[tuple(restore)].toggle(ticklabels=False, label=False)

    def get_tightbbox(self, renderer, call_axes_locator=True,
                      bbox_extra_artists=None):
        bbs = [
            *[ax.get_tightbbox(renderer, call_axes_locator=call_axes_locator)
              for ax in self.parasites],
            super().get_tightbbox(renderer,
                                  call_axes_locator=call_axes_locator,
                                  bbox_extra_artists=bbox_extra_artists)]
        return Bbox.union([b for b in bbs if b.width != 0 or b.height != 0])


host_axes_class_factory = cbook._make_class_factory(
    HostAxesBase, "{}HostAxes", "_base_axes_class")
HostAxes = host_axes_class_factory(Axes)
SubplotHost = subplot_class_factory(HostAxes)


def host_subplot_class_factory(axes_class):
    host_axes_class = host_axes_class_factory(axes_class)
    subplot_host_class = subplot_class_factory(host_axes_class)
    return subplot_host_class


def host_axes(*args, axes_class=Axes, figure=None, **kwargs):
    """
    Create axes that can act as a hosts to parasitic axes.

    Parameters
    ----------
    figure : `matplotlib.figure.Figure`
        Figure to which the axes will be added. Defaults to the current figure
        `.pyplot.gcf()`.

    *args, **kwargs
        Will be passed on to the underlying ``Axes`` object creation.
    """
    import matplotlib.pyplot as plt
    host_axes_class = host_axes_class_factory(axes_class)
    if figure is None:
        figure = plt.gcf()
    ax = host_axes_class(figure, *args, **kwargs)
    figure.add_axes(ax)
    plt.draw_if_interactive()
    return ax


def host_subplot(*args, axes_class=Axes, figure=None, **kwargs):
    """
    Create a subplot that can act as a host to parasitic axes.

    Parameters
    ----------
    figure : `matplotlib.figure.Figure`
        Figure to which the subplot will be added. Defaults to the current
        figure `.pyplot.gcf()`.

    *args, **kwargs
        Will be passed on to the underlying ``Axes`` object creation.
    """
    import matplotlib.pyplot as plt
    host_subplot_class = host_subplot_class_factory(axes_class)
    if figure is None:
        figure = plt.gcf()
    ax = host_subplot_class(figure, *args, **kwargs)
    figure.add_subplot(ax)
    plt.draw_if_interactive()
    return ax
