"""
Builtin colormaps, colormap handling utilities, and the `ScalarMappable` mixin.

.. seealso::

  :doc:`/gallery/color/colormap_reference` for a list of builtin colormaps.

  :doc:`/tutorials/colors/colormap-manipulation` for examples of how to
  make colormaps.

  :doc:`/tutorials/colors/colormaps` an in-depth discussion of
  choosing colormaps.

  :doc:`/tutorials/colors/colormapnorms` for more details about data
  normalization.
"""

from collections.abc import Mapping, MutableMapping

import numpy as np
from numpy import ma

import matplotlib as mpl
from matplotlib import _api, colors, cbook
from matplotlib._cm import datad
from matplotlib._cm_listed import cmaps as cmaps_listed


@_api.caching_module_getattr  # module-level deprecations
class __getattr__:
    LUTSIZE = _api.deprecated(
        "3.5", obj_type="", alternative="rcParams['image.lut']")(
            property(lambda self: _LUTSIZE))


_LUTSIZE = mpl.rcParams['image.lut']


def _gen_cmap_registry():
    """
    Generate a dict mapping standard colormap names to standard colormaps, as
    well as the reversed colormaps.
    """
    cmap_d = {**cmaps_listed}
    for name, spec in datad.items():
        cmap_d[name] = (  # Precache the cmaps at a fixed lutsize..
            colors.LinearSegmentedColormap(name, spec, _LUTSIZE)
            if 'red' in spec else
            colors.ListedColormap(spec['listed'], name)
            if 'listed' in spec else
            colors.LinearSegmentedColormap.from_list(name, spec, _LUTSIZE))
    # Generate reversed cmaps.
    for cmap in list(cmap_d.values()):
        rmap = cmap.reversed()
        cmap._global = True
        rmap._global = True
        cmap_d[rmap.name] = rmap
    return cmap_d


class _DeprecatedCmapDictWrapper(MutableMapping):
    """Dictionary mapping for deprecated _cmap_d access."""

    def __init__(self, cmap_registry):
        self._cmap_registry = cmap_registry

    def __delitem__(self, key):
        self._warn_deprecated()
        self._cmap_registry.__delitem__(key)

    def __getitem__(self, key):
        self._warn_deprecated()
        return self._cmap_registry.__getitem__(key)

    def __iter__(self):
        self._warn_deprecated()
        return self._cmap_registry.__iter__()

    def __len__(self):
        self._warn_deprecated()
        return self._cmap_registry.__len__()

    def __setitem__(self, key, val):
        self._warn_deprecated()
        self._cmap_registry.__setitem__(key, val)

    def get(self, key, default=None):
        self._warn_deprecated()
        return self._cmap_registry.get(key, default)

    def _warn_deprecated(self):
        _api.warn_deprecated(
            "3.3",
            message="The global colormaps dictionary is no longer "
                    "considered public API.",
            alternative="Please use register_cmap() and get_cmap() to "
                        "access the contents of the dictionary."
        )


class ColormapRegistry(Mapping):
    r"""
    Container for colormaps that are known to Matplotlib by name.

    .. admonition:: Experimental

       While we expect the API to be final, we formally mark it as
       experimental for 3.5 because we want to keep the option to still adapt
       the API for 3.6 should the need arise.

    The universal registry instance is `matplotlib.colormaps`. There should be
    no need for users to instantiate `.ColormapRegistry` themselves.

    Read access uses a dict-like interface mapping names to `.Colormap`\s::

        import matplotlib as mpl
        cmap = mpl.colormaps['viridis']

    Returned `.Colormap`\s are copies, so that their modification does not
    change the global definition of the colormap.

    Additional colormaps can be added via `.ColormapRegistry.register`::

        mpl.colormaps.register(my_colormap)
    """
    def __init__(self, cmaps):
        self._cmaps = cmaps

    def __getitem__(self, item):
        try:
            return self._cmaps[item].copy()
        except KeyError:
            raise KeyError(f"{item!r} is not a known colormap name")

    def __iter__(self):
        return iter(self._cmaps)

    def __len__(self):
        return len(self._cmaps)

    def __str__(self):
        return ('ColormapRegistry; available colormaps:\n' +
                ', '.join(f"'{name}'" for name in self))

    def __call__(self):
        """
        Return a list of the registered colormap names.

        This exists only for backward-compatibilty in `.pyplot` which had a
        ``plt.colormaps()`` method. The recommended way to get this list is
        now ``list(colormaps)``.
        """
        return list(self)

    def register(self, cmap, *, name=None, force=False):
        """
        Register a new colormap.

        The colormap name can then be used as a string argument to any ``cmap``
        parameter in Matplotlib. It is also available in ``pyplot.get_cmap``.

        The colormap registry stores a copy of the given colormap, so that
        future changes to the original colormap instance do not affect the
        registered colormap. Think of this as the registry taking a snapshot
        of the colormap at registration.

        Parameters
        ----------
        cmap : matplotlib.colors.Colormap
            The colormap to register.

        name : str, optional
            The name for the colormap. If not given, ``cmap.name`` is used.

        force: bool, default: False
            If False, a ValueError is raised if trying to overwrite an already
            registered name. True supports overwriting registered colormaps
            other than the builtin colormaps.
        """
        name = name or cmap.name
        if name in self and not force:
            raise ValueError(
                f'A colormap named "{name}" is already registered.')
        register_cmap(name, cmap.copy())


_cmap_registry = _gen_cmap_registry()
globals().update(_cmap_registry)
# This is no longer considered public API
cmap_d = _DeprecatedCmapDictWrapper(_cmap_registry)
__builtin_cmaps = tuple(_cmap_registry)

# public access to the colormaps should be via `matplotlib.colormaps`. For now,
# we still create the registry here, but that should stay an implementation
# detail.
_colormaps = ColormapRegistry(_cmap_registry)


def register_cmap(name=None, cmap=None, *, override_builtin=False):
    """
    Add a colormap to the set recognized by :func:`get_cmap`.

    Register a new colormap to be accessed by name ::

        LinearSegmentedColormap('swirly', data, lut)
        register_cmap(cmap=swirly_cmap)

    Parameters
    ----------
    name : str, optional
       The name that can be used in :func:`get_cmap` or :rc:`image.cmap`

       If absent, the name will be the :attr:`~matplotlib.colors.Colormap.name`
       attribute of the *cmap*.

    cmap : matplotlib.colors.Colormap
       Despite being the second argument and having a default value, this
       is a required argument.

    override_builtin : bool

        Allow built-in colormaps to be overridden by a user-supplied
        colormap.

        Please do not use this unless you are sure you need it.

    Notes
    -----
    Registering a colormap stores a reference to the colormap object
    which can currently be modified and inadvertently change the global
    colormap state. This behavior is deprecated and in Matplotlib 3.5
    the registered colormap will be immutable.

    """
    _api.check_isinstance((str, None), name=name)
    if name is None:
        try:
            name = cmap.name
        except AttributeError as err:
            raise ValueError("Arguments must include a name or a "
                             "Colormap") from err
    if name in _cmap_registry:
        if not override_builtin and name in __builtin_cmaps:
            msg = f"Trying to re-register the builtin cmap {name!r}."
            raise ValueError(msg)
        else:
            msg = f"Trying to register the cmap {name!r} which already exists."
            _api.warn_external(msg)

    if not isinstance(cmap, colors.Colormap):
        raise ValueError("You must pass a Colormap instance. "
                         f"You passed {cmap} a {type(cmap)} object.")

    cmap._global = True
    _cmap_registry[name] = cmap
    return


def get_cmap(name=None, lut=None):
    """
    Get a colormap instance, defaulting to rc values if *name* is None.

    Colormaps added with :func:`register_cmap` take precedence over
    built-in colormaps.

    Notes
    -----
    Currently, this returns the global colormap object, which is deprecated.
    In Matplotlib 3.5, you will no longer be able to modify the global
    colormaps in-place.

    Parameters
    ----------
    name : `matplotlib.colors.Colormap` or str or None, default: None
        If a `.Colormap` instance, it will be returned. Otherwise, the name of
        a colormap known to Matplotlib, which will be resampled by *lut*. The
        default, None, means :rc:`image.cmap`.
    lut : int or None, default: None
        If *name* is not already a Colormap instance and *lut* is not None, the
        colormap will be resampled to have *lut* entries in the lookup table.
    """
    if name is None:
        name = mpl.rcParams['image.cmap']
    if isinstance(name, colors.Colormap):
        return name
    _api.check_in_list(sorted(_cmap_registry), name=name)
    if lut is None:
        return _cmap_registry[name]
    else:
        return _cmap_registry[name]._resample(lut)


def unregister_cmap(name):
    """
    Remove a colormap recognized by :func:`get_cmap`.

    You may not remove built-in colormaps.

    If the named colormap is not registered, returns with no error, raises
    if you try to de-register a default colormap.

    .. warning ::

      Colormap names are currently a shared namespace that may be used
      by multiple packages. Use `unregister_cmap` only if you know you
      have registered that name before. In particular, do not
      unregister just in case to clean the name before registering a
      new colormap.

    Parameters
    ----------
    name : str
        The name of the colormap to be un-registered

    Returns
    -------
    ColorMap or None
        If the colormap was registered, return it if not return `None`

    Raises
    ------
    ValueError
       If you try to de-register a default built-in colormap.

    """
    if name not in _cmap_registry:
        return
    if name in __builtin_cmaps:
        raise ValueError(f"cannot unregister {name!r} which is a builtin "
                         "colormap.")
    return _cmap_registry.pop(name)


class ScalarMappable:
    """
    A mixin class to map scalar data to RGBA.

    The ScalarMappable applies data normalization before returning RGBA colors
    from the given colormap.
    """

    def __init__(self, norm=None, cmap=None):
        """

        Parameters
        ----------
        norm : `matplotlib.colors.Normalize` (or subclass thereof)
            The normalizing object which scales data, typically into the
            interval ``[0, 1]``.
            If *None*, *norm* defaults to a *colors.Normalize* object which
            initializes its scaling based on the first data processed.
        cmap : str or `~matplotlib.colors.Colormap`
            The colormap used to map normalized data values to RGBA colors.
        """
        self._A = None
        self._norm = None  # So that the setter knows we're initializing.
        self.set_norm(norm)  # The Normalize instance of this ScalarMappable.
        self.cmap = None  # So that the setter knows we're initializing.
        self.set_cmap(cmap)  # The Colormap instance of this ScalarMappable.
        #: The last colorbar associated with this ScalarMappable. May be None.
        self.colorbar = None
        self.callbacks = cbook.CallbackRegistry()

    callbacksSM = _api.deprecated("3.5", alternative="callbacks")(
        property(lambda self: self.callbacks))

    def _scale_norm(self, norm, vmin, vmax):
        """
        Helper for initial scaling.

        Used by public functions that create a ScalarMappable and support
        parameters *vmin*, *vmax* and *norm*. This makes sure that a *norm*
        will take precedence over *vmin*, *vmax*.

        Note that this method does not set the norm.
        """
        if vmin is not None or vmax is not None:
            self.set_clim(vmin, vmax)
            if norm is not None:
                raise ValueError(
                    "Passing parameters norm and vmin/vmax simultaneously is "
                    "not supported. Please pass vmin/vmax directly to the "
                    "norm when creating it.")

        # always resolve the autoscaling so we have concrete limits
        # rather than deferring to draw time.
        self.autoscale_None()

    def to_rgba(self, x, alpha=None, bytes=False, norm=True):
        """
        Return a normalized rgba array corresponding to *x*.

        In the normal case, *x* is a 1D or 2D sequence of scalars, and
        the corresponding ndarray of rgba values will be returned,
        based on the norm and colormap set for this ScalarMappable.

        There is one special case, for handling images that are already
        rgb or rgba, such as might have been read from an image file.
        If *x* is an ndarray with 3 dimensions,
        and the last dimension is either 3 or 4, then it will be
        treated as an rgb or rgba array, and no mapping will be done.
        The array can be uint8, or it can be floating point with
        values in the 0-1 range; otherwise a ValueError will be raised.
        If it is a masked array, the mask will be ignored.
        If the last dimension is 3, the *alpha* kwarg (defaulting to 1)
        will be used to fill in the transparency.  If the last dimension
        is 4, the *alpha* kwarg is ignored; it does not
        replace the pre-existing alpha.  A ValueError will be raised
        if the third dimension is other than 3 or 4.

        In either case, if *bytes* is *False* (default), the rgba
        array will be floats in the 0-1 range; if it is *True*,
        the returned rgba array will be uint8 in the 0 to 255 range.

        If norm is False, no normalization of the input data is
        performed, and it is assumed to be in the range (0-1).

        """
        # First check for special case, image input:
        try:
            if x.ndim == 3:
                if x.shape[2] == 3:
                    if alpha is None:
                        alpha = 1
                    if x.dtype == np.uint8:
                        alpha = np.uint8(alpha * 255)
                    m, n = x.shape[:2]
                    xx = np.empty(shape=(m, n, 4), dtype=x.dtype)
                    xx[:, :, :3] = x
                    xx[:, :, 3] = alpha
                elif x.shape[2] == 4:
                    xx = x
                else:
                    raise ValueError("Third dimension must be 3 or 4")
                if xx.dtype.kind == 'f':
                    if norm and (xx.max() > 1 or xx.min() < 0):
                        raise ValueError("Floating point image RGB values "
                                         "must be in the 0..1 range.")
                    if bytes:
                        xx = (xx * 255).astype(np.uint8)
                elif xx.dtype == np.uint8:
                    if not bytes:
                        xx = xx.astype(np.float32) / 255
                else:
                    raise ValueError("Image RGB array must be uint8 or "
                                     "floating point; found %s" % xx.dtype)
                return xx
        except AttributeError:
            # e.g., x is not an ndarray; so try mapping it
            pass

        # This is the normal case, mapping a scalar array:
        x = ma.asarray(x)
        if norm:
            x = self.norm(x)
        rgba = self.cmap(x, alpha=alpha, bytes=bytes)
        return rgba

    def set_array(self, A):
        """
        Set the value array from array-like *A*.

        Parameters
        ----------
        A : array-like or None
            The values that are mapped to colors.

            The base class `.ScalarMappable` does not make any assumptions on
            the dimensionality and shape of the value array *A*.
        """
        if A is None:
            self._A = None
            return

        A = cbook.safe_masked_invalid(A, copy=True)
        if not np.can_cast(A.dtype, float, "same_kind"):
            raise TypeError(f"Image data of dtype {A.dtype} cannot be "
                            "converted to float")

        self._A = A

    def get_array(self):
        """
        Return the array of values, that are mapped to colors.

        The base class `.ScalarMappable` does not make any assumptions on
        the dimensionality and shape of the array.
        """
        return self._A

    def get_cmap(self):
        """Return the `.Colormap` instance."""
        return self.cmap

    def get_clim(self):
        """
        Return the values (min, max) that are mapped to the colormap limits.
        """
        return self.norm.vmin, self.norm.vmax

    def set_clim(self, vmin=None, vmax=None):
        """
        Set the norm limits for image scaling.

        Parameters
        ----------
        vmin, vmax : float
             The limits.

             The limits may also be passed as a tuple (*vmin*, *vmax*) as a
             single positional argument.

             .. ACCEPTS: (vmin: float, vmax: float)
        """
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        if vmax is None:
            try:
                vmin, vmax = vmin
            except (TypeError, ValueError):
                pass
        if vmin is not None:
            self.norm.vmin = colors._sanitize_extrema(vmin)
        if vmax is not None:
            self.norm.vmax = colors._sanitize_extrema(vmax)

    def get_alpha(self):
        """
        Returns
        -------
        float
            Always returns 1.
        """
        # This method is intended to be overridden by Artist sub-classes
        return 1.

    def set_cmap(self, cmap):
        """
        Set the colormap for luminance data.

        Parameters
        ----------
        cmap : `.Colormap` or str or None
        """
        in_init = self.cmap is None
        cmap = get_cmap(cmap)
        self.cmap = cmap
        if not in_init:
            self.changed()  # Things are not set up properly yet.

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, norm):
        _api.check_isinstance((colors.Normalize, None), norm=norm)
        if norm is None:
            norm = colors.Normalize()

        if norm is self.norm:
            # We aren't updating anything
            return

        in_init = self.norm is None
        # Remove the current callback and connect to the new one
        if not in_init:
            self.norm.callbacks.disconnect(self._id_norm)
        self._norm = norm
        self._id_norm = self.norm.callbacks.connect('changed',
                                                    self.changed)
        if not in_init:
            self.changed()

    def set_norm(self, norm):
        """
        Set the normalization instance.

        Parameters
        ----------
        norm : `.Normalize` or None

        Notes
        -----
        If there are any colorbars using the mappable for this norm, setting
        the norm of the mappable will reset the norm, locator, and formatters
        on the colorbar to default.
        """
        self.norm = norm

    def autoscale(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array
        """
        if self._A is None:
            raise TypeError('You must first set_array for mappable')
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        self.norm.autoscale(self._A)

    def autoscale_None(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array, changing only limits that are None
        """
        if self._A is None:
            raise TypeError('You must first set_array for mappable')
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        self.norm.autoscale_None(self._A)

    def changed(self):
        """
        Call this whenever the mappable is changed to notify all the
        callbackSM listeners to the 'changed' signal.
        """
        self.callbacks.process('changed', self)
        self.stale = True
