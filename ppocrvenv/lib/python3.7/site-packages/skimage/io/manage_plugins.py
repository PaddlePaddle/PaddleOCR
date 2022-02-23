"""Handle image reading, writing and plotting plugins.

To improve performance, plugins are only loaded as needed. As a result, there
can be multiple states for a given plugin:

    available: Defined in an *ini file located in `skimage.io._plugins`.
        See also `skimage.io.available_plugins`.
    partial definition: Specified in an *ini file, but not defined in the
        corresponding plugin module. This will raise an error when loaded.
    available but not on this system: Defined in `skimage.io._plugins`, but
        a dependent library (e.g. Qt, PIL) is not available on your system.
        This will raise an error when loaded.
    loaded: The real availability is determined when it's explicitly loaded,
        either because it's one of the default plugins, or because it's
        loaded explicitly by the user.

"""
import os.path
import warnings
from configparser import ConfigParser
from glob import glob

from .collection import imread_collection_wrapper

__all__ = ['use_plugin', 'call_plugin', 'plugin_info', 'plugin_order',
           'reset_plugins', 'find_available_plugins', 'available_plugins']

# The plugin store will save a list of *loaded* io functions for each io type
# (e.g. 'imread', 'imsave', etc.). Plugins are loaded as requested.
plugin_store = None
# Dictionary mapping plugin names to a list of functions they provide.
plugin_provides = {}
# The module names for the plugins in `skimage.io._plugins`.
plugin_module_name = {}
# Meta-data about plugins provided by *.ini files.
plugin_meta_data = {}
# For each plugin type, default to the first available plugin as defined by
# the following preferences.
preferred_plugins = {
    # Default plugins for all types (overridden by specific types below).
    'all': ['imageio', 'pil', 'matplotlib', 'qt'],
    'imshow': ['matplotlib'],
    'imshow_collection': ['matplotlib']
}


def _clear_plugins():
    """Clear the plugin state to the default, i.e., where no plugins are loaded
    """
    global plugin_store
    plugin_store = {'imread': [],
                    'imsave': [],
                    'imshow': [],
                    'imread_collection': [],
                    'imshow_collection': [],
                    '_app_show': []}


_clear_plugins()


def _load_preferred_plugins():
    # Load preferred plugin for each io function.
    io_types = ['imsave', 'imshow', 'imread_collection', 'imshow_collection',
                'imread']
    for p_type in io_types:
        _set_plugin(p_type, preferred_plugins['all'])

    plugin_types = (p for p in preferred_plugins.keys() if p != 'all')
    for p_type in plugin_types:
        _set_plugin(p_type, preferred_plugins[p_type])


def _set_plugin(plugin_type, plugin_list):
    for plugin in plugin_list:
        if plugin not in available_plugins:
            continue
        try:
            use_plugin(plugin, kind=plugin_type)
            break
        except (ImportError, RuntimeError, OSError):
            pass


def reset_plugins():
    _clear_plugins()
    _load_preferred_plugins()


def _parse_config_file(filename):
    """Return plugin name and meta-data dict from plugin config file."""
    parser = ConfigParser()
    parser.read(filename)
    name = parser.sections()[0]

    meta_data = {}
    for opt in parser.options(name):
        meta_data[opt] = parser.get(name, opt)

    return name, meta_data


def _scan_plugins():
    """Scan the plugins directory for .ini files and parse them
    to gather plugin meta-data.
    """
    pd = os.path.dirname(__file__)
    config_files = glob(os.path.join(pd, '_plugins', '*.ini'))

    for filename in config_files:
        name, meta_data = _parse_config_file(filename)
        if 'provides' not in meta_data:
            warnings.warn(f'file {filename} not recognized as a scikit-image io plugin, skipping.')
            continue
        plugin_meta_data[name] = meta_data
        provides = [s.strip() for s in meta_data['provides'].split(',')]
        valid_provides = [p for p in provides if p in plugin_store]

        for p in provides:
            if p not in plugin_store:
                print("Plugin `%s` wants to provide non-existent `%s`."
                      " Ignoring." % (name, p))

        # Add plugins that provide 'imread' as provider of 'imread_collection'.
        need_to_add_collection = ('imread_collection' not in valid_provides and
                                  'imread' in valid_provides)
        if need_to_add_collection:
            valid_provides.append('imread_collection')

        plugin_provides[name] = valid_provides

        plugin_module_name[name] = os.path.basename(filename)[:-4]


_scan_plugins()


def find_available_plugins(loaded=False):
    """List available plugins.

    Parameters
    ----------
    loaded : bool
        If True, show only those plugins currently loaded.  By default,
        all plugins are shown.

    Returns
    -------
    p : dict
        Dictionary with plugin names as keys and exposed functions as
        values.

    """
    active_plugins = set()
    for plugin_func in plugin_store.values():
        for plugin, func in plugin_func:
            active_plugins.add(plugin)

    d = {}
    for plugin in plugin_provides:
        if not loaded or plugin in active_plugins:
            d[plugin] = [f for f in plugin_provides[plugin]
                         if not f.startswith('_')]

    return d


available_plugins = find_available_plugins()


def call_plugin(kind, *args, **kwargs):
    """Find the appropriate plugin of 'kind' and execute it.

    Parameters
    ----------
    kind : {'imshow', 'imsave', 'imread', 'imread_collection'}
        Function to look up.
    plugin : str, optional
        Plugin to load.  Defaults to None, in which case the first
        matching plugin is used.
    *args, **kwargs : arguments and keyword arguments
        Passed to the plugin function.

    """
    if kind not in plugin_store:
        raise ValueError('Invalid function (%s) requested.' % kind)

    plugin_funcs = plugin_store[kind]
    if len(plugin_funcs) == 0:
        msg = ("No suitable plugin registered for %s.\n\n"
               "You may load I/O plugins with the `skimage.io.use_plugin` "
               "command.  A list of all available plugins are shown in the "
               "`skimage.io` docstring.")
        raise RuntimeError(msg % kind)

    plugin = kwargs.pop('plugin', None)
    if plugin is None:
        _, func = plugin_funcs[0]
    else:
        _load(plugin)
        try:
            func = [f for (p, f) in plugin_funcs if p == plugin][0]
        except IndexError:
            raise RuntimeError('Could not find the plugin "%s" for %s.' %
                               (plugin, kind))

    return func(*args, **kwargs)


def use_plugin(name, kind=None):
    """Set the default plugin for a specified operation.  The plugin
    will be loaded if it hasn't been already.

    Parameters
    ----------
    name : str
        Name of plugin.
    kind : {'imsave', 'imread', 'imshow', 'imread_collection', 'imshow_collection'}, optional
        Set the plugin for this function.  By default,
        the plugin is set for all functions.

    See Also
    --------
    available_plugins : List of available plugins

    Examples
    --------
    To use Matplotlib as the default image reader, you would write:

    >>> from skimage import io
    >>> io.use_plugin('matplotlib', 'imread')

    To see a list of available plugins run ``io.available_plugins``. Note that
    this lists plugins that are defined, but the full list may not be usable
    if your system does not have the required libraries installed.

    """
    if kind is None:
        kind = plugin_store.keys()
    else:
        if kind not in plugin_provides[name]:
            raise RuntimeError("Plugin %s does not support `%s`." %
                               (name, kind))

        if kind == 'imshow':
            kind = [kind, '_app_show']
        else:
            kind = [kind]

    _load(name)

    for k in kind:
        if k not in plugin_store:
            raise RuntimeError("'%s' is not a known plugin function." % k)

        funcs = plugin_store[k]

        # Shuffle the plugins so that the requested plugin stands first
        # in line
        funcs = [(n, f) for (n, f) in funcs if n == name] + \
                [(n, f) for (n, f) in funcs if n != name]

        plugin_store[k] = funcs


def _inject_imread_collection_if_needed(module):
    """Add `imread_collection` to module if not already present."""
    if not hasattr(module, 'imread_collection') and hasattr(module, 'imread'):
        imread = getattr(module, 'imread')
        func = imread_collection_wrapper(imread)
        setattr(module, 'imread_collection', func)


def _load(plugin):
    """Load the given plugin.

    Parameters
    ----------
    plugin : str
        Name of plugin to load.

    See Also
    --------
    plugins : List of available plugins

    """
    if plugin in find_available_plugins(loaded=True):
        return
    if plugin not in plugin_module_name:
        raise ValueError("Plugin %s not found." % plugin)
    else:
        modname = plugin_module_name[plugin]
        plugin_module = __import__('skimage.io._plugins.' + modname,
                                   fromlist=[modname])

    provides = plugin_provides[plugin]
    for p in provides:
        if p == 'imread_collection':
            _inject_imread_collection_if_needed(plugin_module)
        elif not hasattr(plugin_module, p):
            print("Plugin %s does not provide %s as advertised.  Ignoring." %
                  (plugin, p))
            continue

        store = plugin_store[p]
        func = getattr(plugin_module, p)
        if not (plugin, func) in store:
            store.append((plugin, func))


def plugin_info(plugin):
    """Return plugin meta-data.

    Parameters
    ----------
    plugin : str
        Name of plugin.

    Returns
    -------
    m : dict
        Meta data as specified in plugin ``.ini``.

    """
    try:
        return plugin_meta_data[plugin]
    except KeyError:
        raise ValueError('No information on plugin "%s"' % plugin)


def plugin_order():
    """Return the currently preferred plugin order.

    Returns
    -------
    p : dict
        Dictionary of preferred plugin order, with function name as key and
        plugins (in order of preference) as value.

    """
    p = {}
    for func in plugin_store:
        p[func] = [plugin_name for (plugin_name, f) in plugin_store[func]]
    return p
