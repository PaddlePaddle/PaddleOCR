"""
Core functions and attributes for the matplotlib style library:

``use``
    Select style sheet to override the current matplotlib settings.
``context``
    Context manager to use a style sheet temporarily.
``available``
    List available style sheets.
``library``
    A dictionary of style names and matplotlib settings.
"""

import contextlib
import logging
import os
from pathlib import Path
import re
import warnings

import matplotlib as mpl
from matplotlib import _api, docstring, rc_params_from_file, rcParamsDefault

_log = logging.getLogger(__name__)

__all__ = ['use', 'context', 'available', 'library', 'reload_library']


@_api.caching_module_getattr  # module-level deprecations
class __getattr__:
    STYLE_FILE_PATTERN = _api.deprecated("3.5", obj_type="")(property(
        lambda self: re.compile(r'([\S]+).%s$' % STYLE_EXTENSION)))


BASE_LIBRARY_PATH = os.path.join(mpl.get_data_path(), 'stylelib')
# Users may want multiple library paths, so store a list of paths.
USER_LIBRARY_PATHS = [os.path.join(mpl.get_configdir(), 'stylelib')]
STYLE_EXTENSION = 'mplstyle'
# A list of rcParams that should not be applied from styles
STYLE_BLACKLIST = {
    'interactive', 'backend', 'webagg.port', 'webagg.address',
    'webagg.port_retries', 'webagg.open_in_browser', 'backend_fallback',
    'toolbar', 'timezone', 'datapath', 'figure.max_open_warning',
    'figure.raise_window', 'savefig.directory', 'tk.window_focus',
    'docstring.hardcopy', 'date.epoch'}


def _remove_blacklisted_style_params(d, warn=True):
    o = {}
    for key in d:  # prevent triggering RcParams.__getitem__('backend')
        if key in STYLE_BLACKLIST:
            if warn:
                _api.warn_external(
                    f"Style includes a parameter, {key!r}, that is not "
                    "related to style.  Ignoring this parameter.")
        else:
            o[key] = d[key]
    return o


def _apply_style(d, warn=True):
    mpl.rcParams.update(_remove_blacklisted_style_params(d, warn=warn))


@docstring.Substitution(
    "\n".join(map("- {}".format, sorted(STYLE_BLACKLIST, key=str.lower)))
)
def use(style):
    """
    Use Matplotlib style settings from a style specification.

    The style name of 'default' is reserved for reverting back to
    the default style settings.

    .. note::

       This updates the `.rcParams` with the settings from the style.
       `.rcParams` not defined in the style are kept.

    Parameters
    ----------
    style : str, dict, Path or list
        A style specification. Valid options are:

        +------+-------------------------------------------------------------+
        | str  | The name of a style or a path/URL to a style file. For a    |
        |      | list of available style names, see `.style.available`.      |
        +------+-------------------------------------------------------------+
        | dict | Dictionary with valid key/value pairs for                   |
        |      | `matplotlib.rcParams`.                                      |
        +------+-------------------------------------------------------------+
        | Path | A path-like object which is a path to a style file.         |
        +------+-------------------------------------------------------------+
        | list | A list of style specifiers (str, Path or dict) applied from |
        |      | first to last in the list.                                  |
        +------+-------------------------------------------------------------+

    Notes
    -----
    The following `.rcParams` are not related to style and will be ignored if
    found in a style specification:

    %s
    """
    style_alias = {'mpl20': 'default',
                   'mpl15': 'classic'}
    if isinstance(style, (str, Path)) or hasattr(style, 'keys'):
        # If name is a single str, Path or dict, make it a single element list.
        styles = [style]
    else:
        styles = style

    styles = (style_alias.get(s, s) if isinstance(s, str) else s
              for s in styles)
    for style in styles:
        if not isinstance(style, (str, Path)):
            _apply_style(style)
        elif style == 'default':
            # Deprecation warnings were already handled when creating
            # rcParamsDefault, no need to reemit them here.
            with _api.suppress_matplotlib_deprecation_warning():
                _apply_style(rcParamsDefault, warn=False)
        elif style in library:
            _apply_style(library[style])
        else:
            try:
                rc = rc_params_from_file(style, use_default_template=False)
                _apply_style(rc)
            except IOError as err:
                raise IOError(
                    "{!r} not found in the style library and input is not a "
                    "valid URL or path; see `style.available` for list of "
                    "available styles".format(style)) from err


@contextlib.contextmanager
def context(style, after_reset=False):
    """
    Context manager for using style settings temporarily.

    Parameters
    ----------
    style : str, dict, Path or list
        A style specification. Valid options are:

        +------+-------------------------------------------------------------+
        | str  | The name of a style or a path/URL to a style file. For a    |
        |      | list of available style names, see `.style.available`.      |
        +------+-------------------------------------------------------------+
        | dict | Dictionary with valid key/value pairs for                   |
        |      | `matplotlib.rcParams`.                                      |
        +------+-------------------------------------------------------------+
        | Path | A path-like object which is a path to a style file.         |
        +------+-------------------------------------------------------------+
        | list | A list of style specifiers (str, Path or dict) applied from |
        |      | first to last in the list.                                  |
        +------+-------------------------------------------------------------+

    after_reset : bool
        If True, apply style after resetting settings to their defaults;
        otherwise, apply style on top of the current settings.
    """
    with mpl.rc_context():
        if after_reset:
            mpl.rcdefaults()
        use(style)
        yield


@_api.deprecated("3.5")
def load_base_library():
    """Load style library defined in this package."""
    library = read_style_directory(BASE_LIBRARY_PATH)
    return library


@_api.deprecated("3.5")
def iter_user_libraries():
    for stylelib_path in USER_LIBRARY_PATHS:
        stylelib_path = os.path.expanduser(stylelib_path)
        if os.path.exists(stylelib_path) and os.path.isdir(stylelib_path):
            yield stylelib_path


def update_user_library(library):
    """Update style library with user-defined rc files."""
    for stylelib_path in map(os.path.expanduser, USER_LIBRARY_PATHS):
        styles = read_style_directory(stylelib_path)
        update_nested_dict(library, styles)
    return library


def read_style_directory(style_dir):
    """Return dictionary of styles defined in *style_dir*."""
    styles = dict()
    for path in Path(style_dir).glob(f"*.{STYLE_EXTENSION}"):
        with warnings.catch_warnings(record=True) as warns:
            styles[path.stem] = rc_params_from_file(
                path, use_default_template=False)
        for w in warns:
            _log.warning('In %s: %s', path, w.message)
    return styles


def update_nested_dict(main_dict, new_dict):
    """
    Update nested dict (only level of nesting) with new values.

    Unlike `dict.update`, this assumes that the values of the parent dict are
    dicts (or dict-like), so you shouldn't replace the nested dict if it
    already exists. Instead you should update the sub-dict.
    """
    # update named styles specified by user
    for name, rc_dict in new_dict.items():
        main_dict.setdefault(name, {}).update(rc_dict)
    return main_dict


# Load style library
# ==================
_base_library = read_style_directory(BASE_LIBRARY_PATH)
library = None
available = []


def reload_library():
    """Reload the style library."""
    global library
    library = update_user_library(_base_library)
    available[:] = sorted(library.keys())


reload_library()
