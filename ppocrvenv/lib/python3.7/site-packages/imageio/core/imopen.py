from pathlib import Path
from typing import Any, Union

from .request import (
    IOMode,
    Request,
    InitializationError,
    URI_FILENAME,
    SPECIAL_READ_URIS,
)
from ..config.plugins import PluginConfig
from ..config import known_plugins
from ..config.extensions import known_extensions


def _get_config(plugin: str, legacy_mode: bool) -> PluginConfig:
    """Look up the config for the given plugin name

    Factored out for legacy compatibility with FormatManager. Move
    back into imopen in V3.
    """

    # for v2 compatibility, delete in v3
    extension_name = None

    if legacy_mode and Path(plugin).suffix.lower() in known_extensions:
        # for v2 compatibility, delete in v3
        extension_name = Path(plugin).suffix.lower()
    elif plugin in known_plugins:
        pass
    elif not legacy_mode:
        raise ValueError(f"`{plugin}` is not a registered plugin name.")
    elif plugin.upper() in known_plugins:
        # for v2 compatibility, delete in v3
        plugin = plugin.upper()
    elif plugin.lower() in known_extensions:
        # for v2 compatibility, delete in v3
        extension_name = plugin.lower()
    elif "." + plugin.lower() in known_extensions:
        # for v2 compatibility, delete in v3
        extension_name = "." + plugin.lower()
    else:
        # for v2 compatibility, delete in v3
        raise IndexError(f"No format known by name `{plugin}`.")

    # for v2 compatibility, delete in v3
    if extension_name is not None:
        for plugin_name in [
            x
            for file_extension in known_extensions[extension_name]
            for x in file_extension.priority
        ]:
            if known_plugins[plugin_name].is_legacy:
                plugin = plugin_name
                break
        else:  # pragma: no cover
            # currently there is no format that is only supported
            # by v3 plugins
            raise IndexError(f"No format known by name `{plugin}`.")

    return known_plugins[plugin]


def imopen(
    uri,
    io_mode: str,
    *,
    plugin: Union[str, Any] = None,
    legacy_mode: bool = True,
    **kwargs,
) -> Any:
    """Open an ImageResource.

    .. warning::
        This warning is for pypy users. If you are not using a context manager,
        remember to deconstruct the returned plugin to avoid leaking the file
        handle to an unclosed file.

    Parameters
    ----------
    uri : str or pathlib.Path or bytes or file or Request
        The :doc:`ImageResources <getting_started.request>` to load the image
        from.
    io_mode : str
        The mode in which the file is opened. Possible values are::

            ``r`` - open the file for reading
            ``w`` - open the file for writing

        Depreciated since v2.9:
        A second character can be added to give the reader a hint on what
        the user expects. This will be ignored by new plugins and will
        only have an effect on legacy plugins. Possible values are::

            ``i`` for a single image,
            ``I`` for multiple images,
            ``v`` for a single volume,
            ``V`` for multiple volumes,
            ``?`` for don't care (default)

    plugin : str, Plugin, or None
        The plugin to use. If set to None (default) imopen will perform a
        search for a matching plugin.
    legacy_mode : bool
        If true (default) use the v2 behavior when searching for a suitable
        plugin. This will ignore v3 plugins and will check ``plugin``
        against known extensions if no plugin with the given name can be found.
    **kwargs : Any
        Additional keyword arguments will be passed to the plugin upon
        construction.

    Notes
    -----
    Registered plugins are controlled via the ``known_plugins`` dict in
    ``imageio.config``.

    Passing a ``Request`` as the uri is only supported if ``legacy_mode``
    is ``True``. In this case ``io_mode`` is ignored.

    Examples
    --------

    >>> import imageio.v3 as iio
    >>> with iio.imopen("/path/to/image.png", "r") as file:
    >>>     im = file.read()

    >>> with iio.imopen("/path/to/output.jpg", "w") as file:
    >>>     file.write(im)

    """

    if isinstance(uri, Request) and legacy_mode:
        request = uri
        uri = request.raw_uri
        io_mode = request.mode.io_mode
    else:
        request = Request(uri, io_mode)

    source = "<bytes>" if isinstance(uri, bytes) else uri

    # plugin specified, no search needed
    # (except in legacy mode)
    if plugin is not None:
        if isinstance(plugin, str):
            try:
                config = _get_config(plugin, legacy_mode)
            except (IndexError, ValueError):
                request.finish()
                raise

            def loader(request, **kwargs):
                return config.plugin_class(request, **kwargs)

        elif not legacy_mode:

            def loader(request, **kwargs):
                return plugin(request, **kwargs)

        else:
            raise ValueError("The `plugin` argument must be a string.")

        try:
            return loader(request, **kwargs)
        except InitializationError as class_specific:
            err_from = class_specific
            err_type = RuntimeError if legacy_mode else IOError
            err_msg = f"`{plugin}` can not handle the given uri."
        except ImportError:
            err_from = None
            err_type = ImportError
            err_msg = (
                f"The `{config.name}` plugin is not installed. "
                f"Use `pip install imageio[{config.install_name}]` to install it."
            )
        except Exception as generic_error:
            err_from = generic_error
            err_type = IOError
            err_msg = f"An unknown error occured while initializing plugin `{plugin}`."

        request.finish()
        raise err_type(err_msg) from err_from

    # fast-path based on file extension
    if request.extension in known_extensions:
        for candidate_format in known_extensions[request.extension]:
            for plugin_name in candidate_format.priority:
                config = known_plugins[plugin_name]

                # v2 compatibility; delete in v3
                if legacy_mode and not config.is_legacy:
                    continue

                try:
                    candidate_plugin = config.plugin_class
                except ImportError:
                    # not installed
                    continue

                try:
                    plugin_instance = candidate_plugin(request, **kwargs)
                except InitializationError:
                    # file extension doesn't match file type
                    continue

                return plugin_instance

    # error out for read-only special targets
    # this is again hacky; can we come up with a better solution for this?
    if request.mode.io_mode == IOMode.write:
        if isinstance(uri, str) and uri.startswith(SPECIAL_READ_URIS):
            err_type = ValueError if legacy_mode else IOError
            err_msg = f"`{source}` is read-only."
            raise err_type(err_msg)

    # error out for directories
    # this is a bit hacky and should be cleaned once we decide
    # how to gracefully handle DICOM
    if request._uri_type == URI_FILENAME and Path(request.raw_uri).is_dir():
        err_type = ValueError if legacy_mode else IOError
        err_msg = (
            "ImageIO does not generally support reading folders. "
            "Limited support may be available via specific plugins. "
            "Specify the plugin explicitly using the `plugin` kwarg, e.g. `plugin='DICOM'`"
        )
        raise err_type(err_msg)

    # fallback option: try all plugins
    for config in known_plugins.values():
        # Note: for v2 compatibility
        # this branch can be removed in ImageIO v3.0
        if legacy_mode and not config.is_legacy:
            continue

        try:
            plugin_instance = config.plugin_class(request, **kwargs)
        except InitializationError:
            continue
        except ImportError:
            continue
        else:
            return plugin_instance

    err_type = ValueError if legacy_mode else IOError
    err_msg = f"Could not find a backend to open `{source}`` with iomode `{io_mode}`."

    # check if a missing plugin could help
    if request.extension in known_extensions:
        missing_plugins = list()

        formats = known_extensions[request.extension]
        plugin_names = [
            plugin for file_format in formats for plugin in file_format.priority
        ]
        for name in plugin_names:
            config = known_plugins[name]

            try:
                config.plugin_class
                continue
            except ImportError:
                missing_plugins.append(config)

        if len(missing_plugins) > 0:
            install_candidates = "\n".join(
                [
                    (
                        f"  {config.name}:  "
                        f"pip install imageio[{config.install_name}]"
                    )
                    for config in missing_plugins
                ]
            )
            err_msg += (
                "\nBased on the extension, the following plugins might add capable backends:\n"
                f"{install_candidates}"
            )

    request.finish()
    raise err_type(err_msg)
