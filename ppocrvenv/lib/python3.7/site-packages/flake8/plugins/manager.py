"""Plugin loading and management logic and classes."""
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from flake8 import exceptions
from flake8 import utils
from flake8._compat import importlib_metadata

LOG = logging.getLogger(__name__)

__all__ = ("Checkers", "Plugin", "PluginManager", "ReportFormatters")

NO_GROUP_FOUND = object()


class Plugin:
    """Wrap an EntryPoint from setuptools and other logic."""

    def __init__(self, name, entry_point, local=False):
        """Initialize our Plugin.

        :param str name:
            Name of the entry-point as it was registered with setuptools.
        :param entry_point:
            EntryPoint returned by setuptools.
        :type entry_point:
            setuptools.EntryPoint
        :param bool local:
            Is this a repo-local plugin?
        """
        self.name = name
        self.entry_point = entry_point
        self.local = local
        self._plugin: Any = None
        self._parameters = None
        self._parameter_names: Optional[List[str]] = None
        self._group = None
        self._plugin_name = None
        self._version = None

    def __repr__(self) -> str:
        """Provide an easy to read description of the current plugin."""
        return 'Plugin(name="{}", entry_point="{}")'.format(
            self.name, self.entry_point.value
        )

    def to_dictionary(self):
        """Convert this plugin to a dictionary."""
        return {
            "name": self.name,
            "parameters": self.parameters,
            "parameter_names": self.parameter_names,
            "plugin": self.plugin,
            "plugin_name": self.plugin_name,
        }

    def is_in_a_group(self):
        """Determine if this plugin is in a group.

        :returns:
            True if the plugin is in a group, otherwise False.
        :rtype:
            bool
        """
        return self.group() is not None

    def group(self):
        """Find and parse the group the plugin is in."""
        if self._group is None:
            name = self.name.split(".", 1)
            if len(name) > 1:
                self._group = name[0]
            else:
                self._group = NO_GROUP_FOUND
        if self._group is NO_GROUP_FOUND:
            return None
        return self._group

    @property
    def parameters(self):
        """List of arguments that need to be passed to the plugin."""
        if self._parameters is None:
            self._parameters = utils.parameters_for(self)
        return self._parameters

    @property
    def parameter_names(self) -> List[str]:
        """List of argument names that need to be passed to the plugin."""
        if self._parameter_names is None:
            self._parameter_names = list(self.parameters)
        return self._parameter_names

    @property
    def plugin(self):
        """Load and return the plugin associated with the entry-point.

        This property implicitly loads the plugin and then caches it.
        """
        self.load_plugin()
        return self._plugin

    @property
    def version(self) -> str:
        """Return the version of the plugin."""
        version = self._version
        if version is None:
            if self.is_in_a_group():
                version = self._version = version_for(self)
            else:
                version = self._version = self.plugin.version
        return version

    @property
    def plugin_name(self):
        """Return the name of the plugin."""
        if self._plugin_name is None:
            if self.is_in_a_group():
                self._plugin_name = self.group()
            else:
                self._plugin_name = self.plugin.name

        return self._plugin_name

    @property
    def off_by_default(self):
        """Return whether the plugin is ignored by default."""
        return getattr(self.plugin, "off_by_default", False)

    def execute(self, *args, **kwargs):
        r"""Call the plugin with \*args and \*\*kwargs."""
        return self.plugin(*args, **kwargs)  # pylint: disable=not-callable

    def _load(self):
        self._plugin = self.entry_point.load()
        if not callable(self._plugin):
            msg = (
                f"Plugin {self._plugin!r} is not a callable. It might be "
                f"written for an older version of flake8 and might not work "
                f"with this version"
            )
            LOG.critical(msg)
            raise TypeError(msg)

    def load_plugin(self):
        """Retrieve the plugin for this entry-point.

        This loads the plugin, stores it on the instance and then returns it.
        It does not reload it after the first time, it merely returns the
        cached plugin.

        :returns:
            Nothing
        """
        if self._plugin is None:
            LOG.info('Loading plugin "%s" from entry-point.', self.name)
            try:
                self._load()
            except Exception as load_exception:
                LOG.exception(load_exception)
                failed_to_load = exceptions.FailedToLoadPlugin(
                    plugin_name=self.name, exception=load_exception
                )
                LOG.critical(str(failed_to_load))
                raise failed_to_load

    def enable(self, optmanager, options=None):
        """Remove plugin name from the default ignore list."""
        optmanager.remove_from_default_ignore([self.name])
        optmanager.extend_default_select([self.name])
        if not options:
            return
        try:
            options.ignore.remove(self.name)
        except (ValueError, KeyError):
            LOG.debug(
                "Attempted to remove %s from the ignore list but it was "
                "not a member of the list.",
                self.name,
            )

    def disable(self, optmanager):
        """Add the plugin name to the default ignore list."""
        optmanager.extend_default_ignore([self.name])

    def provide_options(self, optmanager, options, extra_args):
        """Pass the parsed options and extra arguments to the plugin."""
        parse_options = getattr(self.plugin, "parse_options", None)
        if parse_options is not None:
            LOG.debug('Providing options to plugin "%s".', self.name)
            try:
                parse_options(optmanager, options, extra_args)
            except TypeError:
                parse_options(options)

        if self.name in options.enable_extensions:
            self.enable(optmanager, options)

    def register_options(self, optmanager):
        """Register the plugin's command-line options on the OptionManager.

        :param optmanager:
            Instantiated OptionManager to register options on.
        :type optmanager:
            flake8.options.manager.OptionManager
        :returns:
            Nothing
        """
        add_options = getattr(self.plugin, "add_options", None)
        if add_options is not None:
            LOG.debug(
                'Registering options from plugin "%s" on OptionManager %r',
                self.name,
                optmanager,
            )
            with optmanager.group(self.plugin_name):
                add_options(optmanager)

        if self.off_by_default:
            self.disable(optmanager)


class PluginManager:  # pylint: disable=too-few-public-methods
    """Find and manage plugins consistently."""

    def __init__(
        self, namespace: str, local_plugins: Optional[List[str]] = None
    ) -> None:
        """Initialize the manager.

        :param str namespace:
            Namespace of the plugins to manage, e.g., 'flake8.extension'.
        :param list local_plugins:
            Plugins from config (as "X = path.to:Plugin" strings).
        """
        self.namespace = namespace
        self.plugins: Dict[str, Plugin] = {}
        self.names: List[str] = []
        self._load_local_plugins(local_plugins or [])
        self._load_entrypoint_plugins()

    def _load_local_plugins(self, local_plugins):
        """Load local plugins from config.

        :param list local_plugins:
            Plugins from config (as "X = path.to:Plugin" strings).
        """
        for plugin_str in local_plugins:
            name, _, entry_str = plugin_str.partition("=")
            name, entry_str = name.strip(), entry_str.strip()
            entry_point = importlib_metadata.EntryPoint(
                name, entry_str, self.namespace
            )
            self._load_plugin_from_entrypoint(entry_point, local=True)

    def _load_entrypoint_plugins(self):
        LOG.info('Loading entry-points for "%s".', self.namespace)
        eps = importlib_metadata.entry_points().get(self.namespace, ())
        # python2.7 occasionally gives duplicate results due to redundant
        # `local/lib` -> `../lib` symlink on linux in virtualenvs so we
        # eliminate duplicates here
        for entry_point in sorted(frozenset(eps)):
            if entry_point.name == "per-file-ignores":
                LOG.warning(
                    "flake8-per-file-ignores plugin is incompatible with "
                    "flake8>=3.7 (which implements per-file-ignores itself)."
                )
                continue
            self._load_plugin_from_entrypoint(entry_point)

    def _load_plugin_from_entrypoint(self, entry_point, local=False):
        """Load a plugin from a setuptools EntryPoint.

        :param EntryPoint entry_point:
            EntryPoint to load plugin from.
        :param bool local:
            Is this a repo-local plugin?
        """
        name = entry_point.name
        self.plugins[name] = Plugin(name, entry_point, local=local)
        self.names.append(name)
        LOG.debug('Loaded %r for plugin "%s".', self.plugins[name], name)

    def map(self, func, *args, **kwargs):
        r"""Call ``func`` with the plugin and \*args and \**kwargs after.

        This yields the return value from ``func`` for each plugin.

        :param collections.Callable func:
            Function to call with each plugin. Signature should at least be:

            .. code-block:: python

                def myfunc(plugin):
                     pass

            Any extra positional or keyword arguments specified with map will
            be passed along to this function after the plugin. The plugin
            passed is a :class:`~flake8.plugins.manager.Plugin`.
        :param args:
            Positional arguments to pass to ``func`` after each plugin.
        :param kwargs:
            Keyword arguments to pass to ``func`` after each plugin.
        """
        for name in self.names:
            yield func(self.plugins[name], *args, **kwargs)

    def versions(self):
        # () -> (str, str)
        """Generate the versions of plugins.

        :returns:
            Tuples of the plugin_name and version
        :rtype:
            tuple
        """
        plugins_seen: Set[str] = set()
        for entry_point_name in self.names:
            plugin = self.plugins[entry_point_name]
            plugin_name = plugin.plugin_name
            if plugin.plugin_name in plugins_seen:
                continue
            plugins_seen.add(plugin_name)
            yield (plugin_name, plugin.version)


def version_for(plugin):
    # (Plugin) -> Optional[str]
    """Determine the version of a plugin by its module.

    :param plugin:
        The loaded plugin
    :type plugin:
        Plugin
    :returns:
        version string for the module
    :rtype:
        str
    """
    module_name = plugin.plugin.__module__
    try:
        module = __import__(module_name)
    except ImportError:
        return None

    return getattr(module, "__version__", None)


class PluginTypeManager:
    """Parent class for most of the specific plugin types."""

    namespace: str

    def __init__(self, local_plugins=None):
        """Initialize the plugin type's manager.

        :param list local_plugins:
            Plugins from config file instead of entry-points
        """
        self.manager = PluginManager(
            self.namespace, local_plugins=local_plugins
        )
        self.plugins_loaded = False

    def __contains__(self, name):
        """Check if the entry-point name is in this plugin type manager."""
        LOG.debug('Checking for "%s" in plugin type manager.', name)
        return name in self.plugins

    def __getitem__(self, name):
        """Retrieve a plugin by its name."""
        LOG.debug('Retrieving plugin for "%s".', name)
        return self.plugins[name]

    def get(self, name, default=None):
        """Retrieve the plugin referred to by ``name`` or return the default.

        :param str name:
            Name of the plugin to retrieve.
        :param default:
            Default value to return.
        :returns:
            Plugin object referred to by name, if it exists.
        :rtype:
            :class:`Plugin`
        """
        if name in self:
            return self[name]
        return default

    @property
    def names(self):
        """Proxy attribute to underlying manager."""
        return self.manager.names

    @property
    def plugins(self):
        """Proxy attribute to underlying manager."""
        return self.manager.plugins

    @staticmethod
    def _generate_call_function(method_name, optmanager, *args, **kwargs):
        def generated_function(plugin):
            method = getattr(plugin, method_name, None)
            if method is not None and callable(method):
                return method(optmanager, *args, **kwargs)

        return generated_function

    def load_plugins(self):
        """Load all plugins of this type that are managed by this manager."""
        if self.plugins_loaded:
            return

        def load_plugin(plugin):
            """Call each plugin's load_plugin method."""
            return plugin.load_plugin()

        plugins = list(self.manager.map(load_plugin))
        # Do not set plugins_loaded if we run into an exception
        self.plugins_loaded = True
        return plugins

    def register_plugin_versions(self, optmanager):
        """Register the plugins and their versions with the OptionManager."""
        self.load_plugins()
        for (plugin_name, version) in self.manager.versions():
            optmanager.register_plugin(name=plugin_name, version=version)

    def register_options(self, optmanager):
        """Register all of the checkers' options to the OptionManager."""
        self.load_plugins()
        call_register_options = self._generate_call_function(
            "register_options", optmanager
        )

        list(self.manager.map(call_register_options))

    def provide_options(self, optmanager, options, extra_args):
        """Provide parsed options and extra arguments to the plugins."""
        call_provide_options = self._generate_call_function(
            "provide_options", optmanager, options, extra_args
        )

        list(self.manager.map(call_provide_options))


class Checkers(PluginTypeManager):
    """All of the checkers registered through entry-points or config."""

    namespace = "flake8.extension"

    def checks_expecting(self, argument_name):
        """Retrieve checks that expect an argument with the specified name.

        Find all checker plugins that are expecting a specific argument.
        """
        for plugin in self.plugins.values():
            if argument_name == plugin.parameter_names[0]:
                yield plugin

    def to_dictionary(self):
        """Return a dictionary of AST and line-based plugins."""
        return {
            "ast_plugins": [
                plugin.to_dictionary() for plugin in self.ast_plugins
            ],
            "logical_line_plugins": [
                plugin.to_dictionary() for plugin in self.logical_line_plugins
            ],
            "physical_line_plugins": [
                plugin.to_dictionary() for plugin in self.physical_line_plugins
            ],
        }

    def register_options(self, optmanager):
        """Register all of the checkers' options to the OptionManager.

        This also ensures that plugins that are not part of a group and are
        enabled by default are enabled on the option manager.
        """
        # NOTE(sigmavirus24) We reproduce a little of
        # PluginTypeManager.register_options to reduce the number of times
        # that we loop over the list of plugins. Instead of looping twice,
        # option registration and enabling the plugin, we loop once with one
        # function to map over the plugins.
        self.load_plugins()
        call_register_options = self._generate_call_function(
            "register_options", optmanager
        )

        def register_and_enable(plugin):
            call_register_options(plugin)
            if plugin.group() is None and not plugin.off_by_default:
                plugin.enable(optmanager)

        list(self.manager.map(register_and_enable))

    @property
    def ast_plugins(self):
        """List of plugins that expect the AST tree."""
        plugins = getattr(self, "_ast_plugins", [])
        if not plugins:
            plugins = list(self.checks_expecting("tree"))
            self._ast_plugins = plugins
        return plugins

    @property
    def logical_line_plugins(self):
        """List of plugins that expect the logical lines."""
        plugins = getattr(self, "_logical_line_plugins", [])
        if not plugins:
            plugins = list(self.checks_expecting("logical_line"))
            self._logical_line_plugins = plugins
        return plugins

    @property
    def physical_line_plugins(self):
        """List of plugins that expect the physical lines."""
        plugins = getattr(self, "_physical_line_plugins", [])
        if not plugins:
            plugins = list(self.checks_expecting("physical_line"))
            self._physical_line_plugins = plugins
        return plugins


class ReportFormatters(PluginTypeManager):
    """All of the report formatters registered through entry-points/config."""

    namespace = "flake8.report"
