import inspect
import sys
import warnings

from . import _tracing
from ._callers import _Result, _multicall
from ._hooks import HookImpl, _HookRelay, _HookCaller, normalize_hookimpl_opts

if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata


def _warn_for_function(warning, function):
    warnings.warn_explicit(
        warning,
        type(warning),
        lineno=function.__code__.co_firstlineno,
        filename=function.__code__.co_filename,
    )


class PluginValidationError(Exception):
    """plugin failed validation.

    :param object plugin: the plugin which failed validation,
        may be a module or an arbitrary object.
    """

    def __init__(self, plugin, message):
        self.plugin = plugin
        super(Exception, self).__init__(message)


class DistFacade:
    """Emulate a pkg_resources Distribution"""

    def __init__(self, dist):
        self._dist = dist

    @property
    def project_name(self):
        return self.metadata["name"]

    def __getattr__(self, attr, default=None):
        return getattr(self._dist, attr, default)

    def __dir__(self):
        return sorted(dir(self._dist) + ["_dist", "project_name"])


class PluginManager:
    """Core :py:class:`.PluginManager` class which manages registration
    of plugin objects and 1:N hook calling.

    You can register new hooks by calling :py:meth:`add_hookspecs(module_or_class)
    <.PluginManager.add_hookspecs>`.
    You can register plugin objects (which contain hooks) by calling
    :py:meth:`register(plugin) <.PluginManager.register>`.  The :py:class:`.PluginManager`
    is initialized with a prefix that is searched for in the names of the dict
    of registered plugin objects.

    For debugging purposes you can call :py:meth:`.PluginManager.enable_tracing`
    which will subsequently send debug information to the trace helper.
    """

    def __init__(self, project_name):
        self.project_name = project_name
        self._name2plugin = {}
        self._plugin2hookcallers = {}
        self._plugin_distinfo = []
        self.trace = _tracing.TagTracer().get("pluginmanage")
        self.hook = _HookRelay()
        self._inner_hookexec = _multicall

    def _hookexec(self, hook_name, methods, kwargs, firstresult):
        # called from all hookcaller instances.
        # enable_tracing will set its own wrapping function at self._inner_hookexec
        return self._inner_hookexec(hook_name, methods, kwargs, firstresult)

    def register(self, plugin, name=None):
        """Register a plugin and return its canonical name or ``None`` if the name
        is blocked from registering.  Raise a :py:class:`ValueError` if the plugin
        is already registered."""
        plugin_name = name or self.get_canonical_name(plugin)

        if plugin_name in self._name2plugin or plugin in self._plugin2hookcallers:
            if self._name2plugin.get(plugin_name, -1) is None:
                return  # blocked plugin, return None to indicate no registration
            raise ValueError(
                "Plugin already registered: %s=%s\n%s"
                % (plugin_name, plugin, self._name2plugin)
            )

        # XXX if an error happens we should make sure no state has been
        # changed at point of return
        self._name2plugin[plugin_name] = plugin

        # register matching hook implementations of the plugin
        self._plugin2hookcallers[plugin] = hookcallers = []
        for name in dir(plugin):
            hookimpl_opts = self.parse_hookimpl_opts(plugin, name)
            if hookimpl_opts is not None:
                normalize_hookimpl_opts(hookimpl_opts)
                method = getattr(plugin, name)
                hookimpl = HookImpl(plugin, plugin_name, method, hookimpl_opts)
                name = hookimpl_opts.get("specname") or name
                hook = getattr(self.hook, name, None)
                if hook is None:
                    hook = _HookCaller(name, self._hookexec)
                    setattr(self.hook, name, hook)
                elif hook.has_spec():
                    self._verify_hook(hook, hookimpl)
                    hook._maybe_apply_history(hookimpl)
                hook._add_hookimpl(hookimpl)
                hookcallers.append(hook)
        return plugin_name

    def parse_hookimpl_opts(self, plugin, name):
        method = getattr(plugin, name)
        if not inspect.isroutine(method):
            return
        try:
            res = getattr(method, self.project_name + "_impl", None)
        except Exception:
            res = {}
        if res is not None and not isinstance(res, dict):
            # false positive
            res = None
        return res

    def unregister(self, plugin=None, name=None):
        """unregister a plugin object and all its contained hook implementations
        from internal data structures."""
        if name is None:
            assert plugin is not None, "one of name or plugin needs to be specified"
            name = self.get_name(plugin)

        if plugin is None:
            plugin = self.get_plugin(name)

        # if self._name2plugin[name] == None registration was blocked: ignore
        if self._name2plugin.get(name):
            del self._name2plugin[name]

        for hookcaller in self._plugin2hookcallers.pop(plugin, []):
            hookcaller._remove_plugin(plugin)

        return plugin

    def set_blocked(self, name):
        """block registrations of the given name, unregister if already registered."""
        self.unregister(name=name)
        self._name2plugin[name] = None

    def is_blocked(self, name):
        """return ``True`` if the given plugin name is blocked."""
        return name in self._name2plugin and self._name2plugin[name] is None

    def add_hookspecs(self, module_or_class):
        """add new hook specifications defined in the given ``module_or_class``.
        Functions are recognized if they have been decorated accordingly."""
        names = []
        for name in dir(module_or_class):
            spec_opts = self.parse_hookspec_opts(module_or_class, name)
            if spec_opts is not None:
                hc = getattr(self.hook, name, None)
                if hc is None:
                    hc = _HookCaller(name, self._hookexec, module_or_class, spec_opts)
                    setattr(self.hook, name, hc)
                else:
                    # plugins registered this hook without knowing the spec
                    hc.set_specification(module_or_class, spec_opts)
                    for hookfunction in hc.get_hookimpls():
                        self._verify_hook(hc, hookfunction)
                names.append(name)

        if not names:
            raise ValueError(
                f"did not find any {self.project_name!r} hooks in {module_or_class!r}"
            )

    def parse_hookspec_opts(self, module_or_class, name):
        method = getattr(module_or_class, name)
        return getattr(method, self.project_name + "_spec", None)

    def get_plugins(self):
        """return the set of registered plugins."""
        return set(self._plugin2hookcallers)

    def is_registered(self, plugin):
        """Return ``True`` if the plugin is already registered."""
        return plugin in self._plugin2hookcallers

    def get_canonical_name(self, plugin):
        """Return canonical name for a plugin object. Note that a plugin
        may be registered under a different name which was specified
        by the caller of :py:meth:`register(plugin, name) <.PluginManager.register>`.
        To obtain the name of an registered plugin use :py:meth:`get_name(plugin)
        <.PluginManager.get_name>` instead."""
        return getattr(plugin, "__name__", None) or str(id(plugin))

    def get_plugin(self, name):
        """Return a plugin or ``None`` for the given name."""
        return self._name2plugin.get(name)

    def has_plugin(self, name):
        """Return ``True`` if a plugin with the given name is registered."""
        return self.get_plugin(name) is not None

    def get_name(self, plugin):
        """Return name for registered plugin or ``None`` if not registered."""
        for name, val in self._name2plugin.items():
            if plugin == val:
                return name

    def _verify_hook(self, hook, hookimpl):
        if hook.is_historic() and hookimpl.hookwrapper:
            raise PluginValidationError(
                hookimpl.plugin,
                "Plugin %r\nhook %r\nhistoric incompatible to hookwrapper"
                % (hookimpl.plugin_name, hook.name),
            )

        if hook.spec.warn_on_impl:
            _warn_for_function(hook.spec.warn_on_impl, hookimpl.function)

        # positional arg checking
        notinspec = set(hookimpl.argnames) - set(hook.spec.argnames)
        if notinspec:
            raise PluginValidationError(
                hookimpl.plugin,
                "Plugin %r for hook %r\nhookimpl definition: %s\n"
                "Argument(s) %s are declared in the hookimpl but "
                "can not be found in the hookspec"
                % (
                    hookimpl.plugin_name,
                    hook.name,
                    _formatdef(hookimpl.function),
                    notinspec,
                ),
            )

        if hookimpl.hookwrapper and not inspect.isgeneratorfunction(hookimpl.function):
            raise PluginValidationError(
                hookimpl.plugin,
                "Plugin %r for hook %r\nhookimpl definition: %s\n"
                "Declared as hookwrapper=True but function is not a generator function"
                % (hookimpl.plugin_name, hook.name, _formatdef(hookimpl.function)),
            )

    def check_pending(self):
        """Verify that all hooks which have not been verified against
        a hook specification are optional, otherwise raise :py:class:`.PluginValidationError`."""
        for name in self.hook.__dict__:
            if name[0] != "_":
                hook = getattr(self.hook, name)
                if not hook.has_spec():
                    for hookimpl in hook.get_hookimpls():
                        if not hookimpl.optionalhook:
                            raise PluginValidationError(
                                hookimpl.plugin,
                                "unknown hook %r in plugin %r"
                                % (name, hookimpl.plugin),
                            )

    def load_setuptools_entrypoints(self, group, name=None):
        """Load modules from querying the specified setuptools ``group``.

        :param str group: entry point group to load plugins
        :param str name: if given, loads only plugins with the given ``name``.
        :rtype: int
        :return: return the number of loaded plugins by this call.
        """
        count = 0
        for dist in list(importlib_metadata.distributions()):
            for ep in dist.entry_points:
                if (
                    ep.group != group
                    or (name is not None and ep.name != name)
                    # already registered
                    or self.get_plugin(ep.name)
                    or self.is_blocked(ep.name)
                ):
                    continue
                plugin = ep.load()
                self.register(plugin, name=ep.name)
                self._plugin_distinfo.append((plugin, DistFacade(dist)))
                count += 1
        return count

    def list_plugin_distinfo(self):
        """return list of distinfo/plugin tuples for all setuptools registered
        plugins."""
        return list(self._plugin_distinfo)

    def list_name_plugin(self):
        """return list of name/plugin pairs."""
        return list(self._name2plugin.items())

    def get_hookcallers(self, plugin):
        """get all hook callers for the specified plugin."""
        return self._plugin2hookcallers.get(plugin)

    def add_hookcall_monitoring(self, before, after):
        """add before/after tracing functions for all hooks
        and return an undo function which, when called,
        will remove the added tracers.

        ``before(hook_name, hook_impls, kwargs)`` will be called ahead
        of all hook calls and receive a hookcaller instance, a list
        of HookImpl instances and the keyword arguments for the hook call.

        ``after(outcome, hook_name, hook_impls, kwargs)`` receives the
        same arguments as ``before`` but also a :py:class:`pluggy._callers._Result` object
        which represents the result of the overall hook call.
        """
        oldcall = self._inner_hookexec

        def traced_hookexec(hook_name, hook_impls, kwargs, firstresult):
            before(hook_name, hook_impls, kwargs)
            outcome = _Result.from_call(
                lambda: oldcall(hook_name, hook_impls, kwargs, firstresult)
            )
            after(outcome, hook_name, hook_impls, kwargs)
            return outcome.get_result()

        self._inner_hookexec = traced_hookexec

        def undo():
            self._inner_hookexec = oldcall

        return undo

    def enable_tracing(self):
        """enable tracing of hook calls and return an undo function."""
        hooktrace = self.trace.root.get("hook")

        def before(hook_name, methods, kwargs):
            hooktrace.root.indent += 1
            hooktrace(hook_name, kwargs)

        def after(outcome, hook_name, methods, kwargs):
            if outcome.excinfo is None:
                hooktrace("finish", hook_name, "-->", outcome.get_result())
            hooktrace.root.indent -= 1

        return self.add_hookcall_monitoring(before, after)

    def subset_hook_caller(self, name, remove_plugins):
        """Return a new :py:class:`._hooks._HookCaller` instance for the named method
        which manages calls to all registered plugins except the
        ones from remove_plugins."""
        orig = getattr(self.hook, name)
        plugins_to_remove = [plug for plug in remove_plugins if hasattr(plug, name)]
        if plugins_to_remove:
            hc = _HookCaller(
                orig.name, orig._hookexec, orig.spec.namespace, orig.spec.opts
            )
            for hookimpl in orig.get_hookimpls():
                plugin = hookimpl.plugin
                if plugin not in plugins_to_remove:
                    hc._add_hookimpl(hookimpl)
                    # we also keep track of this hook caller so it
                    # gets properly removed on plugin unregistration
                    self._plugin2hookcallers.setdefault(plugin, []).append(hc)
            return hc
        return orig


def _formatdef(func):
    return f"{func.__name__}{inspect.signature(func)}"
