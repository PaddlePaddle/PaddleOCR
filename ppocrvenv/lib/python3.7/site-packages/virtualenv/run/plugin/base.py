from __future__ import absolute_import, unicode_literals

import sys
from collections import OrderedDict

if sys.version_info >= (3, 8):
    from importlib.metadata import entry_points

    importlib_metadata_version = ()
else:
    from importlib_metadata import entry_points, version

    importlib_metadata_version = tuple(int(i) for i in version("importlib_metadata").split(".")[:2])


class PluginLoader(object):
    _OPTIONS = None
    _ENTRY_POINTS = None

    @classmethod
    def entry_points_for(cls, key):
        if sys.version_info >= (3, 10) or importlib_metadata_version >= (3, 6):
            return OrderedDict((e.name, e.load()) for e in cls.entry_points().select(group=key))
        else:
            return OrderedDict((e.name, e.load()) for e in cls.entry_points().get(key, {}))

    @staticmethod
    def entry_points():
        if PluginLoader._ENTRY_POINTS is None:
            PluginLoader._ENTRY_POINTS = entry_points()
        return PluginLoader._ENTRY_POINTS


class ComponentBuilder(PluginLoader):
    def __init__(self, interpreter, parser, name, possible):
        self.interpreter = interpreter
        self.name = name
        self._impl_class = None
        self.possible = possible
        self.parser = parser.add_argument_group(title=name)
        self.add_selector_arg_parse(name, list(self.possible))

    @classmethod
    def options(cls, key):
        if cls._OPTIONS is None:
            cls._OPTIONS = cls.entry_points_for(key)
        return cls._OPTIONS

    def add_selector_arg_parse(self, name, choices):
        raise NotImplementedError

    def handle_selected_arg_parse(self, options):
        selected = getattr(options, self.name)
        if selected not in self.possible:
            raise RuntimeError("No implementation for {}".format(self.interpreter))
        self._impl_class = self.possible[selected]
        self.populate_selected_argparse(selected, options.app_data)
        return selected

    def populate_selected_argparse(self, selected, app_data):
        self.parser.description = "options for {} {}".format(self.name, selected)
        self._impl_class.add_parser_arguments(self.parser, self.interpreter, app_data)

    def create(self, options):
        return self._impl_class(options, self.interpreter)
