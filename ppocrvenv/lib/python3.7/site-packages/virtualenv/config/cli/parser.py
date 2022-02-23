from __future__ import absolute_import, unicode_literals

import os
from argparse import SUPPRESS, ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections import OrderedDict

from virtualenv.config.convert import get_type

from ..env_var import get_env_var
from ..ini import IniConfig


class VirtualEnvOptions(Namespace):
    def __init__(self, **kwargs):
        super(VirtualEnvOptions, self).__init__(**kwargs)
        self._src = None
        self._sources = {}

    def set_src(self, key, value, src):
        setattr(self, key, value)
        if src.startswith("env var"):
            src = "env var"
        self._sources[key] = src

    def __setattr__(self, key, value):
        if getattr(self, "_src", None) is not None:
            self._sources[key] = self._src
        super(VirtualEnvOptions, self).__setattr__(key, value)

    def get_source(self, key):
        return self._sources.get(key)

    @property
    def verbosity(self):
        if not hasattr(self, "verbose") and not hasattr(self, "quiet"):
            return None
        return max(self.verbose - self.quiet, 0)

    def __repr__(self):
        return "{}({})".format(
            type(self).__name__,
            ", ".join("{}={}".format(k, v) for k, v in vars(self).items() if not k.startswith("_")),
        )


class VirtualEnvConfigParser(ArgumentParser):
    """
    Custom option parser which updates its defaults by checking the configuration files and environmental variables
    """

    def __init__(self, options=None, env=None, *args, **kwargs):
        env = os.environ if env is None else env
        self.file_config = IniConfig(env)
        self.epilog_list = []
        self.env = env
        kwargs["epilog"] = self.file_config.epilog
        kwargs["add_help"] = False
        kwargs["formatter_class"] = HelpFormatter
        kwargs["prog"] = "virtualenv"
        super(VirtualEnvConfigParser, self).__init__(*args, **kwargs)
        self._fixed = set()
        if options is not None and not isinstance(options, VirtualEnvOptions):
            raise TypeError("options must be of type VirtualEnvOptions")
        self.options = VirtualEnvOptions() if options is None else options
        self._interpreter = None
        self._app_data = None

    def _fix_defaults(self):
        for action in self._actions:
            action_id = id(action)
            if action_id not in self._fixed:
                self._fix_default(action)
                self._fixed.add(action_id)

    def _fix_default(self, action):
        if hasattr(action, "default") and hasattr(action, "dest") and action.default != SUPPRESS:
            as_type = get_type(action)
            names = OrderedDict((i.lstrip("-").replace("-", "_"), None) for i in action.option_strings)
            outcome = None
            for name in names:
                outcome = get_env_var(name, as_type, self.env)
                if outcome is not None:
                    break
            if outcome is None and self.file_config:
                for name in names:
                    outcome = self.file_config.get(name, as_type)
                    if outcome is not None:
                        break
            if outcome is not None:
                action.default, action.default_source = outcome
            else:
                outcome = action.default, "default"
            self.options.set_src(action.dest, *outcome)

    def enable_help(self):
        self._fix_defaults()
        self.add_argument("-h", "--help", action="help", default=SUPPRESS, help="show this help message and exit")

    def parse_known_args(self, args=None, namespace=None):
        if namespace is None:
            namespace = self.options
        elif namespace is not self.options:
            raise ValueError("can only pass in parser.options")
        self._fix_defaults()
        self.options._src = "cli"
        try:
            namespace.env = self.env
            return super(VirtualEnvConfigParser, self).parse_known_args(args, namespace=namespace)
        finally:
            self.options._src = None


class HelpFormatter(ArgumentDefaultsHelpFormatter):
    def __init__(self, prog):
        super(HelpFormatter, self).__init__(prog, max_help_position=32, width=240)

    def _get_help_string(self, action):
        # noinspection PyProtectedMember
        text = super(HelpFormatter, self)._get_help_string(action)
        if hasattr(action, "default_source"):
            default = " (default: %(default)s)"
            if text.endswith(default):
                text = "{} (default: %(default)s -> from %(default_source)s)".format(text[: -len(default)])
        return text
