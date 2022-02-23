from __future__ import absolute_import, unicode_literals

import logging
import os

from platformdirs import user_config_dir

from virtualenv.info import PY3
from virtualenv.util import ConfigParser
from virtualenv.util.path import Path
from virtualenv.util.six import ensure_str

from .convert import convert


class IniConfig(object):
    VIRTUALENV_CONFIG_FILE_ENV_VAR = ensure_str("VIRTUALENV_CONFIG_FILE")
    STATE = {None: "failed to parse", True: "active", False: "missing"}

    section = "virtualenv"

    def __init__(self, env=None):
        env = os.environ if env is None else env
        config_file = env.get(self.VIRTUALENV_CONFIG_FILE_ENV_VAR, None)
        self.is_env_var = config_file is not None
        config_file = (
            Path(config_file)
            if config_file is not None
            else Path(user_config_dir(appname="virtualenv", appauthor="pypa")) / "virtualenv.ini"
        )
        self.config_file = config_file
        self._cache = {}

        exception = None
        self.has_config_file = None
        try:
            self.has_config_file = self.config_file.exists()
        except OSError as exc:
            exception = exc
        else:
            if self.has_config_file:
                self.config_file = self.config_file.resolve()
                self.config_parser = ConfigParser.ConfigParser()
                try:
                    self._load()
                    self.has_virtualenv_section = self.config_parser.has_section(self.section)
                except Exception as exc:
                    exception = exc
        if exception is not None:
            logging.error("failed to read config file %s because %r", config_file, exception)

    def _load(self):
        with self.config_file.open("rt") as file_handler:
            reader = getattr(self.config_parser, "read_file" if PY3 else "readfp")
            reader(file_handler)

    def get(self, key, as_type):
        cache_key = key, as_type
        if cache_key in self._cache:
            return self._cache[cache_key]
        # noinspection PyBroadException
        try:
            source = "file"
            raw_value = self.config_parser.get(self.section, key.lower())
            value = convert(raw_value, as_type, source)
            result = value, source
        except Exception:
            result = None
        self._cache[cache_key] = result
        return result

    def __bool__(self):
        return bool(self.has_config_file) and bool(self.has_virtualenv_section)

    @property
    def epilog(self):
        msg = "{}config file {} {} (change{} via env var {})"
        return msg.format(
            "\n",
            self.config_file,
            self.STATE[self.has_config_file],
            "d" if self.is_env_var else "",
            self.VIRTUALENV_CONFIG_FILE_ENV_VAR,
        )
