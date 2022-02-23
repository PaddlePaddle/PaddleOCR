from __future__ import absolute_import, unicode_literals

from virtualenv.util.six import ensure_str, ensure_text

from .convert import convert


def get_env_var(key, as_type, env):
    """Get the environment variable option.

    :param key: the config key requested
    :param as_type: the type we would like to convert it to
    :param env: environment variables to use
    :return:
    """
    environ_key = ensure_str("VIRTUALENV_{}".format(key.upper()))
    if env.get(environ_key):
        value = env[environ_key]
        # noinspection PyBroadException
        try:
            source = "env var {}".format(ensure_text(environ_key))
            as_type = convert(value, as_type, source)
            return as_type, source
        except Exception:  # note the converter already logs a warning when failures happen
            pass


__all__ = ("get_env_var",)
