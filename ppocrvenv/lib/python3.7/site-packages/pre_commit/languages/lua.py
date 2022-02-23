import contextlib
import os
import sys
from typing import Generator
from typing import Sequence
from typing import Tuple

import pre_commit.constants as C
from pre_commit.envcontext import envcontext
from pre_commit.envcontext import PatchesT
from pre_commit.envcontext import Var
from pre_commit.hook import Hook
from pre_commit.languages import helpers
from pre_commit.prefix import Prefix
from pre_commit.util import clean_path_on_failure
from pre_commit.util import cmd_output

ENVIRONMENT_DIR = 'lua_env'
get_default_version = helpers.basic_get_default_version
healthy = helpers.basic_healthy


def _get_lua_version() -> str:  # pragma: win32 no cover
    """Get the Lua version used in file paths."""
    _, stdout, _ = cmd_output('luarocks', 'config', '--lua-ver')
    return stdout.strip()


def get_env_patch(d: str) -> PatchesT:  # pragma: win32 no cover
    version = _get_lua_version()
    so_ext = 'dll' if sys.platform == 'win32' else 'so'
    return (
        ('PATH', (os.path.join(d, 'bin'), os.pathsep, Var('PATH'))),
        (
            'LUA_PATH', (
                os.path.join(d, 'share', 'lua', version, '?.lua;'),
                os.path.join(d, 'share', 'lua', version, '?', 'init.lua;;'),
            ),
        ),
        (
            'LUA_CPATH',
            (os.path.join(d, 'lib', 'lua', version, f'?.{so_ext};;'),),
        ),
    )


def _envdir(prefix: Prefix) -> str:  # pragma: win32 no cover
    directory = helpers.environment_dir(ENVIRONMENT_DIR, C.DEFAULT)
    return prefix.path(directory)


@contextlib.contextmanager  # pragma: win32 no cover
def in_env(prefix: Prefix) -> Generator[None, None, None]:
    with envcontext(get_env_patch(_envdir(prefix))):
        yield


def install_environment(
    prefix: Prefix,
    version: str,
    additional_dependencies: Sequence[str],
) -> None:  # pragma: win32 no cover
    helpers.assert_version_default('lua', version)

    envdir = _envdir(prefix)
    with clean_path_on_failure(envdir):
        with in_env(prefix):
            # luarocks doesn't bootstrap a tree prior to installing
            # so ensure the directory exists.
            os.makedirs(envdir, exist_ok=True)

            # Older luarocks (e.g., 2.4.2) expect the rockspec as an arg
            for rockspec in prefix.star('.rockspec'):
                make_cmd = ('luarocks', '--tree', envdir, 'make', rockspec)
                helpers.run_setup_cmd(prefix, make_cmd)

            # luarocks can't install multiple packages at once
            # so install them individually.
            for dependency in additional_dependencies:
                cmd = ('luarocks', '--tree', envdir, 'install', dependency)
                helpers.run_setup_cmd(prefix, cmd)


def run_hook(
    hook: Hook,
    file_args: Sequence[str],
    color: bool,
) -> Tuple[int, bytes]:  # pragma: win32 no cover
    with in_env(hook.prefix):
        return helpers.run_xargs(hook, hook.cmd, file_args, color=color)
