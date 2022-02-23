import contextlib
import os
from typing import Generator
from typing import Sequence
from typing import Tuple

from pre_commit.envcontext import envcontext
from pre_commit.envcontext import PatchesT
from pre_commit.envcontext import Var
from pre_commit.hook import Hook
from pre_commit.languages import helpers
from pre_commit.prefix import Prefix
from pre_commit.util import clean_path_on_failure

ENVIRONMENT_DIR = 'coursier'

get_default_version = helpers.basic_get_default_version
healthy = helpers.basic_healthy


def install_environment(
        prefix: Prefix,
        version: str,
        additional_dependencies: Sequence[str],
) -> None:   # pragma: win32 no cover
    helpers.assert_version_default('coursier', version)
    helpers.assert_no_additional_deps('coursier', additional_dependencies)

    envdir = prefix.path(helpers.environment_dir(ENVIRONMENT_DIR, version))
    channel = prefix.path('.pre-commit-channel')
    with clean_path_on_failure(envdir):
        for app_descriptor in os.listdir(channel):
            _, app_file = os.path.split(app_descriptor)
            app, _ = os.path.splitext(app_file)
            helpers.run_setup_cmd(
                prefix,
                (
                    'cs',
                    'install',
                    '--default-channels=false',
                    f'--channel={channel}',
                    app,
                    f'--dir={envdir}',
                ),
            )


def get_env_patch(target_dir: str) -> PatchesT:   # pragma: win32 no cover
    return (
        ('PATH', (target_dir, os.pathsep, Var('PATH'))),
    )


@contextlib.contextmanager
def in_env(
        prefix: Prefix,
) -> Generator[None, None, None]:   # pragma: win32 no cover
    target_dir = prefix.path(
        helpers.environment_dir(ENVIRONMENT_DIR, get_default_version()),
    )
    with envcontext(get_env_patch(target_dir)):
        yield


def run_hook(
        hook: Hook,
        file_args: Sequence[str],
        color: bool,
) -> Tuple[int, bytes]:   # pragma: win32 no cover
    with in_env(hook.prefix):
        return helpers.run_xargs(hook, hook.cmd, file_args, color=color)
