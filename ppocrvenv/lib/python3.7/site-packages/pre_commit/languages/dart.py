import contextlib
import os.path
import shutil
import tempfile
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
from pre_commit.util import win_exe
from pre_commit.util import yaml_load

ENVIRONMENT_DIR = 'dartenv'

get_default_version = helpers.basic_get_default_version
healthy = helpers.basic_healthy


def get_env_patch(venv: str) -> PatchesT:
    return (
        ('PATH', (os.path.join(venv, 'bin'), os.pathsep, Var('PATH'))),
    )


@contextlib.contextmanager
def in_env(prefix: Prefix) -> Generator[None, None, None]:
    directory = helpers.environment_dir(ENVIRONMENT_DIR, C.DEFAULT)
    envdir = prefix.path(directory)
    with envcontext(get_env_patch(envdir)):
        yield


def install_environment(
        prefix: Prefix,
        version: str,
        additional_dependencies: Sequence[str],
) -> None:
    helpers.assert_version_default('dart', version)

    envdir = prefix.path(helpers.environment_dir(ENVIRONMENT_DIR, version))
    bin_dir = os.path.join(envdir, 'bin')

    def _install_dir(prefix_p: Prefix, pub_cache: str) -> None:
        dart_env = {**os.environ, 'PUB_CACHE': pub_cache}

        with open(prefix_p.path('pubspec.yaml')) as f:
            pubspec_contents = yaml_load(f)

        helpers.run_setup_cmd(prefix_p, ('dart', 'pub', 'get'), env=dart_env)

        for executable in pubspec_contents['executables']:
            helpers.run_setup_cmd(
                prefix_p,
                (
                    'dart', 'compile', 'exe',
                    '--output', os.path.join(bin_dir, win_exe(executable)),
                    prefix_p.path('bin', f'{executable}.dart'),
                ),
                env=dart_env,
            )

    with clean_path_on_failure(envdir):
        os.makedirs(bin_dir)

        with tempfile.TemporaryDirectory() as tmp:
            _install_dir(prefix, tmp)

        for dep_s in additional_dependencies:
            with tempfile.TemporaryDirectory() as dep_tmp:
                dep, _, version = dep_s.partition(':')
                if version:
                    dep_cmd: Tuple[str, ...] = (dep, '--version', version)
                else:
                    dep_cmd = (dep,)

                helpers.run_setup_cmd(
                    prefix,
                    ('dart', 'pub', 'cache', 'add', *dep_cmd),
                    env={**os.environ, 'PUB_CACHE': dep_tmp},
                )

                # try and find the 'pubspec.yaml' that just got added
                for root, _, filenames in os.walk(dep_tmp):
                    if 'pubspec.yaml' in filenames:
                        with tempfile.TemporaryDirectory() as copied:
                            pkg = os.path.join(copied, 'pkg')
                            shutil.copytree(root, pkg)
                            _install_dir(Prefix(pkg), dep_tmp)
                        break
                else:
                    raise AssertionError(
                        f'could not find pubspec.yaml for {dep_s}',
                    )


def run_hook(
        hook: Hook,
        file_args: Sequence[str],
        color: bool,
) -> Tuple[int, bytes]:
    with in_env(hook.prefix):
        return helpers.run_xargs(hook, hook.cmd, file_args, color=color)
