import contextlib
import functools
import os.path
import shutil
import tarfile
from typing import Generator
from typing import Sequence
from typing import Tuple

import pre_commit.constants as C
from pre_commit.envcontext import envcontext
from pre_commit.envcontext import PatchesT
from pre_commit.envcontext import UNSET
from pre_commit.envcontext import Var
from pre_commit.hook import Hook
from pre_commit.languages import helpers
from pre_commit.prefix import Prefix
from pre_commit.util import CalledProcessError
from pre_commit.util import clean_path_on_failure
from pre_commit.util import resource_bytesio

ENVIRONMENT_DIR = 'rbenv'
healthy = helpers.basic_healthy


@functools.lru_cache(maxsize=1)
def get_default_version() -> str:
    if all(helpers.exe_exists(exe) for exe in ('ruby', 'gem')):
        return 'system'
    else:
        return C.DEFAULT


def get_env_patch(
        venv: str,
        language_version: str,
) -> PatchesT:
    patches: PatchesT = (
        ('GEM_HOME', os.path.join(venv, 'gems')),
        ('GEM_PATH', UNSET),
        ('BUNDLE_IGNORE_CONFIG', '1'),
    )
    if language_version == 'system':
        patches += (
            (
                'PATH', (
                    os.path.join(venv, 'gems', 'bin'), os.pathsep,
                    Var('PATH'),
                ),
            ),
        )
    else:  # pragma: win32 no cover
        patches += (
            ('RBENV_ROOT', venv),
            (
                'PATH', (
                    os.path.join(venv, 'gems', 'bin'), os.pathsep,
                    os.path.join(venv, 'shims'), os.pathsep,
                    os.path.join(venv, 'bin'), os.pathsep, Var('PATH'),
                ),
            ),
        )
    if language_version not in {'system', 'default'}:  # pragma: win32 no cover
        patches += (('RBENV_VERSION', language_version),)

    return patches


@contextlib.contextmanager
def in_env(
        prefix: Prefix,
        language_version: str,
) -> Generator[None, None, None]:
    envdir = prefix.path(
        helpers.environment_dir(ENVIRONMENT_DIR, language_version),
    )
    with envcontext(get_env_patch(envdir, language_version)):
        yield


def _extract_resource(filename: str, dest: str) -> None:
    with resource_bytesio(filename) as bio:
        with tarfile.open(fileobj=bio) as tf:
            tf.extractall(dest)


def _install_rbenv(
        prefix: Prefix,
        version: str,
) -> None:  # pragma: win32 no cover
    directory = helpers.environment_dir(ENVIRONMENT_DIR, version)

    _extract_resource('rbenv.tar.gz', prefix.path('.'))
    shutil.move(prefix.path('rbenv'), prefix.path(directory))

    # Only install ruby-build if the version is specified
    if version != C.DEFAULT:
        plugins_dir = prefix.path(directory, 'plugins')
        _extract_resource('ruby-download.tar.gz', plugins_dir)
        _extract_resource('ruby-build.tar.gz', plugins_dir)


def _install_ruby(
        prefix: Prefix,
        version: str,
) -> None:  # pragma: win32 no cover
    try:
        helpers.run_setup_cmd(prefix, ('rbenv', 'download', version))
    except CalledProcessError:  # pragma: no cover (usually find with download)
        # Failed to download from mirror for some reason, build it instead
        helpers.run_setup_cmd(prefix, ('rbenv', 'install', version))


def install_environment(
        prefix: Prefix, version: str, additional_dependencies: Sequence[str],
) -> None:
    additional_dependencies = tuple(additional_dependencies)
    directory = helpers.environment_dir(ENVIRONMENT_DIR, version)
    with clean_path_on_failure(prefix.path(directory)):
        if version != 'system':  # pragma: win32 no cover
            _install_rbenv(prefix, version)
            with in_env(prefix, version):
                # Need to call this before installing so rbenv's directories
                # are set up
                helpers.run_setup_cmd(prefix, ('rbenv', 'init', '-'))
                if version != C.DEFAULT:
                    _install_ruby(prefix, version)
                # Need to call this after installing to set up the shims
                helpers.run_setup_cmd(prefix, ('rbenv', 'rehash'))

        with in_env(prefix, version):
            helpers.run_setup_cmd(
                prefix, ('gem', 'build', *prefix.star('.gemspec')),
            )
            helpers.run_setup_cmd(
                prefix,
                (
                    'gem', 'install',
                    '--no-document', '--no-format-executable',
                    *prefix.star('.gem'), *additional_dependencies,
                ),
            )


def run_hook(
        hook: Hook,
        file_args: Sequence[str],
        color: bool,
) -> Tuple[int, bytes]:
    with in_env(hook.prefix, hook.language_version):
        return helpers.run_xargs(hook, hook.cmd, file_args, color=color)
