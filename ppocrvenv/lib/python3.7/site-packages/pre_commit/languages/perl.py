import contextlib
import os
import shlex
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

ENVIRONMENT_DIR = 'perl_env'
get_default_version = helpers.basic_get_default_version
healthy = helpers.basic_healthy


def _envdir(prefix: Prefix, version: str) -> str:
    directory = helpers.environment_dir(ENVIRONMENT_DIR, version)
    return prefix.path(directory)


def get_env_patch(venv: str) -> PatchesT:
    return (
        ('PATH', (os.path.join(venv, 'bin'), os.pathsep, Var('PATH'))),
        ('PERL5LIB', os.path.join(venv, 'lib', 'perl5')),
        ('PERL_MB_OPT', f'--install_base {shlex.quote(venv)}'),
        (
            'PERL_MM_OPT', (
                f'INSTALL_BASE={shlex.quote(venv)} '
                f'INSTALLSITEMAN1DIR=none INSTALLSITEMAN3DIR=none'
            ),
        ),
    )


@contextlib.contextmanager
def in_env(
        prefix: Prefix,
        language_version: str,
) -> Generator[None, None, None]:
    with envcontext(get_env_patch(_envdir(prefix, language_version))):
        yield


def install_environment(
        prefix: Prefix, version: str, additional_dependencies: Sequence[str],
) -> None:
    helpers.assert_version_default('perl', version)

    with clean_path_on_failure(_envdir(prefix, version)):
        with in_env(prefix, version):
            helpers.run_setup_cmd(
                prefix, ('cpan', '-T', '.', *additional_dependencies),
            )


def run_hook(
        hook: Hook,
        file_args: Sequence[str],
        color: bool,
) -> Tuple[int, bytes]:
    with in_env(hook.prefix, hook.language_version):
        return helpers.run_xargs(hook, hook.cmd, file_args, color=color)
