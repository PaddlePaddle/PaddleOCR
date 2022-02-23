import contextlib
import os.path
import sys
from typing import Generator
from typing import Sequence
from typing import Tuple

import pre_commit.constants as C
from pre_commit import git
from pre_commit.envcontext import envcontext
from pre_commit.envcontext import PatchesT
from pre_commit.envcontext import Var
from pre_commit.hook import Hook
from pre_commit.languages import helpers
from pre_commit.prefix import Prefix
from pre_commit.util import clean_path_on_failure
from pre_commit.util import cmd_output
from pre_commit.util import cmd_output_b
from pre_commit.util import rmtree

ENVIRONMENT_DIR = 'golangenv'
get_default_version = helpers.basic_get_default_version
healthy = helpers.basic_healthy


def get_env_patch(venv: str) -> PatchesT:
    return (
        ('PATH', (os.path.join(venv, 'bin'), os.pathsep, Var('PATH'))),
    )


@contextlib.contextmanager
def in_env(prefix: Prefix) -> Generator[None, None, None]:
    envdir = prefix.path(
        helpers.environment_dir(ENVIRONMENT_DIR, C.DEFAULT),
    )
    with envcontext(get_env_patch(envdir)):
        yield


def guess_go_dir(remote_url: str) -> str:
    if remote_url.endswith('.git'):
        remote_url = remote_url[:-1 * len('.git')]
    looks_like_url = (
        not remote_url.startswith('file://') and
        ('//' in remote_url or '@' in remote_url)
    )
    remote_url = remote_url.replace(':', '/')
    if looks_like_url:
        _, _, remote_url = remote_url.rpartition('//')
        _, _, remote_url = remote_url.rpartition('@')
        return remote_url
    else:
        return 'unknown_src_dir'


def install_environment(
        prefix: Prefix,
        version: str,
        additional_dependencies: Sequence[str],
) -> None:
    helpers.assert_version_default('golang', version)
    directory = prefix.path(
        helpers.environment_dir(ENVIRONMENT_DIR, C.DEFAULT),
    )

    with clean_path_on_failure(directory):
        remote = git.get_remote_url(prefix.prefix_dir)
        repo_src_dir = os.path.join(directory, 'src', guess_go_dir(remote))

        # Clone into the goenv we'll create
        cmd = ('git', 'clone', '--recursive', '.', repo_src_dir)
        helpers.run_setup_cmd(prefix, cmd)

        if sys.platform == 'cygwin':  # pragma: no cover
            _, gopath, _ = cmd_output('cygpath', '-w', directory)
            gopath = gopath.strip()
        else:
            gopath = directory
        env = dict(os.environ, GOPATH=gopath)
        env.pop('GOBIN', None)
        cmd_output_b('go', 'install', './...', cwd=repo_src_dir, env=env)
        for dependency in additional_dependencies:
            cmd_output_b(
                'go', 'install', dependency, cwd=repo_src_dir, env=env,
            )
        # Same some disk space, we don't need these after installation
        rmtree(prefix.path(directory, 'src'))
        pkgdir = prefix.path(directory, 'pkg')
        if os.path.exists(pkgdir):  # pragma: no cover (go<1.10)
            rmtree(pkgdir)


def run_hook(
        hook: Hook,
        file_args: Sequence[str],
        color: bool,
) -> Tuple[int, bytes]:
    with in_env(hook.prefix):
        return helpers.run_xargs(hook, hook.cmd, file_args, color=color)
