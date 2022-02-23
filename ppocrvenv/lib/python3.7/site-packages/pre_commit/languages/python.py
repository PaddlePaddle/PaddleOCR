import contextlib
import functools
import os
import sys
from typing import Dict
from typing import Generator
from typing import Optional
from typing import Sequence
from typing import Tuple

import pre_commit.constants as C
from pre_commit.envcontext import envcontext
from pre_commit.envcontext import PatchesT
from pre_commit.envcontext import UNSET
from pre_commit.envcontext import Var
from pre_commit.hook import Hook
from pre_commit.languages import helpers
from pre_commit.parse_shebang import find_executable
from pre_commit.prefix import Prefix
from pre_commit.util import CalledProcessError
from pre_commit.util import clean_path_on_failure
from pre_commit.util import cmd_output
from pre_commit.util import cmd_output_b
from pre_commit.util import win_exe

ENVIRONMENT_DIR = 'py_env'


@functools.lru_cache(maxsize=None)
def _version_info(exe: str) -> str:
    prog = 'import sys;print(".".join(str(p) for p in sys.version_info))'
    try:
        return cmd_output(exe, '-S', '-c', prog)[1].strip()
    except CalledProcessError:
        return f'<<error retrieving version from {exe}>>'


def _read_pyvenv_cfg(filename: str) -> Dict[str, str]:
    ret = {}
    with open(filename, encoding='UTF-8') as f:
        for line in f:
            try:
                k, v = line.split('=')
            except ValueError:  # blank line / comment / etc.
                continue
            else:
                ret[k.strip()] = v.strip()
    return ret


def bin_dir(venv: str) -> str:
    """On windows there's a different directory for the virtualenv"""
    bin_part = 'Scripts' if os.name == 'nt' else 'bin'
    return os.path.join(venv, bin_part)


def get_env_patch(venv: str) -> PatchesT:
    return (
        ('PIP_DISABLE_PIP_VERSION_CHECK', '1'),
        ('PYTHONHOME', UNSET),
        ('VIRTUAL_ENV', venv),
        ('PATH', (bin_dir(venv), os.pathsep, Var('PATH'))),
    )


def _find_by_py_launcher(
        version: str,
) -> Optional[str]:  # pragma: no cover (windows only)
    if version.startswith('python'):
        num = version[len('python'):]
        cmd = ('py', f'-{num}', '-c', 'import sys; print(sys.executable)')
        env = dict(os.environ, PYTHONIOENCODING='UTF-8')
        try:
            return cmd_output(*cmd, env=env)[1].strip()
        except CalledProcessError:
            pass
    return None


def _find_by_sys_executable() -> Optional[str]:
    def _norm(path: str) -> Optional[str]:
        _, exe = os.path.split(path.lower())
        exe, _, _ = exe.partition('.exe')
        if exe not in {'python', 'pythonw'} and find_executable(exe):
            return exe
        return None

    # On linux, I see these common sys.executables:
    #
    # system `python`: /usr/bin/python -> python2.7
    # system `python2`: /usr/bin/python2 -> python2.7
    # virtualenv v: v/bin/python (will not return from this loop)
    # virtualenv v -ppython2: v/bin/python -> python2
    # virtualenv v -ppython2.7: v/bin/python -> python2.7
    # virtualenv v -ppypy: v/bin/python -> v/bin/pypy
    for path in (sys.executable, os.path.realpath(sys.executable)):
        exe = _norm(path)
        if exe:
            return exe
    return None


@functools.lru_cache(maxsize=1)
def get_default_version() -> str:  # pragma: no cover (platform dependent)
    # First attempt from `sys.executable` (or the realpath)
    exe = _find_by_sys_executable()
    if exe:
        return exe

    # Next try the `pythonX.X` executable
    exe = f'python{sys.version_info[0]}.{sys.version_info[1]}'
    if find_executable(exe):
        return exe

    if _find_by_py_launcher(exe):
        return exe

    # We tried!
    return C.DEFAULT


def _sys_executable_matches(version: str) -> bool:
    if version == 'python':
        return True
    elif not version.startswith('python'):
        return False

    try:
        info = tuple(int(p) for p in version[len('python'):].split('.'))
    except ValueError:
        return False

    return sys.version_info[:len(info)] == info


def norm_version(version: str) -> Optional[str]:
    if version == C.DEFAULT:  # use virtualenv's default
        return None
    elif _sys_executable_matches(version):  # virtualenv defaults to our exe
        return None

    if os.name == 'nt':  # pragma: no cover (windows)
        version_exec = _find_by_py_launcher(version)
        if version_exec:
            return version_exec

        # Try looking up by name
        version_exec = find_executable(version)
        if version_exec and version_exec != version:
            return version_exec

    # Otherwise assume it is a path
    return os.path.expanduser(version)


@contextlib.contextmanager
def in_env(
        prefix: Prefix,
        language_version: str,
) -> Generator[None, None, None]:
    directory = helpers.environment_dir(ENVIRONMENT_DIR, language_version)
    envdir = prefix.path(directory)
    with envcontext(get_env_patch(envdir)):
        yield


def healthy(prefix: Prefix, language_version: str) -> bool:
    directory = helpers.environment_dir(ENVIRONMENT_DIR, language_version)
    envdir = prefix.path(directory)
    pyvenv_cfg = os.path.join(envdir, 'pyvenv.cfg')

    # created with "old" virtualenv
    if not os.path.exists(pyvenv_cfg):
        return False

    exe_name = win_exe('python')
    py_exe = prefix.path(bin_dir(envdir), exe_name)
    cfg = _read_pyvenv_cfg(pyvenv_cfg)

    return (
        'version_info' in cfg and
        # always use uncached lookup here in case we replaced an unhealthy env
        _version_info.__wrapped__(py_exe) == cfg['version_info'] and (
            'base-executable' not in cfg or
            _version_info(cfg['base-executable']) == cfg['version_info']
        )
    )


def install_environment(
        prefix: Prefix,
        version: str,
        additional_dependencies: Sequence[str],
) -> None:
    envdir = prefix.path(helpers.environment_dir(ENVIRONMENT_DIR, version))
    venv_cmd = [sys.executable, '-mvirtualenv', envdir]
    python = norm_version(version)
    if python is not None:
        venv_cmd.extend(('-p', python))
    install_cmd = ('python', '-mpip', 'install', '.', *additional_dependencies)

    with clean_path_on_failure(envdir):
        cmd_output_b(*venv_cmd, cwd='/')
        with in_env(prefix, version):
            helpers.run_setup_cmd(prefix, install_cmd)


def run_hook(
        hook: Hook,
        file_args: Sequence[str],
        color: bool,
) -> Tuple[int, bytes]:
    with in_env(hook.prefix, hook.language_version):
        return helpers.run_xargs(hook, hook.cmd, file_args, color=color)
