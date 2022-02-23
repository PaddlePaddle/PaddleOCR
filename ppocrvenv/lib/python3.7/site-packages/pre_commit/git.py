import logging
import os.path
import sys
from typing import Dict
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Set

from pre_commit.errors import FatalError
from pre_commit.util import CalledProcessError
from pre_commit.util import cmd_output
from pre_commit.util import cmd_output_b

logger = logging.getLogger(__name__)

# see #2046
NO_FS_MONITOR = ('-c', 'core.useBuiltinFSMonitor=false')


def zsplit(s: str) -> List[str]:
    s = s.strip('\0')
    if s:
        return s.split('\0')
    else:
        return []


def no_git_env(
        _env: Optional[MutableMapping[str, str]] = None,
) -> Dict[str, str]:
    # Too many bugs dealing with environment variables and GIT:
    # https://github.com/pre-commit/pre-commit/issues/300
    # In git 2.6.3 (maybe others), git exports GIT_WORK_TREE while running
    # pre-commit hooks
    # In git 1.9.1 (maybe others), git exports GIT_DIR and GIT_INDEX_FILE
    # while running pre-commit hooks in submodules.
    # GIT_DIR: Causes git clone to clone wrong thing
    # GIT_INDEX_FILE: Causes 'error invalid object ...' during commit
    _env = _env if _env is not None else os.environ
    return {
        k: v for k, v in _env.items()
        if not k.startswith('GIT_') or
        k.startswith(('GIT_CONFIG_KEY_', 'GIT_CONFIG_VALUE_')) or
        k in {
            'GIT_EXEC_PATH', 'GIT_SSH', 'GIT_SSH_COMMAND', 'GIT_SSL_CAINFO',
            'GIT_SSL_NO_VERIFY', 'GIT_CONFIG_COUNT',
        }
    }


def get_root() -> str:
    # Git 2.25 introduced a change to "rev-parse --show-toplevel" that exposed
    # underlying volumes for Windows drives mapped with SUBST.  We use
    # "rev-parse --show-cdup" to get the appropriate path, but must perform
    # an extra check to see if we are in the .git directory.
    try:
        root = os.path.abspath(
            cmd_output('git', 'rev-parse', '--show-cdup')[1].strip(),
        )
        git_dir = os.path.abspath(get_git_dir())
    except CalledProcessError:
        raise FatalError(
            'git failed. Is it installed, and are you in a Git repository '
            'directory?',
        )
    if os.path.samefile(root, git_dir):
        raise FatalError(
            'git toplevel unexpectedly empty! make sure you are not '
            'inside the `.git` directory of your repository.',
        )
    return root


def get_git_dir(git_root: str = '.') -> str:
    opts = ('--git-common-dir', '--git-dir')
    _, out, _ = cmd_output('git', 'rev-parse', *opts, cwd=git_root)
    for line, opt in zip(out.splitlines(), opts):
        if line != opt:  # pragma: no branch (git < 2.5)
            return os.path.normpath(os.path.join(git_root, line))
    else:
        raise AssertionError('unreachable: no git dir')


def get_remote_url(git_root: str) -> str:
    _, out, _ = cmd_output('git', 'config', 'remote.origin.url', cwd=git_root)
    return out.strip()


def is_in_merge_conflict() -> bool:
    git_dir = get_git_dir('.')
    return (
        os.path.exists(os.path.join(git_dir, 'MERGE_MSG')) and
        os.path.exists(os.path.join(git_dir, 'MERGE_HEAD'))
    )


def parse_merge_msg_for_conflicts(merge_msg: bytes) -> List[str]:
    # Conflicted files start with tabs
    return [
        line.lstrip(b'#').strip().decode()
        for line in merge_msg.splitlines()
        # '#\t' for git 2.4.1
        if line.startswith((b'\t', b'#\t'))
    ]


def get_conflicted_files() -> Set[str]:
    logger.info('Checking merge-conflict files only.')
    # Need to get the conflicted files from the MERGE_MSG because they could
    # have resolved the conflict by choosing one side or the other
    with open(os.path.join(get_git_dir('.'), 'MERGE_MSG'), 'rb') as f:
        merge_msg = f.read()
    merge_conflict_filenames = parse_merge_msg_for_conflicts(merge_msg)

    # This will get the rest of the changes made after the merge.
    # If they resolved the merge conflict by choosing a mesh of both sides
    # this will also include the conflicted files
    tree_hash = cmd_output('git', 'write-tree')[1].strip()
    merge_diff_filenames = zsplit(
        cmd_output(
            'git', 'diff', '--name-only', '--no-ext-diff', '-z',
            '-m', tree_hash, 'HEAD', 'MERGE_HEAD',
        )[1],
    )
    return set(merge_conflict_filenames) | set(merge_diff_filenames)


def get_staged_files(cwd: Optional[str] = None) -> List[str]:
    return zsplit(
        cmd_output(
            'git', 'diff', '--staged', '--name-only', '--no-ext-diff', '-z',
            # Everything except for D
            '--diff-filter=ACMRTUXB',
            cwd=cwd,
        )[1],
    )


def intent_to_add_files() -> List[str]:
    _, stdout, _ = cmd_output(
        'git', 'status', '--ignore-submodules', '--porcelain', '-z',
    )
    parts = list(reversed(zsplit(stdout)))
    intent_to_add = []
    while parts:
        line = parts.pop()
        status, filename = line[:3], line[3:]
        if status[0] in {'C', 'R'}:  # renames / moves have an additional arg
            parts.pop()
        if status[1] == 'A':
            intent_to_add.append(filename)
    return intent_to_add


def get_all_files() -> List[str]:
    return zsplit(cmd_output('git', 'ls-files', '-z')[1])


def get_changed_files(old: str, new: str) -> List[str]:
    diff_cmd = ('git', 'diff', '--name-only', '--no-ext-diff', '-z')
    try:
        _, out, _ = cmd_output(*diff_cmd, f'{old}...{new}')
    except CalledProcessError:  # pragma: no cover (new git)
        # on newer git where old and new do not have a merge base git fails
        # so we try a full diff (this is what old git did for us!)
        _, out, _ = cmd_output(*diff_cmd, f'{old}..{new}')

    return zsplit(out)


def head_rev(remote: str) -> str:
    _, out, _ = cmd_output('git', 'ls-remote', '--exit-code', remote, 'HEAD')
    return out.split()[0]


def has_diff(*args: str, repo: str = '.') -> bool:
    cmd = ('git', 'diff', '--quiet', '--no-ext-diff', *args)
    return cmd_output_b(*cmd, cwd=repo, retcode=None)[0] == 1


def has_core_hookpaths_set() -> bool:
    _, out, _ = cmd_output_b('git', 'config', 'core.hooksPath', retcode=None)
    return bool(out.strip())


def init_repo(path: str, remote: str) -> None:
    if os.path.isdir(remote):
        remote = os.path.abspath(remote)

    git = ('git', *NO_FS_MONITOR)
    env = no_git_env()
    # avoid the user's template so that hooks do not recurse
    cmd_output_b(*git, 'init', '--template=', path, env=env)
    cmd_output_b(*git, 'remote', 'add', 'origin', remote, cwd=path, env=env)


def commit(repo: str = '.') -> None:
    env = no_git_env()
    name, email = 'pre-commit', 'asottile+pre-commit@umich.edu'
    env['GIT_AUTHOR_NAME'] = env['GIT_COMMITTER_NAME'] = name
    env['GIT_AUTHOR_EMAIL'] = env['GIT_COMMITTER_EMAIL'] = email
    cmd = ('git', 'commit', '--no-edit', '--no-gpg-sign', '-n', '-minit')
    cmd_output_b(*cmd, cwd=repo, env=env)


def git_path(name: str, repo: str = '.') -> str:
    _, out, _ = cmd_output('git', 'rev-parse', '--git-path', name, cwd=repo)
    return os.path.join(repo, out.strip())


def check_for_cygwin_mismatch() -> None:
    """See https://github.com/pre-commit/pre-commit/issues/354"""
    if sys.platform in ('cygwin', 'win32'):  # pragma: no cover (windows)
        is_cygwin_python = sys.platform == 'cygwin'
        try:
            toplevel = get_root()
        except FatalError:  # skip the check if we're not in a git repo
            return
        is_cygwin_git = toplevel.startswith('/')

        if is_cygwin_python ^ is_cygwin_git:
            exe_type = {True: '(cygwin)', False: '(windows)'}
            logger.warn(
                f'pre-commit has detected a mix of cygwin python / git\n'
                f'This combination is not supported, it is likely you will '
                f'receive an error later in the program.\n'
                f'Make sure to use cygwin git+python while using cygwin\n'
                f'These can be installed through the cygwin installer.\n'
                f' - python {exe_type[is_cygwin_python]}\n'
                f' - git {exe_type[is_cygwin_git]}\n',
            )
