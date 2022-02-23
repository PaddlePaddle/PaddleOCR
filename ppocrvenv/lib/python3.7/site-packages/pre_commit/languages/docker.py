import hashlib
import json
import os
from typing import Sequence
from typing import Tuple

import pre_commit.constants as C
from pre_commit.hook import Hook
from pre_commit.languages import helpers
from pre_commit.prefix import Prefix
from pre_commit.util import CalledProcessError
from pre_commit.util import clean_path_on_failure
from pre_commit.util import cmd_output_b

ENVIRONMENT_DIR = 'docker'
PRE_COMMIT_LABEL = 'PRE_COMMIT'
get_default_version = helpers.basic_get_default_version
healthy = helpers.basic_healthy


def _is_in_docker() -> bool:
    try:
        with open('/proc/1/cgroup', 'rb') as f:
            return b'docker' in f.read()
    except FileNotFoundError:
        return False


def _get_container_id() -> str:
    # It's assumed that we already check /proc/1/cgroup in _is_in_docker. The
    # cpuset cgroup controller existed since cgroups were introduced so this
    # way of getting the container ID is pretty reliable.
    with open('/proc/1/cgroup', 'rb') as f:
        for line in f.readlines():
            if line.split(b':')[1] == b'cpuset':
                return os.path.basename(line.split(b':')[2]).strip().decode()
    raise RuntimeError('Failed to find the container ID in /proc/1/cgroup.')


def _get_docker_path(path: str) -> str:
    if not _is_in_docker():
        return path

    container_id = _get_container_id()

    try:
        _, out, _ = cmd_output_b('docker', 'inspect', container_id)
    except CalledProcessError:
        # self-container was not visible from here (perhaps docker-in-docker)
        return path

    container, = json.loads(out)
    for mount in container['Mounts']:
        src_path = mount['Source']
        to_path = mount['Destination']
        if os.path.commonpath((path, to_path)) == to_path:
            # So there is something in common,
            # and we can proceed remapping it
            return path.replace(to_path, src_path)
    # we're in Docker, but the path is not mounted, cannot really do anything,
    # so fall back to original path
    return path


def md5(s: str) -> str:  # pragma: win32 no cover
    return hashlib.md5(s.encode()).hexdigest()


def docker_tag(prefix: Prefix) -> str:  # pragma: win32 no cover
    md5sum = md5(os.path.basename(prefix.prefix_dir)).lower()
    return f'pre-commit-{md5sum}'


def build_docker_image(
        prefix: Prefix,
        *,
        pull: bool,
) -> None:  # pragma: win32 no cover
    cmd: Tuple[str, ...] = (
        'docker', 'build',
        '--tag', docker_tag(prefix),
        '--label', PRE_COMMIT_LABEL,
    )
    if pull:
        cmd += ('--pull',)
    # This must come last for old versions of docker.  See #477
    cmd += ('.',)
    helpers.run_setup_cmd(prefix, cmd)


def install_environment(
        prefix: Prefix, version: str, additional_dependencies: Sequence[str],
) -> None:  # pragma: win32 no cover
    helpers.assert_version_default('docker', version)
    helpers.assert_no_additional_deps('docker', additional_dependencies)

    directory = prefix.path(
        helpers.environment_dir(ENVIRONMENT_DIR, C.DEFAULT),
    )

    # Docker doesn't really have relevant disk environment, but pre-commit
    # still needs to cleanup its state files on failure
    with clean_path_on_failure(directory):
        build_docker_image(prefix, pull=True)
        os.mkdir(directory)


def get_docker_user() -> Tuple[str, ...]:  # pragma: win32 no cover
    try:
        return ('-u', f'{os.getuid()}:{os.getgid()}')
    except AttributeError:
        return ()


def docker_cmd() -> Tuple[str, ...]:  # pragma: win32 no cover
    return (
        'docker', 'run',
        '--rm',
        *get_docker_user(),
        # https://docs.docker.com/engine/reference/commandline/run/#mount-volumes-from-container-volumes-from
        # The `Z` option tells Docker to label the content with a private
        # unshared label. Only the current container can use a private volume.
        '-v', f'{_get_docker_path(os.getcwd())}:/src:rw,Z',
        '--workdir', '/src',
    )


def run_hook(
        hook: Hook,
        file_args: Sequence[str],
        color: bool,
) -> Tuple[int, bytes]:  # pragma: win32 no cover
    # Rebuild the docker image in case it has gone missing, as many people do
    # automated cleanup of docker images.
    build_docker_image(hook.prefix, pull=False)

    entry_exe, *cmd_rest = hook.cmd

    entry_tag = ('--entrypoint', entry_exe, docker_tag(hook.prefix))
    cmd = (*docker_cmd(), *entry_tag, *cmd_rest)
    return helpers.run_xargs(hook, cmd, file_args, color=color)
