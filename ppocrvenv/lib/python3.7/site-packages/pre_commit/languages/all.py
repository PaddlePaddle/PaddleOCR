from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple

from pre_commit.hook import Hook
from pre_commit.languages import conda
from pre_commit.languages import coursier
from pre_commit.languages import dart
from pre_commit.languages import docker
from pre_commit.languages import docker_image
from pre_commit.languages import dotnet
from pre_commit.languages import fail
from pre_commit.languages import golang
from pre_commit.languages import lua
from pre_commit.languages import node
from pre_commit.languages import perl
from pre_commit.languages import pygrep
from pre_commit.languages import python
from pre_commit.languages import r
from pre_commit.languages import ruby
from pre_commit.languages import rust
from pre_commit.languages import script
from pre_commit.languages import swift
from pre_commit.languages import system
from pre_commit.prefix import Prefix


class Language(NamedTuple):
    name: str
    # Use `None` for no installation / environment
    ENVIRONMENT_DIR: Optional[str]
    # return a value to replace `'default` for `language_version`
    get_default_version: Callable[[], str]
    # return whether the environment is healthy (or should be rebuilt)
    healthy: Callable[[Prefix, str], bool]
    # install a repository for the given language and language_version
    install_environment: Callable[[Prefix, str, Sequence[str]], None]
    # execute a hook and return the exit code and output
    run_hook: 'Callable[[Hook, Sequence[str], bool], Tuple[int, bytes]]'


# TODO: back to modules + Protocol: https://github.com/python/mypy/issues/5018
languages = {
    # BEGIN GENERATED (testing/gen-languages-all)
    'conda': Language(name='conda', ENVIRONMENT_DIR=conda.ENVIRONMENT_DIR, get_default_version=conda.get_default_version, healthy=conda.healthy, install_environment=conda.install_environment, run_hook=conda.run_hook),  # noqa: E501
    'coursier': Language(name='coursier', ENVIRONMENT_DIR=coursier.ENVIRONMENT_DIR, get_default_version=coursier.get_default_version, healthy=coursier.healthy, install_environment=coursier.install_environment, run_hook=coursier.run_hook),  # noqa: E501
    'dart': Language(name='dart', ENVIRONMENT_DIR=dart.ENVIRONMENT_DIR, get_default_version=dart.get_default_version, healthy=dart.healthy, install_environment=dart.install_environment, run_hook=dart.run_hook),  # noqa: E501
    'docker': Language(name='docker', ENVIRONMENT_DIR=docker.ENVIRONMENT_DIR, get_default_version=docker.get_default_version, healthy=docker.healthy, install_environment=docker.install_environment, run_hook=docker.run_hook),  # noqa: E501
    'docker_image': Language(name='docker_image', ENVIRONMENT_DIR=docker_image.ENVIRONMENT_DIR, get_default_version=docker_image.get_default_version, healthy=docker_image.healthy, install_environment=docker_image.install_environment, run_hook=docker_image.run_hook),  # noqa: E501
    'dotnet': Language(name='dotnet', ENVIRONMENT_DIR=dotnet.ENVIRONMENT_DIR, get_default_version=dotnet.get_default_version, healthy=dotnet.healthy, install_environment=dotnet.install_environment, run_hook=dotnet.run_hook),  # noqa: E501
    'fail': Language(name='fail', ENVIRONMENT_DIR=fail.ENVIRONMENT_DIR, get_default_version=fail.get_default_version, healthy=fail.healthy, install_environment=fail.install_environment, run_hook=fail.run_hook),  # noqa: E501
    'golang': Language(name='golang', ENVIRONMENT_DIR=golang.ENVIRONMENT_DIR, get_default_version=golang.get_default_version, healthy=golang.healthy, install_environment=golang.install_environment, run_hook=golang.run_hook),  # noqa: E501
    'lua': Language(name='lua', ENVIRONMENT_DIR=lua.ENVIRONMENT_DIR, get_default_version=lua.get_default_version, healthy=lua.healthy, install_environment=lua.install_environment, run_hook=lua.run_hook),  # noqa: E501
    'node': Language(name='node', ENVIRONMENT_DIR=node.ENVIRONMENT_DIR, get_default_version=node.get_default_version, healthy=node.healthy, install_environment=node.install_environment, run_hook=node.run_hook),  # noqa: E501
    'perl': Language(name='perl', ENVIRONMENT_DIR=perl.ENVIRONMENT_DIR, get_default_version=perl.get_default_version, healthy=perl.healthy, install_environment=perl.install_environment, run_hook=perl.run_hook),  # noqa: E501
    'pygrep': Language(name='pygrep', ENVIRONMENT_DIR=pygrep.ENVIRONMENT_DIR, get_default_version=pygrep.get_default_version, healthy=pygrep.healthy, install_environment=pygrep.install_environment, run_hook=pygrep.run_hook),  # noqa: E501
    'python': Language(name='python', ENVIRONMENT_DIR=python.ENVIRONMENT_DIR, get_default_version=python.get_default_version, healthy=python.healthy, install_environment=python.install_environment, run_hook=python.run_hook),  # noqa: E501
    'r': Language(name='r', ENVIRONMENT_DIR=r.ENVIRONMENT_DIR, get_default_version=r.get_default_version, healthy=r.healthy, install_environment=r.install_environment, run_hook=r.run_hook),  # noqa: E501
    'ruby': Language(name='ruby', ENVIRONMENT_DIR=ruby.ENVIRONMENT_DIR, get_default_version=ruby.get_default_version, healthy=ruby.healthy, install_environment=ruby.install_environment, run_hook=ruby.run_hook),  # noqa: E501
    'rust': Language(name='rust', ENVIRONMENT_DIR=rust.ENVIRONMENT_DIR, get_default_version=rust.get_default_version, healthy=rust.healthy, install_environment=rust.install_environment, run_hook=rust.run_hook),  # noqa: E501
    'script': Language(name='script', ENVIRONMENT_DIR=script.ENVIRONMENT_DIR, get_default_version=script.get_default_version, healthy=script.healthy, install_environment=script.install_environment, run_hook=script.run_hook),  # noqa: E501
    'swift': Language(name='swift', ENVIRONMENT_DIR=swift.ENVIRONMENT_DIR, get_default_version=swift.get_default_version, healthy=swift.healthy, install_environment=swift.install_environment, run_hook=swift.run_hook),  # noqa: E501
    'system': Language(name='system', ENVIRONMENT_DIR=system.ENVIRONMENT_DIR, get_default_version=system.get_default_version, healthy=system.healthy, install_environment=system.install_environment, run_hook=system.run_hook),  # noqa: E501
    # END GENERATED
}
# TODO: fully deprecate `python_venv`
languages['python_venv'] = languages['python']
all_languages = sorted(languages)
