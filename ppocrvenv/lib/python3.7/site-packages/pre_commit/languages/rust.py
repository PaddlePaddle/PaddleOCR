import contextlib
import os.path
from typing import Generator
from typing import Sequence
from typing import Set
from typing import Tuple

import toml

import pre_commit.constants as C
from pre_commit.envcontext import envcontext
from pre_commit.envcontext import PatchesT
from pre_commit.envcontext import Var
from pre_commit.hook import Hook
from pre_commit.languages import helpers
from pre_commit.prefix import Prefix
from pre_commit.util import clean_path_on_failure
from pre_commit.util import cmd_output_b

ENVIRONMENT_DIR = 'rustenv'
get_default_version = helpers.basic_get_default_version
healthy = helpers.basic_healthy


def get_env_patch(target_dir: str) -> PatchesT:
    return (
        ('PATH', (os.path.join(target_dir, 'bin'), os.pathsep, Var('PATH'))),
    )


@contextlib.contextmanager
def in_env(prefix: Prefix) -> Generator[None, None, None]:
    target_dir = prefix.path(
        helpers.environment_dir(ENVIRONMENT_DIR, C.DEFAULT),
    )
    with envcontext(get_env_patch(target_dir)):
        yield


def _add_dependencies(
        cargo_toml_path: str,
        additional_dependencies: Set[str],
) -> None:
    with open(cargo_toml_path, 'r+') as f:
        cargo_toml = toml.load(f)
        cargo_toml.setdefault('dependencies', {})
        for dep in additional_dependencies:
            name, _, spec = dep.partition(':')
            cargo_toml['dependencies'][name] = spec or '*'
        f.seek(0)
        toml.dump(cargo_toml, f)
        f.truncate()


def install_environment(
        prefix: Prefix,
        version: str,
        additional_dependencies: Sequence[str],
) -> None:
    helpers.assert_version_default('rust', version)
    directory = prefix.path(
        helpers.environment_dir(ENVIRONMENT_DIR, C.DEFAULT),
    )

    # There are two cases where we might want to specify more dependencies:
    # as dependencies for the library being built, and as binary packages
    # to be `cargo install`'d.
    #
    # Unlike e.g. Python, if we just `cargo install` a library, it won't be
    # used for compilation. And if we add a crate providing a binary to the
    # `Cargo.toml`, the binary won't be built.
    #
    # Because of this, we allow specifying "cli" dependencies by prefixing
    # with 'cli:'.
    cli_deps = {
        dep for dep in additional_dependencies if dep.startswith('cli:')
    }
    lib_deps = set(additional_dependencies) - cli_deps

    if len(lib_deps) > 0:
        _add_dependencies(prefix.path('Cargo.toml'), lib_deps)

    with clean_path_on_failure(directory):
        packages_to_install: Set[Tuple[str, ...]] = {('--path', '.')}
        for cli_dep in cli_deps:
            cli_dep = cli_dep[len('cli:'):]
            package, _, version = cli_dep.partition(':')
            if version != '':
                packages_to_install.add((package, '--version', version))
            else:
                packages_to_install.add((package,))

        for args in packages_to_install:
            cmd_output_b(
                'cargo', 'install', '--bins', '--root', directory, *args,
                cwd=prefix.prefix_dir,
            )


def run_hook(
        hook: Hook,
        file_args: Sequence[str],
        color: bool,
) -> Tuple[int, bytes]:
    with in_env(hook.prefix):
        return helpers.run_xargs(hook, hook.cmd, file_args, color=color)
