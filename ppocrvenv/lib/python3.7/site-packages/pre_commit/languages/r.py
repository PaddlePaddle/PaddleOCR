import contextlib
import os
import shlex
import shutil
from typing import Generator
from typing import Sequence
from typing import Tuple

from pre_commit.envcontext import envcontext
from pre_commit.envcontext import PatchesT
from pre_commit.envcontext import UNSET
from pre_commit.hook import Hook
from pre_commit.languages import helpers
from pre_commit.prefix import Prefix
from pre_commit.util import clean_path_on_failure
from pre_commit.util import cmd_output_b

ENVIRONMENT_DIR = 'renv'
RSCRIPT_OPTS = ('--no-save', '--no-restore', '--no-site-file', '--no-environ')
get_default_version = helpers.basic_get_default_version
healthy = helpers.basic_healthy


def get_env_patch(venv: str) -> PatchesT:
    return (
        ('R_PROFILE_USER', os.path.join(venv, 'activate.R')),
        ('RENV_PROJECT', UNSET),
    )


@contextlib.contextmanager
def in_env(
        prefix: Prefix,
        language_version: str,
) -> Generator[None, None, None]:
    envdir = _get_env_dir(prefix, language_version)
    with envcontext(get_env_patch(envdir)):
        yield


def _get_env_dir(prefix: Prefix, version: str) -> str:
    return prefix.path(helpers.environment_dir(ENVIRONMENT_DIR, version))


def _prefix_if_non_local_file_entry(
    entry: Sequence[str],
    prefix: Prefix,
    src: str,
) -> Sequence[str]:
    if entry[1] == '-e':
        return entry[1:]
    else:
        if src == 'local':
            path = entry[1]
        else:
            path = prefix.path(entry[1])
        return (path,)


def _rscript_exec() -> str:
    return os.path.join(os.getenv('R_HOME', ''), 'Rscript')


def _entry_validate(entry: Sequence[str]) -> None:
    """
    Allowed entries:
    # Rscript -e expr
    # Rscript path/to/file
    """
    if entry[0] != 'Rscript':
        raise ValueError('entry must start with `Rscript`.')

    if entry[1] == '-e':
        if len(entry) > 3:
            raise ValueError('You can supply at most one expression.')
    elif len(entry) > 2:
        raise ValueError(
            'The only valid syntax is `Rscript -e {expr}`',
            'or `Rscript path/to/hook/script`',
        )


def _cmd_from_hook(hook: Hook) -> Tuple[str, ...]:
    entry = shlex.split(hook.entry)
    _entry_validate(entry)

    return (
        *entry[:1], *RSCRIPT_OPTS,
        *_prefix_if_non_local_file_entry(entry, hook.prefix, hook.src),
        *hook.args,
    )


def install_environment(
        prefix: Prefix,
        version: str,
        additional_dependencies: Sequence[str],
) -> None:
    env_dir = _get_env_dir(prefix, version)
    with clean_path_on_failure(env_dir):
        os.makedirs(env_dir, exist_ok=True)
        shutil.copy(prefix.path('renv.lock'), env_dir)
        shutil.copytree(prefix.path('renv'), os.path.join(env_dir, 'renv'))

        cmd_output_b(
            _rscript_exec(), '--vanilla', '-e',
            f"""\
            prefix_dir <- {prefix.prefix_dir!r}
            options(
                repos = c(CRAN = "https://cran.rstudio.com"),
                renv.consent = TRUE
            )
            source("renv/activate.R")
            renv::restore()
            activate_statement <- paste0(
              'suppressWarnings({{',
              'old <- setwd("', getwd(), '"); ',
              'source("renv/activate.R"); ',
              'setwd(old); ',
              'renv::load("', getwd(), '");}})'
            )
            writeLines(activate_statement, 'activate.R')
            is_package <- tryCatch(
              {{
                  path_desc <- file.path(prefix_dir, 'DESCRIPTION')
                  suppressWarnings(desc <- read.dcf(path_desc))
                  "Package" %in% colnames(desc)
              }},
              error = function(...) FALSE
            )
            if (is_package) {{
                renv::install(prefix_dir)
            }}
            """,
            cwd=env_dir,
        )
        if additional_dependencies:
            with in_env(prefix, version):
                cmd_output_b(
                    _rscript_exec(), *RSCRIPT_OPTS, '-e',
                    'renv::install(commandArgs(trailingOnly = TRUE))',
                    *additional_dependencies,
                    cwd=env_dir,
                )


def run_hook(
        hook: Hook,
        file_args: Sequence[str],
        color: bool,
) -> Tuple[int, bytes]:
    with in_env(hook.prefix, hook.language_version):
        return helpers.run_xargs(
            hook, _cmd_from_hook(hook), file_args, color=color,
        )
