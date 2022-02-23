"""Contains the logic for all of the default options for Flake8."""
import argparse
import functools

from flake8 import defaults
from flake8.main import debug


def register_preliminary_options(parser: argparse.ArgumentParser) -> None:
    """Register the preliminary options on our OptionManager.

    The preliminary options include:

    - ``-v``/``--verbose``
    - ``--output-file``
    - ``--append-config``
    - ``--config``
    - ``--isolated``
    """
    add_argument = parser.add_argument

    add_argument(
        "-v",
        "--verbose",
        default=0,
        action="count",
        help="Print more information about what is happening in flake8."
        " This option is repeatable and will increase verbosity each "
        "time it is repeated.",
    )

    add_argument(
        "--output-file", default=None, help="Redirect report to a file."
    )

    # Config file options

    add_argument(
        "--append-config",
        action="append",
        help="Provide extra config files to parse in addition to the files "
        "found by Flake8 by default. These files are the last ones read "
        "and so they take the highest precedence when multiple files "
        "provide the same option.",
    )

    add_argument(
        "--config",
        default=None,
        help="Path to the config file that will be the authoritative config "
        "source. This will cause Flake8 to ignore all other "
        "configuration files.",
    )

    add_argument(
        "--isolated",
        default=False,
        action="store_true",
        help="Ignore all configuration files.",
    )


class JobsArgument:
    """Type callback for the --jobs argument."""

    def __init__(self, arg: str) -> None:
        """Parse and validate the --jobs argument.

        :param str arg:
            The argument passed by argparse for validation
        """
        self.is_auto = False
        self.n_jobs = -1
        if arg == "auto":
            self.is_auto = True
        elif arg.isdigit():
            self.n_jobs = int(arg)
        else:
            raise argparse.ArgumentTypeError(
                f"{arg!r} must be 'auto' or an integer.",
            )

    def __str__(self):
        """Format our JobsArgument class."""
        return "auto" if self.is_auto else str(self.n_jobs)


def register_default_options(option_manager):
    """Register the default options on our OptionManager.

    The default options include:

    - ``-q``/``--quiet``
    - ``--count``
    - ``--diff``
    - ``--exclude``
    - ``--extend-exclude``
    - ``--filename``
    - ``--format``
    - ``--hang-closing``
    - ``--ignore``
    - ``--extend-ignore``
    - ``--per-file-ignores``
    - ``--max-line-length``
    - ``--max-doc-length``
    - ``--indent-size``
    - ``--select``
    - ``--extend-select``
    - ``--disable-noqa``
    - ``--show-source``
    - ``--statistics``
    - ``--enable-extensions``
    - ``--exit-zero``
    - ``-j``/``--jobs``
    - ``--tee``
    - ``--benchmark``
    - ``--bug-report``
    """
    add_option = option_manager.add_option

    # pep8 options
    add_option(
        "-q",
        "--quiet",
        default=0,
        action="count",
        parse_from_config=True,
        help="Report only file names, or nothing. This option is repeatable.",
    )

    add_option(
        "--count",
        action="store_true",
        parse_from_config=True,
        help="Print total number of errors and warnings to standard error and"
        " set the exit code to 1 if total is not empty.",
    )

    add_option(
        "--diff",
        action="store_true",
        help="Report changes only within line number ranges in the unified "
        "diff provided on standard in by the user.",
    )

    add_option(
        "--exclude",
        metavar="patterns",
        default=",".join(defaults.EXCLUDE),
        comma_separated_list=True,
        parse_from_config=True,
        normalize_paths=True,
        help="Comma-separated list of files or directories to exclude."
        " (Default: %(default)s)",
    )

    add_option(
        "--extend-exclude",
        metavar="patterns",
        default="",
        parse_from_config=True,
        comma_separated_list=True,
        normalize_paths=True,
        help="Comma-separated list of files or directories to add to the list"
        " of excluded ones.",
    )

    add_option(
        "--filename",
        metavar="patterns",
        default="*.py",
        parse_from_config=True,
        comma_separated_list=True,
        help="Only check for filenames matching the patterns in this comma-"
        "separated list. (Default: %(default)s)",
    )

    add_option(
        "--stdin-display-name",
        default="stdin",
        help="The name used when reporting errors from code passed via stdin."
        " This is useful for editors piping the file contents to flake8."
        " (Default: %(default)s)",
    )

    # TODO(sigmavirus24): Figure out --first/--repeat

    # NOTE(sigmavirus24): We can't use choices for this option since users can
    # freely provide a format string and that will break if we restrict their
    # choices.
    add_option(
        "--format",
        metavar="format",
        default="default",
        parse_from_config=True,
        help="Format errors according to the chosen formatter.",
    )

    add_option(
        "--hang-closing",
        action="store_true",
        parse_from_config=True,
        help="Hang closing bracket instead of matching indentation of opening"
        " bracket's line.",
    )

    add_option(
        "--ignore",
        metavar="errors",
        default=",".join(defaults.IGNORE),
        parse_from_config=True,
        comma_separated_list=True,
        help="Comma-separated list of errors and warnings to ignore (or skip)."
        " For example, ``--ignore=E4,E51,W234``. (Default: %(default)s)",
    )

    add_option(
        "--extend-ignore",
        metavar="errors",
        default="",
        parse_from_config=True,
        comma_separated_list=True,
        help="Comma-separated list of errors and warnings to add to the list"
        " of ignored ones. For example, ``--extend-ignore=E4,E51,W234``.",
    )

    add_option(
        "--per-file-ignores",
        default="",
        parse_from_config=True,
        help="A pairing of filenames and violation codes that defines which "
        "violations to ignore in a particular file. The filenames can be "
        "specified in a manner similar to the ``--exclude`` option and the "
        "violations work similarly to the ``--ignore`` and ``--select`` "
        "options.",
    )

    add_option(
        "--max-line-length",
        type=int,
        metavar="n",
        default=defaults.MAX_LINE_LENGTH,
        parse_from_config=True,
        help="Maximum allowed line length for the entirety of this run. "
        "(Default: %(default)s)",
    )

    add_option(
        "--max-doc-length",
        type=int,
        metavar="n",
        default=None,
        parse_from_config=True,
        help="Maximum allowed doc line length for the entirety of this run. "
        "(Default: %(default)s)",
    )
    add_option(
        "--indent-size",
        type=int,
        metavar="n",
        default=defaults.INDENT_SIZE,
        parse_from_config=True,
        help="Number of spaces used for indentation (Default: %(default)s)",
    )

    add_option(
        "--select",
        metavar="errors",
        default=",".join(defaults.SELECT),
        parse_from_config=True,
        comma_separated_list=True,
        help="Comma-separated list of errors and warnings to enable."
        " For example, ``--select=E4,E51,W234``. (Default: %(default)s)",
    )

    add_option(
        "--extend-select",
        metavar="errors",
        default="",
        parse_from_config=True,
        comma_separated_list=True,
        help=(
            "Comma-separated list of errors and warnings to add to the list "
            "of selected ones. For example, ``--extend-select=E4,E51,W234``."
        ),
    )

    add_option(
        "--disable-noqa",
        default=False,
        parse_from_config=True,
        action="store_true",
        help='Disable the effect of "# noqa". This will report errors on '
        'lines with "# noqa" at the end.',
    )

    # TODO(sigmavirus24): Decide what to do about --show-pep8

    add_option(
        "--show-source",
        action="store_true",
        parse_from_config=True,
        help="Show the source generate each error or warning.",
    )
    add_option(
        "--no-show-source",
        action="store_false",
        dest="show_source",
        parse_from_config=False,
        help="Negate --show-source",
    )

    add_option(
        "--statistics",
        action="store_true",
        parse_from_config=True,
        help="Count errors and warnings.",
    )

    # Flake8 options
    add_option(
        "--enable-extensions",
        default="",
        parse_from_config=True,
        comma_separated_list=True,
        help="Enable plugins and extensions that are otherwise disabled "
        "by default",
    )

    add_option(
        "--exit-zero",
        action="store_true",
        help='Exit with status code "0" even if there are errors.',
    )

    add_option(
        "-j",
        "--jobs",
        default="auto",
        parse_from_config=True,
        type=JobsArgument,
        help="Number of subprocesses to use to run checks in parallel. "
        'This is ignored on Windows. The default, "auto", will '
        "auto-detect the number of processors available to use."
        " (Default: %(default)s)",
    )

    add_option(
        "--tee",
        default=False,
        parse_from_config=True,
        action="store_true",
        help="Write to stdout and output-file.",
    )

    # Benchmarking

    add_option(
        "--benchmark",
        default=False,
        action="store_true",
        help="Print benchmark information about this run of Flake8",
    )

    # Debugging

    add_option(
        "--bug-report",
        action=functools.partial(
            debug.DebugAction, option_manager=option_manager
        ),
        nargs=0,
        help="Print information necessary when preparing a bug report",
    )
