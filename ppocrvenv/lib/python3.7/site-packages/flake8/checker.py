"""Checker Manager and Checker classes."""
import collections
import errno
import itertools
import logging
import signal
import tokenize
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from flake8 import defaults
from flake8 import exceptions
from flake8 import processor
from flake8 import utils

try:
    import multiprocessing.pool
except ImportError:
    multiprocessing = None  # type: ignore

Results = List[Tuple[str, int, int, str, Optional[str]]]

LOG = logging.getLogger(__name__)

SERIAL_RETRY_ERRNOS = {
    # ENOSPC: Added by sigmavirus24
    # > On some operating systems (OSX), multiprocessing may cause an
    # > ENOSPC error while trying to trying to create a Semaphore.
    # > In those cases, we should replace the customized Queue Report
    # > class with pep8's StandardReport class to ensure users don't run
    # > into this problem.
    # > (See also: https://github.com/pycqa/flake8/issues/117)
    errno.ENOSPC,
    # NOTE(sigmavirus24): When adding to this list, include the reasoning
    # on the lines before the error code and always append your error
    # code. Further, please always add a trailing `,` to reduce the visual
    # noise in diffs.
}


def _multiprocessing_is_fork():  # type () -> bool
    """Class state is only preserved when using the `fork` strategy."""
    return multiprocessing and multiprocessing.get_start_method() == "fork"


class Manager:
    """Manage the parallelism and checker instances for each plugin and file.

    This class will be responsible for the following:

    - Determining the parallelism of Flake8, e.g.:

      * Do we use :mod:`multiprocessing` or is it unavailable?

      * Do we automatically decide on the number of jobs to use or did the
        user provide that?

    - Falling back to a serial way of processing files if we run into an
      OSError related to :mod:`multiprocessing`

    - Organizing the results of each checker so we can group the output
      together and make our output deterministic.
    """

    def __init__(self, style_guide, arguments, checker_plugins):
        """Initialize our Manager instance.

        :param style_guide:
            The instantiated style guide for this instance of Flake8.
        :type style_guide:
            flake8.style_guide.StyleGuide
        :param list arguments:
            The extra arguments parsed from the CLI (if any)
        :param checker_plugins:
            The plugins representing checks parsed from entry-points.
        :type checker_plugins:
            flake8.plugins.manager.Checkers
        """
        self.arguments = arguments
        self.style_guide = style_guide
        self.options = style_guide.options
        self.checks = checker_plugins
        self.jobs = self._job_count()
        self._all_checkers: List[FileChecker] = []
        self.checkers: List[FileChecker] = []
        self.statistics = {
            "files": 0,
            "logical lines": 0,
            "physical lines": 0,
            "tokens": 0,
        }
        self.exclude = tuple(
            itertools.chain(self.options.exclude, self.options.extend_exclude)
        )

    def _process_statistics(self):
        for checker in self.checkers:
            for statistic in defaults.STATISTIC_NAMES:
                self.statistics[statistic] += checker.statistics[statistic]
        self.statistics["files"] += len(self.checkers)

    def _job_count(self) -> int:
        # First we walk through all of our error cases:
        # - multiprocessing library is not present
        # - we're running on windows in which case we know we have significant
        #   implementation issues
        # - the user provided stdin and that's not something we can handle
        #   well
        # - we're processing a diff, which again does not work well with
        #   multiprocessing and which really shouldn't require multiprocessing
        # - the user provided some awful input
        if not _multiprocessing_is_fork():
            LOG.warning(
                "The multiprocessing module is not available. "
                "Ignoring --jobs arguments."
            )
            return 0

        if utils.is_using_stdin(self.arguments):
            LOG.warning(
                "The --jobs option is not compatible with supplying "
                "input using - . Ignoring --jobs arguments."
            )
            return 0

        if self.options.diff:
            LOG.warning(
                "The --diff option was specified with --jobs but "
                "they are not compatible. Ignoring --jobs arguments."
            )
            return 0

        jobs = self.options.jobs

        # If the value is "auto", we want to let the multiprocessing library
        # decide the number based on the number of CPUs. However, if that
        # function is not implemented for this particular value of Python we
        # default to 1
        if jobs.is_auto:
            try:
                return multiprocessing.cpu_count()
            except NotImplementedError:
                return 0

        # Otherwise, we know jobs should be an integer and we can just convert
        # it to an integer
        return jobs.n_jobs

    def _handle_results(self, filename, results):
        style_guide = self.style_guide
        reported_results_count = 0
        for (error_code, line_number, column, text, physical_line) in results:
            reported_results_count += style_guide.handle_error(
                code=error_code,
                filename=filename,
                line_number=line_number,
                column_number=column,
                text=text,
                physical_line=physical_line,
            )
        return reported_results_count

    def is_path_excluded(self, path: str) -> bool:
        """Check if a path is excluded.

        :param str path:
            Path to check against the exclude patterns.
        :returns:
            True if there are exclude patterns and the path matches,
            otherwise False.
        :rtype:
            bool
        """
        if path == "-":
            if self.options.stdin_display_name == "stdin":
                return False
            path = self.options.stdin_display_name

        return utils.matches_filename(
            path,
            patterns=self.exclude,
            log_message='"%(path)s" has %(whether)sbeen excluded',
            logger=LOG,
        )

    def make_checkers(self, paths: Optional[List[str]] = None) -> None:
        """Create checkers for each file."""
        if paths is None:
            paths = self.arguments

        if not paths:
            paths = ["."]

        filename_patterns = self.options.filename
        running_from_diff = self.options.diff

        # NOTE(sigmavirus24): Yes this is a little unsightly, but it's our
        # best solution right now.
        def should_create_file_checker(filename, argument):
            """Determine if we should create a file checker."""
            matches_filename_patterns = utils.fnmatch(
                filename, filename_patterns
            )
            is_stdin = filename == "-"
            # NOTE(sigmavirus24): If a user explicitly specifies something,
            # e.g, ``flake8 bin/script`` then we should run Flake8 against
            # that. Since should_create_file_checker looks to see if the
            # filename patterns match the filename, we want to skip that in
            # the event that the argument and the filename are identical.
            # If it was specified explicitly, the user intended for it to be
            # checked.
            explicitly_provided = not running_from_diff and (
                argument == filename
            )
            return (
                explicitly_provided or matches_filename_patterns
            ) or is_stdin

        checks = self.checks.to_dictionary()
        self._all_checkers = [
            FileChecker(filename, checks, self.options)
            for argument in paths
            for filename in utils.filenames_from(
                argument, self.is_path_excluded
            )
            if should_create_file_checker(filename, argument)
        ]
        self.checkers = [c for c in self._all_checkers if c.should_process]
        LOG.info("Checking %d files", len(self.checkers))

    def report(self) -> Tuple[int, int]:
        """Report all of the errors found in the managed file checkers.

        This iterates over each of the checkers and reports the errors sorted
        by line number.

        :returns:
            A tuple of the total results found and the results reported.
        :rtype:
            tuple(int, int)
        """
        results_reported = results_found = 0
        for checker in self._all_checkers:
            results = sorted(checker.results, key=lambda tup: (tup[1], tup[2]))
            filename = checker.display_name
            with self.style_guide.processing_file(filename):
                results_reported += self._handle_results(filename, results)
            results_found += len(results)
        return (results_found, results_reported)

    def run_parallel(self) -> None:
        """Run the checkers in parallel."""
        # fmt: off
        final_results: Dict[str, List[Tuple[str, int, int, str, Optional[str]]]] = collections.defaultdict(list)  # noqa: E501
        final_statistics: Dict[str, Dict[str, int]] = collections.defaultdict(dict)  # noqa: E501
        # fmt: on

        pool = _try_initialize_processpool(self.jobs)

        if pool is None:
            self.run_serial()
            return

        pool_closed = False
        try:
            pool_map = pool.imap_unordered(
                _run_checks,
                self.checkers,
                chunksize=calculate_pool_chunksize(
                    len(self.checkers), self.jobs
                ),
            )
            for ret in pool_map:
                filename, results, statistics = ret
                final_results[filename] = results
                final_statistics[filename] = statistics
            pool.close()
            pool.join()
            pool_closed = True
        finally:
            if not pool_closed:
                pool.terminate()
                pool.join()

        for checker in self.checkers:
            filename = checker.display_name
            checker.results = final_results[filename]
            checker.statistics = final_statistics[filename]

    def run_serial(self) -> None:
        """Run the checkers in serial."""
        for checker in self.checkers:
            checker.run_checks()

    def run(self) -> None:
        """Run all the checkers.

        This will intelligently decide whether to run the checks in parallel
        or whether to run them in serial.

        If running the checks in parallel causes a problem (e.g.,
        https://github.com/pycqa/flake8/issues/117) this also implements
        fallback to serial processing.
        """
        try:
            if self.jobs > 1 and len(self.checkers) > 1:
                self.run_parallel()
            else:
                self.run_serial()
        except KeyboardInterrupt:
            LOG.warning("Flake8 was interrupted by the user")
            raise exceptions.EarlyQuit("Early quit while running checks")

    def start(self, paths=None):
        """Start checking files.

        :param list paths:
            Path names to check. This is passed directly to
            :meth:`~Manager.make_checkers`.
        """
        LOG.info("Making checkers")
        self.make_checkers(paths)

    def stop(self):
        """Stop checking files."""
        self._process_statistics()


class FileChecker:
    """Manage running checks for a file and aggregate the results."""

    def __init__(self, filename, checks, options):
        """Initialize our file checker.

        :param str filename:
            Name of the file to check.
        :param checks:
            The plugins registered to check the file.
        :type checks:
            dict
        :param options:
            Parsed option values from config and command-line.
        :type options:
            argparse.Namespace
        """
        self.options = options
        self.filename = filename
        self.checks = checks
        self.results: Results = []
        self.statistics = {
            "tokens": 0,
            "logical lines": 0,
            "physical lines": 0,
        }
        self.processor = self._make_processor()
        self.display_name = filename
        self.should_process = False
        if self.processor is not None:
            self.display_name = self.processor.filename
            self.should_process = not self.processor.should_ignore_file()
            self.statistics["physical lines"] = len(self.processor.lines)

    def __repr__(self) -> str:
        """Provide helpful debugging representation."""
        return f"FileChecker for {self.filename}"

    def _make_processor(self) -> Optional[processor.FileProcessor]:
        try:
            return processor.FileProcessor(self.filename, self.options)
        except OSError as e:
            # If we can not read the file due to an IOError (e.g., the file
            # does not exist or we do not have the permissions to open it)
            # then we need to format that exception for the user.
            # NOTE(sigmavirus24): Historically, pep8 has always reported this
            # as an E902. We probably *want* a better error code for this
            # going forward.
            self.report("E902", 0, 0, f"{type(e).__name__}: {e}")
            return None

    def report(
        self,
        error_code: Optional[str],
        line_number: int,
        column: int,
        text: str,
    ) -> str:
        """Report an error by storing it in the results list."""
        if error_code is None:
            error_code, text = text.split(" ", 1)

        # If we're recovering from a problem in _make_processor, we will not
        # have this attribute.
        if hasattr(self, "processor") and self.processor is not None:
            line = self.processor.noqa_line_for(line_number)
        else:
            line = None

        self.results.append((error_code, line_number, column, text, line))
        return error_code

    def run_check(self, plugin, **arguments):
        """Run the check in a single plugin."""
        LOG.debug("Running %r with %r", plugin, arguments)
        assert self.processor is not None
        try:
            self.processor.keyword_arguments_for(
                plugin["parameters"], arguments
            )
        except AttributeError as ae:
            LOG.error("Plugin requested unknown parameters.")
            raise exceptions.PluginRequestedUnknownParameters(
                plugin=plugin, exception=ae
            )
        try:
            return plugin["plugin"](**arguments)
        except Exception as all_exc:
            LOG.critical(
                "Plugin %s raised an unexpected exception",
                plugin["name"],
                exc_info=True,
            )
            raise exceptions.PluginExecutionFailed(
                plugin=plugin, exception=all_exc
            )

    @staticmethod
    def _extract_syntax_information(exception: Exception) -> Tuple[int, int]:
        if (
            len(exception.args) > 1
            and exception.args[1]
            and len(exception.args[1]) > 2
        ):
            token = exception.args[1]
            row, column = token[1:3]
        elif (
            isinstance(exception, tokenize.TokenError)
            and len(exception.args) == 2
            and len(exception.args[1]) == 2
        ):
            token = ()
            row, column = exception.args[1]
        else:
            token = ()
            row, column = (1, 0)

        if (
            column > 0
            and token
            and isinstance(exception, SyntaxError)
            and len(token) == 4  # Python 3.9 or earlier
        ):
            # NOTE(sigmavirus24): SyntaxErrors report 1-indexed column
            # numbers. We need to decrement the column number by 1 at
            # least.
            column_offset = 1
            row_offset = 0
            # See also: https://github.com/pycqa/flake8/issues/169,
            # https://github.com/PyCQA/flake8/issues/1372
            # On Python 3.9 and earlier, token will be a 4-item tuple with the
            # last item being the string. Starting with 3.10, they added to
            # the tuple so now instead of it ending with the code that failed
            # to parse, it ends with the end of the section of code that
            # failed to parse. Luckily the absolute position in the tuple is
            # stable across versions so we can use that here
            physical_line = token[3]

            # NOTE(sigmavirus24): Not all "tokens" have a string as the last
            # argument. In this event, let's skip trying to find the correct
            # column and row values.
            if physical_line is not None:
                # NOTE(sigmavirus24): SyntaxErrors also don't exactly have a
                # "physical" line so much as what was accumulated by the point
                # tokenizing failed.
                # See also: https://github.com/pycqa/flake8/issues/169
                lines = physical_line.rstrip("\n").split("\n")
                row_offset = len(lines) - 1
                logical_line = lines[0]
                logical_line_length = len(logical_line)
                if column > logical_line_length:
                    column = logical_line_length
            row -= row_offset
            column -= column_offset
        return row, column

    def run_ast_checks(self) -> None:
        """Run all checks expecting an abstract syntax tree."""
        assert self.processor is not None
        ast = self.processor.build_ast()

        for plugin in self.checks["ast_plugins"]:
            checker = self.run_check(plugin, tree=ast)
            # If the plugin uses a class, call the run method of it, otherwise
            # the call should return something iterable itself
            try:
                runner = checker.run()
            except AttributeError:
                runner = checker
            for (line_number, offset, text, _) in runner:
                self.report(
                    error_code=None,
                    line_number=line_number,
                    column=offset,
                    text=text,
                )

    def run_logical_checks(self):
        """Run all checks expecting a logical line."""
        assert self.processor is not None
        comments, logical_line, mapping = self.processor.build_logical_line()
        if not mapping:
            return
        self.processor.update_state(mapping)

        LOG.debug('Logical line: "%s"', logical_line.rstrip())

        for plugin in self.checks["logical_line_plugins"]:
            self.processor.update_checker_state_for(plugin)
            results = self.run_check(plugin, logical_line=logical_line) or ()
            for offset, text in results:
                line_number, column_offset = find_offset(offset, mapping)
                if line_number == column_offset == 0:
                    LOG.warning("position of error out of bounds: %s", plugin)
                self.report(
                    error_code=None,
                    line_number=line_number,
                    column=column_offset,
                    text=text,
                )

        self.processor.next_logical_line()

    def run_physical_checks(self, physical_line):
        """Run all checks for a given physical line.

        A single physical check may return multiple errors.
        """
        assert self.processor is not None
        for plugin in self.checks["physical_line_plugins"]:
            self.processor.update_checker_state_for(plugin)
            result = self.run_check(plugin, physical_line=physical_line)

            if result is not None:
                # This is a single result if first element is an int
                column_offset = None
                try:
                    column_offset = result[0]
                except (IndexError, TypeError):
                    pass

                if isinstance(column_offset, int):
                    # If we only have a single result, convert to a collection
                    result = (result,)

                for result_single in result:
                    column_offset, text = result_single
                    self.report(
                        error_code=None,
                        line_number=self.processor.line_number,
                        column=column_offset,
                        text=text,
                    )

    def process_tokens(self):
        """Process tokens and trigger checks.

        Instead of using this directly, you should use
        :meth:`flake8.checker.FileChecker.run_checks`.
        """
        assert self.processor is not None
        parens = 0
        statistics = self.statistics
        file_processor = self.processor
        prev_physical = ""
        for token in file_processor.generate_tokens():
            statistics["tokens"] += 1
            self.check_physical_eol(token, prev_physical)
            token_type, text = token[0:2]
            processor.log_token(LOG, token)
            if token_type == tokenize.OP:
                parens = processor.count_parentheses(parens, text)
            elif parens == 0:
                if processor.token_is_newline(token):
                    self.handle_newline(token_type)
            prev_physical = token[4]

        if file_processor.tokens:
            # If any tokens are left over, process them
            self.run_physical_checks(file_processor.lines[-1])
            self.run_logical_checks()

    def run_checks(self) -> Tuple[str, Results, Dict[str, int]]:
        """Run checks against the file."""
        assert self.processor is not None
        try:
            self.run_ast_checks()
            self.process_tokens()
        except (SyntaxError, tokenize.TokenError) as e:
            code = "E902" if isinstance(e, tokenize.TokenError) else "E999"
            row, column = self._extract_syntax_information(e)
            self.report(code, row, column, f"{type(e).__name__}: {e.args[0]}")
            return self.filename, self.results, self.statistics

        logical_lines = self.processor.statistics["logical lines"]
        self.statistics["logical lines"] = logical_lines
        return self.filename, self.results, self.statistics

    def handle_newline(self, token_type):
        """Handle the logic when encountering a newline token."""
        assert self.processor is not None
        if token_type == tokenize.NEWLINE:
            self.run_logical_checks()
            self.processor.reset_blank_before()
        elif len(self.processor.tokens) == 1:
            # The physical line contains only this token.
            self.processor.visited_new_blank_line()
            self.processor.delete_first_token()
        else:
            self.run_logical_checks()

    def check_physical_eol(
        self, token: processor._Token, prev_physical: str
    ) -> None:
        """Run physical checks if and only if it is at the end of the line."""
        assert self.processor is not None
        # a newline token ends a single physical line.
        if processor.is_eol_token(token):
            # if the file does not end with a newline, the NEWLINE
            # token is inserted by the parser, but it does not contain
            # the previous physical line in `token[4]`
            if token[4] == "":
                self.run_physical_checks(prev_physical)
            else:
                self.run_physical_checks(token[4])
        elif processor.is_multiline_string(token):
            # Less obviously, a string that contains newlines is a
            # multiline string, either triple-quoted or with internal
            # newlines backslash-escaped. Check every physical line in the
            # string *except* for the last one: its newline is outside of
            # the multiline string, so we consider it a regular physical
            # line, and will check it like any other physical line.
            #
            # Subtleties:
            # - have to wind self.line_number back because initially it
            #   points to the last line of the string, and we want
            #   check_physical() to give accurate feedback
            line_no = token[2][0]
            with self.processor.inside_multiline(line_number=line_no):
                for line in self.processor.split_line(token):
                    self.run_physical_checks(line + "\n")


def _pool_init() -> None:
    """Ensure correct signaling of ^C using multiprocessing.Pool."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _try_initialize_processpool(
    job_count: int,
) -> Optional[multiprocessing.pool.Pool]:
    """Return a new process pool instance if we are able to create one."""
    try:
        return multiprocessing.Pool(job_count, _pool_init)
    except OSError as err:
        if err.errno not in SERIAL_RETRY_ERRNOS:
            raise
    except ImportError:
        pass

    return None


def calculate_pool_chunksize(num_checkers, num_jobs):
    """Determine the chunksize for the multiprocessing Pool.

    - For chunksize, see: https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.imap  # noqa
    - This formula, while not perfect, aims to give each worker two batches of
      work.
    - See: https://github.com/pycqa/flake8/issues/829#note_18878876
    - See: https://github.com/pycqa/flake8/issues/197
    """
    return max(num_checkers // (num_jobs * 2), 1)


def _run_checks(checker):
    return checker.run_checks()


def find_offset(
    offset: int, mapping: processor._LogicalMapping
) -> Tuple[int, int]:
    """Find the offset tuple for a single offset."""
    if isinstance(offset, tuple):
        return offset

    for token in mapping:
        token_offset = token[0]
        if offset <= token_offset:
            position = token[1]
            break
    else:
        position = (0, 0)
        offset = token_offset = 0
    return (position[0], position[1] + offset - token_offset)
