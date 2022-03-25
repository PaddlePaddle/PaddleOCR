from __future__ import division
from __future__ import print_function

from functools import partial

import pytest

from .fixture import statistics
from .fixture import statistics_error
from .logger import Logger
from .table import TableResults
from .utils import NAME_FORMATTERS
from .utils import SecondsDecimal
from .utils import first_or_value
from .utils import get_machine_id
from .utils import load_storage
from .utils import load_timer
from .utils import report_noprogress
from .utils import report_progress
from .utils import safe_dumps
from .utils import short_filename


class PerformanceRegression(Exception):
    pass


class BenchmarkSession(object):
    compared_mapping = None
    groups = None

    def __init__(self, config):
        self.verbose = config.getoption("benchmark_verbose")
        self.quiet = False if self.verbose else config.getoption("benchmark_quiet")
        level = Logger.QUIET if self.quiet else Logger.NORMAL
        if self.verbose:
            level = Logger.VERBOSE
        self.logger = Logger(level, config=config)
        self.config = config
        self.performance_regressions = []
        self.benchmarks = []
        self.machine_id = get_machine_id()
        self.storage = load_storage(
            config.getoption("benchmark_storage"),
            logger=self.logger,
            default_machine_id=self.machine_id,
            netrc=config.getoption("benchmark_netrc")
        )
        self.options = dict(
            min_time=SecondsDecimal(config.getoption("benchmark_min_time")),
            min_rounds=config.getoption("benchmark_min_rounds"),
            max_time=SecondsDecimal(config.getoption("benchmark_max_time")),
            timer=load_timer(config.getoption("benchmark_timer")),
            calibration_precision=config.getoption("benchmark_calibration_precision"),
            disable_gc=config.getoption("benchmark_disable_gc"),
            warmup=config.getoption("benchmark_warmup"),
            warmup_iterations=config.getoption("benchmark_warmup_iterations"),
            cprofile=bool(config.getoption("benchmark_cprofile")),
        )
        self.skip = config.getoption("benchmark_skip")
        self.disabled = config.getoption("benchmark_disable") and not config.getoption("benchmark_enable")
        self.cprofile_sort_by = config.getoption("benchmark_cprofile")

        if config.getoption("dist", "no") != "no" and not self.skip and not self.disabled:
            self.logger.warn(
                "Benchmarks are automatically disabled because xdist plugin is active."
                "Benchmarks cannot be performed reliably in a parallelized environment.",
            )
            self.disabled = True
        if hasattr(config, "slaveinput"):
            self.disabled = True
        if not statistics and not self.disabled:
            self.logger.warn(
                "Benchmarks are automatically disabled because we could not import `statistics`\n\n%s" %
                statistics_error,
            )
            self.disabled = True

        self.only = config.getoption("benchmark_only")
        self.sort = config.getoption("benchmark_sort")
        self.columns = config.getoption("benchmark_columns")
        if self.skip and self.only:
            self.skip = False
        if self.disabled and self.only:
            raise pytest.UsageError(
                "Can't have both --benchmark-only and --benchmark-disable options. Note that --benchmark-disable is "
                "automatically activated if xdist is on or you're missing the statistics dependency.")
        self.group_by = config.getoption("benchmark_group_by")
        self.save = config.getoption("benchmark_save")
        self.autosave = config.getoption("benchmark_autosave")
        self.save_data = config.getoption("benchmark_save_data")
        self.json = config.getoption("benchmark_json")
        self.compare = config.getoption("benchmark_compare")
        self.compare_fail = config.getoption("benchmark_compare_fail")
        self.name_format = NAME_FORMATTERS[config.getoption("benchmark_name")]

        self.histogram = first_or_value(config.getoption("benchmark_histogram"), False)

    def get_machine_info(self):
        obj = self.config.hook.pytest_benchmark_generate_machine_info(config=self.config)
        self.config.hook.pytest_benchmark_update_machine_info(
            config=self.config,
            machine_info=obj
        )
        return obj

    def prepare_benchmarks(self):
        for bench in self.benchmarks:
            if bench:
                compared = False
                for path, compared_mapping in self.compared_mapping.items():
                    if bench.fullname in compared_mapping:
                        compared = compared_mapping[bench.fullname]
                        source = short_filename(path, self.machine_id)
                        flat_bench = bench.as_dict(include_data=False, stats=False, cprofile=self.cprofile_sort_by)
                        flat_bench.update(compared["stats"])
                        flat_bench["path"] = str(path)
                        flat_bench["source"] = source
                        if self.compare_fail:
                            for check in self.compare_fail:
                                fail = check.fails(bench, flat_bench)
                                if fail:
                                    self.performance_regressions.append((self.name_format(flat_bench), fail))
                        yield flat_bench
                flat_bench = bench.as_dict(include_data=False, flat=True, cprofile=self.cprofile_sort_by)
                flat_bench["path"] = None
                flat_bench["source"] = compared and "NOW"
                yield flat_bench

    def save_json(self, output_json):
        with self.json as fh:
            fh.write(safe_dumps(output_json, ensure_ascii=True, indent=4).encode())
        self.logger.info("Wrote benchmark data in: %s" % self.json, purple=True)

    def handle_saving(self):
        save = self.save or self.autosave
        if save or self.json:
            if not self.benchmarks and not self.disabled:
                self.logger.warn("Not saving anything, no benchmarks have been run!")
                return
            machine_info = self.get_machine_info()
            commit_info = self.config.hook.pytest_benchmark_generate_commit_info(config=self.config)
            self.config.hook.pytest_benchmark_update_commit_info(config=self.config, commit_info=commit_info)

        if self.json:
            output_json = self.config.hook.pytest_benchmark_generate_json(
                config=self.config,
                benchmarks=self.benchmarks,
                include_data=True,
                machine_info=machine_info,
                commit_info=commit_info,
            )
            self.config.hook.pytest_benchmark_update_json(
                config=self.config,
                benchmarks=self.benchmarks,
                output_json=output_json,
            )
            self.save_json(output_json)

        if save:
            output_json = self.config.hook.pytest_benchmark_generate_json(
                config=self.config,
                benchmarks=self.benchmarks,
                include_data=self.save_data,
                machine_info=machine_info,
                commit_info=commit_info,
            )
            self.config.hook.pytest_benchmark_update_json(
                config=self.config,
                benchmarks=self.benchmarks,
                output_json=output_json,
            )
            self.storage.save(output_json, save)

    def handle_loading(self):
        compared_mapping = {}
        if self.compare:
            if self.compare is True:
                compared_benchmarks = list(self.storage.load())[-1:]
            else:
                compared_benchmarks = list(self.storage.load(self.compare))

            if not compared_benchmarks:
                msg = "Can't compare. No benchmark files in %r" % str(self.storage)
                if self.compare is True:
                    msg += ". Can't load the previous benchmark."
                else:
                    msg += " match %r." % self.compare
                self.logger.warn(msg)

            machine_info = self.get_machine_info()
            for path, compared_benchmark in compared_benchmarks:
                self.config.hook.pytest_benchmark_compare_machine_info(
                    config=self.config,
                    benchmarksession=self,
                    machine_info=machine_info,
                    compared_benchmark=compared_benchmark,
                )
                compared_mapping[path] = dict(
                    (bench['fullname'], bench) for bench in compared_benchmark['benchmarks']
                )
                self.logger.info("Comparing against benchmarks from: %s" % path, newline=False)
        self.compared_mapping = compared_mapping

    def finish(self):
        self.handle_saving()
        prepared_benchmarks = list(self.prepare_benchmarks())
        if prepared_benchmarks:
            self.groups = self.config.hook.pytest_benchmark_group_stats(
                config=self.config,
                benchmarks=prepared_benchmarks,
                group_by=self.group_by
            )

    def display(self, tr):
        if not self.groups:
            return

        tr.ensure_newline()
        results_table = TableResults(
            columns=self.columns,
            sort=self.sort,
            histogram=self.histogram,
            name_format=self.name_format,
            logger=self.logger,
            scale_unit=partial(self.config.hook.pytest_benchmark_scale_unit, config=self.config),
        )
        progress_reporter = report_progress if self.verbose else report_noprogress
        if not self.quiet:
            results_table.display(tr, self.groups, progress_reporter=progress_reporter)
        self.check_regressions()
        if not self.quiet:
            self.display_cprofile(tr)

    def check_regressions(self):
        if self.compare_fail and not self.compared_mapping:
            raise pytest.UsageError("--benchmark-compare-fail requires valid --benchmark-compare.")

        if self.performance_regressions:
            self.logger.error("Performance has regressed:\n%s" % "\n".join(
                "\t%s - %s" % line for line in self.performance_regressions
            ))
            raise PerformanceRegression("Performance has regressed.")

    def display_cprofile(self, tr):
        section_displayed = False
        for group in self.groups:
            group_name, benchmarks = group
            for benchmark in benchmarks:
                if "cprofile" in benchmark:
                    if not section_displayed:
                        tr.section("cProfile (time in s)", sep="-", yellow=True)
                        section_displayed = True
                    tr.write(benchmark["fullname"], yellow=True)
                    if benchmark["source"]:
                        tr.write_line(" ({})".format((benchmark["source"])))
                    else:
                        tr.write("\n")
                    tr.write_line("ncalls\ttottime\tpercall\tcumtime\tpercall\tfilename:lineno(function)")
                    for function_info in benchmark["cprofile"]:
                        line = ("{ncalls_recursion}\t{tottime:.{prec}f}\t{tottime_per:.{prec}f}\t{cumtime:.{prec}f}"
                                "\t{cumtime_per:.{prec}f}\t{function_name}").format(
                            prec=4, **function_info)
                        tr.write_line(line)
                    tr.write("\n")
