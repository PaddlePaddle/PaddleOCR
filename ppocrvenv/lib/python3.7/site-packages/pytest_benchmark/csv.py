from __future__ import absolute_import

import csv
import operator

import py


class CSVResults(object):
    def __init__(self, columns, sort, logger):
        self.columns = columns
        self.sort = sort
        self.logger = logger

    def render(self, output_file, groups):
        output_file = py.path.local(output_file)
        if not output_file.ext:
            output_file = output_file.new(ext='csv')
        with output_file.open('w', ensure=True) as stream:
            writer = csv.writer(stream)
            params = sorted(set(
                param
                for group, benchmarks in groups
                for benchmark in benchmarks
                for param in benchmark.get("params", {}) or ()
            ))
            writer.writerow([
                "name",
            ] + [
                "param:{0}".format(p)
                for p in params
            ] + self.columns)

            for group, benchmarks in groups:
                benchmarks = sorted(benchmarks, key=operator.itemgetter(self.sort))

                for bench in benchmarks:
                    row = [bench.get("fullfunc", bench["fullname"])]
                    row.extend(bench.get('params', {}).get(param, "") for param in params)
                    row.extend(bench[prop] for prop in self.columns)
                    writer.writerow(row)
        self.logger.info("Generated csv: {0}".format(output_file), bold=True)
