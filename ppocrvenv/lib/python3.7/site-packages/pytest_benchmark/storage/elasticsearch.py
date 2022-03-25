from __future__ import absolute_import

import re
import sys
import uuid
from datetime import date
from datetime import datetime
from decimal import Decimal
from functools import partial

from ..compat import reraise
from ..stats import normalize_stats

try:
    import elasticsearch
    from elasticsearch.serializer import JSONSerializer
except ImportError as exc:
    reraise(ImportError, ImportError("Please install elasticsearch or pytest-benchmark[elasticsearch]", exc.args),
            sys.exc_info()[2])


class BenchmarkJSONSerializer(JSONSerializer):
    def default(self, data):
        if isinstance(data, (date, datetime)):
            return data.isoformat()
        elif isinstance(data, Decimal):
            return float(data)
        elif isinstance(data, uuid.UUID):
            return str(data)
        else:
            return "UNSERIALIZABLE[%r]" % data


def _mask_hosts(hosts):
    m = re.compile('^([^:]+)://[^@]+@')
    sub_fun = partial(m.sub, '\\1://***:***@')
    masked_hosts = list(map(sub_fun, hosts))
    return masked_hosts


class ElasticsearchStorage(object):
    def __init__(self, hosts, index, doctype, project_name, logger,
                 default_machine_id=None):
        self._es_hosts = hosts
        self._es_index = index
        self._es_doctype = doctype
        self._es = elasticsearch.Elasticsearch(self._es_hosts, serializer=BenchmarkJSONSerializer())
        self._project_name = project_name
        self.default_machine_id = default_machine_id
        self.logger = logger
        self._cache = {}
        self._create_index()

    def __str__(self):
        return str(self._es_hosts)

    @property
    def location(self):
        return str(self._es_hosts)

    def query(self):
        """
        Returns sorted records names (ids) that corresponds with project.
        """
        body = {
            "size": 0,
            "aggs": {
                "benchmark_ids": {
                    "terms": {
                        "field": "benchmark_id"
                    }
                }
            }
        }
        result = self._es.search(index=self._es_index, doc_type=self._es_doctype, body=body)
        return sorted([record["key"] for record in result["aggregations"]["benchmark_ids"]["buckets"]])

    def load(self, id_prefix=None):
        """
        Yield key and content of records that corresponds with project name.
        """
        r = self._search(self._project_name, id_prefix)
        groupped_data = self._group_by_commit_and_time(r["hits"]["hits"])
        result = [(key, value) for key, value in groupped_data.items()]
        result.sort(key=lambda x: datetime.strptime(x[1]["datetime"], "%Y-%m-%dT%H:%M:%S.%f"))
        for key, data in result:
            for bench in data["benchmarks"]:
                normalize_stats(bench["stats"])
            yield key, data

    def _search(self, project, id_prefix=None):
        body = {
            "size": 1000,
            "sort": [
                {
                    "datetime": {
                        "order": "desc"
                    }
                }
            ],
            "query": {
                "bool": {
                    "filter": {
                        "term": {
                            "commit_info.project": project
                        }
                    }
                }
            }
        }
        if id_prefix:
            body["query"]["bool"]["must"] = {
                "prefix": {
                    "_id": id_prefix
                }
            }

        return self._es.search(index=self._es_index, doc_type=self._es_doctype, body=body)

    @staticmethod
    def _benchmark_from_es_record(source_es_record):
        result = {}
        for benchmark_key in ("group", "stats", "options", "param", "name", "params", "fullname", "benchmark_id"):
            result[benchmark_key] = source_es_record[benchmark_key]
        return result

    @staticmethod
    def _run_info_from_es_record(source_es_record):
        result = {}
        for run_key in ("machine_info", "commit_info", "datetime", "version"):
            result[run_key] = source_es_record[run_key]
        return result

    def _group_by_commit_and_time(self, hits):
        result = {}
        for hit in hits:
            source_hit = hit["_source"]
            key = "%s_%s" % (source_hit["commit_info"]["id"], source_hit["datetime"])
            benchmark = self._benchmark_from_es_record(source_hit)
            if key in result:
                result[key]["benchmarks"].append(benchmark)
            else:
                run_info = self._run_info_from_es_record(source_hit)
                run_info["benchmarks"] = [benchmark]
                result[key] = run_info
        return result

    def load_benchmarks(self, *args):
        """
        Yield benchmarks that corresponds with project. Put path and
        source (uncommon part of path) to benchmark dict.
        """
        id_prefix = args[0] if args else None
        r = self._search(self._project_name, id_prefix)
        for hit in r["hits"]["hits"]:
            bench = self._benchmark_from_es_record(hit["_source"])
            bench.update(bench.pop("stats"))
            bench["source"] = bench["benchmark_id"]
            yield bench

    def save(self, output_json, save):
        output_benchmarks = output_json.pop("benchmarks")
        for bench in output_benchmarks:
            # add top level info from output_json dict to each record
            bench.update(output_json)
            benchmark_id = save
            if self.default_machine_id:
                benchmark_id = self.default_machine_id + "_" + benchmark_id
            doc_id = benchmark_id + "_" + bench["fullname"]
            bench["benchmark_id"] = benchmark_id
            self._es.index(
                index=self._es_index,
                doc_type=self._es_doctype,
                body=bench,
                id=doc_id,
            )
        # hide user's credentials before logging
        masked_hosts = _mask_hosts(self._es_hosts)
        self.logger.info("Saved benchmark data to %s to index %s as doctype %s" % (
            masked_hosts, self._es_index, self._es_doctype))

    def _create_index(self):
        mapping = {
            "mappings": {
                "benchmark": {
                    "properties": {
                        "commit_info": {
                            "properties": {
                                "dirty": {
                                    "type": "boolean"
                                },
                                "id": {
                                    "type": "string",
                                    "index": "not_analyzed"

                                },
                                "project": {
                                    "type": "string",
                                    "index": "not_analyzed"
                                }
                            }
                        },
                        "datetime": {
                            "type": "date",
                            "format": "strict_date_optional_time||epoch_millis"
                        },
                        "name": {
                            "type": "string",
                            "index": "not_analyzed"
                        },
                        "fullname": {
                            "type": "string",
                            "index": "not_analyzed"
                        },
                        "version": {
                            "type": "string",
                            "index": "not_analyzed"
                        },
                        "benchmark_id": {
                            "type": "string",
                            "index": "not_analyzed",
                        },
                        "machine_info": {
                            "properties": {
                                "machine": {
                                    "type": "string",
                                    "index": "not_analyzed"
                                },
                                "node": {
                                    "type": "string",
                                    "index": "not_analyzed"
                                },
                                "processor": {
                                    "type": "string",
                                    "index": "not_analyzed"
                                },
                                "python_build": {
                                    "type": "string",
                                    "index": "not_analyzed"
                                },
                                "python_compiler": {
                                    "type": "string",
                                    "index": "not_analyzed"
                                },
                                "python_implementation": {
                                    "type": "string",
                                    "index": "not_analyzed"
                                },
                                "python_implementation_version": {
                                    "type": "string",
                                    "index": "not_analyzed"
                                },
                                "python_version": {
                                    "type": "string",
                                    "index": "not_analyzed"
                                },
                                "release": {
                                    "type": "string",
                                    "index": "not_analyzed"
                                },
                                "system": {
                                    "type": "string",
                                    "index": "not_analyzed"
                                }
                            }
                        },
                        "options": {
                            "properties": {
                                "disable_gc": {
                                    "type": "boolean"
                                },
                                "max_time": {
                                    "type": "double"
                                },
                                "min_rounds": {
                                    "type": "long"
                                },
                                "min_time": {
                                    "type": "double"
                                },
                                "timer": {
                                    "type": "string"
                                },
                                "warmup": {
                                    "type": "boolean"
                                }
                            }
                        },
                        "stats": {
                            "properties": {
                                "hd15iqr": {
                                    "type": "double"
                                },
                                "iqr": {
                                    "type": "double"
                                },
                                "iqr_outliers": {
                                    "type": "long"
                                },
                                "iterations": {
                                    "type": "long"
                                },
                                "ld15iqr": {
                                    "type": "double"
                                },
                                "max": {
                                    "type": "double"
                                },
                                "mean": {
                                    "type": "double"
                                },
                                "median": {
                                    "type": "double"
                                },
                                "min": {
                                    "type": "double"
                                },
                                "outliers": {
                                    "type": "string"
                                },
                                "q1": {
                                    "type": "double"
                                },
                                "q3": {
                                    "type": "double"
                                },
                                "rounds": {
                                    "type": "long"
                                },
                                "stddev": {
                                    "type": "double"
                                },
                                "stddev_outliers": {
                                    "type": "long"
                                },
                                "ops": {
                                    "type": "double"
                                },
                            }
                        },
                    }
                }
            }
        }
        self._es.indices.create(index=self._es_index, ignore=400, body=mapping)
