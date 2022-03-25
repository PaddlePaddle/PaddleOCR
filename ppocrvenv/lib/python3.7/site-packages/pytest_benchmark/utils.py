from __future__ import division
from __future__ import print_function

import argparse
import json
import netrc
import ntpath
import os
import platform
import re
import subprocess
import sys
import types
from datetime import datetime
from decimal import Decimal
from functools import partial
from os.path import basename
from os.path import dirname
from os.path import exists
from os.path import join
from os.path import split

import genericpath

from .compat import PY3
from .compat import PY38

# This is here (in the utils module) because it might be used by
# various other modules.
try:
    from pathlib2 import Path  # noqa: F401
except ImportError:
    from pathlib import Path  # noqa: F401

try:
    from urllib.parse import parse_qs
    from urllib.parse import urlparse
except ImportError:
    from urlparse import parse_qs
    from urlparse import urlparse

try:
    from subprocess import CalledProcessError
    from subprocess import check_output
except ImportError:
    class CalledProcessError(subprocess.CalledProcessError):
        def __init__(self, returncode, cmd, output=None):
            super(CalledProcessError, self).__init__(returncode, cmd)
            self.output = output

    def check_output(*popenargs, **kwargs):
        if 'stdout' in kwargs:
            raise ValueError('stdout argument not allowed, it will be overridden.')
        process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            raise CalledProcessError(retcode, cmd, output)
        return output

TIME_UNITS = {
    "": "Seconds",
    "m": "Milliseconds (ms)",
    "u": "Microseconds (us)",
    "n": "Nanoseconds (ns)"
}
ALLOWED_COLUMNS = ["min", "max", "mean", "stddev", "median", "iqr", "ops", "outliers", "rounds", "iterations"]


class SecondsDecimal(Decimal):
    def __float__(self):
        return float(super(SecondsDecimal, self).__str__())

    def __str__(self):
        return "{0}s".format(format_time(float(super(SecondsDecimal, self).__str__())))

    @property
    def as_string(self):
        return super(SecondsDecimal, self).__str__()


class NameWrapper(object):
    def __init__(self, target):
        self.target = target

    def __str__(self):
        name = self.target.__module__ + "." if hasattr(self.target, '__module__') else ""
        name += self.target.__name__ if hasattr(self.target, '__name__') else repr(self.target)
        return name

    def __repr__(self):
        return "NameWrapper(%s)" % repr(self.target)


def get_tag(project_name=None):
    info = get_commit_info(project_name)
    parts = [info['id'], get_current_time()]
    if info['dirty']:
        parts.append("uncommited-changes")
    return "_".join(parts)


def get_machine_id():
    return "%s-%s-%s-%s" % (
        platform.system(),
        platform.python_implementation(),
        ".".join(platform.python_version_tuple()[:2]),
        platform.architecture()[0]
    )


class Fallback(object):
    def __init__(self, fallback, exceptions):
        self.fallback = fallback
        self.functions = []
        self.exceptions = exceptions

    def __call__(self, *args, **kwargs):
        for func in self.functions:
            try:
                value = func(*args, **kwargs)
            except self.exceptions:
                continue
            else:
                if value:
                    return value
        else:
            return self.fallback(*args, **kwargs)

    def register(self, other):
        self.functions.append(other)
        return self


@partial(Fallback, exceptions=(IndexError, CalledProcessError, OSError))
def get_project_name():
    return basename(os.getcwd())


@get_project_name.register
def get_project_name_git():
    is_git = check_output(['git', 'rev-parse', '--git-dir'], stderr=subprocess.STDOUT)
    if is_git:
        project_address = check_output(['git', 'config', '--local', 'remote.origin.url'])
        if isinstance(project_address, bytes) and str != bytes:
            project_address = project_address.decode()
        project_name = [i for i in re.split(r'[/:\s\\]|\.git', project_address) if i][-1]
        return project_name.strip()


@get_project_name.register
def get_project_name_hg():
    with open(os.devnull, 'w') as devnull:
        project_address = check_output(['hg', 'path', 'default'], stderr=devnull)
    project_address = project_address.decode()
    project_name = project_address.split("/")[-1]
    return project_name.strip()


def in_any_parent(name, path=None):
    prev = None
    if not path:
        path = os.getcwd()
    while path and prev != path and not exists(join(path, name)):
        prev = path
        path = dirname(path)
    return exists(join(path, name))


def subprocess_output(cmd):
    return check_output(cmd.split(), stderr=subprocess.STDOUT, universal_newlines=True).strip()


def get_commit_info(project_name=None):
    dirty = False
    commit = 'unversioned'
    commit_time = None
    author_time = None
    project_name = project_name or get_project_name()
    branch = '(unknown)'
    try:
        if in_any_parent('.git'):
            desc = subprocess_output('git describe --dirty --always --long --abbrev=40')
            desc = desc.split('-')
            if desc[-1].strip() == 'dirty':
                dirty = True
                desc.pop()
            commit = desc[-1].strip('g')
            commit_time = subprocess_output('git show -s --pretty=format:"%cI"').strip('"')
            author_time = subprocess_output('git show -s --pretty=format:"%aI"').strip('"')
            branch = subprocess_output('git rev-parse --abbrev-ref HEAD')
            if branch == 'HEAD':
                branch = '(detached head)'
        elif in_any_parent('.hg'):
            desc = subprocess_output('hg id --id --debug')
            if desc[-1] == '+':
                dirty = True
            commit = desc.strip('+')
            commit_time = subprocess_output('hg tip --template "{date|rfc3339date}"').strip('"')
            branch = subprocess_output('hg branch')
        return {
            'id': commit,
            'time': commit_time,
            'author_time': author_time,
            'dirty': dirty,
            'project': project_name,
            'branch': branch,
        }
    except Exception as exc:
        return {
            'id': 'unknown',
            'time': None,
            'author_time': None,
            'dirty': dirty,
            'error': 'CalledProcessError({0.returncode}, {0.output!r})'.format(exc)
                     if isinstance(exc, CalledProcessError) else repr(exc),
            'project': project_name,
            'branch': branch,
        }


def get_current_time():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def first_or_value(obj, value):
    if obj:
        value, = obj

    return value


def short_filename(path, machine_id=None):
    parts = []
    try:
        last = len(path.parts) - 1
    except AttributeError:
        return str(path)
    for pos, part in enumerate(path.parts):
        if not pos and part == machine_id:
            continue
        if pos == last:
            part = part.rsplit('.', 1)[0]
            # if len(part) > 16:
            #     part = "%.13s..." % part
        parts.append(part)
    return '/'.join(parts)


def load_timer(string):
    if "." not in string:
        raise argparse.ArgumentTypeError("Value for --benchmark-timer must be in dotted form. Eg: 'module.attr'.")
    mod, attr = string.rsplit(".", 1)
    if mod == 'pep418':
        if PY3:
            import time
            return NameWrapper(getattr(time, attr))
        else:
            from . import pep418
            return NameWrapper(getattr(pep418, attr))
    else:
        __import__(mod)
        mod = sys.modules[mod]
        return NameWrapper(getattr(mod, attr))


class RegressionCheck(object):
    def __init__(self, field, threshold):
        self.field = field
        self.threshold = threshold

    def fails(self, current, compared):
        val = self.compute(current, compared)
        if val > self.threshold:
            return "Field %r has failed %s: %.9f > %.9f" % (
                self.field, self.__class__.__name__, val, self.threshold
            )


class PercentageRegressionCheck(RegressionCheck):
    def compute(self, current, compared):
        val = compared[self.field]
        if not val:
            return float("inf")
        return current[self.field] / val * 100 - 100


class DifferenceRegressionCheck(RegressionCheck):
    def compute(self, current, compared):
        return current[self.field] - compared[self.field]


def parse_compare_fail(string,
                       rex=re.compile(r'^(?P<field>min|max|mean|median|stddev|iqr):'
                                      r'((?P<percentage>[0-9]?[0-9])%|(?P<difference>[0-9]*\.?[0-9]+([eE][-+]?['
                                      r'0-9]+)?))$')):
    m = rex.match(string)
    if m:
        g = m.groupdict()
        if g['percentage']:
            return PercentageRegressionCheck(g['field'], int(g['percentage']))
        elif g['difference']:
            return DifferenceRegressionCheck(g['field'], float(g['difference']))

    raise argparse.ArgumentTypeError("Could not parse value: %r." % string)


def parse_warmup(string):
    string = string.lower().strip()
    if string == "auto":
        return platform.python_implementation() == "PyPy"
    elif string in ["off", "false", "no"]:
        return False
    elif string in ["on", "true", "yes", ""]:
        return True
    else:
        raise argparse.ArgumentTypeError("Could not parse value: %r." % string)


def name_formatter_short(bench):
    name = bench["name"]
    if bench["source"]:
        name = "%s (%.4s)" % (name, split(bench["source"])[-1])
    if name.startswith("test_"):
        name = name[5:]
    return name


def name_formatter_normal(bench):
    name = bench["name"]
    if bench["source"]:
        parts = bench["source"].split('/')
        parts[-1] = parts[-1][:12]
        name = "%s (%s)" % (name, '/'.join(parts))
    return name


def name_formatter_long(bench):
    if bench["source"]:
        return "%(fullname)s (%(source)s)" % bench
    else:
        return bench["fullname"]


def name_formatter_trial(bench):
    if bench["source"]:
        return "%.4s" % split(bench["source"])[-1]
    else:
        return '????'


NAME_FORMATTERS = {
    "short": name_formatter_short,
    "normal": name_formatter_normal,
    "long": name_formatter_long,
    "trial": name_formatter_trial,
}


def parse_name_format(string):
    string = string.lower().strip()
    if string in NAME_FORMATTERS:
        return string
    else:
        raise argparse.ArgumentTypeError("Could not parse value: %r." % string)


def parse_timer(string):
    return str(load_timer(string))


def parse_sort(string):
    string = string.lower().strip()
    if string not in ("min", "max", "mean", "stddev", "name", "fullname"):
        raise argparse.ArgumentTypeError(
            "Unacceptable value: %r. "
            "Value for --benchmark-sort must be one of: 'min', 'max', 'mean', "
            "'stddev', 'name', 'fullname'." % string)
    return string


def parse_columns(string):
    columns = [str.strip(s) for s in string.lower().split(',')]
    invalid = set(columns) - set(ALLOWED_COLUMNS)
    if invalid:
        # there are extra items in columns!
        msg = "Invalid column name(s): %s. " % ', '.join(invalid)
        msg += "The only valid column names are: %s" % ', '.join(ALLOWED_COLUMNS)
        raise argparse.ArgumentTypeError(msg)
    return columns


def parse_rounds(string):
    try:
        value = int(string)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(exc)
    else:
        if value < 1:
            raise argparse.ArgumentTypeError("Value for --benchmark-rounds must be at least 1.")
        return value


def parse_seconds(string):
    try:
        return SecondsDecimal(string).as_string
    except Exception as exc:
        raise argparse.ArgumentTypeError("Invalid decimal value %r: %r" % (string, exc))


def parse_save(string):
    if not string:
        raise argparse.ArgumentTypeError("Can't be empty.")
    illegal = ''.join(c for c in r"\/:*?<>|" if c in string)
    if illegal:
        raise argparse.ArgumentTypeError("Must not contain any of these characters: /:*?<>|\\ (it has %r)" % illegal)
    return string


def _parse_hosts(storage_url, netrc_file):

    # load creds from netrc file
    path = os.path.expanduser(netrc_file)
    creds = None
    if netrc_file and os.path.isfile(path):
        creds = netrc.netrc(path)

    # add creds to urls
    urls = []
    for netloc in storage_url.netloc.split(','):
        auth = ""
        if creds and '@' not in netloc:
            host = netloc.split(':').pop(0)
            res = creds.authenticators(host)
            if res:
                user, _, secret = res
                auth = "{user}:{secret}@".format(user=user, secret=secret)
        url = "{scheme}://{auth}{netloc}".format(scheme=storage_url.scheme,
                                                 netloc=netloc, auth=auth)
        urls.append(url)
    return urls


def parse_elasticsearch_storage(string, default_index="benchmark",
                                default_doctype="benchmark", netrc_file=''):
    storage_url = urlparse(string)
    hosts = _parse_hosts(storage_url, netrc_file)
    index = default_index
    doctype = default_doctype
    if storage_url.path and storage_url.path != "/":
        splitted = storage_url.path.strip("/").split("/")
        index = splitted[0]
        if len(splitted) >= 2:
            doctype = splitted[1]
    query = parse_qs(storage_url.query)
    try:
        project_name = query["project_name"][0]
    except KeyError:
        project_name = get_project_name()
    return hosts, index, doctype, project_name


def load_storage(storage, **kwargs):
    if "://" not in storage:
        storage = "file://" + storage
    netrc_file = kwargs.pop('netrc')  # only used by elasticsearch storage
    if storage.startswith("file://"):
        from .storage.file import FileStorage
        return FileStorage(storage[len("file://"):], **kwargs)
    elif storage.startswith("elasticsearch+"):
        from .storage.elasticsearch import ElasticsearchStorage

        # TODO update benchmark_autosave
        args = parse_elasticsearch_storage(storage[len("elasticsearch+"):],
                                           netrc_file=netrc_file)
        return ElasticsearchStorage(*args, **kwargs)
    else:
        raise argparse.ArgumentTypeError("Storage must be in form of file://path or "
                                         "elasticsearch+http[s]://host1,host2/index/doctype")


def time_unit(value):
    if value < 1e-6:
        return "n", 1e9
    elif value < 1e-3:
        return "u", 1e6
    elif value < 1:
        return "m", 1e3
    else:
        return "", 1.


def operations_unit(value):
    if value > 1e+6:
        return "M", 1e-6
    if value > 1e+3:
        return "K", 1e-3
    return "", 1.


def format_time(value):
    unit, adjustment = time_unit(value)
    return "{0:.2f}{1:s}".format(value * adjustment, unit)


class cached_property(object):
    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def funcname(f):
    try:
        if isinstance(f, partial):
            return f.func.__name__
        else:
            return f.__name__
    except AttributeError:
        return str(f)


# from: https://bitbucket.org/antocuni/pypytools/src/tip/pypytools/util.py?at=default
def clonefunc(f):
    """Deep clone the given function to create a new one.

    By default, the PyPy JIT specializes the assembler based on f.__code__:
    clonefunc makes sure that you will get a new function with a **different**
    __code__, so that PyPy will produce independent assembler. This is useful
    e.g. for benchmarks and microbenchmarks, so you can make sure to compare
    apples to apples.

    Use it with caution: if abused, this might easily produce an explosion of
    produced assembler.
    """
    # first of all, we clone the code object
    if not hasattr(f, '__code__'):
        return f
    co = f.__code__
    args = [co.co_argcount, co.co_nlocals, co.co_stacksize, co.co_flags, co.co_code,
            co.co_consts, co.co_names, co.co_varnames, co.co_filename, co.co_name,
            co.co_firstlineno, co.co_lnotab, co.co_freevars, co.co_cellvars]
    if PY38:
        args.insert(1, co.co_posonlyargcount)
    if PY3:
        args.insert(1, co.co_kwonlyargcount)
    co2 = types.CodeType(*args)
    #
    # then, we clone the function itself, using the new co2
    f2 = types.FunctionType(co2, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
    return f2


def format_dict(obj):
    return "{%s}" % ", ".join("%s: %s" % (k, json.dumps(v)) for k, v in sorted(obj.items()))


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, o):
        return "UNSERIALIZABLE[%r]" % o


def safe_dumps(obj, **kwargs):
    return json.dumps(obj, cls=SafeJSONEncoder, **kwargs)


def report_progress(iterable, terminal_reporter, format_string, **kwargs):
    total = len(iterable)

    def progress_reporting_wrapper():
        for pos, item in enumerate(iterable):
            string = format_string.format(pos=pos + 1, total=total, value=item, **kwargs)
            terminal_reporter.rewrite(string, black=True, bold=True)
            yield string, item

    return progress_reporting_wrapper()


def report_noprogress(iterable, *args, **kwargs):
    for pos, item in enumerate(iterable):
        yield "", item


def report_online_progress(progress_reporter, tr, line):
    next(progress_reporter([line], tr, "{value}"))


def slugify(name):
    for c in r"\/:*?<>| ":
        name = name.replace(c, '_').replace('__', '_')
    return name


def commonpath(paths):
    """Given a sequence of path names, returns the longest common sub-path."""

    if not paths:
        raise ValueError('commonpath() arg is an empty sequence')

    if isinstance(paths[0], bytes):
        sep = b'\\'
        altsep = b'/'
        curdir = b'.'
    else:
        sep = '\\'
        altsep = '/'
        curdir = '.'

    try:
        drivesplits = [ntpath.splitdrive(p.replace(altsep, sep).lower()) for p in paths]
        split_paths = [p.split(sep) for d, p in drivesplits]

        try:
            isabs, = set(p[:1] == sep for d, p in drivesplits)
        except ValueError:
            raise ValueError("Can't mix absolute and relative paths")

        # Check that all drive letters or UNC paths match. The check is made only
        # now otherwise type errors for mixing strings and bytes would not be
        # caught.
        if len(set(d for d, p in drivesplits)) != 1:
            raise ValueError("Paths don't have the same drive")

        drive, path = ntpath.splitdrive(paths[0].replace(altsep, sep))
        common = path.split(sep)
        common = [c for c in common if c and c != curdir]

        split_paths = [[c for c in s if c and c != curdir] for s in split_paths]
        s1 = min(split_paths)
        s2 = max(split_paths)
        for i, c in enumerate(s1):
            if c != s2[i]:
                common = common[:i]
                break
        else:
            common = common[:len(s1)]

        prefix = drive + sep if isabs else drive
        return prefix + sep.join(common)
    except (TypeError, AttributeError):
        genericpath._check_arg_types('commonpath', *paths)
        raise


def get_cprofile_functions(stats):
    """
    Convert pstats structure to list of sorted dicts about each function.
    """
    result = []
    # this assumes that you run py.test from project root dir
    project_dir_parent = dirname(os.getcwd())

    for function_info, run_info in stats.stats.items():
        file_path = function_info[0]
        if file_path.startswith(project_dir_parent):
            file_path = file_path[len(project_dir_parent):].lstrip('/')
        function_name = '{0}:{1}({2})'.format(file_path, function_info[1], function_info[2])

        # if the function is recursive write number of 'total calls/primitive calls'
        if run_info[0] == run_info[1]:
            calls = str(run_info[0])
        else:
            calls = '{1}/{0}'.format(run_info[0], run_info[1])

        result.append(dict(ncalls_recursion=calls,
                           ncalls=run_info[1],
                           tottime=run_info[2],
                           tottime_per=run_info[2] / run_info[0] if run_info[0] > 0 else 0,
                           cumtime=run_info[3],
                           cumtime_per=run_info[3] / run_info[0] if run_info[0] > 0 else 0,
                           function_name=function_name))

    return result
