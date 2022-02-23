# tests __main__ module handling in multiprocessing
from test import support
# Skip tests if _multiprocessing wasn't built.
support.import_module('_multiprocessing')

import importlib
import importlib.machinery
import unittest
import sys
import os
import os.path
import py_compile

from test.support.script_helper import (
    make_pkg, make_script, make_zip_pkg, make_zip_script,
    assert_python_ok)

if support.PGO:
    raise unittest.SkipTest("test is not helpful for PGO")

# Look up which start methods are available to test
import multiprocess as multiprocessing
AVAILABLE_START_METHODS = set(multiprocessing.get_all_start_methods())

# Issue #22332: Skip tests if sem_open implementation is broken.
support.import_module('multiprocess.synchronize')

verbose = support.verbose

test_source = """\
# multiprocessing includes all sorts of shenanigans to make __main__
# attributes accessible in the subprocess in a pickle compatible way.

# We run the "doesn't work in the interactive interpreter" example from
# the docs to make sure it *does* work from an executed __main__,
# regardless of the invocation mechanism

import sys
import time
from multiprocess import Pool, set_start_method

# We use this __main__ defined function in the map call below in order to
# check that multiprocessing in correctly running the unguarded
# code in child processes and then making it available as __main__
def f(x):
    return x*x

# Check explicit relative imports
if "check_sibling" in __file__:
    # We're inside a package and not in a __main__.py file
    # so make sure explicit relative imports work correctly
    from . import sibling

if __name__ == '__main__':
    start_method = sys.argv[1]
    set_start_method(start_method)
    results = []
    with Pool(5) as pool:
        pool.map_async(f, [1, 2, 3], callback=results.extend)
        start_time = getattr(time,'monotonic',time.time)()
        while not results:
            time.sleep(0.05)
            # up to 1 min to report the results
            dt = getattr(time,'monotonic',time.time)() - start_time
            if dt > 60.0:
                raise RuntimeError("Timed out waiting for results (%.1f sec)" % dt)
    results.sort()
    print(start_method, "->", results)

    pool.join()
"""

test_source_main_skipped_in_children = """\
# __main__.py files have an implied "if __name__ == '__main__'" so
# multiprocessing should always skip running them in child processes

# This means we can't use __main__ defined functions in child processes,
# so we just use "int" as a passthrough operation below

if __name__ != "__main__":
    raise RuntimeError("Should only be called as __main__!")

import sys
import time
from multiprocess import Pool, set_start_method

start_method = sys.argv[1]
set_start_method(start_method)
results = []
with Pool(5) as pool:
    pool.map_async(int, [1, 4, 9], callback=results.extend)
    start_time = getattr(time,'monotonic',time.time)()
    while not results:
        time.sleep(0.05)
        # up to 1 min to report the results
        dt = getattr(time,'monotonic',time.time)() - start_time
        if dt > 60.0:
            raise RuntimeError("Timed out waiting for results (%.1f sec)" % dt)

results.sort()
print(start_method, "->", results)

pool.join()
"""

# These helpers were copied from test_cmd_line_script & tweaked a bit...

def _make_test_script(script_dir, script_basename,
                      source=test_source, omit_suffix=False):
    to_return = make_script(script_dir, script_basename,
                            source, omit_suffix)
    # Hack to check explicit relative imports
    if script_basename == "check_sibling":
        make_script(script_dir, "sibling", "")
    importlib.invalidate_caches()
    return to_return

def _make_test_zip_pkg(zip_dir, zip_basename, pkg_name, script_basename,
                       source=test_source, depth=1):
    to_return = make_zip_pkg(zip_dir, zip_basename, pkg_name, script_basename,
                             source, depth)
    importlib.invalidate_caches()
    return to_return

# There's no easy way to pass the script directory in to get
# -m to work (avoiding that is the whole point of making
# directories and zipfiles executable!)
# So we fake it for testing purposes with a custom launch script
launch_source = """\
import sys, os.path, runpy
sys.path.insert(0, %s)
runpy._run_module_as_main(%r)
"""

def _make_launch_script(script_dir, script_basename, module_name, path=None):
    if path is None:
        path = "os.path.dirname(__file__)"
    else:
        path = repr(path)
    source = launch_source % (path, module_name)
    to_return = make_script(script_dir, script_basename, source)
    importlib.invalidate_caches()
    return to_return

class MultiProcessingCmdLineMixin():
    maxDiff = None # Show full tracebacks on subprocess failure

    def setUp(self):
        if self.start_method not in AVAILABLE_START_METHODS:
            self.skipTest("%r start method not available" % self.start_method)

    def _check_output(self, script_name, exit_code, out, err):
        if verbose > 1:
            print("Output from test script %r:" % script_name)
            print(repr(out))
        self.assertEqual(exit_code, 0)
        self.assertEqual(err.decode('utf-8'), '')
        expected_results = "%s -> [1, 4, 9]" % self.start_method
        self.assertEqual(out.decode('utf-8').strip(), expected_results)

    def _check_script(self, script_name, *cmd_line_switches):
        if not __debug__:
            cmd_line_switches += ('-' + 'O' * sys.flags.optimize,)
        run_args = cmd_line_switches + (script_name, self.start_method)
        rc, out, err = assert_python_ok(*run_args, __isolated=False)
        self._check_output(script_name, rc, out, err)

    def test_basic_script(self):
        with support.temp_dir() as script_dir:
            script_name = _make_test_script(script_dir, 'script')
            self._check_script(script_name)

    def test_basic_script_no_suffix(self):
        with support.temp_dir() as script_dir:
            script_name = _make_test_script(script_dir, 'script',
                                            omit_suffix=True)
            self._check_script(script_name)

    def test_ipython_workaround(self):
        # Some versions of the IPython launch script are missing the
        # __name__ = "__main__" guard, and multiprocessing has long had
        # a workaround for that case
        # See https://github.com/ipython/ipython/issues/4698
        source = test_source_main_skipped_in_children
        with support.temp_dir() as script_dir:
            script_name = _make_test_script(script_dir, 'ipython',
                                            source=source)
            self._check_script(script_name)
            script_no_suffix = _make_test_script(script_dir, 'ipython',
                                                 source=source,
                                                 omit_suffix=True)
            self._check_script(script_no_suffix)

    def test_script_compiled(self):
        with support.temp_dir() as script_dir:
            script_name = _make_test_script(script_dir, 'script')
            py_compile.compile(script_name, doraise=True)
            os.remove(script_name)
            pyc_file = support.make_legacy_pyc(script_name)
            self._check_script(pyc_file)

    def test_directory(self):
        source = self.main_in_children_source
        with support.temp_dir() as script_dir:
            script_name = _make_test_script(script_dir, '__main__',
                                            source=source)
            self._check_script(script_dir)

    def test_directory_compiled(self):
        source = self.main_in_children_source
        with support.temp_dir() as script_dir:
            script_name = _make_test_script(script_dir, '__main__',
                                            source=source)
            py_compile.compile(script_name, doraise=True)
            os.remove(script_name)
            pyc_file = support.make_legacy_pyc(script_name)
            self._check_script(script_dir)

    def test_zipfile(self):
        source = self.main_in_children_source
        with support.temp_dir() as script_dir:
            script_name = _make_test_script(script_dir, '__main__',
                                            source=source)
            zip_name, run_name = make_zip_script(script_dir, 'test_zip', script_name)
            self._check_script(zip_name)

    def test_zipfile_compiled(self):
        source = self.main_in_children_source
        with support.temp_dir() as script_dir:
            script_name = _make_test_script(script_dir, '__main__',
                                            source=source)
            compiled_name = py_compile.compile(script_name, doraise=True)
            zip_name, run_name = make_zip_script(script_dir, 'test_zip', compiled_name)
            self._check_script(zip_name)

    def test_module_in_package(self):
        with support.temp_dir() as script_dir:
            pkg_dir = os.path.join(script_dir, 'test_pkg')
            make_pkg(pkg_dir)
            script_name = _make_test_script(pkg_dir, 'check_sibling')
            launch_name = _make_launch_script(script_dir, 'launch',
                                              'test_pkg.check_sibling')
            self._check_script(launch_name)

    def test_module_in_package_in_zipfile(self):
        with support.temp_dir() as script_dir:
            zip_name, run_name = _make_test_zip_pkg(script_dir, 'test_zip', 'test_pkg', 'script')
            launch_name = _make_launch_script(script_dir, 'launch', 'test_pkg.script', zip_name)
            self._check_script(launch_name)

    def test_module_in_subpackage_in_zipfile(self):
        with support.temp_dir() as script_dir:
            zip_name, run_name = _make_test_zip_pkg(script_dir, 'test_zip', 'test_pkg', 'script', depth=2)
            launch_name = _make_launch_script(script_dir, 'launch', 'test_pkg.test_pkg.script', zip_name)
            self._check_script(launch_name)

    def test_package(self):
        source = self.main_in_children_source
        with support.temp_dir() as script_dir:
            pkg_dir = os.path.join(script_dir, 'test_pkg')
            make_pkg(pkg_dir)
            script_name = _make_test_script(pkg_dir, '__main__',
                                            source=source)
            launch_name = _make_launch_script(script_dir, 'launch', 'test_pkg')
            self._check_script(launch_name)

    def test_package_compiled(self):
        source = self.main_in_children_source
        with support.temp_dir() as script_dir:
            pkg_dir = os.path.join(script_dir, 'test_pkg')
            make_pkg(pkg_dir)
            script_name = _make_test_script(pkg_dir, '__main__',
                                            source=source)
            compiled_name = py_compile.compile(script_name, doraise=True)
            os.remove(script_name)
            pyc_file = support.make_legacy_pyc(script_name)
            launch_name = _make_launch_script(script_dir, 'launch', 'test_pkg')
            self._check_script(launch_name)

# Test all supported start methods (setupClass skips as appropriate)

class SpawnCmdLineTest(MultiProcessingCmdLineMixin, unittest.TestCase):
    start_method = 'spawn'
    main_in_children_source = test_source_main_skipped_in_children

class ForkCmdLineTest(MultiProcessingCmdLineMixin, unittest.TestCase):
    start_method = 'fork'
    main_in_children_source = test_source

class ForkServerCmdLineTest(MultiProcessingCmdLineMixin, unittest.TestCase):
    start_method = 'forkserver'
    main_in_children_source = test_source_main_skipped_in_children

def tearDownModule():
    support.reap_children()

if __name__ == '__main__':
    unittest.main()
