"""
Generic test utilities.

"""

import os
import re
import sys


__all__ = ['PytestTester', 'check_free_memory']


class FPUModeChangeWarning(RuntimeWarning):
    """Warning about FPU mode change"""
    pass


class PytestTester:
    """
    Pytest test runner entry point.
    """

    def __init__(self, module_name):
        self.module_name = module_name

    def __call__(self, label="fast", verbose=1, extra_argv=None, doctests=False,
                 coverage=False, tests=None, parallel=None):
        import pytest

        module = sys.modules[self.module_name]
        module_path = os.path.abspath(module.__path__[0])

        pytest_args = ['--showlocals', '--tb=short']

        if doctests:
            raise ValueError("Doctests not supported")

        if extra_argv:
            pytest_args += list(extra_argv)

        if verbose and int(verbose) > 1:
            pytest_args += ["-" + "v"*(int(verbose)-1)]

        if coverage:
            pytest_args += ["--cov=" + module_path]

        if label == "fast":
            pytest_args += ["-m", "not slow"]
        elif label != "full":
            pytest_args += ["-m", label]

        if tests is None:
            tests = [self.module_name]

        if parallel is not None and parallel > 1:
            if _pytest_has_xdist():
                pytest_args += ['-n', str(parallel)]
            else:
                import warnings
                warnings.warn('Could not run tests in parallel because '
                              'pytest-xdist plugin is not available.')

        pytest_args += ['--pyargs'] + list(tests)

        try:
            code = pytest.main(pytest_args)
        except SystemExit as exc:
            code = exc.code

        return (code == 0)


def _pytest_has_xdist():
    """
    Check if the pytest-xdist plugin is installed, providing parallel tests
    """
    # Check xdist exists without importing, otherwise pytests emits warnings
    from importlib.util import find_spec
    return find_spec('xdist') is not None


def check_free_memory(free_mb):
    """
    Check *free_mb* of memory is available, otherwise do pytest.skip
    """
    import pytest

    try:
        mem_free = _parse_size(os.environ['SCIPY_AVAILABLE_MEM'])
        msg = '{0} MB memory required, but environment SCIPY_AVAILABLE_MEM={1}'.format(
            free_mb, os.environ['SCIPY_AVAILABLE_MEM'])
    except KeyError:
        mem_free = _get_mem_available()
        if mem_free is None:
            pytest.skip("Could not determine available memory; set SCIPY_AVAILABLE_MEM "
                        "variable to free memory in MB to run the test.")
        msg = '{0} MB memory required, but {1} MB available'.format(
            free_mb, mem_free/1e6)

    if mem_free < free_mb * 1e6:
        pytest.skip(msg)


def _parse_size(size_str):
    suffixes = {'': 1e6,
                'b': 1.0,
                'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12,
                'kb': 1e3, 'Mb': 1e6, 'Gb': 1e9, 'Tb': 1e12,
                'kib': 1024.0, 'Mib': 1024.0**2, 'Gib': 1024.0**3, 'Tib': 1024.0**4}
    m = re.match(r'^\s*(\d+)\s*({0})\s*$'.format('|'.join(suffixes.keys())),
                 size_str,
                 re.I)
    if not m or m.group(2) not in suffixes:
        raise ValueError("Invalid size string")

    return float(m.group(1)) * suffixes[m.group(2)]


def _get_mem_available():
    """
    Get information about memory available, not counting swap.
    """
    try:
        import psutil
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass

    if sys.platform.startswith('linux'):
        info = {}
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                p = line.split()
                info[p[0].strip(':').lower()] = float(p[1]) * 1e3

        if 'memavailable' in info:
            # Linux >= 3.14
            return info['memavailable']
        else:
            return info['memfree'] + info['cached']

    return None
