# flake8: noqa
import sys

if sys.version_info[:2] < (3, 0):
    def exec_file_wrapper(fpath, g_vars, l_vars):
        execfile(fpath, g_vars, l_vars)
