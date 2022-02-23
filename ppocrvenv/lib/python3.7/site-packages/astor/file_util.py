# -*- coding: utf-8 -*-
"""
Part of the astor library for Python AST manipulation.

License: 3-clause BSD

Copyright (c) 2012-2015 Patrick Maupin
Copyright (c) 2013-2015 Berker Peksag

Functions that interact with the filesystem go here.

"""

import ast
import sys
import os

try:
    from tokenize import open as fopen
except ImportError:
    fopen = open


class CodeToAst(object):
    """Given a module, or a function that was compiled as part
    of a module, re-compile the module into an AST and extract
    the sub-AST for the function.  Allow caching to reduce
    number of compiles.

    Also contains static helper utility functions to
    look for python files, to parse python files, and to extract
    the file/line information from a code object.
    """

    @staticmethod
    def find_py_files(srctree, ignore=None):
        """Return all the python files in a source tree

        Ignores any path that contains the ignore string

        This is not used by other class methods, but is
        designed to be used in code that uses this class.
        """

        if not os.path.isdir(srctree):
            yield os.path.split(srctree)
        for srcpath, _, fnames in os.walk(srctree):
            # Avoid infinite recursion for silly users
            if ignore is not None and ignore in srcpath:
                continue
            for fname in (x for x in fnames if x.endswith('.py')):
                yield srcpath, fname

    @staticmethod
    def parse_file(fname):
        """Parse a python file into an AST.

        This is a very thin wrapper around ast.parse

            TODO: Handle encodings other than the default for Python 2
                        (issue #26)
        """
        try:
            with fopen(fname) as f:
                fstr = f.read()
        except IOError:
            if fname != 'stdin':
                raise
            sys.stdout.write('\nReading from stdin:\n\n')
            fstr = sys.stdin.read()
        fstr = fstr.replace('\r\n', '\n').replace('\r', '\n')
        if not fstr.endswith('\n'):
            fstr += '\n'
        return ast.parse(fstr, filename=fname)

    @staticmethod
    def get_file_info(codeobj):
        """Returns the file and line number of a code object.

            If the code object has a __file__ attribute (e.g. if
            it is a module), then the returned line number will
            be 0
        """
        fname = getattr(codeobj, '__file__', None)
        linenum = 0
        if fname is None:
            func_code = codeobj.__code__
            fname = func_code.co_filename
            linenum = func_code.co_firstlineno
        fname = fname.replace('.pyc', '.py')
        return fname, linenum

    def __init__(self, cache=None):
        self.cache = cache or {}

    def __call__(self, codeobj):
        cache = self.cache
        key = self.get_file_info(codeobj)
        result = cache.get(key)
        if result is not None:
            return result
        fname = key[0]
        cache[(fname, 0)] = mod_ast = self.parse_file(fname)
        for obj in mod_ast.body:
            if not isinstance(obj, ast.FunctionDef):
                continue
            cache[(fname, obj.lineno)] = obj
        return cache[key]


code_to_ast = CodeToAst()
