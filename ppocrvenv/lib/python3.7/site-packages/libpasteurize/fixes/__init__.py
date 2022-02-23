import sys
from lib2to3 import refactor

# The original set of these fixes comes from lib3to2 (https://bitbucket.org/amentajo/lib3to2):
fix_names = set([
                 'libpasteurize.fixes.fix_add_all__future__imports',  # from __future__ import absolute_import etc. on separate lines
                 'libpasteurize.fixes.fix_add_future_standard_library_import',  # we force adding this import for now, even if it doesn't seem necessary to the fix_future_standard_library fixer, for ease of testing
                 # 'libfuturize.fixes.fix_order___future__imports',  # consolidates to a single line to simplify testing -- UNFINISHED
                 'libpasteurize.fixes.fix_future_builtins',   # adds "from future.builtins import *"
                 'libfuturize.fixes.fix_future_standard_library', # adds "from future import standard_library"

                 'libpasteurize.fixes.fix_annotations',
                 # 'libpasteurize.fixes.fix_bitlength',  # ints have this in Py2.7
                 # 'libpasteurize.fixes.fix_bool',    # need a decorator or Mixin
                 # 'libpasteurize.fixes.fix_bytes',   # leave bytes as bytes
                 # 'libpasteurize.fixes.fix_classdecorator',  # available in
                 # Py2.6+
                 # 'libpasteurize.fixes.fix_collections', hmmm ...
                 # 'libpasteurize.fixes.fix_dctsetcomp',  # avail in Py27
                 'libpasteurize.fixes.fix_division',   # yes
                 # 'libpasteurize.fixes.fix_except',   # avail in Py2.6+
                 # 'libpasteurize.fixes.fix_features',  # ?
                 'libpasteurize.fixes.fix_fullargspec',
                 # 'libpasteurize.fixes.fix_funcattrs',
                 'libpasteurize.fixes.fix_getcwd',
                 'libpasteurize.fixes.fix_imports',   # adds "from future import standard_library"
                 'libpasteurize.fixes.fix_imports2',
                 # 'libpasteurize.fixes.fix_input',
                 # 'libpasteurize.fixes.fix_int',
                 # 'libpasteurize.fixes.fix_intern',
                 # 'libpasteurize.fixes.fix_itertools',
                 'libpasteurize.fixes.fix_kwargs',   # yes, we want this
                 # 'libpasteurize.fixes.fix_memoryview',
                 # 'libpasteurize.fixes.fix_metaclass',  # write a custom handler for
                 # this
                 # 'libpasteurize.fixes.fix_methodattrs',  # __func__ and __self__ seem to be defined on Py2.7 already
                 'libpasteurize.fixes.fix_newstyle',   # yes, we want this: explicit inheritance from object. Without new-style classes in Py2, super() will break etc.
                 # 'libpasteurize.fixes.fix_next',   # use a decorator for this
                 # 'libpasteurize.fixes.fix_numliterals',   # prob not
                 # 'libpasteurize.fixes.fix_open',   # huh?
                 # 'libpasteurize.fixes.fix_print',  # no way
                 'libpasteurize.fixes.fix_printfunction',  # adds __future__ import print_function
                 # 'libpasteurize.fixes.fix_raise_',   # TODO: get this working!

                 # 'libpasteurize.fixes.fix_range',  # nope
                 # 'libpasteurize.fixes.fix_reduce',
                 # 'libpasteurize.fixes.fix_setliteral',
                 # 'libpasteurize.fixes.fix_str',
                 # 'libpasteurize.fixes.fix_super',  # maybe, if our magic super() isn't robust enough
                 'libpasteurize.fixes.fix_throw',   # yes, if Py3 supports it
                 # 'libpasteurize.fixes.fix_unittest',
                 'libpasteurize.fixes.fix_unpacking',  # yes, this is useful
                 # 'libpasteurize.fixes.fix_with'      # way out of date
                ])
