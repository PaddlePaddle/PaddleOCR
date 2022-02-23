#
#  Cython - Compilation-wide options and pragma declarations
#

from __future__ import absolute_import


class ShouldBeFromDirective(object):

    known_directives = []

    def __init__(self, options_name, directive_name=None, disallow=False):
        self.options_name = options_name
        self.directive_name = directive_name or options_name
        self.disallow = disallow
        self.known_directives.append(self)

    def __nonzero__(self):
        self._bad_access()

    def __int__(self):
        self._bad_access()

    def _bad_access(self):
        raise RuntimeError(repr(self))

    def __repr__(self):
        return (
        "Illegal access of '%s' from Options module rather than directive '%s'"
        % (self.options_name, self.directive_name))


"""
The members of this module are documented using autodata in
Cython/docs/src/reference/compilation.rst.
See http://www.sphinx-doc.org/en/master/ext/autodoc.html#directive-autoattribute
for how autodata works.
Descriptions of those members should start with a #:
Donc forget to keep the docs in sync by removing and adding
the members in both this file and the .rst file.
"""

#: Whether or not to include docstring in the Python extension. If False, the binary size
#: will be smaller, but the ``__doc__`` attribute of any class or function will be an
#: empty string.
docstrings = True

#: Embed the source code position in the docstrings of functions and classes.
embed_pos_in_docstring = False

#: Copy the original source code line by line into C code comments
#: in the generated code file to help with understanding the output.
#: This is also required for coverage analysis.
emit_code_comments = True

# undocumented
pre_import = None

#: Decref global variables in each module on exit for garbage collection.
#: 0: None, 1+: interned objects, 2+: cdef globals, 3+: types objects
#: Mostly for reducing noise in Valgrind as it typically executes at process exit
#: (when all memory will be reclaimed anyways).
#: Note that directly or indirectly executed cleanup code that makes use of global
#: variables or types may no longer be safe when enabling the respective level since
#: there is no guaranteed order in which the (reference counted) objects will
#: be cleaned up.  The order can change due to live references and reference cycles.
generate_cleanup_code = False

#: Should tp_clear() set object fields to None instead of clearing them to NULL?
clear_to_none = True

#: Generate an annotated HTML version of the input source files for debugging and optimisation purposes.
#: This has the same effect as the ``annotate`` argument in :func:`cythonize`.
annotate = False

# When annotating source files in HTML, include coverage information from
# this file.
annotate_coverage_xml = None

#: This will abort the compilation on the first error occurred rather than trying
#: to keep going and printing further error messages.
fast_fail = False

#: Turn all warnings into errors.
warning_errors = False

#: Make unknown names an error.  Python raises a NameError when
#: encountering unknown names at runtime, whereas this option makes
#: them a compile time error.  If you want full Python compatibility,
#: you should disable this option and also 'cache_builtins'.
error_on_unknown_names = True

#: Make uninitialized local variable reference a compile time error.
#: Python raises UnboundLocalError at runtime, whereas this option makes
#: them a compile time error. Note that this option affects only variables
#: of "python object" type.
error_on_uninitialized = True

#: This will convert statements of the form ``for i in range(...)``
#: to ``for i from ...`` when ``i`` is a C integer type, and the direction
#: (i.e. sign of step) can be determined.
#: WARNING: This may change the semantics if the range causes assignment to
#: i to overflow. Specifically, if this option is set, an error will be
#: raised before the loop is entered, whereas without this option the loop
#: will execute until an overflowing value is encountered.
convert_range = True

#: Perform lookups on builtin names only once, at module initialisation
#: time.  This will prevent the module from getting imported if a
#: builtin name that it uses cannot be found during initialisation.
#: Default is True.
#: Note that some legacy builtins are automatically remapped
#: from their Python 2 names to their Python 3 names by Cython
#: when building in Python 3.x,
#: so that they do not get in the way even if this option is enabled.
cache_builtins = True

#: Generate branch prediction hints to speed up error handling etc.
gcc_branch_hints = True

#: Enable this to allow one to write ``your_module.foo = ...`` to overwrite the
#: definition if the cpdef function foo, at the cost of an extra dictionary
#: lookup on every call.
#: If this is false it generates only the Python wrapper and no override check.
lookup_module_cpdef = False

#: Whether or not to embed the Python interpreter, for use in making a
#: standalone executable or calling from external libraries.
#: This will provide a C function which initialises the interpreter and
#: executes the body of this module.
#: See `this demo <https://github.com/cython/cython/tree/master/Demos/embed>`_
#: for a concrete example.
#: If true, the initialisation function is the C main() function, but
#: this option can also be set to a non-empty string to provide a function name explicitly.
#: Default is False.
embed = None

# In previous iterations of Cython, globals() gave the first non-Cython module
# globals in the call stack.  Sage relies on this behavior for variable injection.
old_style_globals = ShouldBeFromDirective('old_style_globals')

#: Allows cimporting from a pyx file without a pxd file.
cimport_from_pyx = False

#: Maximum number of dimensions for buffers -- set lower than number of
#: dimensions in numpy, as
#: slices are passed by value and involve a lot of copying.
buffer_max_dims = 8

#: Number of function closure instances to keep in a freelist (0: no freelists)
closure_freelist_size = 8


def get_directive_defaults():
    # To add an item to this list, all accesses should be changed to use the new
    # directive, and the global option itself should be set to an instance of
    # ShouldBeFromDirective.
    for old_option in ShouldBeFromDirective.known_directives:
        value = globals().get(old_option.options_name)
        assert old_option.directive_name in _directive_defaults
        if not isinstance(value, ShouldBeFromDirective):
            if old_option.disallow:
                raise RuntimeError(
                    "Option '%s' must be set from directive '%s'" % (
                    old_option.option_name, old_option.directive_name))
            else:
                # Warn?
                _directive_defaults[old_option.directive_name] = value
    return _directive_defaults

# Declare compiler directives
_directive_defaults = {
    'boundscheck' : True,
    'nonecheck' : False,
    'initializedcheck' : True,
    'embedsignature' : False,
    'auto_cpdef': False,
    'auto_pickle': None,
    'cdivision': False,  # was True before 0.12
    'cdivision_warnings': False,
    'c_api_binop_methods': True,
    'overflowcheck': False,
    'overflowcheck.fold': True,
    'always_allow_keywords': False,
    'allow_none_for_extension_args': True,
    'wraparound' : True,
    'ccomplex' : False,  # use C99/C++ for complex types and arith
    'callspec' : "",
    'nogil' : False,
    'profile': False,
    'linetrace': False,
    'emit_code_comments': True,  # copy original source code into C code comments
    'annotation_typing': True,  # read type declarations from Python function annotations
    'infer_types': None,
    'infer_types.verbose': False,
    'autotestdict': True,
    'autotestdict.cdef': False,
    'autotestdict.all': False,
    'language_level': None,
    'fast_getattr': False,  # Undocumented until we come up with a better way to handle this everywhere.
    'py2_import': False,  # For backward compatibility of Cython's source code in Py3 source mode
    'preliminary_late_includes_cy28': False,  # Temporary directive in 0.28, to be removed in a later version (see GH#2079).
    'iterable_coroutine': False,  # Make async coroutines backwards compatible with the old asyncio yield-from syntax.
    'c_string_type': 'bytes',
    'c_string_encoding': '',
    'type_version_tag': True,  # enables Py_TPFLAGS_HAVE_VERSION_TAG on extension types
    'unraisable_tracebacks': True,
    'old_style_globals': False,
    'np_pythran': False,
    'fast_gil': False,

    # set __file__ and/or __path__ to known source/target path at import time (instead of not having them available)
    'set_initial_path' : None,  # SOURCEFILE or "/full/path/to/module"

    'warn': None,
    'warn.undeclared': False,
    'warn.unreachable': True,
    'warn.maybe_uninitialized': False,
    'warn.unused': False,
    'warn.unused_arg': False,
    'warn.unused_result': False,
    'warn.multiple_declarators': True,

# optimizations
    'optimize.inline_defnode_calls': True,
    'optimize.unpack_method_calls': True,  # increases code size when True
    'optimize.unpack_method_calls_in_pyinit': False,  # uselessly increases code size when True
    'optimize.use_switch': True,

# remove unreachable code
    'remove_unreachable': True,

# control flow debug directives
    'control_flow.dot_output': "",  # Graphviz output filename
    'control_flow.dot_annotate_defs': False,  # Annotate definitions

# test support
    'test_assert_path_exists' : [],
    'test_fail_if_path_exists' : [],

# experimental, subject to change
    'binding': None,

    'formal_grammar': False,
}

# Extra warning directives
extra_warnings = {
    'warn.maybe_uninitialized': True,
    'warn.unreachable': True,
    'warn.unused': True,
}

def one_of(*args):
    def validate(name, value):
        if value not in args:
            raise ValueError("%s directive must be one of %s, got '%s'" % (
                name, args, value))
        else:
            return value
    return validate


def normalise_encoding_name(option_name, encoding):
    """
    >>> normalise_encoding_name('c_string_encoding', 'ascii')
    'ascii'
    >>> normalise_encoding_name('c_string_encoding', 'AsCIi')
    'ascii'
    >>> normalise_encoding_name('c_string_encoding', 'us-ascii')
    'ascii'
    >>> normalise_encoding_name('c_string_encoding', 'utF8')
    'utf8'
    >>> normalise_encoding_name('c_string_encoding', 'utF-8')
    'utf8'
    >>> normalise_encoding_name('c_string_encoding', 'deFAuLT')
    'default'
    >>> normalise_encoding_name('c_string_encoding', 'default')
    'default'
    >>> normalise_encoding_name('c_string_encoding', 'SeriousLyNoSuch--Encoding')
    'SeriousLyNoSuch--Encoding'
    """
    if not encoding:
        return ''
    if encoding.lower() in ('default', 'ascii', 'utf8'):
        return encoding.lower()
    import codecs
    try:
        decoder = codecs.getdecoder(encoding)
    except LookupError:
        return encoding  # may exists at runtime ...
    for name in ('ascii', 'utf8'):
        if codecs.getdecoder(name) == decoder:
            return name
    return encoding


# Override types possibilities above, if needed
directive_types = {
    'language_level': str,  # values can be None/2/3/'3str', where None == 2+warning
    'auto_pickle': bool,
    'locals': dict,
    'final' : bool,  # final cdef classes and methods
    'nogil' : bool,
    'internal' : bool,  # cdef class visibility in the module dict
    'infer_types' : bool,  # values can be True/None/False
    'binding' : bool,
    'cfunc' : None,  # decorators do not take directive value
    'ccall' : None,
    'inline' : None,
    'staticmethod' : None,
    'cclass' : None,
    'no_gc_clear' : bool,
    'no_gc' : bool,
    'returns' : type,
    'exceptval': type,  # actually (type, check=True/False), but has its own parser
    'set_initial_path': str,
    'freelist': int,
    'c_string_type': one_of('bytes', 'bytearray', 'str', 'unicode'),
    'c_string_encoding': normalise_encoding_name,
}

for key, val in _directive_defaults.items():
    if key not in directive_types:
        directive_types[key] = type(val)

directive_scopes = {  # defaults to available everywhere
    # 'module', 'function', 'class', 'with statement'
    'auto_pickle': ('module', 'cclass'),
    'final' : ('cclass', 'function'),
    'nogil' : ('function', 'with statement'),
    'inline' : ('function',),
    'cfunc' : ('function', 'with statement'),
    'ccall' : ('function', 'with statement'),
    'returns' : ('function',),
    'exceptval' : ('function',),
    'locals' : ('function',),
    'staticmethod' : ('function',),  # FIXME: analysis currently lacks more specific function scope
    'no_gc_clear' : ('cclass',),
    'no_gc' : ('cclass',),
    'internal' : ('cclass',),
    'cclass' : ('class', 'cclass', 'with statement'),
    'autotestdict' : ('module',),
    'autotestdict.all' : ('module',),
    'autotestdict.cdef' : ('module',),
    'set_initial_path' : ('module',),
    'test_assert_path_exists' : ('function', 'class', 'cclass'),
    'test_fail_if_path_exists' : ('function', 'class', 'cclass'),
    'freelist': ('cclass',),
    'emit_code_comments': ('module',),
    'annotation_typing': ('module',),  # FIXME: analysis currently lacks more specific function scope
    # Avoid scope-specific to/from_py_functions for c_string.
    'c_string_type': ('module',),
    'c_string_encoding': ('module',),
    'type_version_tag': ('module', 'cclass'),
    'language_level': ('module',),
    # globals() could conceivably be controlled at a finer granularity,
    # but that would complicate the implementation
    'old_style_globals': ('module',),
    'np_pythran': ('module',),
    'fast_gil': ('module',),
    'iterable_coroutine': ('module', 'function'),
}


def parse_directive_value(name, value, relaxed_bool=False):
    """
    Parses value as an option value for the given name and returns
    the interpreted value. None is returned if the option does not exist.

    >>> print(parse_directive_value('nonexisting', 'asdf asdfd'))
    None
    >>> parse_directive_value('boundscheck', 'True')
    True
    >>> parse_directive_value('boundscheck', 'true')
    Traceback (most recent call last):
       ...
    ValueError: boundscheck directive must be set to True or False, got 'true'

    >>> parse_directive_value('c_string_encoding', 'us-ascii')
    'ascii'
    >>> parse_directive_value('c_string_type', 'str')
    'str'
    >>> parse_directive_value('c_string_type', 'bytes')
    'bytes'
    >>> parse_directive_value('c_string_type', 'bytearray')
    'bytearray'
    >>> parse_directive_value('c_string_type', 'unicode')
    'unicode'
    >>> parse_directive_value('c_string_type', 'unnicode')
    Traceback (most recent call last):
    ValueError: c_string_type directive must be one of ('bytes', 'bytearray', 'str', 'unicode'), got 'unnicode'
    """
    type = directive_types.get(name)
    if not type:
        return None
    orig_value = value
    if type is bool:
        value = str(value)
        if value == 'True':
            return True
        if value == 'False':
            return False
        if relaxed_bool:
            value = value.lower()
            if value in ("true", "yes"):
                return True
            elif value in ("false", "no"):
                return False
        raise ValueError("%s directive must be set to True or False, got '%s'" % (
            name, orig_value))
    elif type is int:
        try:
            return int(value)
        except ValueError:
            raise ValueError("%s directive must be set to an integer, got '%s'" % (
                name, orig_value))
    elif type is str:
        return str(value)
    elif callable(type):
        return type(name, value)
    else:
        assert False


def parse_directive_list(s, relaxed_bool=False, ignore_unknown=False,
                         current_settings=None):
    """
    Parses a comma-separated list of pragma options. Whitespace
    is not considered.

    >>> parse_directive_list('      ')
    {}
    >>> (parse_directive_list('boundscheck=True') ==
    ... {'boundscheck': True})
    True
    >>> parse_directive_list('  asdf')
    Traceback (most recent call last):
       ...
    ValueError: Expected "=" in option "asdf"
    >>> parse_directive_list('boundscheck=hey')
    Traceback (most recent call last):
       ...
    ValueError: boundscheck directive must be set to True or False, got 'hey'
    >>> parse_directive_list('unknown=True')
    Traceback (most recent call last):
       ...
    ValueError: Unknown option: "unknown"
    >>> warnings = parse_directive_list('warn.all=True')
    >>> len(warnings) > 1
    True
    >>> sum(warnings.values()) == len(warnings)  # all true.
    True
    """
    if current_settings is None:
        result = {}
    else:
        result = current_settings
    for item in s.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError('Expected "=" in option "%s"' % item)
        name, value = [s.strip() for s in item.strip().split('=', 1)]
        if name not in _directive_defaults:
            found = False
            if name.endswith('.all'):
                prefix = name[:-3]
                for directive in _directive_defaults:
                    if directive.startswith(prefix):
                        found = True
                        parsed_value = parse_directive_value(directive, value, relaxed_bool=relaxed_bool)
                        result[directive] = parsed_value
            if not found and not ignore_unknown:
                raise ValueError('Unknown option: "%s"' % name)
        else:
            parsed_value = parse_directive_value(name, value, relaxed_bool=relaxed_bool)
            result[name] = parsed_value
    return result


def parse_variable_value(value):
    """
    Parses value as an option value for the given name and returns
    the interpreted value.

    >>> parse_variable_value('True')
    True
    >>> parse_variable_value('true')
    'true'
    >>> parse_variable_value('us-ascii')
    'us-ascii'
    >>> parse_variable_value('str')
    'str'
    >>> parse_variable_value('123')
    123
    >>> parse_variable_value('1.23')
    1.23

    """
    if value == "True":
        return True
    elif value == "False":
        return False
    elif value == "None":
        return None
    elif value.isdigit():
        return int(value)
    else:
        try:
            value = float(value)
        except Exception:
            # Not a float
            pass
        return value


def parse_compile_time_env(s, current_settings=None):
    """
    Parses a comma-separated list of pragma options. Whitespace
    is not considered.

    >>> parse_compile_time_env('      ')
    {}
    >>> (parse_compile_time_env('HAVE_OPENMP=True') ==
    ... {'HAVE_OPENMP': True})
    True
    >>> parse_compile_time_env('  asdf')
    Traceback (most recent call last):
       ...
    ValueError: Expected "=" in option "asdf"
    >>> parse_compile_time_env('NUM_THREADS=4') == {'NUM_THREADS': 4}
    True
    >>> parse_compile_time_env('unknown=anything') == {'unknown': 'anything'}
    True
    """
    if current_settings is None:
        result = {}
    else:
        result = current_settings
    for item in s.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError('Expected "=" in option "%s"' % item)
        name, value = [s.strip() for s in item.split('=', 1)]
        result[name] = parse_variable_value(value)
    return result
