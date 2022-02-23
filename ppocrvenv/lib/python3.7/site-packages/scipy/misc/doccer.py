''' Utilities to allow inserting docstring fragments for common
parameters into function and method docstrings'''

import numpy as np
from .._lib import doccer as _ld

__all__ = ['docformat', 'inherit_docstring_from', 'indentcount_lines',
           'filldoc', 'unindent_dict', 'unindent_string']

@np.deprecate(message="scipy.misc.docformat is deprecated in Scipy 1.3.0")
def docformat(docstring, docdict=None):
    return _ld.docformat(docstring, docdict)


@np.deprecate(message="scipy.misc.inherit_docstring_from is deprecated "
                      "in SciPy 1.3.0")
def inherit_docstring_from(cls):
    return _ld.inherit_docstring_from(cls)


@np.deprecate(message="scipy.misc.extend_notes_in_docstring is deprecated "
                      "in SciPy 1.3.0")
def extend_notes_in_docstring(cls, notes):
    return _ld.extend_notes_in_docstring(cls, notes)


@np.deprecate(message="scipy.misc.replace_notes_in_docstring is deprecated "
                      "in SciPy 1.3.0")
def replace_notes_in_docstring(cls, notes):
    return _ld.replace_notes_in_docstring(cls, notes)


@np.deprecate(message="scipy.misc.indentcount_lines is deprecated "
                      "in SciPy 1.3.0")
def indentcount_lines(lines):
    return _ld.indentcount_lines(lines)


@np.deprecate(message="scipy.misc.filldoc is deprecated in SciPy 1.3.0")
def filldoc(docdict, unindent_params=True):
    return _ld.filldoc(docdict, unindent_params)


@np.deprecate(message="scipy.misc.unindent_dict is deprecated in SciPy 1.3.0")
def unindent_dict(docdict):
    return _ld.unindent_dict(docdict)


@np.deprecate(message="scipy.misc.unindent_string is deprecated in SciPy 1.3.0")
def unindent_string(docstring):
    return _ld.unindent_string(docstring)
