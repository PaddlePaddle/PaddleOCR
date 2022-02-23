"""
This disables builtin functions (and one exception class) which are
removed from Python 3.3.

This module is designed to be used like this::

    from future.builtins.disabled import *

This disables the following obsolete Py2 builtin functions::

    apply, cmp, coerce, execfile, file, input, long,
    raw_input, reduce, reload, unicode, xrange

We don't hack __builtin__, which is very fragile because it contaminates
imported modules too. Instead, we just create new functions with
the same names as the obsolete builtins from Python 2 which raise
NameError exceptions when called.

Note that both ``input()`` and ``raw_input()`` are among the disabled
functions (in this module). Although ``input()`` exists as a builtin in
Python 3, the Python 2 ``input()`` builtin is unsafe to use because it
can lead to shell injection. Therefore we shadow it by default upon ``from
future.builtins.disabled import *``, in case someone forgets to import our
replacement ``input()`` somehow and expects Python 3 semantics.

See the ``future.builtins.misc`` module for a working version of
``input`` with Python 3 semantics.

(Note that callable() is not among the functions disabled; this was
reintroduced into Python 3.2.)

This exception class is also disabled:

    StandardError

"""

from __future__ import division, absolute_import, print_function

from future import utils


OBSOLETE_BUILTINS = ['apply', 'chr', 'cmp', 'coerce', 'execfile', 'file',
                     'input', 'long', 'raw_input', 'reduce', 'reload',
                     'unicode', 'xrange', 'StandardError']


def disabled_function(name):
    '''
    Returns a function that cannot be called
    '''
    def disabled(*args, **kwargs):
        '''
        A function disabled by the ``future`` module. This function is
        no longer a builtin in Python 3.
        '''
        raise NameError('obsolete Python 2 builtin {0} is disabled'.format(name))
    return disabled


if not utils.PY3:
    for fname in OBSOLETE_BUILTINS:
        locals()[fname] = disabled_function(fname)
    __all__ = OBSOLETE_BUILTINS
else:
    __all__ = []
