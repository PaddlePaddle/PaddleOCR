"""
future: Easy, safe support for Python 2/3 compatibility
=======================================================

``future`` is the missing compatibility layer between Python 2 and Python
3. It allows you to use a single, clean Python 3.x-compatible codebase to
support both Python 2 and Python 3 with minimal overhead.

It is designed to be used as follows::

    from __future__ import (absolute_import, division,
                            print_function, unicode_literals)
    from builtins import (
             bytes, dict, int, list, object, range, str,
             ascii, chr, hex, input, next, oct, open,
             pow, round, super,
             filter, map, zip)

followed by predominantly standard, idiomatic Python 3 code that then runs
similarly on Python 2.6/2.7 and Python 3.3+.

The imports have no effect on Python 3. On Python 2, they shadow the
corresponding builtins, which normally have different semantics on Python 3
versus 2, to provide their Python 3 semantics.


Standard library reorganization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``future`` supports the standard library reorganization (PEP 3108) through the
following Py3 interfaces:

    >>> # Top-level packages with Py3 names provided on Py2:
    >>> import html.parser
    >>> import queue
    >>> import tkinter.dialog
    >>> import xmlrpc.client
    >>> # etc.

    >>> # Aliases provided for extensions to existing Py2 module names:
    >>> from future.standard_library import install_aliases
    >>> install_aliases()

    >>> from collections import Counter, OrderedDict   # backported to Py2.6
    >>> from collections import UserDict, UserList, UserString
    >>> import urllib.request
    >>> from itertools import filterfalse, zip_longest
    >>> from subprocess import getoutput, getstatusoutput


Automatic conversion
--------------------

An included script called `futurize
<http://python-future.org/automatic_conversion.html>`_ aids in converting
code (from either Python 2 or Python 3) to code compatible with both
platforms. It is similar to ``python-modernize`` but goes further in
providing Python 3 compatibility through the use of the backported types
and builtin functions in ``future``.


Documentation
-------------

See: http://python-future.org


Credits
-------

:Author:  Ed Schofield, Jordan M. Adler, et al
:Sponsor: Python Charmers Pty Ltd, Australia, and Python Charmers Pte
          Ltd, Singapore. http://pythoncharmers.com
:Others:  See docs/credits.rst or http://python-future.org/credits.html


Licensing
---------
Copyright 2013-2019 Python Charmers Pty Ltd, Australia.
The software is distributed under an MIT licence. See LICENSE.txt.

"""

__title__ = 'future'
__author__ = 'Ed Schofield'
__license__ = 'MIT'
__copyright__ = 'Copyright 2013-2019 Python Charmers Pty Ltd'
__ver_major__ = 0
__ver_minor__ = 18
__ver_patch__ = 2
__ver_sub__ = ''
__version__ = "%d.%d.%d%s" % (__ver_major__, __ver_minor__,
                              __ver_patch__, __ver_sub__)
