# coding=utf-8
"""
past: compatibility with Python 2 from Python 3
===============================================

``past`` is a package to aid with Python 2/3 compatibility. Whereas ``future``
contains backports of Python 3 constructs to Python 2, ``past`` provides
implementations of some Python 2 constructs in Python 3 and tools to import and
run Python 2 code in Python 3. It is intended to be used sparingly, as a way of
running old Python 2 code from Python 3 until the code is ported properly.

Potential uses for libraries:

- as a step in porting a Python 2 codebase to Python 3 (e.g. with the ``futurize`` script)
- to provide Python 3 support for previously Python 2-only libraries with the
  same APIs as on Python 2 -- particularly with regard to 8-bit strings (the
  ``past.builtins.str`` type).
- to aid in providing minimal-effort Python 3 support for applications using
  libraries that do not yet wish to upgrade their code properly to Python 3, or
  wish to upgrade it gradually to Python 3 style.


Here are some code examples that run identically on Python 3 and 2::

    >>> from past.builtins import str as oldstr

    >>> philosopher = oldstr(u'\u5b54\u5b50'.encode('utf-8'))
    >>> # This now behaves like a Py2 byte-string on both Py2 and Py3.
    >>> # For example, indexing returns a Python 2-like string object, not
    >>> # an integer:
    >>> philosopher[0]
    '\xe5'
    >>> type(philosopher[0])
    <past.builtins.oldstr>

    >>> # List-producing versions of range, reduce, map, filter
    >>> from past.builtins import range, reduce
    >>> range(10)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> reduce(lambda x, y: x+y, [1, 2, 3, 4, 5])
    15

    >>> # Other functions removed in Python 3 are resurrected ...
    >>> from past.builtins import execfile
    >>> execfile('myfile.py')

    >>> from past.builtins import raw_input
    >>> name = raw_input('What is your name? ')
    What is your name? [cursor]

    >>> from past.builtins import reload
    >>> reload(mymodule)   # equivalent to imp.reload(mymodule) in Python 3

    >>> from past.builtins import xrange
    >>> for i in xrange(10):
    ...     pass


It also provides import hooks so you can import and use Python 2 modules like
this::

    $ python3

    >>> from past.translation import autotranslate
    >>> authotranslate('mypy2module')
    >>> import mypy2module

until the authors of the Python 2 modules have upgraded their code. Then, for
example::

    >>> mypy2module.func_taking_py2_string(oldstr(b'abcd'))


Credits
-------

:Author:  Ed Schofield, Jordan M. Adler, et al
:Sponsor: Python Charmers Pty Ltd, Australia: http://pythoncharmers.com


Licensing
---------
Copyright 2013-2019 Python Charmers Pty Ltd, Australia.
The software is distributed under an MIT licence. See LICENSE.txt.
"""

from future import __version__, __copyright__, __license__

__title__ = 'past'
__author__ = 'Ed Schofield'
