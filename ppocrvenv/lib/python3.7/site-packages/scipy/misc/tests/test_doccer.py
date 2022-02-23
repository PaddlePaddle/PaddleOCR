''' Some tests for the documenting decorator and support functions '''

import sys
import pytest
from numpy.testing import assert_equal, suppress_warnings

from scipy.misc import doccer

# python -OO strips docstrings
DOCSTRINGS_STRIPPED = sys.flags.optimize > 1

docstring = \
"""Docstring
    %(strtest1)s
        %(strtest2)s
     %(strtest3)s
"""
param_doc1 = \
"""Another test
   with some indent"""

param_doc2 = \
"""Another test, one line"""

param_doc3 = \
"""    Another test
       with some indent"""

doc_dict = {'strtest1':param_doc1,
            'strtest2':param_doc2,
            'strtest3':param_doc3}

filled_docstring = \
"""Docstring
    Another test
       with some indent
        Another test, one line
     Another test
       with some indent
"""


def test_unindent():
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        assert_equal(doccer.unindent_string(param_doc1), param_doc1)
        assert_equal(doccer.unindent_string(param_doc2), param_doc2)
        assert_equal(doccer.unindent_string(param_doc3), param_doc1)


def test_unindent_dict():
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        d2 = doccer.unindent_dict(doc_dict)
    assert_equal(d2['strtest1'], doc_dict['strtest1'])
    assert_equal(d2['strtest2'], doc_dict['strtest2'])
    assert_equal(d2['strtest3'], doc_dict['strtest1'])


def test_docformat():
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        udd = doccer.unindent_dict(doc_dict)
        formatted = doccer.docformat(docstring, udd)
        assert_equal(formatted, filled_docstring)
        single_doc = 'Single line doc %(strtest1)s'
        formatted = doccer.docformat(single_doc, doc_dict)
        # Note - initial indent of format string does not
        # affect subsequent indent of inserted parameter
        assert_equal(formatted, """Single line doc Another test
   with some indent""")


@pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason="docstrings stripped")
def test_decorator():
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        # with unindentation of parameters
        decorator = doccer.filldoc(doc_dict, True)

        @decorator
        def func():
            """ Docstring
            %(strtest3)s
            """
        assert_equal(func.__doc__, """ Docstring
            Another test
               with some indent
            """)

        # without unindentation of parameters
        decorator = doccer.filldoc(doc_dict, False)

        @decorator
        def func():
            """ Docstring
            %(strtest3)s
            """
        assert_equal(func.__doc__, """ Docstring
                Another test
                   with some indent
            """)


@pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason="docstrings stripped")
def test_inherit_docstring_from():

    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)

        class Foo:
            def func(self):
                '''Do something useful.'''
                return

            def func2(self):
                '''Something else.'''

        class Bar(Foo):
            @doccer.inherit_docstring_from(Foo)
            def func(self):
                '''%(super)sABC'''
                return

            @doccer.inherit_docstring_from(Foo)
            def func2(self):
                # No docstring.
                return

    assert_equal(Bar.func.__doc__, Foo.func.__doc__ + 'ABC')
    assert_equal(Bar.func2.__doc__, Foo.func2.__doc__)
    bar = Bar()
    assert_equal(bar.func.__doc__, Foo.func.__doc__ + 'ABC')
    assert_equal(bar.func2.__doc__, Foo.func2.__doc__)
