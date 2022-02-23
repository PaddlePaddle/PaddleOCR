from __future__ import absolute_import
# Copyright (c) 2010-2015 openpyxl

"""Implements the lxml.etree.xmlfile API using the standard library xml.etree"""


from contextlib import contextmanager

from xml.etree.ElementTree import Element, tostring


class LxmlSyntaxError(Exception):
    pass


class _FakeIncrementalFileWriter(object):
    """Replacement for _IncrementalFileWriter of lxml.
       Uses ElementTree to build xml in memory."""
    def __init__(self, output_file):
        self._element_stack = []
        self._top_element = None
        self._file = output_file
        self._have_root = False

    @contextmanager
    def element(self, tag, attrib=None, nsmap=None, **_extra):
        """Create a new xml element using a context manager.
        The elements are written when the top level context is left.

        This is for code compatibility only as it is quite slow.
        """

        # __enter__ part
        self._have_root = True
        if attrib is None:
            attrib = {}
        self._top_element = Element(tag, attrib=attrib, **_extra)
        self._top_element.text = ''
        self._top_element.tail = ''
        self._element_stack.append(self._top_element)
        yield

        # __exit__ part
        el = self._element_stack.pop()
        if self._element_stack:
            parent = self._element_stack[-1]
            parent.append(self._top_element)
            self._top_element = parent
        else:
            self._write_element(el)
            self._top_element = None

    def write(self, arg):
        """Write a string or subelement."""

        if isinstance(arg, str):
            # it is not allowed to write a string outside of an element
            if self._top_element is None:
                raise LxmlSyntaxError()

            if len(self._top_element) == 0:
                # element has no children: add string to text
                self._top_element.text += arg
            else:
                # element has children: add string to tail of last child
                self._top_element[-1].tail += arg

        else:
            if self._top_element is not None:
                self._top_element.append(arg)
            elif not self._have_root:
                self._write_element(arg)
            else:
                raise LxmlSyntaxError()

    def _write_element(self, element):
        xml = tostring(element)
        self._file.write(xml)

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        # without root the xml document is incomplete
        if not self._have_root:
            raise LxmlSyntaxError()


class xmlfile(object):
    """Context manager that can replace lxml.etree.xmlfile."""
    def __init__(self, output_file, buffered=False, encoding=None, close=False):
        if isinstance(output_file, str):
            self._file = open(output_file, 'wb')
            self._close = True
        else:
            self._file = output_file
            self._close = close

    def __enter__(self):
        return _FakeIncrementalFileWriter(self._file)

    def __exit__(self, type, value, traceback):
        if self._close == True:
            self._file.close()
