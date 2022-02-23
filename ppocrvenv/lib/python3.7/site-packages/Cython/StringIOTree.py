# cython: auto_pickle=False

r"""
Implements a buffer with insertion points. When you know you need to
"get back" to a place and write more later, simply call insertion_point()
at that spot and get a new StringIOTree object that is "left behind".

EXAMPLE:

>>> a = StringIOTree()
>>> _= a.write('first\n')
>>> b = a.insertion_point()
>>> _= a.write('third\n')
>>> _= b.write('second\n')
>>> a.getvalue().split()
['first', 'second', 'third']

>>> c = b.insertion_point()
>>> d = c.insertion_point()
>>> _= d.write('alpha\n')
>>> _= b.write('gamma\n')
>>> _= c.write('beta\n')
>>> b.getvalue().split()
['second', 'alpha', 'beta', 'gamma']

>>> i = StringIOTree()
>>> d.insert(i)
>>> _= i.write('inserted\n')
>>> out = StringIO()
>>> a.copyto(out)
>>> out.getvalue().split()
['first', 'second', 'alpha', 'inserted', 'beta', 'gamma', 'third']
"""

from __future__ import absolute_import  #, unicode_literals

try:
    # Prefer cStringIO since io.StringIO() does not support writing 'str' in Py2.
    from cStringIO import StringIO
except ImportError:
    from io import StringIO


class StringIOTree(object):
    """
    See module docs.
    """

    def __init__(self, stream=None):
        self.prepended_children = []
        if stream is None:
            stream = StringIO()
        self.stream = stream
        self.write = stream.write
        self.markers = []

    def getvalue(self):
        content = [x.getvalue() for x in self.prepended_children]
        content.append(self.stream.getvalue())
        return "".join(content)

    def copyto(self, target):
        """Potentially cheaper than getvalue as no string concatenation
        needs to happen."""
        for child in self.prepended_children:
            child.copyto(target)
        stream_content = self.stream.getvalue()
        if stream_content:
            target.write(stream_content)

    def commit(self):
        # Save what we have written until now so that the buffer
        # itself is empty -- this makes it ready for insertion
        if self.stream.tell():
            self.prepended_children.append(StringIOTree(self.stream))
            self.prepended_children[-1].markers = self.markers
            self.markers = []
            self.stream = StringIO()
            self.write = self.stream.write

    def insert(self, iotree):
        """
        Insert a StringIOTree (and all of its contents) at this location.
        Further writing to self appears after what is inserted.
        """
        self.commit()
        self.prepended_children.append(iotree)

    def insertion_point(self):
        """
        Returns a new StringIOTree, which is left behind at the current position
        (it what is written to the result will appear right before whatever is
        next written to self).

        Calling getvalue() or copyto() on the result will only return the
        contents written to it.
        """
        # Save what we have written until now
        # This is so that getvalue on the result doesn't include it.
        self.commit()
        # Construct the new forked object to return
        other = StringIOTree()
        self.prepended_children.append(other)
        return other

    def allmarkers(self):
        children = self.prepended_children
        return [m for c in children for m in c.allmarkers()] + self.markers
