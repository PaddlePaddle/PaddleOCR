from __future__ import absolute_import, unicode_literals

from abc import ABCMeta

from six import add_metaclass

from virtualenv.create.creator import Creator
from virtualenv.create.describe import Describe


@add_metaclass(ABCMeta)
class VirtualenvBuiltin(Creator, Describe):
    """A creator that does operations itself without delegation, if we can create it we can also describe it"""

    def __init__(self, options, interpreter):
        Creator.__init__(self, options, interpreter)
        Describe.__init__(self, self.dest, interpreter)
