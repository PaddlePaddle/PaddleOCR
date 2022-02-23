#
# The MIT License
#
# Copyright 2017 Joao Felipe Pimentel

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""Helpers for APTED"""

# pylint: disable=no-self-use
# pylint: disable=unused-argument
from __future__ import (absolute_import, division)

from collections import deque
from itertools import chain


class ChainedValue(object):
    """Represents a chained value"""
    # pylint: disable=too-few-public-methods

    def __init__(self, value=0, chain_=None):
        self.value = value
        self.chain = chain_ or []

    def __add__(self, other):
        if not self.chain:
            chain_ = other.chain
        elif not other.chain:
            chain_ = self.chain
        else:
            chain_ = chain(self.chain, other.chain)
        return ChainedValue(self.value + other.value, chain_)

    def __sub__(self, other):
        chain_ = [("R", other.chain)]
        if self.chain:
            chain_ = chain(self.chain, chain_)
        return ChainedValue(
            self.value - other.value, chain(self.chain, chain_)
        )

    def __neg__(self):
        return ChainedValue(
            -self.value, [("R", self.chain)]
        )

    def __bool__(self):
        return bool(self.value)

    def __nonzero__(self):
        return bool(self.value)

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)

    def __hash__(self):
        return hash(self.value)

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value

    def __int__(self):
        return self.value


class Tree(object):
    """Represents a Tree Node"""

    def __init__(self, name, *children):
        self.name = name
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        result = str(self.name)
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)

    def __repr__(self):
        return self.bracket()

    @classmethod
    def from_text(cls, text):
        """Create tree from bracket notation

        Bracket notation encodes the trees with nested parentheses, for example,
        in tree {A{B{X}{Y}{F}}{C}} the root node has label A and two children
        with labels B and C. Node with label B has three children with labels
        X, Y, F.
        """
        tree_stack = []
        stack = []
        for letter in text:
            if letter == "{":
                stack.append("")
            elif letter == "}":
                text = stack.pop()
                children = deque()
                while tree_stack and tree_stack[-1][1] > len(stack):
                    child, _ = tree_stack.pop()
                    children.appendleft(child)

                tree_stack.append((cls(text, *children), len(stack)))
            else:
                stack[-1] += letter
        return tree_stack[0][0]
