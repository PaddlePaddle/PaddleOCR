# -*- coding: utf-8 -*-
# tag: jedi

from __future__ import absolute_import

import sys
import os.path

from textwrap import dedent
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

from Cython.Compiler.ParseTreeTransforms import NormalizeTree, InterpretCompilerDirectives
from Cython.Compiler import Main, Symtab, Visitor
from Cython.TestUtils import TransformTest

TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Tools'))


@contextmanager
def _tempfile(code):
    code = dedent(code)
    if not isinstance(code, bytes):
        code = code.encode('utf8')

    with NamedTemporaryFile(suffix='.py') as f:
        f.write(code)
        f.seek(0)
        yield f


def _test_typing(code, inject=False):
    sys.path.insert(0, TOOLS_DIR)
    try:
        import jedityper
    finally:
        sys.path.remove(TOOLS_DIR)
    lines = []
    with _tempfile(code) as f:
        types = jedityper.analyse(f.name)
        if inject:
            lines = jedityper.inject_types(f.name, types)
    return types, lines


class DeclarationsFinder(Visitor.VisitorTransform):
    directives = None

    visit_Node = Visitor.VisitorTransform.recurse_to_children

    def visit_CompilerDirectivesNode(self, node):
        if not self.directives:
            self.directives = []
        self.directives.append(node)
        self.visitchildren(node)
        return node


class TestJediTyper(TransformTest):
    def _test(self, code):
        return _test_typing(code)[0]

    def test_typing_global_int_loop(self):
        code = '''\
        for i in range(10):
            a = i + 1
        '''
        types = self._test(code)
        self.assertIn((None, (1, 0)), types)
        variables = types.pop((None, (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['int']), 'i': set(['int'])}, variables)

    def test_typing_function_int_loop(self):
        code = '''\
        def func(x):
            for i in range(x):
                a = i + 1
            return a
        '''
        types = self._test(code)
        self.assertIn(('func', (1, 0)), types)
        variables = types.pop(('func', (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['int']), 'i': set(['int'])}, variables)

    def test_conflicting_types_in_function(self):
        code = '''\
        def func(a, b):
            print(a)
            a = 1
            b += a
            a = 'abc'
            return a, str(b)

        print(func(1.5, 2))
        '''
        types = self._test(code)
        self.assertIn(('func', (1, 0)), types)
        variables = types.pop(('func', (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['float', 'int', 'str']), 'b': set(['int'])}, variables)

    def _test_typing_function_char_loop(self):
        code = '''\
        def func(x):
            l = []
            for c in x:
                l.append(c)
            return l

        print(func('abcdefg'))
        '''
        types = self._test(code)
        self.assertIn(('func', (1, 0)), types)
        variables = types.pop(('func', (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['int']), 'i': set(['int'])}, variables)

    def test_typing_global_list(self):
        code = '''\
        a = [x for x in range(10)]
        b = list(range(10))
        c = a + b
        d = [0]*10
        '''
        types = self._test(code)
        self.assertIn((None, (1, 0)), types)
        variables = types.pop((None, (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['list']), 'b': set(['list']), 'c': set(['list']), 'd': set(['list'])}, variables)

    def test_typing_function_list(self):
        code = '''\
        def func(x):
            a = [[], []]
            b = [0]* 10 + a
            c = a[0]

        print(func([0]*100))
        '''
        types = self._test(code)
        self.assertIn(('func', (1, 0)), types)
        variables = types.pop(('func', (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['list']), 'b': set(['list']), 'c': set(['list']), 'x': set(['list'])}, variables)

    def test_typing_global_dict(self):
        code = '''\
        a = dict()
        b = {i: i**2 for i in range(10)}
        c = a
        '''
        types = self._test(code)
        self.assertIn((None, (1, 0)), types)
        variables = types.pop((None, (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['dict']), 'b': set(['dict']), 'c': set(['dict'])}, variables)

    def test_typing_function_dict(self):
        code = '''\
        def func(x):
            a = dict()
            b = {i: i**2 for i in range(10)}
            c = x

        print(func({1:2, 'x':7}))
        '''
        types = self._test(code)
        self.assertIn(('func', (1, 0)), types)
        variables = types.pop(('func', (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['dict']), 'b': set(['dict']), 'c': set(['dict']), 'x': set(['dict'])}, variables)


    def test_typing_global_set(self):
        code = '''\
        a = set()
        # b = {i for i in range(10)} # jedi does not support set comprehension yet
        c = a
        d = {1,2,3}
        e = a | b
        '''
        types = self._test(code)
        self.assertIn((None, (1, 0)), types)
        variables = types.pop((None, (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['set']), 'c': set(['set']), 'd': set(['set']), 'e': set(['set'])}, variables)

    def test_typing_function_set(self):
        code = '''\
        def func(x):
            a = set()
            # b = {i for i in range(10)} # jedi does not support set comprehension yet
            c = a
            d = a | b

        print(func({1,2,3}))
        '''
        types = self._test(code)
        self.assertIn(('func', (1, 0)), types)
        variables = types.pop(('func', (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['set']), 'c': set(['set']), 'd': set(['set']), 'x': set(['set'])}, variables)


class TestTypeInjection(TestJediTyper):
    """
    Subtype of TestJediTyper that additionally tests type injection and compilation.
    """
    def setUp(self):
        super(TestTypeInjection, self).setUp()
        compilation_options = Main.CompilationOptions(Main.default_options)
        ctx = compilation_options.create_context()
        transform = InterpretCompilerDirectives(ctx, ctx.compiler_directives)
        transform.module_scope = Symtab.ModuleScope('__main__', None, ctx)
        self.declarations_finder = DeclarationsFinder()
        self.pipeline = [NormalizeTree(None), transform, self.declarations_finder]

    def _test(self, code):
        types, lines = _test_typing(code, inject=True)
        tree = self.run_pipeline(self.pipeline, ''.join(lines))
        directives = self.declarations_finder.directives
        # TODO: validate directives
        return types
