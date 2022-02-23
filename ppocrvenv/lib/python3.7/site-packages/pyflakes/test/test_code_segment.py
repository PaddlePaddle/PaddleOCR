from sys import version_info

from pyflakes import messages as m
from pyflakes.checker import (FunctionScope, ClassScope, ModuleScope,
                              Argument, FunctionDefinition, Assignment)
from pyflakes.test.harness import TestCase, skipIf


class TestCodeSegments(TestCase):
    """
    Tests for segments of a module
    """

    def test_function_segment(self):
        self.flakes('''
        def foo():
            def bar():
                pass
        ''', is_segment=True)

        self.flakes('''
        def foo():
            def bar():
                x = 0
        ''', m.UnusedVariable, is_segment=True)

    def test_class_segment(self):
        self.flakes('''
        class Foo:
            class Bar:
                pass
        ''', is_segment=True)

        self.flakes('''
        class Foo:
            def bar():
                x = 0
        ''', m.UnusedVariable, is_segment=True)

    def test_scope_class(self):
        checker = self.flakes('''
        class Foo:
            x = 0
            def bar(a, b=1, *d, **e):
                pass
        ''', is_segment=True)

        scopes = checker.deadScopes
        module_scopes = [
            scope for scope in scopes if scope.__class__ is ModuleScope]
        class_scopes = [
            scope for scope in scopes if scope.__class__ is ClassScope]
        function_scopes = [
            scope for scope in scopes if scope.__class__ is FunctionScope]

        # Ensure module scope is not present because we are analysing
        # the inner contents of Foo
        self.assertEqual(len(module_scopes), 0)
        self.assertEqual(len(class_scopes), 1)
        self.assertEqual(len(function_scopes), 1)

        class_scope = class_scopes[0]
        function_scope = function_scopes[0]

        self.assertIsInstance(class_scope, ClassScope)
        self.assertIsInstance(function_scope, FunctionScope)

        self.assertIn('x', class_scope)
        self.assertIn('bar', class_scope)

        self.assertIn('a', function_scope)
        self.assertIn('b', function_scope)
        self.assertIn('d', function_scope)
        self.assertIn('e', function_scope)

        self.assertIsInstance(class_scope['bar'], FunctionDefinition)
        self.assertIsInstance(class_scope['x'], Assignment)

        self.assertIsInstance(function_scope['a'], Argument)
        self.assertIsInstance(function_scope['b'], Argument)
        self.assertIsInstance(function_scope['d'], Argument)
        self.assertIsInstance(function_scope['e'], Argument)

    def test_scope_function(self):
        checker = self.flakes('''
        def foo(a, b=1, *d, **e):
            def bar(f, g=1, *h, **i):
                pass
        ''', is_segment=True)

        scopes = checker.deadScopes
        module_scopes = [
            scope for scope in scopes if scope.__class__ is ModuleScope]
        function_scopes = [
            scope for scope in scopes if scope.__class__ is FunctionScope]

        # Ensure module scope is not present because we are analysing
        # the inner contents of foo
        self.assertEqual(len(module_scopes), 0)
        self.assertEqual(len(function_scopes), 2)

        function_scope_foo = function_scopes[1]
        function_scope_bar = function_scopes[0]

        self.assertIsInstance(function_scope_foo, FunctionScope)
        self.assertIsInstance(function_scope_bar, FunctionScope)

        self.assertIn('a', function_scope_foo)
        self.assertIn('b', function_scope_foo)
        self.assertIn('d', function_scope_foo)
        self.assertIn('e', function_scope_foo)
        self.assertIn('bar', function_scope_foo)

        self.assertIn('f', function_scope_bar)
        self.assertIn('g', function_scope_bar)
        self.assertIn('h', function_scope_bar)
        self.assertIn('i', function_scope_bar)

        self.assertIsInstance(function_scope_foo['bar'], FunctionDefinition)
        self.assertIsInstance(function_scope_foo['a'], Argument)
        self.assertIsInstance(function_scope_foo['b'], Argument)
        self.assertIsInstance(function_scope_foo['d'], Argument)
        self.assertIsInstance(function_scope_foo['e'], Argument)

        self.assertIsInstance(function_scope_bar['f'], Argument)
        self.assertIsInstance(function_scope_bar['g'], Argument)
        self.assertIsInstance(function_scope_bar['h'], Argument)
        self.assertIsInstance(function_scope_bar['i'], Argument)

    @skipIf(version_info < (3, 5), 'new in Python 3.5')
    def test_scope_async_function(self):
        self.flakes('async def foo(): pass', is_segment=True)
