import ast
from sys import version_info

from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skipIf, skip


class Test(TestCase):
    def test_undefined(self):
        self.flakes('bar', m.UndefinedName)

    def test_definedInListComp(self):
        self.flakes('[a for a in range(10) if a]')

    @skipIf(version_info < (3,),
            'in Python 2 list comprehensions execute in the same scope')
    def test_undefinedInListComp(self):
        self.flakes('''
        [a for a in range(10)]
        a
        ''',
                    m.UndefinedName)

    @skipIf(version_info < (3,),
            'in Python 2 exception names stay bound after the except: block')
    def test_undefinedExceptionName(self):
        """Exception names can't be used after the except: block.

        The exc variable is unused inside the exception handler."""
        self.flakes('''
        try:
            raise ValueError('ve')
        except ValueError as exc:
            pass
        exc
        ''', m.UndefinedName, m.UnusedVariable)

    def test_namesDeclaredInExceptBlocks(self):
        """Locals declared in except: blocks can be used after the block.

        This shows the example in test_undefinedExceptionName is
        different."""
        self.flakes('''
        try:
            raise ValueError('ve')
        except ValueError as exc:
            e = exc
        e
        ''')

    @skip('error reporting disabled due to false positives below')
    def test_undefinedExceptionNameObscuringLocalVariable(self):
        """Exception names obscure locals, can't be used after.

        Last line will raise UnboundLocalError on Python 3 after exiting
        the except: block. Note next two examples for false positives to
        watch out for."""
        self.flakes('''
        exc = 'Original value'
        try:
            raise ValueError('ve')
        except ValueError as exc:
            pass
        exc
        ''',
                    m.UndefinedName)

    @skipIf(version_info < (3,),
            'in Python 2 exception names stay bound after the except: block')
    def test_undefinedExceptionNameObscuringLocalVariable2(self):
        """Exception names are unbound after the `except:` block.

        Last line will raise UnboundLocalError on Python 3 but would print out
        've' on Python 2. The exc variable is unused inside the exception
        handler."""
        self.flakes('''
        try:
            raise ValueError('ve')
        except ValueError as exc:
            pass
        print(exc)
        exc = 'Original value'
        ''', m.UndefinedName, m.UnusedVariable)

    def test_undefinedExceptionNameObscuringLocalVariableFalsePositive1(self):
        """Exception names obscure locals, can't be used after. Unless.

        Last line will never raise UnboundLocalError because it's only
        entered if no exception was raised."""
        # The exc variable is unused inside the exception handler.
        expected = [] if version_info < (3,) else [m.UnusedVariable]
        self.flakes('''
        exc = 'Original value'
        try:
            raise ValueError('ve')
        except ValueError as exc:
            print('exception logged')
            raise
        exc
        ''', *expected)

    def test_delExceptionInExcept(self):
        """The exception name can be deleted in the except: block."""
        self.flakes('''
        try:
            pass
        except Exception as exc:
            del exc
        ''')

    def test_undefinedExceptionNameObscuringLocalVariableFalsePositive2(self):
        """Exception names obscure locals, can't be used after. Unless.

        Last line will never raise UnboundLocalError because `error` is
        only falsy if the `except:` block has not been entered."""
        # The exc variable is unused inside the exception handler.
        expected = [] if version_info < (3,) else [m.UnusedVariable]
        self.flakes('''
        exc = 'Original value'
        error = None
        try:
            raise ValueError('ve')
        except ValueError as exc:
            error = 'exception logged'
        if error:
            print(error)
        else:
            exc
        ''', *expected)

    @skip('error reporting disabled due to false positives below')
    def test_undefinedExceptionNameObscuringGlobalVariable(self):
        """Exception names obscure globals, can't be used after.

        Last line will raise UnboundLocalError on both Python 2 and
        Python 3 because the existence of that exception name creates
        a local scope placeholder for it, obscuring any globals, etc."""
        self.flakes('''
        exc = 'Original value'
        def func():
            try:
                pass  # nothing is raised
            except ValueError as exc:
                pass  # block never entered, exc stays unbound
            exc
        ''',
                    m.UndefinedLocal)

    @skip('error reporting disabled due to false positives below')
    def test_undefinedExceptionNameObscuringGlobalVariable2(self):
        """Exception names obscure globals, can't be used after.

        Last line will raise NameError on Python 3 because the name is
        locally unbound after the `except:` block, even if it's
        nonlocal. We should issue an error in this case because code
        only working correctly if an exception isn't raised, is invalid.
        Unless it's explicitly silenced, see false positives below."""
        self.flakes('''
        exc = 'Original value'
        def func():
            global exc
            try:
                raise ValueError('ve')
            except ValueError as exc:
                pass  # block never entered, exc stays unbound
            exc
        ''',
                    m.UndefinedLocal)

    def test_undefinedExceptionNameObscuringGlobalVariableFalsePositive1(self):
        """Exception names obscure globals, can't be used after. Unless.

        Last line will never raise NameError because it's only entered
        if no exception was raised."""
        # The exc variable is unused inside the exception handler.
        expected = [] if version_info < (3,) else [m.UnusedVariable]
        self.flakes('''
        exc = 'Original value'
        def func():
            global exc
            try:
                raise ValueError('ve')
            except ValueError as exc:
                print('exception logged')
                raise
            exc
        ''', *expected)

    def test_undefinedExceptionNameObscuringGlobalVariableFalsePositive2(self):
        """Exception names obscure globals, can't be used after. Unless.

        Last line will never raise NameError because `error` is only
        falsy if the `except:` block has not been entered."""
        # The exc variable is unused inside the exception handler.
        expected = [] if version_info < (3,) else [m.UnusedVariable]
        self.flakes('''
        exc = 'Original value'
        def func():
            global exc
            error = None
            try:
                raise ValueError('ve')
            except ValueError as exc:
                error = 'exception logged'
            if error:
                print(error)
            else:
                exc
        ''', *expected)

    def test_functionsNeedGlobalScope(self):
        self.flakes('''
        class a:
            def b():
                fu
        fu = 1
        ''')

    def test_builtins(self):
        self.flakes('range(10)')

    def test_builtinWindowsError(self):
        """
        C{WindowsError} is sometimes a builtin name, so no warning is emitted
        for using it.
        """
        self.flakes('WindowsError')

    @skipIf(version_info < (3, 6), 'new feature in 3.6')
    def test_moduleAnnotations(self):
        """
        Use of the C{__annotations__} in module scope should not emit
        an undefined name warning when version is greater than or equal to 3.6.
        """
        self.flakes('__annotations__')

    def test_magicGlobalsFile(self):
        """
        Use of the C{__file__} magic global should not emit an undefined name
        warning.
        """
        self.flakes('__file__')

    def test_magicGlobalsBuiltins(self):
        """
        Use of the C{__builtins__} magic global should not emit an undefined
        name warning.
        """
        self.flakes('__builtins__')

    def test_magicGlobalsName(self):
        """
        Use of the C{__name__} magic global should not emit an undefined name
        warning.
        """
        self.flakes('__name__')

    def test_magicGlobalsPath(self):
        """
        Use of the C{__path__} magic global should not emit an undefined name
        warning, if you refer to it from a file called __init__.py.
        """
        self.flakes('__path__', m.UndefinedName)
        self.flakes('__path__', filename='package/__init__.py')

    def test_magicModuleInClassScope(self):
        """
        Use of the C{__module__} magic builtin should not emit an undefined
        name warning if used in class scope.
        """
        self.flakes('__module__', m.UndefinedName)
        self.flakes('''
        class Foo:
            __module__
        ''')
        self.flakes('''
        class Foo:
            def bar(self):
                __module__
        ''', m.UndefinedName)

    @skipIf(version_info < (3, 3), "Python >= 3.3 only")
    def test_magicQualnameInClassScope(self):
        """
        Use of the C{__qualname__} magic builtin should not emit an undefined
        name warning if used in class scope.
        """
        self.flakes('__qualname__', m.UndefinedName)
        self.flakes('''
        class Foo:
            __qualname__
        ''')
        self.flakes('''
        class Foo:
            def bar(self):
                __qualname__
        ''', m.UndefinedName)

    def test_globalImportStar(self):
        """Can't find undefined names with import *."""
        self.flakes('from fu import *; bar',
                    m.ImportStarUsed, m.ImportStarUsage)

    @skipIf(version_info >= (3,), 'obsolete syntax')
    def test_localImportStar(self):
        """
        A local import * still allows undefined names to be found
        in upper scopes.
        """
        self.flakes('''
        def a():
            from fu import *
        bar
        ''', m.ImportStarUsed, m.UndefinedName, m.UnusedImport)

    @skipIf(version_info >= (3,), 'obsolete syntax')
    def test_unpackedParameter(self):
        """Unpacked function parameters create bindings."""
        self.flakes('''
        def a((bar, baz)):
            bar; baz
        ''')

    def test_definedByGlobal(self):
        """
        "global" can make an otherwise undefined name in another function
        defined.
        """
        self.flakes('''
        def a(): global fu; fu = 1
        def b(): fu
        ''')
        self.flakes('''
        def c(): bar
        def b(): global bar; bar = 1
        ''')

    def test_definedByGlobalMultipleNames(self):
        """
        "global" can accept multiple names.
        """
        self.flakes('''
        def a(): global fu, bar; fu = 1; bar = 2
        def b(): fu; bar
        ''')

    def test_globalInGlobalScope(self):
        """
        A global statement in the global scope is ignored.
        """
        self.flakes('''
        global x
        def foo():
            print(x)
        ''', m.UndefinedName)

    def test_global_reset_name_only(self):
        """A global statement does not prevent other names being undefined."""
        # Only different undefined names are reported.
        # See following test that fails where the same name is used.
        self.flakes('''
        def f1():
            s

        def f2():
            global m
        ''', m.UndefinedName)

    @skip("todo")
    def test_unused_global(self):
        """An unused global statement does not define the name."""
        self.flakes('''
        def f1():
            m

        def f2():
            global m
        ''', m.UndefinedName)

    def test_del(self):
        """Del deletes bindings."""
        self.flakes('a = 1; del a; a', m.UndefinedName)

    def test_delGlobal(self):
        """Del a global binding from a function."""
        self.flakes('''
        a = 1
        def f():
            global a
            del a
        a
        ''')

    def test_delUndefined(self):
        """Del an undefined name."""
        self.flakes('del a', m.UndefinedName)

    def test_delConditional(self):
        """
        Ignores conditional bindings deletion.
        """
        self.flakes('''
        context = None
        test = True
        if False:
            del(test)
        assert(test)
        ''')

    def test_delConditionalNested(self):
        """
        Ignored conditional bindings deletion even if they are nested in other
        blocks.
        """
        self.flakes('''
        context = None
        test = True
        if False:
            with context():
                del(test)
        assert(test)
        ''')

    def test_delWhile(self):
        """
        Ignore bindings deletion if called inside the body of a while
        statement.
        """
        self.flakes('''
        def test():
            foo = 'bar'
            while False:
                del foo
            assert(foo)
        ''')

    def test_delWhileTestUsage(self):
        """
        Ignore bindings deletion if called inside the body of a while
        statement and name is used inside while's test part.
        """
        self.flakes('''
        def _worker():
            o = True
            while o is not True:
                del o
                o = False
        ''')

    def test_delWhileNested(self):
        """
        Ignore bindings deletions if node is part of while's test, even when
        del is in a nested block.
        """
        self.flakes('''
        context = None
        def _worker():
            o = True
            while o is not True:
                while True:
                    with context():
                        del o
                o = False
        ''')

    def test_globalFromNestedScope(self):
        """Global names are available from nested scopes."""
        self.flakes('''
        a = 1
        def b():
            def c():
                a
        ''')

    def test_laterRedefinedGlobalFromNestedScope(self):
        """
        Test that referencing a local name that shadows a global, before it is
        defined, generates a warning.
        """
        self.flakes('''
        a = 1
        def fun():
            a
            a = 2
            return a
        ''', m.UndefinedLocal)

    def test_laterRedefinedGlobalFromNestedScope2(self):
        """
        Test that referencing a local name in a nested scope that shadows a
        global declared in an enclosing scope, before it is defined, generates
        a warning.
        """
        self.flakes('''
            a = 1
            def fun():
                global a
                def fun2():
                    a
                    a = 2
                    return a
        ''', m.UndefinedLocal)

    def test_intermediateClassScopeIgnored(self):
        """
        If a name defined in an enclosing scope is shadowed by a local variable
        and the name is used locally before it is bound, an unbound local
        warning is emitted, even if there is a class scope between the enclosing
        scope and the local scope.
        """
        self.flakes('''
        def f():
            x = 1
            class g:
                def h(self):
                    a = x
                    x = None
                    print(x, a)
            print(x)
        ''', m.UndefinedLocal)

    def test_doubleNestingReportsClosestName(self):
        """
        Test that referencing a local name in a nested scope that shadows a
        variable declared in two different outer scopes before it is defined
        in the innermost scope generates an UnboundLocal warning which
        refers to the nearest shadowed name.
        """
        exc = self.flakes('''
            def a():
                x = 1
                def b():
                    x = 2 # line 5
                    def c():
                        x
                        x = 3
                        return x
                    return x
                return x
        ''', m.UndefinedLocal).messages[0]

        # _DoctestMixin.flakes adds two lines preceding the code above.
        expected_line_num = 7 if self.withDoctest else 5

        self.assertEqual(exc.message_args, ('x', expected_line_num))

    def test_laterRedefinedGlobalFromNestedScope3(self):
        """
        Test that referencing a local name in a nested scope that shadows a
        global, before it is defined, generates a warning.
        """
        self.flakes('''
            def fun():
                a = 1
                def fun2():
                    a
                    a = 1
                    return a
                return a
        ''', m.UndefinedLocal)

    def test_undefinedAugmentedAssignment(self):
        self.flakes(
            '''
            def f(seq):
                a = 0
                seq[a] += 1
                seq[b] /= 2
                c[0] *= 2
                a -= 3
                d += 4
                e[any] = 5
            ''',
            m.UndefinedName,    # b
            m.UndefinedName,    # c
            m.UndefinedName, m.UnusedVariable,  # d
            m.UndefinedName,    # e
        )

    def test_nestedClass(self):
        """Nested classes can access enclosing scope."""
        self.flakes('''
        def f(foo):
            class C:
                bar = foo
                def f(self):
                    return foo
            return C()

        f(123).f()
        ''')

    def test_badNestedClass(self):
        """Free variables in nested classes must bind at class creation."""
        self.flakes('''
        def f():
            class C:
                bar = foo
            foo = 456
            return foo
        f()
        ''', m.UndefinedName)

    def test_definedAsStarArgs(self):
        """Star and double-star arg names are defined."""
        self.flakes('''
        def f(a, *b, **c):
            print(a, b, c)
        ''')

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_definedAsStarUnpack(self):
        """Star names in unpack are defined."""
        self.flakes('''
        a, *b = range(10)
        print(a, b)
        ''')
        self.flakes('''
        *a, b = range(10)
        print(a, b)
        ''')
        self.flakes('''
        a, *b, c = range(10)
        print(a, b, c)
        ''')

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_usedAsStarUnpack(self):
        """
        Star names in unpack are used if RHS is not a tuple/list literal.
        """
        self.flakes('''
        def f():
            a, *b = range(10)
        ''')
        self.flakes('''
        def f():
            (*a, b) = range(10)
        ''')
        self.flakes('''
        def f():
            [a, *b, c] = range(10)
        ''')

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_unusedAsStarUnpack(self):
        """
        Star names in unpack are unused if RHS is a tuple/list literal.
        """
        self.flakes('''
        def f():
            a, *b = any, all, 4, 2, 'un'
        ''', m.UnusedVariable, m.UnusedVariable)
        self.flakes('''
        def f():
            (*a, b) = [bool, int, float, complex]
        ''', m.UnusedVariable, m.UnusedVariable)
        self.flakes('''
        def f():
            [a, *b, c] = 9, 8, 7, 6, 5, 4
        ''', m.UnusedVariable, m.UnusedVariable, m.UnusedVariable)

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_keywordOnlyArgs(self):
        """Keyword-only arg names are defined."""
        self.flakes('''
        def f(*, a, b=None):
            print(a, b)
        ''')

        self.flakes('''
        import default_b
        def f(*, a, b=default_b):
            print(a, b)
        ''')

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_keywordOnlyArgsUndefined(self):
        """Typo in kwonly name."""
        self.flakes('''
        def f(*, a, b=default_c):
            print(a, b)
        ''', m.UndefinedName)

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_annotationUndefined(self):
        """Undefined annotations."""
        self.flakes('''
        from abc import note1, note2, note3, note4, note5
        def func(a: note1, *args: note2,
                 b: note3=12, **kw: note4) -> note5: pass
        ''')

        self.flakes('''
        def func():
            d = e = 42
            def func(a: {1, d}) -> (lambda c: e): pass
        ''')

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_metaClassUndefined(self):
        self.flakes('''
        from abc import ABCMeta
        class A(metaclass=ABCMeta): pass
        ''')

    def test_definedInGenExp(self):
        """
        Using the loop variable of a generator expression results in no
        warnings.
        """
        self.flakes('(a for a in [1, 2, 3] if a)')

        self.flakes('(b for b in (a for a in [1, 2, 3] if a) if b)')

    def test_undefinedInGenExpNested(self):
        """
        The loop variables of generator expressions nested together are
        not defined in the other generator.
        """
        self.flakes('(b for b in (a for a in [1, 2, 3] if b) if b)',
                    m.UndefinedName)

        self.flakes('(b for b in (a for a in [1, 2, 3] if a) if a)',
                    m.UndefinedName)

    def test_undefinedWithErrorHandler(self):
        """
        Some compatibility code checks explicitly for NameError.
        It should not trigger warnings.
        """
        self.flakes('''
        try:
            socket_map
        except NameError:
            socket_map = {}
        ''')
        self.flakes('''
        try:
            _memoryview.contiguous
        except (NameError, AttributeError):
            raise RuntimeError("Python >= 3.3 is required")
        ''')
        # If NameError is not explicitly handled, generate a warning
        self.flakes('''
        try:
            socket_map
        except:
            socket_map = {}
        ''', m.UndefinedName)
        self.flakes('''
        try:
            socket_map
        except Exception:
            socket_map = {}
        ''', m.UndefinedName)

    def test_definedInClass(self):
        """
        Defined name for generator expressions and dict/set comprehension.
        """
        self.flakes('''
        class A:
            T = range(10)

            Z = (x for x in T)
            L = [x for x in T]
            B = dict((i, str(i)) for i in T)
        ''')

        self.flakes('''
        class A:
            T = range(10)

            X = {x for x in T}
            Y = {x:x for x in T}
        ''')

    def test_definedInClassNested(self):
        """Defined name for nested generator expressions in a class."""
        self.flakes('''
        class A:
            T = range(10)

            Z = (x for x in (a for a in T))
        ''')

    def test_undefinedInLoop(self):
        """
        The loop variable is defined after the expression is computed.
        """
        self.flakes('''
        for i in range(i):
            print(i)
        ''', m.UndefinedName)
        self.flakes('''
        [42 for i in range(i)]
        ''', m.UndefinedName)
        self.flakes('''
        (42 for i in range(i))
        ''', m.UndefinedName)

    def test_definedFromLambdaInDictionaryComprehension(self):
        """
        Defined name referenced from a lambda function within a dict/set
        comprehension.
        """
        self.flakes('''
        {lambda: id(x) for x in range(10)}
        ''')

    def test_definedFromLambdaInGenerator(self):
        """
        Defined name referenced from a lambda function within a generator
        expression.
        """
        self.flakes('''
        any(lambda: id(x) for x in range(10))
        ''')

    def test_undefinedFromLambdaInDictionaryComprehension(self):
        """
        Undefined name referenced from a lambda function within a dict/set
        comprehension.
        """
        self.flakes('''
        {lambda: id(y) for x in range(10)}
        ''', m.UndefinedName)

    def test_undefinedFromLambdaInComprehension(self):
        """
        Undefined name referenced from a lambda function within a generator
        expression.
        """
        self.flakes('''
        any(lambda: id(y) for x in range(10))
        ''', m.UndefinedName)

    def test_dunderClass(self):
        """
        `__class__` is defined in class scope under Python 3, but is not
        in Python 2.
        """
        code = '''
        class Test(object):
            def __init__(self):
                print(__class__.__name__)
                self.x = 1

        t = Test()
        '''
        if version_info < (3,):
            self.flakes(code, m.UndefinedName)
        else:
            self.flakes(code)


class NameTests(TestCase):
    """
    Tests for some extra cases of name handling.
    """
    def test_impossibleContext(self):
        """
        A Name node with an unrecognized context results in a RuntimeError being
        raised.
        """
        tree = ast.parse("x = 10")
        file_tokens = checker.make_tokens("x = 10")
        # Make it into something unrecognizable.
        tree.body[0].targets[0].ctx = object()
        self.assertRaises(RuntimeError, checker.Checker, tree, file_tokens=file_tokens)
