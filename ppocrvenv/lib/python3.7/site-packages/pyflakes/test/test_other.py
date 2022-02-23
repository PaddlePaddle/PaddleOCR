"""
Tests for various Pyflakes behavior.
"""

from sys import version_info

from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf


class Test(TestCase):

    def test_duplicateArgs(self):
        self.flakes('def fu(bar, bar): pass', m.DuplicateArgument)

    def test_localReferencedBeforeAssignment(self):
        self.flakes('''
        a = 1
        def f():
            a; a=1
        f()
        ''', m.UndefinedLocal, m.UnusedVariable)

    @skipIf(version_info >= (3,),
            'in Python 3 list comprehensions execute in a separate scope')
    def test_redefinedInListComp(self):
        """
        Test that shadowing a variable in a list comprehension raises
        a warning.
        """
        self.flakes('''
        a = 1
        [1 for a, b in [(1, 2)]]
        ''', m.RedefinedInListComp)
        self.flakes('''
        class A:
            a = 1
            [1 for a, b in [(1, 2)]]
        ''', m.RedefinedInListComp)
        self.flakes('''
        def f():
            a = 1
            [1 for a, b in [(1, 2)]]
        ''', m.RedefinedInListComp)
        self.flakes('''
        [1 for a, b in [(1, 2)]]
        [1 for a, b in [(1, 2)]]
        ''')
        self.flakes('''
        for a, b in [(1, 2)]:
            pass
        [1 for a, b in [(1, 2)]]
        ''')

    def test_redefinedInGenerator(self):
        """
        Test that reusing a variable in a generator does not raise
        a warning.
        """
        self.flakes('''
        a = 1
        (1 for a, b in [(1, 2)])
        ''')
        self.flakes('''
        class A:
            a = 1
            list(1 for a, b in [(1, 2)])
        ''')
        self.flakes('''
        def f():
            a = 1
            (1 for a, b in [(1, 2)])
        ''', m.UnusedVariable)
        self.flakes('''
        (1 for a, b in [(1, 2)])
        (1 for a, b in [(1, 2)])
        ''')
        self.flakes('''
        for a, b in [(1, 2)]:
            pass
        (1 for a, b in [(1, 2)])
        ''')

    def test_redefinedInSetComprehension(self):
        """
        Test that reusing a variable in a set comprehension does not raise
        a warning.
        """
        self.flakes('''
        a = 1
        {1 for a, b in [(1, 2)]}
        ''')
        self.flakes('''
        class A:
            a = 1
            {1 for a, b in [(1, 2)]}
        ''')
        self.flakes('''
        def f():
            a = 1
            {1 for a, b in [(1, 2)]}
        ''', m.UnusedVariable)
        self.flakes('''
        {1 for a, b in [(1, 2)]}
        {1 for a, b in [(1, 2)]}
        ''')
        self.flakes('''
        for a, b in [(1, 2)]:
            pass
        {1 for a, b in [(1, 2)]}
        ''')

    def test_redefinedInDictComprehension(self):
        """
        Test that reusing a variable in a dict comprehension does not raise
        a warning.
        """
        self.flakes('''
        a = 1
        {1: 42 for a, b in [(1, 2)]}
        ''')
        self.flakes('''
        class A:
            a = 1
            {1: 42 for a, b in [(1, 2)]}
        ''')
        self.flakes('''
        def f():
            a = 1
            {1: 42 for a, b in [(1, 2)]}
        ''', m.UnusedVariable)
        self.flakes('''
        {1: 42 for a, b in [(1, 2)]}
        {1: 42 for a, b in [(1, 2)]}
        ''')
        self.flakes('''
        for a, b in [(1, 2)]:
            pass
        {1: 42 for a, b in [(1, 2)]}
        ''')

    def test_redefinedFunction(self):
        """
        Test that shadowing a function definition with another one raises a
        warning.
        """
        self.flakes('''
        def a(): pass
        def a(): pass
        ''', m.RedefinedWhileUnused)

    def test_redefinedUnderscoreFunction(self):
        """
        Test that shadowing a function definition named with underscore doesn't
        raise anything.
        """
        self.flakes('''
        def _(): pass
        def _(): pass
        ''')

    def test_redefinedUnderscoreImportation(self):
        """
        Test that shadowing an underscore importation raises a warning.
        """
        self.flakes('''
        from .i18n import _
        def _(): pass
        ''', m.RedefinedWhileUnused)

    def test_redefinedClassFunction(self):
        """
        Test that shadowing a function definition in a class suite with another
        one raises a warning.
        """
        self.flakes('''
        class A:
            def a(): pass
            def a(): pass
        ''', m.RedefinedWhileUnused)

    def test_redefinedIfElseFunction(self):
        """
        Test that shadowing a function definition twice in an if
        and else block does not raise a warning.
        """
        self.flakes('''
        if True:
            def a(): pass
        else:
            def a(): pass
        ''')

    def test_redefinedIfFunction(self):
        """
        Test that shadowing a function definition within an if block
        raises a warning.
        """
        self.flakes('''
        if True:
            def a(): pass
            def a(): pass
        ''', m.RedefinedWhileUnused)

    def test_redefinedTryExceptFunction(self):
        """
        Test that shadowing a function definition twice in try
        and except block does not raise a warning.
        """
        self.flakes('''
        try:
            def a(): pass
        except:
            def a(): pass
        ''')

    def test_redefinedTryFunction(self):
        """
        Test that shadowing a function definition within a try block
        raises a warning.
        """
        self.flakes('''
        try:
            def a(): pass
            def a(): pass
        except:
            pass
        ''', m.RedefinedWhileUnused)

    def test_redefinedIfElseInListComp(self):
        """
        Test that shadowing a variable in a list comprehension in
        an if and else block does not raise a warning.
        """
        self.flakes('''
        if False:
            a = 1
        else:
            [a for a in '12']
        ''')

    @skipIf(version_info >= (3,),
            'in Python 3 list comprehensions execute in a separate scope')
    def test_redefinedElseInListComp(self):
        """
        Test that shadowing a variable in a list comprehension in
        an else (or if) block raises a warning.
        """
        self.flakes('''
        if False:
            pass
        else:
            a = 1
            [a for a in '12']
        ''', m.RedefinedInListComp)

    def test_functionDecorator(self):
        """
        Test that shadowing a function definition with a decorated version of
        that function does not raise a warning.
        """
        self.flakes('''
        from somewhere import somedecorator

        def a(): pass
        a = somedecorator(a)
        ''')

    def test_classFunctionDecorator(self):
        """
        Test that shadowing a function definition in a class suite with a
        decorated version of that function does not raise a warning.
        """
        self.flakes('''
        class A:
            def a(): pass
            a = classmethod(a)
        ''')

    def test_modernProperty(self):
        self.flakes("""
        class A:
            @property
            def t(self):
                pass
            @t.setter
            def t(self, value):
                pass
            @t.deleter
            def t(self):
                pass
        """)

    def test_unaryPlus(self):
        """Don't die on unary +."""
        self.flakes('+1')

    def test_undefinedBaseClass(self):
        """
        If a name in the base list of a class definition is undefined, a
        warning is emitted.
        """
        self.flakes('''
        class foo(foo):
            pass
        ''', m.UndefinedName)

    def test_classNameUndefinedInClassBody(self):
        """
        If a class name is used in the body of that class's definition and
        the name is not already defined, a warning is emitted.
        """
        self.flakes('''
        class foo:
            foo
        ''', m.UndefinedName)

    def test_classNameDefinedPreviously(self):
        """
        If a class name is used in the body of that class's definition and
        the name was previously defined in some other way, no warning is
        emitted.
        """
        self.flakes('''
        foo = None
        class foo:
            foo
        ''')

    def test_classRedefinition(self):
        """
        If a class is defined twice in the same module, a warning is emitted.
        """
        self.flakes('''
        class Foo:
            pass
        class Foo:
            pass
        ''', m.RedefinedWhileUnused)

    def test_functionRedefinedAsClass(self):
        """
        If a function is redefined as a class, a warning is emitted.
        """
        self.flakes('''
        def Foo():
            pass
        class Foo:
            pass
        ''', m.RedefinedWhileUnused)

    def test_classRedefinedAsFunction(self):
        """
        If a class is redefined as a function, a warning is emitted.
        """
        self.flakes('''
        class Foo:
            pass
        def Foo():
            pass
        ''', m.RedefinedWhileUnused)

    def test_classWithReturn(self):
        """
        If a return is used inside a class, a warning is emitted.
        """
        self.flakes('''
        class Foo(object):
            return
        ''', m.ReturnOutsideFunction)

    def test_moduleWithReturn(self):
        """
        If a return is used at the module level, a warning is emitted.
        """
        self.flakes('''
        return
        ''', m.ReturnOutsideFunction)

    def test_classWithYield(self):
        """
        If a yield is used inside a class, a warning is emitted.
        """
        self.flakes('''
        class Foo(object):
            yield
        ''', m.YieldOutsideFunction)

    def test_moduleWithYield(self):
        """
        If a yield is used at the module level, a warning is emitted.
        """
        self.flakes('''
        yield
        ''', m.YieldOutsideFunction)

    @skipIf(version_info < (3, 3), "Python >= 3.3 only")
    def test_classWithYieldFrom(self):
        """
        If a yield from is used inside a class, a warning is emitted.
        """
        self.flakes('''
        class Foo(object):
            yield from range(10)
        ''', m.YieldOutsideFunction)

    @skipIf(version_info < (3, 3), "Python >= 3.3 only")
    def test_moduleWithYieldFrom(self):
        """
        If a yield from is used at the module level, a warning is emitted.
        """
        self.flakes('''
        yield from range(10)
        ''', m.YieldOutsideFunction)

    def test_continueOutsideLoop(self):
        self.flakes('''
        continue
        ''', m.ContinueOutsideLoop)

        self.flakes('''
        def f():
            continue
        ''', m.ContinueOutsideLoop)

        self.flakes('''
        while True:
            pass
        else:
            continue
        ''', m.ContinueOutsideLoop)

        self.flakes('''
        while True:
            pass
        else:
            if 1:
                if 2:
                    continue
        ''', m.ContinueOutsideLoop)

        self.flakes('''
        while True:
            def f():
                continue
        ''', m.ContinueOutsideLoop)

        self.flakes('''
        while True:
            class A:
                continue
        ''', m.ContinueOutsideLoop)

    def test_continueInsideLoop(self):
        self.flakes('''
        while True:
            continue
        ''')

        self.flakes('''
        for i in range(10):
            continue
        ''')

        self.flakes('''
        while True:
            if 1:
                continue
        ''')

        self.flakes('''
        for i in range(10):
            if 1:
                continue
        ''')

        self.flakes('''
        while True:
            while True:
                pass
            else:
                continue
        else:
            pass
        ''')

        self.flakes('''
        while True:
            try:
                pass
            finally:
                while True:
                    continue
        ''')

    @skipIf(version_info > (3, 8), "Python <= 3.8 only")
    def test_continueInFinally(self):
        # 'continue' inside 'finally' is a special syntax error
        # that is removed in 3.8
        self.flakes('''
        while True:
            try:
                pass
            finally:
                continue
        ''', m.ContinueInFinally)

        self.flakes('''
        while True:
            try:
                pass
            finally:
                if 1:
                    if 2:
                        continue
        ''', m.ContinueInFinally)

        # Even when not in a loop, this is the error Python gives
        self.flakes('''
        try:
            pass
        finally:
            continue
        ''', m.ContinueInFinally)

    def test_breakOutsideLoop(self):
        self.flakes('''
        break
        ''', m.BreakOutsideLoop)

        self.flakes('''
        def f():
            break
        ''', m.BreakOutsideLoop)

        self.flakes('''
        while True:
            pass
        else:
            break
        ''', m.BreakOutsideLoop)

        self.flakes('''
        while True:
            pass
        else:
            if 1:
                if 2:
                    break
        ''', m.BreakOutsideLoop)

        self.flakes('''
        while True:
            def f():
                break
        ''', m.BreakOutsideLoop)

        self.flakes('''
        while True:
            class A:
                break
        ''', m.BreakOutsideLoop)

        self.flakes('''
        try:
            pass
        finally:
            break
        ''', m.BreakOutsideLoop)

    def test_breakInsideLoop(self):
        self.flakes('''
        while True:
            break
        ''')

        self.flakes('''
        for i in range(10):
            break
        ''')

        self.flakes('''
        while True:
            if 1:
                break
        ''')

        self.flakes('''
        for i in range(10):
            if 1:
                break
        ''')

        self.flakes('''
        while True:
            while True:
                pass
            else:
                break
        else:
            pass
        ''')

        self.flakes('''
        while True:
            try:
                pass
            finally:
                while True:
                    break
        ''')

        self.flakes('''
        while True:
            try:
                pass
            finally:
                break
        ''')

        self.flakes('''
        while True:
            try:
                pass
            finally:
                if 1:
                    if 2:
                        break
        ''')

    def test_defaultExceptLast(self):
        """
        A default except block should be last.

        YES:

        try:
            ...
        except Exception:
            ...
        except:
            ...

        NO:

        try:
            ...
        except:
            ...
        except Exception:
            ...
        """
        self.flakes('''
        try:
            pass
        except ValueError:
            pass
        ''')

        self.flakes('''
        try:
            pass
        except ValueError:
            pass
        except:
            pass
        ''')

        self.flakes('''
        try:
            pass
        except:
            pass
        ''')

        self.flakes('''
        try:
            pass
        except ValueError:
            pass
        else:
            pass
        ''')

        self.flakes('''
        try:
            pass
        except:
            pass
        else:
            pass
        ''')

        self.flakes('''
        try:
            pass
        except ValueError:
            pass
        except:
            pass
        else:
            pass
        ''')

    def test_defaultExceptNotLast(self):
        self.flakes('''
        try:
            pass
        except:
            pass
        except ValueError:
            pass
        ''', m.DefaultExceptNotLast)

        self.flakes('''
        try:
            pass
        except:
            pass
        except:
            pass
        ''', m.DefaultExceptNotLast)

        self.flakes('''
        try:
            pass
        except:
            pass
        except ValueError:
            pass
        except:
            pass
        ''', m.DefaultExceptNotLast)

        self.flakes('''
        try:
            pass
        except:
            pass
        except ValueError:
            pass
        except:
            pass
        except ValueError:
            pass
        ''', m.DefaultExceptNotLast, m.DefaultExceptNotLast)

        self.flakes('''
        try:
            pass
        except:
            pass
        except ValueError:
            pass
        else:
            pass
        ''', m.DefaultExceptNotLast)

        self.flakes('''
        try:
            pass
        except:
            pass
        except:
            pass
        else:
            pass
        ''', m.DefaultExceptNotLast)

        self.flakes('''
        try:
            pass
        except:
            pass
        except ValueError:
            pass
        except:
            pass
        else:
            pass
        ''', m.DefaultExceptNotLast)

        self.flakes('''
        try:
            pass
        except:
            pass
        except ValueError:
            pass
        except:
            pass
        except ValueError:
            pass
        else:
            pass
        ''', m.DefaultExceptNotLast, m.DefaultExceptNotLast)

        self.flakes('''
        try:
            pass
        except:
            pass
        except ValueError:
            pass
        finally:
            pass
        ''', m.DefaultExceptNotLast)

        self.flakes('''
        try:
            pass
        except:
            pass
        except:
            pass
        finally:
            pass
        ''', m.DefaultExceptNotLast)

        self.flakes('''
        try:
            pass
        except:
            pass
        except ValueError:
            pass
        except:
            pass
        finally:
            pass
        ''', m.DefaultExceptNotLast)

        self.flakes('''
        try:
            pass
        except:
            pass
        except ValueError:
            pass
        except:
            pass
        except ValueError:
            pass
        finally:
            pass
        ''', m.DefaultExceptNotLast, m.DefaultExceptNotLast)

        self.flakes('''
        try:
            pass
        except:
            pass
        except ValueError:
            pass
        else:
            pass
        finally:
            pass
        ''', m.DefaultExceptNotLast)

        self.flakes('''
        try:
            pass
        except:
            pass
        except:
            pass
        else:
            pass
        finally:
            pass
        ''', m.DefaultExceptNotLast)

        self.flakes('''
        try:
            pass
        except:
            pass
        except ValueError:
            pass
        except:
            pass
        else:
            pass
        finally:
            pass
        ''', m.DefaultExceptNotLast)

        self.flakes('''
        try:
            pass
        except:
            pass
        except ValueError:
            pass
        except:
            pass
        except ValueError:
            pass
        else:
            pass
        finally:
            pass
        ''', m.DefaultExceptNotLast, m.DefaultExceptNotLast)

    @skipIf(version_info < (3,), "Python 3 only")
    def test_starredAssignmentNoError(self):
        """
        Python 3 extended iterable unpacking
        """
        self.flakes('''
        a, *b = range(10)
        ''')

        self.flakes('''
        *a, b = range(10)
        ''')

        self.flakes('''
        a, *b, c = range(10)
        ''')

        self.flakes('''
        (a, *b) = range(10)
        ''')

        self.flakes('''
        (*a, b) = range(10)
        ''')

        self.flakes('''
        (a, *b, c) = range(10)
        ''')

        self.flakes('''
        [a, *b] = range(10)
        ''')

        self.flakes('''
        [*a, b] = range(10)
        ''')

        self.flakes('''
        [a, *b, c] = range(10)
        ''')

        # Taken from test_unpack_ex.py in the cPython source
        s = ", ".join("a%d" % i for i in range(1 << 8 - 1)) + \
            ", *rest = range(1<<8)"
        self.flakes(s)

        s = "(" + ", ".join("a%d" % i for i in range(1 << 8 - 1)) + \
            ", *rest) = range(1<<8)"
        self.flakes(s)

        s = "[" + ", ".join("a%d" % i for i in range(1 << 8 - 1)) + \
            ", *rest] = range(1<<8)"
        self.flakes(s)

    @skipIf(version_info < (3, ), "Python 3 only")
    def test_starredAssignmentErrors(self):
        """
        SyntaxErrors (not encoded in the ast) surrounding Python 3 extended
        iterable unpacking
        """
        # Taken from test_unpack_ex.py in the cPython source
        s = ", ".join("a%d" % i for i in range(1 << 8)) + \
            ", *rest = range(1<<8 + 1)"
        self.flakes(s, m.TooManyExpressionsInStarredAssignment)

        s = "(" + ", ".join("a%d" % i for i in range(1 << 8)) + \
            ", *rest) = range(1<<8 + 1)"
        self.flakes(s, m.TooManyExpressionsInStarredAssignment)

        s = "[" + ", ".join("a%d" % i for i in range(1 << 8)) + \
            ", *rest] = range(1<<8 + 1)"
        self.flakes(s, m.TooManyExpressionsInStarredAssignment)

        s = ", ".join("a%d" % i for i in range(1 << 8 + 1)) + \
            ", *rest = range(1<<8 + 2)"
        self.flakes(s, m.TooManyExpressionsInStarredAssignment)

        s = "(" + ", ".join("a%d" % i for i in range(1 << 8 + 1)) + \
            ", *rest) = range(1<<8 + 2)"
        self.flakes(s, m.TooManyExpressionsInStarredAssignment)

        s = "[" + ", ".join("a%d" % i for i in range(1 << 8 + 1)) + \
            ", *rest] = range(1<<8 + 2)"
        self.flakes(s, m.TooManyExpressionsInStarredAssignment)

        # No way we can actually test this!
        # s = "*rest, " + ", ".join("a%d" % i for i in range(1<<24)) + \
        #    ", *rest = range(1<<24 + 1)"
        # self.flakes(s, m.TooManyExpressionsInStarredAssignment)

        self.flakes('''
        a, *b, *c = range(10)
        ''', m.TwoStarredExpressions)

        self.flakes('''
        a, *b, c, *d = range(10)
        ''', m.TwoStarredExpressions)

        self.flakes('''
        *a, *b, *c = range(10)
        ''', m.TwoStarredExpressions)

        self.flakes('''
        (a, *b, *c) = range(10)
        ''', m.TwoStarredExpressions)

        self.flakes('''
        (a, *b, c, *d) = range(10)
        ''', m.TwoStarredExpressions)

        self.flakes('''
        (*a, *b, *c) = range(10)
        ''', m.TwoStarredExpressions)

        self.flakes('''
        [a, *b, *c] = range(10)
        ''', m.TwoStarredExpressions)

        self.flakes('''
        [a, *b, c, *d] = range(10)
        ''', m.TwoStarredExpressions)

        self.flakes('''
        [*a, *b, *c] = range(10)
        ''', m.TwoStarredExpressions)

    @skip("todo: Too hard to make this warn but other cases stay silent")
    def test_doubleAssignment(self):
        """
        If a variable is re-assigned to without being used, no warning is
        emitted.
        """
        self.flakes('''
        x = 10
        x = 20
        ''', m.RedefinedWhileUnused)

    def test_doubleAssignmentConditionally(self):
        """
        If a variable is re-assigned within a conditional, no warning is
        emitted.
        """
        self.flakes('''
        x = 10
        if True:
            x = 20
        ''')

    def test_doubleAssignmentWithUse(self):
        """
        If a variable is re-assigned to after being used, no warning is
        emitted.
        """
        self.flakes('''
        x = 10
        y = x * 2
        x = 20
        ''')

    def test_comparison(self):
        """
        If a defined name is used on either side of any of the six comparison
        operators, no warning is emitted.
        """
        self.flakes('''
        x = 10
        y = 20
        x < y
        x <= y
        x == y
        x != y
        x >= y
        x > y
        ''')

    def test_identity(self):
        """
        If a defined name is used on either side of an identity test, no
        warning is emitted.
        """
        self.flakes('''
        x = 10
        y = 20
        x is y
        x is not y
        ''')

    def test_containment(self):
        """
        If a defined name is used on either side of a containment test, no
        warning is emitted.
        """
        self.flakes('''
        x = 10
        y = 20
        x in y
        x not in y
        ''')

    def test_loopControl(self):
        """
        break and continue statements are supported.
        """
        self.flakes('''
        for x in [1, 2]:
            break
        ''')
        self.flakes('''
        for x in [1, 2]:
            continue
        ''')

    def test_ellipsis(self):
        """
        Ellipsis in a slice is supported.
        """
        self.flakes('''
        [1, 2][...]
        ''')

    def test_extendedSlice(self):
        """
        Extended slices are supported.
        """
        self.flakes('''
        x = 3
        [1, 2][x,:]
        ''')

    def test_varAugmentedAssignment(self):
        """
        Augmented assignment of a variable is supported.
        We don't care about var refs.
        """
        self.flakes('''
        foo = 0
        foo += 1
        ''')

    def test_attrAugmentedAssignment(self):
        """
        Augmented assignment of attributes is supported.
        We don't care about attr refs.
        """
        self.flakes('''
        foo = None
        foo.bar += foo.baz
        ''')

    def test_globalDeclaredInDifferentScope(self):
        """
        A 'global' can be declared in one scope and reused in another.
        """
        self.flakes('''
        def f(): global foo
        def g(): foo = 'anything'; foo.is_used()
        ''')

    def test_function_arguments(self):
        """
        Test to traverse ARG and ARGUMENT handler
        """
        self.flakes('''
        def foo(a, b):
            pass
        ''')

        self.flakes('''
        def foo(a, b, c=0):
            pass
        ''')

        self.flakes('''
        def foo(a, b, c=0, *args):
            pass
        ''')

        self.flakes('''
        def foo(a, b, c=0, *args, **kwargs):
            pass
        ''')

    @skipIf(version_info < (3, 3), "Python >= 3.3 only")
    def test_function_arguments_python3(self):
        self.flakes('''
        def foo(a, b, c=0, *args, d=0, **kwargs):
            pass
        ''')


class TestUnusedAssignment(TestCase):
    """
    Tests for warning about unused assignments.
    """

    def test_unusedVariable(self):
        """
        Warn when a variable in a function is assigned a value that's never
        used.
        """
        self.flakes('''
        def a():
            b = 1
        ''', m.UnusedVariable)

    def test_unusedUnderscoreVariable(self):
        """
        Don't warn when the magic "_" (underscore) variable is unused.
        See issue #202.
        """
        self.flakes('''
        def a(unused_param):
            _ = unused_param
        ''')

    def test_unusedVariableAsLocals(self):
        """
        Using locals() it is perfectly valid to have unused variables
        """
        self.flakes('''
        def a():
            b = 1
            return locals()
        ''')

    def test_unusedVariableNoLocals(self):
        """
        Using locals() in wrong scope should not matter
        """
        self.flakes('''
        def a():
            locals()
            def a():
                b = 1
                return
        ''', m.UnusedVariable)

    @skip("todo: Difficult because it doesn't apply in the context of a loop")
    def test_unusedReassignedVariable(self):
        """
        Shadowing a used variable can still raise an UnusedVariable warning.
        """
        self.flakes('''
        def a():
            b = 1
            b.foo()
            b = 2
        ''', m.UnusedVariable)

    def test_variableUsedInLoop(self):
        """
        Shadowing a used variable cannot raise an UnusedVariable warning in the
        context of a loop.
        """
        self.flakes('''
        def a():
            b = True
            while b:
                b = False
        ''')

    def test_assignToGlobal(self):
        """
        Assigning to a global and then not using that global is perfectly
        acceptable. Do not mistake it for an unused local variable.
        """
        self.flakes('''
        b = 0
        def a():
            global b
            b = 1
        ''')

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_assignToNonlocal(self):
        """
        Assigning to a nonlocal and then not using that binding is perfectly
        acceptable. Do not mistake it for an unused local variable.
        """
        self.flakes('''
        b = b'0'
        def a():
            nonlocal b
            b = b'1'
        ''')

    def test_assignToMember(self):
        """
        Assigning to a member of another object and then not using that member
        variable is perfectly acceptable. Do not mistake it for an unused
        local variable.
        """
        # XXX: Adding this test didn't generate a failure. Maybe not
        # necessary?
        self.flakes('''
        class b:
            pass
        def a():
            b.foo = 1
        ''')

    def test_assignInForLoop(self):
        """
        Don't warn when a variable in a for loop is assigned to but not used.
        """
        self.flakes('''
        def f():
            for i in range(10):
                pass
        ''')

    def test_assignInListComprehension(self):
        """
        Don't warn when a variable in a list comprehension is
        assigned to but not used.
        """
        self.flakes('''
        def f():
            [None for i in range(10)]
        ''')

    def test_generatorExpression(self):
        """
        Don't warn when a variable in a generator expression is
        assigned to but not used.
        """
        self.flakes('''
        def f():
            (None for i in range(10))
        ''')

    def test_assignmentInsideLoop(self):
        """
        Don't warn when a variable assignment occurs lexically after its use.
        """
        self.flakes('''
        def f():
            x = None
            for i in range(10):
                if i > 2:
                    return x
                x = i * 2
        ''')

    def test_tupleUnpacking(self):
        """
        Don't warn when a variable included in tuple unpacking is unused. It's
        very common for variables in a tuple unpacking assignment to be unused
        in good Python code, so warning will only create false positives.
        """
        self.flakes('''
        def f(tup):
            (x, y) = tup
        ''')
        self.flakes('''
        def f():
            (x, y) = 1, 2
        ''', m.UnusedVariable, m.UnusedVariable)
        self.flakes('''
        def f():
            (x, y) = coords = 1, 2
            if x > 1:
                print(coords)
        ''')
        self.flakes('''
        def f():
            (x, y) = coords = 1, 2
        ''', m.UnusedVariable)
        self.flakes('''
        def f():
            coords = (x, y) = 1, 2
        ''', m.UnusedVariable)

    def test_listUnpacking(self):
        """
        Don't warn when a variable included in list unpacking is unused.
        """
        self.flakes('''
        def f(tup):
            [x, y] = tup
        ''')
        self.flakes('''
        def f():
            [x, y] = [1, 2]
        ''', m.UnusedVariable, m.UnusedVariable)

    def test_closedOver(self):
        """
        Don't warn when the assignment is used in an inner function.
        """
        self.flakes('''
        def barMaker():
            foo = 5
            def bar():
                return foo
            return bar
        ''')

    def test_doubleClosedOver(self):
        """
        Don't warn when the assignment is used in an inner function, even if
        that inner function itself is in an inner function.
        """
        self.flakes('''
        def barMaker():
            foo = 5
            def bar():
                def baz():
                    return foo
            return bar
        ''')

    def test_tracebackhideSpecialVariable(self):
        """
        Do not warn about unused local variable __tracebackhide__, which is
        a special variable for py.test.
        """
        self.flakes("""
            def helper():
                __tracebackhide__ = True
        """)

    def test_ifexp(self):
        """
        Test C{foo if bar else baz} statements.
        """
        self.flakes("a = 'moo' if True else 'oink'")
        self.flakes("a = foo if True else 'oink'", m.UndefinedName)
        self.flakes("a = 'moo' if True else bar", m.UndefinedName)

    def test_if_tuple(self):
        """
        Test C{if (foo,)} conditions.
        """
        self.flakes("""if (): pass""")
        self.flakes("""
        if (
            True
        ):
            pass
        """)
        self.flakes("""
        if (
            True,
        ):
            pass
        """, m.IfTuple)
        self.flakes("""
        x = 1 if (
            True,
        ) else 2
        """, m.IfTuple)

    def test_withStatementNoNames(self):
        """
        No warnings are emitted for using inside or after a nameless C{with}
        statement a name defined beforehand.
        """
        self.flakes('''
        from __future__ import with_statement
        bar = None
        with open("foo"):
            bar
        bar
        ''')

    def test_withStatementSingleName(self):
        """
        No warnings are emitted for using a name defined by a C{with} statement
        within the suite or afterwards.
        """
        self.flakes('''
        from __future__ import with_statement
        with open('foo') as bar:
            bar
        bar
        ''')

    def test_withStatementAttributeName(self):
        """
        No warnings are emitted for using an attribute as the target of a
        C{with} statement.
        """
        self.flakes('''
        from __future__ import with_statement
        import foo
        with open('foo') as foo.bar:
            pass
        ''')

    def test_withStatementSubscript(self):
        """
        No warnings are emitted for using a subscript as the target of a
        C{with} statement.
        """
        self.flakes('''
        from __future__ import with_statement
        import foo
        with open('foo') as foo[0]:
            pass
        ''')

    def test_withStatementSubscriptUndefined(self):
        """
        An undefined name warning is emitted if the subscript used as the
        target of a C{with} statement is not defined.
        """
        self.flakes('''
        from __future__ import with_statement
        import foo
        with open('foo') as foo[bar]:
            pass
        ''', m.UndefinedName)

    def test_withStatementTupleNames(self):
        """
        No warnings are emitted for using any of the tuple of names defined by
        a C{with} statement within the suite or afterwards.
        """
        self.flakes('''
        from __future__ import with_statement
        with open('foo') as (bar, baz):
            bar, baz
        bar, baz
        ''')

    def test_withStatementListNames(self):
        """
        No warnings are emitted for using any of the list of names defined by a
        C{with} statement within the suite or afterwards.
        """
        self.flakes('''
        from __future__ import with_statement
        with open('foo') as [bar, baz]:
            bar, baz
        bar, baz
        ''')

    def test_withStatementComplicatedTarget(self):
        """
        If the target of a C{with} statement uses any or all of the valid forms
        for that part of the grammar (See
        U{http://docs.python.org/reference/compound_stmts.html#the-with-statement}),
        the names involved are checked both for definedness and any bindings
        created are respected in the suite of the statement and afterwards.
        """
        self.flakes('''
        from __future__ import with_statement
        c = d = e = g = h = i = None
        with open('foo') as [(a, b), c[d], e.f, g[h:i]]:
            a, b, c, d, e, g, h, i
        a, b, c, d, e, g, h, i
        ''')

    def test_withStatementSingleNameUndefined(self):
        """
        An undefined name warning is emitted if the name first defined by a
        C{with} statement is used before the C{with} statement.
        """
        self.flakes('''
        from __future__ import with_statement
        bar
        with open('foo') as bar:
            pass
        ''', m.UndefinedName)

    def test_withStatementTupleNamesUndefined(self):
        """
        An undefined name warning is emitted if a name first defined by the
        tuple-unpacking form of the C{with} statement is used before the
        C{with} statement.
        """
        self.flakes('''
        from __future__ import with_statement
        baz
        with open('foo') as (bar, baz):
            pass
        ''', m.UndefinedName)

    def test_withStatementSingleNameRedefined(self):
        """
        A redefined name warning is emitted if a name bound by an import is
        rebound by the name defined by a C{with} statement.
        """
        self.flakes('''
        from __future__ import with_statement
        import bar
        with open('foo') as bar:
            pass
        ''', m.RedefinedWhileUnused)

    def test_withStatementTupleNamesRedefined(self):
        """
        A redefined name warning is emitted if a name bound by an import is
        rebound by one of the names defined by the tuple-unpacking form of a
        C{with} statement.
        """
        self.flakes('''
        from __future__ import with_statement
        import bar
        with open('foo') as (bar, baz):
            pass
        ''', m.RedefinedWhileUnused)

    def test_withStatementUndefinedInside(self):
        """
        An undefined name warning is emitted if a name is used inside the
        body of a C{with} statement without first being bound.
        """
        self.flakes('''
        from __future__ import with_statement
        with open('foo') as bar:
            baz
        ''', m.UndefinedName)

    def test_withStatementNameDefinedInBody(self):
        """
        A name defined in the body of a C{with} statement can be used after
        the body ends without warning.
        """
        self.flakes('''
        from __future__ import with_statement
        with open('foo') as bar:
            baz = 10
        baz
        ''')

    def test_withStatementUndefinedInExpression(self):
        """
        An undefined name warning is emitted if a name in the I{test}
        expression of a C{with} statement is undefined.
        """
        self.flakes('''
        from __future__ import with_statement
        with bar as baz:
            pass
        ''', m.UndefinedName)

        self.flakes('''
        from __future__ import with_statement
        with bar as bar:
            pass
        ''', m.UndefinedName)

    def test_dictComprehension(self):
        """
        Dict comprehensions are properly handled.
        """
        self.flakes('''
        a = {1: x for x in range(10)}
        ''')

    def test_setComprehensionAndLiteral(self):
        """
        Set comprehensions are properly handled.
        """
        self.flakes('''
        a = {1, 2, 3}
        b = {x for x in range(10)}
        ''')

    def test_exceptionUsedInExcept(self):
        self.flakes('''
        try: pass
        except Exception as e: e
        ''')

        self.flakes('''
        def download_review():
            try: pass
            except Exception as e: e
        ''')

    @skipIf(version_info < (3,),
            "In Python 2 exception names stay bound after the exception handler")
    def test_exceptionUnusedInExcept(self):
        self.flakes('''
        try: pass
        except Exception as e: pass
        ''', m.UnusedVariable)

    def test_exceptionUnusedInExceptInFunction(self):
        self.flakes('''
        def download_review():
            try: pass
            except Exception as e: pass
        ''', m.UnusedVariable)

    def test_exceptWithoutNameInFunction(self):
        """
        Don't issue false warning when an unnamed exception is used.
        Previously, there would be a false warning, but only when the
        try..except was in a function
        """
        self.flakes('''
        import tokenize
        def foo():
            try: pass
            except tokenize.TokenError: pass
        ''')

    def test_exceptWithoutNameInFunctionTuple(self):
        """
        Don't issue false warning when an unnamed exception is used.
        This example catches a tuple of exception types.
        """
        self.flakes('''
        import tokenize
        def foo():
            try: pass
            except (tokenize.TokenError, IndentationError): pass
        ''')

    def test_augmentedAssignmentImportedFunctionCall(self):
        """
        Consider a function that is called on the right part of an
        augassign operation to be used.
        """
        self.flakes('''
        from foo import bar
        baz = 0
        baz += bar()
        ''')

    def test_assert_without_message(self):
        """An assert without a message is not an error."""
        self.flakes('''
        a = 1
        assert a
        ''')

    def test_assert_with_message(self):
        """An assert with a message is not an error."""
        self.flakes('''
        a = 1
        assert a, 'x'
        ''')

    def test_assert_tuple(self):
        """An assert of a non-empty tuple is always True."""
        self.flakes('''
        assert (False, 'x')
        assert (False, )
        ''', m.AssertTuple, m.AssertTuple)

    def test_assert_tuple_empty(self):
        """An assert of an empty tuple is always False."""
        self.flakes('''
        assert ()
        ''')

    def test_assert_static(self):
        """An assert of a static value is not an error."""
        self.flakes('''
        assert True
        assert 1
        ''')

    @skipIf(version_info < (3, 3), 'new in Python 3.3')
    def test_yieldFromUndefined(self):
        """
        Test C{yield from} statement
        """
        self.flakes('''
        def bar():
            yield from foo()
        ''', m.UndefinedName)

    @skipIf(version_info < (3, 6), 'new in Python 3.6')
    def test_f_string(self):
        """Test PEP 498 f-strings are treated as a usage."""
        self.flakes('''
        baz = 0
        print(f'\x7b4*baz\N{RIGHT CURLY BRACKET}')
        ''')

    @skipIf(version_info < (3, 8), 'new in Python 3.8')
    def test_assign_expr(self):
        """Test PEP 572 assignment expressions are treated as usage / write."""
        self.flakes('''
        from foo import y
        print(x := y)
        print(x)
        ''')


class TestStringFormatting(TestCase):

    @skipIf(version_info < (3, 6), 'new in Python 3.6')
    def test_f_string_without_placeholders(self):
        self.flakes("f'foo'", m.FStringMissingPlaceholders)
        self.flakes('''
            f"""foo
            bar
            """
        ''', m.FStringMissingPlaceholders)
        self.flakes('''
            print(
                f'foo'
                f'bar'
            )
        ''', m.FStringMissingPlaceholders)
        # this is an "escaped placeholder" but not a placeholder
        self.flakes("f'{{}}'", m.FStringMissingPlaceholders)
        # ok: f-string with placeholders
        self.flakes('''
            x = 5
            print(f'{x}')
        ''')
        # ok: f-string with format specifiers
        self.flakes('''
            x = 'a' * 90
            print(f'{x:.8}')
        ''')
        # ok: f-string with multiple format specifiers
        self.flakes('''
            x = y = 5
            print(f'{x:>2} {y:>2}')
        ''')

    def test_invalid_dot_format_calls(self):
        self.flakes('''
            '{'.format(1)
        ''', m.StringDotFormatInvalidFormat)
        self.flakes('''
            '{} {1}'.format(1, 2)
        ''', m.StringDotFormatMixingAutomatic)
        self.flakes('''
            '{0} {}'.format(1, 2)
        ''', m.StringDotFormatMixingAutomatic)
        self.flakes('''
            '{}'.format(1, 2)
        ''', m.StringDotFormatExtraPositionalArguments)
        self.flakes('''
            '{}'.format(1, bar=2)
        ''', m.StringDotFormatExtraNamedArguments)
        self.flakes('''
            '{} {}'.format(1)
        ''', m.StringDotFormatMissingArgument)
        self.flakes('''
            '{2}'.format()
        ''', m.StringDotFormatMissingArgument)
        self.flakes('''
            '{bar}'.format()
        ''', m.StringDotFormatMissingArgument)
        # too much string recursion (placeholder-in-placeholder)
        self.flakes('''
            '{:{:{}}}'.format(1, 2, 3)
        ''', m.StringDotFormatInvalidFormat)
        # ok: dotted / bracketed names need to handle the param differently
        self.flakes("'{.__class__}'.format('')")
        self.flakes("'{foo[bar]}'.format(foo={'bar': 'barv'})")
        # ok: placeholder-placeholders
        self.flakes('''
            print('{:{}} {}'.format(1, 15, 2))
        ''')
        # ok: not a placeholder-placeholder
        self.flakes('''
            print('{:2}'.format(1))
        ''')
        # ok: not mixed automatic
        self.flakes('''
            '{foo}-{}'.format(1, foo=2)
        ''')
        # ok: we can't determine statically the format args
        self.flakes('''
            a = ()
            "{}".format(*a)
        ''')
        self.flakes('''
            k = {}
            "{foo}".format(**k)
        ''')

    def test_invalid_percent_format_calls(self):
        self.flakes('''
            '%(foo)' % {'foo': 'bar'}
        ''', m.PercentFormatInvalidFormat)
        self.flakes('''
            '%s %(foo)s' % {'foo': 'bar'}
        ''', m.PercentFormatMixedPositionalAndNamed)
        self.flakes('''
            '%(foo)s %s' % {'foo': 'bar'}
        ''', m.PercentFormatMixedPositionalAndNamed)
        self.flakes('''
            '%j' % (1,)
        ''', m.PercentFormatUnsupportedFormatCharacter)
        self.flakes('''
            '%s %s' % (1,)
        ''', m.PercentFormatPositionalCountMismatch)
        self.flakes('''
            '%s %s' % (1, 2, 3)
        ''', m.PercentFormatPositionalCountMismatch)
        self.flakes('''
            '%(bar)s' % {}
        ''', m.PercentFormatMissingArgument,)
        self.flakes('''
            '%(bar)s' % {'bar': 1, 'baz': 2}
        ''', m.PercentFormatExtraNamedArguments)
        self.flakes('''
            '%(bar)s' % (1, 2, 3)
        ''', m.PercentFormatExpectedMapping)
        self.flakes('''
            '%s %s' % {'k': 'v'}
        ''', m.PercentFormatExpectedSequence)
        self.flakes('''
            '%(bar)*s' % {'bar': 'baz'}
        ''', m.PercentFormatStarRequiresSequence)
        # ok: single %s with mapping
        self.flakes('''
            '%s' % {'foo': 'bar', 'baz': 'womp'}
        ''')
        # ok: does not cause a MemoryError (the strings aren't evaluated)
        self.flakes('''
            "%1000000000000f" % 1
        ''')
        # ok: %% should not count towards placeholder count
        self.flakes('''
            '%% %s %% %s' % (1, 2)
        ''')
        # ok: * consumes one positional argument
        self.flakes('''
            '%.*f' % (2, 1.1234)
            '%*.*f' % (5, 2, 3.1234)
        ''')

    @skipIf(version_info < (3, 5), 'new in Python 3.5')
    def test_ok_percent_format_cannot_determine_element_count(self):
        self.flakes('''
            a = []
            '%s %s' % [*a]
            '%s %s' % (*a,)
        ''')
        self.flakes('''
            k = {}
            '%(k)s' % {**k}
        ''')


class TestAsyncStatements(TestCase):

    @skipIf(version_info < (3, 5), 'new in Python 3.5')
    def test_asyncDef(self):
        self.flakes('''
        async def bar():
            return 42
        ''')

    @skipIf(version_info < (3, 5), 'new in Python 3.5')
    def test_asyncDefAwait(self):
        self.flakes('''
        async def read_data(db):
            await db.fetch('SELECT ...')
        ''')

    @skipIf(version_info < (3, 5), 'new in Python 3.5')
    def test_asyncDefUndefined(self):
        self.flakes('''
        async def bar():
            return foo()
        ''', m.UndefinedName)

    @skipIf(version_info < (3, 5), 'new in Python 3.5')
    def test_asyncFor(self):
        self.flakes('''
        async def read_data(db):
            output = []
            async for row in db.cursor():
                output.append(row)
            return output
        ''')

    @skipIf(version_info < (3, 5), 'new in Python 3.5')
    def test_asyncForUnderscoreLoopVar(self):
        self.flakes('''
        async def coro(it):
            async for _ in it:
                pass
        ''')

    @skipIf(version_info < (3, 5), 'new in Python 3.5')
    def test_loopControlInAsyncFor(self):
        self.flakes('''
        async def read_data(db):
            output = []
            async for row in db.cursor():
                if row[0] == 'skip':
                    continue
                output.append(row)
            return output
        ''')

        self.flakes('''
        async def read_data(db):
            output = []
            async for row in db.cursor():
                if row[0] == 'stop':
                    break
                output.append(row)
            return output
        ''')

    @skipIf(version_info < (3, 5), 'new in Python 3.5')
    def test_loopControlInAsyncForElse(self):
        self.flakes('''
        async def read_data(db):
            output = []
            async for row in db.cursor():
                output.append(row)
            else:
                continue
            return output
        ''', m.ContinueOutsideLoop)

        self.flakes('''
        async def read_data(db):
            output = []
            async for row in db.cursor():
                output.append(row)
            else:
                break
            return output
        ''', m.BreakOutsideLoop)

    @skipIf(version_info < (3, 5), 'new in Python 3.5')
    @skipIf(version_info > (3, 8), "Python <= 3.8 only")
    def test_continueInAsyncForFinally(self):
        self.flakes('''
        async def read_data(db):
            output = []
            async for row in db.cursor():
                try:
                    output.append(row)
                finally:
                    continue
            return output
        ''', m.ContinueInFinally)

    @skipIf(version_info < (3, 5), 'new in Python 3.5')
    def test_asyncWith(self):
        self.flakes('''
        async def commit(session, data):
            async with session.transaction():
                await session.update(data)
        ''')

    @skipIf(version_info < (3, 5), 'new in Python 3.5')
    def test_asyncWithItem(self):
        self.flakes('''
        async def commit(session, data):
            async with session.transaction() as trans:
                await trans.begin()
                ...
                await trans.end()
        ''')

    @skipIf(version_info < (3, 5), 'new in Python 3.5')
    def test_matmul(self):
        self.flakes('''
        def foo(a, b):
            return a @ b
        ''')

    @skipIf(version_info < (3, 6), 'new in Python 3.6')
    def test_formatstring(self):
        self.flakes('''
        hi = 'hi'
        mom = 'mom'
        f'{hi} {mom}'
        ''')

    def test_raise_notimplemented(self):
        self.flakes('''
        raise NotImplementedError("This is fine")
        ''')

        self.flakes('''
        raise NotImplementedError
        ''')

        self.flakes('''
        raise NotImplemented("This isn't gonna work")
        ''', m.RaiseNotImplemented)

        self.flakes('''
        raise NotImplemented
        ''', m.RaiseNotImplemented)


class TestIncompatiblePrintOperator(TestCase):
    """
    Tests for warning about invalid use of print function.
    """

    def test_valid_print(self):
        self.flakes('''
        print("Hello")
        ''')

    def test_invalid_print_when_imported_from_future(self):
        exc = self.flakes('''
        from __future__ import print_function
        import sys
        print >>sys.stderr, "Hello"
        ''', m.InvalidPrintSyntax).messages[0]

        self.assertEqual(exc.lineno, 4)
        self.assertEqual(exc.col, 0)

    def test_print_function_assignment(self):
        """
        A valid assignment, tested for catching false positives.
        """
        self.flakes('''
        from __future__ import print_function
        log = print
        log("Hello")
        ''')

    def test_print_in_lambda(self):
        self.flakes('''
        from __future__ import print_function
        a = lambda: print
        ''')

    def test_print_returned_in_function(self):
        self.flakes('''
        from __future__ import print_function
        def a():
            return print
        ''')

    def test_print_as_condition_test(self):
        self.flakes('''
        from __future__ import print_function
        if print: pass
        ''')
