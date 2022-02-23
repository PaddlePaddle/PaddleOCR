from sys import version_info

from pyflakes import messages as m
from pyflakes.checker import (
    FutureImportation,
    Importation,
    ImportationFrom,
    StarImportation,
    SubmoduleImportation,
)
from pyflakes.test.harness import TestCase, skip, skipIf


class TestImportationObject(TestCase):

    def test_import_basic(self):
        binding = Importation('a', None, 'a')
        assert binding.source_statement == 'import a'
        assert str(binding) == 'a'

    def test_import_as(self):
        binding = Importation('c', None, 'a')
        assert binding.source_statement == 'import a as c'
        assert str(binding) == 'a as c'

    def test_import_submodule(self):
        binding = SubmoduleImportation('a.b', None)
        assert binding.source_statement == 'import a.b'
        assert str(binding) == 'a.b'

    def test_import_submodule_as(self):
        # A submodule import with an as clause is not a SubmoduleImportation
        binding = Importation('c', None, 'a.b')
        assert binding.source_statement == 'import a.b as c'
        assert str(binding) == 'a.b as c'

    def test_import_submodule_as_source_name(self):
        binding = Importation('a', None, 'a.b')
        assert binding.source_statement == 'import a.b as a'
        assert str(binding) == 'a.b as a'

    def test_importfrom_relative(self):
        binding = ImportationFrom('a', None, '.', 'a')
        assert binding.source_statement == 'from . import a'
        assert str(binding) == '.a'

    def test_importfrom_relative_parent(self):
        binding = ImportationFrom('a', None, '..', 'a')
        assert binding.source_statement == 'from .. import a'
        assert str(binding) == '..a'

    def test_importfrom_relative_with_module(self):
        binding = ImportationFrom('b', None, '..a', 'b')
        assert binding.source_statement == 'from ..a import b'
        assert str(binding) == '..a.b'

    def test_importfrom_relative_with_module_as(self):
        binding = ImportationFrom('c', None, '..a', 'b')
        assert binding.source_statement == 'from ..a import b as c'
        assert str(binding) == '..a.b as c'

    def test_importfrom_member(self):
        binding = ImportationFrom('b', None, 'a', 'b')
        assert binding.source_statement == 'from a import b'
        assert str(binding) == 'a.b'

    def test_importfrom_submodule_member(self):
        binding = ImportationFrom('c', None, 'a.b', 'c')
        assert binding.source_statement == 'from a.b import c'
        assert str(binding) == 'a.b.c'

    def test_importfrom_member_as(self):
        binding = ImportationFrom('c', None, 'a', 'b')
        assert binding.source_statement == 'from a import b as c'
        assert str(binding) == 'a.b as c'

    def test_importfrom_submodule_member_as(self):
        binding = ImportationFrom('d', None, 'a.b', 'c')
        assert binding.source_statement == 'from a.b import c as d'
        assert str(binding) == 'a.b.c as d'

    def test_importfrom_star(self):
        binding = StarImportation('a.b', None)
        assert binding.source_statement == 'from a.b import *'
        assert str(binding) == 'a.b.*'

    def test_importfrom_star_relative(self):
        binding = StarImportation('.b', None)
        assert binding.source_statement == 'from .b import *'
        assert str(binding) == '.b.*'

    def test_importfrom_future(self):
        binding = FutureImportation('print_function', None, None)
        assert binding.source_statement == 'from __future__ import print_function'
        assert str(binding) == '__future__.print_function'

    def test_unusedImport_underscore(self):
        """
        The magic underscore var should be reported as unused when used as an
        import alias.
        """
        self.flakes('import fu as _', m.UnusedImport)


class Test(TestCase):

    def test_unusedImport(self):
        self.flakes('import fu, bar', m.UnusedImport, m.UnusedImport)
        self.flakes('from baz import fu, bar', m.UnusedImport, m.UnusedImport)

    def test_unusedImport_relative(self):
        self.flakes('from . import fu', m.UnusedImport)
        self.flakes('from . import fu as baz', m.UnusedImport)
        self.flakes('from .. import fu', m.UnusedImport)
        self.flakes('from ... import fu', m.UnusedImport)
        self.flakes('from .. import fu as baz', m.UnusedImport)
        self.flakes('from .bar import fu', m.UnusedImport)
        self.flakes('from ..bar import fu', m.UnusedImport)
        self.flakes('from ...bar import fu', m.UnusedImport)
        self.flakes('from ...bar import fu as baz', m.UnusedImport)

        checker = self.flakes('from . import fu', m.UnusedImport)

        error = checker.messages[0]
        assert error.message == '%r imported but unused'
        assert error.message_args == ('.fu', )

        checker = self.flakes('from . import fu as baz', m.UnusedImport)

        error = checker.messages[0]
        assert error.message == '%r imported but unused'
        assert error.message_args == ('.fu as baz', )

    def test_aliasedImport(self):
        self.flakes('import fu as FU, bar as FU',
                    m.RedefinedWhileUnused, m.UnusedImport)
        self.flakes('from moo import fu as FU, bar as FU',
                    m.RedefinedWhileUnused, m.UnusedImport)

    def test_aliasedImportShadowModule(self):
        """Imported aliases can shadow the source of the import."""
        self.flakes('from moo import fu as moo; moo')
        self.flakes('import fu as fu; fu')
        self.flakes('import fu.bar as fu; fu')

    def test_usedImport(self):
        self.flakes('import fu; print(fu)')
        self.flakes('from baz import fu; print(fu)')
        self.flakes('import fu; del fu')

    def test_usedImport_relative(self):
        self.flakes('from . import fu; assert fu')
        self.flakes('from .bar import fu; assert fu')
        self.flakes('from .. import fu; assert fu')
        self.flakes('from ..bar import fu as baz; assert baz')

    def test_redefinedWhileUnused(self):
        self.flakes('import fu; fu = 3', m.RedefinedWhileUnused)
        self.flakes('import fu; fu, bar = 3', m.RedefinedWhileUnused)
        self.flakes('import fu; [fu, bar] = 3', m.RedefinedWhileUnused)

    def test_redefinedIf(self):
        """
        Test that importing a module twice within an if
        block does raise a warning.
        """
        self.flakes('''
        i = 2
        if i==1:
            import os
            import os
        os.path''', m.RedefinedWhileUnused)

    def test_redefinedIfElse(self):
        """
        Test that importing a module twice in if
        and else blocks does not raise a warning.
        """
        self.flakes('''
        i = 2
        if i==1:
            import os
        else:
            import os
        os.path''')

    def test_redefinedTry(self):
        """
        Test that importing a module twice in a try block
        does raise a warning.
        """
        self.flakes('''
        try:
            import os
            import os
        except:
            pass
        os.path''', m.RedefinedWhileUnused)

    def test_redefinedTryExcept(self):
        """
        Test that importing a module twice in a try
        and except block does not raise a warning.
        """
        self.flakes('''
        try:
            import os
        except:
            import os
        os.path''')

    def test_redefinedTryNested(self):
        """
        Test that importing a module twice using a nested
        try/except and if blocks does not issue a warning.
        """
        self.flakes('''
        try:
            if True:
                if True:
                    import os
        except:
            import os
        os.path''')

    def test_redefinedTryExceptMulti(self):
        self.flakes("""
        try:
            from aa import mixer
        except AttributeError:
            from bb import mixer
        except RuntimeError:
            from cc import mixer
        except:
            from dd import mixer
        mixer(123)
        """)

    def test_redefinedTryElse(self):
        self.flakes("""
        try:
            from aa import mixer
        except ImportError:
            pass
        else:
            from bb import mixer
        mixer(123)
        """, m.RedefinedWhileUnused)

    def test_redefinedTryExceptElse(self):
        self.flakes("""
        try:
            import funca
        except ImportError:
            from bb import funca
            from bb import funcb
        else:
            from bbb import funcb
        print(funca, funcb)
        """)

    def test_redefinedTryExceptFinally(self):
        self.flakes("""
        try:
            from aa import a
        except ImportError:
            from bb import a
        finally:
            a = 42
        print(a)
        """)

    def test_redefinedTryExceptElseFinally(self):
        self.flakes("""
        try:
            import b
        except ImportError:
            b = Ellipsis
            from bb import a
        else:
            from aa import a
        finally:
            a = 42
        print(a, b)
        """)

    def test_redefinedByFunction(self):
        self.flakes('''
        import fu
        def fu():
            pass
        ''', m.RedefinedWhileUnused)

    def test_redefinedInNestedFunction(self):
        """
        Test that shadowing a global name with a nested function definition
        generates a warning.
        """
        self.flakes('''
        import fu
        def bar():
            def baz():
                def fu():
                    pass
        ''', m.RedefinedWhileUnused, m.UnusedImport)

    def test_redefinedInNestedFunctionTwice(self):
        """
        Test that shadowing a global name with a nested function definition
        generates a warning.
        """
        self.flakes('''
        import fu
        def bar():
            import fu
            def baz():
                def fu():
                    pass
        ''',
                    m.RedefinedWhileUnused, m.RedefinedWhileUnused,
                    m.UnusedImport, m.UnusedImport)

    def test_redefinedButUsedLater(self):
        """
        Test that a global import which is redefined locally,
        but used later in another scope does not generate a warning.
        """
        self.flakes('''
        import unittest, transport

        class GetTransportTestCase(unittest.TestCase):
            def test_get_transport(self):
                transport = 'transport'
                self.assertIsNotNone(transport)

        class TestTransportMethodArgs(unittest.TestCase):
            def test_send_defaults(self):
                transport.Transport()
        ''')

    def test_redefinedByClass(self):
        self.flakes('''
        import fu
        class fu:
            pass
        ''', m.RedefinedWhileUnused)

    def test_redefinedBySubclass(self):
        """
        If an imported name is redefined by a class statement which also uses
        that name in the bases list, no warning is emitted.
        """
        self.flakes('''
        from fu import bar
        class bar(bar):
            pass
        ''')

    def test_redefinedInClass(self):
        """
        Test that shadowing a global with a class attribute does not produce a
        warning.
        """
        self.flakes('''
        import fu
        class bar:
            fu = 1
        print(fu)
        ''')

    def test_importInClass(self):
        """
        Test that import within class is a locally scoped attribute.
        """
        self.flakes('''
        class bar:
            import fu
        ''')

        self.flakes('''
        class bar:
            import fu

        fu
        ''', m.UndefinedName)

    def test_usedInFunction(self):
        self.flakes('''
        import fu
        def fun():
            print(fu)
        ''')

    def test_shadowedByParameter(self):
        self.flakes('''
        import fu
        def fun(fu):
            print(fu)
        ''', m.UnusedImport, m.RedefinedWhileUnused)

        self.flakes('''
        import fu
        def fun(fu):
            print(fu)
        print(fu)
        ''')

    def test_newAssignment(self):
        self.flakes('fu = None')

    def test_usedInGetattr(self):
        self.flakes('import fu; fu.bar.baz')
        self.flakes('import fu; "bar".fu.baz', m.UnusedImport)

    def test_usedInSlice(self):
        self.flakes('import fu; print(fu.bar[1:])')

    def test_usedInIfBody(self):
        self.flakes('''
        import fu
        if True: print(fu)
        ''')

    def test_usedInIfConditional(self):
        self.flakes('''
        import fu
        if fu: pass
        ''')

    def test_usedInElifConditional(self):
        self.flakes('''
        import fu
        if False: pass
        elif fu: pass
        ''')

    def test_usedInElse(self):
        self.flakes('''
        import fu
        if False: pass
        else: print(fu)
        ''')

    def test_usedInCall(self):
        self.flakes('import fu; fu.bar()')

    def test_usedInClass(self):
        self.flakes('''
        import fu
        class bar:
            bar = fu
        ''')

    def test_usedInClassBase(self):
        self.flakes('''
        import fu
        class bar(object, fu.baz):
            pass
        ''')

    def test_notUsedInNestedScope(self):
        self.flakes('''
        import fu
        def bleh():
            pass
        print(fu)
        ''')

    def test_usedInFor(self):
        self.flakes('''
        import fu
        for bar in range(9):
            print(fu)
        ''')

    def test_usedInForElse(self):
        self.flakes('''
        import fu
        for bar in range(10):
            pass
        else:
            print(fu)
        ''')

    def test_redefinedByFor(self):
        self.flakes('''
        import fu
        for fu in range(2):
            pass
        ''', m.ImportShadowedByLoopVar)

    def test_shadowedByFor(self):
        """
        Test that shadowing a global name with a for loop variable generates a
        warning.
        """
        self.flakes('''
        import fu
        fu.bar()
        for fu in ():
            pass
        ''', m.ImportShadowedByLoopVar)

    def test_shadowedByForDeep(self):
        """
        Test that shadowing a global name with a for loop variable nested in a
        tuple unpack generates a warning.
        """
        self.flakes('''
        import fu
        fu.bar()
        for (x, y, z, (a, b, c, (fu,))) in ():
            pass
        ''', m.ImportShadowedByLoopVar)
        # Same with a list instead of a tuple
        self.flakes('''
        import fu
        fu.bar()
        for [x, y, z, (a, b, c, (fu,))] in ():
            pass
        ''', m.ImportShadowedByLoopVar)

    def test_usedInReturn(self):
        self.flakes('''
        import fu
        def fun():
            return fu
        ''')

    def test_usedInOperators(self):
        self.flakes('import fu; 3 + fu.bar')
        self.flakes('import fu; 3 % fu.bar')
        self.flakes('import fu; 3 - fu.bar')
        self.flakes('import fu; 3 * fu.bar')
        self.flakes('import fu; 3 ** fu.bar')
        self.flakes('import fu; 3 / fu.bar')
        self.flakes('import fu; 3 // fu.bar')
        self.flakes('import fu; -fu.bar')
        self.flakes('import fu; ~fu.bar')
        self.flakes('import fu; 1 == fu.bar')
        self.flakes('import fu; 1 | fu.bar')
        self.flakes('import fu; 1 & fu.bar')
        self.flakes('import fu; 1 ^ fu.bar')
        self.flakes('import fu; 1 >> fu.bar')
        self.flakes('import fu; 1 << fu.bar')

    def test_usedInAssert(self):
        self.flakes('import fu; assert fu.bar')

    def test_usedInSubscript(self):
        self.flakes('import fu; fu.bar[1]')

    def test_usedInLogic(self):
        self.flakes('import fu; fu and False')
        self.flakes('import fu; fu or False')
        self.flakes('import fu; not fu.bar')

    def test_usedInList(self):
        self.flakes('import fu; [fu]')

    def test_usedInTuple(self):
        self.flakes('import fu; (fu,)')

    def test_usedInTry(self):
        self.flakes('''
        import fu
        try: fu
        except: pass
        ''')

    def test_usedInExcept(self):
        self.flakes('''
        import fu
        try: fu
        except: pass
        ''')

    def test_redefinedByExcept(self):
        expected = [m.RedefinedWhileUnused]
        if version_info >= (3,):
            # The exc variable is unused inside the exception handler.
            expected.append(m.UnusedVariable)
        self.flakes('''
        import fu
        try: pass
        except Exception as fu: pass
        ''', *expected)

    def test_usedInRaise(self):
        self.flakes('''
        import fu
        raise fu.bar
        ''')

    def test_usedInYield(self):
        self.flakes('''
        import fu
        def gen():
            yield fu
        ''')

    def test_usedInDict(self):
        self.flakes('import fu; {fu:None}')
        self.flakes('import fu; {1:fu}')

    def test_usedInParameterDefault(self):
        self.flakes('''
        import fu
        def f(bar=fu):
            pass
        ''')

    def test_usedInAttributeAssign(self):
        self.flakes('import fu; fu.bar = 1')

    def test_usedInKeywordArg(self):
        self.flakes('import fu; fu.bar(stuff=fu)')

    def test_usedInAssignment(self):
        self.flakes('import fu; bar=fu')
        self.flakes('import fu; n=0; n+=fu')

    def test_usedInListComp(self):
        self.flakes('import fu; [fu for _ in range(1)]')
        self.flakes('import fu; [1 for _ in range(1) if fu]')

    @skipIf(version_info >= (3,),
            'in Python 3 list comprehensions execute in a separate scope')
    def test_redefinedByListComp(self):
        self.flakes('import fu; [1 for fu in range(1)]',
                    m.RedefinedInListComp)

    def test_usedInTryFinally(self):
        self.flakes('''
        import fu
        try: pass
        finally: fu
        ''')

        self.flakes('''
        import fu
        try: fu
        finally: pass
        ''')

    def test_usedInWhile(self):
        self.flakes('''
        import fu
        while 0:
            fu
        ''')

        self.flakes('''
        import fu
        while fu: pass
        ''')

    def test_usedInGlobal(self):
        """
        A 'global' statement shadowing an unused import should not prevent it
        from being reported.
        """
        self.flakes('''
        import fu
        def f(): global fu
        ''', m.UnusedImport)

    def test_usedAndGlobal(self):
        """
        A 'global' statement shadowing a used import should not cause it to be
        reported as unused.
        """
        self.flakes('''
            import foo
            def f(): global foo
            def g(): foo.is_used()
        ''')

    def test_assignedToGlobal(self):
        """
        Binding an import to a declared global should not cause it to be
        reported as unused.
        """
        self.flakes('''
            def f(): global foo; import foo
            def g(): foo.is_used()
        ''')

    @skipIf(version_info >= (3,), 'deprecated syntax')
    def test_usedInBackquote(self):
        self.flakes('import fu; `fu`')

    def test_usedInExec(self):
        if version_info < (3,):
            exec_stmt = 'exec "print 1" in fu.bar'
        else:
            exec_stmt = 'exec("print(1)", fu.bar)'
        self.flakes('import fu; %s' % exec_stmt)

    def test_usedInLambda(self):
        self.flakes('import fu; lambda: fu')

    def test_shadowedByLambda(self):
        self.flakes('import fu; lambda fu: fu',
                    m.UnusedImport, m.RedefinedWhileUnused)
        self.flakes('import fu; lambda fu: fu\nfu()')

    def test_usedInSliceObj(self):
        self.flakes('import fu; "meow"[::fu]')

    def test_unusedInNestedScope(self):
        self.flakes('''
        def bar():
            import fu
        fu
        ''', m.UnusedImport, m.UndefinedName)

    def test_methodsDontUseClassScope(self):
        self.flakes('''
        class bar:
            import fu
            def fun(self):
                fu
        ''', m.UndefinedName)

    def test_nestedFunctionsNestScope(self):
        self.flakes('''
        def a():
            def b():
                fu
            import fu
        ''')

    def test_nestedClassAndFunctionScope(self):
        self.flakes('''
        def a():
            import fu
            class b:
                def c(self):
                    print(fu)
        ''')

    def test_importStar(self):
        """Use of import * at module level is reported."""
        self.flakes('from fu import *', m.ImportStarUsed, m.UnusedImport)
        self.flakes('''
        try:
            from fu import *
        except:
            pass
        ''', m.ImportStarUsed, m.UnusedImport)

        checker = self.flakes('from fu import *',
                              m.ImportStarUsed, m.UnusedImport)

        error = checker.messages[0]
        assert error.message.startswith("'from %s import *' used; unable ")
        assert error.message_args == ('fu', )

        error = checker.messages[1]
        assert error.message == '%r imported but unused'
        assert error.message_args == ('fu.*', )

    def test_importStar_relative(self):
        """Use of import * from a relative import is reported."""
        self.flakes('from .fu import *', m.ImportStarUsed, m.UnusedImport)
        self.flakes('''
        try:
            from .fu import *
        except:
            pass
        ''', m.ImportStarUsed, m.UnusedImport)

        checker = self.flakes('from .fu import *',
                              m.ImportStarUsed, m.UnusedImport)

        error = checker.messages[0]
        assert error.message.startswith("'from %s import *' used; unable ")
        assert error.message_args == ('.fu', )

        error = checker.messages[1]
        assert error.message == '%r imported but unused'
        assert error.message_args == ('.fu.*', )

        checker = self.flakes('from .. import *',
                              m.ImportStarUsed, m.UnusedImport)

        error = checker.messages[0]
        assert error.message.startswith("'from %s import *' used; unable ")
        assert error.message_args == ('..', )

        error = checker.messages[1]
        assert error.message == '%r imported but unused'
        assert error.message_args == ('from .. import *', )

    @skipIf(version_info < (3,),
            'import * below module level is a warning on Python 2')
    def test_localImportStar(self):
        """import * is only allowed at module level."""
        self.flakes('''
        def a():
            from fu import *
        ''', m.ImportStarNotPermitted)
        self.flakes('''
        class a:
            from fu import *
        ''', m.ImportStarNotPermitted)

        checker = self.flakes('''
        class a:
            from .. import *
        ''', m.ImportStarNotPermitted)
        error = checker.messages[0]
        assert error.message == "'from %s import *' only allowed at module level"
        assert error.message_args == ('..', )

    @skipIf(version_info > (3,),
            'import * below module level is an error on Python 3')
    def test_importStarNested(self):
        """All star imports are marked as used by an undefined variable."""
        self.flakes('''
        from fu import *
        def f():
            from bar import *
            x
        ''', m.ImportStarUsed, m.ImportStarUsed, m.ImportStarUsage)

    def test_packageImport(self):
        """
        If a dotted name is imported and used, no warning is reported.
        """
        self.flakes('''
        import fu.bar
        fu.bar
        ''')

    def test_unusedPackageImport(self):
        """
        If a dotted name is imported and not used, an unused import warning is
        reported.
        """
        self.flakes('import fu.bar', m.UnusedImport)

    def test_duplicateSubmoduleImport(self):
        """
        If a submodule of a package is imported twice, an unused import warning
        and a redefined while unused warning are reported.
        """
        self.flakes('''
        import fu.bar, fu.bar
        fu.bar
        ''', m.RedefinedWhileUnused)
        self.flakes('''
        import fu.bar
        import fu.bar
        fu.bar
        ''', m.RedefinedWhileUnused)

    def test_differentSubmoduleImport(self):
        """
        If two different submodules of a package are imported, no duplicate
        import warning is reported for the package.
        """
        self.flakes('''
        import fu.bar, fu.baz
        fu.bar, fu.baz
        ''')
        self.flakes('''
        import fu.bar
        import fu.baz
        fu.bar, fu.baz
        ''')

    def test_used_package_with_submodule_import(self):
        """
        Usage of package marks submodule imports as used.
        """
        self.flakes('''
        import fu
        import fu.bar
        fu.x
        ''')

        self.flakes('''
        import fu.bar
        import fu
        fu.x
        ''')

    def test_used_package_with_submodule_import_of_alias(self):
        """
        Usage of package by alias marks submodule imports as used.
        """
        self.flakes('''
        import foo as f
        import foo.bar
        f.bar.do_something()
        ''')

        self.flakes('''
        import foo as f
        import foo.bar.blah
        f.bar.blah.do_something()
        ''')

    def test_unused_package_with_submodule_import(self):
        """
        When a package and its submodule are imported, only report once.
        """
        checker = self.flakes('''
        import fu
        import fu.bar
        ''', m.UnusedImport)
        error = checker.messages[0]
        assert error.message == '%r imported but unused'
        assert error.message_args == ('fu.bar', )
        assert error.lineno == 5 if self.withDoctest else 3

    def test_assignRHSFirst(self):
        self.flakes('import fu; fu = fu')
        self.flakes('import fu; fu, bar = fu')
        self.flakes('import fu; [fu, bar] = fu')
        self.flakes('import fu; fu += fu')

    def test_tryingMultipleImports(self):
        self.flakes('''
        try:
            import fu
        except ImportError:
            import bar as fu
        fu
        ''')

    def test_nonGlobalDoesNotRedefine(self):
        self.flakes('''
        import fu
        def a():
            fu = 3
            return fu
        fu
        ''')

    def test_functionsRunLater(self):
        self.flakes('''
        def a():
            fu
        import fu
        ''')

    def test_functionNamesAreBoundNow(self):
        self.flakes('''
        import fu
        def fu():
            fu
        fu
        ''', m.RedefinedWhileUnused)

    def test_ignoreNonImportRedefinitions(self):
        self.flakes('a = 1; a = 2')

    @skip("todo")
    def test_importingForImportError(self):
        self.flakes('''
        try:
            import fu
        except ImportError:
            pass
        ''')

    def test_importedInClass(self):
        """Imports in class scope can be used through self."""
        self.flakes('''
        class c:
            import i
            def __init__(self):
                self.i
        ''')

    def test_importUsedInMethodDefinition(self):
        """
        Method named 'foo' with default args referring to module named 'foo'.
        """
        self.flakes('''
        import foo

        class Thing(object):
            def foo(self, parser=foo.parse_foo):
                pass
        ''')

    def test_futureImport(self):
        """__future__ is special."""
        self.flakes('from __future__ import division')
        self.flakes('''
        "docstring is allowed before future import"
        from __future__ import division
        ''')

    def test_futureImportFirst(self):
        """
        __future__ imports must come before anything else.
        """
        self.flakes('''
        x = 5
        from __future__ import division
        ''', m.LateFutureImport)
        self.flakes('''
        from foo import bar
        from __future__ import division
        bar
        ''', m.LateFutureImport)

    def test_futureImportUsed(self):
        """__future__ is special, but names are injected in the namespace."""
        self.flakes('''
        from __future__ import division
        from __future__ import print_function

        assert print_function is not division
        ''')

    def test_futureImportUndefined(self):
        """Importing undefined names from __future__ fails."""
        self.flakes('''
        from __future__ import print_statement
        ''', m.FutureFeatureNotDefined)

    def test_futureImportStar(self):
        """Importing '*' from __future__ fails."""
        self.flakes('''
        from __future__ import *
        ''', m.FutureFeatureNotDefined)


class TestSpecialAll(TestCase):
    """
    Tests for suppression of unused import warnings by C{__all__}.
    """
    def test_ignoredInFunction(self):
        """
        An C{__all__} definition does not suppress unused import warnings in a
        function scope.
        """
        self.flakes('''
        def foo():
            import bar
            __all__ = ["bar"]
        ''', m.UnusedImport, m.UnusedVariable)

    def test_ignoredInClass(self):
        """
        An C{__all__} definition in a class does not suppress unused import warnings.
        """
        self.flakes('''
        import bar
        class foo:
            __all__ = ["bar"]
        ''', m.UnusedImport)

    def test_warningSuppressed(self):
        """
        If a name is imported and unused but is named in C{__all__}, no warning
        is reported.
        """
        self.flakes('''
        import foo
        __all__ = ["foo"]
        ''')
        self.flakes('''
        import foo
        __all__ = ("foo",)
        ''')

    def test_augmentedAssignment(self):
        """
        The C{__all__} variable is defined incrementally.
        """
        self.flakes('''
        import a
        import c
        __all__ = ['a']
        __all__ += ['b']
        if 1 < 3:
            __all__ += ['c', 'd']
        ''', m.UndefinedExport, m.UndefinedExport)

    def test_list_concatenation_assignment(self):
        """
        The C{__all__} variable is defined through list concatenation.
        """
        self.flakes('''
        import sys
        __all__ = ['a'] + ['b'] + ['c']
        ''', m.UndefinedExport, m.UndefinedExport, m.UndefinedExport, m.UnusedImport)

    def test_tuple_concatenation_assignment(self):
        """
        The C{__all__} variable is defined through tuple concatenation.
        """
        self.flakes('''
        import sys
        __all__ = ('a',) + ('b',) + ('c',)
        ''', m.UndefinedExport, m.UndefinedExport, m.UndefinedExport, m.UnusedImport)

    def test_all_with_attributes(self):
        self.flakes('''
        from foo import bar
        __all__ = [bar.__name__]
        ''')

    def test_all_with_names(self):
        # not actually valid, but shouldn't produce a crash
        self.flakes('''
        from foo import bar
        __all__ = [bar]
        ''')

    def test_all_with_attributes_added(self):
        self.flakes('''
        from foo import bar
        from bar import baz
        __all__ = [bar.__name__] + [baz.__name__]
        ''')

    def test_all_mixed_attributes_and_strings(self):
        self.flakes('''
        from foo import bar
        from foo import baz
        __all__ = ['bar', baz.__name__]
        ''')

    def test_unboundExported(self):
        """
        If C{__all__} includes a name which is not bound, a warning is emitted.
        """
        self.flakes('''
        __all__ = ["foo"]
        ''', m.UndefinedExport)

        # Skip this in __init__.py though, since the rules there are a little
        # different.
        for filename in ["foo/__init__.py", "__init__.py"]:
            self.flakes('''
            __all__ = ["foo"]
            ''', filename=filename)

    def test_importStarExported(self):
        """
        Report undefined if import * is used
        """
        self.flakes('''
        from math import *
        __all__ = ['sin', 'cos']
        csc(1)
        ''', m.ImportStarUsed, m.ImportStarUsage, m.ImportStarUsage, m.ImportStarUsage)

    def test_importStarNotExported(self):
        """Report unused import when not needed to satisfy __all__."""
        self.flakes('''
        from foolib import *
        a = 1
        __all__ = ['a']
        ''', m.ImportStarUsed, m.UnusedImport)

    def test_usedInGenExp(self):
        """
        Using a global in a generator expression results in no warnings.
        """
        self.flakes('import fu; (fu for _ in range(1))')
        self.flakes('import fu; (1 for _ in range(1) if fu)')

    def test_redefinedByGenExp(self):
        """
        Re-using a global name as the loop variable for a generator
        expression results in a redefinition warning.
        """
        self.flakes('import fu; (1 for fu in range(1))',
                    m.RedefinedWhileUnused, m.UnusedImport)

    def test_usedAsDecorator(self):
        """
        Using a global name in a decorator statement results in no warnings,
        but using an undefined name in a decorator statement results in an
        undefined name warning.
        """
        self.flakes('''
        from interior import decorate
        @decorate
        def f():
            return "hello"
        ''')

        self.flakes('''
        from interior import decorate
        @decorate('value')
        def f():
            return "hello"
        ''')

        self.flakes('''
        @decorate
        def f():
            return "hello"
        ''', m.UndefinedName)

    def test_usedAsClassDecorator(self):
        """
        Using an imported name as a class decorator results in no warnings,
        but using an undefined name as a class decorator results in an
        undefined name warning.
        """
        self.flakes('''
        from interior import decorate
        @decorate
        class foo:
            pass
        ''')

        self.flakes('''
        from interior import decorate
        @decorate("foo")
        class bar:
            pass
        ''')

        self.flakes('''
        @decorate
        class foo:
            pass
        ''', m.UndefinedName)
