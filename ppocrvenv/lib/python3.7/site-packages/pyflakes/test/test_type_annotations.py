"""
Tests for behaviour related to type annotations.
"""

from sys import version_info

from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf


class TestTypeAnnotations(TestCase):

    def test_typingOverload(self):
        """Allow intentional redefinitions via @typing.overload"""
        self.flakes("""
        import typing
        from typing import overload

        @overload
        def f(s):  # type: (None) -> None
            pass

        @overload
        def f(s):  # type: (int) -> int
            pass

        def f(s):
            return s

        @typing.overload
        def g(s):  # type: (None) -> None
            pass

        @typing.overload
        def g(s):  # type: (int) -> int
            pass

        def g(s):
            return s
        """)

    def test_typingExtensionsOverload(self):
        """Allow intentional redefinitions via @typing_extensions.overload"""
        self.flakes("""
        import typing_extensions
        from typing_extensions import overload

        @overload
        def f(s):  # type: (None) -> None
            pass

        @overload
        def f(s):  # type: (int) -> int
            pass

        def f(s):
            return s

        @typing_extensions.overload
        def g(s):  # type: (None) -> None
            pass

        @typing_extensions.overload
        def g(s):  # type: (int) -> int
            pass

        def g(s):
            return s
        """)

    @skipIf(version_info < (3, 5), 'new in Python 3.5')
    def test_typingOverloadAsync(self):
        """Allow intentional redefinitions via @typing.overload (async)"""
        self.flakes("""
        from typing import overload

        @overload
        async def f(s):  # type: (None) -> None
            pass

        @overload
        async def f(s):  # type: (int) -> int
            pass

        async def f(s):
            return s
        """)

    def test_overload_with_multiple_decorators(self):
        self.flakes("""
            from typing import overload
            dec = lambda f: f

            @dec
            @overload
            def f(x):  # type: (int) -> int
                pass

            @dec
            @overload
            def f(x):  # type: (str) -> str
                pass

            @dec
            def f(x): return x
       """)

    def test_overload_in_class(self):
        self.flakes("""
        from typing import overload

        class C:
            @overload
            def f(self, x):  # type: (int) -> int
                pass

            @overload
            def f(self, x):  # type: (str) -> str
                pass

            def f(self, x): return x
        """)

    def test_aliased_import(self):
        """Detect when typing is imported as another name"""
        self.flakes("""
        import typing as t

        @t.overload
        def f(s):  # type: (None) -> None
            pass

        @t.overload
        def f(s):  # type: (int) -> int
            pass

        def f(s):
            return s
        """)

    def test_not_a_typing_overload(self):
        """regression test for @typing.overload detection bug in 2.1.0"""
        self.flakes("""
            def foo(x):
                return x

            @foo
            def bar():
                pass

            def bar():
                pass
        """, m.RedefinedWhileUnused)

    @skipIf(version_info < (3, 6), 'new in Python 3.6')
    def test_variable_annotations(self):
        self.flakes('''
        name: str
        age: int
        ''')
        self.flakes('''
        name: str = 'Bob'
        age: int = 18
        ''')
        self.flakes('''
        class C:
            name: str
            age: int
        ''')
        self.flakes('''
        class C:
            name: str = 'Bob'
            age: int = 18
        ''')
        self.flakes('''
        def f():
            name: str
            age: int
        ''')
        self.flakes('''
        def f():
            name: str = 'Bob'
            age: int = 18
            foo: not_a_real_type = None
        ''', m.UnusedVariable, m.UnusedVariable, m.UnusedVariable, m.UndefinedName)
        self.flakes('''
        def f():
            name: str
            print(name)
        ''', m.UndefinedName)
        self.flakes('''
        from typing import Any
        def f():
            a: Any
        ''')
        self.flakes('''
        foo: not_a_real_type
        ''', m.UndefinedName)
        self.flakes('''
        foo: not_a_real_type = None
        ''', m.UndefinedName)
        self.flakes('''
        class C:
            foo: not_a_real_type
        ''', m.UndefinedName)
        self.flakes('''
        class C:
            foo: not_a_real_type = None
        ''', m.UndefinedName)
        self.flakes('''
        def f():
            class C:
                foo: not_a_real_type
        ''', m.UndefinedName)
        self.flakes('''
        def f():
            class C:
                foo: not_a_real_type = None
        ''', m.UndefinedName)
        self.flakes('''
        from foo import Bar
        bar: Bar
        ''')
        self.flakes('''
        from foo import Bar
        bar: 'Bar'
        ''')
        self.flakes('''
        import foo
        bar: foo.Bar
        ''')
        self.flakes('''
        import foo
        bar: 'foo.Bar'
        ''')
        self.flakes('''
        from foo import Bar
        def f(bar: Bar): pass
        ''')
        self.flakes('''
        from foo import Bar
        def f(bar: 'Bar'): pass
        ''')
        self.flakes('''
        from foo import Bar
        def f(bar) -> Bar: return bar
        ''')
        self.flakes('''
        from foo import Bar
        def f(bar) -> 'Bar': return bar
        ''')
        self.flakes('''
        bar: 'Bar'
        ''', m.UndefinedName)
        self.flakes('''
        bar: 'foo.Bar'
        ''', m.UndefinedName)
        self.flakes('''
        from foo import Bar
        bar: str
        ''', m.UnusedImport)
        self.flakes('''
        from foo import Bar
        def f(bar: str): pass
        ''', m.UnusedImport)
        self.flakes('''
        def f(a: A) -> A: pass
        class A: pass
        ''', m.UndefinedName, m.UndefinedName)
        self.flakes('''
        def f(a: 'A') -> 'A': return a
        class A: pass
        ''')
        self.flakes('''
        a: A
        class A: pass
        ''', m.UndefinedName)
        self.flakes('''
        a: 'A'
        class A: pass
        ''')
        self.flakes('''
        T: object
        def f(t: T): pass
        ''', m.UndefinedName)
        self.flakes('''
        T: object
        def g(t: 'T'): pass
        ''')
        self.flakes('''
        a: 'A B'
        ''', m.ForwardAnnotationSyntaxError)
        self.flakes('''
        a: 'A; B'
        ''', m.ForwardAnnotationSyntaxError)
        self.flakes('''
        a: '1 + 2'
        ''')
        self.flakes('''
        a: 'a: "A"'
        ''', m.ForwardAnnotationSyntaxError)

    @skipIf(version_info < (3, 6), 'new in Python 3.6')
    def test_annotating_an_import(self):
        self.flakes('''
            from a import b, c
            b: c
            print(b)
        ''')

    @skipIf(version_info < (3, 6), 'new in Python 3.6')
    def test_unused_annotation(self):
        # Unused annotations are fine in module and class scope
        self.flakes('''
        x: int
        class Cls:
            y: int
        ''')
        # TODO: this should print a UnusedVariable message
        self.flakes('''
        def f():
            x: int
        ''')
        # This should only print one UnusedVariable message
        self.flakes('''
        def f():
            x: int
            x = 3
        ''', m.UnusedVariable)

    @skipIf(version_info < (3, 5), 'new in Python 3.5')
    def test_annotated_async_def(self):
        self.flakes('''
        class c: pass
        async def func(c: c) -> None: pass
        ''')

    @skipIf(version_info < (3, 7), 'new in Python 3.7')
    def test_postponed_annotations(self):
        self.flakes('''
        from __future__ import annotations
        def f(a: A) -> A: pass
        class A:
            b: B
        class B: pass
        ''')

        self.flakes('''
        from __future__ import annotations
        def f(a: A) -> A: pass
        class A:
            b: Undefined
        class B: pass
        ''', m.UndefinedName)

        self.flakes('''
        from __future__ import annotations
        T: object
        def f(t: T): pass
        def g(t: 'T'): pass
        ''')

    @skipIf(version_info < (3, 6), 'new in Python 3.6')
    def test_type_annotation_clobbers_all(self):
        self.flakes('''\
        from typing import TYPE_CHECKING, List

        from y import z

        if not TYPE_CHECKING:
            __all__ = ("z",)
        else:
            __all__: List[str]
        ''')

    def test_typeCommentsMarkImportsAsUsed(self):
        self.flakes("""
        from mod import A, B, C, D, E, F, G


        def f(
            a,  # type: A
        ):
            # type: (...) -> B
            for b in a:  # type: C
                with b as c:  # type: D
                    d = c.x  # type: E
                    return d


        def g(x):  # type: (F) -> G
            return x.y
        """)

    def test_typeCommentsFullSignature(self):
        self.flakes("""
        from mod import A, B, C, D
        def f(a, b):
            # type: (A, B[C]) -> D
            return a + b
        """)

    def test_typeCommentsStarArgs(self):
        self.flakes("""
        from mod import A, B, C, D
        def f(a, *b, **c):
            # type: (A, *B, **C) -> D
            return a + b
        """)

    def test_typeCommentsFullSignatureWithDocstring(self):
        self.flakes('''
        from mod import A, B, C, D
        def f(a, b):
            # type: (A, B[C]) -> D
            """do the thing!"""
            return a + b
        ''')

    def test_typeCommentsAdditionalComment(self):
        self.flakes("""
        from mod import F

        x = 1 # type: F  # noqa
        """)

    def test_typeCommentsNoWhitespaceAnnotation(self):
        self.flakes("""
        from mod import F

        x = 1  #type:F
        """)

    def test_typeCommentsInvalidDoesNotMarkAsUsed(self):
        self.flakes("""
        from mod import F

        # type: F
        """, m.UnusedImport)

    def test_typeCommentsSyntaxError(self):
        self.flakes("""
        def f(x):  # type: (F[) -> None
            pass
        """, m.CommentAnnotationSyntaxError)

    def test_typeCommentsSyntaxErrorCorrectLine(self):
        checker = self.flakes("""\
        x = 1
        # type: definitely not a PEP 484 comment
        """, m.CommentAnnotationSyntaxError)
        self.assertEqual(checker.messages[0].lineno, 2)

    def test_typeCommentsAssignedToPreviousNode(self):
        # This test demonstrates an issue in the implementation which
        # associates the type comment with a node above it, however the type
        # comment isn't valid according to mypy.  If an improved approach
        # which can detect these "invalid" type comments is implemented, this
        # test should be removed / improved to assert that new check.
        self.flakes("""
        from mod import F
        x = 1
        # type: F
        """)

    def test_typeIgnore(self):
        self.flakes("""
        a = 0  # type: ignore
        b = 0  # type: ignore[excuse]
        c = 0  # type: ignore=excuse
        d = 0  # type: ignore [excuse]
        e = 0  # type: ignore whatever
        """)

    def test_typeIgnoreBogus(self):
        self.flakes("""
        x = 1  # type: ignored
        """, m.UndefinedName)

    def test_typeIgnoreBogusUnicode(self):
        error = (m.CommentAnnotationSyntaxError if version_info < (3,)
                 else m.UndefinedName)
        self.flakes("""
        x = 2  # type: ignore\xc3
        """, error)

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_return_annotation_is_class_scope_variable(self):
        self.flakes("""
        from typing import TypeVar
        class Test:
            Y = TypeVar('Y')

            def t(self, x: Y) -> Y:
                return x
        """)

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_return_annotation_is_function_body_variable(self):
        self.flakes("""
        class Test:
            def t(self) -> Y:
                Y = 2
                return Y
        """, m.UndefinedName)

    @skipIf(version_info < (3, 8), 'new in Python 3.8')
    def test_positional_only_argument_annotations(self):
        self.flakes("""
        from x import C

        def f(c: C, /): ...
        """)

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_partially_quoted_type_annotation(self):
        self.flakes("""
        from queue import Queue
        from typing import Optional

        def f() -> Optional['Queue[str]']:
            return None
        """)

    def test_partially_quoted_type_assignment(self):
        self.flakes("""
        from queue import Queue
        from typing import Optional

        MaybeQueue = Optional['Queue[str]']
        """)

    def test_nested_partially_quoted_type_assignment(self):
        self.flakes("""
        from queue import Queue
        from typing import Callable

        Func = Callable[['Queue[str]'], None]
        """)

    def test_quoted_type_cast(self):
        self.flakes("""
        from typing import cast, Optional

        maybe_int = cast('Optional[int]', 42)
        """)

    def test_type_cast_literal_str_to_str(self):
        # Checks that our handling of quoted type annotations in the first
        # argument to `cast` doesn't cause issues when (only) the _second_
        # argument is a literal str which looks a bit like a type annotation.
        self.flakes("""
        from typing import cast

        a_string = cast(str, 'Optional[int]')
        """)

    def test_quoted_type_cast_renamed_import(self):
        self.flakes("""
        from typing import cast as tsac, Optional as Maybe

        maybe_int = tsac('Maybe[int]', 42)
        """)

    def test_quoted_TypeVar_constraints(self):
        self.flakes("""
        from typing import TypeVar, Optional

        T = TypeVar('T', 'str', 'Optional[int]', bytes)
        """)

    def test_quoted_TypeVar_bound(self):
        self.flakes("""
        from typing import TypeVar, Optional, List

        T = TypeVar('T', bound='Optional[int]')
        S = TypeVar('S', int, bound='List[int]')
        """)

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_literal_type_typing(self):
        self.flakes("""
        from typing import Literal

        def f(x: Literal['some string']) -> None:
            return None
        """)

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_literal_type_typing_extensions(self):
        self.flakes("""
        from typing_extensions import Literal

        def f(x: Literal['some string']) -> None:
            return None
        """)

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_annotated_type_typing_missing_forward_type(self):
        self.flakes("""
        from typing import Annotated

        def f(x: Annotated['integer']) -> None:
            return None
        """, m.UndefinedName)

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_annotated_type_typing_missing_forward_type_multiple_args(self):
        self.flakes("""
        from typing import Annotated

        def f(x: Annotated['integer', 1]) -> None:
            return None
        """, m.UndefinedName)

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_annotated_type_typing_with_string_args(self):
        self.flakes("""
        from typing import Annotated

        def f(x: Annotated[int, '> 0']) -> None:
            return None
        """)

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_annotated_type_typing_with_string_args_in_union(self):
        self.flakes("""
        from typing import Annotated, Union

        def f(x: Union[Annotated['int', '>0'], 'integer']) -> None:
            return None
        """, m.UndefinedName)

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_literal_type_some_other_module(self):
        """err on the side of false-negatives for types named Literal"""
        self.flakes("""
        from my_module import compat
        from my_module.compat import Literal

        def f(x: compat.Literal['some string']) -> None:
            return None
        def g(x: Literal['some string']) -> None:
            return None
        """)

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_literal_union_type_typing(self):
        self.flakes("""
        from typing import Literal

        def f(x: Literal['some string', 'foo bar']) -> None:
            return None
        """)

    @skipIf(version_info < (3,), 'new in Python 3')
    def test_deferred_twice_annotation(self):
        self.flakes("""
            from queue import Queue
            from typing import Optional


            def f() -> "Optional['Queue[str]']":
                return None
        """)

    @skipIf(version_info < (3, 7), 'new in Python 3.7')
    def test_partial_string_annotations_with_future_annotations(self):
        self.flakes("""
            from __future__ import annotations

            from queue import Queue
            from typing import Optional


            def f() -> Optional['Queue[str]']:
                return None
        """)

    def test_idomiatic_typing_guards(self):
        # typing.TYPE_CHECKING: python3.5.3+
        self.flakes("""
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                from t import T

            def f():  # type: () -> T
                pass
        """)
        # False: the old, more-compatible approach
        self.flakes("""
            if False:
                from t import T

            def f():  # type: () -> T
                pass
        """)
        # some choose to assign a constant and do it that way
        self.flakes("""
            MYPY = False

            if MYPY:
                from t import T

            def f():  # type: () -> T
                pass
        """)

    def test_typing_guard_for_protocol(self):
        self.flakes("""
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                from typing import Protocol
            else:
                Protocol = object

            class C(Protocol):
                def f():  # type: () -> int
                    pass
        """)

    def test_typednames_correct_forward_ref(self):
        self.flakes("""
            from typing import TypedDict, List, NamedTuple

            List[TypedDict("x", {})]
            List[TypedDict("x", x=int)]
            List[NamedTuple("a", a=int)]
            List[NamedTuple("a", [("a", int)])]
        """)
        self.flakes("""
            from typing import TypedDict, List, NamedTuple, TypeVar

            List[TypedDict("x", {"x": "Y"})]
            List[TypedDict("x", x="Y")]
            List[NamedTuple("a", [("a", "Y")])]
            List[NamedTuple("a", a="Y")]
            List[TypedDict("x", {"x": List["a"]})]
            List[TypeVar("A", bound="C")]
            List[TypeVar("A", List["C"])]
        """, *[m.UndefinedName]*7)
        self.flakes("""
            from typing import NamedTuple, TypeVar, cast
            from t import A, B, C, D, E

            NamedTuple("A", [("a", A["C"])])
            TypeVar("A", bound=A["B"])
            TypeVar("A", A["D"])
            cast(A["E"], [])
        """)

    @skipIf(version_info < (3, 6), 'new in Python 3.6')
    def test_namedtypes_classes(self):
        self.flakes("""
            from typing import TypedDict, NamedTuple
            class X(TypedDict):
                y: TypedDict("z", {"zz":int})

            class Y(NamedTuple):
                y: NamedTuple("v", [("vv", int)])
        """)
