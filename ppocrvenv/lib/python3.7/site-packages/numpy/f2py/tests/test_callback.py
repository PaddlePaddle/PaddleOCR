import math
import textwrap
import sys
import pytest
import threading
import traceback
import time
import random

import numpy as np
from numpy.testing import assert_, assert_equal, IS_PYPY
from . import util


class TestF77Callback(util.F2PyTest):
    code = """
       subroutine t(fun,a)
       integer a
cf2py  intent(out) a
       external fun
       call fun(a)
       end

       subroutine func(a)
cf2py  intent(in,out) a
       integer a
       a = a + 11
       end

       subroutine func0(a)
cf2py  intent(out) a
       integer a
       a = 11
       end

       subroutine t2(a)
cf2py  intent(callback) fun
       integer a
cf2py  intent(out) a
       external fun
       call fun(a)
       end

       subroutine string_callback(callback, a)
       external callback
       double precision callback
       double precision a
       character*1 r
cf2py  intent(out) a
       r = 'r'
       a = callback(r)
       end

       subroutine string_callback_array(callback, cu, lencu, a)
       external callback
       integer callback
       integer lencu
       character*8 cu(lencu)
       integer a
cf2py  intent(out) a

       a = callback(cu, lencu)
       end

       subroutine hidden_callback(a, r)
       external global_f
cf2py  intent(callback, hide) global_f
       integer a, r, global_f
cf2py  intent(out) r
       r = global_f(a)
       end

       subroutine hidden_callback2(a, r)
       external global_f
       integer a, r, global_f
cf2py  intent(out) r
       r = global_f(a)
       end
    """

    @pytest.mark.parametrize('name', 't,t2'.split(','))
    def test_all(self, name):
        self.check_function(name)

    @pytest.mark.xfail(IS_PYPY,
                       reason="PyPy cannot modify tp_doc after PyType_Ready")
    def test_docstring(self):
        expected = textwrap.dedent("""\
        a = t(fun,[fun_extra_args])

        Wrapper for ``t``.

        Parameters
        ----------
        fun : call-back function

        Other Parameters
        ----------------
        fun_extra_args : input tuple, optional
            Default: ()

        Returns
        -------
        a : int

        Notes
        -----
        Call-back functions::

          def fun(): return a
          Return objects:
            a : int
        """)
        assert_equal(self.module.t.__doc__, expected)

    def check_function(self, name):
        t = getattr(self.module, name)
        r = t(lambda: 4)
        assert_(r == 4, repr(r))
        r = t(lambda a: 5, fun_extra_args=(6,))
        assert_(r == 5, repr(r))
        r = t(lambda a: a, fun_extra_args=(6,))
        assert_(r == 6, repr(r))
        r = t(lambda a: 5 + a, fun_extra_args=(7,))
        assert_(r == 12, repr(r))
        r = t(lambda a: math.degrees(a), fun_extra_args=(math.pi,))
        assert_(r == 180, repr(r))
        r = t(math.degrees, fun_extra_args=(math.pi,))
        assert_(r == 180, repr(r))

        r = t(self.module.func, fun_extra_args=(6,))
        assert_(r == 17, repr(r))
        r = t(self.module.func0)
        assert_(r == 11, repr(r))
        r = t(self.module.func0._cpointer)
        assert_(r == 11, repr(r))

        class A:

            def __call__(self):
                return 7

            def mth(self):
                return 9
        a = A()
        r = t(a)
        assert_(r == 7, repr(r))
        r = t(a.mth)
        assert_(r == 9, repr(r))

    @pytest.mark.skipif(sys.platform=='win32',
                        reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_string_callback(self):

        def callback(code):
            if code == 'r':
                return 0
            else:
                return 1

        f = getattr(self.module, 'string_callback')
        r = f(callback)
        assert_(r == 0, repr(r))

    @pytest.mark.skipif(sys.platform=='win32',
                        reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_string_callback_array(self):
        # See gh-10027
        cu = np.zeros((1, 8), 'S1')

        def callback(cu, lencu):
            if cu.shape != (lencu, 8):
                return 1
            if cu.dtype != 'S1':
                return 2
            if not np.all(cu == b''):
                return 3
            return 0

        f = getattr(self.module, 'string_callback_array')
        res = f(callback, cu, len(cu))
        assert_(res == 0, repr(res))

    def test_threadsafety(self):
        # Segfaults if the callback handling is not threadsafe

        errors = []

        def cb():
            # Sleep here to make it more likely for another thread
            # to call their callback at the same time.
            time.sleep(1e-3)

            # Check reentrancy
            r = self.module.t(lambda: 123)
            assert_(r == 123)

            return 42

        def runner(name):
            try:
                for j in range(50):
                    r = self.module.t(cb)
                    assert_(r == 42)
                    self.check_function(name)
            except Exception:
                errors.append(traceback.format_exc())

        threads = [threading.Thread(target=runner, args=(arg,))
                   for arg in ("t", "t2") for n in range(20)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        errors = "\n\n".join(errors)
        if errors:
            raise AssertionError(errors)

    def test_hidden_callback(self):
        try:
            self.module.hidden_callback(2)
        except Exception as msg:
            assert_(str(msg).startswith('Callback global_f not defined'))

        try:
            self.module.hidden_callback2(2)
        except Exception as msg:
            assert_(str(msg).startswith('cb: Callback global_f not defined'))

        self.module.global_f = lambda x: x + 1
        r = self.module.hidden_callback(2)
        assert_(r == 3)

        self.module.global_f = lambda x: x + 2
        r = self.module.hidden_callback(2)
        assert_(r == 4)

        del self.module.global_f
        try:
            self.module.hidden_callback(2)
        except Exception as msg:
            assert_(str(msg).startswith('Callback global_f not defined'))

        self.module.global_f = lambda x=0: x + 3
        r = self.module.hidden_callback(2)
        assert_(r == 5)

        # reproducer of gh18341
        r = self.module.hidden_callback2(2)
        assert_(r == 3)


class TestF77CallbackPythonTLS(TestF77Callback):
    """
    Callback tests using Python thread-local storage instead of
    compiler-provided
    """
    options = ["-DF2PY_USE_PYTHON_TLS"]


class TestF90Callback(util.F2PyTest):

    suffix = '.f90'

    code = textwrap.dedent(
        """
        function gh17797(f, y) result(r)
          external f
          integer(8) :: r, f
          integer(8), dimension(:) :: y
          r = f(0)
          r = r + sum(y)
        end function gh17797
        """)

    def test_gh17797(self):

        def incr(x):
            return x + 123

        y = np.array([1, 2, 3], dtype=np.int64)
        r = self.module.gh17797(incr, y)
        assert r == 123 + 1 + 2 + 3


class TestGH18335(util.F2PyTest):
    """The reproduction of the reported issue requires specific input that
    extensions may break the issue conditions, so the reproducer is
    implemented as a separate test class. Do not extend this test with
    other tests!
    """

    suffix = '.f90'

    code = textwrap.dedent(
        """
        ! When gh18335_workaround is defined as an extension,
        ! the issue cannot be reproduced.
        !subroutine gh18335_workaround(f, y)
        !  implicit none
        !  external f
        !  integer(kind=1) :: y(1)
        !  call f(y)
        !end subroutine gh18335_workaround

        function gh18335(f) result (r)
          implicit none
          external f
          integer(kind=1) :: y(1), r
          y(1) = 123
          call f(y)
          r = y(1)
        end function gh18335
        """)

    def test_gh18335(self):

        def foo(x):
            x[0] += 1

        y = np.array([1, 2, 3], dtype=np.int8)
        r = self.module.gh18335(foo)
        assert r == 123 + 1
