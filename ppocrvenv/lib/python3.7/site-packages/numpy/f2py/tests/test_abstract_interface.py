import textwrap
from . import util
from numpy.f2py import crackfortran


class TestAbstractInterface(util.F2PyTest):
    suffix = '.f90'

    skip = ['add1', 'add2']

    code = textwrap.dedent("""
        module ops_module

          abstract interface
            subroutine op(x, y, z)
              integer, intent(in) :: x, y
              integer, intent(out) :: z
            end subroutine
          end interface

        contains

          subroutine foo(x, y, r1, r2)
            integer, intent(in) :: x, y
            integer, intent(out) :: r1, r2
            procedure (op) add1, add2
            procedure (op), pointer::p
            p=>add1
            call p(x, y, r1)
            p=>add2
            call p(x, y, r2)
          end subroutine
        end module

        subroutine add1(x, y, z)
          integer, intent(in) :: x, y
          integer, intent(out) :: z
          z = x + y
        end subroutine

        subroutine add2(x, y, z)
          integer, intent(in) :: x, y
          integer, intent(out) :: z
          z = x + 2 * y
        end subroutine
        """)

    def test_abstract_interface(self):
        assert self.module.ops_module.foo(3, 5) == (8, 13)

    def test_parse_abstract_interface(self, tmp_path):
        # Test gh18403
        f_path = tmp_path / "gh18403_mod.f90"
        with f_path.open('w') as ff:
            ff.write(textwrap.dedent("""\
                module test
                  abstract interface
                    subroutine foo()
                    end subroutine
                  end interface
                end module test
                """))
        mod = crackfortran.crackfortran([str(f_path)])
        assert len(mod) == 1
        assert len(mod[0]['body']) == 1
        assert mod[0]['body'][0]['block'] == 'abstract interface'
