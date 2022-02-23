
import pytest
import pickle
from numpy.testing import assert_equal
from scipy._lib._bunch import _make_tuple_bunch


# `Result` is defined at the top level of the module so it can be
# used to test pickling.
Result = _make_tuple_bunch('Result', ['x', 'y', 'z'], ['w', 'beta'])


class TestMakeTupleBunch:

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Tests with Result
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def setup(self):
        # Set up an instance of Result.
        self.result = Result(x=1, y=2, z=3, w=99, beta=0.5)

    def test_attribute_access(self):
        assert_equal(self.result.x, 1)
        assert_equal(self.result.y, 2)
        assert_equal(self.result.z, 3)
        assert_equal(self.result.w, 99)
        assert_equal(self.result.beta, 0.5)

    def test_indexing(self):
        assert_equal(self.result[0], 1)
        assert_equal(self.result[1], 2)
        assert_equal(self.result[2], 3)
        assert_equal(self.result[-1], 3)
        with pytest.raises(IndexError, match='index out of range'):
            self.result[3]

    def test_unpacking(self):
        x0, y0, z0 = self.result
        assert_equal((x0, y0, z0), (1, 2, 3))
        assert_equal(self.result, (1, 2, 3))

    def test_slice(self):
        assert_equal(self.result[1:], (2, 3))
        assert_equal(self.result[::2], (1, 3))
        assert_equal(self.result[::-1], (3, 2, 1))

    def test_len(self):
        assert_equal(len(self.result), 3)

    def test_repr(self):
        s = repr(self.result)
        assert_equal(s, 'Result(x=1, y=2, z=3, w=99, beta=0.5)')

    def test_hash(self):
        assert_equal(hash(self.result), hash((1, 2, 3)))

    def test_pickle(self):
        s = pickle.dumps(self.result)
        obj = pickle.loads(s)
        assert isinstance(obj, Result)
        assert_equal(obj.x, self.result.x)
        assert_equal(obj.y, self.result.y)
        assert_equal(obj.z, self.result.z)
        assert_equal(obj.w, self.result.w)
        assert_equal(obj.beta, self.result.beta)

    def test_read_only_existing(self):
        with pytest.raises(AttributeError, match="can't set attribute"):
            self.result.x = -1

    def test_read_only_new(self):
        with pytest.raises(AttributeError, match="can't set attribute"):
            self.result.plate_of_shrimp = "lattice of coincidence"

    def test_constructor_missing_parameter(self):
        with pytest.raises(TypeError, match='missing'):
            # `w` is missing.
            Result(x=1, y=2, z=3, beta=0.75)

    def test_constructor_incorrect_parameter(self):
        with pytest.raises(TypeError, match='unexpected'):
            # `foo` is not an existing field.
            Result(x=1, y=2, z=3, w=123, beta=0.75, foo=999)

    def test_module(self):
        m = 'scipy._lib.tests.test_bunch'
        assert_equal(Result.__module__, m)
        assert_equal(self.result.__module__, m)

    def test_extra_fields_per_instance(self):
        # This test exists to ensure that instances of the same class
        # store their own values for the extra fields. That is, the values
        # are stored per instance and not in the class.
        result1 = Result(x=1, y=2, z=3, w=-1, beta=0.0)
        result2 = Result(x=4, y=5, z=6, w=99, beta=1.0)
        assert_equal(result1.w, -1)
        assert_equal(result1.beta, 0.0)
        # The rest of these checks aren't essential, but let's check
        # them anyway.
        assert_equal(result1[:], (1, 2, 3))
        assert_equal(result2.w, 99)
        assert_equal(result2.beta, 1.0)
        assert_equal(result2[:], (4, 5, 6))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Other tests
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_extra_field_names_is_optional(self):
        Square = _make_tuple_bunch('Square', ['width', 'height'])
        sq = Square(width=1, height=2)
        assert_equal(sq.width, 1)
        assert_equal(sq.height, 2)
        s = repr(sq)
        assert_equal(s, 'Square(width=1, height=2)')

    def test_tuple_like(self):
        Tup = _make_tuple_bunch('Tup', ['a', 'b'])
        tu = Tup(a=1, b=2)
        assert isinstance(tu, tuple)
        assert isinstance(tu + (1,), tuple)

    def test_explicit_module(self):
        m = 'some.module.name'
        Foo = _make_tuple_bunch('Foo', ['x'], ['a', 'b'], module=m)
        foo = Foo(x=1, a=355, b=113)
        assert_equal(Foo.__module__, m)
        assert_equal(foo.__module__, m)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Argument validation
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    @pytest.mark.parametrize('args', [('123', ['a'], ['b']),
                                      ('Foo', ['-3'], ['x']),
                                      ('Foo', ['a'], ['+-*/'])])
    def test_identifiers_not_allowed(self, args):
        with pytest.raises(ValueError, match='identifiers'):
            _make_tuple_bunch(*args)

    @pytest.mark.parametrize('args', [('Foo', ['a', 'b', 'a'], ['x']),
                                      ('Foo', ['a', 'b'], ['b', 'x'])])
    def test_repeated_field_names(self, args):
        with pytest.raises(ValueError, match='Duplicate'):
            _make_tuple_bunch(*args)

    @pytest.mark.parametrize('args', [('Foo', ['_a'], ['x']),
                                      ('Foo', ['a'], ['_x'])])
    def test_leading_underscore_not_allowed(self, args):
        with pytest.raises(ValueError, match='underscore'):
            _make_tuple_bunch(*args)

    @pytest.mark.parametrize('args', [('Foo', ['def'], ['x']),
                                      ('Foo', ['a'], ['or']),
                                      ('and', ['a'], ['x'])])
    def test_keyword_not_allowed_in_fields(self, args):
        with pytest.raises(ValueError, match='keyword'):
            _make_tuple_bunch(*args)

    def test_at_least_one_field_name_required(self):
        with pytest.raises(ValueError, match='at least one name'):
            _make_tuple_bunch('Qwerty', [], ['a', 'b'])
