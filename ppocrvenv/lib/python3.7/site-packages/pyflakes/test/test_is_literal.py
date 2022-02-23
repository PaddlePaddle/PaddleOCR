from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase


class Test(TestCase):
    def test_is_str(self):
        self.flakes("""
        x = 'foo'
        if x is 'foo':
            pass
        """, IsLiteral)

    def test_is_bytes(self):
        self.flakes("""
        x = b'foo'
        if x is b'foo':
            pass
        """, IsLiteral)

    def test_is_unicode(self):
        self.flakes("""
        x = u'foo'
        if x is u'foo':
            pass
        """, IsLiteral)

    def test_is_int(self):
        self.flakes("""
        x = 10
        if x is 10:
            pass
        """, IsLiteral)

    def test_is_true(self):
        self.flakes("""
        x = True
        if x is True:
            pass
        """)

    def test_is_false(self):
        self.flakes("""
        x = False
        if x is False:
            pass
        """)

    def test_is_not_str(self):
        self.flakes("""
        x = 'foo'
        if x is not 'foo':
            pass
        """, IsLiteral)

    def test_is_not_bytes(self):
        self.flakes("""
        x = b'foo'
        if x is not b'foo':
            pass
        """, IsLiteral)

    def test_is_not_unicode(self):
        self.flakes("""
        x = u'foo'
        if x is not u'foo':
            pass
        """, IsLiteral)

    def test_is_not_int(self):
        self.flakes("""
        x = 10
        if x is not 10:
            pass
        """, IsLiteral)

    def test_is_not_true(self):
        self.flakes("""
        x = True
        if x is not True:
            pass
        """)

    def test_is_not_false(self):
        self.flakes("""
        x = False
        if x is not False:
            pass
        """)

    def test_left_is_str(self):
        self.flakes("""
        x = 'foo'
        if 'foo' is x:
            pass
        """, IsLiteral)

    def test_left_is_bytes(self):
        self.flakes("""
        x = b'foo'
        if b'foo' is x:
            pass
        """, IsLiteral)

    def test_left_is_unicode(self):
        self.flakes("""
        x = u'foo'
        if u'foo' is x:
            pass
        """, IsLiteral)

    def test_left_is_int(self):
        self.flakes("""
        x = 10
        if 10 is x:
            pass
        """, IsLiteral)

    def test_left_is_true(self):
        self.flakes("""
        x = True
        if True is x:
            pass
        """)

    def test_left_is_false(self):
        self.flakes("""
        x = False
        if False is x:
            pass
        """)

    def test_left_is_not_str(self):
        self.flakes("""
        x = 'foo'
        if 'foo' is not x:
            pass
        """, IsLiteral)

    def test_left_is_not_bytes(self):
        self.flakes("""
        x = b'foo'
        if b'foo' is not x:
            pass
        """, IsLiteral)

    def test_left_is_not_unicode(self):
        self.flakes("""
        x = u'foo'
        if u'foo' is not x:
            pass
        """, IsLiteral)

    def test_left_is_not_int(self):
        self.flakes("""
        x = 10
        if 10 is not x:
            pass
        """, IsLiteral)

    def test_left_is_not_true(self):
        self.flakes("""
        x = True
        if True is not x:
            pass
        """)

    def test_left_is_not_false(self):
        self.flakes("""
        x = False
        if False is not x:
            pass
        """)

    def test_chained_operators_is_true(self):
        self.flakes("""
        x = 5
        if x is True < 4:
            pass
        """)

    def test_chained_operators_is_str(self):
        self.flakes("""
        x = 5
        if x is 'foo' < 4:
            pass
        """, IsLiteral)

    def test_chained_operators_is_true_end(self):
        self.flakes("""
        x = 5
        if 4 < x is True:
            pass
        """)

    def test_chained_operators_is_str_end(self):
        self.flakes("""
        x = 5
        if 4 < x is 'foo':
            pass
        """, IsLiteral)

    def test_is_tuple_constant(self):
        self.flakes('''\
            x = 5
            if x is ():
                pass
        ''', IsLiteral)

    def test_is_tuple_constant_containing_constants(self):
        self.flakes('''\
            x = 5
            if x is (1, '2', True, (1.5, ())):
                pass
        ''', IsLiteral)

    def test_is_tuple_containing_variables_ok(self):
        # a bit nonsensical, but does not trigger a SyntaxWarning
        self.flakes('''\
            x = 5
            if x is (x,):
                pass
        ''')
