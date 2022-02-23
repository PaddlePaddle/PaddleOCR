from sys import version_info

from pyflakes.test.harness import TestCase, skipIf


@skipIf(version_info < (3, 10), "Python >= 3.10 only")
class TestMatch(TestCase):
    def test_match_bindings(self):
        self.flakes('''
            def f():
                x = 1
                match x:
                    case 1 as y:
                        print(f'matched as {y}')
        ''')
        self.flakes('''
            def f():
                x = [1, 2, 3]
                match x:
                    case [1, y, 3]:
                        print(f'matched {y}')
        ''')
        self.flakes('''
            def f():
                x = {'foo': 1}
                match x:
                    case {'foo': y}:
                        print(f'matched {y}')
        ''')

    def test_match_pattern_matched_class(self):
        self.flakes('''
            from a import B

            match 1:
                case B(x=1) as y:
                    print(f'matched {y}')
        ''')
        self.flakes('''
            from a import B

            match 1:
                case B(a, x=z) as y:
                    print(f'matched {y} {a} {z}')
        ''')

    def test_match_placeholder(self):
        self.flakes('''
            def f():
                match 1:
                    case _:
                        print('catchall!')
        ''')

    def test_match_singleton(self):
        self.flakes('''
            match 1:
                case True:
                    print('true')
        ''')

    def test_match_or_pattern(self):
        self.flakes('''
            match 1:
                case 1 | 2:
                    print('one or two')
        ''')

    def test_match_star(self):
        self.flakes('''
            x = [1, 2, 3]
            match x:
                case [1, *y]:
                    print(f'captured: {y}')
        ''')

    def test_match_double_star(self):
        self.flakes('''
            x = {'foo': 'bar', 'baz': 'womp'}
            match x:
                case {'foo': k1, **rest}:
                    print(f'{k1=} {rest=}')
        ''')
