from pandas import Categorical
import pandas._testing as tm


class TestCategoricalSubclassing:
    def test_constructor(self):
        sc = tm.SubclassedCategorical(["a", "b", "c"])
        assert isinstance(sc, tm.SubclassedCategorical)
        tm.assert_categorical_equal(sc, Categorical(["a", "b", "c"]))

    def test_from_codes(self):
        sc = tm.SubclassedCategorical.from_codes([1, 0, 2], ["a", "b", "c"])
        assert isinstance(sc, tm.SubclassedCategorical)
        exp = Categorical.from_codes([1, 0, 2], ["a", "b", "c"])
        tm.assert_categorical_equal(sc, exp)

    def test_map(self):
        sc = tm.SubclassedCategorical(["a", "b", "c"])
        res = sc.map(lambda x: x.upper())
        assert isinstance(res, tm.SubclassedCategorical)
        exp = Categorical(["A", "B", "C"])
        tm.assert_categorical_equal(res, exp)
