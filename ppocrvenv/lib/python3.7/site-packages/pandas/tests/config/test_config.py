import warnings

import pytest

from pandas._config import config as cf
from pandas._config.config import OptionError

import pandas as pd


class TestConfig:
    @classmethod
    def setup_class(cls):
        from copy import deepcopy

        cls.cf = cf
        cls.gc = deepcopy(getattr(cls.cf, "_global_config"))
        cls.do = deepcopy(getattr(cls.cf, "_deprecated_options"))
        cls.ro = deepcopy(getattr(cls.cf, "_registered_options"))

    def setup_method(self, method):
        setattr(self.cf, "_global_config", {})
        setattr(self.cf, "options", self.cf.DictWrapper(self.cf._global_config))
        setattr(self.cf, "_deprecated_options", {})
        setattr(self.cf, "_registered_options", {})

        # Our test fixture in conftest.py sets "chained_assignment"
        # to "raise" only after all test methods have been setup.
        # However, after this setup, there is no longer any
        # "chained_assignment" option, so re-register it.
        self.cf.register_option("chained_assignment", "raise")

    def teardown_method(self, method):
        setattr(self.cf, "_global_config", self.gc)
        setattr(self.cf, "_deprecated_options", self.do)
        setattr(self.cf, "_registered_options", self.ro)

    def test_api(self):

        # the pandas object exposes the user API
        assert hasattr(pd, "get_option")
        assert hasattr(pd, "set_option")
        assert hasattr(pd, "reset_option")
        assert hasattr(pd, "describe_option")

    def test_is_one_of_factory(self):
        v = self.cf.is_one_of_factory([None, 12])

        v(12)
        v(None)
        msg = r"Value must be one of None\|12"
        with pytest.raises(ValueError, match=msg):
            v(1.1)

    def test_register_option(self):
        self.cf.register_option("a", 1, "doc")

        # can't register an already registered option
        msg = "Option 'a' has already been registered"
        with pytest.raises(OptionError, match=msg):
            self.cf.register_option("a", 1, "doc")

        # can't register an already registered option
        msg = "Path prefix to option 'a' is already an option"
        with pytest.raises(OptionError, match=msg):
            self.cf.register_option("a.b.c.d1", 1, "doc")
        with pytest.raises(OptionError, match=msg):
            self.cf.register_option("a.b.c.d2", 1, "doc")

        # no python keywords
        msg = "for is a python keyword"
        with pytest.raises(ValueError, match=msg):
            self.cf.register_option("for", 0)
        with pytest.raises(ValueError, match=msg):
            self.cf.register_option("a.for.b", 0)
        # must be valid identifier (ensure attribute access works)
        msg = "oh my goddess! is not a valid identifier"
        with pytest.raises(ValueError, match=msg):
            self.cf.register_option("Oh my Goddess!", 0)

        # we can register options several levels deep
        # without predefining the intermediate steps
        # and we can define differently named options
        # in the same namespace
        self.cf.register_option("k.b.c.d1", 1, "doc")
        self.cf.register_option("k.b.c.d2", 1, "doc")

    def test_describe_option(self):
        self.cf.register_option("a", 1, "doc")
        self.cf.register_option("b", 1, "doc2")
        self.cf.deprecate_option("b")

        self.cf.register_option("c.d.e1", 1, "doc3")
        self.cf.register_option("c.d.e2", 1, "doc4")
        self.cf.register_option("f", 1)
        self.cf.register_option("g.h", 1)
        self.cf.register_option("k", 2)
        self.cf.deprecate_option("g.h", rkey="k")
        self.cf.register_option("l", "foo")

        # non-existent keys raise KeyError
        msg = r"No such keys\(s\)"
        with pytest.raises(OptionError, match=msg):
            self.cf.describe_option("no.such.key")

        # we can get the description for any key we registered
        assert "doc" in self.cf.describe_option("a", _print_desc=False)
        assert "doc2" in self.cf.describe_option("b", _print_desc=False)
        assert "precated" in self.cf.describe_option("b", _print_desc=False)
        assert "doc3" in self.cf.describe_option("c.d.e1", _print_desc=False)
        assert "doc4" in self.cf.describe_option("c.d.e2", _print_desc=False)

        # if no doc is specified we get a default message
        # saying "description not available"
        assert "available" in self.cf.describe_option("f", _print_desc=False)
        assert "available" in self.cf.describe_option("g.h", _print_desc=False)
        assert "precated" in self.cf.describe_option("g.h", _print_desc=False)
        assert "k" in self.cf.describe_option("g.h", _print_desc=False)

        # default is reported
        assert "foo" in self.cf.describe_option("l", _print_desc=False)
        # current value is reported
        assert "bar" not in self.cf.describe_option("l", _print_desc=False)
        self.cf.set_option("l", "bar")
        assert "bar" in self.cf.describe_option("l", _print_desc=False)

    def test_case_insensitive(self):
        self.cf.register_option("KanBAN", 1, "doc")

        assert "doc" in self.cf.describe_option("kanbaN", _print_desc=False)
        assert self.cf.get_option("kanBaN") == 1
        self.cf.set_option("KanBan", 2)
        assert self.cf.get_option("kAnBaN") == 2

        # gets of non-existent keys fail
        msg = r"No such keys\(s\): 'no_such_option'"
        with pytest.raises(OptionError, match=msg):
            self.cf.get_option("no_such_option")
        self.cf.deprecate_option("KanBan")

        assert self.cf._is_deprecated("kAnBaN")

    def test_get_option(self):
        self.cf.register_option("a", 1, "doc")
        self.cf.register_option("b.c", "hullo", "doc2")
        self.cf.register_option("b.b", None, "doc2")

        # gets of existing keys succeed
        assert self.cf.get_option("a") == 1
        assert self.cf.get_option("b.c") == "hullo"
        assert self.cf.get_option("b.b") is None

        # gets of non-existent keys fail
        msg = r"No such keys\(s\): 'no_such_option'"
        with pytest.raises(OptionError, match=msg):
            self.cf.get_option("no_such_option")

    def test_set_option(self):
        self.cf.register_option("a", 1, "doc")
        self.cf.register_option("b.c", "hullo", "doc2")
        self.cf.register_option("b.b", None, "doc2")

        assert self.cf.get_option("a") == 1
        assert self.cf.get_option("b.c") == "hullo"
        assert self.cf.get_option("b.b") is None

        self.cf.set_option("a", 2)
        self.cf.set_option("b.c", "wurld")
        self.cf.set_option("b.b", 1.1)

        assert self.cf.get_option("a") == 2
        assert self.cf.get_option("b.c") == "wurld"
        assert self.cf.get_option("b.b") == 1.1

        msg = r"No such keys\(s\): 'no.such.key'"
        with pytest.raises(OptionError, match=msg):
            self.cf.set_option("no.such.key", None)

    def test_set_option_empty_args(self):
        msg = "Must provide an even number of non-keyword arguments"
        with pytest.raises(ValueError, match=msg):
            self.cf.set_option()

    def test_set_option_uneven_args(self):
        msg = "Must provide an even number of non-keyword arguments"
        with pytest.raises(ValueError, match=msg):
            self.cf.set_option("a.b", 2, "b.c")

    def test_set_option_invalid_single_argument_type(self):
        msg = "Must provide an even number of non-keyword arguments"
        with pytest.raises(ValueError, match=msg):
            self.cf.set_option(2)

    def test_set_option_multiple(self):
        self.cf.register_option("a", 1, "doc")
        self.cf.register_option("b.c", "hullo", "doc2")
        self.cf.register_option("b.b", None, "doc2")

        assert self.cf.get_option("a") == 1
        assert self.cf.get_option("b.c") == "hullo"
        assert self.cf.get_option("b.b") is None

        self.cf.set_option("a", "2", "b.c", None, "b.b", 10.0)

        assert self.cf.get_option("a") == "2"
        assert self.cf.get_option("b.c") is None
        assert self.cf.get_option("b.b") == 10.0

    def test_validation(self):
        self.cf.register_option("a", 1, "doc", validator=self.cf.is_int)
        self.cf.register_option("d", 1, "doc", validator=self.cf.is_nonnegative_int)
        self.cf.register_option("b.c", "hullo", "doc2", validator=self.cf.is_text)

        msg = "Value must have type '<class 'int'>'"
        with pytest.raises(ValueError, match=msg):
            self.cf.register_option("a.b.c.d2", "NO", "doc", validator=self.cf.is_int)

        self.cf.set_option("a", 2)  # int is_int
        self.cf.set_option("b.c", "wurld")  # str is_str
        self.cf.set_option("d", 2)
        self.cf.set_option("d", None)  # non-negative int can be None

        # None not is_int
        with pytest.raises(ValueError, match=msg):
            self.cf.set_option("a", None)
        with pytest.raises(ValueError, match=msg):
            self.cf.set_option("a", "ab")

        msg = "Value must be a nonnegative integer or None"
        with pytest.raises(ValueError, match=msg):
            self.cf.register_option(
                "a.b.c.d3", "NO", "doc", validator=self.cf.is_nonnegative_int
            )
        with pytest.raises(ValueError, match=msg):
            self.cf.register_option(
                "a.b.c.d3", -2, "doc", validator=self.cf.is_nonnegative_int
            )

        msg = r"Value must be an instance of <class 'str'>\|<class 'bytes'>"
        with pytest.raises(ValueError, match=msg):
            self.cf.set_option("b.c", 1)

        validator = self.cf.is_one_of_factory([None, self.cf.is_callable])
        self.cf.register_option("b", lambda: None, "doc", validator=validator)
        self.cf.set_option("b", "%.1f".format)  # Formatter is callable
        self.cf.set_option("b", None)  # Formatter is none (default)
        with pytest.raises(ValueError, match="Value must be a callable"):
            self.cf.set_option("b", "%.1f")

    def test_reset_option(self):
        self.cf.register_option("a", 1, "doc", validator=self.cf.is_int)
        self.cf.register_option("b.c", "hullo", "doc2", validator=self.cf.is_str)
        assert self.cf.get_option("a") == 1
        assert self.cf.get_option("b.c") == "hullo"

        self.cf.set_option("a", 2)
        self.cf.set_option("b.c", "wurld")
        assert self.cf.get_option("a") == 2
        assert self.cf.get_option("b.c") == "wurld"

        self.cf.reset_option("a")
        assert self.cf.get_option("a") == 1
        assert self.cf.get_option("b.c") == "wurld"
        self.cf.reset_option("b.c")
        assert self.cf.get_option("a") == 1
        assert self.cf.get_option("b.c") == "hullo"

    def test_reset_option_all(self):
        self.cf.register_option("a", 1, "doc", validator=self.cf.is_int)
        self.cf.register_option("b.c", "hullo", "doc2", validator=self.cf.is_str)
        assert self.cf.get_option("a") == 1
        assert self.cf.get_option("b.c") == "hullo"

        self.cf.set_option("a", 2)
        self.cf.set_option("b.c", "wurld")
        assert self.cf.get_option("a") == 2
        assert self.cf.get_option("b.c") == "wurld"

        self.cf.reset_option("all")
        assert self.cf.get_option("a") == 1
        assert self.cf.get_option("b.c") == "hullo"

    def test_deprecate_option(self):
        # we can deprecate non-existent options
        self.cf.deprecate_option("foo")

        assert self.cf._is_deprecated("foo")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(KeyError, match="No such keys.s.: 'foo'"):
                self.cf.get_option("foo")
            assert len(w) == 1  # should have raised one warning
            assert "deprecated" in str(w[-1])  # we get the default message

        self.cf.register_option("a", 1, "doc", validator=self.cf.is_int)
        self.cf.register_option("b.c", "hullo", "doc2")
        self.cf.register_option("foo", "hullo", "doc2")

        self.cf.deprecate_option("a", removal_ver="nifty_ver")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.cf.get_option("a")

            assert len(w) == 1  # should have raised one warning
            assert "eprecated" in str(w[-1])  # we get the default message
            assert "nifty_ver" in str(w[-1])  # with the removal_ver quoted

            msg = "Option 'a' has already been defined as deprecated"
            with pytest.raises(OptionError, match=msg):
                self.cf.deprecate_option("a")

        self.cf.deprecate_option("b.c", "zounds!")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.cf.get_option("b.c")

            assert len(w) == 1  # should have raised one warning
            assert "zounds!" in str(w[-1])  # we get the custom message

        # test rerouting keys
        self.cf.register_option("d.a", "foo", "doc2")
        self.cf.register_option("d.dep", "bar", "doc2")
        assert self.cf.get_option("d.a") == "foo"
        assert self.cf.get_option("d.dep") == "bar"

        self.cf.deprecate_option("d.dep", rkey="d.a")  # reroute d.dep to d.a
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert self.cf.get_option("d.dep") == "foo"

            assert len(w) == 1  # should have raised one warning
            assert "eprecated" in str(w[-1])  # we get the custom message

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.cf.set_option("d.dep", "baz")  # should overwrite "d.a"

            assert len(w) == 1  # should have raised one warning
            assert "eprecated" in str(w[-1])  # we get the custom message

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert self.cf.get_option("d.dep") == "baz"

            assert len(w) == 1  # should have raised one warning
            assert "eprecated" in str(w[-1])  # we get the custom message

    def test_config_prefix(self):
        with self.cf.config_prefix("base"):
            self.cf.register_option("a", 1, "doc1")
            self.cf.register_option("b", 2, "doc2")
            assert self.cf.get_option("a") == 1
            assert self.cf.get_option("b") == 2

            self.cf.set_option("a", 3)
            self.cf.set_option("b", 4)
            assert self.cf.get_option("a") == 3
            assert self.cf.get_option("b") == 4

        assert self.cf.get_option("base.a") == 3
        assert self.cf.get_option("base.b") == 4
        assert "doc1" in self.cf.describe_option("base.a", _print_desc=False)
        assert "doc2" in self.cf.describe_option("base.b", _print_desc=False)

        self.cf.reset_option("base.a")
        self.cf.reset_option("base.b")

        with self.cf.config_prefix("base"):
            assert self.cf.get_option("a") == 1
            assert self.cf.get_option("b") == 2

    def test_callback(self):
        k = [None]
        v = [None]

        def callback(key):
            k.append(key)
            v.append(self.cf.get_option(key))

        self.cf.register_option("d.a", "foo", cb=callback)
        self.cf.register_option("d.b", "foo", cb=callback)

        del k[-1], v[-1]
        self.cf.set_option("d.a", "fooz")
        assert k[-1] == "d.a"
        assert v[-1] == "fooz"

        del k[-1], v[-1]
        self.cf.set_option("d.b", "boo")
        assert k[-1] == "d.b"
        assert v[-1] == "boo"

        del k[-1], v[-1]
        self.cf.reset_option("d.b")
        assert k[-1] == "d.b"

    def test_set_ContextManager(self):
        def eq(val):
            assert self.cf.get_option("a") == val

        self.cf.register_option("a", 0)
        eq(0)
        with self.cf.option_context("a", 15):
            eq(15)
            with self.cf.option_context("a", 25):
                eq(25)
            eq(15)
        eq(0)

        self.cf.set_option("a", 17)
        eq(17)

        # Test that option_context can be used as a decorator too (#34253).
        @self.cf.option_context("a", 123)
        def f():
            eq(123)

        f()

    def test_attribute_access(self):
        holder = []

        def f3(key):
            holder.append(True)

        self.cf.register_option("a", 0)
        self.cf.register_option("c", 0, cb=f3)
        options = self.cf.options

        assert options.a == 0
        with self.cf.option_context("a", 15):
            assert options.a == 15

        options.a = 500
        assert self.cf.get_option("a") == 500

        self.cf.reset_option("a")
        assert options.a == self.cf.get_option("a", 0)

        msg = "You can only set the value of existing options"
        with pytest.raises(OptionError, match=msg):
            options.b = 1
        with pytest.raises(OptionError, match=msg):
            options.display = 1

        # make sure callback kicks when using this form of setting
        options.c = 1
        assert len(holder) == 1

    def test_option_context_scope(self):
        # Ensure that creating a context does not affect the existing
        # environment as it is supposed to be used with the `with` statement.
        # See https://github.com/pandas-dev/pandas/issues/8514

        original_value = 60
        context_value = 10
        option_name = "a"

        self.cf.register_option(option_name, original_value)

        # Ensure creating contexts didn't affect the current context.
        ctx = self.cf.option_context(option_name, context_value)
        assert self.cf.get_option(option_name) == original_value

        # Ensure the correct value is available inside the context.
        with ctx:
            assert self.cf.get_option(option_name) == context_value

        # Ensure the current context is reset
        assert self.cf.get_option(option_name) == original_value

    def test_dictwrapper_getattr(self):
        options = self.cf.options
        # GH 19789
        with pytest.raises(OptionError, match="No such option"):
            options.bananas
        assert not hasattr(options, "bananas")
