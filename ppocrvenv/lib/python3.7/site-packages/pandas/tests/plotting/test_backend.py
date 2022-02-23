import sys
import types

import pkg_resources
import pytest

import pandas.util._test_decorators as td

import pandas

dummy_backend = types.ModuleType("pandas_dummy_backend")
setattr(dummy_backend, "plot", lambda *args, **kwargs: "used_dummy")


pytestmark = pytest.mark.slow


@pytest.fixture
def restore_backend():
    """Restore the plotting backend to matplotlib"""
    pandas.set_option("plotting.backend", "matplotlib")
    yield
    pandas.set_option("plotting.backend", "matplotlib")


def test_backend_is_not_module():
    msg = "Could not find plotting backend 'not_an_existing_module'."
    with pytest.raises(ValueError, match=msg):
        pandas.set_option("plotting.backend", "not_an_existing_module")

    assert pandas.options.plotting.backend == "matplotlib"


def test_backend_is_correct(monkeypatch, restore_backend):
    monkeypatch.setitem(sys.modules, "pandas_dummy_backend", dummy_backend)

    pandas.set_option("plotting.backend", "pandas_dummy_backend")
    assert pandas.get_option("plotting.backend") == "pandas_dummy_backend"
    assert (
        pandas.plotting._core._get_plot_backend("pandas_dummy_backend") is dummy_backend
    )


def test_backend_can_be_set_in_plot_call(monkeypatch, restore_backend):
    monkeypatch.setitem(sys.modules, "pandas_dummy_backend", dummy_backend)
    df = pandas.DataFrame([1, 2, 3])

    assert pandas.get_option("plotting.backend") == "matplotlib"
    assert df.plot(backend="pandas_dummy_backend") == "used_dummy"


@td.skip_if_no_mpl
def test_register_entrypoint(restore_backend):

    dist = pkg_resources.get_distribution("pandas")
    if dist.module_path not in pandas.__file__:
        # We are running from a non-installed pandas, and this test is invalid
        pytest.skip("Testing a non-installed pandas")

    mod = types.ModuleType("my_backend")
    mod.plot = lambda *args, **kwargs: 1

    backends = pkg_resources.get_entry_map("pandas")
    my_entrypoint = pkg_resources.EntryPoint(
        "pandas_plotting_backend", mod.__name__, dist=dist
    )
    backends["pandas_plotting_backends"]["my_backend"] = my_entrypoint
    # TODO: the docs recommend importlib.util.module_from_spec. But this works for now.
    sys.modules["my_backend"] = mod

    result = pandas.plotting._core._get_plot_backend("my_backend")
    assert result is mod

    # TODO: https://github.com/pandas-dev/pandas/issues/27517
    # Remove the td.skip_if_no_mpl
    with pandas.option_context("plotting.backend", "my_backend"):
        result = pandas.plotting._core._get_plot_backend()

    assert result is mod


def test_setting_backend_without_plot_raises():
    # GH-28163
    module = types.ModuleType("pandas_plot_backend")
    sys.modules["pandas_plot_backend"] = module

    assert pandas.options.plotting.backend == "matplotlib"
    with pytest.raises(
        ValueError, match="Could not find plotting backend 'pandas_plot_backend'."
    ):
        pandas.set_option("plotting.backend", "pandas_plot_backend")

    assert pandas.options.plotting.backend == "matplotlib"


@td.skip_if_mpl
def test_no_matplotlib_ok():
    msg = (
        'matplotlib is required for plotting when the default backend "matplotlib" is '
        "selected."
    )
    with pytest.raises(ImportError, match=msg):
        pandas.plotting._core._get_plot_backend("matplotlib")


def test_extra_kinds_ok(monkeypatch, restore_backend):
    # https://github.com/pandas-dev/pandas/pull/28647
    monkeypatch.setitem(sys.modules, "pandas_dummy_backend", dummy_backend)
    pandas.set_option("plotting.backend", "pandas_dummy_backend")
    df = pandas.DataFrame({"A": [1, 2, 3]})
    df.plot(kind="not a real kind")
