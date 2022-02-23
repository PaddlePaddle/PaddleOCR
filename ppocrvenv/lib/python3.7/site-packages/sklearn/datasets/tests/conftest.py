""" Network tests are only run, if data is already locally available,
or if download is specifically requested by environment variable."""
import builtins
import pytest


@pytest.fixture
def hide_available_pandas(monkeypatch):
    """Pretend pandas was not installed."""
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)
