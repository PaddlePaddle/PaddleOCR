from sklearn.utils._show_versions import _get_sys_info
from sklearn.utils._show_versions import _get_deps_info
from sklearn.utils._show_versions import show_versions
from sklearn.utils._testing import ignore_warnings


def test_get_sys_info():
    sys_info = _get_sys_info()

    assert "python" in sys_info
    assert "executable" in sys_info
    assert "machine" in sys_info


def test_get_deps_info():
    with ignore_warnings():
        deps_info = _get_deps_info()

    assert "pip" in deps_info
    assert "setuptools" in deps_info
    assert "sklearn" in deps_info
    assert "numpy" in deps_info
    assert "scipy" in deps_info
    assert "Cython" in deps_info
    assert "pandas" in deps_info
    assert "matplotlib" in deps_info
    assert "joblib" in deps_info


def test_show_versions(capsys):
    with ignore_warnings():
        show_versions()
        out, err = capsys.readouterr()

    assert "python" in out
    assert "numpy" in out
