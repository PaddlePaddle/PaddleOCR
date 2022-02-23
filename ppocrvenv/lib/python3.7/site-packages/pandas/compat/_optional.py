from __future__ import annotations

import importlib
import sys
import types
import warnings

from pandas.util.version import Version

# Update install.rst when updating versions!

VERSIONS = {
    "bs4": "4.6.0",
    "bottleneck": "1.2.1",
    "fsspec": "0.7.4",
    "fastparquet": "0.4.0",
    "gcsfs": "0.6.0",
    "lxml.etree": "4.3.0",
    "matplotlib": "2.2.3",
    "numexpr": "2.7.0",
    "odfpy": "1.3.0",
    "openpyxl": "3.0.0",
    "pandas_gbq": "0.12.0",
    "pyarrow": "0.17.0",
    "pytest": "6.0",
    "pyxlsb": "1.0.6",
    "s3fs": "0.4.0",
    "scipy": "1.2.0",
    "sqlalchemy": "1.3.0",
    "tables": "3.5.1",
    "tabulate": "0.8.7",
    "xarray": "0.12.3",
    "xlrd": "1.2.0",
    "xlwt": "1.3.0",
    "xlsxwriter": "1.0.2",
    "numba": "0.46.0",
}

# A mapping from import name to package name (on PyPI) for packages where
# these two names are different.

INSTALL_MAPPING = {
    "bs4": "beautifulsoup4",
    "bottleneck": "Bottleneck",
    "lxml.etree": "lxml",
    "odf": "odfpy",
    "pandas_gbq": "pandas-gbq",
    "sqlalchemy": "SQLAlchemy",
    "jinja2": "Jinja2",
}


def get_version(module: types.ModuleType) -> str:
    version = getattr(module, "__version__", None)
    if version is None:
        # xlrd uses a capitalized attribute name
        version = getattr(module, "__VERSION__", None)

    if version is None:
        raise ImportError(f"Can't determine version for {module.__name__}")
    return version


def import_optional_dependency(
    name: str,
    extra: str = "",
    errors: str = "raise",
    min_version: str | None = None,
):
    """
    Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice
    message will be raised. If a dependency is present, but too old,
    we raise.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.
    errors : str {'raise', 'warn', 'ignore'}
        What to do when a dependency is not found or its version is too old.

        * raise : Raise an ImportError
        * warn : Only applicable when a module's version is to old.
          Warns that the version is too old and returns None
        * ignore: If the module is not installed, return None, otherwise,
          return the module, even if the version is too old.
          It's expected that users validate the version locally when
          using ``errors="ignore"`` (see. ``io/html.py``)
    min_version : str, default None
        Specify a minimum version that is different from the global pandas
        minimum version required.
    Returns
    -------
    maybe_module : Optional[ModuleType]
        The imported module, when found and the version is correct.
        None is returned when the package is not found and `errors`
        is False, or when the package's version is too old and `errors`
        is ``'warn'``.
    """

    assert errors in {"warn", "raise", "ignore"}

    package_name = INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name

    msg = (
        f"Missing optional dependency '{install_name}'. {extra} "
        f"Use pip or conda to install {install_name}."
    )
    try:
        module = importlib.import_module(name)
    except ImportError:
        if errors == "raise":
            raise ImportError(msg) from None
        else:
            return None

    # Handle submodules: if we have submodule, grab parent module from sys.modules
    parent = name.split(".")[0]
    if parent != name:
        install_name = parent
        module_to_get = sys.modules[install_name]
    else:
        module_to_get = module
    minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
    if minimum_version:
        version = get_version(module_to_get)
        if Version(version) < Version(minimum_version):
            msg = (
                f"Pandas requires version '{minimum_version}' or newer of '{parent}' "
                f"(version '{version}' currently installed)."
            )
            if errors == "warn":
                warnings.warn(msg, UserWarning)
                return None
            elif errors == "raise":
                raise ImportError(msg)

    return module
