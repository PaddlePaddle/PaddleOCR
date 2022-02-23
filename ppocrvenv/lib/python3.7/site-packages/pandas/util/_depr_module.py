"""
This module houses a utility class for mocking deprecated modules.
It is for internal use only and should not be used beyond this purpose.
"""

import importlib
from typing import Iterable
import warnings


class _DeprecatedModule:
    """
    Class for mocking deprecated modules.

    Parameters
    ----------
    deprmod : name of module to be deprecated.
    deprmodto : name of module as a replacement, optional.
                If not given, the __module__ attribute will
                be used when needed.
    removals : objects or methods in module that will no longer be
               accessible once module is removed.
    moved : dict, optional
            dictionary of function name -> new location for moved
            objects
    """

    def __init__(self, deprmod, deprmodto=None, removals=None, moved=None):
        self.deprmod = deprmod
        self.deprmodto = deprmodto
        self.removals = removals
        if self.removals is not None:
            self.removals = frozenset(self.removals)
        self.moved = moved

        # For introspection purposes.
        self.self_dir = frozenset(dir(type(self)))

    def __dir__(self) -> Iterable[str]:
        deprmodule = self._import_deprmod()
        return dir(deprmodule)

    def __repr__(self) -> str:
        deprmodule = self._import_deprmod()
        return repr(deprmodule)

    __str__ = __repr__

    def __getattr__(self, name: str):
        if name in self.self_dir:
            return object.__getattribute__(self, name)

        try:
            deprmodule = self._import_deprmod(self.deprmod)
        except ImportError:
            if self.deprmodto is None:
                raise

            # a rename
            deprmodule = self._import_deprmod(self.deprmodto)

        obj = getattr(deprmodule, name)

        if self.removals is not None and name in self.removals:
            warnings.warn(
                f"{self.deprmod}.{name} is deprecated and will be removed in "
                "a future version.",
                FutureWarning,
                stacklevel=2,
            )
        elif self.moved is not None and name in self.moved:
            warnings.warn(
                f"{self.deprmod} is deprecated and will be removed in "
                f"a future version.\nYou can access {name} as {self.moved[name]}",
                FutureWarning,
                stacklevel=2,
            )
        else:
            deprmodto = self.deprmodto
            if deprmodto is False:
                warnings.warn(
                    f"{self.deprmod}.{name} is deprecated and will be removed in "
                    "a future version.",
                    FutureWarning,
                    stacklevel=2,
                )
            else:
                if deprmodto is None:
                    deprmodto = obj.__module__
                # The object is actually located in another module.
                warnings.warn(
                    f"{self.deprmod}.{name} is deprecated. Please use "
                    f"{deprmodto}.{name} instead.",
                    FutureWarning,
                    stacklevel=2,
                )

        return obj

    def _import_deprmod(self, mod=None):
        if mod is None:
            mod = self.deprmod

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            deprmodule = importlib.import_module(mod)
            return deprmodule
