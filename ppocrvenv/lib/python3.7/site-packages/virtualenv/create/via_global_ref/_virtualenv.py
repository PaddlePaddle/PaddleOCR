"""Patches that are applied at runtime to the virtual environment"""
# -*- coding: utf-8 -*-

import os
import sys

VIRTUALENV_PATCH_FILE = os.path.join(__file__)


def patch_dist(dist):
    """
    Distutils allows user to configure some arguments via a configuration file:
    https://docs.python.org/3/install/index.html#distutils-configuration-files

    Some of this arguments though don't make sense in context of the virtual environment files, let's fix them up.
    """
    # we cannot allow some install config as that would get packages installed outside of the virtual environment
    old_parse_config_files = dist.Distribution.parse_config_files

    def parse_config_files(self, *args, **kwargs):
        result = old_parse_config_files(self, *args, **kwargs)
        install = self.get_option_dict("install")

        if "prefix" in install:  # the prefix governs where to install the libraries
            install["prefix"] = VIRTUALENV_PATCH_FILE, os.path.abspath(sys.prefix)
        for base in ("purelib", "platlib", "headers", "scripts", "data"):
            key = "install_{}".format(base)
            if key in install:  # do not allow global configs to hijack venv paths
                install.pop(key, None)
        return result

    dist.Distribution.parse_config_files = parse_config_files


# Import hook that patches some modules to ignore configuration values that break package installation in case
# of virtual environments.
_DISTUTILS_PATCH = "distutils.dist", "setuptools.dist"
if sys.version_info > (3, 4):
    # https://docs.python.org/3/library/importlib.html#setting-up-an-importer
    from functools import partial
    from importlib.abc import MetaPathFinder
    from importlib.util import find_spec

    class _Finder(MetaPathFinder):
        """A meta path finder that allows patching the imported distutils modules"""

        fullname = None

        # lock[0] is threading.Lock(), but initialized lazily to avoid importing threading very early at startup,
        # because there are gevent-based applications that need to be first to import threading by themselves.
        # See https://github.com/pypa/virtualenv/issues/1895 for details.
        lock = []

        def find_spec(self, fullname, path, target=None):
            if fullname in _DISTUTILS_PATCH and self.fullname is None:
                # initialize lock[0] lazily
                if len(self.lock) == 0:
                    import threading

                    lock = threading.Lock()
                    # there is possibility that two threads T1 and T2 are simultaneously running into find_spec,
                    # observing .lock as empty, and further going into hereby initialization. However due to the GIL,
                    # list.append() operation is atomic and this way only one of the threads will "win" to put the lock
                    # - that every thread will use - into .lock[0].
                    # https://docs.python.org/3/faq/library.html#what-kinds-of-global-value-mutation-are-thread-safe
                    self.lock.append(lock)

                with self.lock[0]:
                    self.fullname = fullname
                    try:
                        spec = find_spec(fullname, path)
                        if spec is not None:
                            # https://www.python.org/dev/peps/pep-0451/#how-loading-will-work
                            is_new_api = hasattr(spec.loader, "exec_module")
                            func_name = "exec_module" if is_new_api else "load_module"
                            old = getattr(spec.loader, func_name)
                            func = self.exec_module if is_new_api else self.load_module
                            if old is not func:
                                try:
                                    setattr(spec.loader, func_name, partial(func, old))
                                except AttributeError:
                                    pass  # C-Extension loaders are r/o such as zipimporter with <python 3.7
                            return spec
                    finally:
                        self.fullname = None

        @staticmethod
        def exec_module(old, module):
            old(module)
            if module.__name__ in _DISTUTILS_PATCH:
                patch_dist(module)

        @staticmethod
        def load_module(old, name):
            module = old(name)
            if module.__name__ in _DISTUTILS_PATCH:
                patch_dist(module)
            return module

    sys.meta_path.insert(0, _Finder())
else:
    # https://www.python.org/dev/peps/pep-0302/
    from imp import find_module
    from pkgutil import ImpImporter, ImpLoader

    class _VirtualenvImporter(object, ImpImporter):
        def __init__(self, path=None):
            object.__init__(self)
            ImpImporter.__init__(self, path)

        def find_module(self, fullname, path=None):
            if fullname in _DISTUTILS_PATCH:
                try:
                    return _VirtualenvLoader(fullname, *find_module(fullname.split(".")[-1], path))
                except ImportError:
                    pass
            return None

    class _VirtualenvLoader(object, ImpLoader):
        def __init__(self, fullname, file, filename, etc):
            object.__init__(self)
            ImpLoader.__init__(self, fullname, file, filename, etc)

        def load_module(self, fullname):
            module = super(_VirtualenvLoader, self).load_module(fullname)
            patch_dist(module)
            module.__loader__ = None  # distlib fallback
            return module

    sys.meta_path.append(_VirtualenvImporter())
