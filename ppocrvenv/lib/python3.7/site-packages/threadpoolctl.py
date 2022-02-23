"""threadpoolctl

This module provides utilities to introspect native libraries that relies on
thread pools (notably BLAS and OpenMP implementations) and dynamically set the
maximal number of threads they can use.
"""
# License: BSD 3-Clause

# The code to introspect dynamically loaded libraries on POSIX systems is
# adapted from code by Intel developer @anton-malakhov available at
# https://github.com/IntelPython/smp (Copyright (c) 2017, Intel Corporation)
# and also published under the BSD 3-Clause license
import os
import re
import sys
import ctypes
import textwrap
import warnings
from ctypes.util import find_library
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import ContextDecorator

__version__ = "3.1.0"
__all__ = ["threadpool_limits", "threadpool_info", "ThreadpoolController"]


# One can get runtime errors or even segfaults due to multiple OpenMP libraries
# loaded simultaneously which can happen easily in Python when importing and
# using compiled extensions built with different compilers and therefore
# different OpenMP runtimes in the same program. In particular libiomp (used by
# Intel ICC) and libomp used by clang/llvm tend to crash. This can happen for
# instance when calling BLAS inside a prange. Setting the following environment
# variable allows multiple OpenMP libraries to be loaded. It should not degrade
# performances since we manually take care of potential over-subscription
# performance issues, in sections of the code where nested OpenMP loops can
# happen, by dynamically reconfiguring the inner OpenMP runtime to temporarily
# disable it while under the scope of the outer OpenMP parallel section.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

# Structure to cast the info on dynamically loaded library. See
# https://linux.die.net/man/3/dl_iterate_phdr for more details.
_SYSTEM_UINT = ctypes.c_uint64 if sys.maxsize > 2 ** 32 else ctypes.c_uint32
_SYSTEM_UINT_HALF = ctypes.c_uint32 if sys.maxsize > 2 ** 32 else ctypes.c_uint16


class _dl_phdr_info(ctypes.Structure):
    _fields_ = [
        ("dlpi_addr", _SYSTEM_UINT),  # Base address of object
        ("dlpi_name", ctypes.c_char_p),  # path to the library
        ("dlpi_phdr", ctypes.c_void_p),  # pointer on dlpi_headers
        ("dlpi_phnum", _SYSTEM_UINT_HALF),  # number of elements in dlpi_phdr
    ]


# The RTLD_NOLOAD flag for loading shared libraries is not defined on Windows.
try:
    _RTLD_NOLOAD = os.RTLD_NOLOAD
except AttributeError:
    _RTLD_NOLOAD = ctypes.DEFAULT_MODE


# List of the supported libraries. The items are indexed by the name of the
# class to instantiate to create the library controller objects. The items hold
# the possible prefixes of loaded shared objects, the name of the internal_api
# to call, the name of the user_api and potentially some symbols that the library is
# expected to have (this is necessary to distinguish between the blas implementations
# when they are all renamed "libblas.dll" on conda-forge on windows).
_SUPPORTED_LIBRARIES = {
    "OpenMPController": {
        "user_api": "openmp",
        "internal_api": "openmp",
        "filename_prefixes": ("libiomp", "libgomp", "libomp", "vcomp"),
    },
    "OpenBLASController": {
        "user_api": "blas",
        "internal_api": "openblas",
        "filename_prefixes": ("libopenblas", "libblas"),
        "check_symbols": ("openblas_get_num_threads", "openblas_get_num_threads64_"),
    },
    "MKLController": {
        "user_api": "blas",
        "internal_api": "mkl",
        "filename_prefixes": ("libmkl_rt", "mkl_rt", "libblas"),
        "check_symbols": ("MKL_Get_Max_Threads",),
    },
    "BLISController": {
        "user_api": "blas",
        "internal_api": "blis",
        "filename_prefixes": ("libblis", "libblas"),
        "check_symbols": ("bli_thread_get_num_threads",),
    },
}

# Helpers for the doc and test names
_ALL_USER_APIS = list(set(lib["user_api"] for lib in _SUPPORTED_LIBRARIES.values()))
_ALL_INTERNAL_APIS = [lib["internal_api"] for lib in _SUPPORTED_LIBRARIES.values()]
_ALL_PREFIXES = list(
    set(
        prefix
        for lib in _SUPPORTED_LIBRARIES.values()
        for prefix in lib["filename_prefixes"]
    )
)
_ALL_BLAS_LIBRARIES = [
    lib["internal_api"]
    for lib in _SUPPORTED_LIBRARIES.values()
    if lib["user_api"] == "blas"
]
_ALL_OPENMP_LIBRARIES = list(
    _SUPPORTED_LIBRARIES["OpenMPController"]["filename_prefixes"]
)


def _format_docstring(*args, **kwargs):
    def decorator(o):
        if o.__doc__ is not None:
            o.__doc__ = o.__doc__.format(*args, **kwargs)
        return o

    return decorator


@lru_cache(maxsize=10000)
def _realpath(filepath):
    """Small caching wrapper around os.path.realpath to limit system calls"""
    return os.path.realpath(filepath)


@_format_docstring(USER_APIS=list(_ALL_USER_APIS), INTERNAL_APIS=_ALL_INTERNAL_APIS)
def threadpool_info():
    """Return the maximal number of threads for each detected library.

    Return a list with all the supported libraries that have been found. Each
    library is represented by a dict with the following information:

      - "user_api" : user API. Possible values are {USER_APIS}.
      - "internal_api": internal API. Possible values are {INTERNAL_APIS}.
      - "prefix" : filename prefix of the specific implementation.
      - "filepath": path to the loaded library.
      - "version": version of the library (if available).
      - "num_threads": the current thread limit.

    In addition, each library may contain internal_api specific entries.
    """
    return ThreadpoolController().info()


class _ThreadpoolLimiter:
    """The guts of ThreadpoolController.limit

    Refer to the docstring of ThreadpoolController.limit for more details.

    It will only act on the library controllers held by the provided `controller`.
    Using the default constructor sets the limits right away such that it can be used as
    a callable. Setting the limits can be delayed by using the `wrap` class method such
    that it can be used as a decorator.
    """

    def __init__(self, controller, *, limits=None, user_api=None):
        self._controller = controller
        self._limits, self._user_api, self._prefixes = self._check_params(
            limits, user_api
        )
        self._original_info = self._controller.info()
        self._set_threadpool_limits()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.restore_original_limits()

    @classmethod
    def wrap(cls, controller, *, limits=None, user_api=None):
        """Return an instance of this class that can be used as a decorator"""
        return _ThreadpoolLimiterDecorator(
            controller=controller, limits=limits, user_api=user_api
        )

    def restore_original_limits(self):
        """Set the limits back to their original values"""
        for lib_controller, original_info in zip(
            self._controller.lib_controllers, self._original_info
        ):
            lib_controller.set_num_threads(original_info["num_threads"])

    # Alias of `restore_original_limits` for backward compatibility
    unregister = restore_original_limits

    def get_original_num_threads(self):
        """Original num_threads from before calling threadpool_limits

        Return a dict `{user_api: num_threads}`.
        """
        num_threads = {}
        warning_apis = []

        for user_api in self._user_api:
            limits = [
                lib_info["num_threads"]
                for lib_info in self._original_info
                if lib_info["user_api"] == user_api
            ]
            limits = set(limits)
            n_limits = len(limits)

            if n_limits == 1:
                limit = limits.pop()
            elif n_limits == 0:
                limit = None
            else:
                limit = min(limits)
                warning_apis.append(user_api)

            num_threads[user_api] = limit

        if warning_apis:
            warnings.warn(
                "Multiple value possible for following user apis: "
                + ", ".join(warning_apis)
                + ". Returning the minimum."
            )

        return num_threads

    def _check_params(self, limits, user_api):
        """Suitable values for the _limits, _user_api and _prefixes attributes"""

        if isinstance(limits, str) and limits == "sequential_blas_under_openmp":
            (
                limits,
                user_api,
            ) = self._controller._get_params_for_sequential_blas_under_openmp().values()

        if limits is None or isinstance(limits, int):
            if user_api is None:
                user_api = _ALL_USER_APIS
            elif user_api in _ALL_USER_APIS:
                user_api = [user_api]
            else:
                raise ValueError(
                    f"user_api must be either in {_ALL_USER_APIS} or None. Got "
                    f"{user_api} instead."
                )

            if limits is not None:
                limits = {api: limits for api in user_api}
            prefixes = []
        else:
            if isinstance(limits, list):
                # This should be a list of dicts of library info, for
                # compatibility with the result from threadpool_info.
                limits = {
                    lib_info["prefix"]: lib_info["num_threads"] for lib_info in limits
                }
            elif isinstance(limits, ThreadpoolController):
                # To set the limits from the library controllers of a
                # ThreadpoolController object.
                limits = {
                    lib_controller.prefix: lib_controller.num_threads
                    for lib_controller in limits.lib_controllers
                }

            if not isinstance(limits, dict):
                raise TypeError(
                    "limits must either be an int, a list, a dict, or "
                    f"'sequential_blas_under_openmp'. Got {type(limits)} instead"
                )

            # With a dictionary, can set both specific limit for given
            # libraries and global limit for user_api. Fetch each separately.
            prefixes = [prefix for prefix in limits if prefix in _ALL_PREFIXES]
            user_api = [api for api in limits if api in _ALL_USER_APIS]

        return limits, user_api, prefixes

    def _set_threadpool_limits(self):
        """Change the maximal number of threads in selected thread pools.

        Return a list with all the supported libraries that have been found
        matching `self._prefixes` and `self._user_api`.
        """
        if self._limits is None:
            return

        for lib_controller in self._controller.lib_controllers:
            # self._limits is a dict {key: num_threads} where key is either
            # a prefix or a user_api. If a library matches both, the limit
            # corresponding to the prefix is chosen.
            if lib_controller.prefix in self._limits:
                num_threads = self._limits[lib_controller.prefix]
            elif lib_controller.user_api in self._limits:
                num_threads = self._limits[lib_controller.user_api]
            else:
                continue

            if num_threads is not None:
                lib_controller.set_num_threads(num_threads)


class _ThreadpoolLimiterDecorator(_ThreadpoolLimiter, ContextDecorator):
    """Same as _ThreadpoolLimiter but to be used as a decorator"""

    def __init__(self, controller, *, limits=None, user_api=None):
        self._limits, self._user_api, self._prefixes = self._check_params(
            limits, user_api
        )
        self._controller = controller

    def __enter__(self):
        # we need to set the limits here and not in the __init__ because we want the
        # limits to be set when calling the decorated function, not when creating the
        # decorator.
        self._original_info = self._controller.info()
        self._set_threadpool_limits()
        return self


@_format_docstring(
    USER_APIS=", ".join(f'"{api}"' for api in _ALL_USER_APIS),
    BLAS_LIBS=", ".join(_ALL_BLAS_LIBRARIES),
    OPENMP_LIBS=", ".join(_ALL_OPENMP_LIBRARIES),
)
class threadpool_limits(_ThreadpoolLimiter):
    """Change the maximal number of threads that can be used in thread pools.

    This object can be used either as a callable (the construction of this object
    limits the number of threads), as a context manager in a `with` block to
    automatically restore the original state of the controlled libraries when exiting
    the block, or as a decorator through its `wrap` method.

    Set the maximal number of threads that can be used in thread pools used in
    the supported libraries to `limit`. This function works for libraries that
    are already loaded in the interpreter and can be changed dynamically.

    This effect is global and impacts the whole Python process. There is no thread level
    isolation as these libraries do not offer thread-local APIs to configure the number
    of threads to use in nested parallel calls.

    Parameters
    ----------
    limits : int, dict, 'sequential_blas_under_openmp' or None (default=None)
        The maximal number of threads that can be used in thread pools

        - If int, sets the maximum number of threads to `limits` for each
          library selected by `user_api`.

        - If it is a dictionary `{{key: max_threads}}`, this function sets a
          custom maximum number of threads for each `key` which can be either a
          `user_api` or a `prefix` for a specific library.

        - If 'sequential_blas_under_openmp', it will chose the appropriate `limits`
          and `user_api` parameters for the specific use case of sequential BLAS
          calls within an OpenMP parallel region. The `user_api` parameter is
          ignored.

        - If None, this function does not do anything.

    user_api : {USER_APIS} or None (default=None)
        APIs of libraries to limit. Used only if `limits` is an int.

        - If "blas", it will only limit BLAS supported libraries ({BLAS_LIBS}).

        - If "openmp", it will only limit OpenMP supported libraries
          ({OPENMP_LIBS}). Note that it can affect the number of threads used
          by the BLAS libraries if they rely on OpenMP.

        - If None, this function will apply to all supported libraries.
    """

    def __init__(self, limits=None, user_api=None):
        super().__init__(ThreadpoolController(), limits=limits, user_api=user_api)

    @classmethod
    def wrap(cls, limits=None, user_api=None):
        return super().wrap(ThreadpoolController(), limits=limits, user_api=user_api)


@_format_docstring(
    PREFIXES=", ".join(f'"{prefix}"' for prefix in _ALL_PREFIXES),
    USER_APIS=", ".join(f'"{api}"' for api in _ALL_USER_APIS),
    BLAS_LIBS=", ".join(_ALL_BLAS_LIBRARIES),
    OPENMP_LIBS=", ".join(_ALL_OPENMP_LIBRARIES),
)
class ThreadpoolController:
    """Collection of LibController objects for all loaded supported libraries

    Attributes
    ----------
    lib_controllers : list of `LibController` objects
        The list of library controllers of all loaded supported libraries.
    """

    # Cache for libc under POSIX and a few system libraries under Windows.
    # We use a class level cache instead of an instance level cache because
    # it's very unlikely that a shared library will be unloaded and reloaded
    # during the lifetime of a program.
    _system_libraries = dict()

    def __init__(self):
        self.lib_controllers = []
        self._load_libraries()
        self._warn_if_incompatible_openmp()

    @classmethod
    def _from_controllers(cls, lib_controllers):
        new_controller = cls.__new__(cls)
        new_controller.lib_controllers = lib_controllers
        return new_controller

    def info(self):
        """Return lib_controllers info as a list of dicts"""
        return [lib_controller.info() for lib_controller in self.lib_controllers]

    def select(self, **kwargs):
        """Return a ThreadpoolController containing a subset of its current
        library controllers

        It will select all libraries matching at least one pair (key, value) from kwargs
        where key is an entry of the library info dict (like "user_api", "internal_api",
        "prefix", ...) and value is the value or a list of acceptable values for that
        entry.

        For instance, `ThreadpoolController().select(internal_api=["blis", "openblas"])`
        will select all library controllers whose internal_api is either "blis" or
        "openblas".
        """
        for key, vals in kwargs.items():
            kwargs[key] = [vals] if not isinstance(vals, list) else vals

        lib_controllers = [
            lib_controller
            for lib_controller in self.lib_controllers
            if any(
                getattr(lib_controller, key, None) in vals
                for key, vals in kwargs.items()
            )
        ]

        return ThreadpoolController._from_controllers(lib_controllers)

    def _get_params_for_sequential_blas_under_openmp(self):
        """Return appropriate params to use for a sequential BLAS call in an OpenMP loop

        This function takes into account the unexpected behavior of OpenBLAS with the
        OpenMP threading layer.
        """
        if self.select(
            internal_api="openblas", threading_layer="openmp"
        ).lib_controllers:
            return {"limits": None, "user_api": None}
        return {"limits": 1, "user_api": "blas"}

    @_format_docstring(
        USER_APIS=", ".join('"{}"'.format(api) for api in _ALL_USER_APIS),
        BLAS_LIBS=", ".join(_ALL_BLAS_LIBRARIES),
        OPENMP_LIBS=", ".join(_ALL_OPENMP_LIBRARIES),
    )
    def limit(self, *, limits=None, user_api=None):
        """Change the maximal number of threads that can be used in thread pools.

        This function returns an object that can be used either as a callable (the
        construction of this object limits the number of threads) or as a context
        manager, in a `with` block to automatically restore the original state of the
        controlled libraries when exiting the block.

        Set the maximal number of threads that can be used in thread pools used in
        the supported libraries to `limits`. This function works for libraries that
        are already loaded in the interpreter and can be changed dynamically.

        This effect is global and impacts the whole Python process. There is no thread
        level isolation as these libraries do not offer thread-local APIs to configure
        the number of threads to use in nested parallel calls.

        Parameters
        ----------
        limits : int, dict, 'sequential_blas_under_openmp' or None (default=None)
            The maximal number of threads that can be used in thread pools

            - If int, sets the maximum number of threads to `limits` for each
              library selected by `user_api`.

            - If it is a dictionary `{{key: max_threads}}`, this function sets a
              custom maximum number of threads for each `key` which can be either a
              `user_api` or a `prefix` for a specific library.

            - If 'sequential_blas_under_openmp', it will chose the appropriate `limits`
              and `user_api` parameters for the specific use case of sequential BLAS
              calls within an OpenMP parallel region. The `user_api` parameter is
              ignored.

            - If None, this function does not do anything.

        user_api : {USER_APIS} or None (default=None)
            APIs of libraries to limit. Used only if `limits` is an int.

            - If "blas", it will only limit BLAS supported libraries ({BLAS_LIBS}).

            - If "openmp", it will only limit OpenMP supported libraries
              ({OPENMP_LIBS}). Note that it can affect the number of threads used
              by the BLAS libraries if they rely on OpenMP.

            - If None, this function will apply to all supported libraries.
        """
        return _ThreadpoolLimiter(self, limits=limits, user_api=user_api)

    @_format_docstring(
        USER_APIS=", ".join('"{}"'.format(api) for api in _ALL_USER_APIS),
        BLAS_LIBS=", ".join(_ALL_BLAS_LIBRARIES),
        OPENMP_LIBS=", ".join(_ALL_OPENMP_LIBRARIES),
    )
    def wrap(self, *, limits=None, user_api=None):
        """Change the maximal number of threads that can be used in thread pools.

        This function returns an object that can be used as a decorator.

        Set the maximal number of threads that can be used in thread pools used in
        the supported libraries to `limits`. This function works for libraries that
        are already loaded in the interpreter and can be changed dynamically.

        Parameters
        ----------
        limits : int, dict or None (default=None)
            The maximal number of threads that can be used in thread pools

            - If int, sets the maximum number of threads to `limits` for each
              library selected by `user_api`.

            - If it is a dictionary `{{key: max_threads}}`, this function sets a
              custom maximum number of threads for each `key` which can be either a
              `user_api` or a `prefix` for a specific library.

            - If None, this function does not do anything.

        user_api : {USER_APIS} or None (default=None)
            APIs of libraries to limit. Used only if `limits` is an int.

            - If "blas", it will only limit BLAS supported libraries ({BLAS_LIBS}).

            - If "openmp", it will only limit OpenMP supported libraries
              ({OPENMP_LIBS}). Note that it can affect the number of threads used
              by the BLAS libraries if they rely on OpenMP.

            - If None, this function will apply to all supported libraries.
        """
        return _ThreadpoolLimiter.wrap(self, limits=limits, user_api=user_api)

    def __len__(self):
        return len(self.lib_controllers)

    def _load_libraries(self):
        """Loop through loaded shared libraries and store the supported ones"""
        if sys.platform == "darwin":
            self._find_libraries_with_dyld()
        elif sys.platform == "win32":
            self._find_libraries_with_enum_process_module_ex()
        else:
            self._find_libraries_with_dl_iterate_phdr()

    def _find_libraries_with_dl_iterate_phdr(self):
        """Loop through loaded libraries and return binders on supported ones

        This function is expected to work on POSIX system only.
        This code is adapted from code by Intel developer @anton-malakhov
        available at https://github.com/IntelPython/smp

        Copyright (c) 2017, Intel Corporation published under the BSD 3-Clause
        license
        """
        libc = self._get_libc()
        if not hasattr(libc, "dl_iterate_phdr"):  # pragma: no cover
            return []

        # Callback function for `dl_iterate_phdr` which is called for every
        # library loaded in the current process until it returns 1.
        def match_library_callback(info, size, data):
            # Get the path of the current library
            filepath = info.contents.dlpi_name
            if filepath:
                filepath = filepath.decode("utf-8")

                # Store the library controller if it is supported and selected
                self._make_controller_from_path(filepath)
            return 0

        c_func_signature = ctypes.CFUNCTYPE(
            ctypes.c_int,  # Return type
            ctypes.POINTER(_dl_phdr_info),
            ctypes.c_size_t,
            ctypes.c_char_p,
        )
        c_match_library_callback = c_func_signature(match_library_callback)

        data = ctypes.c_char_p(b"")
        libc.dl_iterate_phdr(c_match_library_callback, data)

    def _find_libraries_with_dyld(self):
        """Loop through loaded libraries and return binders on supported ones

        This function is expected to work on OSX system only
        """
        libc = self._get_libc()
        if not hasattr(libc, "_dyld_image_count"):  # pragma: no cover
            return []

        n_dyld = libc._dyld_image_count()
        libc._dyld_get_image_name.restype = ctypes.c_char_p

        for i in range(n_dyld):
            filepath = ctypes.string_at(libc._dyld_get_image_name(i))
            filepath = filepath.decode("utf-8")

            # Store the library controller if it is supported and selected
            self._make_controller_from_path(filepath)

    def _find_libraries_with_enum_process_module_ex(self):
        """Loop through loaded libraries and return binders on supported ones

        This function is expected to work on windows system only.
        This code is adapted from code by Philipp Hagemeister @phihag available
        at https://stackoverflow.com/questions/17474574
        """
        from ctypes.wintypes import DWORD, HMODULE, MAX_PATH

        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010

        LIST_LIBRARIES_ALL = 0x03

        ps_api = self._get_windll("Psapi")
        kernel_32 = self._get_windll("kernel32")

        h_process = kernel_32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, os.getpid()
        )
        if not h_process:  # pragma: no cover
            raise OSError(f"Could not open PID {os.getpid()}")

        try:
            buf_count = 256
            needed = DWORD()
            # Grow the buffer until it becomes large enough to hold all the
            # module headers
            while True:
                buf = (HMODULE * buf_count)()
                buf_size = ctypes.sizeof(buf)
                if not ps_api.EnumProcessModulesEx(
                    h_process,
                    ctypes.byref(buf),
                    buf_size,
                    ctypes.byref(needed),
                    LIST_LIBRARIES_ALL,
                ):
                    raise OSError("EnumProcessModulesEx failed")
                if buf_size >= needed.value:
                    break
                buf_count = needed.value // (buf_size // buf_count)

            count = needed.value // (buf_size // buf_count)
            h_modules = map(HMODULE, buf[:count])

            # Loop through all the module headers and get the library path
            buf = ctypes.create_unicode_buffer(MAX_PATH)
            n_size = DWORD()
            for h_module in h_modules:

                # Get the path of the current module
                if not ps_api.GetModuleFileNameExW(
                    h_process, h_module, ctypes.byref(buf), ctypes.byref(n_size)
                ):
                    raise OSError("GetModuleFileNameEx failed")
                filepath = buf.value

                # Store the library controller if it is supported and selected
                self._make_controller_from_path(filepath)
        finally:
            kernel_32.CloseHandle(h_process)

    def _make_controller_from_path(self, filepath):
        """Store a library controller if it is supported and selected"""
        # Required to resolve symlinks
        filepath = _realpath(filepath)
        # `lower` required to take account of OpenMP dll case on Windows
        # (vcomp, VCOMP, Vcomp, ...)
        filename = os.path.basename(filepath).lower()

        # Loop through supported libraries to find if this filename corresponds
        # to a supported one.
        for controller_class, candidate_lib in _SUPPORTED_LIBRARIES.items():
            # check if filename matches a supported prefix
            prefix = self._check_prefix(filename, candidate_lib["filename_prefixes"])

            # filename does not match any of the prefixes of the candidate
            # library. move to next library.
            if prefix is None:
                continue

            # workaround for BLAS libraries packaged by conda-forge on windows, which
            # are all renamed "libblas.dll". We thus have to check to which BLAS
            # implementation it actually corresponds looking for implementation
            # specific symbols.
            if prefix == "libblas":
                if filename.endswith(".dll"):
                    libblas = ctypes.CDLL(filepath, _RTLD_NOLOAD)
                    if not any(
                        hasattr(libblas, func)
                        for func in candidate_lib["check_symbols"]
                    ):
                        continue
                else:
                    # We ignore libblas on other platforms than windows because there
                    # might be a libblas dso comming with openblas for instance that
                    # can't be used to instantiate a pertinent LibController (many
                    # symbols are missing) and would create confusion by making a
                    # duplicate entry in threadpool_info.
                    continue

            # filename matches a prefix. Create and store the library
            # controller.
            user_api = candidate_lib["user_api"]
            internal_api = candidate_lib["internal_api"]

            lib_controller_class = globals()[controller_class]
            lib_controller = lib_controller_class(
                filepath=filepath,
                prefix=prefix,
                user_api=user_api,
                internal_api=internal_api,
            )
            self.lib_controllers.append(lib_controller)

    def _check_prefix(self, library_basename, filename_prefixes):
        """Return the prefix library_basename starts with

        Return None if none matches.
        """
        for prefix in filename_prefixes:
            if library_basename.startswith(prefix):
                return prefix
        return None

    def _warn_if_incompatible_openmp(self):
        """Raise a warning if llvm-OpenMP and intel-OpenMP are both loaded"""
        if sys.platform != "linux":
            # Only raise the warning on linux
            return

        prefixes = [lib_controller.prefix for lib_controller in self.lib_controllers]
        msg = textwrap.dedent(
            """
            Found Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp') loaded at
            the same time. Both libraries are known to be incompatible and this
            can cause random crashes or deadlocks on Linux when loaded in the
            same Python program.
            Using threadpoolctl may cause crashes or deadlocks. For more
            information and possible workarounds, please see
                https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md
            """
        )
        if "libomp" in prefixes and "libiomp" in prefixes:
            warnings.warn(msg, RuntimeWarning)

    @classmethod
    def _get_libc(cls):
        """Load the lib-C for unix systems."""
        libc = cls._system_libraries.get("libc")
        if libc is None:
            libc_name = find_library("c")
            if libc_name is None:  # pragma: no cover
                return None
            libc = ctypes.CDLL(libc_name, mode=_RTLD_NOLOAD)
            cls._system_libraries["libc"] = libc
        return libc

    @classmethod
    def _get_windll(cls, dll_name):
        """Load a windows DLL"""
        dll = cls._system_libraries.get(dll_name)
        if dll is None:
            dll = ctypes.WinDLL(f"{dll_name}.dll")
            cls._system_libraries[dll_name] = dll
        return dll


@_format_docstring(
    USER_APIS=", ".join('"{}"'.format(api) for api in _ALL_USER_APIS),
    INTERNAL_APIS=", ".join('"{}"'.format(api) for api in _ALL_INTERNAL_APIS),
)
class LibController(ABC):
    """Abstract base class for the individual library controllers

    A library controller is represented by the following information:
      - "user_api" : user API. Possible values are {USER_APIS}.
      - "internal_api" : internal API. Possible values are {INTERNAL_APIS}.
      - "prefix" : prefix of the shared library's name.
      - "filepath" : path to the loaded library.
      - "version" : version of the library (if available).
      - "num_threads" : the current thread limit.

    In addition, each library controller may contain internal_api specific
    entries.
    """

    def __init__(self, *, filepath=None, prefix=None, user_api=None, internal_api=None):
        self.user_api = user_api
        self.internal_api = internal_api
        self.prefix = prefix
        self.filepath = filepath
        self._dynlib = ctypes.CDLL(filepath, mode=_RTLD_NOLOAD)
        self.version = self.get_version()

    def info(self):
        """Return relevant info wrapped in a dict"""
        all_attrs = dict(vars(self), **{"num_threads": self.num_threads})
        return {k: v for k, v in all_attrs.items() if not k.startswith("_")}

    @property
    def num_threads(self):
        return self.get_num_threads()

    @abstractmethod
    def get_num_threads(self):
        """Return the maximum number of threads available to use"""
        pass  # pragma: no cover

    @abstractmethod
    def set_num_threads(self, num_threads):
        """Set the maximum number of threads to use"""
        pass  # pragma: no cover

    @abstractmethod
    def get_version(self):
        """Return the version of the shared library"""
        pass  # pragma: no cover


class OpenBLASController(LibController):
    """Controller class for OpenBLAS"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threading_layer = self._get_threading_layer()
        self.architecture = self._get_architecture()

    def get_num_threads(self):
        get_func = getattr(
            self._dynlib,
            "openblas_get_num_threads",
            # Symbols differ when built for 64bit integers in Fortran
            getattr(self._dynlib, "openblas_get_num_threads64_", lambda: None),
        )

        return get_func()

    def set_num_threads(self, num_threads):
        set_func = getattr(
            self._dynlib,
            "openblas_set_num_threads",
            # Symbols differ when built for 64bit integers in Fortran
            getattr(
                self._dynlib, "openblas_set_num_threads64_", lambda num_threads: None
            ),
        )
        return set_func(num_threads)

    def get_version(self):
        # None means OpenBLAS is not loaded or version < 0.3.4, since OpenBLAS
        # did not expose its version before that.
        get_config = getattr(
            self._dynlib,
            "openblas_get_config",
            getattr(self._dynlib, "openblas_get_config64_", None),
        )
        if get_config is None:
            return None

        get_config.restype = ctypes.c_char_p
        config = get_config().split()
        if config[0] == b"OpenBLAS":
            return config[1].decode("utf-8")
        return None

    def _get_threading_layer(self):
        """Return the threading layer of OpenBLAS"""
        openblas_get_parallel = getattr(
            self._dynlib,
            "openblas_get_parallel",
            getattr(self._dynlib, "openblas_get_parallel64_", None),
        )
        if openblas_get_parallel is None:
            return "unknown"
        threading_layer = openblas_get_parallel()
        if threading_layer == 2:
            return "openmp"
        elif threading_layer == 1:
            return "pthreads"
        return "disabled"

    def _get_architecture(self):
        """Return the architecture detected by OpenBLAS"""
        get_corename = getattr(
            self._dynlib,
            "openblas_get_corename",
            getattr(self._dynlib, "openblas_get_corename64_", None),
        )
        if get_corename is None:
            return None

        get_corename.restype = ctypes.c_char_p
        return get_corename().decode("utf-8")


class BLISController(LibController):
    """Controller class for BLIS"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threading_layer = self._get_threading_layer()
        self.architecture = self._get_architecture()

    def get_num_threads(self):
        get_func = getattr(self._dynlib, "bli_thread_get_num_threads", lambda: None)
        num_threads = get_func()
        # by default BLIS is single-threaded and get_num_threads
        # returns -1. We map it to 1 for consistency with other libraries.
        return 1 if num_threads == -1 else num_threads

    def set_num_threads(self, num_threads):
        set_func = getattr(
            self._dynlib, "bli_thread_set_num_threads", lambda num_threads: None
        )
        return set_func(num_threads)

    def get_version(self):
        get_version_ = getattr(self._dynlib, "bli_info_get_version_str", None)
        if get_version_ is None:
            return None

        get_version_.restype = ctypes.c_char_p
        return get_version_().decode("utf-8")

    def _get_threading_layer(self):
        """Return the threading layer of BLIS"""
        if self._dynlib.bli_info_get_enable_openmp():
            return "openmp"
        elif self._dynlib.bli_info_get_enable_pthreads():
            return "pthreads"
        return "disabled"

    def _get_architecture(self):
        """Return the architecture detected by BLIS"""
        bli_arch_query_id = getattr(self._dynlib, "bli_arch_query_id", None)
        bli_arch_string = getattr(self._dynlib, "bli_arch_string", None)
        if bli_arch_query_id is None or bli_arch_string is None:
            return None

        # the true restype should be BLIS' arch_t (enum) but int should work
        # for us:
        bli_arch_query_id.restype = ctypes.c_int
        bli_arch_string.restype = ctypes.c_char_p
        return bli_arch_string(bli_arch_query_id()).decode("utf-8")


class MKLController(LibController):
    """Controller class for MKL"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threading_layer = self._get_threading_layer()

    def get_num_threads(self):
        get_func = getattr(self._dynlib, "MKL_Get_Max_Threads", lambda: None)
        return get_func()

    def set_num_threads(self, num_threads):
        set_func = getattr(
            self._dynlib, "MKL_Set_Num_Threads", lambda num_threads: None
        )
        return set_func(num_threads)

    def get_version(self):
        if not hasattr(self._dynlib, "MKL_Get_Version_String"):
            return None

        res = ctypes.create_string_buffer(200)
        self._dynlib.MKL_Get_Version_String(res, 200)

        version = res.value.decode("utf-8")
        group = re.search(r"Version ([^ ]+) ", version)
        if group is not None:
            version = group.groups()[0]
        return version.strip()

    def _get_threading_layer(self):
        """Return the threading layer of MKL"""
        # The function mkl_set_threading_layer returns the current threading
        # layer. Calling it with an invalid threading layer allows us to safely
        # get the threading layer
        set_threading_layer = getattr(
            self._dynlib, "MKL_Set_Threading_Layer", lambda layer: -1
        )
        layer_map = {
            0: "intel",
            1: "sequential",
            2: "pgi",
            3: "gnu",
            4: "tbb",
            -1: "not specified",
        }
        return layer_map[set_threading_layer(-1)]


class OpenMPController(LibController):
    """Controller class for OpenMP"""

    def get_num_threads(self):
        get_func = getattr(self._dynlib, "omp_get_max_threads", lambda: None)
        return get_func()

    def set_num_threads(self, num_threads):
        set_func = getattr(
            self._dynlib, "omp_set_num_threads", lambda num_threads: None
        )
        return set_func(num_threads)

    def get_version(self):
        # There is no way to get the version number programmatically in OpenMP.
        return None


def _main():
    """Commandline interface to display thread-pool information and exit."""
    import argparse
    import importlib
    import json
    import sys

    parser = argparse.ArgumentParser(
        usage="python -m threadpoolctl -i numpy scipy.linalg xgboost",
        description="Display thread-pool information and exit.",
    )
    parser.add_argument(
        "-i",
        "--import",
        dest="modules",
        nargs="*",
        default=(),
        help="Python modules to import before introspecting thread-pools.",
    )
    parser.add_argument(
        "-c",
        "--command",
        help="a Python statement to execute before introspecting thread-pools.",
    )

    options = parser.parse_args(sys.argv[1:])
    for module in options.modules:
        try:
            importlib.import_module(module, package=None)
        except ImportError:
            print("WARNING: could not import", module, file=sys.stderr)

    if options.command:
        exec(options.command)

    print(json.dumps(threadpool_info(), indent=2))


if __name__ == "__main__":
    _main()
