# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import base64
import concurrent.futures
import errno
import logging
import os
import shutil
import tempfile
import traceback
import uuid
from collections import OrderedDict
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Set,
    Union,
)
from urllib.parse import urlparse

import portalocker  # type: ignore
from iopath.common.download import download
from iopath.common.event_logger import VTYPE, EventLogger
from iopath.common.non_blocking_io import NonBlockingIOManager


__all__ = ["LazyPath", "PathManager", "get_cache_dir", "file_lock"]


def get_cache_dir(cache_dir: Optional[str] = None) -> str:
    """
    Returns a default directory to cache static files
    (usually downloaded from Internet), if None is provided.

    Args:
        cache_dir (None or str): if not None, will be returned as is.
            If None, returns the default cache directory as:

        1) $FVCORE_CACHE, if set
        2) otherwise ~/.torch/iopath_cache
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(
            os.getenv("FVCORE_CACHE", "~/.torch/iopath_cache")
        )
    try:
        g_pathmgr.mkdirs(cache_dir)
        assert os.access(cache_dir, os.W_OK)
    except (OSError, AssertionError):
        tmp_dir = os.path.join(tempfile.gettempdir(), "iopath_cache")
        logger = logging.getLogger(__name__)
        logger.warning(f"{cache_dir} is not accessible! Using {tmp_dir} instead!")
        cache_dir = tmp_dir
    return cache_dir


def file_lock(path: str):  # type: ignore
    """
    A file lock. Once entered, it is guaranteed that no one else holds the
    same lock. Others trying to enter the lock will block for 30 minutes and
    raise an exception.

    This is useful to make sure workers don't cache files to the same location.

    Args:
        path (str): a path to be locked. This function will create a lock named
            `path + ".lock"`

    Examples:

        filename = "/path/to/file"
        with file_lock(filename):
            if not os.path.isfile(filename):
                do_create_file()
    """
    dirname = os.path.dirname(path)
    try:
        os.makedirs(dirname, exist_ok=True)
    except OSError:
        # makedir is not atomic. Exceptions can happen when multiple workers try
        # to create the same dir, despite exist_ok=True.
        # When this happens, we assume the dir is created and proceed to creating
        # the lock. If failed to create the directory, the next line will raise
        # exceptions.
        pass
    return portalocker.Lock(path + ".lock", timeout=3600)  # type: ignore


class LazyPath(os.PathLike):
    """
    A path that's lazily evaluated when it's used.

    Users should be careful to not use it like a str, because
    it behaves differently from a str.
    Path manipulation functions in Python such as `os.path.*` all accept
    PathLike objects already.

    It can be materialized to a str using `os.fspath`.
    """

    KEPT_ATTRIBUTES = ["__getstate__", "__setstate__"]

    def __init__(self, func: Callable[[], str]) -> None:
        """
        Args:
            func: a function that takes no arguments and returns the
                actual path as a str. It will be called at most once.
        """
        self._func = func
        self._value: Optional[str] = None

    def _get_value(self) -> str:
        if self._value is None:
            self._value = self._func()
        return self._value  # pyre-ignore

    def __fspath__(self) -> str:
        return self._get_value()

    # behave more like a str after evaluated
    def __getattr__(self, name: str):  # type: ignore
        if name in LazyPath.KEPT_ATTRIBUTES:
            return super().__getattr__(name)
        if self._value is None:
            raise AttributeError(f"Uninitialized LazyPath has no attribute: {name}.")
        return getattr(self._value, name)

    def __getitem__(self, key):  # type: ignore
        if self._value is None:
            raise TypeError("Uninitialized LazyPath is not subscriptable.")
        return self._value[key]  # type: ignore

    def __str__(self) -> str:
        if self._value is not None:
            return self._value  # type: ignore
        else:
            return super().__str__()


class PathHandler(EventLogger):
    """
    PathHandler is a base class that defines common I/O functionality for a URI
    protocol. It routes I/O for a generic URI which may look like "protocol://*"
    or a canonical filepath "/foo/bar/baz".
    """

    _strict_kwargs_check = True

    def __init__(
        self,
        async_executor: Optional[concurrent.futures.Executor] = None,
    ) -> None:
        """
        When registering a `PathHandler`, the user can optionally pass in a
        `Executor` to run the asynchronous file operations.
        NOTE: For regular non-async operations of `PathManager`, there is
        no need to pass `async_executor`.

        Args:
            async_executor (optional `Executor`): Used for async file operations.
                Usage:
                ```
                    path_handler = NativePathHandler(async_executor=exe)
                    path_manager.register_handler(path_handler)
                ```
        """
        super().__init__()
        self._non_blocking_io_manager = None
        self._non_blocking_io_executor = async_executor
        try:
            from iopath.common.setup_defaults import setup_handler_defaults

            setup_handler_defaults(self)
        except ImportError:
            pass

    def _check_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """
        Checks if the given arguments are empty. Throws a ValueError if strict
        kwargs checking is enabled and args are non-empty. If strict kwargs
        checking is disabled, only a warning is logged.

        Args:
            kwargs (Dict[str, Any])
        """
        if self._strict_kwargs_check:
            if len(kwargs) > 0:
                raise ValueError("Unused arguments: {}".format(kwargs))
        else:
            logger = logging.getLogger(__name__)
            for k, v in kwargs.items():
                logger.warning("[PathManager] {}={} argument ignored".format(k, v))

    def _get_supported_prefixes(self) -> List[str]:
        """
        Returns:
            List[str]: the list of URI prefixes this PathHandler can support
        """
        raise NotImplementedError()

    def _get_local_path(self, path: str, force: bool = False, **kwargs: Any) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk. In this case, the cache stays on filesystem
        (under `file_io.get_cache_dir()`) and will be used by a different run.
        Therefore this function is meant to be used with read-only resources.

        Args:
            path (str): A URI supported by this PathHandler
            force(bool): Forces a download from backend if set to True.

        Returns:
            local_path (str): a file path which exists on the local file system
        """
        raise NotImplementedError()

    def _copy_from_local(
        self, local_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> None:
        """
        Copies a local file to the specified URI.

        If the URI is another local path, this should be functionally identical
        to copy.

        Args:
            local_path (str): a file path which exists on the local file system
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing URI

        Returns:
            status (bool): True on success
        """
        raise NotImplementedError()

    def _opent(
        self, path: str, mode: str = "r", buffering: int = 32, **kwargs: Any
    ) -> Iterable[Any]:
        raise NotImplementedError()

    def _open(
        self, path: str, mode: str = "r", buffering: int = -1, **kwargs: Any
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a stream to a URI, similar to the built-in `open`.

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): An optional integer used to set the buffering policy.
                Pass 0 to switch buffering off and an integer >= 1 to indicate the
                size in bytes of a fixed-size chunk buffer. When no buffering
                argument is given, the default buffering policy depends on the
                underlying I/O implementation.

        Returns:
            file: a file-like object.
        """
        raise NotImplementedError()

    def _opena(
        self,
        path: str,
        mode: str = "r",
        callback_after_file_close: Optional[Callable[[None], None]] = None,
        buffering: int = -1,
        **kwargs: Any,
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a stream to a URI with asynchronous permissions.

        NOTE: Writes to the same path are serialized so they are written in
        the same order as they were called but writes to distinct paths can
        happen concurrently.

        Usage (default / without callback function):
            for n in range(50):
                results = run_a_large_task(n)
                with path_manager.opena(uri, "w") as f:
                    f.write(results)            # Runs in separate thread
                # Main process returns immediately and continues to next iteration
            path_manager.async_close()

        Usage (advanced / with callback function):
            # To write local and then copy to Manifold:
            def cb():
                path_manager.copy_from_local(
                    "checkpoint.pt", "manifold://path/to/bucket"
                )
            f = pm.opena("checkpoint.pt", "wb", callback_after_file_close=cb)
            torch.save({...}, f)
            f.close()

        Args:
            ...same args as `_open`...
            callback_after_file_close (Callable): An optional argument that can
                be passed to perform operations that depend on the asynchronous
                writes being completed. The file is first written to the local
                disk and then the callback is executed.
            buffering (int): An optional argument to set the buffer size for
                buffered asynchronous writing.

        Returns:
            file: a file-like object with asynchronous methods.
        """
        # Restrict mode until `NonBlockingIO` has async read feature.
        valid_modes = {"w", "a", "b"}
        if not all(m in valid_modes for m in mode):
            raise ValueError("`opena` mode must be write or append")

        # TODO: Each `PathHandler` should set its own `self._buffered`
        # parameter and pass that in here. Until then, we assume no
        # buffering for any storage backend.
        if not self._non_blocking_io_manager:
            self._non_blocking_io_manager = NonBlockingIOManager(
                buffered=False,
                executor=self._non_blocking_io_executor,
            )

        try:
            return self._non_blocking_io_manager.get_non_blocking_io(
                path=self._get_path_with_cwd(path),
                io_obj=self._open(path, mode, **kwargs),
                callback_after_file_close=callback_after_file_close,
                buffering=buffering,
            )
        except ValueError:
            # When `_strict_kwargs_check = True`, then `open_callable`
            # will throw a `ValueError`. This generic `_opena` function
            # does not check the kwargs since it may include any `_open`
            # args like `encoding`, `ttl`, `has_user_data`, etc.
            logger = logging.getLogger(__name__)
            logger.exception(
                "An exception occurred in `NonBlockingIOManager`. This "
                "is most likely due to invalid `opena` args. Make sure "
                "they match the `open` args for the `PathHandler`."
            )
            self._async_close()

    def _async_join(self, path: Optional[str] = None, **kwargs: Any) -> bool:
        """
        Ensures that desired async write threads are properly joined.

        Args:
            path (str): Pass in a file path to wait until all asynchronous
                activity for that path is complete. If no path is passed in,
                then this will wait until all asynchronous jobs are complete.

        Returns:
            status (bool): True on success
        """
        if not self._non_blocking_io_manager:
            logger = logging.getLogger(__name__)
            logger.warning(
                "This is an async feature. No threads to join because "
                "`opena` was not used."
            )
        self._check_kwargs(kwargs)
        return self._non_blocking_io_manager._join(
            self._get_path_with_cwd(path) if path else None
        )

    def _async_close(self, **kwargs: Any) -> bool:
        """
        Closes the thread pool used for the asynchronous operations.

        Returns:
            status (bool): True on success
        """
        if not self._non_blocking_io_manager:
            logger = logging.getLogger(__name__)
            logger.warning(
                "This is an async feature. No threadpool to close because "
                "`opena` was not used."
            )
        self._check_kwargs(kwargs)
        return self._non_blocking_io_manager._close_thread_pool()

    def _copy(
        self, src_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file

        Returns:
            status (bool): True on success
        """
        raise NotImplementedError()

    def _mv(self, src_path: str, dst_path: str, **kwargs: Any) -> bool:
        """
        Moves (renames) a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler

        Returns:
            status (bool): True on success
        """
        raise NotImplementedError()

    def _exists(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        raise NotImplementedError()

    def _isfile(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a file.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a file
        """
        raise NotImplementedError()

    def _isdir(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a directory.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a directory
        """
        raise NotImplementedError()

    def _ls(self, path: str, **kwargs: Any) -> List[str]:
        """
        List the contents of the directory at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            List[str]: list of contents in given path
        """
        raise NotImplementedError()

    def _mkdirs(self, path: str, **kwargs: Any) -> None:
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.

        Args:
            path (str): A URI supported by this PathHandler
        """
        raise NotImplementedError()

    def _rm(self, path: str, **kwargs: Any) -> None:
        """
        Remove the file (not directory) at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler
        """
        raise NotImplementedError()

    def _symlink(self, src_path: str, dst_path: str, **kwargs: Any) -> bool:
        """
        Symlink the src_path to the dst_path

        Args:
            src_path (str): A URI supported by this PathHandler to symlink from
            dst_path (str): A URI supported by this PathHandler to symlink to
        """
        raise NotImplementedError()

    def _set_cwd(self, path: Union[str, None], **kwargs: Any) -> bool:
        """
        Set the current working directory. PathHandler classes prepend the cwd
        to all URI paths that are handled.

        Args:
            path (str) or None: A URI supported by this PathHandler. Must be a valid
                absolute path or None to set the cwd to None.

        Returns:
            bool: true if cwd was set without errors
        """
        raise NotImplementedError()

    def _get_path_with_cwd(self, path: str) -> str:
        """
        Default implementation. PathHandler classes that provide a `_set_cwd`
        feature should also override this `_get_path_with_cwd` method.

        Args:
            path (str): A URI supported by this PathHandler.

        Returns:
            path (str): Full path with the cwd attached.
        """
        return path


class NativePathHandler(PathHandler):
    """
    Handles paths that can be accessed using Python native system calls. This
    handler uses `open()` and `os.*` calls on the given path.
    """

    _cwd = None

    def __init__(
        self,
        async_executor: Optional[concurrent.futures.Executor] = None,
    ) -> None:
        super().__init__(async_executor)

    def _get_local_path(self, path: str, force: bool = False, **kwargs: Any) -> str:
        self._check_kwargs(kwargs)
        return os.fspath(path)

    def _copy_from_local(
        self, local_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> None:
        self._check_kwargs(kwargs)
        local_path = self._get_path_with_cwd(local_path)
        dst_path = self._get_path_with_cwd(dst_path)
        assert self._copy(
            src_path=local_path, dst_path=dst_path, overwrite=overwrite, **kwargs
        )

    def _open(
        self,
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        closefd: bool = True,
        opener: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a path.

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): An optional integer used to set the buffering policy.
                Pass 0 to switch buffering off and an integer >= 1 to indicate the
                size in bytes of a fixed-size chunk buffer. When no buffering
                argument is given, the default buffering policy works as follows:
                    * Binary files are buffered in fixed-size chunks; the size of
                    the buffer is chosen using a heuristic trying to determine the
                    underlying device’s “block size” and falling back on
                    io.DEFAULT_BUFFER_SIZE. On many systems, the buffer will
                    typically be 4096 or 8192 bytes long.
            encoding (Optional[str]): the name of the encoding used to decode or
                encode the file. This should only be used in text mode.
            errors (Optional[str]): an optional string that specifies how encoding
                and decoding errors are to be handled. This cannot be used in binary
                mode.
            newline (Optional[str]): controls how universal newlines mode works
                (it only applies to text mode). It can be None, '', '\n', '\r',
                and '\r\n'.
            closefd (bool): If closefd is False and a file descriptor rather than
                a filename was given, the underlying file descriptor will be kept
                open when the file is closed. If a filename is given closefd must
                be True (the default) otherwise an error will be raised.
            opener (Optional[Callable]): A custom opener can be used by passing
                a callable as opener. The underlying file descriptor for the file
                object is then obtained by calling opener with (file, flags).
                opener must return an open file descriptor (passing os.open as opener
                results in functionality similar to passing None).

            See https://docs.python.org/3/library/functions.html#open for details.

        Returns:
            file: a file-like object.
        """
        self._check_kwargs(kwargs)
        return open(  # type: ignore
            self._get_path_with_cwd(path),
            mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            closefd=closefd,
            opener=opener,
        )

    def _copy(
        self, src_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file

        Returns:
            status (bool): True on success
        """
        self._check_kwargs(kwargs)
        src_path = self._get_path_with_cwd(src_path)
        dst_path = self._get_path_with_cwd(dst_path)
        if os.path.exists(dst_path) and not overwrite:
            logger = logging.getLogger(__name__)
            logger.error("Destination file {} already exists.".format(dst_path))
            return False

        try:
            shutil.copyfile(src_path, dst_path)
            return True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error("Error in file copy - {}".format(str(e)))
            return False

    def _mv(self, src_path: str, dst_path: str, **kwargs: Any) -> bool:
        """
        Moves (renames) a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler

        Returns:
            status (bool): True on success
        """
        self._check_kwargs(kwargs)
        src_path = self._get_path_with_cwd(src_path)
        dst_path = self._get_path_with_cwd(dst_path)
        if os.path.exists(dst_path):
            logger = logging.getLogger(__name__)
            logger.error("Destination file {} already exists.".format(dst_path))
            return False

        try:
            shutil.move(src_path, dst_path)
            return True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error("Error in move operation - {}".format(str(e)))
            return False

    def _symlink(self, src_path: str, dst_path: str, **kwargs: Any) -> bool:
        """
        Creates a symlink to the src_path at the dst_path

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler

        Returns:
            status (bool): True on success
        """
        self._check_kwargs(kwargs)
        src_path = self._get_path_with_cwd(src_path)
        dst_path = self._get_path_with_cwd(dst_path)
        logger = logging.getLogger(__name__)
        if not os.path.exists(src_path):
            logger.error("Source path {} does not exist".format(src_path))
            return False
        if os.path.exists(dst_path):
            logger.error("Destination path {} already exists.".format(dst_path))
            return False
        try:
            os.symlink(src_path, dst_path)
            return True
        except Exception as e:
            logger.error("Error in symlink - {}".format(str(e)))
            return False

    def _exists(self, path: str, **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        return os.path.exists(self._get_path_with_cwd(path))

    def _isfile(self, path: str, **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        return os.path.isfile(self._get_path_with_cwd(path))

    def _isdir(self, path: str, **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        return os.path.isdir(self._get_path_with_cwd(path))

    def _ls(self, path: str, **kwargs: Any) -> List[str]:
        self._check_kwargs(kwargs)
        return os.listdir(self._get_path_with_cwd(path))

    def _mkdirs(self, path: str, **kwargs: Any) -> None:
        self._check_kwargs(kwargs)
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            # EEXIST it can still happen if multiple processes are creating the dir
            if e.errno != errno.EEXIST:
                raise

    def _rm(self, path: str, **kwargs: Any) -> None:
        self._check_kwargs(kwargs)
        os.remove(path)

    def _set_cwd(self, path: Union[str, None], **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        # Remove cwd path if None
        if path is None:
            self._cwd = None
            return True

        # Make sure path is a valid Unix path
        if not os.path.exists(path):
            raise ValueError(f"{path} is not a valid Unix path")
        # Make sure path is an absolute path
        if not os.path.isabs(path):
            raise ValueError(f"{path} is not an absolute path")
        self._cwd = path
        return True

    def _get_path_with_cwd(self, path: str) -> str:
        if not path:
            return path
        return os.path.normpath(
            path if not self._cwd else os.path.join(self._cwd, path)
        )


class HTTPURLHandler(PathHandler):
    """
    Download URLs and cache them to disk.
    """

    MAX_FILENAME_LEN = 250

    def __init__(self) -> None:
        super().__init__()
        self.cache_map: Dict[str, str] = {}

    def _get_supported_prefixes(self) -> List[str]:
        return ["http://", "https://", "ftp://"]

    def _get_local_path(
        self,
        path: str,
        force: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        This implementation downloads the remote resource and caches it locally.
        The resource will only be downloaded if not previously requested.
        """
        self._check_kwargs(kwargs)
        if (
            force
            or path not in self.cache_map
            or not os.path.exists(self.cache_map[path])
        ):
            logger = logging.getLogger(__name__)
            parsed_url = urlparse(path)
            dirname = os.path.join(
                get_cache_dir(cache_dir), os.path.dirname(parsed_url.path.lstrip("/"))
            )
            filename = path.split("/")[-1]
            if len(filename) > self.MAX_FILENAME_LEN:
                filename = filename[:100] + "_" + uuid.uuid4().hex

            cached = os.path.join(dirname, filename)
            with file_lock(cached):
                if not os.path.isfile(cached):
                    logger.info("Downloading {} ...".format(path))
                    cached = download(path, dirname, filename=filename)
            logger.info("URL {} cached in {}".format(path, cached))
            self.cache_map[path] = cached
        return self.cache_map[path]

    def _open(
        self, path: str, mode: str = "r", buffering: int = -1, **kwargs: Any
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a remote HTTP path. The resource is first downloaded and cached
        locally.

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): Not used for this PathHandler.

        Returns:
            file: a file-like object.
        """
        self._check_kwargs(kwargs)
        assert mode in ("r", "rb"), "{} does not support open with {} mode".format(
            self.__class__.__name__, mode
        )
        assert (
            buffering == -1
        ), f"{self.__class__.__name__} does not support the `buffering` argument"
        local_path = self._get_local_path(path, force=False)
        return open(local_path, mode)


class OneDrivePathHandler(HTTPURLHandler):
    """
    Map OneDrive (short) URLs to direct download links
    """

    ONE_DRIVE_PREFIX = "https://1drv.ms/u/s!"

    def create_one_drive_direct_download(self, one_drive_url: str) -> str:
        """
        Converts a short OneDrive URI into a download link that can be used with wget

        Args:
            one_drive_url (str): A OneDrive URI supported by this PathHandler

        Returns:
            result_url (str): A direct download URI for the file
        """
        data_b64 = base64.b64encode(bytes(one_drive_url, "utf-8"))
        data_b64_string = (
            data_b64.decode("utf-8").replace("/", "_").replace("+", "-").rstrip("=")
        )
        result_url = (
            f"https://api.onedrive.com/v1.0/shares/u!{data_b64_string}/root/content"
        )
        return result_url

    def _get_supported_prefixes(self) -> List[str]:
        return [self.ONE_DRIVE_PREFIX]

    def _get_local_path(self, path: str, force: bool = False, **kwargs: Any) -> str:
        """
        This implementation downloads the remote resource and caches it locally.
        The resource will only be downloaded if not previously requested.
        """
        logger = logging.getLogger(__name__)
        direct_url = self.create_one_drive_direct_download(path)

        logger.info(f"URL {path} mapped to direct download link {direct_url}")

        return super()._get_local_path(os.fspath(direct_url), force=force, **kwargs)


class PathManager:
    """
    A class for users to open generic paths or translate generic paths to file names.

    path_manager.method(path) will do the following:
    1. Find a handler by checking the prefixes in `self._path_handlers`.
    2. Call handler.method(path) on the handler that's found
    """

    def __init__(self) -> None:
        self._path_handlers: MutableMapping[str, PathHandler] = OrderedDict()
        """
        Dict from path prefix to handler.
        """

        self._native_path_handler: PathHandler = NativePathHandler()
        """
        A NativePathHandler that works on posix paths. This is used as the fallback.
        """

        self._cwd: Optional[str] = None
        """
        Keeps track of the single cwd (if set).
        NOTE: Only one PathHandler can have a cwd set at a time.
        """

        self._async_handlers: Set[PathHandler] = set()
        """
        Keeps track of the PathHandler subclasses where `opena` was used so
        all of the threads can be properly joined when calling
        `PathManager.join`.
        """

        self._enable_logging = True
        """
        Flag for enabling / disabling telemetry.
        """

    def __get_path_handler(self, path: Union[str, os.PathLike]) -> PathHandler:
        """
        Finds a PathHandler that supports the given path. Falls back to the native
        PathHandler if no other handler is found.

        Args:
            path (str or os.PathLike): URI path to resource

        Returns:
            handler (PathHandler)
        """
        path = os.fspath(path)  # pyre-ignore
        for p in self._path_handlers.keys():
            if path.startswith(p):
                return self._path_handlers[p]
        return self._native_path_handler

    def __log_tmetry_keys(self, handler: PathHandler, kvs: Dict[str, VTYPE]) -> None:
        """
        Logs a dictionary of key-value pairs to a given path handler.

        Args:
            handler (PathHandler): The path handler to send the key-value pairs to.
            kvs (Dict): Dict of Key-value pairs to be logged.

        """
        if self._enable_logging:
            handler.add_keys(kvs)
            try:
                handler.log_event()
            except Exception:
                logger = logging.getLogger(__name__)
                logger.exception(
                    "An exception occurred in telemetry logging."
                    "Disabling telemetry to prevent further exceptions."
                )
                self._enable_logging = False

    def __get_open_keys(self, path: str, mode: str, buffering: int) -> Dict[str, VTYPE]:
        """
        Helper function to return common set of key-value pairs applicable to open apis.

        Args:
            path (str):
            mode (str):
            buffering (int):

        """
        kvs = {}
        kvs["op"] = "open"
        if "r" in mode:
            kvs["mode"] = "read"
        elif "w" in mode:
            kvs["mode"] = "write"
        elif "a" in mode:
            kvs["mode"] = "append"
        kvs["buffering"] = buffering
        if "b" in mode:
            kvs["format"] = "binary"
        else:
            kvs["format"] = "text"
        kvs["path"] = path

        return kvs

    def opent(
        self, path: str, mode: str = "r", buffering: int = 32, **kwargs: Any
    ) -> Iterable[Any]:
        """
        Open a tabular data source. Only reading is supported.
        The opent() returns a Python iterable collection object, compared to bytes/text data with open()

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'
            buffering (int): number of rows fetched and cached

        Returns:
            An iterable collection object.
        """
        return self.__get_path_handler(path)._opent(path, mode, buffering, **kwargs)

    def open(
        self, path: str, mode: str = "r", buffering: int = -1, **kwargs: Any
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a stream to a URI, similar to the built-in `open`.

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): An optional integer used to set the buffering policy.
                Pass 0 to switch buffering off and an integer >= 1 to indicate the
                size in bytes of a fixed-size chunk buffer. When no buffering
                argument is given, the default buffering policy depends on the
                underlying I/O implementation.

        Returns:
            file: a file-like object.
        """
        handler = self.__get_path_handler(path)
        bret = handler._open(path, mode, buffering=buffering, **kwargs)  # type: ignore

        kvs = self.__get_open_keys(path, mode, buffering)
        self.__log_tmetry_keys(handler, kvs)
        return bret

    # NOTE: This feature is only implemented for `NativePathHandler` and can
    # currently only be used in write mode.
    def opena(
        self,
        path: str,
        mode: str = "r",
        buffering: int = -1,
        callback_after_file_close: Optional[Callable[[None], None]] = None,
        **kwargs: Any,
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a file with asynchronous permissions. `f.write()` calls (and
        potentially `f.read()` calls in the future) will be dispatched
        asynchronously such that the main program can continue running.

        NOTE: Writes to the same path are serialized so they are written in
        the same order as they were called but writes to distinct paths can
        happen concurrently.

        Usage (default / without callback function):
            for n in range(50):
                results = run_a_large_task(n)
                # `f` is a file-like object with asynchronous methods
                with path_manager.opena(uri, "w") as f:
                    f.write(results)            # Runs in separate thread
                # Main process returns immediately and continues to next iteration
            path_manager.async_close()

        Usage (advanced / with callback function):
            # To asynchronously write to Manifold:
            def cb():
                path_manager.copy_from_local(
                    "checkpoint.pt", "manifold://path/to/bucket"
                )
            f = pm.opena("checkpoint.pt", "wb", callback_after_file_close=cb)
            torch.save({...}, f)
            f.close()

        Args:
            ...
            callback_after_file_close (Callable): An optional argument that can
                be passed to perform operations that depend on the asynchronous
                writes being completed. The file is first written to the local
                disk and then the callback is executed.

        Returns:
            file: a file-like object with asynchronous methods.
        """
        non_blocking_io = self.__get_path_handler(path)._opena(
            path,
            mode,
            buffering=buffering,
            callback_after_file_close=callback_after_file_close,
            **kwargs,
        )
        # Keep track of the path handlers where `opena` is used so that all of the
        # threads can be properly joined on `PathManager.join`.
        self._async_handlers.add(self.__get_path_handler(path))
        return non_blocking_io

    def async_join(self, *paths: str, **kwargs: Any) -> bool:
        """
        Ensures that desired async write threads are properly joined.

        Usage:
            Wait for asynchronous methods operating on specific file paths to
            complete.
                async_join("path/to/file1.txt")
                async_join("path/to/file2.txt", "path/to/file3.txt")
            Wait for all asynchronous methods to complete.
                async_join()

        Args:
            *paths (str): Pass in any number of file paths and `async_join` will wait
                until all asynchronous activity for those paths is complete. If no
                paths are passed in, then `async_join` will wait until all asynchronous
                jobs are complete.

        Returns:
            status (bool): True on success
        """
        success = True
        if not paths:  # Join all.
            for handler in self._async_handlers:
                success = handler._async_join(**kwargs) and success
        else:  # Join specific paths.
            for path in paths:
                success = (
                    self.__get_path_handler(path)._async_join(path, **kwargs)
                    and success
                )
        return success

    def async_close(self, **kwargs: Any) -> bool:
        """
        `async_close()` must be called at the very end of any script that uses the
        asynchronous `opena` feature. This calls `async_join()` first and then closes
        the thread pool used for the asynchronous operations.

        Returns:
            status (bool): True on success
        """
        success = self.async_join(**kwargs)
        for handler in self._async_handlers:
            success = handler._async_close(**kwargs) and success
        self._async_handlers.clear()
        return success

    def copy(
        self, src_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file

        Returns:
            status (bool): True on success
        """

        # Copying across handlers is not supported.
        assert self.__get_path_handler(  # type: ignore
            src_path
        ) == self.__get_path_handler(dst_path)
        handler = self.__get_path_handler(src_path)
        bret = handler._copy(src_path, dst_path, overwrite, **kwargs)
        kvs = {"op": "copy", "path": src_path, "dst_path": dst_path}
        self.__log_tmetry_keys(handler, kvs)
        return bret

    def mv(self, src_path: str, dst_path: str, **kwargs: Any) -> bool:
        """
        Moves (renames) a source path supported by NativePathHandler to
        a destination path.

        Args:
            src_path (str): A URI supported by NativePathHandler
            dst_path (str): A URI supported by NativePathHandler

        Returns:
            status (bool): True on success
        Exception:
            Asserts if both the src and dest paths are not supported by
            NativePathHandler.
        """

        # Moving across handlers is not supported.
        assert self.__get_path_handler(  # type: ignore
            src_path
        ) == self.__get_path_handler(
            dst_path
        ), "Src and dest paths must be supported by the same path handler."
        handler = self.__get_path_handler(src_path)
        bret = handler._mv(src_path, dst_path, **kwargs)
        kvs = {"op": "mv", "path": src_path, "dst_path": dst_path}
        self.__log_tmetry_keys(handler, kvs)
        return bret

    def get_local_path(self, path: str, force: bool = False, **kwargs: Any) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk.

        Args:
            path (str): A URI supported by this PathHandler
            force(bool): Forces a download from backend if set to True.

        Returns:
            local_path (str): a file path which exists on the local file system
        """
        path = os.fspath(path)
        handler = self.__get_path_handler(path)  # type: ignore
        try:
            bret = handler._get_local_path(path, force=force, **kwargs)
        except TypeError:
            bret = handler._get_local_path(path, **kwargs)
        kvs = {"op": "get_local_path", "path": path, "force": force}
        self.__log_tmetry_keys(handler, kvs)
        return bret

    def copy_from_local(
        self, local_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> None:
        """
        Copies a local file to the specified URI.

        If the URI is another local path, this should be functionally identical
        to copy.

        Args:
            local_path (str): a file path which exists on the local file system
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing URI

        Returns:
            status (bool): True on success
        """
        assert os.path.exists(local_path)
        handler = self.__get_path_handler(dst_path)

        kvs = {
            "op": "copy_from_local",
            "path": local_path,
            "dst_path": dst_path,
            "overwrite": overwrite,
        }
        bret = handler._copy_from_local(
            local_path=local_path, dst_path=dst_path, overwrite=overwrite, **kwargs
        )
        self.__log_tmetry_keys(handler, kvs)
        return bret

    def exists(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        handler = self.__get_path_handler(path)
        bret = handler._exists(path, **kwargs)  # type: ignore
        kvs = {"op": "exists", "path": path}
        self.__log_tmetry_keys(handler, kvs)
        return bret

    def isfile(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if there the resource at the given URI is a file.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a file
        """
        handler = self.__get_path_handler(path)
        bret = handler._isfile(path, **kwargs)  # type: ignore
        kvs = {"op": "isfile", "path": path}
        self.__log_tmetry_keys(handler, kvs)
        return bret

    def isdir(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a directory.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a directory
        """
        handler = self.__get_path_handler(path)
        bret = handler._isdir(path, **kwargs)  # type: ignore
        kvs = {"op": "isdir", "path": path}
        self.__log_tmetry_keys(handler, kvs)
        return bret

    def ls(self, path: str, **kwargs: Any) -> List[str]:
        """
        List the contents of the directory at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            List[str]: list of contents in given path
        """
        return self.__get_path_handler(path)._ls(path, **kwargs)

    def mkdirs(self, path: str, **kwargs: Any) -> None:
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.

        Args:
            path (str): A URI supported by this PathHandler
        """
        handler = self.__get_path_handler(path)
        bret = handler._mkdirs(path, **kwargs)  # type: ignore
        kvs = {"op": "mkdirs", "path": path}
        self.__log_tmetry_keys(handler, kvs)
        return bret

    def rm(self, path: str, **kwargs: Any) -> None:
        """
        Remove the file (not directory) at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler
        """
        handler = self.__get_path_handler(path)
        bret = handler._rm(path, **kwargs)  # type: ignore
        kvs = {"op": "rm", "path": path}
        self.__log_tmetry_keys(handler, kvs)
        return bret

    def symlink(self, src_path: str, dst_path: str, **kwargs: Any) -> bool:
        """Symlink the src_path to the dst_path

        Args:
            src_path (str): A URI supported by this PathHandler to symlink from
            dst_path (str): A URI supported by this PathHandler to symlink to
        """
        # Copying across handlers is not supported.
        assert self.__get_path_handler(  # type: ignore
            src_path
        ) == self.__get_path_handler(dst_path)
        handler = self.__get_path_handler(src_path)
        bret = handler._symlink(src_path, dst_path, **kwargs)  # type: ignore
        kvs = {"op": "symlink", "path": src_path, "dst_path": dst_path}
        self.__log_tmetry_keys(handler, kvs)
        return bret

    def set_cwd(self, path: Union[str, None], **kwargs: Any) -> bool:
        """
        Set the current working directory. PathHandler classes prepend the cwd
        to all URI paths that are handled.

        Args:
            path (str) or None: A URI supported by this PathHandler. Must be a valid
                absolute Unix path or None to set the cwd to None.

        Returns:
            bool: true if cwd was set without errors
        """
        if path is None and self._cwd is None:
            return True
        handler = self.__get_path_handler(path or self._cwd)
        if self.__get_path_handler(path or self._cwd)._set_cwd(path, **kwargs):  # type: ignore
            self._cwd = path
            bret = True
        else:
            bret = False
        kvs = {"op": "set_cwd", "path": path}
        self.__log_tmetry_keys(handler, kvs)
        return bret

    def register_handler(
        self, handler: PathHandler, allow_override: bool = True
    ) -> None:
        """
        Register a path handler associated with `handler._get_supported_prefixes`
        URI prefixes.

        Args:
            handler (PathHandler)
            allow_override (bool): allow overriding existing handler for prefix
        """
        logger = logging.getLogger(__name__)
        assert isinstance(handler, PathHandler), handler

        # Allow override of `NativePathHandler` which is automatically
        # instantiated by `PathManager`.
        if isinstance(handler, NativePathHandler):
            if allow_override:
                self._native_path_handler = handler
            else:
                raise ValueError(
                    "`NativePathHandler` is registered by default. Use the "
                    "`allow_override=True` kwarg to override it."
                )
            return

        for prefix in handler._get_supported_prefixes():
            if prefix not in self._path_handlers:
                self._path_handlers[prefix] = handler
                continue

            old_handler_type = type(self._path_handlers[prefix])
            if allow_override:
                # if using the global PathManager, show the warnings
                global g_pathmgr
                if self == g_pathmgr:
                    logger.warning(
                        f"[PathManager] Attempting to register prefix '{prefix}' from "
                        "the following call stack:\n"
                        + "".join(traceback.format_stack(limit=5))
                        # show the most recent callstack
                    )
                    logger.warning(
                        f"[PathManager] Prefix '{prefix}' is already registered "
                        f"by {old_handler_type}. We will override the old handler. "
                        "To avoid such conflicts, create a project-specific PathManager "
                        "instead."
                    )
                self._path_handlers[prefix] = handler
            else:
                raise KeyError(
                    f"[PathManager] Prefix '{prefix}' already registered by {old_handler_type}!"
                )

        # Sort path handlers in reverse order so longer prefixes take priority,
        # eg: http://foo/bar before http://foo
        self._path_handlers = OrderedDict(
            sorted(self._path_handlers.items(), key=lambda t: t[0], reverse=True)
        )

    def set_strict_kwargs_checking(self, enable: bool) -> None:
        """
        Toggles strict kwargs checking. If enabled, a ValueError is thrown if any
        unused parameters are passed to a PathHandler function. If disabled, only
        a warning is given.

        With a centralized file API, there's a tradeoff of convenience and
        correctness delegating arguments to the proper I/O layers. An underlying
        `PathHandler` may support custom arguments which should not be statically
        exposed on the `PathManager` function. For example, a custom `HTTPURLHandler`
        may want to expose a `cache_timeout` argument for `open()` which specifies
        how old a locally cached resource can be before it's refetched from the
        remote server. This argument would not make sense for a `NativePathHandler`.
        If strict kwargs checking is disabled, `cache_timeout` can be passed to
        `PathManager.open` which will forward the arguments to the underlying
        handler. By default, checking is enabled since it is innately unsafe:
        multiple `PathHandler`s could reuse arguments with different semantic
        meanings or types.

        Args:
            enable (bool)
        """
        self._native_path_handler._strict_kwargs_check = enable
        for handler in self._path_handlers.values():
            handler._strict_kwargs_check = enable

    def set_logging(self, enable_logging=True):
        self._enable_logging = enable_logging


"""

History: at first, PathManager is part of detectron2 *only*, and therefore
does not consider cross-projects conflict issues. It was designed to provide
a global interface with static methods.

It is later used by more projects and moved to iopath to faciliate more use
across projects and lead to some conflicts.

Now 'PathManager' class has been changed to provide a per-project interface.
This means that any project using PathManager will first have to instantiate the
class and call methods on the instantiated object.

  example:

  _pathmgr = PathManager()
  _pathmgr.register_handler(<a PathHandler instance>)
  _pathmgr.mkdir(<dir-path>)

"""


class PathManagerFactory:
    """
    PathManagerFactory is the class responsible for creating new PathManager
    instances and removing them when no longer needed.
    PathManager can be instantiated directly too, but it is recommended that
    you use PathManagerFactory to create them.
    """

    GLOBAL_PATH_MANAGER = "global_path_manager"
    pm_list = {}

    @staticmethod
    def get(key=GLOBAL_PATH_MANAGER, defaults_setup=False) -> PathManager:
        """
        Get the path manager instance associated with a key.
        A new instance will be created if there is no existing
        instance associated with the key passed in.
        Args:
            key (str):
            defaults_setup (bool): If True, setup_defaults is called.
        """
        if key not in PathManagerFactory.pm_list:
            PathManagerFactory.pm_list[key] = PathManager()
            if defaults_setup:
                try:
                    from iopath.common.setup_defaults import setup_defaults

                    setup_defaults(PathManagerFactory.pm_list[key])
                except ImportError:
                    pass

        return PathManagerFactory.pm_list[key]

    @staticmethod
    def remove(key):
        """
        Remove the path manager instance associated with a key.
        Args:
            key (str):
        """
        if key in PathManagerFactory.pm_list:
            _pm = PathManagerFactory.pm_list.pop(key)  # noqa: F841
            del _pm


"""
A global instance of PathManager.
This global instance is provided for backward compatibility, but it is
recommended that clients use PathManagerFactory
"""
g_pathmgr = PathManagerFactory.get(defaults_setup=True)
