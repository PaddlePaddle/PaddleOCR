"""Custom implementation of multiprocessing.Pool with custom pickler.

This module provides efficient ways of working with data stored in
shared memory with numpy.memmap arrays without inducing any memory
copy between the parent and child processes.

This module should not be imported if multiprocessing is not
available as it implements subclasses of multiprocessing Pool
that uses a custom alternative to SimpleQueue.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# Copyright: 2012, Olivier Grisel
# License: BSD 3 clause

import copyreg
import sys
import warnings
from time import sleep

try:
    WindowsError
except NameError:
    WindowsError = type(None)

from pickle import Pickler

from pickle import HIGHEST_PROTOCOL
from io import BytesIO

from ._memmapping_reducer import get_memmapping_reducers
from ._memmapping_reducer import TemporaryResourcesManager
from ._multiprocessing_helpers import mp, assert_spawning

# We need the class definition to derive from it, not the multiprocessing.Pool
# factory function
from multiprocessing.pool import Pool

try:
    import numpy as np
except ImportError:
    np = None


###############################################################################
# Enable custom pickling in Pool queues

class CustomizablePickler(Pickler):
    """Pickler that accepts custom reducers.

    TODO python2_drop : can this be simplified ?

    HIGHEST_PROTOCOL is selected by default as this pickler is used
    to pickle ephemeral datastructures for interprocess communication
    hence no backward compatibility is required.

    `reducers` is expected to be a dictionary with key/values
    being `(type, callable)` pairs where `callable` is a function that
    give an instance of `type` will return a tuple `(constructor,
    tuple_of_objects)` to rebuild an instance out of the pickled
    `tuple_of_objects` as would return a `__reduce__` method. See the
    standard library documentation on pickling for more details.

    """

    # We override the pure Python pickler as its the only way to be able to
    # customize the dispatch table without side effects in Python 2.7
    # to 3.2. For Python 3.3+ leverage the new dispatch_table
    # feature from https://bugs.python.org/issue14166 that makes it possible
    # to use the C implementation of the Pickler which is faster.

    def __init__(self, writer, reducers=None, protocol=HIGHEST_PROTOCOL):
        Pickler.__init__(self, writer, protocol=protocol)
        if reducers is None:
            reducers = {}
        if hasattr(Pickler, 'dispatch'):
            # Make the dispatch registry an instance level attribute instead of
            # a reference to the class dictionary under Python 2
            self.dispatch = Pickler.dispatch.copy()
        else:
            # Under Python 3 initialize the dispatch table with a copy of the
            # default registry
            self.dispatch_table = copyreg.dispatch_table.copy()
        for type, reduce_func in reducers.items():
            self.register(type, reduce_func)

    def register(self, type, reduce_func):
        """Attach a reducer function to a given type in the dispatch table."""
        if hasattr(Pickler, 'dispatch'):
            # Python 2 pickler dispatching is not explicitly customizable.
            # Let us use a closure to workaround this limitation.
            def dispatcher(self, obj):
                reduced = reduce_func(obj)
                self.save_reduce(obj=obj, *reduced)
            self.dispatch[type] = dispatcher
        else:
            self.dispatch_table[type] = reduce_func


class CustomizablePicklingQueue(object):
    """Locked Pipe implementation that uses a customizable pickler.

    This class is an alternative to the multiprocessing implementation
    of SimpleQueue in order to make it possible to pass custom
    pickling reducers, for instance to avoid memory copy when passing
    memory mapped datastructures.

    `reducers` is expected to be a dict with key / values being
    `(type, callable)` pairs where `callable` is a function that, given an
    instance of `type`, will return a tuple `(constructor, tuple_of_objects)`
    to rebuild an instance out of the pickled `tuple_of_objects` as would
    return a `__reduce__` method.

    See the standard library documentation on pickling for more details.
    """

    def __init__(self, context, reducers=None):
        self._reducers = reducers
        self._reader, self._writer = context.Pipe(duplex=False)
        self._rlock = context.Lock()
        if sys.platform == 'win32':
            self._wlock = None
        else:
            self._wlock = context.Lock()
        self._make_methods()

    def __getstate__(self):
        assert_spawning(self)
        return (self._reader, self._writer, self._rlock, self._wlock,
                self._reducers)

    def __setstate__(self, state):
        (self._reader, self._writer, self._rlock, self._wlock,
         self._reducers) = state
        self._make_methods()

    def empty(self):
        return not self._reader.poll()

    def _make_methods(self):
        self._recv = recv = self._reader.recv
        racquire, rrelease = self._rlock.acquire, self._rlock.release

        def get():
            racquire()
            try:
                return recv()
            finally:
                rrelease()

        self.get = get

        if self._reducers:
            def send(obj):
                buffer = BytesIO()
                CustomizablePickler(buffer, self._reducers).dump(obj)
                self._writer.send_bytes(buffer.getvalue())
            self._send = send
        else:
            self._send = send = self._writer.send
        if self._wlock is None:
            # writes to a message oriented win32 pipe are atomic
            self.put = send
        else:
            wlock_acquire, wlock_release = (
                self._wlock.acquire, self._wlock.release)

            def put(obj):
                wlock_acquire()
                try:
                    return send(obj)
                finally:
                    wlock_release()

            self.put = put


class PicklingPool(Pool):
    """Pool implementation with customizable pickling reducers.

    This is useful to control how data is shipped between processes
    and makes it possible to use shared memory without useless
    copies induces by the default pickling methods of the original
    objects passed as arguments to dispatch.

    `forward_reducers` and `backward_reducers` are expected to be
    dictionaries with key/values being `(type, callable)` pairs where
    `callable` is a function that, given an instance of `type`, will return a
    tuple `(constructor, tuple_of_objects)` to rebuild an instance out of the
    pickled `tuple_of_objects` as would return a `__reduce__` method.
    See the standard library documentation about pickling for more details.

    """

    def __init__(self, processes=None, forward_reducers=None,
                 backward_reducers=None, **kwargs):
        if forward_reducers is None:
            forward_reducers = dict()
        if backward_reducers is None:
            backward_reducers = dict()
        self._forward_reducers = forward_reducers
        self._backward_reducers = backward_reducers
        poolargs = dict(processes=processes)
        poolargs.update(kwargs)
        super(PicklingPool, self).__init__(**poolargs)

    def _setup_queues(self):
        context = getattr(self, '_ctx', mp)
        self._inqueue = CustomizablePicklingQueue(context,
                                                  self._forward_reducers)
        self._outqueue = CustomizablePicklingQueue(context,
                                                   self._backward_reducers)
        self._quick_put = self._inqueue._send
        self._quick_get = self._outqueue._recv


class MemmappingPool(PicklingPool):
    """Process pool that shares large arrays to avoid memory copy.

    This drop-in replacement for `multiprocessing.pool.Pool` makes
    it possible to work efficiently with shared memory in a numpy
    context.

    Existing instances of numpy.memmap are preserved: the child
    suprocesses will have access to the same shared memory in the
    original mode except for the 'w+' mode that is automatically
    transformed as 'r+' to avoid zeroing the original data upon
    instantiation.

    Furthermore large arrays from the parent process are automatically
    dumped to a temporary folder on the filesystem such as child
    processes to access their content via memmapping (file system
    backed shared memory).

    Note: it is important to call the terminate method to collect
    the temporary folder used by the pool.

    Parameters
    ----------
    processes: int, optional
        Number of worker processes running concurrently in the pool.
    initializer: callable, optional
        Callable executed on worker process creation.
    initargs: tuple, optional
        Arguments passed to the initializer callable.
    temp_folder: (str, callable) optional
        If str:
          Folder to be used by the pool for memmapping large arrays
          for sharing memory with worker processes. If None, this will try in
          order:
          - a folder pointed by the JOBLIB_TEMP_FOLDER environment variable,
          - /dev/shm if the folder exists and is writable: this is a RAMdisk
            filesystem available by default on modern Linux distributions,
          - the default system temporary folder that can be overridden
            with TMP, TMPDIR or TEMP environment variables, typically /tmp
            under Unix operating systems.
        if callable:
            An callable in charge of dynamically resolving a temporary folder
            for memmapping large arrays.
    max_nbytes int or None, optional, 1e6 by default
        Threshold on the size of arrays passed to the workers that
        triggers automated memory mapping in temp_folder.
        Use None to disable memmapping of large arrays.
    mmap_mode: {'r+', 'r', 'w+', 'c'}
        Memmapping mode for numpy arrays passed to workers.
        See 'max_nbytes' parameter documentation for more details.
    forward_reducers: dictionary, optional
        Reducers used to pickle objects passed from master to worker
        processes: see below.
    backward_reducers: dictionary, optional
        Reducers used to pickle return values from workers back to the
        master process.
    verbose: int, optional
        Make it possible to monitor how the communication of numpy arrays
        with the subprocess is handled (pickling or memmapping)
    prewarm: bool or str, optional, "auto" by default.
        If True, force a read on newly memmapped array to make sure that OS
        pre-cache it in memory. This can be useful to avoid concurrent disk
        access when the same data array is passed to different worker
        processes. If "auto" (by default), prewarm is set to True, unless the
        Linux shared memory partition /dev/shm is available and used as temp
        folder.

    `forward_reducers` and `backward_reducers` are expected to be
    dictionaries with key/values being `(type, callable)` pairs where
    `callable` is a function that give an instance of `type` will return
    a tuple `(constructor, tuple_of_objects)` to rebuild an instance out
    of the pickled `tuple_of_objects` as would return a `__reduce__`
    method. See the standard library documentation on pickling for more
    details.

    """

    def __init__(self, processes=None, temp_folder=None, max_nbytes=1e6,
                 mmap_mode='r', forward_reducers=None, backward_reducers=None,
                 verbose=0, context_id=None, prewarm=False, **kwargs):

        if context_id is not None:
            warnings.warn('context_id is deprecated and ignored in joblib'
                          ' 0.9.4 and will be removed in 0.11',
                          DeprecationWarning)

        manager = TemporaryResourcesManager(temp_folder)
        self._temp_folder_manager = manager

        # The usage of a temp_folder_resolver over a simple temp_folder is
        # superfluous for multiprocessing pools, as they don't get reused, see
        # get_memmapping_executor for more details. We still use it for code
        # simplicity.
        forward_reducers, backward_reducers = \
            get_memmapping_reducers(
                temp_folder_resolver=manager.resolve_temp_folder_name,
                max_nbytes=max_nbytes, mmap_mode=mmap_mode,
                forward_reducers=forward_reducers,
                backward_reducers=backward_reducers, verbose=verbose,
                unlink_on_gc_collect=False, prewarm=prewarm)

        poolargs = dict(
            processes=processes,
            forward_reducers=forward_reducers,
            backward_reducers=backward_reducers)
        poolargs.update(kwargs)
        super(MemmappingPool, self).__init__(**poolargs)

    def terminate(self):
        n_retries = 10
        for i in range(n_retries):
            try:
                super(MemmappingPool, self).terminate()
                break
            except OSError as e:
                if isinstance(e, WindowsError):
                    # Workaround  occasional "[Error 5] Access is denied" issue
                    # when trying to terminate a process under windows.
                    sleep(0.1)
                    if i + 1 == n_retries:
                        warnings.warn("Failed to terminate worker processes in"
                                      " multiprocessing pool: %r" % e)
        self._temp_folder_manager._unlink_temporary_resources()

    @property
    def _temp_folder(self):
        # Legacy property in tests. could be removed if we refactored the
        # memmapping tests. SHOULD ONLY BE USED IN TESTS!
        # We cache this property because it is called late in the tests - at
        # this point, all context have been unregistered, and
        # resolve_temp_folder_name raises an error.
        if getattr(self, '_cached_temp_folder', None) is not None:
            return self._cached_temp_folder
        else:
            self._cached_temp_folder = self._temp_folder_manager.resolve_temp_folder_name()  # noqa
            return self._cached_temp_folder
