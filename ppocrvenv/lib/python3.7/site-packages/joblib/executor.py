"""Utility function to construct a loky.ReusableExecutor with custom pickler.

This module provides efficient ways of working with data stored in
shared memory with numpy.memmap arrays without inducing any memory
copy between the parent and child processes.
"""
# Author: Thomas Moreau <thomas.moreau.2010@gmail.com>
# Copyright: 2017, Thomas Moreau
# License: BSD 3 clause

from ._memmapping_reducer import get_memmapping_reducers
from ._memmapping_reducer import TemporaryResourcesManager
from .externals.loky.reusable_executor import _ReusablePoolExecutor


_executor_args = None


def get_memmapping_executor(n_jobs, **kwargs):
    return MemmappingExecutor.get_memmapping_executor(n_jobs, **kwargs)


class MemmappingExecutor(_ReusablePoolExecutor):

    @classmethod
    def get_memmapping_executor(cls, n_jobs, timeout=300, initializer=None,
                                initargs=(), env=None, temp_folder=None,
                                context_id=None, **backend_args):
        """Factory for ReusableExecutor with automatic memmapping for large numpy
        arrays.
        """
        global _executor_args
        # Check if we can reuse the executor here instead of deferring the test
        # to loky as the reducers are objects that changes at each call.
        executor_args = backend_args.copy()
        executor_args.update(env if env else {})
        executor_args.update(dict(
            timeout=timeout, initializer=initializer, initargs=initargs))
        reuse = _executor_args is None or _executor_args == executor_args
        _executor_args = executor_args

        manager = TemporaryResourcesManager(temp_folder)

        # reducers access the temporary folder in which to store temporary
        # pickles through a call to manager.resolve_temp_folder_name. resolving
        # the folder name dynamically is useful to use different folders across
        # calls of a same reusable executor
        job_reducers, result_reducers = get_memmapping_reducers(
            unlink_on_gc_collect=True,
            temp_folder_resolver=manager.resolve_temp_folder_name,
            **backend_args)
        _executor, executor_is_reused = super().get_reusable_executor(
            n_jobs, job_reducers=job_reducers, result_reducers=result_reducers,
            reuse=reuse, timeout=timeout, initializer=initializer,
            initargs=initargs, env=env
        )

        if not executor_is_reused:
            # Only set a _temp_folder_manager for new executors. Reused
            # executors already have a _temporary_folder_manager that must not
            # be re-assigned like that because it is referenced in various
            # places in the reducing machinery of the executor.
            _executor._temp_folder_manager = manager

        if context_id is not None:
            # Only register the specified context once we know which manager
            # the current executor is using, in order to not register an atexit
            # finalizer twice for the same folder.
            _executor._temp_folder_manager.register_new_context(context_id)

        return _executor

    def terminate(self, kill_workers=False):
        self.shutdown(kill_workers=kill_workers)
        if kill_workers:
            # When workers are killed in such a brutal manner, they cannot
            # execute the finalizer of their shared memmaps. The refcount of
            # those memmaps may be off by an unknown number, so instead of
            # decref'ing them, we delete the whole temporary folder, and
            # unregister them. There is no risk of PermissionError at folder
            # deletion because because at this point, all child processes are
            # dead, so all references to temporary memmaps are closed.

            # unregister temporary resources from all contexts
            with self._submit_resize_lock:
                self._temp_folder_manager._unregister_temporary_resources()
                self._temp_folder_manager._try_delete_folder(
                    allow_non_empty=True
                )
        else:
            self._temp_folder_manager._unlink_temporary_resources()
            self._temp_folder_manager._try_delete_folder(allow_non_empty=True)

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


class _TestingMemmappingExecutor(MemmappingExecutor):
    """Wrapper around ReusableExecutor to ease memmapping testing with Pool
    and Executor. This is only for testing purposes.

    """
    def apply_async(self, func, args):
        """Schedule a func to be run"""
        future = self.submit(func, *args)
        future.get = future.result
        return future

    def map(self, f, *args):
        return list(super().map(f, *args))
