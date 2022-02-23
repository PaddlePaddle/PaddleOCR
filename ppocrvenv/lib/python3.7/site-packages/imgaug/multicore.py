"""Classes and functions dealing with augmentation on multiple CPU cores."""
from __future__ import print_function, division, absolute_import
import sys
import multiprocessing
import threading
import traceback
import time
import random
import platform

import numpy as np
import cv2

import imgaug.imgaug as ia
import imgaug.random as iarandom
from imgaug.augmentables.batches import Batch, UnnormalizedBatch

if sys.version_info[0] == 2:
    # pylint: disable=redefined-builtin, import-error
    import cPickle as pickle
    from Queue import Empty as QueueEmpty, Full as QueueFull
    import socket
    BrokenPipeError = socket.error
elif sys.version_info[0] == 3:
    import pickle
    from queue import Empty as QueueEmpty, Full as QueueFull


_CONTEXT = None


# Added in 0.4.0.
def _get_context_method():
    vinfo = sys.version_info

    # get_context() is only supported in 3.5 and later (same for
    # set_start_method)
    get_context_unsupported = (
        vinfo[0] == 2
        or (vinfo[0] == 3 and vinfo[1] <= 3))

    method = None
    # Fix random hanging code in NixOS by switching to spawn method,
    # see issue #414
    # TODO This is only a workaround and doesn't really fix the underlying
    #      issue. The cause of the underlying issue is currently unknown.
    #      Its possible that #535 fixes the issue, though earlier tests
    #      indicated that the cause was something else.
    # TODO this might break the semaphore used to prevent out of memory
    #      errors
    if "NixOS" in platform.version():
        method = "spawn"
        if get_context_unsupported:
            ia.warn("Detected usage of imgaug.multicore in python <=3.4 "
                    "and NixOS. This is known to sometimes cause endlessly "
                    "hanging programs when also making use of multicore "
                    "augmentation (aka background augmentation). Use "
                    "python 3.5 or later to prevent this.")

    if get_context_unsupported:
        return False
    return method


# Added in 0.4.0.
def _set_context(method):
    # method=False indicates that multiprocessing module (i.e. no context)
    # should be used, e.g. because get_context() is not supported
    globals()["_CONTEXT"] = (
        multiprocessing if method is False
        else multiprocessing.get_context(method))


# Added in 0.4.0.
def _reset_context():
    globals()["_CONTEXT"] = None


# Added in 0.4.0.
def _autoset_context():
    _set_context(_get_context_method())


# Added in 0.4.0.
def _get_context():
    if _CONTEXT is None:
        _autoset_context()
    return _CONTEXT


class Pool(object):
    """
    Wrapper around ``multiprocessing.Pool`` for multicore augmentation.

    Parameters
    ----------
    augseq : imgaug.augmenters.meta.Augmenter
        The augmentation sequence to apply to batches.

    processes : None or int, optional
        The number of background workers, similar to the same parameter in
        multiprocessing.Pool. If ``None``, the number of the machine's CPU
        cores will be used (this counts hyperthreads as CPU cores). If this is
        set to a negative value ``p``, then ``P - abs(p)`` will be used,
        where ``P`` is the number of CPU cores. E.g. ``-1`` would use all
        cores except one (this is useful to e.g. reserve one core to feed
        batches to the GPU).

    maxtasksperchild : None or int, optional
        The number of tasks done per worker process before the process is
        killed and restarted, similar to the same parameter in
        multiprocessing.Pool. If ``None``, worker processes will not be
        automatically restarted.

    seed : None or int, optional
        The seed to use for child processes. If ``None``, a random seed will
        be used.

    """
    # This attribute saves the augmentation sequence for background workers so
    # that it does not have to be resend with every batch. The attribute is set
    # once per worker in the worker's initializer. As each worker has its own
    # process, it is a different variable per worker (though usually should be
    # of equal content).
    _WORKER_AUGSEQ = None

    # This attribute saves the initial seed for background workers so that for
    # any future batch the batch's specific seed can be derived, roughly via
    # SEED_START+SEED_BATCH. As each worker has its own process, this seed can
    # be unique per worker even though all seemingly use the same constant
    # attribute.
    _WORKER_SEED_START = None

    def __init__(self, augseq, processes=None, maxtasksperchild=None,
                 seed=None):
        # make sure that don't call pool again in a child process
        assert Pool._WORKER_AUGSEQ is None, (
            "_WORKER_AUGSEQ was already set when calling Pool.__init__(). "
            "Did you try to instantiate a Pool within a Pool?")
        assert processes is None or processes != 0, (
            "Expected `processes` to be `None` (\"use as many cores as "
            "available\") or a negative integer (\"use as many as available "
            "MINUS this number\") or an integer>1 (\"use exactly that many "
            "processes\"). Got type %s, value %s instead." % (
                type(processes), str(processes))
        )

        self.augseq = augseq
        self.processes = processes
        self.maxtasksperchild = maxtasksperchild

        if seed is not None:
            assert iarandom.SEED_MIN_VALUE <= seed <= iarandom.SEED_MAX_VALUE, (
                "Expected `seed` to be either `None` or a value between "
                "%d and %d. Got type %s, value %s instead." % (
                    iarandom.SEED_MIN_VALUE,
                    iarandom.SEED_MAX_VALUE,
                    type(seed),
                    str(seed)
                )
            )
        self.seed = seed

        # multiprocessing.Pool instance
        self._pool = None

        # Running counter of the number of augmented batches. This will be
        # used to send indexes for each batch to the workers so that they can
        # augment using SEED_BASE+SEED_BATCH and ensure consistency of applied
        # augmentation order between script runs.
        self._batch_idx = 0

    @property
    def pool(self):
        """Return or create the ``multiprocessing.Pool`` instance.

        This creates a new instance upon the first call and afterwards
        returns that instance (until the property ``_pool`` is set to
        ``None`` again).

        Returns
        -------
        multiprocessing.Pool
            The ``multiprocessing.Pool`` used internally by this
            ``imgaug.multicore.Pool``.

        """
        if self._pool is None:
            processes = self.processes
            if processes is not None and processes < 0:
                # cpu count returns the number of logical cpu cores, i.e.
                # including hyperthreads could also use
                # os.sched_getaffinity(0) here, which seems to not exist on
                # BSD though.
                # In python 3.4+, there is also os.cpu_count(), which
                # multiprocessing.cpu_count() then redirects to.
                # At least one guy on stackoverflow.com/questions/1006289
                # reported that only os.* existed, not the multiprocessing
                # method.
                # TODO make this also check if os.cpu_count exists as a
                #      fallback
                try:
                    processes = _get_context().cpu_count() - abs(processes)
                    processes = max(processes, 1)
                except (ImportError, NotImplementedError):
                    ia.warn(
                        "Could not find method multiprocessing.cpu_count(). "
                        "This will likely lead to more CPU cores being used "
                        "for the background augmentation than originally "
                        "intended.")
                    processes = None

            self._pool = _get_context().Pool(
                processes,
                initializer=_Pool_initialize_worker,
                initargs=(self.augseq, self.seed),
                maxtasksperchild=self.maxtasksperchild)
        return self._pool

    def map_batches(self, batches, chunksize=None):
        """
        Augment a list of batches.

        Parameters
        ----------
        batches : list of imgaug.augmentables.batches.Batch
            The batches to augment.

        chunksize : None or int, optional
            Rough indicator of how many tasks should be sent to each worker.
            Increasing this number can improve performance.

        Returns
        -------
        list of imgaug.augmentables.batches.Batch
            Augmented batches.

        """
        self._assert_batches_is_list(batches)
        return self.pool.map(
            _Pool_starworker,
            self._handle_batch_ids(batches),
            chunksize=chunksize)

    def map_batches_async(self, batches, chunksize=None, callback=None,
                          error_callback=None):
        """
        Augment batches asynchonously.

        Parameters
        ----------
        batches : list of imgaug.augmentables.batches.Batch
            The batches to augment.

        chunksize : None or int, optional
            Rough indicator of how many tasks should be sent to each worker.
            Increasing this number can improve performance.

        callback : None or callable, optional
            Function to call upon finish. See ``multiprocessing.Pool``.

        error_callback : None or callable, optional
            Function to call upon errors. See ``multiprocessing.Pool``.

        Returns
        -------
        multiprocessing.MapResult
            Asynchonous result. See ``multiprocessing.Pool``.

        """
        self._assert_batches_is_list(batches)
        return self.pool.map_async(
            _Pool_starworker,
            self._handle_batch_ids(batches),
            chunksize=chunksize,
            callback=callback,
            error_callback=error_callback)

    @classmethod
    def _assert_batches_is_list(cls, batches):
        assert isinstance(batches, list), (
            "Expected `batches` to be a list, got type %s. Call "
            "imap_batches() if you use generators.") % (type(batches),)

    def imap_batches(self, batches, chunksize=1, output_buffer_size=None):
        """
        Augment batches from a generator.

        Pattern for output buffer constraint is from
        https://stackoverflow.com/a/47058399.

        Parameters
        ----------
        batches : generator of imgaug.augmentables.batches.Batch
            The batches to augment, provided as a generator. Each call to the
            generator should yield exactly one batch.

        chunksize : None or int, optional
            Rough indicator of how many tasks should be sent to each worker.
            Increasing this number can improve performance.

        output_buffer_size : None or int, optional
            Max number of batches to handle *at the same time* in the *whole*
            pipeline (including already augmented batches that are waiting to
            be requested). If the buffer size is reached, no new batches will
            be loaded from `batches` until a produced (i.e. augmented) batch is
            consumed (i.e. requested from this method).
            The buffer is unlimited if this is set to ``None``. For large
            datasets, this should be set to an integer value to avoid filling
            the whole RAM if loading+augmentation happens faster than training.

            *New in version 0.3.0.*

        Yields
        ------
        imgaug.augmentables.batches.Batch
            Augmented batch.

        """
        self._assert_batches_is_generator(batches)

        # buffer is either None or a Semaphore
        output_buffer_left = _create_output_buffer_left(output_buffer_size)

        # TODO change this to 'yield from' once switched to 3.3+
        gen = self.pool.imap(
            _Pool_starworker,
            self._ibuffer_batch_loading(
                self._handle_batch_ids_gen(batches),
                output_buffer_left
            ),
            chunksize=chunksize)

        for batch in gen:
            yield batch
            if output_buffer_left is not None:
                output_buffer_left.release()

    def imap_batches_unordered(self, batches, chunksize=1,
                               output_buffer_size=None):
        """Augment batches from a generator (without preservation of order).

        Pattern for output buffer constraint is from
        https://stackoverflow.com/a/47058399.

        Parameters
        ----------
        batches : generator of imgaug.augmentables.batches.Batch
            The batches to augment, provided as a generator. Each call to the
            generator should yield exactly one batch.

        chunksize : None or int, optional
            Rough indicator of how many tasks should be sent to each worker.
            Increasing this number can improve performance.

        output_buffer_size : None or int, optional
            Max number of batches to handle *at the same time* in the *whole*
            pipeline (including already augmented batches that are waiting to
            be requested). If the buffer size is reached, no new batches will
            be loaded from `batches` until a produced (i.e. augmented) batch is
            consumed (i.e. requested from this method).
            The buffer is unlimited if this is set to ``None``. For large
            datasets, this should be set to an integer value to avoid filling
            the whole RAM if loading+augmentation happens faster than training.

            *New in version 0.3.0.*

        Yields
        ------
        imgaug.augmentables.batches.Batch
            Augmented batch.

        """
        self._assert_batches_is_generator(batches)

        # buffer is either None or a Semaphore
        output_buffer_left = _create_output_buffer_left(output_buffer_size)

        gen = self.pool.imap_unordered(
            _Pool_starworker,
            self._ibuffer_batch_loading(
                self._handle_batch_ids_gen(batches),
                output_buffer_left
            ),
            chunksize=chunksize
        )

        for batch in gen:
            yield batch
            if output_buffer_left is not None:
                output_buffer_left.release()

    @classmethod
    def _assert_batches_is_generator(cls, batches):
        assert ia.is_generator(batches), (
            "Expected `batches` to be generator, got type %s. Call "
            "map_batches() if you use lists.") % (type(batches),)

    def __enter__(self):
        assert self._pool is None, (
            "Tried to __enter__ a pool that has already been initialized.")
        _ = self.pool  # initialize internal multiprocessing pool instance
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the pool gracefully."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def terminate(self):
        """Terminate the pool immediately."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    # TODO why does this function exist if it may only be called after
    #      close/terminate and both of these two already call join() themselves
    def join(self):
        """
        Wait for the workers to exit.

        This may only be called after first calling
        :func:`~imgaug.multicore.Pool.close` or
        :func:`~imgaug.multicore.Pool.terminate`.

        """
        if self._pool is not None:
            self._pool.join()

    def _handle_batch_ids(self, batches):
        ids = np.arange(self._batch_idx, self._batch_idx + len(batches))
        inputs = list(zip(ids, batches))
        self._batch_idx += len(batches)
        return inputs

    def _handle_batch_ids_gen(self, batches):
        for batch in batches:
            batch_idx = self._batch_idx
            yield batch_idx, batch
            self._batch_idx += 1

    @classmethod
    def _ibuffer_batch_loading(cls, batches, output_buffer_left):
        for batch in batches:
            if output_buffer_left is not None:
                output_buffer_left.acquire()
            yield batch


def _create_output_buffer_left(output_buffer_size):
    output_buffer_left = None
    if output_buffer_size:
        assert output_buffer_size > 0, (
            "Expected buffer size to be greater than zero, but got size %d "
            "instead." % (output_buffer_size,))
        output_buffer_left = _get_context().Semaphore(output_buffer_size)
    return output_buffer_left


# This could be a classmethod or staticmethod of Pool in 3.x, but in 2.7 that
# leads to pickle errors.
def _Pool_initialize_worker(augseq, seed_start):
    # pylint: disable=invalid-name, protected-access

    # Not using this seems to have caused infinite hanging in the case
    # of gaussian blur on at least MacOSX.
    # It is also in most cases probably not sensible to use multiple
    # threads while already running augmentation in multiple processes.
    cv2.setNumThreads(0)

    if seed_start is None:
        # pylint falsely thinks in older versions that
        # multiprocessing.current_process() was not callable, see
        # https://github.com/PyCQA/pylint/issues/1699
        # pylint: disable=not-callable
        process_name = _get_context().current_process().name
        # pylint: enable=not-callable

        # time_ns() exists only in 3.7+
        if sys.version_info[0] == 3 and sys.version_info[1] >= 7:
            seed_offset = time.time_ns()
        else:
            seed_offset = int(time.time() * 10**6) % 10**6
        seed = hash(process_name) + seed_offset
        _reseed_global_local(seed, augseq)
    Pool._WORKER_SEED_START = seed_start
    Pool._WORKER_AUGSEQ = augseq
    # not sure if really necessary, but shouldn't hurt either
    Pool._WORKER_AUGSEQ.localize_random_state_()


# This could be a classmethod or staticmethod of Pool in 3.x, but in 2.7 that
# leads to pickle errors.
def _Pool_worker(batch_idx, batch):
    # pylint: disable=invalid-name, protected-access
    assert ia.is_single_integer(batch_idx), (
        "Expected `batch_idx` to be an integer. Got type %s instead." % (
            type(batch_idx)
        ))
    assert isinstance(batch, (UnnormalizedBatch, Batch)), (
        "Expected `batch` to be either an instance of "
        "`imgaug.augmentables.batches.UnnormalizedBatch` or "
        "`imgaug.augmentables.batches.Batch`. Got type %s instead." % (
            type(batch)
        ))
    assert Pool._WORKER_AUGSEQ is not None, (
        "Expected `Pool._WORKER_AUGSEQ` to NOT be `None`. Did you manually "
        "call _Pool_worker()?")

    augseq = Pool._WORKER_AUGSEQ
    # TODO why is this if here? _WORKER_SEED_START should always be set?
    if Pool._WORKER_SEED_START is not None:
        seed = Pool._WORKER_SEED_START + batch_idx
        _reseed_global_local(seed, augseq)
    result = augseq.augment_batch_(batch)
    return result


# could be a classmethod or staticmethod of Pool in 3.x, but in 2.7 that leads
# to pickle errors starworker is here necessary, because starmap does not exist
# in 2.7
def _Pool_starworker(inputs):
    # pylint: disable=invalid-name
    return _Pool_worker(*inputs)


def _reseed_global_local(base_seed, augseq):
    seed_global = _derive_seed(base_seed, -10**9)
    seed_local = _derive_seed(base_seed)
    iarandom.seed(seed_global)
    augseq.seed_(seed_local)


def _derive_seed(base_seed, offset=0):
    return (
        iarandom.SEED_MIN_VALUE
        + (base_seed + offset)
        % (iarandom.SEED_MAX_VALUE - iarandom.SEED_MIN_VALUE)
    )


class BatchLoader(object):
    """**Deprecated**. Load batches in the background.

    Deprecated. Use ``imgaug.multicore.Pool`` instead.

    Loaded batches can be accesses using :attr:`imgaug.BatchLoader.queue`.

    Parameters
    ----------
    load_batch_func : callable or generator
        Generator or generator function (i.e. function that yields Batch
        objects) or a function that returns a list of Batch objects.
        Background loading automatically stops when the last batch was yielded
        or the last batch in the list was reached.

    queue_size : int, optional
        Maximum number of batches to store in the queue. May be set higher
        for small images and/or small batches.

    nb_workers : int, optional
        Number of workers to run in the background.

    threaded : bool, optional
        Whether to run the background processes using threads (True) or full
        processes (False).

    """

    @ia.deprecated(alt_func="imgaug.multicore.Pool")
    def __init__(self, load_batch_func, queue_size=50, nb_workers=1,
                 threaded=True):
        assert queue_size >= 2, (
            "Queue size for BatchLoader must be at least 2, "
            "got %d." % (queue_size,))
        assert nb_workers >= 1, (
            "Number of workers for BatchLoader must be at least 1, "
            "got %d" % (nb_workers,))
        self._queue_internal = multiprocessing.Queue(queue_size//2)
        self.queue = multiprocessing.Queue(queue_size//2)
        self.join_signal = multiprocessing.Event()
        self.workers = []
        self.threaded = threaded
        seeds = iarandom.get_global_rng().generate_seeds_(nb_workers)
        for i in range(nb_workers):
            if threaded:
                worker = threading.Thread(
                    target=self._load_batches,
                    args=(load_batch_func, self._queue_internal,
                          self.join_signal, None)
                )
            else:
                worker = multiprocessing.Process(
                    target=self._load_batches,
                    args=(load_batch_func, self._queue_internal,
                          self.join_signal, seeds[i])
                )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

        self.main_worker_thread = threading.Thread(
            target=self._main_worker,
            args=()
        )
        self.main_worker_thread.daemon = True
        self.main_worker_thread.start()

    def count_workers_alive(self):
        return sum([int(worker.is_alive()) for worker in self.workers])

    def all_finished(self):
        """
        Determine whether the workers have finished the loading process.

        Returns
        -------
        out : bool
            True if all workers have finished. Else False.

        """
        return self.count_workers_alive() == 0

    def _main_worker(self):
        workers_running = self.count_workers_alive()

        while workers_running > 0 and not self.join_signal.is_set():
            # wait for a new batch in the source queue and load it
            try:
                batch_str = self._queue_internal.get(timeout=0.1)
                if batch_str == "":
                    workers_running -= 1
                else:
                    self.queue.put(batch_str)
            except QueueEmpty:
                time.sleep(0.01)
            except (EOFError, BrokenPipeError):
                break

            workers_running = self.count_workers_alive()

        # All workers have finished, move the remaining entries from internal
        # to external queue
        while True:
            try:
                batch_str = self._queue_internal.get(timeout=0.005)
                if batch_str != "":
                    self.queue.put(batch_str)
            except QueueEmpty:
                break
            except (EOFError, BrokenPipeError):
                break

        self.queue.put(pickle.dumps(None, protocol=-1))
        time.sleep(0.01)

    @classmethod
    def _load_batches(cls, load_batch_func, queue_internal, join_signal,
                      seedval):
        # pylint: disable=broad-except
        if seedval is not None:
            random.seed(seedval)
            np.random.seed(seedval)
            iarandom.seed(seedval)

        try:
            gen = (
                load_batch_func()
                if not ia.is_generator(load_batch_func)
                else load_batch_func
            )
            for batch in gen:
                assert isinstance(batch, Batch), (
                    "Expected batch returned by load_batch_func to "
                    "be of class imgaug.Batch, got %s." % (
                        type(batch),))
                batch_pickled = pickle.dumps(batch, protocol=-1)
                while not join_signal.is_set():
                    try:
                        queue_internal.put(batch_pickled, timeout=0.005)
                        break
                    except QueueFull:
                        pass
                if join_signal.is_set():
                    break
        except Exception:
            traceback.print_exc()
        finally:
            queue_internal.put("")
        time.sleep(0.01)

    def terminate(self):
        """Stop all workers."""
        # pylint: disable=protected-access
        if not self.join_signal.is_set():
            self.join_signal.set()
        # give minimal time to put generated batches in queue and gracefully
        # shut down
        time.sleep(0.01)

        if self.main_worker_thread.is_alive():
            self.main_worker_thread.join()

        if self.threaded:
            for worker in self.workers:
                if worker.is_alive():
                    worker.join()
        else:
            for worker in self.workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()

            # wait until all workers are fully terminated
            while not self.all_finished():
                time.sleep(0.001)

        # empty queue until at least one element can be added and place None
        # as signal that BL finished
        if self.queue.full():
            self.queue.get()
        self.queue.put(pickle.dumps(None, protocol=-1))
        time.sleep(0.01)

        # clean the queue, this reportedly prevents hanging threads
        while True:
            try:
                self._queue_internal.get(timeout=0.005)
            except QueueEmpty:
                break

        if not self._queue_internal._closed:
            self._queue_internal.close()
        if not self.queue._closed:
            self.queue.close()
        self._queue_internal.join_thread()
        self.queue.join_thread()
        time.sleep(0.025)

    def __del__(self):
        if not self.join_signal.is_set():
            self.join_signal.set()


class BackgroundAugmenter(object):
    """
    **Deprecated**. Augment batches in the background processes.

    Deprecated. Use ``imgaug.multicore.Pool`` instead.

    This is a wrapper around the multiprocessing module.

    Parameters
    ----------
    batch_loader : BatchLoader or multiprocessing.Queue
        BatchLoader object that loads the data fed into the
        BackgroundAugmenter, or alternatively a Queue. If a Queue, then it
        must be made sure that a final ``None`` in the Queue signals that the
        loading is finished and no more batches will follow. Otherwise the
        BackgroundAugmenter will wait forever for the next batch.

    augseq : Augmenter
        An augmenter to apply to all loaded images.
        This may be e.g. a Sequential to apply multiple augmenters.

    queue_size : int
        Size of the queue that is used to temporarily save the augmentation
        results. Larger values offer the background processes more room
        to save results when the main process doesn't load much, i.e. they
        can lead to smoother and faster training. For large images, high
        values can block a lot of RAM though.

    nb_workers : 'auto' or int
        Number of background workers to spawn.
        If ``auto``, it will be set to ``C-1``, where ``C`` is the number of
        CPU cores.

    """

    @ia.deprecated(alt_func="imgaug.multicore.Pool")
    def __init__(self, batch_loader, augseq, queue_size=50, nb_workers="auto"):
        assert queue_size > 0, (
            "Expected 'queue_size' to be at least 1, got %d." % (queue_size,))
        self.augseq = augseq
        self.queue_source = (
            batch_loader
            if isinstance(batch_loader, multiprocessing.queues.Queue)
            else batch_loader.queue
        )
        self.queue_result = multiprocessing.Queue(queue_size)

        if nb_workers == "auto":
            try:
                nb_workers = multiprocessing.cpu_count()
            except (ImportError, NotImplementedError):
                nb_workers = 1
            # try to reserve at least one core for the main process
            nb_workers = max(1, nb_workers - 1)
        else:
            assert nb_workers >= 1, (
                "Expected 'nb_workers' to be \"auto\" or at least 1, "
                "got %d instead." % (nb_workers,))

        self.nb_workers = nb_workers
        self.workers = []
        self.nb_workers_finished = 0

        seeds = iarandom.get_global_rng().generate_seeds_(nb_workers)
        for i in range(nb_workers):
            worker = multiprocessing.Process(
                target=self._augment_images_worker,
                args=(augseq, self.queue_source, self.queue_result, seeds[i])
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def all_finished(self):
        return self.nb_workers_finished == self.nb_workers

    def get_batch(self):
        """
        Returns a batch from the queue of augmented batches.

        If workers are still running and there are no batches in the queue,
        it will automatically wait for the next batch.

        Returns
        -------
        out : None or imgaug.Batch
            One batch or None if all workers have finished.

        """
        if self.all_finished():
            return None

        batch_str = self.queue_result.get()
        batch = pickle.loads(batch_str)
        if batch is not None:
            return batch

        self.nb_workers_finished += 1
        if self.nb_workers_finished >= self.nb_workers:
            try:
                # remove `None` from the source queue
                self.queue_source.get(timeout=0.001)
            except QueueEmpty:
                pass
            return None
        return self.get_batch()

    @classmethod
    def _augment_images_worker(cls, augseq, queue_source, queue_result,
                               seedval):
        """
        Augment endlessly images in the source queue.

        This is a worker function for that endlessly queries the source queue
        (input batches), augments batches in it and sends the result to the
        output queue.

        """
        np.random.seed(seedval)
        random.seed(seedval)
        augseq.seed_(seedval)
        iarandom.seed(seedval)

        loader_finished = False

        while not loader_finished:
            # wait for a new batch in the source queue and load it
            try:
                batch_str = queue_source.get(timeout=0.1)
                batch = pickle.loads(batch_str)
                if batch is None:
                    loader_finished = True
                    # put it back in so that other workers know that the
                    # loading queue is finished
                    queue_source.put(pickle.dumps(None, protocol=-1))
                else:
                    batch_aug = augseq.augment_batch_(batch)

                    # send augmented batch to output queue
                    batch_str = pickle.dumps(batch_aug, protocol=-1)
                    queue_result.put(batch_str)
            except QueueEmpty:
                time.sleep(0.01)

        queue_result.put(pickle.dumps(None, protocol=-1))
        time.sleep(0.01)

    def terminate(self):
        """
        Terminates all background processes immediately.

        This will also free their RAM.

        """
        # pylint: disable=protected-access
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
        self.nb_workers_finished = len(self.workers)

        if not self.queue_result._closed:
            self.queue_result.close()
        time.sleep(0.01)

    def __del__(self):
        time.sleep(0.1)
        self.terminate()
