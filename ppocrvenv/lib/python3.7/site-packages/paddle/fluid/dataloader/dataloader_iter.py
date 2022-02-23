# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import six
import sys
import time
import signal
import numbers
import logging
import itertools
import threading
import numpy as np
import multiprocessing
from collections import namedtuple
from paddle.fluid.framework import _set_expected_place, _current_expected_place

# NOTE: queue has a different name in python2 and python3
import queue

import paddle
from .. import core, layers
from ..framework import in_dygraph_mode
from ..multiprocess_utils import _set_SIGCHLD_handler, MP_STATUS_CHECK_INTERVAL, CleanupFuncRegistrar
from .fetcher import _IterableDatasetFetcher, _MapDatasetFetcher
from .batch_sampler import _InfiniteIterableSampler
from .collate import default_collate_fn, default_convert_fn
from .worker import ParentWatchDog, get_worker_info, _worker_loop, \
        _DatasetKind, _IterableDatasetStopIteration, _WorkerException, \
        _ResumeIteration
from .flat import _flatten_batch, _restore_batch

__all__ = ['get_worker_info']

# NOTE: fix `terminate called without an active exception`
# if for loop break and program exit immediately(with no model
# layers processing) after iterate **the first few data** in
# distributed lauch mode, distributed launch will call
# terminate() to kill main process on each devices, but thread
# is still iterating to fullfill blocking queue caches, which
# may cause thread error `terminate called without an active
# exception` for terminate is a strong singal and `__del__`
# of DataLoader may not be called, so we add a global link to
# the last DataLoader instance to call `__del__` to clean up
# resources
# NOTE: cannot simply as `__del__` to CleanupFuncRegistrar,
# for this will remain a link to each DataLoader instance in
# global, and will precludes GC to auto collect DataLoader
# instance and will cause memory leak
_loader = None


def _clear_loader():
    global _loader
    if _loader is not None:
        try:
            _loader.__del__()
            del _loader
        except:
            pass


CleanupFuncRegistrar.register(_clear_loader)


class _DataLoaderIterBase(object):
    """
    Iterator implement of DataLoader, will load and feed mini-batch
    data by setting in given dataloader.

    Args:
        loader(instance of DataLoader): instance of `fluid.io.DataLoader`
    """

    def __init__(self, loader):
        self._dataset = loader.dataset
        self._feed_list = loader.feed_list or []
        self._places = loader.places
        self._return_list = loader.return_list
        self._batch_sampler = loader.batch_sampler
        self._drop_last = loader.drop_last
        self._auto_collate_batch = loader.auto_collate_batch
        self._num_workers = loader.num_workers
        self._use_buffer_reader = loader.use_buffer_reader
        self._use_shared_memory = loader.use_shared_memory
        self._timeout = loader.timeout if loader.timeout > 0 else MP_STATUS_CHECK_INTERVAL
        self._worker_init_fn = loader.worker_init_fn
        self._dataset_kind = loader.dataset_kind
        self._pin_memory = loader.pin_memory

        self._sampler_iter = iter(self._index_sampler)
        if self._auto_collate_batch:
            self._collate_fn = loader.collate_fn or default_collate_fn
        else:
            self._collate_fn = loader.collate_fn or default_convert_fn

        # LoDTensorBlockingQueue instance for create_py_reader and a thread
        # to put mini-batch data to self._blocking_queue, mini-batch data
        # will be get from:
        # 1. multi-process mode: get data from workers' result queue
        # 2. single-process mode: read mini-batch data in main process
        self._blocking_queue = None
        self._thread = None
        self._thread_done_event = threading.Event()

    @property
    def _index_sampler(self):
        if self._auto_collate_batch:
            return self._batch_sampler
        else:
            if self._dataset_kind == _DatasetKind.MAP:
                return list(range(len(self._dataset)))
            else:
                return _InfiniteIterableSampler(self._dataset, 1)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._batch_sampler)

    def _exit_thread_expectedly(self):
        self._thread_done_event.set()
        if self._blocking_queue:
            self._blocking_queue.close()

    def _exit_thread_unexpectedly(self):
        self._thread_done_event.set()
        if self._blocking_queue:
            self._blocking_queue.kill()


class _DataLoaderIterSingleProcess(_DataLoaderIterBase):
    """
    Single process implement of DataLoaderIter, loading data from
    loader.data in main process
    """

    def __init__(self, loader):
        super(_DataLoaderIterSingleProcess, self).__init__(loader)

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collate_batch,
            self._collate_fn, self._drop_last)

        # NOTE: _structrue_infos used to record the data structure of
        # batch to restore batch structure after reading Tensor
        # from blocking_queue in single-process mode. Note that
        # only single process is used in single-process mode, we
        # can record the data structure sequencely in a list without
        # recording the send and recv index
        self._structure_infos = []

        # NOTE: len(self._places) batch data compose as an output
        # iteration, set blocking_queue can cache 2 iteration datas
        # at most here
        self._blocking_queue_capacity = 1 * len(self._places)

        self._init_thread()
        self._shutdown = False

        global _loader
        _loader = self

    def _init_thread(self):
        self._var_names = [v.name for v in self._feed_list]
        self._shapes = [v.shape for v in self._feed_list]
        self._dtypes = [v.dtype for v in self._feed_list]
        self._need_check_feed = [
            v.desc.need_check_feed() for v in self._feed_list
        ]
        # if only 1 place, do not need to keep order
        self._blocking_queue = core.init_lod_tensor_blocking_queue(
            core.Variable(), self._blocking_queue_capacity,
            len(self._places) > 1)
        self._reader = core.create_py_reader(
            self._blocking_queue, self._var_names, self._shapes, self._dtypes,
            self._need_check_feed, self._places, self._use_buffer_reader, True,
            self._pin_memory)

        self._thread = threading.Thread(
            target=self._thread_loop, args=(_current_expected_place(), ))
        self._thread.daemon = True
        self._thread.start()

    def _thread_loop(self, legacy_expected_place):
        #NOTE(zhiqiu): Set the expected place for new thread as the same as father thread,
        # and it will call platform::SetDeviceId() in c++ internally.
        # If we do not set cudaDeviceId in new thread, the default cudaDeviceId will be 0,
        # Which may cost hundreds of MB of GPU memory on CUDAPlace(0) if calling some cuda 
        # APIs in this thread.
        _set_expected_place(legacy_expected_place)

        while not self._thread_done_event.is_set():
            try:
                indices = next(self._sampler_iter)

                # read data from dataset in mini-batch
                # with paddle.fluid.dygraph.guard(place=paddle.CPUPlace()):
                # read data from dataset in mini-batch
                batch = self._dataset_fetcher.fetch(indices,
                                                    self._thread_done_event)
            except StopIteration:
                self._exit_thread_expectedly()
                return

            if batch is None or self._thread_done_event.is_set(): break

            # flat batch and record structure infos
            batch, structure = _flatten_batch(batch)
            self._structure_infos.append(structure)

            if self._thread_done_event.is_set(): break

            try:
                # pack as LoDTensorArray
                array = core.LoDTensorArray()
                for slot in batch:
                    if isinstance(slot, paddle.Tensor):
                        slot = slot.value().get_tensor()
                    elif not isinstance(slot, core.LoDTensor):
                        tmp = core.LoDTensor()
                        tmp.set(slot, core.CPUPlace())
                        slot = tmp

                    array.append(slot)

                if self._thread_done_event.is_set(): break

                try:
                    self._blocking_queue.push(array)
                except:
                    self._exit_thread_expectedly()

            except:
                self._exit_thread_unexpectedly()
                six.reraise(*sys.exc_info())

        self._exit_thread_expectedly()

    def __next__(self):
        try:
            if in_dygraph_mode():
                data = self._reader.read_next_var_list()
                data = _restore_batch(data, self._structure_infos.pop(0))
            else:
                if self._return_list:
                    data = self._reader.read_next_list()
                    data = [
                        _restore_batch(d, s)
                        for d, s in zip(data, self._structure_infos[:len(
                            self._places)])
                    ]
                    self._structure_infos = self._structure_infos[len(
                        self._places):]
                    # static graph organized data on multi-device with list, if
                    # place number is 1, there is only 1 device, extra the data
                    # from list for devices to be compatible with dygraph mode
                    if len(self._places) == 1:
                        data = data[0]
                else:
                    data = self._reader.read_next()

            return data
        except StopIteration:
            self._reader.shutdown()
            self._try_shutdown_all()
            six.reraise(*sys.exc_info())

    def _shutdown_thread(self):
        if self._thread:
            self._thread_done_event.set()
            # NOTE: we wait for _thread exit for 3 seconds, if
            #       thread not exit normally, force kill it
            for _ in range(3):
                if self._thread.is_alive():
                    time.sleep(1)
                else:
                    break
            else:
                if self._thread is not threading.current_thread():
                    self._thread.join()

            self._thread = None

    # python2 compatibility
    def next(self):
        return self.__next__()

    def _try_shutdown_all(self):
        if not self._shutdown:
            try:
                # # _blocking_queue in keep order mode holds sub-threads
                # # need to release thread resources on unexpected exit
                if self._blocking_queue:
                    self._blocking_queue.close()
                    self._blocking_queue = None
                # NOTE: blocking queue should be closed firstly for
                # blocking queue read may hang and _thread_done_event
                # cannot be checked
                self._shutdown_thread()
            finally:
                self._shutdown = True

    def __del__(self):
        self._try_shutdown_all()


class _DataLoaderIterMultiProcess(_DataLoaderIterBase):
    def __init__(self, loader):
        super(_DataLoaderIterMultiProcess, self).__init__(loader)

        self._persistent_workers = loader._persistent_workers
        self._resume_worker_cnt = 0

        assert self._num_workers > 0,  "Multi-process DataLoader " \
                    "invalid num_workers({})".format(self._num_workers)

        # subprocess wrokers' result queue
        self._data_queue = None

        # data get from _data_queue will be reordered by _rcvd_idx
        # for data order keeping, data index not equal _rcvd_idx 
        # will be cached in _task_infos
        self._send_idx = 0
        self._rcvd_idx = 0
        self._batches_outstanding = 0
        self._task_infos = {}
        self._structure_infos = []

        # indices outstand as _outstanding_capacity at first, and
        # blocking_queue capacity is also _outstanding_capacity.
        # _outstanding_capacity here to make sure each indices_queue
        # has at least 2 indices, and outstanding batch cached
        # output data for at least 2 iterations(Note that len(_places)
        # batches will be composed as an iteration output)
        self._outstanding_capacity = 2 * max(self._num_workers,
                                             len(self._places))

        # see _try_put_indices
        self._thread_lock = threading.Lock()

        # init workers and indices queues and put 2 indices in each indices queue
        self._init_workers()
        for _ in range(self._outstanding_capacity):
            self._try_put_indices()

        self._init_thread()
        self._shutdown = False

    def _init_workers(self):
        # multiprocess worker and indice queue list initial as empty
        self._workers = []
        self._worker_status = []
        self._indices_queues = []
        self._workers_idx_cycle = itertools.cycle(range(self._num_workers))

        # create data_queue for workers
        self._data_queue = multiprocessing.Queue()

        # event for workers and thread, thread event is only need 
        # in multi-processing mode
        self._workers_done_event = multiprocessing.Event()
        self._thread_done_event = threading.Event()

        for i in range(self._num_workers):
            indices_queue = multiprocessing.Queue()
            self._indices_queues.append(indices_queue)
            worker = multiprocessing.Process(
                target=_worker_loop,
                args=(self._dataset, self._dataset_kind, indices_queue,
                      self._data_queue, self._workers_done_event,
                      self._auto_collate_batch, self._collate_fn,
                      self._drop_last, self._worker_init_fn, i,
                      self._num_workers, self._use_shared_memory))
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
            self._worker_status.append(True)

        core._set_process_pids(id(self), tuple(w.pid for w in self._workers))
        _set_SIGCHLD_handler()

    def _clear_and_remove_data_queue(self):
        if self._data_queue is not None:
            while True:
                try:
                    self._data_queue.get_nowait()
                except:
                    self._data_queue.cancel_join_thread()
                    self._data_queue.close()
                    break

    def _init_thread(self):
        self._var_names = [v.name for v in self._feed_list]
        self._shapes = [v.shape for v in self._feed_list]
        self._dtypes = [v.dtype for v in self._feed_list]
        self._need_check_feed = [
            v.desc.need_check_feed() for v in self._feed_list
        ]
        # if only 1 place, do not need to keep order
        self._blocking_queue = core.init_lod_tensor_blocking_queue(
            core.Variable(), self._outstanding_capacity, len(self._places) > 1)
        self._reader = core.create_py_reader(
            self._blocking_queue, self._var_names, self._shapes, self._dtypes,
            self._need_check_feed, self._places, self._use_buffer_reader, True,
            self._pin_memory)

        self._thread_done_event = threading.Event()
        # thread event is only need in multi-processing mode
        self._thread = threading.Thread(
            target=self._thread_loop, args=(_current_expected_place(), ))
        self._thread.daemon = True
        self._thread.start()

    def _reset(self):
        # resume iteration in following steps
        # 1. Resume workers, clear worker caches
        # put _ResumeIteration to all worker as resume iteration flag
        with self._thread_lock:
            self._resume_worker_cnt = self._num_workers
            for worker_id in range(self._num_workers):
                self._indices_queues[worker_id].put(_ResumeIteration())
                self._batches_outstanding += 1
        # all flag will be check in _thread_loop, simply wait here
        while self._resume_worker_cnt > 0:
            time.sleep(0.5)

        # 2. clear blocking_queue caches
        # in order not to restart the thread, we just clear
        # the blocking_queue cachees instead of recreating one
        while self._blocking_queue.size() >= len(self._places):
            if in_dygraph_mode():
                self._reader.read_next_var_list()
            elif self._return_list:
                self._reader.read_next_list()
            else:
                data = self._reader.read_next()

        # 3. reset all states
        self._send_idx = 0
        self._rcvd_idx = 0
        self._batches_outstanding = 0
        self._task_infos = {}
        self._structure_infos = []

        # set all worker status available
        self._worker_status = [True] * self._num_workers

        # 4. reset _sampler_iter and put prefetch indices to start next epoch
        # init workers and indices queues and put 2 indices in each indices queue
        self._sampler_iter = iter(self._index_sampler)
        for _ in range(self._outstanding_capacity):
            self._try_put_indices()

    def _shutdown_worker(self, worker_id, shutdown=False):
        if self._worker_status[worker_id] or (self._persistent_workers and
                                              shutdown):
            self._indices_queues[worker_id].put(None)
            self._worker_status[worker_id] = False

    def _try_shutdown_all(self, timeout=None):
        if not self._shutdown:
            try:
                self._exit_thread_expectedly()
                self._clear_and_remove_data_queue()

                # set _workers_done_event should be set before put None
                # to indices_queue, workers wll exit on reading None from
                # indices_queue
                self._workers_done_event.set()
                for i in range(self._num_workers):
                    self._shutdown_worker(i, shutdown=True)

                if not self._shutdown:
                    for w in self._workers:
                        w.join(timeout)
                    for q in self._indices_queues:
                        q.cancel_join_thread()
                        q.close()
            finally:
                core._erase_process_pids(id(self))
                self._shutdown = True

    def _thread_loop(self, legacy_expected_place):
        #NOTE(zhiqiu): Set the expected place for new thread as the same as father thread,
        # and it will call platform::SetDeviceId() in c++ internally.
        # If we do not set cudaDeviceId in new thread, the default cudaDeviceId will be 0,
        # Which may cost hundreds of MB of GPU memory on CUDAPlace(0) if calling some cuda 
        # APIs in this thread.
        _set_expected_place(legacy_expected_place)

        while not self._thread_done_event.is_set():
            batch = self._get_data()
            if not self._thread_done_event.is_set():
                if batch is None:
                    self._exit_thread_expectedly()
                else:
                    if isinstance(batch, _ResumeIteration):
                        assert self._resume_worker_cnt > 0
                        self._resume_worker_cnt -= 1
                        continue
                    try:
                        # pack as LoDTensorArray
                        array = core.LoDTensorArray()
                        if self._use_shared_memory:
                            for tensor in batch:
                                array.append(tensor)
                        else:
                            # LoDTensor not in shared memory is not
                            # serializable, cannot be create in workers
                            for slot in batch:
                                if isinstance(slot, paddle.Tensor):
                                    slot = slot.value().get_tensor()
                                elif not isinstance(slot, core.LoDTensor):
                                    tmp = core.LoDTensor()
                                    tmp.set(slot, core.CPUPlace())
                                    slot = tmp
                                array.append(slot)

                        if not self._blocking_queue.push(array):
                            self._blocking_queue.close()
                    except Exception as e:
                        self._exit_thread_unexpectedly()
                        six.reraise(*sys.exc_info())
                    finally:
                        self._rcvd_idx += 1

    def _get_data(self):
        while not self._thread_done_event.is_set():
            # For IterableDataset, batch indices is generated infinitely
            # for each worker to raise StopIteration, but a StopIteration
            # raising process will discard a batch indices which is count
            # in _send_idx but will not increase _rcvd_idx, so we check 
            # whether the worker is still alive here to skip the discarded
            # batch indices and increase _rcvd_idx
            if self._dataset_kind == _DatasetKind.ITER:
                while self._rcvd_idx < self._send_idx:
                    info = self._task_infos[self._rcvd_idx]
                    if len(info) == 3 or self._worker_status[info[0]]:
                        break
                    del self._task_infos[self._rcvd_idx]
                    self._rcvd_idx += 1
                    self._batches_outstanding -= 1
                else:
                    # NOTE: in persistent workers mode, do not check data
                    #       drained here, simply let it go to _data_queue
                    #       reading to get _ResumeIteration
                    if not self._persistent_workers:
                        # NOTE: _rcvd_idx and _send_idx only record batches among
                        #       workers, if batches among workers drained, there
                        #       may also be data in blocking queue
                        if self._batches_outstanding < len(self._places):
                            return None
                        continue

            if self._rcvd_idx in self._task_infos and \
                    len(self._task_infos[self._rcvd_idx]) == 3:
                info = self._task_infos.pop(self._rcvd_idx)
                self._structure_infos.append(info[2])
                return info[1]

            try:
                # [ avoid hang ]: main process may blocking at _reader.read_next when
                # KeyboardInterrupt, we do following tradeoff:
                # 1. get data with timeout, MP_STATUS_CHECK_INTERVAL(5s) as timeout
                #    default, if KeyboardInterrupt blocking, failed workers will be
                #    checked and raise RuntimeError to quit DataLoader in timeout
                #    exception handling.
                # 2. if get data timeout and check workers all alive, continue to
                #    get data again
                data = self._data_queue.get(timeout=self._timeout)
            except Exception as e:
                # check if thread done event set when waiting data
                if self._thread_done_event.is_set():
                    continue

                # check failed workers
                failed_workers = []
                for i, w in enumerate(self._workers):
                    if self._worker_status[i] and not w.is_alive():
                        failed_workers.append(w)
                        self._shutdown_worker(i)
                if len(failed_workers) > 0:
                    self._exit_thread_unexpectedly()
                    pids = ', '.join(str(w.pid) for w in failed_workers)
                    raise RuntimeError("DataLoader {} workers exit unexpectedly, " \
                                "pids: {}".format(len(failed_workers), pids))

                # get(timeout) will call _poll(timeout) and may raise IOError
                if isinstance(e, queue.Empty) or isinstance(e, IOError):
                    # continue on timeout to keep getting data from queue
                    continue

                self._exit_thread_unexpectedly()
                logging.error("DataLoader reader thread failed({}) to read data from " \
                              "workers' result queue.".format(e))
                six.reraise(*sys.exc_info())
            else:
                if self._dataset_kind == _DatasetKind.ITER and isinstance(
                        data, _IterableDatasetStopIteration):
                    # if a worker get StopIteraion, we shutdown this worker,
                    # note that this batch indices to trigger StopIteration
                    # is discard, outstanding batch number should be decrease
                    # and another indices should be put for other workers
                    # may still working.
                    if self._persistent_workers:
                        self._worker_status[data.worker_id] = False
                    else:
                        self._shutdown_worker(data.worker_id)
                        self._batches_outstanding -= 1
                    self._try_put_indices()
                    continue

                idx, batch, structure = data

                if isinstance(idx, _ResumeIteration) and batch is None \
                        and structure is None:
                    return idx

                if isinstance(batch, _WorkerException):
                    self._exit_thread_unexpectedly()
                    batch.reraise()

                if idx == self._rcvd_idx:
                    del self._task_infos[idx]
                    self._structure_infos.append(structure)
                    return batch
                else:
                    self._task_infos[idx] += (batch, structure)
                    continue

    def _try_put_indices(self):
        assert self._batches_outstanding <= self._outstanding_capacity, \
                    "too many indices have been put to queue"
        # In multi-process mode for IterableDataset, _try_put_indices will
        # be called both in main process(for our implement has blocking queue,
        # and blocking queue read is in main process) and thread, which may
        # cause error following error
        #   1. "ValueError: generator already executing" in next(self._sampler_iter)
        #   2. re-enter in increase _send_idx
        # add a lock for threading save, for _try_put_indices is only a slight
        # function which is not in data reading pipeline, this lock almost no
        # influence on performance
        with self._thread_lock:
            try:
                indices = next(self._sampler_iter)
            except StopIteration:
                return

            for i in range(self._num_workers):
                worker_idx = next(self._workers_idx_cycle)
                if self._worker_status[worker_idx]:
                    break
            else:
                return

            self._indices_queues[worker_idx].put((self._send_idx, indices))
            self._task_infos[self._send_idx] = (worker_idx, )
            self._batches_outstanding += 1
            self._send_idx += 1

    def __del__(self):
        self._try_shutdown_all()

    def _shutdown_on_exit(self):
        self._try_shutdown_all(1)

    def __next__(self):
        try:
            # _batches_outstanding here record the total batch data number
            # in 'from after _try_put_indices to beforeoutput data', this
            # value should be _outstanding_capacity if data is not drained,
            # if _batches_outstanding is less than _places number, there are
            # no enough data to generate next output, close blocking_queue and
            # set _thread_done_event here, py_reader will raise StopIteration,
            # end workers and indices_queues in StopIteration handling
            if self._batches_outstanding < len(self._places):
                if self._persistent_workers:
                    raise StopIteration
                else:
                    self._thread_done_event.set()
                    self._blocking_queue.close()

            if in_dygraph_mode():
                data = self._reader.read_next_var_list()
                data = _restore_batch(data, self._structure_infos.pop(0))
            else:
                if self._return_list:
                    data = self._reader.read_next_list()
                    data = [
                        _restore_batch(d, s)
                        for d, s in zip(data, self._structure_infos[:len(
                            self._places)])
                    ]
                    self._structure_infos = self._structure_infos[len(
                        self._places):]
                    # static graph organized data on multi-device with list, if
                    # place number is 1, there is only 1 device, extra the data
                    # from list for devices to be compatible with dygraph mode
                    if len(self._places) == 1:
                        data = data[0]
                else:
                    data = self._reader.read_next()
            self._on_output_batch()
            return data
        except StopIteration:
            if not self._persistent_workers:
                self._reader.shutdown()
                self._try_shutdown_all()
            six.reraise(*sys.exc_info())

    # python2 compatibility
    def next(self):
        return self.__next__()

    def _on_output_batch(self):
        for _ in range(len(self._places)):
            self._batches_outstanding -= 1
            self._try_put_indices()
