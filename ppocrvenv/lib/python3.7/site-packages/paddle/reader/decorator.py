# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from threading import Thread
import subprocess
import multiprocessing
import six
import sys
import warnings
import logging

from six.moves.queue import Queue
from six.moves import zip_longest
from six.moves import map
from six.moves import zip
import itertools
import random
import zlib

import paddle.compat as cpt
from paddle.fluid.reader import QUEUE_GET_TIMEOUT

__all__ = []

# On macOS, the 'spawn' start method is now the default in Python3.8 multiprocessing,
# Paddle is currently unable to solve this, so forces the process to start using 
# the 'fork' start method.
#
# TODO: This solution is not good, because the fork start method could lead to 
# crashes of the subprocess. Figure out how to make 'spawn' work.
#
# For more details, please refer to
# https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
# https://bugs.python.org/issue33725
if sys.version_info >= (3, 8) and sys.platform == 'darwin':
    fork_context = multiprocessing.get_context('fork')
else:
    fork_context = multiprocessing


def cache(reader):
    """
    Cache the reader data into memory. 

    Be careful that this method may take long time to process, 
    and consume lots of memory. :code:`reader()` would only 
    call once. 

    Args:
        reader (generator): a reader object which yields 
            data each time.

    Returns:
        generator: a decorated reader object which yields data from cached memory.
    
    Examples:
        .. code-block:: python

            import paddle
            
            def reader():
                for i in range(3):
                    yield i
            
            # All data is cached into memory
            cached_reader = paddle.io.cache(reader)
            
            # Output: 0 1 2
            for i in cached_reader():
                print(i)
    """
    all_data = tuple(reader())

    def __impl__():
        for item in all_data:
            yield item

    return __impl__


def map_readers(func, *readers):
    """
    Creates a data reader that outputs return value of function using
    output of each data reader as arguments.

    If input readers output the following data entries: 2 3,
    and the input func is mul(x, y),
    the output of the resulted reader will be 6.


    Args:
        func: a function to read data and compute result, the output of this function 
              will be set as the output of the resulted data reader.
        readers (Reader|list of Reader): list of readers whose outputs will be used as arguments of func.
 
    Returns:
        the resulted data reader (Reader)

    Examples:

        .. code-block:: python

         import paddle.reader
         d = {"h": 0, "i": 1}
         def func(x):
             return d[x]
         def reader():
             yield "h"
             yield "i"
         map_reader_result = paddle.reader.map_readers(func, reader)
    """

    def reader():
        rs = []
        for r in readers:
            rs.append(r())
        for e in map(func, *rs):
            yield e

    return reader


def shuffle(reader, buf_size):
    """
    paddle.fluid.io.shuffle ( :ref:`api_fluid_io_shuffle` ) is recommended to use,
    and paddle.reader.shuffle is an alias.

    This API creates a decorated reader that outputs the shuffled data.

    The output data from the origin reader will be saved into a buffer, 
    and then shuffle the data. The size of buffer is determined by argument buf_size.
 
    Args:
        reader(callable): the original reader whose data will be shuffled.
        buf_size(int): the size of shuffled buffer.

    Returns:
        callable: a decorated reader.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            def reader():
                for i in range(5):
                    yield i
            shuffled_reader = fluid.io.shuffle(reader, 3)
            for e in shuffled_reader():
                print(e)
            # outputs are 0~4 unordered arrangement
    """

    def data_reader():
        buf = []
        for e in reader():
            buf.append(e)
            if len(buf) >= buf_size:
                random.shuffle(buf)
                for b in buf:
                    yield b
                buf = []

        if len(buf) > 0:
            random.shuffle(buf)
            for b in buf:
                yield b

    return data_reader


def chain(*readers):
    """
    Use the input data readers to create a chained data reader. The new created reader
    chains the outputs of input readers together as its output, and it do not change
    the format of the outputs.

    **Note**:
        ``paddle.reader.chain`` is the alias of ``paddle.fluid.io.chain``, and
        ``paddle.fluid.io.chain`` is recommended to use.

    For example, if three input readers' outputs are as follows:
    [0, 0, 0],
    [10, 10, 10],
    [20, 20, 20].
    The chained reader will output:
    [0, 0, 0], [10, 10, 10], [20, 20, 20].

    Args:
        readers(list): input data readers.

    Returns:
        callable: the new chained data reader.

    Examples:
        ..  code-block:: python

            import paddle

            def reader_creator_3(start):
                def reader():
                    for i in range(start, start + 3):
                        yield [i, i, i]
                return reader

            c = paddle.reader.chain(reader_creator_3(0), reader_creator_3(10), reader_creator_3(20))
            for e in c():
                print(e)
            # Output:
            # [0, 0, 0]
            # [1, 1, 1]
            # [2, 2, 2]
            # [10, 10, 10]
            # [11, 11, 11]
            # [12, 12, 12]
            # [20, 20, 20]
            # [21, 21, 21]
            # [22, 22, 22]

    """

    def reader():
        rs = []
        for r in readers:
            rs.append(r())

        for e in itertools.chain(*rs):
            yield e

    return reader


class ComposeNotAligned(ValueError):
    pass


def compose(*readers, **kwargs):
    """
    Creates a data reader whose output is the combination of input readers.

    If input readers output following data entries:
    (1, 2)    3    (4, 5)
    The composed reader will output:
    (1, 2, 3, 4, 5)

    Args:
        readers (Reader|list of Reader): readers that will be composed together. 
        check_alignment(bool, optional): Indicates whether the input readers are checked for
                              alignment. If True, whether input readers are aligned
                              correctly will be checked, else alignment will not be checkout and trailing outputs
                              will be discarded. Defaults to True.

    Returns: 
        the new data reader (Reader).

    Raises:
        ComposeNotAligned: outputs of readers are not aligned. This will not raise if check_alignment is set to False.
  
    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          def reader_creator_10(dur):
              def reader():
                 for i in range(10):
                     yield i
              return reader
          reader = fluid.io.compose(reader_creator_10(0), reader_creator_10(0))
    """
    check_alignment = kwargs.pop('check_alignment', True)

    def make_tuple(x):
        if isinstance(x, tuple):
            return x
        else:
            return (x, )

    def reader():
        rs = []
        for r in readers:
            rs.append(r())
        if not check_alignment:
            for outputs in zip(*rs):
                yield sum(list(map(make_tuple, outputs)), ())
        else:
            for outputs in zip_longest(*rs):
                for o in outputs:
                    if o is None:
                        # None will be not be present if compose is aligned
                        raise ComposeNotAligned(
                            "outputs of readers are not aligned.")
                yield sum(list(map(make_tuple, outputs)), ())

    return reader


def buffered(reader, size):
    """
    Creates a buffered data reader.

    The buffered data reader will read and save data entries into a
    buffer. Reading from the buffered data reader will proceed as long
    as the buffer is not empty.

    Args:
        reader(generator): the data reader to read from.
        size(int): max buffer size.

    Returns:
        generator: the buffered data reader.
    
    Examples:
        .. code-block:: python

            import paddle
            
            def reader():
                for i in range(3):
                    yield i
            
            # Create a buffered reader, and the buffer size is 2.
            buffered_reader = paddle.io.buffered(reader, 2)
            
            # Output: 0 1 2
            for i in buffered_reader():
                print(i)
    """

    class EndSignal():
        pass

    end = EndSignal()

    def read_worker(r, q):
        for d in r:
            q.put(d)
        q.put(end)

    def data_reader():
        r = reader()
        q = Queue(maxsize=size)
        t = Thread(
            target=read_worker, args=(
                r,
                q, ))
        t.daemon = True
        t.start()
        e = q.get()
        while e != end:
            yield e
            e = q.get()

    return data_reader


def firstn(reader, n):
    """
    paddle.fluid.io.firstn ( :ref:`api_fluid_io_firstn` ) is recommended to use,
    and paddle.reader.firstn is an alias.
    
    This API creates a decorated reader, and limits the max number of 
    samples that reader could return.

    Args:
        reader(callable): the input reader.
        n(int): the max number of samples in the reader.

    Returns:
        callable: the decorated reader.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            def reader():
                for i in range(100):
                    yield i
            firstn_reader = fluid.io.firstn(reader, 5)
            for e in firstn_reader():
                print(e)
            # the outputs are: 0 1 2 3 4  
    """

    # TODO(yuyang18): Check if just drop the reader, could clean the opened
    # resource or not?

    def firstn_reader():
        for i, item in enumerate(reader()):
            if i == n:
                break
            yield item

    return firstn_reader


class XmapEndSignal():
    pass


def xmap_readers(mapper, reader, process_num, buffer_size, order=False):
    """
    Use multi-threads to map samples from reader by a mapper defined by user.

    Args:
        mapper (callable): a function to map the data from reader.
        reader (callable): a data reader which yields the data. 
        process_num (int): thread number to handle original sample.
        buffer_size (int): size of the queue to read data in. 
        order (bool): whether to keep the data order from original reader. 
            Default False.

    Returns:
        callable: a decorated reader with data mapping. 
    """
    end = XmapEndSignal()

    # define a worker to read samples from reader to in_queue
    def read_worker(reader, in_queue):
        for i in reader():
            in_queue.put(i)
        in_queue.put(end)

    # define a worker to read samples from reader to in_queue with order flag
    def order_read_worker(reader, in_queue):
        in_order = 0
        for i in reader():
            in_queue.put((in_order, i))
            in_order += 1
        in_queue.put(end)

    # define a worker to handle samples from in_queue by mapper
    # and put mapped samples into out_queue
    def handle_worker(in_queue, out_queue, mapper):
        sample = in_queue.get()
        while not isinstance(sample, XmapEndSignal):
            r = mapper(sample)
            out_queue.put(r)
            sample = in_queue.get()
        in_queue.put(end)
        out_queue.put(end)

    # define a worker to handle samples from in_queue by mapper
    # and put mapped samples into out_queue by order
    def order_handle_worker(in_queue, out_queue, mapper, out_order):
        ins = in_queue.get()
        while not isinstance(ins, XmapEndSignal):
            order, sample = ins
            r = mapper(sample)
            while order != out_order[0]:
                pass
            out_queue.put(r)
            out_order[0] += 1
            ins = in_queue.get()
        in_queue.put(end)
        out_queue.put(end)

    def xreader():
        in_queue = Queue(buffer_size)
        out_queue = Queue(buffer_size)
        out_order = [0]
        # start a read worker in a thread
        target = order_read_worker if order else read_worker
        t = Thread(target=target, args=(reader, in_queue))
        t.daemon = True
        t.start()
        # start several handle_workers
        target = order_handle_worker if order else handle_worker
        args = (in_queue, out_queue, mapper, out_order) if order else (
            in_queue, out_queue, mapper)
        workers = []
        for i in range(process_num):
            worker = Thread(target=target, args=args)
            worker.daemon = True
            workers.append(worker)
        for w in workers:
            w.start()

        sample = out_queue.get()
        while not isinstance(sample, XmapEndSignal):
            yield sample
            sample = out_queue.get()
        finish = 1
        while finish < process_num:
            sample = out_queue.get()
            if isinstance(sample, XmapEndSignal):
                finish += 1
            else:
                yield sample

    return xreader


def multiprocess_reader(readers, use_pipe=True, queue_size=1000):
    """
    This API use python ``multiprocessing`` to read data from ``readers`` parallelly,
    and then ``multiprocess.Queue`` or ``multiprocess.Pipe`` is used to merge 
    these data. A separate process will be created for each reader in the 
    ``readers`` list, please guarantee every reader can work independently 
    to avoid conflicts in parallel environment.
    

    ``Multiprocess.Queue`` require the rw access right to /dev/shm, and it's not supported 
    in some platforms.

    Parameters:
       readers (list( ``generator`` ) | tuple( ``generator`` )): a python ``generator`` list 
           used to read input data
       use_pipe (bool, optional): control the inner API used to implement the multi-processing,
           default True - use ``multiprocess.Pipe`` which is recommended
       queue_size (int, optional): only useful when ``use_pipe`` is False - ``multiprocess.Queue``
           is used, default 1000. Increase this value can speed up the data reading, and more memory
           will be consumed.

    Returns:
        ``generator``: a new reader which can be run parallelly


    Example:

    .. code-block:: python

        import paddle.fluid as fluid
        from paddle.fluid.io import multiprocess_reader
        import numpy as np
        
        sample_files = ['sample_file_1', 'sample_file_2']
        
        def fake_input_files():
            with open(sample_files[0], 'w') as f:
               np.savez(f, a=np.array([1, 2]), b=np.array([3, 4]), c=np.array([5, 6]), d=np.array([7, 8]))
            with open(sample_files[1], 'w') as f:
               np.savez(f, a=np.array([9, 10]), b=np.array([11, 12]), c=np.array([13, 14]))
        
        
        def generate_reader(file_name):
            # load data file
            def _impl():
                data = np.load(file_name)
                for item in sorted(data.files):
                    yield data[item],
            return _impl
        
        if __name__ == '__main__':
            # generate sample input files
            fake_input_files()
            
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                place = fluid.CPUPlace()
                # the 1st 2 is batch size
                image = fluid.data(name='image', dtype='int64', shape=[2, 1, 2]) 
                fluid.layers.Print(image)
                # print detailed tensor info of image variable
            
                reader = fluid.io.PyReader(feed_list=[image], capacity=2)
            
                decorated_reader = multiprocess_reader(
                    [generate_reader(sample_files[0]), generate_reader(sample_files[1])], False)
            
                reader.decorate_sample_generator(decorated_reader, batch_size=2, places=[place])
            
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
            
                for data in reader():
                    res = exe.run(feed=data, fetch_list=[image])
                    print(res[0])
                    # print below content in this case
                    # [[[1 2]], [[3 4]]]
                    # [[[5 6]], [[7 8]]]
                    # [[[9 10]], [[11 12]]]
                    # [13,14] will be dropped

    """

    if sys.platform == 'win32':
        raise NotImplementedError(
            "The multiprocess_reader method is not supported on windows.")

    # ujson is ultra fast json encoder and decoder written in pure C with bindings for Python 3.6+.
    try:
        import ujson as json
    except Exception as e:
        warnings.warn(
            "The `ujson` module is not found, use the `json` module, `ujson` encodes and decodes faster, "
            "you can install `ujson` through `pip install ujson`.")
        import json

    assert isinstance(readers, (list, tuple)) and len(readers) > 0, (
        "`readers` must be list or tuple.")

    def _read_into_queue(reader, queue):
        try:
            for sample in reader():
                if sample is None:
                    raise ValueError("sample has None")
                queue.put(sample)
            queue.put(None)
        except:
            queue.put("")
            six.reraise(*sys.exc_info())

    def queue_reader():
        queue = fork_context.Queue(queue_size)
        for reader in readers:
            p = fork_context.Process(
                target=_read_into_queue, args=(reader, queue))
            p.start()

        reader_num = len(readers)
        finish_num = 0
        while finish_num < reader_num:
            try:
                sample = queue.get(timeout=QUEUE_GET_TIMEOUT)
            except:
                logging.error(
                    "multiprocess_reader failed to get data from the multiprocessing.Queue."
                )
                six.reraise(*sys.exc_info())

            if sample is None:
                finish_num += 1
            elif sample == "":
                raise ValueError(
                    "multiprocess_reader failed to put data into the multiprocessing.Queue."
                )
            else:
                yield sample

    def _read_into_pipe(reader, conn):
        try:
            for sample in reader():
                if sample is None:
                    raise ValueError("sample has None!")
                conn.send(json.dumps(sample))
            conn.send(json.dumps(None))
            conn.close()
        except:
            conn.send(json.dumps(""))
            conn.close()
            six.reraise(*sys.exc_info())

    def pipe_reader():
        conns = []
        for reader in readers:
            parent_conn, child_conn = fork_context.Pipe()
            conns.append(parent_conn)
            p = fork_context.Process(
                target=_read_into_pipe, args=(reader, child_conn))
            p.start()

        reader_num = len(readers)
        finish_num = 0
        conn_to_remove = []
        while finish_num < reader_num:
            for conn in conn_to_remove:
                conns.remove(conn)
            conn_to_remove = []
            for conn in conns:
                sample = json.loads(conn.recv())
                if sample is None:
                    finish_num += 1
                    conn.close()
                    conn_to_remove.append(conn)
                elif sample == "":
                    conn.close()
                    conn_to_remove.append(conn)
                    raise ValueError(
                        "multiprocess_reader failed to send data into the multiprocessing.Pipe."
                    )
                else:
                    yield sample

    if use_pipe:
        return pipe_reader
    else:
        return queue_reader
