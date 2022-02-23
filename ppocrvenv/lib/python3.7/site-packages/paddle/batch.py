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

__all__ = []


def batch(reader, batch_size, drop_last=False):
    """
    This operator creates a batched reader which combines the data from the 
    input reader to batched data.
    
    Args:
        reader(generator): the data reader to read from.
        batch_size(int): size of each mini-batch.
        drop_last(bool, optional): If set to True, the last batch is dropped when 
            the size of last batch is not equal to batch_size, if set to False,
            it will not. Default: False.
    Returns:
        The batched reader. 
    
    Return Type:
        generator   

    Examples:
        .. code-block:: python
           
            import paddle
            def reader():
                for i in range(10):
                    yield i
            batch_reader = paddle.batch(reader, batch_size=2)
            
            for data in batch_reader():
                print(data)

            # Output is
            # [0, 1]
            # [2, 3]
            # [4, 5]
            # [6, 7]
            # [8, 9]
    """

    def batch_reader():
        r = reader()
        b = []
        for instance in r:
            b.append(instance)
            if len(b) == batch_size:
                yield b
                b = []
        if drop_last is False and len(b) != 0:
            yield b

    # Batch size check
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size should be a positive integeral value, "
                         "but got batch_size={}".format(batch_size))

    return batch_reader
