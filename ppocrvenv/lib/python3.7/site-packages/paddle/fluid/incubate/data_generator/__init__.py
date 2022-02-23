# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import sys

__all__ = ['MultiSlotDataGenerator', 'MultiSlotStringDataGenerator']


class DataGenerator(object):
    """
    DataGenerator is a general Base class for user to inherit
    A user who wants to define his/her own python processing logic
    with paddle.fluid.dataset should inherit this class
    """

    def __init__(self):
        self._proto_info = None
        self.batch_size_ = 32

    def _set_line_limit(self, line_limit):
        if not isinstance(line_limit, int):
            raise ValueError("line_limit%s must be in int type" %
                             type(line_limit))
        if line_limit < 1:
            raise ValueError("line_limit can not less than 1")
        self._line_limit = line_limit

    def set_batch(self, batch_size):
        '''
        Set batch size of current DataGenerator
        This is necessary only if a user wants to define generator_batch
        
        Example:
            .. code-block:: python
                import paddle.fluid.incubate.data_generator as dg
                class MyData(dg.DataGenerator):
                    def generate_sample(self, line):
                        def local_iter():
                            int_words = [int(x) for x in line.split()]
                            yield ("words", int_words)
                        return local_iter
                    def generate_batch(self, samples):
                        def local_iter():
                            for s in samples:
                                yield ("words", s[1].extend([s[1][0]]))
                mydata = MyData()
                mydata.set_batch(128)
                    
        '''
        self.batch_size_ = batch_size

    def run_from_memory(self):
        '''
        This function generator data from memory, it is usually used for
        debug and benchmarking
        Example:
            .. code-block:: python
                import paddle.fluid.incubate.data_generator as dg
                class MyData(dg.DataGenerator):
                    def generate_sample(self, line):
                        def local_iter():
                            yield ("words", [1, 2, 3, 4])
                        return local_iter
                mydata = MyData()
                mydata.run_from_memory()
        '''
        batch_samples = []
        line_iter = self.generate_sample(None)
        for user_parsed_line in line_iter():
            if user_parsed_line == None:
                continue
            batch_samples.append(user_parsed_line)
            if len(batch_samples) == self.batch_size_:
                batch_iter = self.generate_batch(batch_samples)
                for sample in batch_iter():
                    sys.stdout.write(self._gen_str(sample))
                batch_samples = []
        if len(batch_samples) > 0:
            batch_iter = self.generate_batch(batch_samples)
            for sample in batch_iter():
                sys.stdout.write(self._gen_str(sample))

    def run_from_stdin(self):
        '''
        This function reads the data row from stdin, parses it with the
        process function, and further parses the return value of the 
        process function with the _gen_str function. The parsed data will
        be wrote to stdout and the corresponding protofile will be
        generated.
        Example:
        
            .. code-block:: python
                import paddle.fluid.incubate.data_generator as dg
                class MyData(dg.DataGenerator):
                    def generate_sample(self, line):
                        def local_iter():
                            int_words = [int(x) for x in line.split()]
                            yield ("words", [int_words])
                        return local_iter
                mydata = MyData()
                mydata.run_from_stdin()
        '''
        batch_samples = []
        for line in sys.stdin:
            line_iter = self.generate_sample(line)
            for user_parsed_line in line_iter():
                if user_parsed_line == None:
                    continue
                batch_samples.append(user_parsed_line)
                if len(batch_samples) == self.batch_size_:
                    batch_iter = self.generate_batch(batch_samples)
                    for sample in batch_iter():
                        sys.stdout.write(self._gen_str(sample))
                    batch_samples = []
        if len(batch_samples) > 0:
            batch_iter = self.generate_batch(batch_samples)
            for sample in batch_iter():
                sys.stdout.write(self._gen_str(sample))

    def _gen_str(self, line):
        '''
        Further processing the output of the process() function rewritten by
        user, outputting data that can be directly read by the datafeed,and
        updating proto_info information.
        Args:
            line(str): the output of the process() function rewritten by user.
        Returns:
            Return a string data that can be read directly by the datafeed.
        '''
        raise NotImplementedError(
            "pls use MultiSlotDataGenerator or PairWiseDataGenerator")

    def generate_sample(self, line):
        '''
        This function needs to be overridden by the user to process the 
        original data row into a list or tuple.
        Args:
            line(str): the original data row
        Returns:
            Returns the data processed by the user.
              The data format is list or tuple: 
            [(name, [feasign, ...]), ...] 
              or ((name, [feasign, ...]), ...)
             
            For example:
            [("words", [1926, 08, 17]), ("label", [1])]
              or (("words", [1926, 08, 17]), ("label", [1]))
        Note:
            The type of feasigns must be in int or float. Once the float
            element appears in the feasign, the type of that slot will be
            processed into a float.
        Example:
            .. code-block:: python
                import paddle.fluid.incubate.data_generator as dg
                class MyData(dg.DataGenerator):
                    def generate_sample(self, line):
                        def local_iter():
                            int_words = [int(x) for x in line.split()]
                            yield ("words", [int_words])
                        return local_iter
        '''
        raise NotImplementedError(
            "Please rewrite this function to return a list or tuple: " +
            "[(name, [feasign, ...]), ...] or ((name, [feasign, ...]), ...)")

    def generate_batch(self, samples):
        '''
        This function needs to be overridden by the user to process the
        generated samples from generate_sample(self, str) function
        It is usually used as batch processing when a user wants to
        do preprocessing on a batch of samples, e.g. padding according to
        the max length of a sample in the batch
        Args:
            samples(list tuple): generated sample from generate_sample
        Returns:
            a python generator, the same format as return value of generate_sample
        Example:
            .. code-block:: python
                import paddle.fluid.incubate.data_generator as dg
                class MyData(dg.DataGenerator):
                    def generate_sample(self, line):
                        def local_iter():
                            int_words = [int(x) for x in line.split()]
                            yield ("words", int_words)
                        return local_iter
                    def generate_batch(self, samples):
                        def local_iter():
                            for s in samples:
                                yield ("words", s[1].extend([s[1][0]]))
                mydata = MyData()
                mydata.set_batch(128)
        '''

        def local_iter():
            for sample in samples:
                yield sample

        return local_iter


# TODO: guru4elephant
# add more generalized DataGenerator that can adapt user-defined slot
# for example, [(name, float_list), (name, str_list), (name, int_list)]
class MultiSlotStringDataGenerator(DataGenerator):
    def _gen_str(self, line):
        '''
        Further processing the output of the process() function rewritten by
        user, outputting data that can be directly read by the MultiSlotDataFeed,
        and updating proto_info information.
        The input line will be in this format:
            >>> [(name, [str(feasign), ...]), ...]
            >>> or ((name, [str(feasign), ...]), ...)
        The output will be in this format:
            >>> [ids_num id1 id2 ...] ...
        For example, if the input is like this:
            >>> [("words", ["1926", "08", "17"]), ("label", ["1"])]
            >>> or (("words", ["1926", "08", "17"]), ("label", ["1"]))
        the output will be:
            >>> 3 1234 2345 3456 1 1
        Args:
            line(str): the output of the process() function rewritten by user.
        Returns:
            Return a string data that can be read directly by the MultiSlotDataFeed.
        '''
        if not isinstance(line, list) and not isinstance(line, tuple):
            raise ValueError(
                "the output of process() must be in list or tuple type"
                "Examples: [('words', ['1926', '08', '17']), ('label', ['1'])]")
        output = ""
        for index, item in enumerate(line):
            name, elements = item
            if output:
                output += " "
            out_str = []
            out_str.append(str(len(elements)))
            out_str.extend(elements)
            output += " ".join(out_str)
        return output + "\n"


class MultiSlotDataGenerator(DataGenerator):
    def _gen_str(self, line):
        '''
        Further processing the output of the process() function rewritten by
        user, outputting data that can be directly read by the MultiSlotDataFeed,
        and updating proto_info information.
        The input line will be in this format:
            >>> [(name, [feasign, ...]), ...] 
            >>> or ((name, [feasign, ...]), ...)
        The output will be in this format:
            >>> [ids_num id1 id2 ...] ...
        The proto_info will be in this format:
            >>> [(name, type), ...]
        
        For example, if the input is like this:
            >>> [("words", [1926, 08, 17]), ("label", [1])]
            >>> or (("words", [1926, 08, 17]), ("label", [1]))
        the output will be:
            >>> 3 1234 2345 3456 1 1
        the proto_info will be:
            >>> [("words", "uint64"), ("label", "uint64")]
        Args:
            line(str): the output of the process() function rewritten by user.
        Returns:
            Return a string data that can be read directly by the MultiSlotDataFeed.
        '''
        if not isinstance(line, list) and not isinstance(line, tuple):
            raise ValueError(
                "the output of process() must be in list or tuple type"
                "Example: [('words', [1926, 08, 17]), ('label', [1])]")
        output = ""

        if self._proto_info is None:
            self._proto_info = []
            for item in line:
                name, elements = item
                if not isinstance(name, str):
                    raise ValueError("name%s must be in str type" % type(name))
                if not isinstance(elements, list):
                    raise ValueError("elements%s must be in list type" %
                                     type(elements))
                if not elements:
                    raise ValueError(
                        "the elements of each field can not be empty, you need padding it in process()."
                    )
                self._proto_info.append((name, "uint64"))
                if output:
                    output += " "
                output += str(len(elements))
                for elem in elements:
                    if isinstance(elem, float):
                        self._proto_info[-1] = (name, "float")
                    elif not isinstance(elem, int) and not isinstance(elem,
                                                                      long):
                        raise ValueError(
                            "the type of element%s must be in int or float" %
                            type(elem))
                    output += " " + str(elem)
        else:
            if len(line) != len(self._proto_info):
                raise ValueError(
                    "the complete field set of two given line are inconsistent.")
            for index, item in enumerate(line):
                name, elements = item
                if not isinstance(name, str):
                    raise ValueError("name%s must be in str type" % type(name))
                if not isinstance(elements, list):
                    raise ValueError("elements%s must be in list type" %
                                     type(elements))
                if not elements:
                    raise ValueError(
                        "the elements of each field can not be empty, you need padding it in process()."
                    )
                if name != self._proto_info[index][0]:
                    raise ValueError(
                        "the field name of two given line are not match: require<%s>, get<%s>."
                        % (self._proto_info[index][0], name))
                if output:
                    output += " "
                output += str(len(elements))
                for elem in elements:
                    if self._proto_info[index][1] != "float":
                        if isinstance(elem, float):
                            self._proto_info[index] = (name, "float")
                        elif not isinstance(elem, int) and not isinstance(elem,
                                                                          long):
                            raise ValueError(
                                "the type of element%s must be in int or float"
                                % type(elem))
                    output += " " + str(elem)
        return output + "\n"
