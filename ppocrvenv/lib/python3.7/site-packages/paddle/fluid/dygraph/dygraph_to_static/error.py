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
import traceback
import linecache
import re

from paddle.fluid.dygraph.dygraph_to_static.origin_info import Location, OriginInfo, global_origin_info_map

ERROR_DATA = "Error data about original source code information and traceback."

# A flag to set whether to open the dygraph2static error reporting module
SIMPLIFY_ERROR_ENV_NAME = "TRANSLATOR_SIMPLIFY_NEW_ERROR"
DEFAULT_SIMPLIFY_NEW_ERROR = 1

# A flag to set whether to display the simplified error stack
DISABLE_ERROR_ENV_NAME = "TRANSLATOR_DISABLE_NEW_ERROR"
DEFAULT_DISABLE_NEW_ERROR = 0

SOURCE_CODE_RANGE = 5
BLANK_COUNT_BEFORE_FILE_STR = 4


def attach_error_data(error, in_runtime=False):
    """
    Attachs error data about original source code information and traceback to an error.

    Args:
        error(Exception): An native error.
        in_runtime(bool): `error` is raised in runtime if in_runtime is True, otherwise in compile time
    Returns:
        An error attached data about original source code information and traceback.
    """

    e_type, e_value, e_traceback = sys.exc_info()
    tb = traceback.extract_tb(e_traceback)[1:]

    error_data = ErrorData(e_type, e_value, tb, global_origin_info_map)
    error_data.in_runtime = in_runtime

    setattr(error, ERROR_DATA, error_data)

    return error


class TraceBackFrame(OriginInfo):
    """
    Traceback frame information.
    """

    def __init__(self, location, function_name, source_code):
        self.location = location
        self.function_name = function_name
        self.source_code = source_code

    def formated_message(self):
        # self.source_code may be empty in some functions.
        # For example, decorator generated function
        return ' ' * BLANK_COUNT_BEFORE_FILE_STR + 'File "{}", line {}, in {}\n\t{}'.format(
            self.location.filepath, self.location.lineno, self.function_name,
            self.source_code.lstrip()
            if isinstance(self.source_code, str) else self.source_code)


class TraceBackFrameRange(OriginInfo):
    """
    Traceback frame information.
    """

    def __init__(self, location, function_name):
        self.location = location
        self.function_name = function_name
        self.source_code = []
        blank_count = []
        begin_lineno = max(1, self.location.lineno - int(SOURCE_CODE_RANGE / 2))

        for i in range(begin_lineno, begin_lineno + SOURCE_CODE_RANGE):
            line = linecache.getline(self.location.filepath, i).rstrip('\n')
            line_lstrip = line.lstrip()
            self.source_code.append(line_lstrip)
            if not line_lstrip:  # empty line from source code
                blank_count.append(-1)
            else:
                blank_count.append(len(line) - len(line_lstrip))

            if i == self.location.lineno:
                hint_msg = '~' * len(self.source_code[-1]) + ' <--- HERE'
                self.source_code.append(hint_msg)
                blank_count.append(blank_count[-1])
        linecache.clearcache()
        # remove top and bottom empty line in source code
        while len(self.source_code) > 0 and not self.source_code[0]:
            self.source_code.pop(0)
            blank_count.pop(0)
        while len(self.source_code) > 0 and not self.source_code[-1]:
            self.source_code.pop(-1)
            blank_count.pop(-1)

        min_black_count = min([i for i in blank_count if i >= 0])
        for i in range(len(self.source_code)):
            # if source_code[i] is empty line between two code line, dont add blank
            if self.source_code[i]:
                self.source_code[i] = ' ' * (blank_count[i] - min_black_count +
                                             BLANK_COUNT_BEFORE_FILE_STR * 2
                                             ) + self.source_code[i]

    def formated_message(self):
        msg = ' ' * BLANK_COUNT_BEFORE_FILE_STR + 'File "{}", line {}, in {}\n'.format(
            self.location.filepath, self.location.lineno, self.function_name)
        # add empty line after range code
        return msg + '\n'.join(self.source_code)


class SuggestionDict(object):
    def __init__(self):
        # {(keywords): (suggestions)}
        self.suggestion_dict = {
            ('is not initialized.', 'Hint:', 'IsInitialized'):
            ("Please ensure all your sublayers are inheritted from nn.Layer.",
             "Please ensure there is no tensor created explicitly depended on external data, we suggest to register it as buffer tensor. See https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/04_dygraph_to_static/export_model/principle_cn.html#parameters-buffers for details"
             )
        }

    def keys(self):
        return self.suggestion_dict.keys()

    def __getitem__(self, key):
        return self.suggestion_dict[key]


class ErrorData(object):
    """
    Error data attached to an exception which is raised in un-transformed code.
    """

    def __init__(self, error_type, error_value, origin_traceback,
                 origin_info_map):
        self.error_type = error_type
        self.error_value = error_value
        self.origin_traceback = origin_traceback
        self.origin_info_map = origin_info_map
        self.in_runtime = False
        self.suggestion_dict = SuggestionDict()

    def create_exception(self):
        message = self.create_message()
        new_exception = self.error_type(message)
        setattr(new_exception, ERROR_DATA, self)
        return new_exception

    def create_message(self):
        """
        Creates a custom error message which includes trace stack with source code information of dygraph from user.
        """
        message_lines = []

        # Step1: Adds header message to prompt users that the following is the original information.
        header_message = "In transformed code:"
        message_lines.append(header_message)
        message_lines.append("")

        # Simplify error value to improve readability if error is raised in runtime
        if self.in_runtime:
            if int(
                    os.getenv(SIMPLIFY_ERROR_ENV_NAME,
                              DEFAULT_SIMPLIFY_NEW_ERROR)):
                self._simplify_error_value()
            message_lines.append(str(self.error_value))
            return '\n'.join(message_lines)

        # Step2: Optimizes stack information with source code information of dygraph from user.
        user_code_traceback_index = []
        for i, (filepath, lineno, funcname,
                code) in enumerate(self.origin_traceback):
            dygraph_func_info = self.origin_info_map.get((filepath, lineno),
                                                         None)
            if dygraph_func_info:
                user_code_traceback_index.append(i)

        # Add user code traceback
        for i in user_code_traceback_index:
            filepath, lineno, funcname, code = self.origin_traceback[i]
            dygraph_func_info = self.origin_info_map.get((filepath, lineno),
                                                         None)
            if i == user_code_traceback_index[-1]:
                traceback_frame = TraceBackFrameRange(
                    dygraph_func_info.location, dygraph_func_info.function_name)
            else:
                traceback_frame = TraceBackFrame(
                    dygraph_func_info.location, dygraph_func_info.function_name,
                    dygraph_func_info.source_code)

            message_lines.append(traceback_frame.formated_message())
        message_lines.append("")

        # Add paddle traceback after user code traceback
        paddle_traceback_start_idnex = user_code_traceback_index[
            -1] + 1 if user_code_traceback_index else 0
        for filepath, lineno, funcname, code in self.origin_traceback[
                paddle_traceback_start_idnex:]:
            traceback_frame = TraceBackFrame(
                Location(filepath, lineno), funcname, code)
            message_lines.append(traceback_frame.formated_message())
        message_lines.append("")

        # Step3: Adds error message like "TypeError: dtype must be int32, but received float32".
        # NOTE: `format_exception` is a list, its length is 1 in most cases, but sometimes its length
        # is gather than 1, for example, the error_type is IndentationError.
        format_exception = traceback.format_exception_only(self.error_type,
                                                           self.error_value)
        error_message = [
            " " * BLANK_COUNT_BEFORE_FILE_STR + line
            for line in format_exception
        ]
        message_lines.extend(error_message)

        return '\n'.join(message_lines)

    def _create_revise_suggestion(self, bottom_error_message):
        revise_suggestions = [
            '', ' ' * BLANK_COUNT_BEFORE_FILE_STR + 'Revise suggestion: '
        ]
        for keywords in self.suggestion_dict.keys():
            contain_keywords = [
                True for i in keywords if i in ''.join(bottom_error_message)
            ]
            if len(contain_keywords) == len(
                    keywords):  # all keywords should be in bottom_error_message
                for suggestion in self.suggestion_dict[keywords]:
                    suggestion_msg = ' ' * BLANK_COUNT_BEFORE_FILE_STR * 2 + '{}. {}'.format(
                        str(len(revise_suggestions) - 1), suggestion)
                    revise_suggestions.append(suggestion_msg)
        return revise_suggestions if len(revise_suggestions) > 2 else []

    def _simplify_error_value(self):
        """
        Simplifies error value to improve readability if error is raised in runtime.

        NOTE(liym27): The op callstack information about transformed static code has been replaced with original dygraph code.

        TODO(liym27):
            1. Need a more robust way because the code of start_trace may change.
            2. Set the switch to determine whether to simplify error_value
        """

        assert self.in_runtime is True

        error_value_lines = str(self.error_value).split("\n")
        error_value_lines_strip = [mes.lstrip(" ") for mes in error_value_lines]

        start_trace = "outputs = static_func(*inputs)"
        start_idx = error_value_lines_strip.index(start_trace)

        error_value_lines = error_value_lines[start_idx + 1:]
        error_value_lines_strip = error_value_lines_strip[start_idx + 1:]

        # use empty line to locate the bottom_error_message
        empty_line_idx = error_value_lines_strip.index('')
        bottom_error_message = error_value_lines[empty_line_idx + 1:]
        revise_suggestion = self._create_revise_suggestion(bottom_error_message)

        user_filepath = ''
        error_traceback = []
        user_code_traceback_index = []
        pattern = 'File "(?P<filepath>.+)", line (?P<lineno>.+), in (?P<function_name>.+)'
        for i in range(0, len(error_value_lines_strip), 2):
            if error_value_lines_strip[i].startswith("File "):
                re_result = re.search(pattern, error_value_lines_strip[i])
                tmp_filepath, lineno_str, function_name = re_result.groups()
                code = error_value_lines_strip[i + 1] if i + 1 < len(
                    error_value_lines_strip) else ''
                if i == 0:
                    user_filepath = tmp_filepath
                if tmp_filepath == user_filepath:
                    user_code_traceback_index.append(len(error_traceback))

                error_traceback.append(
                    (tmp_filepath, int(lineno_str), function_name, code))

        error_frame = []
        # Add user code traceback
        for i in user_code_traceback_index:
            filepath, lineno, funcname, code = error_traceback[i]
            if i == user_code_traceback_index[-1]:
                traceback_frame = TraceBackFrameRange(
                    Location(filepath, lineno), funcname)
            else:
                traceback_frame = TraceBackFrame(
                    Location(filepath, lineno), funcname, code)
            error_frame.append(traceback_frame.formated_message())
        error_frame.append("")

        # Add paddle traceback after user code traceback
        paddle_traceback_start_idnex = user_code_traceback_index[
            -1] + 1 if user_code_traceback_index else 0
        for filepath, lineno, funcname, code in error_traceback[
                paddle_traceback_start_idnex:]:
            traceback_frame = TraceBackFrame(
                Location(filepath, lineno), funcname, code)
            error_frame.append(traceback_frame.formated_message())
        error_frame.append("")

        error_frame.extend(bottom_error_message)
        error_frame.extend(revise_suggestion)
        error_value_str = '\n'.join(error_frame)
        self.error_value = self.error_type(error_value_str)

    def raise_new_exception(self):
        # Raises the origin error if disable dygraph2static error module,
        if int(os.getenv(DISABLE_ERROR_ENV_NAME, DEFAULT_DISABLE_NEW_ERROR)):
            raise

        new_exception = self.create_exception()
        if six.PY3:
            # NOTE(liym27):
            # 1. Why `raise new_exception from None`?
            #   In Python 3, by default, an new exception is raised with trace information of the caught exception.
            #   This only raises new_exception and hides unwanted implementation details from tracebacks of the
            #   caught exception.
            # 2. Use exec to bypass syntax error checking in Python 2.

            six.exec_("raise new_exception from None")
        else:
            raise new_exception
