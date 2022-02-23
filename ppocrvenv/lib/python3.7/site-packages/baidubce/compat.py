# Copyright 2014 Baidu, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions
# and limitations under the License.
"""
This module provides string converting tools and compatibility on py2 vs py3
"""

import functools
import itertools
import operator
import sys
import types

PY2 = sys.version_info[0]==2
PY3 = sys.version_info[0]==3


if PY3:
    string_types = str,
    integer_types = int,
    class_types = type,
    text_type = str
    binary_type = bytes

    def convert_to_bytes(idata):
        """
        convert source type idata to bytes string

        :type idata: any valid python type
        :param idata: source data
        :return : bytes string
        """
        # unicode
        if isinstance(idata, str):
            return idata.encode(encoding='utf-8')
        # Ascii
        elif isinstance(idata, bytes):
            return idata
        # int,dict,list
        else:
            return str(idata).encode(encoding='utf-8')

    def convert_to_string(idata):
        """
        convert source data to str string on py3

        :type idata:any valid python type
        :param idata:source data
        :return :uniocde string on py3
        """
        return convert_to_unicode(idata)

    def convert_to_unicode(idata):
        """
        convert source type idata to unicode string

        :type idata: any valid python type
        :param idata: source data
        :return : unicode  string
        """
        # Ascii
        if isinstance(idata, bytes):
            return idata.decode(encoding='utf-8')
        # unicode
        elif isinstance(idata, str):
            return idata
        # int,dict,list
        else:
            return str(idata)

else:   # py2
    string_types = basestring,
    integer_types = (int, long)
    class_types = (type, types.ClassType)
    text_type = unicode
    binary_type = str

    def convert_to_bytes(idata):
        """
        convert source type idata to bytes string

        :type idata: any valid python type
        :param idata: source data
        :return : bytes string
        """
        if isinstance(idata, unicode):
            return idata.encode(encoding='utf-8')
        elif isinstance(idata, str):
            return idata
        # int ,long, dict, list
        else:
            return str(idata)

    def convert_to_string(idata):
        """
        convert source data to str string on py2

        :type idata:any valid python type
        :param idata:source data
        :return :bytes string on py2
        """
        return convert_to_bytes(idata)

    def convert_to_unicode(idata):
        """
        convert source type idata to unicode string

        :type idata: any valid python type
        :param idata: source data
        :return : unicode  string
        """
        if isinstance(idata, str):  #Ascii
            return idata.decode(encoding='utf-8')
        elif isinstance(idata, unicode):
            return idata
        else:
            return str(idata).decode(encoding='utf-8')
