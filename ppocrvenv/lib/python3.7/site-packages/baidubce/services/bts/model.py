# coding=utf-8
# Copyright 2014 Baidu, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""
This module defines some Argument classes for BTS
"""


class CreateInstanceArgs(object):
    """
    Create Instance Args
    :param storage_type  instance's storage type. eg.CommonPerformance
    :type storage_type string
    """
    def __init__(self, storage_type=None):
        self.storage_type = storage_type


def create_instance_args_2_dict(args):
    """
    change create_instance_args to dict

    :param args: create instance args
    :type args: CreateInstanceArgs

    :return:
    :rtype dict
    """
    return {
        'storageType': args.storage_type
    }


class CreateTableArgs(object):
    """
    Create Table Args
    :param table_version  table's version
    :type table_version int64
    :param compress_type  table's compress type. eg.SNAPPY_ALL
    :type compress_type string
    :param ttl  time to live
    :type ttl int
    :param storage_type  instance's storage type. eg.CommonPerformance
    :type storage_type string
    :param max_versions  table's max data versions.
    :type max_versions int
    """
    def __init__(self, table_version=0, compress_type=None, ttl=0, storage_type=None, max_versions=1):
        self.table_version = table_version
        self.compress_type = compress_type
        self.ttl = ttl
        self.storage_type = storage_type
        self.max_versions = max_versions


def create_table_args_2_dict(args):
    """
    change create_table_args to dict

    :param args: create table args
    :type args: CreateTableArgs

    :return:
    :rtype dict
    """
    return {
        'tableVersion': args.table_version,
        'compressType': args.compress_type,
        'ttl': args.ttl,
        'storageType': args.storage_type,
        'maxVersions': args.max_versions
    }


class UpdateTableArgs(object):
    """
    Update Table Args
    :param table_version  table's version
    :type table_version int64
    :param compress_type  table's compress type. eg.SNAPPY_ALL
    :type compress_type string
    :param ttl time to live
    :type ttl int
    :param max_versions  table's max data versions.
    :type max_versions int
    """
    # 不能将table_version初始化为None，否则后端会认为是创建表
    def __init__(self, table_version=1, compress_type=None, ttl=None, max_versions=None):
        self.table_version = table_version
        self.compress_type = compress_type
        self.ttl = ttl
        self.max_versions = max_versions


def update_table_args_2_dict(args):
    """
    change update_table_args to dict

    :param args: update table args
    :type args: UpdateTableArgs

    :return:
    :rtype dict
    """
    return {
        'tableVersion': args.table_version,
        'compressType': args.compress_type,
        'ttl': args.ttl,
        'maxVersions': args.max_versions
    }


class Cell(object):
    """
    Cell
    :param column
    :type column string
    :param value
    :type value string
    """
    def __init__(self, column="", value=""):
        self.column = column
        self.value = value


class Row(object):
    """
    Row
    :param rowkey
    :type rowkey string
    :param cells
    :type cells []
    """
    def __init__(self, rowkey=""):
        self.rowkey = rowkey
        self.cells = []

    def append_cell(self, cell):
        """
        append cell

        :param cell: column & value
        :type cell: Cell

        :return:
        :rtype
        """
        self.cells.append(cell)

    def get_cell(self):
        """
        get cell

        :return cells:
        :rtype Cell[]
        """
        return self.cells


class BatchPutRowArgs(object):
    """
    Batch Put Row Args
    :param rows
    :type rows []
    """
    def __init__(self):
        self.rows = []

    def append_row(self, row):
        """
        append row

        :param row:
        :type row: Row

        :return:
        :rtype
        """
        self.rows.append(row)

    def get_row(self):
        """
        get row

        :return rows:
        :rtype Row[]
        """
        return self.rows


class QueryCell(object):
    """
    Query Cell
    :param column
    :type column string
    """
    def __init__(self, column=""):
        self.column = column


class QueryRowArgs(object):
    """
    Query Row Arg
    :param rowkey
    :type rowkey string
    :param max_versions
    :type max_versions int
    :param cells
    :type cells []
    """
    def __init__(self, rowkey="", max_versions=0):
        self.rowkey = rowkey
        self.max_versions = max_versions
        self.cells = []

    def append_cell(self, cell):
        """
        append cell

        :param cell: column & value
        :type cell: Cell

        :return:
        :rtype
        """
        self.cells.append(cell)

    def get_cell(self):
        """
        get cell

        :return cells:
        :rtype Cell[]
        """
        return self.cells


def query_row_args_2_dict(args):
    """
    change query_row_args to dict

    :param args: query row args
    :type args: QueryRowArgs

    :return:
    :rtype dict
    """
    return {
        'rowkey': args.rowkey,
        'maxVersions': args.max_versions,
        'cells': args.cells
    }


class BatchQueryRowArgs(object):
    """
    Batch Query Row Args
    :param rows
    :type rows []
    :param max_versions
    :type max_versions int
    """
    def __init__(self, max_versions=0):
        self.rows = []
        self.max_versions = max_versions

    def append_row(self, row):
        """
        append row

        :param row:
        :type row: Row

        :return:
        :rtype
        """
        self.rows.append(row)

    def get_rows(self):
        """
        get row

        :return rows:
        :rtype Row[]
        """
        return self.rows


def batch_query_row_args_2_dict(args):
    """
    change batch_query_row_args to dict

    :param args: batch query row args
    :type args: BatchQueryRowArgs

    :return:
    :rtype dict
    """
    return {
        'maxVersions': args.max_versions,
        'rows': args.rows
    }


class ScanArgs(object):
    """
    Scan Args
    :param start_rowkey
    :type start_rowkey string
    :param include_start
    :type include_start bool
    :param stop_rowkey
    :type stop_rowkey string
    :param include_stop
    :type include_stop bool
    :param limit
    :type limit int
    :param max_versions
    :type max_versions int
    :param selector
    :type selector []
    """
    def __init__(self, start_rowkey="", include_start=True, stop_rowkey="",
                 include_stop=False, limit=0, max_versions=0):
        self.start_rowkey = start_rowkey
        self.include_start = include_start
        self.stop_rowkey = stop_rowkey
        self.include_stop = include_stop
        self.limit = limit
        self.max_versions = max_versions
        self.selector = []

    def append_selector(self, query_cell):
        """
        append selector

        :param query_cell:
        :type query_cell: QueryCell

        :return:
        :rtype
        """
        self.selector.append(query_cell)

    def get_selector(self):
        """
        get selector

        :return selector:
        :rtype query_cell[]
        """
        return self.selector


def scan_args_2_dict(args):
    """
    change scan_args to dict

    :param args: scan row args
    :type args: ScanArgs

    :return:
    :rtype dict
    """
    return {
        'startRowkey': args.start_rowkey,
        'includeStart': args.include_start,
        'stopRowkey': args.stop_rowkey,
        'includeStop': args.include_stop,
        'selector': args.selector,
        'limit': args.limit,
        'maxVersions': args.max_versions
    }

