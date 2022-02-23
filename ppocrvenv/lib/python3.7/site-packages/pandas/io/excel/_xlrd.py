from datetime import time

import numpy as np

from pandas._typing import StorageOptions
from pandas.compat._optional import import_optional_dependency

from pandas.io.excel._base import BaseExcelReader


class XlrdReader(BaseExcelReader):
    def __init__(self, filepath_or_buffer, storage_options: StorageOptions = None):
        """
        Reader using xlrd engine.

        Parameters
        ----------
        filepath_or_buffer : str, path object or Workbook
            Object to be parsed.
        storage_options : dict, optional
            passed to fsspec for appropriate URLs (see ``_get_filepath_or_buffer``)
        """
        err_msg = "Install xlrd >= 1.0.0 for Excel support"
        import_optional_dependency("xlrd", extra=err_msg)
        super().__init__(filepath_or_buffer, storage_options=storage_options)

    @property
    def _workbook_class(self):
        from xlrd import Book

        return Book

    def load_workbook(self, filepath_or_buffer):
        from xlrd import open_workbook

        if hasattr(filepath_or_buffer, "read"):
            data = filepath_or_buffer.read()
            return open_workbook(file_contents=data)
        else:
            return open_workbook(filepath_or_buffer)

    @property
    def sheet_names(self):
        return self.book.sheet_names()

    def get_sheet_by_name(self, name):
        self.raise_if_bad_sheet_by_name(name)
        return self.book.sheet_by_name(name)

    def get_sheet_by_index(self, index):
        self.raise_if_bad_sheet_by_index(index)
        return self.book.sheet_by_index(index)

    def get_sheet_data(self, sheet, convert_float):
        from xlrd import (
            XL_CELL_BOOLEAN,
            XL_CELL_DATE,
            XL_CELL_ERROR,
            XL_CELL_NUMBER,
            xldate,
        )

        epoch1904 = self.book.datemode

        def _parse_cell(cell_contents, cell_typ):
            """
            converts the contents of the cell into a pandas appropriate object
            """
            if cell_typ == XL_CELL_DATE:

                # Use the newer xlrd datetime handling.
                try:
                    cell_contents = xldate.xldate_as_datetime(cell_contents, epoch1904)
                except OverflowError:
                    return cell_contents

                # Excel doesn't distinguish between dates and time,
                # so we treat dates on the epoch as times only.
                # Also, Excel supports 1900 and 1904 epochs.
                year = (cell_contents.timetuple())[0:3]
                if (not epoch1904 and year == (1899, 12, 31)) or (
                    epoch1904 and year == (1904, 1, 1)
                ):
                    cell_contents = time(
                        cell_contents.hour,
                        cell_contents.minute,
                        cell_contents.second,
                        cell_contents.microsecond,
                    )

            elif cell_typ == XL_CELL_ERROR:
                cell_contents = np.nan
            elif cell_typ == XL_CELL_BOOLEAN:
                cell_contents = bool(cell_contents)
            elif convert_float and cell_typ == XL_CELL_NUMBER:
                # GH5394 - Excel 'numbers' are always floats
                # it's a minimal perf hit and less surprising
                val = int(cell_contents)
                if val == cell_contents:
                    cell_contents = val
            return cell_contents

        data = []

        for i in range(sheet.nrows):
            row = [
                _parse_cell(value, typ)
                for value, typ in zip(sheet.row_values(i), sheet.row_types(i))
            ]
            data.append(row)

        return data
