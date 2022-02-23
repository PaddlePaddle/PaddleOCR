from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import pandas._libs.json as json
from pandas._typing import StorageOptions

from pandas.io.excel._base import ExcelWriter
from pandas.io.excel._util import (
    combine_kwargs,
    validate_freeze_panes,
)

if TYPE_CHECKING:
    from xlwt import XFStyle


class XlwtWriter(ExcelWriter):
    engine = "xlwt"
    supported_extensions = (".xls",)

    def __init__(
        self,
        path,
        engine=None,
        date_format=None,
        datetime_format=None,
        encoding=None,
        mode: str = "w",
        storage_options: StorageOptions = None,
        if_sheet_exists: str | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        # Use the xlwt module as the Excel writer.
        import xlwt

        engine_kwargs = combine_kwargs(engine_kwargs, kwargs)

        if mode == "a":
            raise ValueError("Append mode is not supported with xlwt!")

        super().__init__(
            path,
            mode=mode,
            storage_options=storage_options,
            if_sheet_exists=if_sheet_exists,
            engine_kwargs=engine_kwargs,
        )

        if encoding is None:
            encoding = "ascii"
        self.book = xlwt.Workbook(encoding=encoding)
        self.fm_datetime = xlwt.easyxf(num_format_str=self.datetime_format)
        self.fm_date = xlwt.easyxf(num_format_str=self.date_format)

    def save(self):
        """
        Save workbook to disk.
        """
        if self.sheets:
            # fails when the ExcelWriter is just opened and then closed
            self.book.save(self.handles.handle)

    def write_cells(
        self, cells, sheet_name=None, startrow=0, startcol=0, freeze_panes=None
    ):

        sheet_name = self._get_sheet_name(sheet_name)

        if sheet_name in self.sheets:
            wks = self.sheets[sheet_name]
        else:
            wks = self.book.add_sheet(sheet_name)
            self.sheets[sheet_name] = wks

        if validate_freeze_panes(freeze_panes):
            wks.set_panes_frozen(True)
            wks.set_horz_split_pos(freeze_panes[0])
            wks.set_vert_split_pos(freeze_panes[1])

        style_dict: dict[str, XFStyle] = {}

        for cell in cells:
            val, fmt = self._value_with_fmt(cell.val)

            stylekey = json.dumps(cell.style)
            if fmt:
                stylekey += fmt

            if stylekey in style_dict:
                style = style_dict[stylekey]
            else:
                style = self._convert_to_style(cell.style, fmt)
                style_dict[stylekey] = style

            if cell.mergestart is not None and cell.mergeend is not None:
                wks.write_merge(
                    startrow + cell.row,
                    startrow + cell.mergestart,
                    startcol + cell.col,
                    startcol + cell.mergeend,
                    val,
                    style,
                )
            else:
                wks.write(startrow + cell.row, startcol + cell.col, val, style)

    @classmethod
    def _style_to_xlwt(
        cls, item, firstlevel: bool = True, field_sep=",", line_sep=";"
    ) -> str:
        """
        helper which recursively generate an xlwt easy style string
        for example:

            hstyle = {"font": {"bold": True},
            "border": {"top": "thin",
                    "right": "thin",
                    "bottom": "thin",
                    "left": "thin"},
            "align": {"horiz": "center"}}
            will be converted to
            font: bold on; \
                    border: top thin, right thin, bottom thin, left thin; \
                    align: horiz center;
        """
        if hasattr(item, "items"):
            if firstlevel:
                it = [
                    f"{key}: {cls._style_to_xlwt(value, False)}"
                    for key, value in item.items()
                ]
                out = f"{line_sep.join(it)} "
                return out
            else:
                it = [
                    f"{key} {cls._style_to_xlwt(value, False)}"
                    for key, value in item.items()
                ]
                out = f"{field_sep.join(it)} "
                return out
        else:
            item = f"{item}"
            item = item.replace("True", "on")
            item = item.replace("False", "off")
            return item

    @classmethod
    def _convert_to_style(cls, style_dict, num_format_str=None):
        """
        converts a style_dict to an xlwt style object

        Parameters
        ----------
        style_dict : style dictionary to convert
        num_format_str : optional number format string
        """
        import xlwt

        if style_dict:
            xlwt_stylestr = cls._style_to_xlwt(style_dict)
            style = xlwt.easyxf(xlwt_stylestr, field_sep=",", line_sep=";")
        else:
            style = xlwt.XFStyle()
        if num_format_str is not None:
            style.num_format_str = num_format_str

        return style
