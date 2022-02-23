# Copyright (c) 2010-2021 openpyxl

from openpyxl.compat import safe_string

from openpyxl.descriptors import (
    Typed,
    Integer,
    Bool,
    String,
    Sequence,
)
from openpyxl.descriptors.excel import ExtensionList
from openpyxl.descriptors.serialisable import Serialisable

from .fills import PatternFill, Fill
from .fonts import Font
from .borders import Border
from .alignment import Alignment
from .protection import Protection
from .numbers import (
    NumberFormatDescriptor,
    BUILTIN_FORMATS_MAX_SIZE,
    BUILTIN_FORMATS_REVERSE,
)
from .cell_style import (
    StyleArray,
    CellStyle,
)


class NamedStyle(Serialisable):

    """
    Named and editable styles
    """

    font = Typed(expected_type=Font)
    fill = Typed(expected_type=Fill)
    border = Typed(expected_type=Border)
    alignment = Typed(expected_type=Alignment)
    number_format = NumberFormatDescriptor()
    protection = Typed(expected_type=Protection)
    builtinId = Integer(allow_none=True)
    hidden = Bool(allow_none=True)
    xfId = Integer(allow_none=True)
    name = String()
    _wb = None
    _style = StyleArray()


    def __init__(self,
                 name="Normal",
                 font=Font(),
                 fill=PatternFill(),
                 border=Border(),
                 alignment=Alignment(),
                 number_format=None,
                 protection=Protection(),
                 builtinId=None,
                 hidden=False,
                 xfId=None,
                 ):
        self.name = name
        self.font = font
        self.fill = fill
        self.border = border
        self.alignment = alignment
        self.number_format = number_format
        self.protection = protection
        self.builtinId = builtinId
        self.hidden = hidden
        self._wb = None
        self._style = StyleArray()


    def __setattr__(self, attr, value):
        super(NamedStyle, self).__setattr__(attr, value)
        if getattr(self, '_wb', None) and attr in (
           'font', 'fill', 'border', 'alignment', 'number_format', 'protection',
            ):
            self._recalculate()


    def __iter__(self):
        for key in ('name', 'builtinId', 'hidden', 'xfId'):
            value = getattr(self, key, None)
            if value is not None:
                yield key, safe_string(value)


    @property
    def xfId(self):
        """
        Index of the style in the list of named styles
        """
        return self._style.xfId


    def _set_index(self, idx):
        """
        Allow the containing list to set the index
        """
        self._style.xfId = idx


    def bind(self, wb):
        """
        Bind a named style to a workbook
        """
        self._wb = wb
        self._recalculate()


    def _recalculate(self):
        self._style.fontId =  self._wb._fonts.add(self.font)
        self._style.borderId = self._wb._borders.add(self.border)
        self._style.fillId =  self._wb._fills.add(self.fill)
        self._style.protectionId = self._wb._protections.add(self.protection)
        self._style.alignmentId = self._wb._alignments.add(self.alignment)
        fmt = self.number_format
        if fmt in BUILTIN_FORMATS_REVERSE:
            fmt = BUILTIN_FORMATS_REVERSE[fmt]
        else:
            fmt = self._wb._number_formats.add(self.number_format) + (
                  BUILTIN_FORMATS_MAX_SIZE)
        self._style.numFmtId = fmt


    def as_tuple(self):
        """Return a style array representing the current style"""
        return self._style


    def as_xf(self):
        """
        Return equivalent XfStyle
        """
        xf = CellStyle.from_array(self._style)
        xf.xfId = None
        xf.pivotButton = None
        xf.quotePrefix = None
        if self.alignment != Alignment():
            xf.alignment = self.alignment
        if self.protection != Protection():
            xf.protection = self.protection
        return xf


    def as_name(self):
        """
        Return relevant named style

        """
        named = _NamedCellStyle(
            name=self.name,
            builtinId=self.builtinId,
            hidden=self.hidden,
            xfId=self.xfId
        )
        return named


class NamedStyleList(list):
    """
    Named styles are editable and can be applied to multiple objects

    As only the index is stored in referencing objects the order mus
    be preserved.
    """

    @property
    def names(self):
        return [s.name for s in self]


    def __getitem__(self, key):
        if isinstance(key, int):
            return super(NamedStyleList, self).__getitem__(key)

        names = self.names
        if key not in names:
            raise KeyError("No named style with the name{0} exists".format(key))

        for idx, name in enumerate(names):
            if name == key:
                return self[idx]


    def append(self, style):
        if not isinstance(style, NamedStyle):
            raise TypeError("""Only NamedStyle instances can be added""")
        elif style.name in self.names:
            raise ValueError("""Style {0} exists already""".format(style.name))
        style._set_index(len(self))
        super(NamedStyleList, self).append(style)


class _NamedCellStyle(Serialisable):

    """
    Pointer-based representation of named styles in XML
    xfId refers to the corresponding CellStyleXfs

    Not used in client code.
    """

    tagname = "cellStyle"

    name = String()
    xfId = Integer()
    builtinId = Integer(allow_none=True)
    iLevel = Integer(allow_none=True)
    hidden = Bool(allow_none=True)
    customBuiltin = Bool(allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ()


    def __init__(self,
                 name=None,
                 xfId=None,
                 builtinId=None,
                 iLevel=None,
                 hidden=None,
                 customBuiltin=None,
                 extLst=None,
                ):
        self.name = name
        self.xfId = xfId
        self.builtinId = builtinId
        self.iLevel = iLevel
        self.hidden = hidden
        self.customBuiltin = customBuiltin


class _NamedCellStyleList(Serialisable):
    """
    Container for named cell style objects

    Not used in client code
    """

    tagname = "cellStyles"

    count = Integer(allow_none=True)
    cellStyle = Sequence(expected_type=_NamedCellStyle)

    __attrs__ = ("count",)

    def __init__(self,
                 count=None,
                 cellStyle=(),
                ):
        self.cellStyle = cellStyle


    @property
    def count(self):
        return len(self.cellStyle)


    @property
    def names(self):
        """
        Convert to NamedStyle objects and remove duplicates.

        In theory the highest xfId wins but in practice they are duplicates
        so it doesn't matter.
        """

        def sort_fn(v):
            return v.xfId

        styles = []
        names = set()

        for ns in sorted(self.cellStyle, key=sort_fn):
            if ns.name in names:
                continue

            style = NamedStyle(
                name=ns.name,
                hidden=ns.hidden,
                builtinId = ns.builtinId
            )
            names.add(ns.name)
            style._set_index(len(styles)) # assign xfId
            styles.append(style)

        return NamedStyleList(styles)
