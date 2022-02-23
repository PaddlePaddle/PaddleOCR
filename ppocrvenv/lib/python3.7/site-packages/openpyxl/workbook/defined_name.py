# Copyright (c) 2010-2021 openpyxl

import re

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Alias,
    Typed,
    String,
    Float,
    Integer,
    Bool,
    NoneSet,
    Set,
    Sequence,
    Descriptor,
)
from openpyxl.compat import safe_string
from openpyxl.formula import Tokenizer
from openpyxl.utils.cell import (
    SHEETRANGE_RE,
    SHEET_TITLE,
)

RESERVED = frozenset(["Print_Area", "Print_Titles", "Criteria",
                      "_FilterDatabase", "Extract", "Consolidate_Area",
                      "Sheet_Title"])

_names = "|".join(RESERVED)
RESERVED_REGEX = re.compile(r"^_xlnm\.(?P<name>{0})".format(_names))
COL_RANGE = r"""(?P<cols>[$]?[a-zA-Z]{1,3}:[$]?[a-zA-Z]{1,3})"""
COL_RANGE_RE = re.compile(COL_RANGE)
ROW_RANGE = r"""(?P<rows>[$]?\d+:[$]?\d+)"""
ROW_RANGE_RE = re.compile(ROW_RANGE)
TITLES_REGEX = re.compile("""{0}{1}?,?{2}?""".format(SHEET_TITLE, ROW_RANGE, COL_RANGE),
                          re.VERBOSE)


### utilities

def _unpack_print_titles(defn):
    """
    Extract rows and or columns from print titles so that they can be
    assigned to a worksheet
    """
    scanner = TITLES_REGEX.finditer(defn.value)
    kw = dict((k, v) for match in scanner
              for k, v in match.groupdict().items() if v)

    return kw.get('rows'), kw.get('cols')


def _unpack_print_area(defn):
    """
    Extract print area
    """
    new = []
    for m in SHEETRANGE_RE.finditer(defn.value): # can be multiple
        coord = m.group("cells")
        if coord:
            new.append(coord)
    return new


class DefinedName(Serialisable):

    tagname = "definedName"

    name = String() # unique per workbook/worksheet
    comment = String(allow_none=True)
    customMenu = String(allow_none=True)
    description = String(allow_none=True)
    help = String(allow_none=True)
    statusBar = String(allow_none=True)
    localSheetId = Integer(allow_none=True)
    hidden = Bool(allow_none=True)
    function = Bool(allow_none=True)
    vbProcedure = Bool(allow_none=True)
    xlm = Bool(allow_none=True)
    functionGroupId = Integer(allow_none=True)
    shortcutKey = String(allow_none=True)
    publishToServer = Bool(allow_none=True)
    workbookParameter = Bool(allow_none=True)
    attr_text = Descriptor()
    value = Alias("attr_text")


    def __init__(self,
                 name=None,
                 comment=None,
                 customMenu=None,
                 description=None,
                 help=None,
                 statusBar=None,
                 localSheetId=None,
                 hidden=None,
                 function=None,
                 vbProcedure=None,
                 xlm=None,
                 functionGroupId=None,
                 shortcutKey=None,
                 publishToServer=None,
                 workbookParameter=None,
                 attr_text=None
                ):
        self.name = name
        self.comment = comment
        self.customMenu = customMenu
        self.description = description
        self.help = help
        self.statusBar = statusBar
        self.localSheetId = localSheetId
        self.hidden = hidden
        self.function = function
        self.vbProcedure = vbProcedure
        self.xlm = xlm
        self.functionGroupId = functionGroupId
        self.shortcutKey = shortcutKey
        self.publishToServer = publishToServer
        self.workbookParameter = workbookParameter
        self.attr_text = attr_text


    @property
    def type(self):
        tok = Tokenizer("=" + self.value)
        parsed = tok.items[0]
        if parsed.type == "OPERAND":
            return parsed.subtype
        return parsed.type


    @property
    def destinations(self):
        if self.type == "RANGE":
            tok = Tokenizer("=" + self.value)
            for part in tok.items:
                if part.subtype == "RANGE":
                    m = SHEETRANGE_RE.match(part.value)
                    sheetname = m.group('notquoted') or m.group('quoted')
                    yield sheetname, m.group('cells')


    @property
    def is_reserved(self):
        m = RESERVED_REGEX.match(self.name)
        if m:
            return m.group("name")


    @property
    def is_external(self):
        return re.compile(r"^\[\d+\].*").match(self.value) is not None


    def __iter__(self):
        for key in self.__attrs__:
            if key == "attr_text":
                continue
            v = getattr(self, key)
            if v is not None:
                if v in RESERVED:
                    v = "_xlnm." + v
                yield key, safe_string(v)


class DefinedNameList(Serialisable):

    tagname = "definedNames"

    definedName = Sequence(expected_type=DefinedName)


    def __init__(self, definedName=()):
        self.definedName = definedName


    def _cleanup(self):
        """
        Strip invalid definitions and remove special hidden ones
        """
        valid_names = []
        for n in self.definedName:
            if n.name in ("_xlnm.Print_Titles", "_xlnm.Print_Area") and n.localSheetId is None:
                continue
            elif n.name == "_xlnm._FilterDatabase":
                continue
            valid_names.append(n)
        self.definedName = valid_names


    def _duplicate(self, defn):
        """
        Check for whether DefinedName with the same name and scope already
        exists
        """
        for d in self.definedName:
            if d.name == defn.name and d.localSheetId == defn.localSheetId:
                return True


    def append(self, defn):
        if not isinstance(defn, DefinedName):
            raise TypeError("""You can only append DefinedNames""")
        if self._duplicate(defn):
            raise ValueError("""DefinedName with the same name and scope already exists""")
        names = self.definedName[:]
        names.append(defn)
        self.definedName = names


    def __len__(self):
        return len(self.definedName)


    def __contains__(self, name):
        """
        See if a globaly defined name exists
        """
        for defn in self.definedName:
            if defn.name == name and defn.localSheetId is None:
                return True


    def __getitem__(self, name):
        """
        Get globally defined name
        """
        defn = self.get(name)
        if not defn:
            raise KeyError("No definition called {0}".format(name))
        return defn


    def get(self, name, scope=None):
        """
        Get the name assigned to a specicic sheet or global
        """
        for defn in self.definedName:
            if defn.name == name and defn.localSheetId == scope:
                return defn


    def __delitem__(self, name):
        """
        Delete a globally defined name
        """
        if not self.delete(name):
            raise KeyError("No globally defined name {0}".format(name))


    def delete(self, name, scope=None):
        """
        Delete a name assigned to a specific or global
        """
        for idx, defn in enumerate(self.definedName):
            if defn.name == name and defn.localSheetId == scope:
                del self.definedName[idx]
                return True


    def localnames(self, scope):
        """
        Provide a list of all names for a particular worksheet
        """
        return [defn.name for defn in self.definedName if defn.localSheetId == scope]
