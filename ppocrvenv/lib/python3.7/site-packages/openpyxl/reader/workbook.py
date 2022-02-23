# Copyright (c) 2010-2021 openpyxl

import posixpath
from warnings import warn

from openpyxl.xml.functions import fromstring

from openpyxl.packaging.relationship import (
    get_dependents,
    get_rels_path,
    get_rel,
)
from openpyxl.packaging.manifest import Manifest
from openpyxl.packaging.workbook import WorkbookPackage
from openpyxl.workbook import Workbook
from openpyxl.workbook.defined_name import (
    _unpack_print_area,
    _unpack_print_titles,
)
from openpyxl.workbook.external_link.external import read_external_link
from openpyxl.pivot.cache import CacheDefinition
from openpyxl.pivot.record import RecordList

from openpyxl.utils.datetime import CALENDAR_MAC_1904


class WorkbookParser:

    _rels = None

    def __init__(self, archive, workbook_part_name, keep_links=True):
        self.archive = archive
        self.workbook_part_name = workbook_part_name
        self.wb = Workbook()
        self.keep_links = keep_links
        self.sheets = []


    @property
    def rels(self):
        if self._rels is None:
            self._rels = get_dependents(self.archive, get_rels_path(self.workbook_part_name))
        return self._rels


    def parse(self):
        src = self.archive.read(self.workbook_part_name)
        node = fromstring(src)
        package = WorkbookPackage.from_tree(node)
        if package.properties.date1904:
            self.wb.epoch = CALENDAR_MAC_1904

        self.wb.code_name = package.properties.codeName
        self.wb.active = package.active
        self.wb.views = package.bookViews
        self.sheets = package.sheets
        self.wb.calculation = package.calcPr
        self.caches = package.pivotCaches

        #external links contain cached worksheets and can be very big
        if not self.keep_links:
            package.externalReferences = []

        for ext_ref in package.externalReferences:
            rel = self.rels[ext_ref.id]
            self.wb._external_links.append(
                read_external_link(self.archive, rel.Target)
            )

        if package.definedNames:
            package.definedNames._cleanup()
            self.wb.defined_names = package.definedNames

        self.wb.security = package.workbookProtection


    def find_sheets(self):
        """
        Find all sheets in the workbook and return the link to the source file.

        Older XLSM files sometimes contain invalid sheet elements.
        Warn user when these are removed.
        """

        for sheet in self.sheets:
            if not sheet.id:
                msg = "File contains an invalid specification for {0}. This will be removed".format(sheet.name)
                warn(msg)
                continue
            yield sheet, self.rels[sheet.id]


    def assign_names(self):
        """
        Bind reserved names to parsed worksheets
        """
        defns = []

        for defn in self.wb.defined_names.definedName:
            reserved = defn.is_reserved
            if reserved in ("Print_Titles", "Print_Area"):
                sheet = self.wb._sheets[defn.localSheetId]
                if reserved == "Print_Titles":
                    rows, cols = _unpack_print_titles(defn)
                    sheet.print_title_rows = rows
                    sheet.print_title_cols = cols
                elif reserved == "Print_Area":
                    sheet.print_area = _unpack_print_area(defn)
            else:
                defns.append(defn)
        self.wb.defined_names.definedName = defns


    @property
    def pivot_caches(self):
        """
        Get PivotCache objects
        """
        d = {}
        for c in self.caches:
            cache = get_rel(self.archive, self.rels, id=c.id, cls=CacheDefinition)
            if cache.deps:
                records = get_rel(self.archive, cache.deps, cache.id, RecordList)
                cache.records = records
            d[c.cacheId]  = cache
        return d
