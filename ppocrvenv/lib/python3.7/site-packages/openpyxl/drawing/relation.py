# Copyright (c) 2010-2021 openpyxl

from openpyxl.xml.constants import CHART_NS

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors.excel import Relation


class ChartRelation(Serialisable):

    tagname = "chart"
    namespace = CHART_NS

    id = Relation()

    def __init__(self, id):
        self.id = id
