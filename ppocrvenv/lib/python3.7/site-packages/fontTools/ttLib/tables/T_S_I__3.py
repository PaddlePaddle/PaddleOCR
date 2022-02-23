""" TSI{0,1,2,3,5} are private tables used by Microsoft Visual TrueType (VTT)
tool to store its hinting source data.

TSI3 contains the text of the glyph programs in the form of 'VTTTalk' code.
"""
from fontTools import ttLib

superclass = ttLib.getTableClass("TSI1")

class table_T_S_I__3(superclass):

	extras = {0xfffa: "reserved0", 0xfffb: "reserved1", 0xfffc: "reserved2", 0xfffd: "reserved3"}

	indextable = "TSI2"
