from io import BytesIO
from fontTools.ttLib.tables.C_F_F_ import table_C_F_F_


class table_C_F_F__2(table_C_F_F_):

    def decompile(self, data, otFont):
        self.cff.decompile(BytesIO(data), otFont, isCFF2=True)
        assert len(self.cff) == 1, "can't deal with multi-font CFF tables."

    def compile(self, otFont):
        f = BytesIO()
        self.cff.compile(f, otFont, isCFF2=True)
        return f.getvalue()
