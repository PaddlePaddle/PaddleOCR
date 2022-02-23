import json

from . import DefaultTable


class table_D__e_b_g(DefaultTable.DefaultTable):
    def decompile(self, data, ttFont):
        self.data = json.loads(data)

    def compile(self, ttFont):
        return json.dumps(self.data).encode("utf-8")

    def toXML(self, writer, ttFont):
        writer.writecdata(json.dumps(self.data))

    def fromXML(self, name, attrs, content, ttFont):
        self.data = json.loads(content)
