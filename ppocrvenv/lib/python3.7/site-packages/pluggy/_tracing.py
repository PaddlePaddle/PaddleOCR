"""
Tracing utils
"""


class TagTracer:
    def __init__(self):
        self._tags2proc = {}
        self._writer = None
        self.indent = 0

    def get(self, name):
        return TagTracerSub(self, (name,))

    def _format_message(self, tags, args):
        if isinstance(args[-1], dict):
            extra = args[-1]
            args = args[:-1]
        else:
            extra = {}

        content = " ".join(map(str, args))
        indent = "  " * self.indent

        lines = ["{}{} [{}]\n".format(indent, content, ":".join(tags))]

        for name, value in extra.items():
            lines.append(f"{indent}    {name}: {value}\n")

        return "".join(lines)

    def _processmessage(self, tags, args):
        if self._writer is not None and args:
            self._writer(self._format_message(tags, args))
        try:
            processor = self._tags2proc[tags]
        except KeyError:
            pass
        else:
            processor(tags, args)

    def setwriter(self, writer):
        self._writer = writer

    def setprocessor(self, tags, processor):
        if isinstance(tags, str):
            tags = tuple(tags.split(":"))
        else:
            assert isinstance(tags, tuple)
        self._tags2proc[tags] = processor


class TagTracerSub:
    def __init__(self, root, tags):
        self.root = root
        self.tags = tags

    def __call__(self, *args):
        self.root._processmessage(self.tags, args)

    def get(self, name):
        return self.__class__(self.root, self.tags + (name,))
