from __future__ import absolute_import, unicode_literals

import logging
from collections import OrderedDict

from virtualenv.util.six import ensure_text


class PyEnvCfg(object):
    def __init__(self, content, path):
        self.content = content
        self.path = path

    @classmethod
    def from_folder(cls, folder):
        return cls.from_file(folder / "pyvenv.cfg")

    @classmethod
    def from_file(cls, path):
        content = cls._read_values(path) if path.exists() else OrderedDict()
        return PyEnvCfg(content, path)

    @staticmethod
    def _read_values(path):
        content = OrderedDict()
        for line in path.read_text(encoding="utf-8").splitlines():
            equals_at = line.index("=")
            key = line[:equals_at].strip()
            value = line[equals_at + 1 :].strip()
            content[key] = value
        return content

    def write(self):
        logging.debug("write %s", ensure_text(str(self.path)))
        text = ""
        for key, value in self.content.items():
            line = "{} = {}".format(key, value)
            logging.debug("\t%s", line)
            text += line
            text += "\n"
        self.path.write_text(text, encoding="utf-8")

    def refresh(self):
        self.content = self._read_values(self.path)
        return self.content

    def __setitem__(self, key, value):
        self.content[key] = value

    def __getitem__(self, key):
        return self.content[key]

    def __contains__(self, item):
        return item in self.content

    def update(self, other):
        self.content.update(other)
        return self

    def __repr__(self):
        return "{}(path={})".format(self.__class__.__name__, self.path)
