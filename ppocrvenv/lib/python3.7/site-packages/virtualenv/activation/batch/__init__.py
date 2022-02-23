from __future__ import absolute_import, unicode_literals

import os

from virtualenv.util.path import Path

from ..via_template import ViaTemplateActivator


class BatchActivator(ViaTemplateActivator):
    @classmethod
    def supports(cls, interpreter):
        return interpreter.os == "nt"

    def templates(self):
        yield Path("activate.bat")
        yield Path("deactivate.bat")
        yield Path("pydoc.bat")

    def instantiate_template(self, replacements, template, creator):
        # ensure the text has all newlines as \r\n - required by batch
        base = super(BatchActivator, self).instantiate_template(replacements, template, creator)
        return base.replace(os.linesep, "\n").replace("\n", os.linesep)
