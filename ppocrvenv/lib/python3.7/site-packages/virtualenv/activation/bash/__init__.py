from __future__ import absolute_import, unicode_literals

from virtualenv.util.path import Path

from ..via_template import ViaTemplateActivator


class BashActivator(ViaTemplateActivator):
    def templates(self):
        yield Path("activate.sh")

    def as_name(self, template):
        return template.stem
