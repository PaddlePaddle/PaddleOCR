from __future__ import absolute_import, unicode_literals

from virtualenv.util.path import Path

from ..via_template import ViaTemplateActivator


class PowerShellActivator(ViaTemplateActivator):
    def templates(self):
        yield Path("activate.ps1")
