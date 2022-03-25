from __future__ import absolute_import

from pylama.lint import Linter as BaseLinter

from flake8_import_order import __version__
from flake8_import_order.checker import (
    DEFAULT_IMPORT_ORDER_STYLE, ImportOrderChecker,
)
from flake8_import_order.styles import lookup_entry_point


class Linter(ImportOrderChecker, BaseLinter):
    name = "import-order"
    version = __version__

    def __init__(self):
        super(Linter, self).__init__(None, None)

    def allow(self, path):
        return path.endswith(".py")

    def error(self, error):
        return {
            'lnum': error.lineno,
            'col': 0,
            'text': error.message,
            'type': error.code,
        }

    def run(self, path, **meta):
        self.filename = path
        self.ast_tree = None
        meta.setdefault('import_order_style', DEFAULT_IMPORT_ORDER_STYLE)
        meta['import_order_style'] = lookup_entry_point(
            meta['import_order_style']
        )
        self.options = meta

        for error in self.check_order():
            yield error
