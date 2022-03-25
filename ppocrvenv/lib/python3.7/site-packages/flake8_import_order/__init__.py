import ast
from collections import namedtuple
from enum import IntEnum

from flake8_import_order.__about__ import (
    __author__, __copyright__, __email__, __license__, __summary__, __title__,
    __uri__, __version__,
)
from flake8_import_order.stdlib_list import STDLIB_NAMES

__all__ = [
    "__title__", "__summary__", "__uri__", "__version__", "__author__",
    "__email__", "__license__", "__copyright__",
]

DEFAULT_IMPORT_ORDER_STYLE = 'cryptography'

ClassifiedImport = namedtuple(
    'ClassifiedImport',
    ['type', 'is_from', 'modules', 'names', 'lineno', 'level', 'package'],
)
NewLine = namedtuple('NewLine', ['lineno'])


class ImportType(IntEnum):
    FUTURE = 0
    STDLIB = 10
    THIRD_PARTY = 20
    APPLICATION_PACKAGE = 30
    APPLICATION = 40
    APPLICATION_RELATIVE = 50
    MIXED = -1


def get_package_names(name):
    tree = ast.parse(name)
    parts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts.append(node.attr)
        if isinstance(node, ast.Name):
            parts.append(node.id)

    if not parts:
        return []

    last_package_name = parts.pop()
    package_names = [last_package_name]

    for part in reversed(parts):
        last_package_name = '%s.%s' % (last_package_name, part)
        package_names.append(last_package_name)

    return package_names


def root_package_name(name):
    tree = ast.parse(name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            return node.id
    else:
        return None


class ImportVisitor(ast.NodeVisitor):

    def __init__(self, application_import_names, application_package_names):
        self.imports = []
        self.application_import_names = frozenset(application_import_names)
        self.application_package_names = frozenset(application_package_names)

    def visit_Import(self, node):  # noqa: N802
        if node.col_offset == 0:
            modules = [alias.name for alias in node.names]
            types_ = {self._classify_type(module) for module in modules}
            if len(types_) == 1:
                type_ = types_.pop()
            else:
                type_ = ImportType.MIXED
            classified_import = ClassifiedImport(
                type_, False, modules, [], node.lineno, 0,
                root_package_name(modules[0]),
            )
            self.imports.append(classified_import)

    def visit_ImportFrom(self, node):  # noqa: N802
        if node.col_offset == 0:
            module = node.module or ''
            if node.level > 0:
                type_ = ImportType.APPLICATION_RELATIVE
            else:
                type_ = self._classify_type(module)
            names = [alias.name for alias in node.names]
            classified_import = ClassifiedImport(
                type_, True, [module], names,
                node.lineno, node.level,
                root_package_name(module),
            )
            self.imports.append(classified_import)

    def _classify_type(self, module):
        package_names = get_package_names(module)

        # Walk through package names from most-specific to least-specific,
        # taking the first match found.
        for package in reversed(package_names):
            if package == "__future__":
                return ImportType.FUTURE
            elif package in self.application_import_names:
                return ImportType.APPLICATION
            elif package in self.application_package_names:
                return ImportType.APPLICATION_PACKAGE
            elif package in STDLIB_NAMES:
                return ImportType.STDLIB

        # Not future, stdlib or an application import.
        # Must be 3rd party.
        return ImportType.THIRD_PARTY
