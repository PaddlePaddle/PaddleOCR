from __future__ import absolute_import

import optparse

from flake8_import_order import __version__
from flake8_import_order.checker import (
    DEFAULT_IMPORT_ORDER_STYLE, ImportOrderChecker,
)
from flake8_import_order.styles import list_entry_points, lookup_entry_point


class Linter(ImportOrderChecker):
    name = "import-order"
    version = __version__

    def __init__(self, tree, filename, lines=None):
        super(Linter, self).__init__(filename, tree)
        self.lines = lines

    @classmethod
    def add_options(cls, parser):
        # List of application import names. They go last.
        register_opt(
            parser,
            "--application-import-names",
            default="",
            action="store",
            type="string",
            help="Import names to consider as application-specific",
            parse_from_config=True,
            comma_separated_list=True,
        )
        register_opt(
            parser,
            "--application-package-names",
            default="",
            action="store",
            type="string",
            help=("Package names to consider as company-specific "
                  "(used only by 'appnexus' style)"),
            parse_from_config=True,
            comma_separated_list=True,
        )
        register_opt(
            parser,
            "--import-order-style",
            default=DEFAULT_IMPORT_ORDER_STYLE,
            action="store",
            type="string",
            help=("Style to follow. Available: " +
                  ", ".join(cls.list_available_styles())),
            parse_from_config=True,
        )

    @staticmethod
    def list_available_styles():
        entry_points = list_entry_points()
        return sorted(entry_point.name for entry_point in entry_points)

    @classmethod
    def parse_options(cls, options):
        optdict = {}

        names = options.application_import_names
        if not isinstance(names, list):
            names = options.application_import_names.split(",")

        pkg_names = options.application_package_names
        if not isinstance(pkg_names, list):
            pkg_names = options.application_package_names.split(",")

        style_entry_point = lookup_entry_point(options.import_order_style)

        optdict = dict(
            application_import_names=[n.strip() for n in names],
            application_package_names=[p.strip() for p in pkg_names],
            import_order_style=style_entry_point,
        )

        cls.options = optdict

    def error(self, error):
        return (
            error.lineno,
            0,
            "{0} {1}".format(error.code, error.message),
            Linter,
        )

    def run(self):
        for error in self.check_order():
            yield error


def register_opt(parser, *args, **kwargs):
    try:
        # Flake8 3.x registration
        parser.add_option(*args, **kwargs)
    except (optparse.OptionError, TypeError):
        # Flake8 2.x registration
        parse_from_config = kwargs.pop('parse_from_config', False)
        kwargs.pop('comma_separated_list', False)
        kwargs.pop('normalize_paths', False)
        parser.add_option(*args, **kwargs)
        if parse_from_config:
            parser.config_options.append(args[-1].lstrip('-'))
