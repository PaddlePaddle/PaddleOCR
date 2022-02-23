"""Utilities to read and write images in various formats.

The following plug-ins are available:

"""

from .manage_plugins import *
from .sift import *
from .collection import *

from ._io import *
from ._image_stack import *


reset_plugins()

WRAP_LEN = 73


def _separator(char, lengths):
    return [char * separator_length for separator_length in lengths]


def _format_plugin_info_table(info_table, column_lengths):
    """Add separators and column titles to plugin info table."""
    info_table.insert(0, _separator('=', column_lengths))
    info_table.insert(1, ('Plugin', 'Description'))
    info_table.insert(2, _separator('-', column_lengths))
    info_table.append(_separator('=', column_lengths))


def _update_doc(doc):
    """Add a list of plugins to the module docstring, formatted as
    a ReStructuredText table.
    """
    from textwrap import wrap

    info_table = [(p, plugin_info(p).get('description', 'no description'))
                  for p in available_plugins if not p == 'test']

    if len(info_table) > 0:
        name_length = max([len(n) for (n, _) in info_table])
    else:
        name_length = 0

    description_length = WRAP_LEN - 1 - name_length
    column_lengths = [name_length, description_length]
    _format_plugin_info_table(info_table, column_lengths)

    for (name, plugin_description) in info_table:
        description_lines = wrap(plugin_description, description_length)
        name_column = [name]
        name_column.extend(['' for _ in range(len(description_lines) - 1)])
        for name, description in zip(name_column, description_lines):
            doc += "%s %s\n" % (name.ljust(name_length), description)
    doc = doc.strip()

    return doc


if __doc__ is not None:
    __doc__ = _update_doc(__doc__)
