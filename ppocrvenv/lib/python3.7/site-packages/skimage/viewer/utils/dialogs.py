import os

from ..qt import QtWidgets


__all__ = ['open_file_dialog', 'save_file_dialog']


def _format_filename(filename):
    if isinstance(filename, tuple):
        # Handle discrepancy between PyQt4 and PySide APIs.
        filename = filename[0]
    if len(filename) == 0:
        return None
    return str(filename)


def open_file_dialog():
    """Return user-selected file path."""
    filename = QtWidgets.QFileDialog.getOpenFileName()
    filename = _format_filename(filename)
    return filename


def save_file_dialog(default_format='png'):
    """Return user-selected file path."""
    filename = QtWidgets.QFileDialog.getSaveFileName()
    filename = _format_filename(filename)
    if filename is None:
        return None
    # TODO: io plugins should assign default image formats
    basename, ext = os.path.splitext(filename)
    if not ext:
        filename = '%s.%s' % (filename, default_format)
    return filename
