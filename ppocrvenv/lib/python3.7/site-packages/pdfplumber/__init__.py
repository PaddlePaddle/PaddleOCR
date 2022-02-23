__all__ = [
    "__version__",
    "utils",
    "pdfminer",
    "open",
    "set_debug",
]

import pdfminer
import pdfminer.pdftypes

from . import utils
from ._version import __version__
from .pdf import PDF

pdfminer.pdftypes.STRICT = False
pdfminer.pdfinterp.STRICT = False

open = PDF.open


def set_debug(debug=0):
    pdfminer.debug = debug


set_debug(0)
