# SPDX-License-Identifier: MIT

from __future__ import absolute_import, division, print_function

import sys

from functools import partial

from . import converters, exceptions, filters, setters, validators
from ._cmp import cmp_using
from ._config import get_run_validators, set_run_validators
from ._funcs import asdict, assoc, astuple, evolve, has, resolve_types
from ._make import (
    NOTHING,
    Attribute,
    Factory,
    attrib,
    attrs,
    fields,
    fields_dict,
    make_class,
    validate,
)
from ._version_info import VersionInfo


__version__ = "21.4.0"
__version_info__ = VersionInfo._from_version_string(__version__)

__title__ = "attrs"
__description__ = "Classes Without Boilerplate"
__url__ = "https://www.attrs.org/"
__uri__ = __url__
__doc__ = __description__ + " <" + __uri__ + ">"

__author__ = "Hynek Schlawack"
__email__ = "hs@ox.cx"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2015 Hynek Schlawack"


s = attributes = attrs
ib = attr = attrib
dataclass = partial(attrs, auto_attribs=True)  # happy Easter ;)

__all__ = [
    "Attribute",
    "Factory",
    "NOTHING",
    "asdict",
    "assoc",
    "astuple",
    "attr",
    "attrib",
    "attributes",
    "attrs",
    "cmp_using",
    "converters",
    "evolve",
    "exceptions",
    "fields",
    "fields_dict",
    "filters",
    "get_run_validators",
    "has",
    "ib",
    "make_class",
    "resolve_types",
    "s",
    "set_run_validators",
    "setters",
    "validate",
    "validators",
]

if sys.version_info[:2] >= (3, 6):
    from ._next_gen import define, field, frozen, mutable  # noqa: F401

    __all__.extend(("define", "field", "frozen", "mutable"))
