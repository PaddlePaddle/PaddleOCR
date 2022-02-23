# -*- coding: utf-8 -*-
"""
Interface for natsort to access fastnumbers functions without
having to worry if it is actually installed.
"""
import re

__all__ = ["fast_float", "fast_int"]


def is_supported_fastnumbers(fastnumbers_version: str) -> bool:
    match = re.match(
        r"^(\d+)\.(\d+)(\.(\d+))?([ab](\d+))?$",
        fastnumbers_version,
        flags=re.ASCII,
    )

    if not match:
        raise ValueError(
            "Invalid fastnumbers version number '{}'".format(fastnumbers_version)
        )

    (major, minor, patch) = match.group(1, 2, 4)

    return (int(major), int(minor), int(patch)) >= (2, 0, 0)


# If the user has fastnumbers installed, they will get great speed
# benefits. If not, we use the simulated functions that come with natsort.
try:
    # noinspection PyPackageRequirements
    from fastnumbers import fast_float, fast_int, __version__ as fn_ver

    # Require >= version 2.0.0.
    if not is_supported_fastnumbers(fn_ver):
        raise ImportError  # pragma: no cover
except ImportError:
    from natsort.compat.fake_fastnumbers import fast_float, fast_int  # type: ignore
