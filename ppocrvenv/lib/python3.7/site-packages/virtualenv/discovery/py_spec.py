"""A Python specification is an abstract requirement definition of a interpreter"""
from __future__ import absolute_import, unicode_literals

import os
import re
import sys
from collections import OrderedDict

from virtualenv.info import fs_is_case_sensitive
from virtualenv.util.six import ensure_str

PATTERN = re.compile(r"^(?P<impl>[a-zA-Z]+)?(?P<version>[0-9.]+)?(?:-(?P<arch>32|64))?$")
IS_WIN = sys.platform == "win32"


class PythonSpec(object):
    """Contains specification about a Python Interpreter"""

    def __init__(self, str_spec, implementation, major, minor, micro, architecture, path):
        self.str_spec = str_spec
        self.implementation = implementation
        self.major = major
        self.minor = minor
        self.micro = micro
        self.architecture = architecture
        self.path = path

    @classmethod
    def from_string_spec(cls, string_spec):
        impl, major, minor, micro, arch, path = None, None, None, None, None, None
        if os.path.isabs(string_spec):
            path = string_spec
        else:
            ok = False
            match = re.match(PATTERN, string_spec)
            if match:

                def _int_or_none(val):
                    return None if val is None else int(val)

                try:
                    groups = match.groupdict()
                    version = groups["version"]
                    if version is not None:
                        versions = tuple(int(i) for i in version.split(".") if i)
                        if len(versions) > 3:
                            raise ValueError
                        if len(versions) == 3:
                            major, minor, micro = versions
                        elif len(versions) == 2:
                            major, minor = versions
                        elif len(versions) == 1:
                            version_data = versions[0]
                            major = int(str(version_data)[0])  # first digit major
                            if version_data > 9:
                                minor = int(str(version_data)[1:])
                    ok = True
                except ValueError:
                    pass
                else:
                    impl = groups["impl"]
                    if impl == "py" or impl == "python":
                        impl = "CPython"
                    arch = _int_or_none(groups["arch"])

            if not ok:
                path = string_spec

        return cls(string_spec, impl, major, minor, micro, arch, path)

    def generate_names(self):
        impls = OrderedDict()
        if self.implementation:
            # first consider implementation as it is
            impls[self.implementation] = False
            if fs_is_case_sensitive():
                # for case sensitive file systems consider lower and upper case versions too
                # trivia: MacBooks and all pre 2018 Windows-es were case insensitive by default
                impls[self.implementation.lower()] = False
                impls[self.implementation.upper()] = False
        impls["python"] = True  # finally consider python as alias, implementation must match now
        version = self.major, self.minor, self.micro
        try:
            version = version[: version.index(None)]
        except ValueError:
            pass
        for impl, match in impls.items():
            for at in range(len(version), -1, -1):
                cur_ver = version[0:at]
                spec = "{}{}".format(impl, ".".join(str(i) for i in cur_ver))
                yield spec, match

    @property
    def is_abs(self):
        return self.path is not None and os.path.isabs(self.path)

    def satisfies(self, spec):
        """called when there's a candidate metadata spec to see if compatible - e.g. PEP-514 on Windows"""
        if spec.is_abs and self.is_abs and self.path != spec.path:
            return False
        if spec.implementation is not None and spec.implementation.lower() != self.implementation.lower():
            return False
        if spec.architecture is not None and spec.architecture != self.architecture:
            return False

        for our, req in zip((self.major, self.minor, self.micro), (spec.major, spec.minor, spec.micro)):
            if req is not None and our is not None and our != req:
                return False
        return True

    def __unicode__(self):
        return "{}({})".format(
            type(self).__name__,
            ", ".join(
                "{}={}".format(k, getattr(self, k))
                for k in ("implementation", "major", "minor", "micro", "architecture", "path")
                if getattr(self, k) is not None
            ),
        )

    def __repr__(self):
        return ensure_str(self.__unicode__())
