from __future__ import absolute_import, unicode_literals

from ..py_info import PythonInfo
from ..py_spec import PythonSpec
from .pep514 import discover_pythons


class Pep514PythonInfo(PythonInfo):
    """ """


def propose_interpreters(spec, cache_dir, env):
    # see if PEP-514 entries are good

    # start with higher python versions in an effort to use the latest version available
    # and prefer PythonCore over conda pythons (as virtualenv is mostly used by non conda tools)
    existing = list(discover_pythons())
    existing.sort(
        key=lambda i: tuple(-1 if j is None else j for j in i[1:4]) + (1 if i[0] == "PythonCore" else 0,), reverse=True
    )

    for name, major, minor, arch, exe, _ in existing:
        # pre-filter
        if name in ("PythonCore", "ContinuumAnalytics"):
            name = "CPython"
        registry_spec = PythonSpec(None, name, major, minor, None, arch, exe)
        if registry_spec.satisfies(spec):
            interpreter = Pep514PythonInfo.from_exe(exe, cache_dir, env=env, raise_on_error=False)
            if interpreter is not None:
                if interpreter.satisfies(spec, impl_must_match=True):
                    yield interpreter
