from pip._vendor.packaging.utils import canonicalize_name

from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import Optional, Sequence, Set

    from pip._internal.req.req_install import InstallRequirement
    from pip._vendor.packaging.version import _BaseVersion


def format_name(project, extras):
    # type: (str, Set[str]) -> str
    if not extras:
        return project
    canonical_extras = sorted(canonicalize_name(e) for e in extras)
    return "{}[{}]".format(project, ",".join(canonical_extras))


class Requirement(object):
    @property
    def name(self):
        # type: () -> str
        raise NotImplementedError("Subclass should override")

    def find_matches(self):
        # type: () -> Sequence[Candidate]
        raise NotImplementedError("Subclass should override")

    def is_satisfied_by(self, candidate):
        # type: (Candidate) -> bool
        return False


class Candidate(object):
    @property
    def name(self):
        # type: () -> str
        raise NotImplementedError("Override in subclass")

    @property
    def version(self):
        # type: () -> _BaseVersion
        raise NotImplementedError("Override in subclass")

    def get_dependencies(self):
        # type: () -> Sequence[Requirement]
        raise NotImplementedError("Override in subclass")

    def get_install_requirement(self):
        # type: () -> Optional[InstallRequirement]
        raise NotImplementedError("Override in subclass")
