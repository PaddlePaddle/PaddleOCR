from pip._vendor.packaging.utils import canonicalize_name

from pip._internal.utils.typing import MYPY_CHECK_RUNNING

from .base import Requirement, format_name

if MYPY_CHECK_RUNNING:
    from typing import Sequence

    from pip._vendor.packaging.specifiers import SpecifierSet

    from pip._internal.req.req_install import InstallRequirement

    from .base import Candidate
    from .factory import Factory


class ExplicitRequirement(Requirement):
    def __init__(self, candidate):
        # type: (Candidate) -> None
        self.candidate = candidate

    def __repr__(self):
        # type: () -> str
        return "{class_name}({candidate!r})".format(
            class_name=self.__class__.__name__,
            candidate=self.candidate,
        )

    @property
    def name(self):
        # type: () -> str
        # No need to canonicalise - the candidate did this
        return self.candidate.name

    def find_matches(self):
        # type: () -> Sequence[Candidate]
        return [self.candidate]

    def is_satisfied_by(self, candidate):
        # type: (Candidate) -> bool
        return candidate == self.candidate


class SpecifierRequirement(Requirement):
    def __init__(self, ireq, factory):
        # type: (InstallRequirement, Factory) -> None
        assert ireq.link is None, "This is a link, not a specifier"
        self._ireq = ireq
        self._factory = factory
        self.extras = ireq.req.extras

    def __str__(self):
        # type: () -> str
        return str(self._ireq.req)

    def __repr__(self):
        # type: () -> str
        return "{class_name}({requirement!r})".format(
            class_name=self.__class__.__name__,
            requirement=str(self._ireq.req),
        )

    @property
    def name(self):
        # type: () -> str
        canonical_name = canonicalize_name(self._ireq.req.name)
        return format_name(canonical_name, self.extras)

    def find_matches(self):
        # type: () -> Sequence[Candidate]
        it = self._factory.iter_found_candidates(self._ireq, self.extras)
        return list(it)

    def is_satisfied_by(self, candidate):
        # type: (Candidate) -> bool
        assert candidate.name == self.name, \
            "Internal issue: Candidate is not for this requirement " \
            " {} vs {}".format(candidate.name, self.name)
        # We can safely always allow prereleases here since PackageFinder
        # already implements the prerelease logic, and would have filtered out
        # prerelease candidates if the user does not expect them.
        spec = self._ireq.req.specifier
        return spec.contains(candidate.version, prereleases=True)


class RequiresPythonRequirement(Requirement):
    """A requirement representing Requires-Python metadata.
    """
    def __init__(self, specifier, match):
        # type: (SpecifierSet, Candidate) -> None
        self.specifier = specifier
        self._candidate = match

    def __repr__(self):
        # type: () -> str
        return "{class_name}({specifier!r})".format(
            class_name=self.__class__.__name__,
            specifier=str(self.specifier),
        )

    @property
    def name(self):
        # type: () -> str
        return self._candidate.name

    def find_matches(self):
        # type: () -> Sequence[Candidate]
        if self._candidate.version in self.specifier:
            return [self._candidate]
        return []

    def is_satisfied_by(self, candidate):
        # type: (Candidate) -> bool
        assert candidate.name == self._candidate.name, "Not Python candidate"
        # We can safely always allow prereleases here since PackageFinder
        # already implements the prerelease logic, and would have filtered out
        # prerelease candidates if the user does not expect them.
        return self.specifier.contains(candidate.version, prereleases=True)
