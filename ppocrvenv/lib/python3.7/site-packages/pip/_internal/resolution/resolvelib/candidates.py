import logging
import sys

from pip._vendor.packaging.specifiers import InvalidSpecifier, SpecifierSet
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import Version

from pip._internal.req.constructors import (
    install_req_from_editable,
    install_req_from_line,
)
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.misc import normalize_version_info
from pip._internal.utils.packaging import get_requires_python
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

from .base import Candidate, format_name

if MYPY_CHECK_RUNNING:
    from typing import Any, Optional, Sequence, Set, Tuple, Union

    from pip._vendor.packaging.version import _BaseVersion
    from pip._vendor.pkg_resources import Distribution

    from pip._internal.distributions import AbstractDistribution
    from pip._internal.models.link import Link

    from .base import Requirement
    from .factory import Factory

    BaseCandidate = Union[
        "AlreadyInstalledCandidate",
        "EditableCandidate",
        "LinkCandidate",
    ]


logger = logging.getLogger(__name__)


def make_install_req_from_link(link, parent):
    # type: (Link, InstallRequirement) -> InstallRequirement
    assert not parent.editable, "parent is editable"
    return install_req_from_line(
        link.url,
        comes_from=parent.comes_from,
        use_pep517=parent.use_pep517,
        isolated=parent.isolated,
        constraint=parent.constraint,
        options=dict(
            install_options=parent.install_options,
            global_options=parent.global_options,
            hashes=parent.hash_options
        ),
    )


def make_install_req_from_editable(link, parent):
    # type: (Link, InstallRequirement) -> InstallRequirement
    assert parent.editable, "parent not editable"
    return install_req_from_editable(
        link.url,
        comes_from=parent.comes_from,
        use_pep517=parent.use_pep517,
        isolated=parent.isolated,
        constraint=parent.constraint,
        options=dict(
            install_options=parent.install_options,
            global_options=parent.global_options,
            hashes=parent.hash_options
        ),
    )


def make_install_req_from_dist(dist, parent):
    # type: (Distribution, InstallRequirement) -> InstallRequirement
    ireq = install_req_from_line(
        "{}=={}".format(
            canonicalize_name(dist.project_name),
            dist.parsed_version,
        ),
        comes_from=parent.comes_from,
        use_pep517=parent.use_pep517,
        isolated=parent.isolated,
        constraint=parent.constraint,
        options=dict(
            install_options=parent.install_options,
            global_options=parent.global_options,
            hashes=parent.hash_options
        ),
    )
    ireq.satisfied_by = dist
    return ireq


class _InstallRequirementBackedCandidate(Candidate):
    def __init__(
        self,
        link,          # type: Link
        ireq,          # type: InstallRequirement
        factory,       # type: Factory
        name=None,     # type: Optional[str]
        version=None,  # type: Optional[_BaseVersion]
    ):
        # type: (...) -> None
        self.link = link
        self._factory = factory
        self._ireq = ireq
        self._name = name
        self._version = version
        self._dist = None  # type: Optional[Distribution]

    def __repr__(self):
        # type: () -> str
        return "{class_name}({link!r})".format(
            class_name=self.__class__.__name__,
            link=str(self.link),
        )

    def __eq__(self, other):
        # type: (Any) -> bool
        if isinstance(other, self.__class__):
            return self.link == other.link
        return False

    # Needed for Python 2, which does not implement this by default
    def __ne__(self, other):
        # type: (Any) -> bool
        return not self.__eq__(other)

    @property
    def name(self):
        # type: () -> str
        """The normalised name of the project the candidate refers to"""
        if self._name is None:
            self._name = canonicalize_name(self.dist.project_name)
        return self._name

    @property
    def version(self):
        # type: () -> _BaseVersion
        if self._version is None:
            self._version = self.dist.parsed_version
        return self._version

    def _prepare_abstract_distribution(self):
        # type: () -> AbstractDistribution
        raise NotImplementedError("Override in subclass")

    def _prepare(self):
        # type: () -> None
        if self._dist is not None:
            return

        abstract_dist = self._prepare_abstract_distribution()
        self._dist = abstract_dist.get_pkg_resources_distribution()
        assert self._dist is not None, "Distribution already installed"

        # TODO: Abort cleanly here, as the resolution has been
        #       based on the wrong name/version until now, and
        #       so is wrong.
        # TODO: (Longer term) Rather than abort, reject this candidate
        #       and backtrack. This would need resolvelib support.
        # These should be "proper" errors, not just asserts, as they
        # can result from user errors like a requirement "foo @ URL"
        # when the project at URL has a name of "bar" in its metadata.
        assert (
            self._name is None or
            self._name == canonicalize_name(self._dist.project_name)
        ), "Name mismatch: {!r} vs {!r}".format(
            self._name, canonicalize_name(self._dist.project_name),
        )
        assert (
            self._version is None or
            self._version == self._dist.parsed_version
        ), "Version mismatch: {!r} vs {!r}".format(
            self._version, self._dist.parsed_version,
        )

    @property
    def dist(self):
        # type: () -> Distribution
        self._prepare()
        return self._dist

    def _get_requires_python_specifier(self):
        # type: () -> Optional[SpecifierSet]
        requires_python = get_requires_python(self.dist)
        if requires_python is None:
            return None
        try:
            spec = SpecifierSet(requires_python)
        except InvalidSpecifier as e:
            logger.warning(
                "Package %r has an invalid Requires-Python: %s", self.name, e,
            )
            return None
        return spec

    def get_dependencies(self):
        # type: () -> Sequence[Requirement]
        deps = [
            self._factory.make_requirement_from_spec(str(r), self._ireq)
            for r in self.dist.requires()
        ]
        python_dep = self._factory.make_requires_python_requirement(
            self._get_requires_python_specifier(),
        )
        if python_dep:
            deps.append(python_dep)
        return deps

    def get_install_requirement(self):
        # type: () -> Optional[InstallRequirement]
        self._prepare()
        return self._ireq


class LinkCandidate(_InstallRequirementBackedCandidate):
    def __init__(
        self,
        link,          # type: Link
        parent,        # type: InstallRequirement
        factory,       # type: Factory
        name=None,     # type: Optional[str]
        version=None,  # type: Optional[_BaseVersion]
    ):
        # type: (...) -> None
        super(LinkCandidate, self).__init__(
            link=link,
            ireq=make_install_req_from_link(link, parent),
            factory=factory,
            name=name,
            version=version,
        )

    def _prepare_abstract_distribution(self):
        # type: () -> AbstractDistribution
        return self._factory.preparer.prepare_linked_requirement(self._ireq)


class EditableCandidate(_InstallRequirementBackedCandidate):
    def __init__(
        self,
        link,          # type: Link
        parent,        # type: InstallRequirement
        factory,       # type: Factory
        name=None,     # type: Optional[str]
        version=None,  # type: Optional[_BaseVersion]
    ):
        # type: (...) -> None
        super(EditableCandidate, self).__init__(
            link=link,
            ireq=make_install_req_from_editable(link, parent),
            factory=factory,
            name=name,
            version=version,
        )

    def _prepare_abstract_distribution(self):
        # type: () -> AbstractDistribution
        return self._factory.preparer.prepare_editable_requirement(self._ireq)


class AlreadyInstalledCandidate(Candidate):
    def __init__(
        self,
        dist,  # type: Distribution
        parent,  # type: InstallRequirement
        factory,  # type: Factory
    ):
        # type: (...) -> None
        self.dist = dist
        self._ireq = make_install_req_from_dist(dist, parent)
        self._factory = factory

        # This is just logging some messages, so we can do it eagerly.
        # The returned dist would be exactly the same as self.dist because we
        # set satisfied_by in make_install_req_from_dist.
        # TODO: Supply reason based on force_reinstall and upgrade_strategy.
        skip_reason = "already satisfied"
        factory.preparer.prepare_installed_requirement(self._ireq, skip_reason)

    def __repr__(self):
        # type: () -> str
        return "{class_name}({distribution!r})".format(
            class_name=self.__class__.__name__,
            distribution=self.dist,
        )

    def __eq__(self, other):
        # type: (Any) -> bool
        if isinstance(other, self.__class__):
            return self.name == other.name and self.version == other.version
        return False

    # Needed for Python 2, which does not implement this by default
    def __ne__(self, other):
        # type: (Any) -> bool
        return not self.__eq__(other)

    @property
    def name(self):
        # type: () -> str
        return canonicalize_name(self.dist.project_name)

    @property
    def version(self):
        # type: () -> _BaseVersion
        return self.dist.parsed_version

    def get_dependencies(self):
        # type: () -> Sequence[Requirement]
        return [
            self._factory.make_requirement_from_spec(str(r), self._ireq)
            for r in self.dist.requires()
        ]

    def get_install_requirement(self):
        # type: () -> Optional[InstallRequirement]
        return None


class ExtrasCandidate(Candidate):
    """A candidate that has 'extras', indicating additional dependencies.

    Requirements can be for a project with dependencies, something like
    foo[extra].  The extras don't affect the project/version being installed
    directly, but indicate that we need additional dependencies. We model that
    by having an artificial ExtrasCandidate that wraps the "base" candidate.

    The ExtrasCandidate differs from the base in the following ways:

    1. It has a unique name, of the form foo[extra]. This causes the resolver
       to treat it as a separate node in the dependency graph.
    2. When we're getting the candidate's dependencies,
       a) We specify that we want the extra dependencies as well.
       b) We add a dependency on the base candidate (matching the name and
          version).  See below for why this is needed.
    3. We return None for the underlying InstallRequirement, as the base
       candidate will provide it, and we don't want to end up with duplicates.

    The dependency on the base candidate is needed so that the resolver can't
    decide that it should recommend foo[extra1] version 1.0 and foo[extra2]
    version 2.0. Having those candidates depend on foo=1.0 and foo=2.0
    respectively forces the resolver to recognise that this is a conflict.
    """
    def __init__(
        self,
        base,  # type: BaseCandidate
        extras,  # type: Set[str]
    ):
        # type: (...) -> None
        self.base = base
        self.extras = extras

    def __repr__(self):
        # type: () -> str
        return "{class_name}(base={base!r}, extras={extras!r})".format(
            class_name=self.__class__.__name__,
            base=self.base,
            extras=self.extras,
        )

    def __eq__(self, other):
        # type: (Any) -> bool
        if isinstance(other, self.__class__):
            return self.base == other.base and self.extras == other.extras
        return False

    # Needed for Python 2, which does not implement this by default
    def __ne__(self, other):
        # type: (Any) -> bool
        return not self.__eq__(other)

    @property
    def name(self):
        # type: () -> str
        """The normalised name of the project the candidate refers to"""
        return format_name(self.base.name, self.extras)

    @property
    def version(self):
        # type: () -> _BaseVersion
        return self.base.version

    def get_dependencies(self):
        # type: () -> Sequence[Requirement]
        factory = self.base._factory

        # The user may have specified extras that the candidate doesn't
        # support. We ignore any unsupported extras here.
        valid_extras = self.extras.intersection(self.base.dist.extras)
        invalid_extras = self.extras.difference(self.base.dist.extras)
        if invalid_extras:
            logger.warning(
                "Invalid extras specified in %s: %s",
                self.name,
                ','.join(sorted(invalid_extras))
            )

        deps = [
            factory.make_requirement_from_spec(str(r), self.base._ireq)
            for r in self.base.dist.requires(valid_extras)
        ]
        # Add a dependency on the exact base.
        # (See note 2b in the class docstring)
        spec = "{}=={}".format(self.base.name, self.base.version)
        deps.append(factory.make_requirement_from_spec(spec, self.base._ireq))
        return deps

    def get_install_requirement(self):
        # type: () -> Optional[InstallRequirement]
        # We don't return anything here, because we always
        # depend on the base candidate, and we'll get the
        # install requirement from that.
        return None


class RequiresPythonCandidate(Candidate):
    def __init__(self, py_version_info):
        # type: (Optional[Tuple[int, ...]]) -> None
        if py_version_info is not None:
            version_info = normalize_version_info(py_version_info)
        else:
            version_info = sys.version_info[:3]
        self._version = Version(".".join(str(c) for c in version_info))

    # We don't need to implement __eq__() and __ne__() since there is always
    # only one RequiresPythonCandidate in a resolution, i.e. the host Python.
    # The built-in object.__eq__() and object.__ne__() do exactly what we want.

    @property
    def name(self):
        # type: () -> str
        # Avoid conflicting with the PyPI package "Python".
        return "<Python fom Requires-Python>"

    @property
    def version(self):
        # type: () -> _BaseVersion
        return self._version

    def get_dependencies(self):
        # type: () -> Sequence[Requirement]
        return []

    def get_install_requirement(self):
        # type: () -> Optional[InstallRequirement]
        return None
