# The following comment should be removed at some point in the future.
# mypy: strict-optional=False

from __future__ import absolute_import

import logging

from pip._internal.utils.logging import indent_log
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

from .req_file import parse_requirements
from .req_install import InstallRequirement
from .req_set import RequirementSet

if MYPY_CHECK_RUNNING:
    from typing import Any, List, Sequence

__all__ = [
    "RequirementSet", "InstallRequirement",
    "parse_requirements", "install_given_reqs",
]

logger = logging.getLogger(__name__)


class InstallationResult(object):
    def __init__(self, name):
        # type: (str) -> None
        self.name = name

    def __repr__(self):
        # type: () -> str
        return "InstallationResult(name={!r})".format(self.name)


def install_given_reqs(
    to_install,  # type: List[InstallRequirement]
    install_options,  # type: List[str]
    global_options=(),  # type: Sequence[str]
    *args,  # type: Any
    **kwargs  # type: Any
):
    # type: (...) -> List[InstallationResult]
    """
    Install everything in the given list.

    (to be called after having downloaded and unpacked the packages)
    """

    if to_install:
        logger.info(
            'Installing collected packages: %s',
            ', '.join([req.name for req in to_install]),
        )

    installed = []

    with indent_log():
        for requirement in to_install:
            if requirement.should_reinstall:
                logger.info('Attempting uninstall: %s', requirement.name)
                with indent_log():
                    uninstalled_pathset = requirement.uninstall(
                        auto_confirm=True
                    )
            try:
                requirement.install(
                    install_options,
                    global_options,
                    *args,
                    **kwargs
                )
            except Exception:
                should_rollback = (
                    requirement.should_reinstall and
                    not requirement.install_succeeded
                )
                # if install did not succeed, rollback previous uninstall
                if should_rollback:
                    uninstalled_pathset.rollback()
                raise
            else:
                should_commit = (
                    requirement.should_reinstall and
                    requirement.install_succeeded
                )
                if should_commit:
                    uninstalled_pathset.commit()

            installed.append(InstallationResult(requirement.name))

    return installed
