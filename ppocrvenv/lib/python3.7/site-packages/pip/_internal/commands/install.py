# The following comment should be removed at some point in the future.
# It's included for now because without it InstallCommand.run() has a
# couple errors where we have to know req.name is str rather than
# Optional[str] for the InstallRequirement req.
# mypy: strict-optional=False
# mypy: disallow-untyped-defs=False

from __future__ import absolute_import

import errno
import logging
import operator
import os
import shutil
import site
from optparse import SUPPRESS_HELP

from pip._vendor import pkg_resources
from pip._vendor.packaging.utils import canonicalize_name

from pip._internal.cache import WheelCache
from pip._internal.cli import cmdoptions
from pip._internal.cli.cmdoptions import make_target_python
from pip._internal.cli.req_command import RequirementCommand, with_cleanup
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.exceptions import CommandError, InstallationError
from pip._internal.locations import distutils_scheme
from pip._internal.operations.check import check_install_conflicts
from pip._internal.req import install_given_reqs
from pip._internal.req.req_tracker import get_requirement_tracker
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.distutils_args import parse_distutils_args
from pip._internal.utils.filesystem import test_writable_dir
from pip._internal.utils.misc import (
    ensure_dir,
    get_installed_version,
    protect_pip_from_modification_on_windows,
    write_output,
)
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.typing import MYPY_CHECK_RUNNING
from pip._internal.utils.virtualenv import virtualenv_no_global
from pip._internal.wheel_builder import build, should_build_for_install_command

if MYPY_CHECK_RUNNING:
    from optparse import Values
    from typing import Any, Iterable, List, Optional

    from pip._internal.models.format_control import FormatControl
    from pip._internal.req.req_install import InstallRequirement
    from pip._internal.wheel_builder import BinaryAllowedPredicate


logger = logging.getLogger(__name__)


def get_check_binary_allowed(format_control):
    # type: (FormatControl) -> BinaryAllowedPredicate
    def check_binary_allowed(req):
        # type: (InstallRequirement) -> bool
        if req.use_pep517:
            return True
        canonical_name = canonicalize_name(req.name)
        allowed_formats = format_control.get_allowed_formats(canonical_name)
        return "binary" in allowed_formats

    return check_binary_allowed


class InstallCommand(RequirementCommand):
    """
    Install packages from:

    - PyPI (and other indexes) using requirement specifiers.
    - VCS project urls.
    - Local project directories.
    - Local or remote source archives.

    pip also supports installing from "requirements files", which provide
    an easy way to specify a whole environment to be installed.
    """

    usage = """
      %prog [options] <requirement specifier> [package-index-options] ...
      %prog [options] -r <requirements file> [package-index-options] ...
      %prog [options] [-e] <vcs project url> ...
      %prog [options] [-e] <local project path> ...
      %prog [options] <archive url/path> ..."""

    def __init__(self, *args, **kw):
        super(InstallCommand, self).__init__(*args, **kw)

        cmd_opts = self.cmd_opts

        cmd_opts.add_option(cmdoptions.requirements())
        cmd_opts.add_option(cmdoptions.constraints())
        cmd_opts.add_option(cmdoptions.no_deps())
        cmd_opts.add_option(cmdoptions.pre())

        cmd_opts.add_option(cmdoptions.editable())
        cmd_opts.add_option(
            '-t', '--target',
            dest='target_dir',
            metavar='dir',
            default=None,
            help='Install packages into <dir>. '
                 'By default this will not replace existing files/folders in '
                 '<dir>. Use --upgrade to replace existing packages in <dir> '
                 'with new versions.'
        )
        cmdoptions.add_target_python_options(cmd_opts)

        cmd_opts.add_option(
            '--user',
            dest='use_user_site',
            action='store_true',
            help="Install to the Python user install directory for your "
                 "platform. Typically ~/.local/, or %APPDATA%\\Python on "
                 "Windows. (See the Python documentation for site.USER_BASE "
                 "for full details.)")
        cmd_opts.add_option(
            '--no-user',
            dest='use_user_site',
            action='store_false',
            help=SUPPRESS_HELP)
        cmd_opts.add_option(
            '--root',
            dest='root_path',
            metavar='dir',
            default=None,
            help="Install everything relative to this alternate root "
                 "directory.")
        cmd_opts.add_option(
            '--prefix',
            dest='prefix_path',
            metavar='dir',
            default=None,
            help="Installation prefix where lib, bin and other top-level "
                 "folders are placed")

        cmd_opts.add_option(cmdoptions.build_dir())

        cmd_opts.add_option(cmdoptions.src())

        cmd_opts.add_option(
            '-U', '--upgrade',
            dest='upgrade',
            action='store_true',
            help='Upgrade all specified packages to the newest available '
                 'version. The handling of dependencies depends on the '
                 'upgrade-strategy used.'
        )

        cmd_opts.add_option(
            '--upgrade-strategy',
            dest='upgrade_strategy',
            default='only-if-needed',
            choices=['only-if-needed', 'eager'],
            help='Determines how dependency upgrading should be handled '
                 '[default: %default]. '
                 '"eager" - dependencies are upgraded regardless of '
                 'whether the currently installed version satisfies the '
                 'requirements of the upgraded package(s). '
                 '"only-if-needed" -  are upgraded only when they do not '
                 'satisfy the requirements of the upgraded package(s).'
        )

        cmd_opts.add_option(
            '--force-reinstall',
            dest='force_reinstall',
            action='store_true',
            help='Reinstall all packages even if they are already '
                 'up-to-date.')

        cmd_opts.add_option(
            '-I', '--ignore-installed',
            dest='ignore_installed',
            action='store_true',
            help='Ignore the installed packages, overwriting them. '
                 'This can break your system if the existing package '
                 'is of a different version or was installed '
                 'with a different package manager!'
        )

        cmd_opts.add_option(cmdoptions.ignore_requires_python())
        cmd_opts.add_option(cmdoptions.no_build_isolation())
        cmd_opts.add_option(cmdoptions.use_pep517())
        cmd_opts.add_option(cmdoptions.no_use_pep517())

        cmd_opts.add_option(cmdoptions.install_options())
        cmd_opts.add_option(cmdoptions.global_options())

        cmd_opts.add_option(
            "--compile",
            action="store_true",
            dest="compile",
            default=True,
            help="Compile Python source files to bytecode",
        )

        cmd_opts.add_option(
            "--no-compile",
            action="store_false",
            dest="compile",
            help="Do not compile Python source files to bytecode",
        )

        cmd_opts.add_option(
            "--no-warn-script-location",
            action="store_false",
            dest="warn_script_location",
            default=True,
            help="Do not warn when installing scripts outside PATH",
        )
        cmd_opts.add_option(
            "--no-warn-conflicts",
            action="store_false",
            dest="warn_about_conflicts",
            default=True,
            help="Do not warn about broken dependencies",
        )

        cmd_opts.add_option(cmdoptions.no_binary())
        cmd_opts.add_option(cmdoptions.only_binary())
        cmd_opts.add_option(cmdoptions.prefer_binary())
        cmd_opts.add_option(cmdoptions.require_hashes())
        cmd_opts.add_option(cmdoptions.progress_bar())

        index_opts = cmdoptions.make_option_group(
            cmdoptions.index_group,
            self.parser,
        )

        self.parser.insert_option_group(0, index_opts)
        self.parser.insert_option_group(0, cmd_opts)

    @with_cleanup
    def run(self, options, args):
        # type: (Values, List[Any]) -> int
        if options.use_user_site and options.target_dir is not None:
            raise CommandError("Can not combine '--user' and '--target'")

        cmdoptions.check_install_build_global(options)
        upgrade_strategy = "to-satisfy-only"
        if options.upgrade:
            upgrade_strategy = options.upgrade_strategy

        cmdoptions.check_dist_restriction(options, check_target=True)

        install_options = options.install_options or []

        options.use_user_site = decide_user_install(
            options.use_user_site,
            prefix_path=options.prefix_path,
            target_dir=options.target_dir,
            root_path=options.root_path,
            isolated_mode=options.isolated_mode,
        )

        target_temp_dir = None  # type: Optional[TempDirectory]
        target_temp_dir_path = None  # type: Optional[str]
        if options.target_dir:
            options.ignore_installed = True
            options.target_dir = os.path.abspath(options.target_dir)
            if (os.path.exists(options.target_dir) and not
                    os.path.isdir(options.target_dir)):
                raise CommandError(
                    "Target path exists but is not a directory, will not "
                    "continue."
                )

            # Create a target directory for using with the target option
            target_temp_dir = TempDirectory(kind="target")
            target_temp_dir_path = target_temp_dir.path

        global_options = options.global_options or []

        session = self.get_default_session(options)

        target_python = make_target_python(options)
        finder = self._build_package_finder(
            options=options,
            session=session,
            target_python=target_python,
            ignore_requires_python=options.ignore_requires_python,
        )
        build_delete = (not (options.no_clean or options.build_dir))
        wheel_cache = WheelCache(options.cache_dir, options.format_control)

        req_tracker = self.enter_context(get_requirement_tracker())

        directory = TempDirectory(
            options.build_dir,
            delete=build_delete,
            kind="install",
            globally_managed=True,
        )

        try:
            reqs = self.get_requirements(
                args, options, finder, session,
                check_supported_wheels=not options.target_dir,
            )

            warn_deprecated_install_options(
                reqs, options.install_options
            )

            preparer = self.make_requirement_preparer(
                temp_build_dir=directory,
                options=options,
                req_tracker=req_tracker,
                session=session,
                finder=finder,
                use_user_site=options.use_user_site,
            )
            resolver = self.make_resolver(
                preparer=preparer,
                finder=finder,
                options=options,
                wheel_cache=wheel_cache,
                use_user_site=options.use_user_site,
                ignore_installed=options.ignore_installed,
                ignore_requires_python=options.ignore_requires_python,
                force_reinstall=options.force_reinstall,
                upgrade_strategy=upgrade_strategy,
                use_pep517=options.use_pep517,
            )

            self.trace_basic_info(finder)

            requirement_set = resolver.resolve(
                reqs, check_supported_wheels=not options.target_dir
            )

            try:
                pip_req = requirement_set.get_requirement("pip")
            except KeyError:
                modifying_pip = None
            else:
                # If we're not replacing an already installed pip,
                # we're not modifying it.
                modifying_pip = pip_req.satisfied_by is None
            protect_pip_from_modification_on_windows(
                modifying_pip=modifying_pip
            )

            check_binary_allowed = get_check_binary_allowed(
                finder.format_control
            )

            reqs_to_build = [
                r for r in requirement_set.requirements.values()
                if should_build_for_install_command(
                    r, check_binary_allowed
                )
            ]

            _, build_failures = build(
                reqs_to_build,
                wheel_cache=wheel_cache,
                build_options=[],
                global_options=[],
            )

            # If we're using PEP 517, we cannot do a direct install
            # so we fail here.
            # We don't care about failures building legacy
            # requirements, as we'll fall through to a direct
            # install for those.
            pep517_build_failures = [
                r for r in build_failures if r.use_pep517
            ]
            if pep517_build_failures:
                raise InstallationError(
                    "Could not build wheels for {} which use"
                    " PEP 517 and cannot be installed directly".format(
                        ", ".join(r.name for r in pep517_build_failures)))

            to_install = resolver.get_installation_order(
                requirement_set
            )

            # Consistency Checking of the package set we're installing.
            should_warn_about_conflicts = (
                not options.ignore_dependencies and
                options.warn_about_conflicts
            )
            if should_warn_about_conflicts:
                self._warn_about_conflicts(to_install)

            # Don't warn about script install locations if
            # --target has been specified
            warn_script_location = options.warn_script_location
            if options.target_dir:
                warn_script_location = False

            installed = install_given_reqs(
                to_install,
                install_options,
                global_options,
                root=options.root_path,
                home=target_temp_dir_path,
                prefix=options.prefix_path,
                pycompile=options.compile,
                warn_script_location=warn_script_location,
                use_user_site=options.use_user_site,
            )

            lib_locations = get_lib_location_guesses(
                user=options.use_user_site,
                home=target_temp_dir_path,
                root=options.root_path,
                prefix=options.prefix_path,
                isolated=options.isolated_mode,
            )
            working_set = pkg_resources.WorkingSet(lib_locations)

            installed.sort(key=operator.attrgetter('name'))
            items = []
            for result in installed:
                item = result.name
                try:
                    installed_version = get_installed_version(
                        result.name, working_set=working_set
                    )
                    if installed_version:
                        item += '-' + installed_version
                except Exception:
                    pass
                items.append(item)
            installed_desc = ' '.join(items)
            if installed_desc:
                write_output(
                    'Successfully installed %s', installed_desc,
                )
        except EnvironmentError as error:
            show_traceback = (self.verbosity >= 1)

            message = create_env_error_message(
                error, show_traceback, options.use_user_site,
            )
            logger.error(message, exc_info=show_traceback)

            return ERROR

        if options.target_dir:
            self._handle_target_dir(
                options.target_dir, target_temp_dir, options.upgrade
            )

        return SUCCESS

    def _handle_target_dir(self, target_dir, target_temp_dir, upgrade):
        ensure_dir(target_dir)

        # Checking both purelib and platlib directories for installed
        # packages to be moved to target directory
        lib_dir_list = []

        with target_temp_dir:
            # Checking both purelib and platlib directories for installed
            # packages to be moved to target directory
            scheme = distutils_scheme('', home=target_temp_dir.path)
            purelib_dir = scheme['purelib']
            platlib_dir = scheme['platlib']
            data_dir = scheme['data']

            if os.path.exists(purelib_dir):
                lib_dir_list.append(purelib_dir)
            if os.path.exists(platlib_dir) and platlib_dir != purelib_dir:
                lib_dir_list.append(platlib_dir)
            if os.path.exists(data_dir):
                lib_dir_list.append(data_dir)

            for lib_dir in lib_dir_list:
                for item in os.listdir(lib_dir):
                    if lib_dir == data_dir:
                        ddir = os.path.join(data_dir, item)
                        if any(s.startswith(ddir) for s in lib_dir_list[:-1]):
                            continue
                    target_item_dir = os.path.join(target_dir, item)
                    if os.path.exists(target_item_dir):
                        if not upgrade:
                            logger.warning(
                                'Target directory %s already exists. Specify '
                                '--upgrade to force replacement.',
                                target_item_dir
                            )
                            continue
                        if os.path.islink(target_item_dir):
                            logger.warning(
                                'Target directory %s already exists and is '
                                'a link. pip will not automatically replace '
                                'links, please remove if replacement is '
                                'desired.',
                                target_item_dir
                            )
                            continue
                        if os.path.isdir(target_item_dir):
                            shutil.rmtree(target_item_dir)
                        else:
                            os.remove(target_item_dir)

                    shutil.move(
                        os.path.join(lib_dir, item),
                        target_item_dir
                    )

    def _warn_about_conflicts(self, to_install):
        try:
            package_set, _dep_info = check_install_conflicts(to_install)
        except Exception:
            logger.error("Error checking for conflicts.", exc_info=True)
            return
        missing, conflicting = _dep_info

        # NOTE: There is some duplication here from pip check
        for project_name in missing:
            version = package_set[project_name][0]
            for dependency in missing[project_name]:
                logger.critical(
                    "%s %s requires %s, which is not installed.",
                    project_name, version, dependency[1],
                )

        for project_name in conflicting:
            version = package_set[project_name][0]
            for dep_name, dep_version, req in conflicting[project_name]:
                logger.critical(
                    "%s %s has requirement %s, but you'll have %s %s which is "
                    "incompatible.",
                    project_name, version, req, dep_name, dep_version,
                )


def get_lib_location_guesses(*args, **kwargs):
    scheme = distutils_scheme('', *args, **kwargs)
    return [scheme['purelib'], scheme['platlib']]


def site_packages_writable(**kwargs):
    return all(
        test_writable_dir(d) for d in set(get_lib_location_guesses(**kwargs))
    )


def decide_user_install(
    use_user_site,  # type: Optional[bool]
    prefix_path=None,  # type: Optional[str]
    target_dir=None,  # type: Optional[str]
    root_path=None,  # type: Optional[str]
    isolated_mode=False,  # type: bool
):
    # type: (...) -> bool
    """Determine whether to do a user install based on the input options.

    If use_user_site is False, no additional checks are done.
    If use_user_site is True, it is checked for compatibility with other
    options.
    If use_user_site is None, the default behaviour depends on the environment,
    which is provided by the other arguments.
    """
    # In some cases (config from tox), use_user_site can be set to an integer
    # rather than a bool, which 'use_user_site is False' wouldn't catch.
    if (use_user_site is not None) and (not use_user_site):
        logger.debug("Non-user install by explicit request")
        return False

    if use_user_site:
        if prefix_path:
            raise CommandError(
                "Can not combine '--user' and '--prefix' as they imply "
                "different installation locations"
            )
        if virtualenv_no_global():
            raise InstallationError(
                "Can not perform a '--user' install. User site-packages "
                "are not visible in this virtualenv."
            )
        logger.debug("User install by explicit request")
        return True

    # If we are here, user installs have not been explicitly requested/avoided
    assert use_user_site is None

    # user install incompatible with --prefix/--target
    if prefix_path or target_dir:
        logger.debug("Non-user install due to --prefix or --target option")
        return False

    # If user installs are not enabled, choose a non-user install
    if not site.ENABLE_USER_SITE:
        logger.debug("Non-user install because user site-packages disabled")
        return False

    # If we have permission for a non-user install, do that,
    # otherwise do a user install.
    if site_packages_writable(root=root_path, isolated=isolated_mode):
        logger.debug("Non-user install because site-packages writeable")
        return False

    logger.info("Defaulting to user installation because normal site-packages "
                "is not writeable")
    return True


def warn_deprecated_install_options(requirements, options):
    # type: (List[InstallRequirement], Optional[List[str]]) -> None
    """If any location-changing --install-option arguments were passed for
    requirements or on the command-line, then show a deprecation warning.
    """
    def format_options(option_names):
        # type: (Iterable[str]) -> List[str]
        return ["--{}".format(name.replace("_", "-")) for name in option_names]

    offenders = []

    for requirement in requirements:
        install_options = requirement.install_options
        location_options = parse_distutils_args(install_options)
        if location_options:
            offenders.append(
                "{!r} from {}".format(
                    format_options(location_options.keys()), requirement
                )
            )

    if options:
        location_options = parse_distutils_args(options)
        if location_options:
            offenders.append(
                "{!r} from command line".format(
                    format_options(location_options.keys())
                )
            )

    if not offenders:
        return

    deprecated(
        reason=(
            "Location-changing options found in --install-option: {}. "
            "This configuration may cause unexpected behavior and is "
            "unsupported.".format(
                "; ".join(offenders)
            )
        ),
        replacement=(
            "using pip-level options like --user, --prefix, --root, and "
            "--target"
        ),
        gone_in="20.2",
        issue=7309,
    )


def create_env_error_message(error, show_traceback, using_user_site):
    """Format an error message for an EnvironmentError

    It may occur anytime during the execution of the install command.
    """
    parts = []

    # Mention the error if we are not going to show a traceback
    parts.append("Could not install packages due to an EnvironmentError")
    if not show_traceback:
        parts.append(": ")
        parts.append(str(error))
    else:
        parts.append(".")

    # Spilt the error indication from a helper message (if any)
    parts[-1] += "\n"

    # Suggest useful actions to the user:
    #  (1) using user site-packages or (2) verifying the permissions
    if error.errno == errno.EACCES:
        user_option_part = "Consider using the `--user` option"
        permissions_part = "Check the permissions"

        if not using_user_site:
            parts.extend([
                user_option_part, " or ",
                permissions_part.lower(),
            ])
        else:
            parts.append(permissions_part)
        parts.append(".\n")

    return "".join(parts).strip() + "\n"
