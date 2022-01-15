# The following comment should be removed at some point in the future.
# mypy: disallow-untyped-defs=False

from __future__ import absolute_import

import logging
import os

from pip._internal.cli import cmdoptions
from pip._internal.cli.cmdoptions import make_target_python
from pip._internal.cli.req_command import RequirementCommand, with_cleanup
from pip._internal.req.req_tracker import get_requirement_tracker
from pip._internal.utils.misc import ensure_dir, normalize_path, write_output
from pip._internal.utils.temp_dir import TempDirectory

logger = logging.getLogger(__name__)


class DownloadCommand(RequirementCommand):
    """
    Download packages from:

    - PyPI (and other indexes) using requirement specifiers.
    - VCS project urls.
    - Local project directories.
    - Local or remote source archives.

    pip also supports downloading from "requirements files", which provide
    an easy way to specify a whole environment to be downloaded.
    """

    usage = """
      %prog [options] <requirement specifier> [package-index-options] ...
      %prog [options] -r <requirements file> [package-index-options] ...
      %prog [options] <vcs project url> ...
      %prog [options] <local project path> ...
      %prog [options] <archive url/path> ..."""

    def __init__(self, *args, **kw):
        super(DownloadCommand, self).__init__(*args, **kw)

        cmd_opts = self.cmd_opts

        cmd_opts.add_option(cmdoptions.constraints())
        cmd_opts.add_option(cmdoptions.requirements())
        cmd_opts.add_option(cmdoptions.build_dir())
        cmd_opts.add_option(cmdoptions.no_deps())
        cmd_opts.add_option(cmdoptions.global_options())
        cmd_opts.add_option(cmdoptions.no_binary())
        cmd_opts.add_option(cmdoptions.only_binary())
        cmd_opts.add_option(cmdoptions.prefer_binary())
        cmd_opts.add_option(cmdoptions.src())
        cmd_opts.add_option(cmdoptions.pre())
        cmd_opts.add_option(cmdoptions.require_hashes())
        cmd_opts.add_option(cmdoptions.progress_bar())
        cmd_opts.add_option(cmdoptions.no_build_isolation())
        cmd_opts.add_option(cmdoptions.use_pep517())
        cmd_opts.add_option(cmdoptions.no_use_pep517())

        cmd_opts.add_option(
            '-d', '--dest', '--destination-dir', '--destination-directory',
            dest='download_dir',
            metavar='dir',
            default=os.curdir,
            help=("Download packages into <dir>."),
        )

        cmdoptions.add_target_python_options(cmd_opts)

        index_opts = cmdoptions.make_option_group(
            cmdoptions.index_group,
            self.parser,
        )

        self.parser.insert_option_group(0, index_opts)
        self.parser.insert_option_group(0, cmd_opts)

    @with_cleanup
    def run(self, options, args):
        options.ignore_installed = True
        # editable doesn't really make sense for `pip download`, but the bowels
        # of the RequirementSet code require that property.
        options.editables = []

        cmdoptions.check_dist_restriction(options)

        options.download_dir = normalize_path(options.download_dir)

        ensure_dir(options.download_dir)

        session = self.get_default_session(options)

        target_python = make_target_python(options)
        finder = self._build_package_finder(
            options=options,
            session=session,
            target_python=target_python,
        )
        build_delete = (not (options.no_clean or options.build_dir))

        req_tracker = self.enter_context(get_requirement_tracker())

        directory = TempDirectory(
            options.build_dir,
            delete=build_delete,
            kind="download",
            globally_managed=True,
        )

        reqs = self.get_requirements(args, options, finder, session)

        preparer = self.make_requirement_preparer(
            temp_build_dir=directory,
            options=options,
            req_tracker=req_tracker,
            session=session,
            finder=finder,
            download_dir=options.download_dir,
            use_user_site=False,
        )

        resolver = self.make_resolver(
            preparer=preparer,
            finder=finder,
            options=options,
            py_version_info=options.python_version,
        )

        self.trace_basic_info(finder)

        requirement_set = resolver.resolve(
            reqs, check_supported_wheels=True
        )

        downloaded = ' '.join([
            req.name for req in requirement_set.requirements.values()
            if req.successfully_downloaded
        ])
        if downloaded:
            write_output('Successfully downloaded %s', downloaded)

        return requirement_set
