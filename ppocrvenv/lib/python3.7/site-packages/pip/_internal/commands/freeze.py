# The following comment should be removed at some point in the future.
# mypy: disallow-untyped-defs=False

from __future__ import absolute_import

import sys

from pip._internal.cache import WheelCache
from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.models.format_control import FormatControl
from pip._internal.operations.freeze import freeze
from pip._internal.utils.compat import stdlib_pkgs

DEV_PKGS = {'pip', 'setuptools', 'distribute', 'wheel'}


class FreezeCommand(Command):
    """
    Output installed packages in requirements format.

    packages are listed in a case-insensitive sorted order.
    """

    usage = """
      %prog [options]"""
    log_streams = ("ext://sys.stderr", "ext://sys.stderr")

    def __init__(self, *args, **kw):
        super(FreezeCommand, self).__init__(*args, **kw)

        self.cmd_opts.add_option(
            '-r', '--requirement',
            dest='requirements',
            action='append',
            default=[],
            metavar='file',
            help="Use the order in the given requirements file and its "
                 "comments when generating output. This option can be "
                 "used multiple times.")
        self.cmd_opts.add_option(
            '-f', '--find-links',
            dest='find_links',
            action='append',
            default=[],
            metavar='URL',
            help='URL for finding packages, which will be added to the '
                 'output.')
        self.cmd_opts.add_option(
            '-l', '--local',
            dest='local',
            action='store_true',
            default=False,
            help='If in a virtualenv that has global access, do not output '
                 'globally-installed packages.')
        self.cmd_opts.add_option(
            '--user',
            dest='user',
            action='store_true',
            default=False,
            help='Only output packages installed in user-site.')
        self.cmd_opts.add_option(cmdoptions.list_path())
        self.cmd_opts.add_option(
            '--all',
            dest='freeze_all',
            action='store_true',
            help='Do not skip these packages in the output:'
                 ' {}'.format(', '.join(DEV_PKGS)))
        self.cmd_opts.add_option(
            '--exclude-editable',
            dest='exclude_editable',
            action='store_true',
            help='Exclude editable package from output.')

        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options, args):
        format_control = FormatControl(set(), set())
        wheel_cache = WheelCache(options.cache_dir, format_control)
        skip = set(stdlib_pkgs)
        if not options.freeze_all:
            skip.update(DEV_PKGS)

        cmdoptions.check_list_path_option(options)

        freeze_kwargs = dict(
            requirement=options.requirements,
            find_links=options.find_links,
            local_only=options.local,
            user_only=options.user,
            paths=options.path,
            isolated=options.isolated_mode,
            wheel_cache=wheel_cache,
            skip=skip,
            exclude_editable=options.exclude_editable,
        )

        for line in freeze(**freeze_kwargs):
            sys.stdout.write(line + '\n')
