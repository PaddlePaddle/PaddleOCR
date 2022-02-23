"""
pasteurize: automatic conversion of Python 3 code to clean 2/3 code
===================================================================

``pasteurize`` attempts to convert existing Python 3 code into source-compatible
Python 2 and 3 code.

Use it like this on Python 3 code:

  $ pasteurize --verbose mypython3script.py

This removes any Py3-only syntax (e.g. new metaclasses) and adds these
import lines:

    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function
    from __future__ import unicode_literals
    from future import standard_library
    standard_library.install_hooks()
    from builtins import *

To write changes to the files, use the -w flag.

It also adds any other wrappers needed for Py2/3 compatibility.

Note that separate stages are not available (or needed) when converting from
Python 3 with ``pasteurize`` as they are when converting from Python 2 with
``futurize``.

The --all-imports option forces adding all ``__future__`` imports,
``builtins`` imports, and standard library aliases, even if they don't
seem necessary for the current state of each module. (This can simplify
testing, and can reduce the need to think about Py2 compatibility when editing
the code further.)

"""

from __future__ import (absolute_import, print_function, unicode_literals)

import sys
import logging
import optparse
from lib2to3.main import main, warn, StdoutRefactoringTool
from lib2to3 import refactor

from future import __version__
from libpasteurize.fixes import fix_names


def main(args=None):
    """Main program.

    Returns a suggested exit status (0, 1, 2).
    """
    # Set up option parser
    parser = optparse.OptionParser(usage="pasteurize [options] file|dir ...")
    parser.add_option("-V", "--version", action="store_true",
                      help="Report the version number of pasteurize")
    parser.add_option("-a", "--all-imports", action="store_true",
                      help="Adds all __future__ and future imports to each module")
    parser.add_option("-f", "--fix", action="append", default=[],
                      help="Each FIX specifies a transformation; default: all")
    parser.add_option("-j", "--processes", action="store", default=1,
                      type="int", help="Run 2to3 concurrently")
    parser.add_option("-x", "--nofix", action="append", default=[],
                      help="Prevent a fixer from being run.")
    parser.add_option("-l", "--list-fixes", action="store_true",
                      help="List available transformations")
    # parser.add_option("-p", "--print-function", action="store_true",
    #                   help="Modify the grammar so that print() is a function")
    parser.add_option("-v", "--verbose", action="store_true",
                      help="More verbose logging")
    parser.add_option("--no-diffs", action="store_true",
                      help="Don't show diffs of the refactoring")
    parser.add_option("-w", "--write", action="store_true",
                      help="Write back modified files")
    parser.add_option("-n", "--nobackups", action="store_true", default=False,
                      help="Don't write backups for modified files.")

    # Parse command line arguments
    refactor_stdin = False
    flags = {}
    options, args = parser.parse_args(args)
    fixer_pkg = 'libpasteurize.fixes'
    avail_fixes = fix_names
    flags["print_function"] = True

    if not options.write and options.no_diffs:
        warn("not writing files and not printing diffs; that's not very useful")
    if not options.write and options.nobackups:
        parser.error("Can't use -n without -w")
    if options.version:
        print(__version__)
        return 0
    if options.list_fixes:
        print("Available transformations for the -f/--fix option:")
        for fixname in sorted(avail_fixes):
            print(fixname)
        if not args:
            return 0
    if not args:
        print("At least one file or directory argument required.",
              file=sys.stderr)
        print("Use --help to show usage.", file=sys.stderr)
        return 2
    if "-" in args:
        refactor_stdin = True
        if options.write:
            print("Can't write to stdin.", file=sys.stderr)
            return 2

    # Set up logging handler
    level = logging.DEBUG if options.verbose else logging.INFO
    logging.basicConfig(format='%(name)s: %(message)s', level=level)

    unwanted_fixes = set()
    for fix in options.nofix:
        if ".fix_" in fix:
            unwanted_fixes.add(fix)
        else:
            # Infer the full module name for the fixer.
            # First ensure that no names clash (e.g.
            # lib2to3.fixes.fix_blah and libfuturize.fixes.fix_blah):
            found = [f for f in avail_fixes
                     if f.endswith('fix_{0}'.format(fix))]
            if len(found) > 1:
                print("Ambiguous fixer name. Choose a fully qualified "
                      "module name instead from these:\n" +
                      "\n".join("  " + myf for myf in found),
                      file=sys.stderr)
                return 2
            elif len(found) == 0:
                print("Unknown fixer. Use --list-fixes or -l for a list.",
                      file=sys.stderr)
                return 2
            unwanted_fixes.add(found[0])

    extra_fixes = set()
    if options.all_imports:
        prefix = 'libpasteurize.fixes.'
        extra_fixes.add(prefix + 'fix_add_all__future__imports')
        extra_fixes.add(prefix + 'fix_add_future_standard_library_import')
        extra_fixes.add(prefix + 'fix_add_all_future_builtins')

    explicit = set()
    if options.fix:
        all_present = False
        for fix in options.fix:
            if fix == 'all':
                all_present = True
            else:
                if ".fix_" in fix:
                    explicit.add(fix)
                else:
                    # Infer the full module name for the fixer.
                    # First ensure that no names clash (e.g.
                    # lib2to3.fixes.fix_blah and libpasteurize.fixes.fix_blah):
                    found = [f for f in avail_fixes
                             if f.endswith('fix_{0}'.format(fix))]
                    if len(found) > 1:
                        print("Ambiguous fixer name. Choose a fully qualified "
                              "module name instead from these:\n" +
                              "\n".join("  " + myf for myf in found),
                              file=sys.stderr)
                        return 2
                    elif len(found) == 0:
                        print("Unknown fixer. Use --list-fixes or -l for a list.",
                              file=sys.stderr)
                        return 2
                    explicit.add(found[0])
        if len(explicit & unwanted_fixes) > 0:
            print("Conflicting usage: the following fixers have been "
                  "simultaneously requested and disallowed:\n" +
                  "\n".join("  " + myf for myf in (explicit & unwanted_fixes)),
                  file=sys.stderr)
            return 2
        requested = avail_fixes.union(explicit) if all_present else explicit
    else:
        requested = avail_fixes.union(explicit)

    fixer_names = requested | extra_fixes - unwanted_fixes

    # Initialize the refactoring tool
    rt = StdoutRefactoringTool(sorted(fixer_names), flags, set(),
                               options.nobackups, not options.no_diffs)

    # Refactor all files and directories passed as arguments
    if not rt.errors:
        if refactor_stdin:
            rt.refactor_stdin()
        else:
            try:
                rt.refactor(args, options.write, None,
                            options.processes)
            except refactor.MultiprocessingUnsupported:
                assert options.processes > 1
                print("Sorry, -j isn't " \
                      "supported on this platform.", file=sys.stderr)
                return 1
        rt.summarize()

    # Return error status (0 if rt.errors is zero)
    return int(bool(rt.errors))
