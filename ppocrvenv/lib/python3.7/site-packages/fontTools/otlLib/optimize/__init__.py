from argparse import RawTextHelpFormatter
from textwrap import dedent

from fontTools.ttLib import TTFont
from fontTools.otlLib.optimize.gpos import compact, GPOS_COMPACT_MODE_DEFAULT

def main(args=None):
    """Optimize the layout tables of an existing font."""
    from argparse import ArgumentParser
    from fontTools import configLogger

    parser = ArgumentParser(prog="otlLib.optimize", description=main.__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("font")
    parser.add_argument(
        "-o", metavar="OUTPUTFILE", dest="outfile", default=None, help="output file"
    )
    parser.add_argument(
        "--gpos-compact-mode",
        help=dedent(
            f"""\
            GPOS Lookup type 2 (PairPos) compaction mode:
                0 = do not attempt to compact PairPos lookups;
                1 to 8 = create at most 1 to 8 new subtables for each existing
                    subtable, provided that it would yield a 50%% file size saving;
                9 = create as many new subtables as needed to yield a file size saving.
            Default: {GPOS_COMPACT_MODE_DEFAULT}.

            This compaction aims to save file size, by splitting large class
            kerning subtables (Format 2) that contain many zero values into
            smaller and denser subtables. It's a trade-off between the overhead
            of several subtables versus the sparseness of one big subtable.

            See the pull request: https://github.com/fonttools/fonttools/pull/2326
            """
        ),
        default=int(GPOS_COMPACT_MODE_DEFAULT),
        choices=list(range(10)),
        type=int,
    )
    logging_group = parser.add_mutually_exclusive_group(required=False)
    logging_group.add_argument(
        "-v", "--verbose", action="store_true", help="Run more verbosely."
    )
    logging_group.add_argument(
        "-q", "--quiet", action="store_true", help="Turn verbosity off."
    )
    options = parser.parse_args(args)

    configLogger(
        level=("DEBUG" if options.verbose else "ERROR" if options.quiet else "INFO")
    )

    font = TTFont(options.font)
    # TODO: switch everything to have type(mode) = int when using the Config class
    compact(font, str(options.gpos_compact_mode))
    font.save(options.outfile or options.font)



if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        sys.exit(main())
    import doctest

    sys.exit(doctest.testmod().failed)

