"""utility script to parse given filenames or string
"""

import cssutils
import logging
import optparse
import sys


def main(args=None):
    """
    Parses given filename(s) or string or URL (using optional encoding) and
    prints the parsed style sheet to stdout.

    Redirect stdout to save CSS. Redirect stderr to save parser log infos.
    """
    usage = """usage: %prog [options] filename1.css [filename2.css ...]
        [>filename_combined.css] [2>parserinfo.log] """
    p = optparse.OptionParser(usage=usage)
    p.add_option(
        '-s', '--string', action='store_true', dest='string', help='parse given string'
    )
    p.add_option('-u', '--url', action='store', dest='url', help='parse given url')
    p.add_option(
        '-e',
        '--encoding',
        action='store',
        dest='encoding',
        help='encoding of the file or override encoding found',
    )
    p.add_option(
        '-m',
        '--minify',
        action='store_true',
        dest='minify',
        help='minify parsed CSS',
        default=False,
    )
    p.add_option(
        '-d',
        '--debug',
        action='store_true',
        dest='debug',
        help='activate debugging output',
    )

    (options, params) = p.parse_args(args)

    if not params and not options.url:
        p.error("no filename given")

    if options.debug:
        p = cssutils.CSSParser(loglevel=logging.DEBUG)
    else:
        p = cssutils.CSSParser()

    if options.minify:
        cssutils.ser.prefs.useMinified()

    if options.string:
        sheet = p.parseString(''.join(params), encoding=options.encoding)
        print(sheet.cssText)
    elif options.url:
        sheet = p.parseUrl(options.url, encoding=options.encoding)
        print(sheet.cssText)
    else:
        for filename in params:
            sys.stderr.write('=== CSS FILE: "%s" ===\n' % filename)
            sheet = p.parseFile(filename, encoding=options.encoding)
            print(sheet.cssText)
            print()
            sys.stderr.write('\n')


if __name__ == "__main__":
    sys.exit(main())
