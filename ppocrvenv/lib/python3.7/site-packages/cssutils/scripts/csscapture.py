"""Retrieve all CSS stylesheets including embedded for a given URL.
Retrieve as StyleSheetList or save to disk - raw, parsed or minified version.

TODO:
- maybe use DOM 3 load/save?
- logger class which handles all cases when no log is given...
- saveto: why does urllib2 hang?
"""
__all__ = ['CSSCapture']

from cssutils.script import CSSCapture
import logging
import optparse
import sys


def main(args=None):
    usage = "usage: %prog [options] URL"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option(
        '-d',
        '--debug',
        action='store_true',
        dest='debug',
        help='show debug messages during capturing',
    )
    parser.add_option(
        '-m',
        '--minified',
        action='store_true',
        dest='minified',
        help='saves minified version of captured files',
    )
    parser.add_option(
        '-n',
        '--notsave',
        action='store_true',
        dest='notsave',
        help='if given files are NOT saved, only log is written',
    )
    #    parser.add_option('-r', '--saveraw', action='store_true', dest='saveraw',
    #        help='if given saves raw css otherwise cssutils\' parsed files')
    parser.add_option(
        '-s',
        '--saveto',
        action='store',
        dest='saveto',
        help='saving retrieved files to "saveto", defaults to "_CSSCapture_SAVED"',
    )
    parser.add_option(
        '-u',
        '--useragent',
        action='store',
        dest='ua',
        help='useragent to use for request of URL, default is urllib2s default',
    )
    options, url = parser.parse_args()

    # TODO:
    options.saveraw = False

    if not url:
        parser.error('no URL given')
    else:
        url = url[0]

    if options.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # START
    c = CSSCapture(ua=options.ua, defaultloglevel=level)

    stylesheetlist = c.capture(url)

    if options.notsave is None or not options.notsave:
        if options.saveto:
            saveto = options.saveto
        else:
            saveto = '_CSSCapture_SAVED'
        c.saveto(saveto, saveraw=options.saveraw, minified=options.minified)
    else:
        for i, s in enumerate(stylesheetlist):
            print(
                '''%s.
    encoding: %r
    title: %r
    href: %r'''
                % (i + 1, s.encoding, s.title, s.href)
            )


if __name__ == "__main__":
    sys.exit(main())
