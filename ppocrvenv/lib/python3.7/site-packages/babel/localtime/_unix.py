# -*- coding: utf-8 -*-
from __future__ import with_statement
import os
import re
import sys
import pytz
import subprocess

_systemconfig_tz = re.compile(r'^Time Zone: (.*)$', re.MULTILINE)


def _tz_from_env(tzenv):
    if tzenv[0] == ':':
        tzenv = tzenv[1:]

    # TZ specifies a file
    if os.path.exists(tzenv):
        with open(tzenv, 'rb') as tzfile:
            return pytz.tzfile.build_tzinfo('local', tzfile)

    # TZ specifies a zoneinfo zone.
    try:
        tz = pytz.timezone(tzenv)
        # That worked, so we return this:
        return tz
    except pytz.UnknownTimeZoneError:
        raise pytz.UnknownTimeZoneError(
            "tzlocal() does not support non-zoneinfo timezones like %s. \n"
            "Please use a timezone in the form of Continent/City")


def _get_localzone(_root='/'):
    """Tries to find the local timezone configuration.
    This method prefers finding the timezone name and passing that to pytz,
    over passing in the localtime file, as in the later case the zoneinfo
    name is unknown.
    The parameter _root makes the function look for files like /etc/localtime
    beneath the _root directory. This is primarily used by the tests.
    In normal usage you call the function without parameters.
    """

    tzenv = os.environ.get('TZ')
    if tzenv:
        return _tz_from_env(tzenv)

    # This is actually a pretty reliable way to test for the local time
    # zone on operating systems like OS X.  On OS X especially this is the
    # only one that actually works.
    try:
        link_dst = os.readlink('/etc/localtime')
    except OSError:
        pass
    else:
        pos = link_dst.find('/zoneinfo/')
        if pos >= 0:
            zone_name = link_dst[pos + 10:]
            try:
                return pytz.timezone(zone_name)
            except pytz.UnknownTimeZoneError:
                pass

    # If we are on OS X now we are pretty sure that the rest of the
    # code will fail and just fall through until it hits the reading
    # of /etc/localtime and using it without name.  At this point we
    # can invoke systemconfig which internally invokes ICU.  ICU itself
    # does the same thing we do (readlink + compare file contents) but
    # since it knows where the zone files are that should be a bit
    # better than reimplementing the logic here.
    if sys.platform == 'darwin':
        c = subprocess.Popen(['systemsetup', '-gettimezone'],
                             stdout=subprocess.PIPE)
        sys_result = c.communicate()[0]
        c.wait()
        tz_match = _systemconfig_tz.search(sys_result)
        if tz_match is not None:
            zone_name = tz_match.group(1)
            try:
                return pytz.timezone(zone_name)
            except pytz.UnknownTimeZoneError:
                pass

    # Now look for distribution specific configuration files
    # that contain the timezone name.
    tzpath = os.path.join(_root, 'etc/timezone')
    if os.path.exists(tzpath):
        with open(tzpath, 'rb') as tzfile:
            data = tzfile.read()

            # Issue #3 in tzlocal was that /etc/timezone was a zoneinfo file.
            # That's a misconfiguration, but we need to handle it gracefully:
            if data[:5] != b'TZif2':
                etctz = data.strip().decode()
                # Get rid of host definitions and comments:
                if ' ' in etctz:
                    etctz, dummy = etctz.split(' ', 1)
                if '#' in etctz:
                    etctz, dummy = etctz.split('#', 1)
                return pytz.timezone(etctz.replace(' ', '_'))

    # CentOS has a ZONE setting in /etc/sysconfig/clock,
    # OpenSUSE has a TIMEZONE setting in /etc/sysconfig/clock and
    # Gentoo has a TIMEZONE setting in /etc/conf.d/clock
    # We look through these files for a timezone:
    timezone_re = re.compile(r'\s*(TIME)?ZONE\s*=\s*"(?P<etctz>.+)"')

    for filename in ('etc/sysconfig/clock', 'etc/conf.d/clock'):
        tzpath = os.path.join(_root, filename)
        if not os.path.exists(tzpath):
            continue
        with open(tzpath, 'rt') as tzfile:
            for line in tzfile:
                match = timezone_re.match(line)
                if match is not None:
                    # We found a timezone
                    etctz = match.group("etctz")
                    return pytz.timezone(etctz.replace(' ', '_'))

    # No explicit setting existed. Use localtime
    for filename in ('etc/localtime', 'usr/local/etc/localtime'):
        tzpath = os.path.join(_root, filename)

        if not os.path.exists(tzpath):
            continue

        with open(tzpath, 'rb') as tzfile:
            return pytz.tzfile.build_tzinfo('local', tzfile)

    raise pytz.UnknownTimeZoneError('Can not find any timezone configuration')
