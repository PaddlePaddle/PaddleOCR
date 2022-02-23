"""
Helpers for logging.

This module needs much love to become useful.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2008 Gael Varoquaux
# License: BSD Style, 3 clauses.

from __future__ import print_function

import time
import sys
import os
import shutil
import logging
import pprint

from .disk import mkdirp


def _squeeze_time(t):
    """Remove .1s to the time under Windows: this is the time it take to
    stat files. This is needed to make results similar to timings under
    Unix, for tests
    """
    if sys.platform.startswith('win'):
        return max(0, t - .1)
    else:
        return t


def format_time(t):
    t = _squeeze_time(t)
    return "%.1fs, %.1fmin" % (t, t / 60.)


def short_format_time(t):
    t = _squeeze_time(t)
    if t > 60:
        return "%4.1fmin" % (t / 60.)
    else:
        return " %5.1fs" % (t)


def pformat(obj, indent=0, depth=3):
    if 'numpy' in sys.modules:
        import numpy as np
        print_options = np.get_printoptions()
        np.set_printoptions(precision=6, threshold=64, edgeitems=1)
    else:
        print_options = None
    out = pprint.pformat(obj, depth=depth, indent=indent)
    if print_options:
        np.set_printoptions(**print_options)
    return out


###############################################################################
# class `Logger`
###############################################################################
class Logger(object):
    """ Base class for logging messages.
    """

    def __init__(self, depth=3):
        """
            Parameters
            ----------
            depth: int, optional
                The depth of objects printed.
        """
        self.depth = depth

    def warn(self, msg):
        logging.warning("[%s]: %s" % (self, msg))

    def debug(self, msg):
        # XXX: This conflicts with the debug flag used in children class
        logging.debug("[%s]: %s" % (self, msg))

    def format(self, obj, indent=0):
        """Return the formatted representation of the object."""
        return pformat(obj, indent=indent, depth=self.depth)


###############################################################################
# class `PrintTime`
###############################################################################
class PrintTime(object):
    """ Print and log messages while keeping track of time.
    """

    def __init__(self, logfile=None, logdir=None):
        if logfile is not None and logdir is not None:
            raise ValueError('Cannot specify both logfile and logdir')
        # XXX: Need argument docstring
        self.last_time = time.time()
        self.start_time = self.last_time
        if logdir is not None:
            logfile = os.path.join(logdir, 'joblib.log')
        self.logfile = logfile
        if logfile is not None:
            mkdirp(os.path.dirname(logfile))
            if os.path.exists(logfile):
                # Rotate the logs
                for i in range(1, 9):
                    try:
                        shutil.move(logfile + '.%i' % i,
                                    logfile + '.%i' % (i + 1))
                    except:
                        "No reason failing here"
                # Use a copy rather than a move, so that a process
                # monitoring this file does not get lost.
                try:
                    shutil.copy(logfile, logfile + '.1')
                except:
                    "No reason failing here"
            try:
                with open(logfile, 'w') as logfile:
                    logfile.write('\nLogging joblib python script\n')
                    logfile.write('\n---%s---\n' % time.ctime(self.last_time))
            except:
                """ Multiprocessing writing to files can create race
                    conditions. Rather fail silently than crash the
                    computation.
                """
                # XXX: We actually need a debug flag to disable this
                # silent failure.

    def __call__(self, msg='', total=False):
        """ Print the time elapsed between the last call and the current
            call, with an optional message.
        """
        if not total:
            time_lapse = time.time() - self.last_time
            full_msg = "%s: %s" % (msg, format_time(time_lapse))
        else:
            # FIXME: Too much logic duplicated
            time_lapse = time.time() - self.start_time
            full_msg = "%s: %.2fs, %.1f min" % (msg, time_lapse,
                                                time_lapse / 60)
        print(full_msg, file=sys.stderr)
        if self.logfile is not None:
            try:
                with open(self.logfile, 'a') as f:
                    print(full_msg, file=f)
            except:
                """ Multiprocessing writing to files can create race
                    conditions. Rather fail silently than crash the
                    calculation.
                """
                # XXX: We actually need a debug flag to disable this
                # silent failure.
        self.last_time = time.time()
