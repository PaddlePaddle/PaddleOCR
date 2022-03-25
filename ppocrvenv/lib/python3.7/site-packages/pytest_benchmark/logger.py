from __future__ import division
from __future__ import print_function

import sys
import warnings

import py
from pytest import PytestWarning


class PytestBenchmarkWarning(PytestWarning):
    pass


class Logger(object):
    QUIET, NORMAL, VERBOSE = range(3)

    def __init__(self, level=NORMAL, config=None):
        self.level = level
        self.term = py.io.TerminalWriter(file=sys.stderr)
        self.suspend_capture = None
        self.resume_capture = None
        if config:
            capman = config.pluginmanager.getplugin("capturemanager")
            if capman:
                self.suspend_capture = getattr(capman,
                                               'suspend_global_capture',
                                               getattr('capman', 'suspendcapture', None))
                self.resume_capture = getattr(capman,
                                              'resume_global_capture',
                                              getattr('capman', 'resumecapture', None))

    def warn(self, text, warner=None, suspend=False):
        if self.level >= self.VERBOSE:
            if suspend and self.suspend_capture:
                self.suspend_capture(in_=True)
            self.term.line("")
            self.term.sep("-", red=True, bold=True)
            self.term.write(" WARNING: ", red=True, bold=True)
            self.term.line(text, red=True)
            self.term.sep("-", red=True, bold=True)
            if suspend and self.resume_capture:
                self.resume_capture()
        if warner is None:
            warner = warnings.warn
        warner(PytestBenchmarkWarning(text))

    def error(self, text):
        self.term.line("")
        self.term.sep("-", red=True, bold=True)
        self.term.line(text, red=True, bold=True)
        self.term.sep("-", red=True, bold=True)

    def info(self, text, newline=True, **kwargs):
        if self.level >= self.NORMAL:
            if not kwargs or kwargs == {'bold': True}:
                kwargs['purple'] = True
            if newline:
                self.term.line("")
            self.term.line(text, **kwargs)

    def debug(self, text, newline=False, **kwargs):
        if self.level >= self.VERBOSE:
            if self.suspend_capture:
                self.suspend_capture(in_=True)
            self.info(text, newline=newline, **kwargs)
            if self.resume_capture:
                self.resume_capture()
