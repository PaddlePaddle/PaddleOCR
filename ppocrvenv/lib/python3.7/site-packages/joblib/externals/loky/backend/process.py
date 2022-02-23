###############################################################################
# LokyProcess implementation
#
# authors: Thomas Moreau and Olivier Grisel
#
# based on multiprocessing/process.py  (17/02/2017)
# * Add some compatibility function for python2.7 and 3.3
#
import os
import sys
from .compat import BaseProcess


class LokyProcess(BaseProcess):
    _start_method = 'loky'

    def __init__(self, group=None, target=None, name=None, args=(),
                 kwargs={}, daemon=None, init_main_module=False,
                 env=None):
        if sys.version_info < (3, 3):
            super(LokyProcess, self).__init__(
                group=group, target=target, name=name, args=args,
                kwargs=kwargs)
            self.daemon = daemon
        else:
            super(LokyProcess, self).__init__(
                group=group, target=target, name=name, args=args,
                kwargs=kwargs, daemon=daemon)
        self.env = {} if env is None else env
        self.authkey = self.authkey
        self.init_main_module = init_main_module

    @staticmethod
    def _Popen(process_obj):
        if sys.platform == "win32":
            from .popen_loky_win32 import Popen
        else:
            from .popen_loky_posix import Popen
        return Popen(process_obj)

    if sys.version_info < (3, 3):
        def start(self):
            '''
            Start child process
            '''
            from multiprocessing.process import _current_process, _cleanup
            assert self._popen is None, 'cannot start a process twice'
            assert self._parent_pid == os.getpid(), \
                'can only start a process object created by current process'
            _cleanup()
            self._popen = self._Popen(self)
            self._sentinel = self._popen.sentinel
            _current_process._children.add(self)

        @property
        def sentinel(self):
            '''
            Return a file descriptor (Unix) or handle (Windows) suitable for
            waiting for process termination.
            '''
            try:
                return self._sentinel
            except AttributeError:
                raise ValueError("process not started")

    if sys.version_info < (3, 4):
        @property
        def authkey(self):
            return self._authkey

        @authkey.setter
        def authkey(self, authkey):
            '''
            Set authorization key of process
            '''
            self._authkey = AuthenticationKey(authkey)

        def _bootstrap(self):
            from .context import set_start_method
            set_start_method(self._start_method)
            super(LokyProcess, self)._bootstrap()


class LokyInitMainProcess(LokyProcess):
    _start_method = 'loky_init_main'

    def __init__(self, group=None, target=None, name=None, args=(),
                 kwargs={}, daemon=None):
        super(LokyInitMainProcess, self).__init__(
            group=group, target=target, name=name, args=args, kwargs=kwargs,
            daemon=daemon, init_main_module=True)


#
# We subclass bytes to avoid accidental transmission of auth keys over network
#

class AuthenticationKey(bytes):
    def __reduce__(self):
        from .context import assert_spawning
        try:
            assert_spawning(self)
        except RuntimeError:
            raise TypeError(
                'Pickling an AuthenticationKey object is '
                'disallowed for security reasons'
            )
        return AuthenticationKey, (bytes(self),)
