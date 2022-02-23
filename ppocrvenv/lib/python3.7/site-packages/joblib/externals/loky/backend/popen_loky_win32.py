import os
import sys
from pickle import load
from multiprocessing import process, util

from . import spawn
from . import reduction
from .context import get_spawning_popen, set_spawning_popen

if sys.platform == "win32":
    # Avoid import error by code introspection tools such as test runners
    # trying to import this module while running on non-Windows systems.
    import msvcrt
    from .compat_win32 import _winapi
    from .compat_win32 import Popen as _Popen
    from .reduction import duplicate
else:
    _Popen = object

if sys.version_info[:2] < (3, 3):
    from os import fdopen as open

__all__ = ['Popen']

#
#
#

TERMINATE = 0x10000
WINEXE = (sys.platform == 'win32' and getattr(sys, 'frozen', False))
WINSERVICE = sys.executable.lower().endswith("pythonservice.exe")


def _path_eq(p1, p2):
    return p1 == p2 or os.path.normcase(p1) == os.path.normcase(p2)


WINENV = (hasattr(sys, "_base_executable")
          and not _path_eq(sys.executable, sys._base_executable))

#
# We define a Popen class similar to the one from subprocess, but
# whose constructor takes a process object as its argument.
#


class Popen(_Popen):
    '''
    Start a subprocess to run the code of a process object
    '''
    method = 'loky'

    def __init__(self, process_obj):
        prep_data = spawn.get_preparation_data(
            process_obj._name, getattr(process_obj, "init_main_module", True))

        # read end of pipe will be "stolen" by the child process
        # -- see spawn_main() in spawn.py.
        rfd, wfd = os.pipe()
        rhandle = duplicate(msvcrt.get_osfhandle(rfd), inheritable=True)
        os.close(rfd)

        cmd = get_command_line(parent_pid=os.getpid(), pipe_handle=rhandle)
        cmd = ' '.join('"%s"' % x for x in cmd)

        python_exe = spawn.get_executable()

        # copy the environment variables to set in the child process
        child_env = os.environ.copy()
        child_env.update(process_obj.env)

        # bpo-35797: When running in a venv, we bypass the redirect
        # executor and launch our base Python.
        if WINENV and _path_eq(python_exe, sys.executable):
            python_exe = sys._base_executable
            child_env["__PYVENV_LAUNCHER__"] = sys.executable

        try:
            with open(wfd, 'wb') as to_child:
                # start process
                try:
                    # This flag allows to pass inheritable handles from the
                    # parent to the child process in a python2-3 compatible way
                    # (see
                    # https://github.com/tomMoral/loky/pull/204#discussion_r290719629
                    # for more detail). When support for Python 2 is dropped,
                    # the cleaner multiprocessing.reduction.steal_handle should
                    # be used instead.
                    inherit = True
                    hp, ht, pid, tid = _winapi.CreateProcess(
                        python_exe, cmd,
                        None, None, inherit, 0,
                        child_env, None, None)
                    _winapi.CloseHandle(ht)
                except BaseException:
                    _winapi.CloseHandle(rhandle)
                    raise

                # set attributes of self
                self.pid = pid
                self.returncode = None
                self._handle = hp
                self.sentinel = int(hp)
                util.Finalize(self, _winapi.CloseHandle, (self.sentinel,))

                # send information to child
                set_spawning_popen(self)
                if sys.version_info[:2] < (3, 4):
                    Popen._tls.process_handle = int(hp)
                try:
                    reduction.dump(prep_data, to_child)
                    reduction.dump(process_obj, to_child)
                finally:
                    set_spawning_popen(None)
                    if sys.version_info[:2] < (3, 4):
                        del Popen._tls.process_handle
        except IOError as exc:
            # IOError 22 happens when the launched subprocess terminated before
            # wfd.close is called. Thus we can safely ignore it.
            if exc.errno != 22:
                raise
            util.debug("While starting {}, ignored a IOError 22"
                       .format(process_obj._name))

    def duplicate_for_child(self, handle):
        assert self is get_spawning_popen()
        return duplicate(handle, self.sentinel)


def get_command_line(pipe_handle, **kwds):
    '''
    Returns prefix of command line used for spawning a child process
    '''
    if getattr(sys, 'frozen', False):
        return ([sys.executable, '--multiprocessing-fork', pipe_handle])
    else:
        prog = 'from joblib.externals.loky.backend.popen_loky_win32 import main; main()'
        opts = util._args_from_interpreter_flags()
        return [spawn.get_executable()] + opts + [
            '-c', prog, '--multiprocessing-fork', pipe_handle]


def is_forking(argv):
    '''
    Return whether commandline indicates we are forking
    '''
    if len(argv) >= 2 and argv[1] == '--multiprocessing-fork':
        assert len(argv) == 3
        return True
    else:
        return False


def main():
    '''
    Run code specified by data received over pipe
    '''
    assert is_forking(sys.argv)

    handle = int(sys.argv[-1])
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
    from_parent = os.fdopen(fd, 'rb')

    process.current_process()._inheriting = True
    preparation_data = load(from_parent)
    spawn.prepare(preparation_data)
    self = load(from_parent)
    process.current_process()._inheriting = False

    from_parent.close()

    exitcode = self._bootstrap()
    sys.exit(exitcode)
