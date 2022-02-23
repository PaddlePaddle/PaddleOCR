###############################################################################
# Popen for LokyProcess.
#
# author: Thomas Moreau and Olivier Grisel
#
import os
import sys
import signal
import pickle
from io import BytesIO

from . import reduction, spawn
from .context import get_spawning_popen, set_spawning_popen
from multiprocessing import util, process

if sys.version_info[:2] < (3, 3):
    ProcessLookupError = OSError

if sys.platform != "win32":
    from . import resource_tracker


__all__ = []

if sys.platform != "win32":
    #
    # Wrapper for an fd used while launching a process
    #

    class _DupFd(object):
        def __init__(self, fd):
            self.fd = reduction._mk_inheritable(fd)

        def detach(self):
            return self.fd

    #
    # Start child process using subprocess.Popen
    #

    __all__.append('Popen')

    class Popen(object):
        method = 'loky'
        DupFd = _DupFd

        def __init__(self, process_obj):
            sys.stdout.flush()
            sys.stderr.flush()
            self.returncode = None
            self._fds = []
            self._launch(process_obj)

        if sys.version_info < (3, 4):
            @classmethod
            def duplicate_for_child(cls, fd):
                popen = get_spawning_popen()
                popen._fds.append(fd)
                return reduction._mk_inheritable(fd)

        else:
            def duplicate_for_child(self, fd):
                self._fds.append(fd)
                return reduction._mk_inheritable(fd)

        def poll(self, flag=os.WNOHANG):
            if self.returncode is None:
                while True:
                    try:
                        pid, sts = os.waitpid(self.pid, flag)
                    except OSError:
                        # Child process not yet created. See #1731717
                        # e.errno == errno.ECHILD == 10
                        return None
                    else:
                        break
                if pid == self.pid:
                    if os.WIFSIGNALED(sts):
                        self.returncode = -os.WTERMSIG(sts)
                    else:
                        assert os.WIFEXITED(sts)
                        self.returncode = os.WEXITSTATUS(sts)
            return self.returncode

        def wait(self, timeout=None):
            if sys.version_info < (3, 3):
                import time
                if timeout is None:
                    return self.poll(0)
                deadline = time.time() + timeout
                delay = 0.0005
                while 1:
                    res = self.poll()
                    if res is not None:
                        break
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    delay = min(delay * 2, remaining, 0.05)
                    time.sleep(delay)
                return res

            if self.returncode is None:
                if timeout is not None:
                    from multiprocessing.connection import wait
                    if not wait([self.sentinel], timeout):
                        return None
                # This shouldn't block if wait() returned successfully.
                return self.poll(os.WNOHANG if timeout == 0.0 else 0)
            return self.returncode

        def terminate(self):
            if self.returncode is None:
                try:
                    os.kill(self.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except OSError:
                    if self.wait(timeout=0.1) is None:
                        raise

        def _launch(self, process_obj):

            tracker_fd = resource_tracker._resource_tracker.getfd()

            fp = BytesIO()
            set_spawning_popen(self)
            try:
                prep_data = spawn.get_preparation_data(
                    process_obj._name,
                    getattr(process_obj, "init_main_module", True))
                reduction.dump(prep_data, fp)
                reduction.dump(process_obj, fp)

            finally:
                set_spawning_popen(None)

            try:
                parent_r, child_w = os.pipe()
                child_r, parent_w = os.pipe()
                # for fd in self._fds:
                #     _mk_inheritable(fd)

                cmd_python = [sys.executable]
                cmd_python += ['-m', self.__module__]
                cmd_python += ['--process-name', str(process_obj.name)]
                cmd_python += ['--pipe',
                               str(reduction._mk_inheritable(child_r))]
                reduction._mk_inheritable(child_w)
                reduction._mk_inheritable(tracker_fd)
                self._fds.extend([child_r, child_w, tracker_fd])
                if sys.version_info >= (3, 8) and os.name == 'posix':
                    mp_tracker_fd = prep_data['mp_tracker_args']['fd']
                    self.duplicate_for_child(mp_tracker_fd)

                from .fork_exec import fork_exec
                pid = fork_exec(cmd_python, self._fds, env=process_obj.env)
                util.debug("launched python with pid {} and cmd:\n{}"
                           .format(pid, cmd_python))
                self.sentinel = parent_r

                method = 'getbuffer'
                if not hasattr(fp, method):
                    method = 'getvalue'
                with os.fdopen(parent_w, 'wb') as f:
                    f.write(getattr(fp, method)())
                self.pid = pid
            finally:
                if parent_r is not None:
                    util.Finalize(self, os.close, (parent_r,))
                for fd in (child_r, child_w):
                    if fd is not None:
                        os.close(fd)

        @staticmethod
        def thread_is_spawning():
            return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Command line parser')
    parser.add_argument('--pipe', type=int, required=True,
                        help='File handle for the pipe')
    parser.add_argument('--process-name', type=str, default=None,
                        help='Identifier for debugging purpose')

    args = parser.parse_args()

    info = dict()

    exitcode = 1
    try:
        with os.fdopen(args.pipe, 'rb') as from_parent:
            process.current_process()._inheriting = True
            try:
                prep_data = pickle.load(from_parent)
                spawn.prepare(prep_data)
                process_obj = pickle.load(from_parent)
            finally:
                del process.current_process()._inheriting

        exitcode = process_obj._bootstrap()
    except Exception:
        print('\n\n' + '-' * 80)
        print('{} failed with traceback: '.format(args.process_name))
        print('-' * 80)
        import traceback
        print(traceback.format_exc())
        print('\n' + '-' * 80)
    finally:
        if from_parent is not None:
            from_parent.close()

        sys.exit(exitcode)
