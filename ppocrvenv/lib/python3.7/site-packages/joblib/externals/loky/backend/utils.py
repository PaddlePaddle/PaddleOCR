import os
import sys
import time
import errno
import signal
import warnings
import threading
import subprocess
try:
    import psutil
except ImportError:
    psutil = None


WIN32 = sys.platform == "win32"


def _flag_current_thread_clean_exit():
    """Put a ``_clean_exit`` flag on the current thread"""
    thread = threading.current_thread()
    thread._clean_exit = True


def recursive_terminate(process, use_psutil=True):
    if use_psutil and psutil is not None:
        _recursive_terminate_with_psutil(process)
    else:
        _recursive_terminate_without_psutil(process)


def _recursive_terminate_with_psutil(process, retries=5):
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return

    # Kill the children in reverse order to avoid killing the parents before
    # the children in cases where there are more processes nested.
    for child in children[::-1]:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    process.terminate()
    process.join()


def _recursive_terminate_without_psutil(process):
    """Terminate a process and its descendants.
    """
    try:
        _recursive_terminate(process.pid)
    except OSError as e:
        warnings.warn("Failed to kill subprocesses on this platform. Please"
                      "install psutil: https://github.com/giampaolo/psutil")
        # In case we cannot introspect the children, we fall back to the
        # classic Process.terminate.
        process.terminate()
    process.join()


def _recursive_terminate(pid):
    """Recursively kill the descendants of a process before killing it.
    """

    if sys.platform == "win32":
        # On windows, the taskkill function with option `/T` terminate a given
        # process pid and its children.
        try:
            subprocess.check_output(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stderr=None)
        except subprocess.CalledProcessError as e:
            # In windows, taskkill return 1 for permission denied and 128, 255
            # for no process found.
            if e.returncode not in [1, 128, 255]:
                raise
            elif e.returncode == 1:
                # Try to kill the process without its descendants if taskkill
                # was denied permission. If this fails too, with an error
                # different from process not found, let the top level function
                # raise a warning and retry to kill the process.
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError as e:
                    if e.errno != errno.ESRCH:
                        raise

    else:
        try:
            children_pids = subprocess.check_output(
                ["pgrep", "-P", str(pid)],
                stderr=None
            )
        except subprocess.CalledProcessError as e:
            # `ps` returns 1 when no child process has been found
            if e.returncode == 1:
                children_pids = b''
            else:
                raise

        # Decode the result, split the cpid and remove the trailing line
        children_pids = children_pids.decode().split('\n')[:-1]
        for cpid in children_pids:
            cpid = int(cpid)
            _recursive_terminate(cpid)

        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            # if OSError is raised with [Errno 3] no such process, the process
            # is already terminated, else, raise the error and let the top
            # level function raise a warning and retry to kill the process.
            if e.errno != errno.ESRCH:
                raise


def get_exitcodes_terminated_worker(processes):
    """Return a formated string with the exitcodes of terminated workers.

    If necessary, wait (up to .25s) for the system to correctly set the
    exitcode of one terminated worker.
    """
    patience = 5

    # Catch the exitcode of the terminated workers. There should at least be
    # one. If not, wait a bit for the system to correctly set the exitcode of
    # the terminated worker.
    exitcodes = [p.exitcode for p in list(processes.values())
                 if p.exitcode is not None]
    while len(exitcodes) == 0 and patience > 0:
        patience -= 1
        exitcodes = [p.exitcode for p in list(processes.values())
                     if p.exitcode is not None]
        time.sleep(.05)

    return _format_exitcodes(exitcodes)


def _format_exitcodes(exitcodes):
    """Format a list of exit code with names of the signals if possible"""
    str_exitcodes = ["{}({})".format(_get_exitcode_name(e), e)
                     for e in exitcodes if e is not None]
    return "{" + ", ".join(str_exitcodes) + "}"


def _get_exitcode_name(exitcode):
    if sys.platform == "win32":
        # The exitcode are unreliable  on windows (see bpo-31863).
        # For this case, return UNKNOWN
        return "UNKNOWN"

    if exitcode < 0:
        try:
            import signal
            if sys.version_info > (3, 5):
                return signal.Signals(-exitcode).name

            # construct an inverse lookup table
            for v, k in signal.__dict__.items():
                if (v.startswith('SIG') and not v.startswith('SIG_') and
                        k == -exitcode):
                        return v
        except ValueError:
            return "UNKNOWN"
    elif exitcode != 255:
        # The exitcode are unreliable on forkserver were 255 is always returned
        # (see bpo-30589). For this case, return UNKNOWN
        return "EXIT"

    return "UNKNOWN"
