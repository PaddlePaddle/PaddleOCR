import py
import os, sys

if sys.platform == "win32" or getattr(os, '_name', '') == 'nt':
    try:
        import ctypes
    except ImportError:
        def dokill(pid):
            py.process.cmdexec("taskkill /F /PID %d" %(pid,))
    else:
        def dokill(pid):
            PROCESS_TERMINATE = 1
            handle = ctypes.windll.kernel32.OpenProcess(
                        PROCESS_TERMINATE, False, pid)
            ctypes.windll.kernel32.TerminateProcess(handle, -1)
            ctypes.windll.kernel32.CloseHandle(handle)
else:
    def dokill(pid):
        os.kill(pid, 15)

def kill(pid):
    """ kill process by id. """
    dokill(pid)
