import sys
import subprocess
import py
from subprocess import Popen, PIPE

def cmdexec(cmd):
    """ return unicode output of executing 'cmd' in a separate process.

    raise cmdexec.Error exeception if the command failed.
    the exception will provide an 'err' attribute containing
    the error-output from the command.
    if the subprocess module does not provide a proper encoding/unicode strings
    sys.getdefaultencoding() will be used, if that does not exist, 'UTF-8'.
    """
    process = subprocess.Popen(cmd, shell=True,
            universal_newlines=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if sys.version_info[0] < 3: # on py3 we get unicode strings, on py2 not
        try:
            default_encoding = sys.getdefaultencoding() # jython may not have it
        except AttributeError:
            default_encoding = sys.stdout.encoding or 'UTF-8'
        out = unicode(out, process.stdout.encoding or default_encoding)
        err = unicode(err, process.stderr.encoding or default_encoding)
    status = process.poll()
    if status:
        raise ExecutionFailed(status, status, cmd, out, err)
    return out

class ExecutionFailed(py.error.Error):
    def __init__(self, status, systemstatus, cmd, out, err):
        Exception.__init__(self)
        self.status = status
        self.systemstatus = systemstatus
        self.cmd = cmd
        self.err = err
        self.out = out

    def __str__(self):
        return "ExecutionFailed: %d  %s\n%s" %(self.status, self.cmd, self.err)

# export the exception under the name 'py.process.cmdexec.Error'
cmdexec.Error = ExecutionFailed
try:
    ExecutionFailed.__module__ = 'py.process.cmdexec'
    ExecutionFailed.__name__ = 'Error'
except (AttributeError, TypeError):
    pass
