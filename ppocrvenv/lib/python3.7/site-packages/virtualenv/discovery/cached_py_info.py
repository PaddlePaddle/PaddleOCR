"""

We acquire the python information by running an interrogation script via subprocess trigger. This operation is not
cheap, especially not on Windows. To not have to pay this hefty cost every time we apply multiple levels of
caching.
"""
from __future__ import absolute_import, unicode_literals

import logging
import os
import pipes
import sys
from collections import OrderedDict

from virtualenv.app_data import AppDataDisabled
from virtualenv.discovery.py_info import PythonInfo
from virtualenv.info import PY2
from virtualenv.util.path import Path
from virtualenv.util.six import ensure_text
from virtualenv.util.subprocess import Popen, subprocess

_CACHE = OrderedDict()
_CACHE[Path(sys.executable)] = PythonInfo()


def from_exe(cls, app_data, exe, env=None, raise_on_error=True, ignore_cache=False):
    env = os.environ if env is None else env
    result = _get_from_cache(cls, app_data, exe, env, ignore_cache=ignore_cache)
    if isinstance(result, Exception):
        if raise_on_error:
            raise result
        else:
            logging.info("%s", str(result))
        result = None
    return result


def _get_from_cache(cls, app_data, exe, env, ignore_cache=True):
    # note here we cannot resolve symlinks, as the symlink may trigger different prefix information if there's a
    # pyenv.cfg somewhere alongside on python3.5+
    exe_path = Path(exe)
    if not ignore_cache and exe_path in _CACHE:  # check in the in-memory cache
        result = _CACHE[exe_path]
    else:  # otherwise go through the app data cache
        py_info = _get_via_file_cache(cls, app_data, exe_path, exe, env)
        result = _CACHE[exe_path] = py_info
    # independent if it was from the file or in-memory cache fix the original executable location
    if isinstance(result, PythonInfo):
        result.executable = exe
    return result


def _get_via_file_cache(cls, app_data, path, exe, env):
    path_text = ensure_text(str(path))
    try:
        path_modified = path.stat().st_mtime
    except OSError:
        path_modified = -1
    if app_data is None:
        app_data = AppDataDisabled()
    py_info, py_info_store = None, app_data.py_info(path)
    with py_info_store.locked():
        if py_info_store.exists():  # if exists and matches load
            data = py_info_store.read()
            of_path, of_st_mtime, of_content = data["path"], data["st_mtime"], data["content"]
            if of_path == path_text and of_st_mtime == path_modified:
                py_info = cls._from_dict({k: v for k, v in of_content.items()})
                sys_exe = py_info.system_executable
                if sys_exe is not None and not os.path.exists(sys_exe):
                    py_info_store.remove()
                    py_info = None
            else:
                py_info_store.remove()
        if py_info is None:  # if not loaded run and save
            failure, py_info = _run_subprocess(cls, exe, app_data, env)
            if failure is None:
                data = {"st_mtime": path_modified, "path": path_text, "content": py_info._to_dict()}
                py_info_store.write(data)
            else:
                py_info = failure
    return py_info


def _run_subprocess(cls, exe, app_data, env):
    py_info_script = Path(os.path.abspath(__file__)).parent / "py_info.py"
    with app_data.ensure_extracted(py_info_script) as py_info_script:
        cmd = [exe, str(py_info_script)]
        # prevent sys.prefix from leaking into the child process - see https://bugs.python.org/issue22490
        env = env.copy()
        env.pop("__PYVENV_LAUNCHER__", None)
        logging.debug("get interpreter info via cmd: %s", LogCmd(cmd))
        try:
            process = Popen(
                cmd,
                universal_newlines=True,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env=env,
            )
            out, err = process.communicate()
            code = process.returncode
        except OSError as os_error:
            out, err, code = "", os_error.strerror, os_error.errno
    result, failure = None, None
    if code == 0:
        result = cls._from_json(out)
        result.executable = exe  # keep original executable as this may contain initialization code
    else:
        msg = "failed to query {} with code {}{}{}".format(
            exe,
            code,
            " out: {!r}".format(out) if out else "",
            " err: {!r}".format(err) if err else "",
        )
        failure = RuntimeError(msg)
    return failure, result


class LogCmd(object):
    def __init__(self, cmd, env=None):
        self.cmd = cmd
        self.env = env

    def __repr__(self):
        def e(v):
            return v.decode("utf-8") if isinstance(v, bytes) else v

        cmd_repr = e(" ").join(pipes.quote(e(c)) for c in self.cmd)
        if self.env is not None:
            cmd_repr += e(" env of {!r}").format(self.env)
        if PY2:
            return cmd_repr.encode("utf-8")
        return cmd_repr

    def __unicode__(self):
        raw = repr(self)
        if PY2:
            return raw.decode("utf-8")
        return raw


def clear(app_data):
    app_data.py_info_clear()
    _CACHE.clear()


___all___ = (
    "from_exe",
    "clear",
    "LogCmd",
)
