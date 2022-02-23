from __future__ import absolute_import, unicode_literals

import logging
from copy import copy

from virtualenv.create.via_global_ref.store import handle_store_python
from virtualenv.discovery.py_info import PythonInfo
from virtualenv.util.error import ProcessCallFailed
from virtualenv.util.path import ensure_dir
from virtualenv.util.subprocess import run_cmd

from .api import ViaGlobalRefApi, ViaGlobalRefMeta


class Venv(ViaGlobalRefApi):
    def __init__(self, options, interpreter):
        self.describe = options.describe
        super(Venv, self).__init__(options, interpreter)
        self.can_be_inline = (
            interpreter is PythonInfo.current() and interpreter.executable == interpreter.system_executable
        )
        self._context = None

    def _args(self):
        return super(Venv, self)._args() + ([("describe", self.describe.__class__.__name__)] if self.describe else [])

    @classmethod
    def can_create(cls, interpreter):
        if interpreter.has_venv:
            meta = ViaGlobalRefMeta()
            if interpreter.platform == "win32" and interpreter.version_info.major == 3:
                meta = handle_store_python(meta, interpreter)
            return meta
        return None

    def create(self):
        if self.can_be_inline:
            self.create_inline()
        else:
            self.create_via_sub_process()
        for lib in self.libs:
            ensure_dir(lib)
        super(Venv, self).create()

    def create_inline(self):
        from venv import EnvBuilder

        builder = EnvBuilder(
            system_site_packages=self.enable_system_site_package,
            clear=False,
            symlinks=self.symlinks,
            with_pip=False,
        )
        builder.create(str(self.dest))

    def create_via_sub_process(self):
        cmd = self.get_host_create_cmd()
        logging.info("using host built-in venv to create via %s", " ".join(cmd))
        code, out, err = run_cmd(cmd)
        if code != 0:
            raise ProcessCallFailed(code, out, err, cmd)

    def get_host_create_cmd(self):
        cmd = [self.interpreter.system_executable, "-m", "venv", "--without-pip"]
        if self.enable_system_site_package:
            cmd.append("--system-site-packages")
        cmd.append("--symlinks" if self.symlinks else "--copies")
        cmd.append(str(self.dest))
        return cmd

    def set_pyenv_cfg(self):
        # prefer venv options over ours, but keep our extra
        venv_content = copy(self.pyenv_cfg.refresh())
        super(Venv, self).set_pyenv_cfg()
        self.pyenv_cfg.update(venv_content)

    def __getattribute__(self, item):
        describe = object.__getattribute__(self, "describe")
        if describe is not None and hasattr(describe, item):
            element = getattr(describe, item)
            if not callable(element) or item in ("script",):
                return element
        return object.__getattribute__(self, item)
