from __future__ import absolute_import, unicode_literals

from abc import ABCMeta

from six import add_metaclass

from virtualenv.create.via_global_ref.builtin.ref import ExePathRefToDest, RefMust, RefWhen
from virtualenv.util.path import ensure_dir

from ..api import ViaGlobalRefApi, ViaGlobalRefMeta
from .builtin_way import VirtualenvBuiltin


class BuiltinViaGlobalRefMeta(ViaGlobalRefMeta):
    def __init__(self):
        super(BuiltinViaGlobalRefMeta, self).__init__()
        self.sources = []


@add_metaclass(ABCMeta)
class ViaGlobalRefVirtualenvBuiltin(ViaGlobalRefApi, VirtualenvBuiltin):
    def __init__(self, options, interpreter):
        super(ViaGlobalRefVirtualenvBuiltin, self).__init__(options, interpreter)
        self._sources = getattr(options.meta, "sources", None)  # if we're created as a describer this might be missing

    @classmethod
    def can_create(cls, interpreter):
        """By default all built-in methods assume that if we can describe it we can create it"""
        # first we must be able to describe it
        if not cls.can_describe(interpreter):
            return None
        meta = cls.setup_meta(interpreter)
        if meta is not None and meta:
            cls._sources_can_be_applied(interpreter, meta)
        return meta

    @classmethod
    def _sources_can_be_applied(cls, interpreter, meta):
        for src in cls.sources(interpreter):
            if src.exists:
                if meta.can_copy and not src.can_copy:
                    meta.copy_error = "cannot copy {}".format(src)
                if meta.can_symlink and not src.can_symlink:
                    meta.symlink_error = "cannot symlink {}".format(src)
            else:
                msg = "missing required file {}".format(src)
                if src.when == RefMust.NA:
                    meta.error = msg
                elif src.when == RefMust.COPY:
                    meta.copy_error = msg
                elif src.when == RefMust.SYMLINK:
                    meta.symlink_error = msg
            if not meta.can_copy and not meta.can_symlink:
                meta.error = "neither copy or symlink supported, copy: {} symlink: {}".format(
                    meta.copy_error,
                    meta.symlink_error,
                )
            if meta.error:
                break
            meta.sources.append(src)

    @classmethod
    def setup_meta(cls, interpreter):
        return BuiltinViaGlobalRefMeta()

    @classmethod
    def sources(cls, interpreter):
        for host_exe, targets, must, when in cls._executables(interpreter):
            yield ExePathRefToDest(host_exe, dest=cls.to_bin, targets=targets, must=must, when=when)

    def to_bin(self, src):
        return self.bin_dir / src.name

    @classmethod
    def _executables(cls, interpreter):
        raise NotImplementedError

    def create(self):
        dirs = self.ensure_directories()
        for directory in list(dirs):
            if any(i for i in dirs if i is not directory and directory.parts == i.parts[: len(directory.parts)]):
                dirs.remove(directory)
        for directory in sorted(dirs):
            ensure_dir(directory)

        self.set_pyenv_cfg()
        self.pyenv_cfg.write()
        true_system_site = self.enable_system_site_package
        try:
            self.enable_system_site_package = False
            for src in self._sources:
                if (
                    src.when == RefWhen.ANY
                    or (src.when == RefWhen.SYMLINK and self.symlinks is True)
                    or (src.when == RefWhen.COPY and self.symlinks is False)
                ):
                    src.run(self, self.symlinks)
        finally:
            if true_system_site != self.enable_system_site_package:
                self.enable_system_site_package = true_system_site
        super(ViaGlobalRefVirtualenvBuiltin, self).create()

    def ensure_directories(self):
        return {self.dest, self.bin_dir, self.script_dir, self.stdlib} | set(self.libs)

    def set_pyenv_cfg(self):
        """
        We directly inject the base prefix and base exec prefix to avoid site.py needing to discover these
        from home (which usually is done within the interpreter itself)
        """
        super(ViaGlobalRefVirtualenvBuiltin, self).set_pyenv_cfg()
        self.pyenv_cfg["base-prefix"] = self.interpreter.system_prefix
        self.pyenv_cfg["base-exec-prefix"] = self.interpreter.system_exec_prefix
        self.pyenv_cfg["base-executable"] = self.interpreter.system_executable
