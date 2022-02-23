from __future__ import absolute_import, unicode_literals

import os
import subprocess
from stat import S_IREAD, S_IRGRP, S_IROTH

from virtualenv.util.path import safe_delete, set_tree
from virtualenv.util.six import ensure_text
from virtualenv.util.subprocess import Popen

from .base import PipInstall


class SymlinkPipInstall(PipInstall):
    def _sync(self, src, dst):
        src_str = ensure_text(str(src))
        dest_str = ensure_text(str(dst))
        os.symlink(src_str, dest_str)

    def _generate_new_files(self):
        # create the pyc files, as the build image will be R/O
        process = Popen(
            [ensure_text(str(self._creator.exe)), "-m", "compileall", ensure_text(str(self._image_dir))],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        process.communicate()
        # the root pyc is shared, so we'll not symlink that - but still add the pyc files to the RECORD for close
        root_py_cache = self._image_dir / "__pycache__"
        new_files = set()
        if root_py_cache.exists():
            new_files.update(root_py_cache.iterdir())
            new_files.add(root_py_cache)
            safe_delete(root_py_cache)
        core_new_files = super(SymlinkPipInstall, self)._generate_new_files()
        # remove files that are within the image folder deeper than one level (as these will be not linked directly)
        for file in core_new_files:
            try:
                rel = file.relative_to(self._image_dir)
                if len(rel.parts) > 1:
                    continue
            except ValueError:
                pass
            new_files.add(file)
        return new_files

    def _fix_records(self, new_files):
        new_files.update(i for i in self._image_dir.iterdir())
        extra_record_data_str = self._records_text(sorted(new_files, key=str))
        with open(ensure_text(str(self._dist_info / "RECORD")), "wb") as file_handler:
            file_handler.write(extra_record_data_str.encode("utf-8"))

    def build_image(self):
        super(SymlinkPipInstall, self).build_image()
        # protect the image by making it read only
        set_tree(self._image_dir, S_IREAD | S_IRGRP | S_IROTH)

    def clear(self):
        if self._image_dir.exists():
            safe_delete(self._image_dir)
        super(SymlinkPipInstall, self).clear()
