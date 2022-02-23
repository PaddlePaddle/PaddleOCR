import sys

if 'setuptools' in sys.modules:
    try:
        from setuptools.command.build_ext import build_ext as _build_ext
    except ImportError:
        # We may be in the process of importing setuptools, which tries
        # to import this.
        from distutils.command.build_ext import build_ext as _build_ext
else:
    from distutils.command.build_ext import build_ext as _build_ext


class new_build_ext(_build_ext, object):
    def finalize_options(self):
        if self.distribution.ext_modules:
            nthreads = getattr(self, 'parallel', None)  # -j option in Py3.5+
            nthreads = int(nthreads) if nthreads else None
            from Cython.Build.Dependencies import cythonize
            self.distribution.ext_modules[:] = cythonize(
                self.distribution.ext_modules, nthreads=nthreads, force=self.force)
        super(new_build_ext, self).finalize_options()

# This will become new_build_ext in the future.
from .old_build_ext import old_build_ext as build_ext
