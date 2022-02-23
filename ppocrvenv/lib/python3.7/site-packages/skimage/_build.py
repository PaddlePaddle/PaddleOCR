import sys
import os
from packaging import version
from multiprocessing import cpu_count

CYTHON_VERSION = '0.23.4'

# WindowsError is not defined on unix systems
try:
    WindowsError
except NameError:
    class WindowsError(Exception):
        pass


def _compiled_filename(f):
    """Check for the presence of a .pyx[.in] file as a .c or .cpp."""
    basename = f.replace('.in', '').replace('.pyx', '')
    for ext in ('.c', '.cpp'):
        filename = basename + ext
        if os.path.exists(filename):
            return filename
    else:
        raise RuntimeError('Cython >= %s is required to build '
                           'scikit-image from git checkout' %
                           CYTHON_VERSION)


def cython(pyx_files, working_path=''):
    """Use Cython to convert the given files to C.

    Parameters
    ----------
    pyx_files : list of str
        The input .pyx files.

    """
    # Do not build cython files if target is clean
    if len(sys.argv) >= 2 and sys.argv[1] == 'clean':
        return

    try:
        from Cython import __version__
        if version.parse(__version__) < version.parse(CYTHON_VERSION):
            raise RuntimeError('Cython >= %s needed to build scikit-image' % CYTHON_VERSION)

        from Cython.Build import cythonize
    except ImportError:
        # If cython is not found, the build will make use of
        # the distributed .c or .cpp files if present
        c_files_used = [_compiled_filename(os.path.join(working_path, f))
                        for f in pyx_files]

        print("Cython >= %s not found; falling back to pre-built %s" \
              % (CYTHON_VERSION, " ".join(c_files_used)))
    else:
        pyx_files = [os.path.join(working_path, f) for f in pyx_files]
        for i, pyxfile in enumerate(pyx_files):
            if pyxfile.endswith('.pyx.in'):
                process_tempita_pyx(pyxfile)
                pyx_files[i] = pyxfile.replace('.pyx.in', '.pyx')

        # skip cythonize when creating an sdist
        # (we do not want the large cython-generated sources to be included)
        if 'sdist' not in sys.argv:
            # Cython doesn't automatically choose a number of threads > 1
            # https://github.com/cython/cython/blob/a0bbb940c847dfe92cac446c8784c34c28c92836/Cython/Build/Dependencies.py#L923-L925
            cythonize(pyx_files, nthreads=cpu_count(),
                      compiler_directives={'language_level': 3})


def process_tempita_pyx(fromfile):
    try:
        try:
            from Cython import Tempita as tempita
        except ImportError:
            import tempita
    except ImportError:
        raise Exception('Building requires Tempita: '
                        'pip install --user Tempita')
    template = tempita.Template.from_filename(fromfile,
                                              encoding=sys.getdefaultencoding())
    pyxcontent = template.substitute()
    if not fromfile.endswith('.pyx.in'):
        raise ValueError("Unexpected extension of %s." % fromfile)

    pyxfile = os.path.splitext(fromfile)[0]    # split off the .in ending
    with open(pyxfile, "w") as f:
        f.write(pyxcontent)
