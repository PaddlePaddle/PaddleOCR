"""Support for installing and building the "wheel" binary package format.
"""

# The following comment should be removed at some point in the future.
# mypy: strict-optional=False

from __future__ import absolute_import

import collections
import compileall
import contextlib
import csv
import logging
import os.path
import re
import shutil
import stat
import sys
import warnings
from base64 import urlsafe_b64encode
from itertools import starmap
from zipfile import ZipFile

from pip._vendor import pkg_resources
from pip._vendor.distlib.scripts import ScriptMaker
from pip._vendor.distlib.util import get_export_entry
from pip._vendor.six import StringIO

from pip._internal.exceptions import InstallationError
from pip._internal.locations import get_major_minor_version
from pip._internal.models.direct_url import DIRECT_URL_METADATA_NAME, DirectUrl
from pip._internal.utils.filesystem import adjacent_tmp_file, replace
from pip._internal.utils.misc import captured_stdout, ensure_dir, hash_file
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.typing import MYPY_CHECK_RUNNING
from pip._internal.utils.unpacking import current_umask, unpack_file
from pip._internal.utils.wheel import parse_wheel

if MYPY_CHECK_RUNNING:
    from email.message import Message
    from typing import (
        Dict, List, Optional, Sequence, Tuple, Any,
        Iterable, Iterator, Callable, Set,
    )

    from pip._internal.models.scheme import Scheme
    from pip._internal.utils.filesystem import NamedTemporaryFileResult

    InstalledCSVRow = Tuple[str, ...]


logger = logging.getLogger(__name__)


def normpath(src, p):
    # type: (str, str) -> str
    return os.path.relpath(src, p).replace(os.path.sep, '/')


def rehash(path, blocksize=1 << 20):
    # type: (str, int) -> Tuple[str, str]
    """Return (encoded_digest, length) for path using hashlib.sha256()"""
    h, length = hash_file(path, blocksize)
    digest = 'sha256=' + urlsafe_b64encode(
        h.digest()
    ).decode('latin1').rstrip('=')
    # unicode/str python2 issues
    return (digest, str(length))  # type: ignore


def csv_io_kwargs(mode):
    # type: (str) -> Dict[str, Any]
    """Return keyword arguments to properly open a CSV file
    in the given mode.
    """
    if sys.version_info.major < 3:
        return {'mode': '{}b'.format(mode)}
    else:
        return {'mode': mode, 'newline': ''}


def fix_script(path):
    # type: (str) -> Optional[bool]
    """Replace #!python with #!/path/to/python
    Return True if file was changed.
    """
    # XXX RECORD hashes will need to be updated
    if os.path.isfile(path):
        with open(path, 'rb') as script:
            firstline = script.readline()
            if not firstline.startswith(b'#!python'):
                return False
            exename = sys.executable.encode(sys.getfilesystemencoding())
            firstline = b'#!' + exename + os.linesep.encode("ascii")
            rest = script.read()
        with open(path, 'wb') as script:
            script.write(firstline)
            script.write(rest)
        return True
    return None


def wheel_root_is_purelib(metadata):
    # type: (Message) -> bool
    return metadata.get("Root-Is-Purelib", "").lower() == "true"


def get_entrypoints(filename):
    # type: (str) -> Tuple[Dict[str, str], Dict[str, str]]
    if not os.path.exists(filename):
        return {}, {}

    # This is done because you can pass a string to entry_points wrappers which
    # means that they may or may not be valid INI files. The attempt here is to
    # strip leading and trailing whitespace in order to make them valid INI
    # files.
    with open(filename) as fp:
        data = StringIO()
        for line in fp:
            data.write(line.strip())
            data.write("\n")
        data.seek(0)

    # get the entry points and then the script names
    entry_points = pkg_resources.EntryPoint.parse_map(data)
    console = entry_points.get('console_scripts', {})
    gui = entry_points.get('gui_scripts', {})

    def _split_ep(s):
        # type: (pkg_resources.EntryPoint) -> Tuple[str, str]
        """get the string representation of EntryPoint,
        remove space and split on '='
        """
        split_parts = str(s).replace(" ", "").split("=")
        return split_parts[0], split_parts[1]

    # convert the EntryPoint objects into strings with module:function
    console = dict(_split_ep(v) for v in console.values())
    gui = dict(_split_ep(v) for v in gui.values())
    return console, gui


def message_about_scripts_not_on_PATH(scripts):
    # type: (Sequence[str]) -> Optional[str]
    """Determine if any scripts are not on PATH and format a warning.
    Returns a warning message if one or more scripts are not on PATH,
    otherwise None.
    """
    if not scripts:
        return None

    # Group scripts by the path they were installed in
    grouped_by_dir = collections.defaultdict(set)  # type: Dict[str, Set[str]]
    for destfile in scripts:
        parent_dir = os.path.dirname(destfile)
        script_name = os.path.basename(destfile)
        grouped_by_dir[parent_dir].add(script_name)

    # We don't want to warn for directories that are on PATH.
    not_warn_dirs = [
        os.path.normcase(i).rstrip(os.sep) for i in
        os.environ.get("PATH", "").split(os.pathsep)
    ]
    # If an executable sits with sys.executable, we don't warn for it.
    #     This covers the case of venv invocations without activating the venv.
    not_warn_dirs.append(os.path.normcase(os.path.dirname(sys.executable)))
    warn_for = {
        parent_dir: scripts for parent_dir, scripts in grouped_by_dir.items()
        if os.path.normcase(parent_dir) not in not_warn_dirs
    }  # type: Dict[str, Set[str]]
    if not warn_for:
        return None

    # Format a message
    msg_lines = []
    for parent_dir, dir_scripts in warn_for.items():
        sorted_scripts = sorted(dir_scripts)  # type: List[str]
        if len(sorted_scripts) == 1:
            start_text = "script {} is".format(sorted_scripts[0])
        else:
            start_text = "scripts {} are".format(
                ", ".join(sorted_scripts[:-1]) + " and " + sorted_scripts[-1]
            )

        msg_lines.append(
            "The {} installed in '{}' which is not on PATH."
            .format(start_text, parent_dir)
        )

    last_line_fmt = (
        "Consider adding {} to PATH or, if you prefer "
        "to suppress this warning, use --no-warn-script-location."
    )
    if len(msg_lines) == 1:
        msg_lines.append(last_line_fmt.format("this directory"))
    else:
        msg_lines.append(last_line_fmt.format("these directories"))

    # Add a note if any directory starts with ~
    warn_for_tilde = any(
        i[0] == "~" for i in os.environ.get("PATH", "").split(os.pathsep) if i
    )
    if warn_for_tilde:
        tilde_warning_msg = (
            "NOTE: The current PATH contains path(s) starting with `~`, "
            "which may not be expanded by all applications."
        )
        msg_lines.append(tilde_warning_msg)

    # Returns the formatted multiline message
    return "\n".join(msg_lines)


def sorted_outrows(outrows):
    # type: (Iterable[InstalledCSVRow]) -> List[InstalledCSVRow]
    """Return the given rows of a RECORD file in sorted order.

    Each row is a 3-tuple (path, hash, size) and corresponds to a record of
    a RECORD file (see PEP 376 and PEP 427 for details).  For the rows
    passed to this function, the size can be an integer as an int or string,
    or the empty string.
    """
    # Normally, there should only be one row per path, in which case the
    # second and third elements don't come into play when sorting.
    # However, in cases in the wild where a path might happen to occur twice,
    # we don't want the sort operation to trigger an error (but still want
    # determinism).  Since the third element can be an int or string, we
    # coerce each element to a string to avoid a TypeError in this case.
    # For additional background, see--
    # https://github.com/pypa/pip/issues/5868
    return sorted(outrows, key=lambda row: tuple(str(x) for x in row))


def get_csv_rows_for_installed(
    old_csv_rows,  # type: Iterable[List[str]]
    installed,  # type: Dict[str, str]
    changed,  # type: Set[str]
    generated,  # type: List[str]
    lib_dir,  # type: str
):
    # type: (...) -> List[InstalledCSVRow]
    """
    :param installed: A map from archive RECORD path to installation RECORD
        path.
    """
    installed_rows = []  # type: List[InstalledCSVRow]
    for row in old_csv_rows:
        if len(row) > 3:
            logger.warning(
                'RECORD line has more than three elements: {}'.format(row)
            )
        # Make a copy because we are mutating the row.
        row = list(row)
        old_path = row[0]
        new_path = installed.pop(old_path, old_path)
        row[0] = new_path
        if new_path in changed:
            digest, length = rehash(new_path)
            row[1] = digest
            row[2] = length
        installed_rows.append(tuple(row))
    for f in generated:
        digest, length = rehash(f)
        installed_rows.append((normpath(f, lib_dir), digest, str(length)))
    for f in installed:
        installed_rows.append((installed[f], '', ''))
    return installed_rows


class MissingCallableSuffix(Exception):
    pass


def _raise_for_invalid_entrypoint(specification):
    # type: (str) -> None
    entry = get_export_entry(specification)
    if entry is not None and entry.suffix is None:
        raise MissingCallableSuffix(str(entry))


class PipScriptMaker(ScriptMaker):
    def make(self, specification, options=None):
        # type: (str, Dict[str, Any]) -> List[str]
        _raise_for_invalid_entrypoint(specification)
        return super(PipScriptMaker, self).make(specification, options)


def install_unpacked_wheel(
    name,  # type: str
    wheeldir,  # type: str
    wheel_zip,  # type: ZipFile
    scheme,  # type: Scheme
    req_description,  # type: str
    pycompile=True,  # type: bool
    warn_script_location=True,  # type: bool
    direct_url=None,  # type: Optional[DirectUrl]
):
    # type: (...) -> None
    """Install a wheel.

    :param name: Name of the project to install
    :param wheeldir: Base directory of the unpacked wheel
    :param wheel_zip: open ZipFile for wheel being installed
    :param scheme: Distutils scheme dictating the install directories
    :param req_description: String used in place of the requirement, for
        logging
    :param pycompile: Whether to byte-compile installed Python files
    :param warn_script_location: Whether to check that scripts are installed
        into a directory on PATH
    :raises UnsupportedWheel:
        * when the directory holds an unpacked wheel with incompatible
          Wheel-Version
        * when the .dist-info dir does not match the wheel
    """
    # TODO: Investigate and break this up.
    # TODO: Look into moving this into a dedicated class for representing an
    #       installation.

    source = wheeldir.rstrip(os.path.sep) + os.path.sep

    info_dir, metadata = parse_wheel(wheel_zip, name)

    if wheel_root_is_purelib(metadata):
        lib_dir = scheme.purelib
    else:
        lib_dir = scheme.platlib

    subdirs = os.listdir(source)
    data_dirs = [s for s in subdirs if s.endswith('.data')]

    # Record details of the files moved
    #   installed = files copied from the wheel to the destination
    #   changed = files changed while installing (scripts #! line typically)
    #   generated = files newly generated during the install (script wrappers)
    installed = {}  # type: Dict[str, str]
    changed = set()
    generated = []  # type: List[str]

    # Compile all of the pyc files that we're going to be installing
    if pycompile:
        with captured_stdout() as stdout:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                compileall.compile_dir(source, force=True, quiet=True)
        logger.debug(stdout.getvalue())

    def record_installed(srcfile, destfile, modified=False):
        # type: (str, str, bool) -> None
        """Map archive RECORD paths to installation RECORD paths."""
        oldpath = normpath(srcfile, wheeldir)
        newpath = normpath(destfile, lib_dir)
        installed[oldpath] = newpath
        if modified:
            changed.add(destfile)

    def clobber(
            source,  # type: str
            dest,  # type: str
            is_base,  # type: bool
            fixer=None,  # type: Optional[Callable[[str], Any]]
            filter=None  # type: Optional[Callable[[str], bool]]
    ):
        # type: (...) -> None
        ensure_dir(dest)  # common for the 'include' path

        for dir, subdirs, files in os.walk(source):
            basedir = dir[len(source):].lstrip(os.path.sep)
            destdir = os.path.join(dest, basedir)
            if is_base and basedir == '':
                subdirs[:] = [s for s in subdirs if not s.endswith('.data')]
            for f in files:
                # Skip unwanted files
                if filter and filter(f):
                    continue
                srcfile = os.path.join(dir, f)
                destfile = os.path.join(dest, basedir, f)
                # directory creation is lazy and after the file filtering above
                # to ensure we don't install empty dirs; empty dirs can't be
                # uninstalled.
                ensure_dir(destdir)

                # copyfile (called below) truncates the destination if it
                # exists and then writes the new contents. This is fine in most
                # cases, but can cause a segfault if pip has loaded a shared
                # object (e.g. from pyopenssl through its vendored urllib3)
                # Since the shared object is mmap'd an attempt to call a
                # symbol in it will then cause a segfault. Unlinking the file
                # allows writing of new contents while allowing the process to
                # continue to use the old copy.
                if os.path.exists(destfile):
                    os.unlink(destfile)

                # We use copyfile (not move, copy, or copy2) to be extra sure
                # that we are not moving directories over (copyfile fails for
                # directories) as well as to ensure that we are not copying
                # over any metadata because we want more control over what
                # metadata we actually copy over.
                shutil.copyfile(srcfile, destfile)

                # Copy over the metadata for the file, currently this only
                # includes the atime and mtime.
                st = os.stat(srcfile)
                if hasattr(os, "utime"):
                    os.utime(destfile, (st.st_atime, st.st_mtime))

                # If our file is executable, then make our destination file
                # executable.
                if os.access(srcfile, os.X_OK):
                    st = os.stat(srcfile)
                    permissions = (
                        st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                    )
                    os.chmod(destfile, permissions)

                changed = False
                if fixer:
                    changed = fixer(destfile)
                record_installed(srcfile, destfile, changed)

    clobber(source, lib_dir, True)

    dest_info_dir = os.path.join(lib_dir, info_dir)

    # Get the defined entry points
    ep_file = os.path.join(dest_info_dir, 'entry_points.txt')
    console, gui = get_entrypoints(ep_file)

    def is_entrypoint_wrapper(name):
        # type: (str) -> bool
        # EP, EP.exe and EP-script.py are scripts generated for
        # entry point EP by setuptools
        if name.lower().endswith('.exe'):
            matchname = name[:-4]
        elif name.lower().endswith('-script.py'):
            matchname = name[:-10]
        elif name.lower().endswith(".pya"):
            matchname = name[:-4]
        else:
            matchname = name
        # Ignore setuptools-generated scripts
        return (matchname in console or matchname in gui)

    for datadir in data_dirs:
        fixer = None
        filter = None
        for subdir in os.listdir(os.path.join(wheeldir, datadir)):
            fixer = None
            if subdir == 'scripts':
                fixer = fix_script
                filter = is_entrypoint_wrapper
            source = os.path.join(wheeldir, datadir, subdir)
            dest = getattr(scheme, subdir)
            clobber(source, dest, False, fixer=fixer, filter=filter)

    maker = PipScriptMaker(None, scheme.scripts)

    # Ensure old scripts are overwritten.
    # See https://github.com/pypa/pip/issues/1800
    maker.clobber = True

    # Ensure we don't generate any variants for scripts because this is almost
    # never what somebody wants.
    # See https://bitbucket.org/pypa/distlib/issue/35/
    maker.variants = {''}

    # This is required because otherwise distlib creates scripts that are not
    # executable.
    # See https://bitbucket.org/pypa/distlib/issue/32/
    maker.set_mode = True

    scripts_to_generate = []

    # Special case pip and setuptools to generate versioned wrappers
    #
    # The issue is that some projects (specifically, pip and setuptools) use
    # code in setup.py to create "versioned" entry points - pip2.7 on Python
    # 2.7, pip3.3 on Python 3.3, etc. But these entry points are baked into
    # the wheel metadata at build time, and so if the wheel is installed with
    # a *different* version of Python the entry points will be wrong. The
    # correct fix for this is to enhance the metadata to be able to describe
    # such versioned entry points, but that won't happen till Metadata 2.0 is
    # available.
    # In the meantime, projects using versioned entry points will either have
    # incorrect versioned entry points, or they will not be able to distribute
    # "universal" wheels (i.e., they will need a wheel per Python version).
    #
    # Because setuptools and pip are bundled with _ensurepip and virtualenv,
    # we need to use universal wheels. So, as a stopgap until Metadata 2.0, we
    # override the versioned entry points in the wheel and generate the
    # correct ones. This code is purely a short-term measure until Metadata 2.0
    # is available.
    #
    # To add the level of hack in this section of code, in order to support
    # ensurepip this code will look for an ``ENSUREPIP_OPTIONS`` environment
    # variable which will control which version scripts get installed.
    #
    # ENSUREPIP_OPTIONS=altinstall
    #   - Only pipX.Y and easy_install-X.Y will be generated and installed
    # ENSUREPIP_OPTIONS=install
    #   - pipX.Y, pipX, easy_install-X.Y will be generated and installed. Note
    #     that this option is technically if ENSUREPIP_OPTIONS is set and is
    #     not altinstall
    # DEFAULT
    #   - The default behavior is to install pip, pipX, pipX.Y, easy_install
    #     and easy_install-X.Y.
    pip_script = console.pop('pip', None)
    if pip_script:
        if "ENSUREPIP_OPTIONS" not in os.environ:
            scripts_to_generate.append('pip = ' + pip_script)

        if os.environ.get("ENSUREPIP_OPTIONS", "") != "altinstall":
            scripts_to_generate.append(
                'pip{} = {}'.format(sys.version_info[0], pip_script)
            )

        scripts_to_generate.append(
            'pip{} = {}'.format(get_major_minor_version(), pip_script)
        )
        # Delete any other versioned pip entry points
        pip_ep = [k for k in console if re.match(r'pip(\d(\.\d)?)?$', k)]
        for k in pip_ep:
            del console[k]
    easy_install_script = console.pop('easy_install', None)
    if easy_install_script:
        if "ENSUREPIP_OPTIONS" not in os.environ:
            scripts_to_generate.append(
                'easy_install = ' + easy_install_script
            )

        scripts_to_generate.append(
            'easy_install-{} = {}'.format(
                get_major_minor_version(), easy_install_script
            )
        )
        # Delete any other versioned easy_install entry points
        easy_install_ep = [
            k for k in console if re.match(r'easy_install(-\d\.\d)?$', k)
        ]
        for k in easy_install_ep:
            del console[k]

    # Generate the console and GUI entry points specified in the wheel
    scripts_to_generate.extend(starmap('{} = {}'.format, console.items()))

    gui_scripts_to_generate = list(starmap('{} = {}'.format, gui.items()))

    generated_console_scripts = []  # type: List[str]

    try:
        generated_console_scripts = maker.make_multiple(scripts_to_generate)
        generated.extend(generated_console_scripts)

        generated.extend(
            maker.make_multiple(gui_scripts_to_generate, {'gui': True})
        )
    except MissingCallableSuffix as e:
        entry = e.args[0]
        raise InstallationError(
            "Invalid script entry point: {} for req: {} - A callable "
            "suffix is required. Cf https://packaging.python.org/"
            "specifications/entry-points/#use-for-scripts for more "
            "information.".format(entry, req_description)
        )

    if warn_script_location:
        msg = message_about_scripts_not_on_PATH(generated_console_scripts)
        if msg is not None:
            logger.warning(msg)

    generated_file_mode = 0o666 & ~current_umask()

    @contextlib.contextmanager
    def _generate_file(path, **kwargs):
        # type: (str, **Any) -> Iterator[NamedTemporaryFileResult]
        with adjacent_tmp_file(path, **kwargs) as f:
            yield f
        os.chmod(f.name, generated_file_mode)
        replace(f.name, path)

    # Record pip as the installer
    installer_path = os.path.join(dest_info_dir, 'INSTALLER')
    with _generate_file(installer_path) as installer_file:
        installer_file.write(b'pip\n')
    generated.append(installer_path)

    # Record the PEP 610 direct URL reference
    if direct_url is not None:
        direct_url_path = os.path.join(dest_info_dir, DIRECT_URL_METADATA_NAME)
        with _generate_file(direct_url_path) as direct_url_file:
            direct_url_file.write(direct_url.to_json().encode("utf-8"))
        generated.append(direct_url_path)

    # Record details of all files installed
    record_path = os.path.join(dest_info_dir, 'RECORD')
    with open(record_path, **csv_io_kwargs('r')) as record_file:
        rows = get_csv_rows_for_installed(
            csv.reader(record_file),
            installed=installed,
            changed=changed,
            generated=generated,
            lib_dir=lib_dir)
    with _generate_file(record_path, **csv_io_kwargs('w')) as record_file:
        writer = csv.writer(record_file)
        writer.writerows(sorted_outrows(rows))  # sort to simplify testing


def install_wheel(
    name,  # type: str
    wheel_path,  # type: str
    scheme,  # type: Scheme
    req_description,  # type: str
    pycompile=True,  # type: bool
    warn_script_location=True,  # type: bool
    _temp_dir_for_testing=None,  # type: Optional[str]
    direct_url=None,  # type: Optional[DirectUrl]
):
    # type: (...) -> None
    with TempDirectory(
        path=_temp_dir_for_testing, kind="unpacked-wheel"
    ) as unpacked_dir, ZipFile(wheel_path, allowZip64=True) as z:
        unpack_file(wheel_path, unpacked_dir.path)
        install_unpacked_wheel(
            name=name,
            wheeldir=unpacked_dir.path,
            wheel_zip=z,
            scheme=scheme,
            req_description=req_description,
            pycompile=pycompile,
            warn_script_location=warn_script_location,
            direct_url=direct_url,
        )
