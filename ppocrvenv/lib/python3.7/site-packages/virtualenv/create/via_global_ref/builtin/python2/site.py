# -*- coding: utf-8 -*-
"""
A simple shim module to fix up things on Python 2 only.

Note: until we setup correctly the paths we can only import built-ins.
"""
import sys


def main():
    """Patch what needed, and invoke the original site.py"""
    here = __file__  # the distutils.install patterns will be injected relative to this site.py, save it here
    config = read_pyvenv()
    sys.real_prefix = sys.base_prefix = config["base-prefix"]
    sys.base_exec_prefix = config["base-exec-prefix"]
    sys.base_executable = config["base-executable"]
    global_site_package_enabled = config.get("include-system-site-packages", False) == "true"
    rewrite_standard_library_sys_path()
    disable_user_site_package()
    load_host_site(here)
    if global_site_package_enabled:
        add_global_site_package()
    rewrite_getsitepackages(here)


def load_host_site(here):
    """trigger reload of site.py - now it will use the standard library instance that will take care of init"""
    # we have a duality here, we generate the platform and pure library path based on what distutils.install specifies
    # because this is what pip will be using; the host site.py though may contain it's own pattern for where the
    # platform and pure library paths should exist

    # notably on Ubuntu there's a patch for getsitepackages to point to
    # - prefix + local/lib/pythonx.y/dist-packages
    # - prefix + lib/pythonx.y/dist-packages
    # while distutils.install.cmd still points both of these to
    # - prefix + lib/python2.7/site-packages

    # to facilitate when the two match, or not we first reload the site.py, now triggering the import of host site.py,
    # as this will ensure that initialization code within host site.py runs

    # ___RELOAD_CODE___

    # and then if the distutils site packages are not on the sys.path we add them via add_site_dir; note we must add
    # them by invoking add_site_dir to trigger the processing of pth files

    add_site_dir = sys.modules["site"].addsitedir
    for path in get_site_packages_dirs(here):
        add_site_dir(path)


def get_site_packages_dirs(here):
    import json
    import os

    site_packages = r"""
    ___EXPECTED_SITE_PACKAGES___
    """

    for path in json.loads(site_packages):
        yield os.path.abspath(os.path.join(here, path.encode("utf-8")))


sep = "\\" if sys.platform == "win32" else "/"  # no os module here yet - poor mans version


def read_pyvenv():
    """read pyvenv.cfg"""
    config_file = "{}{}pyvenv.cfg".format(sys.prefix, sep)
    with open(config_file) as file_handler:
        lines = file_handler.readlines()
    config = {}
    for line in lines:
        try:
            split_at = line.index("=")
        except ValueError:
            continue  # ignore bad/empty lines
        else:
            config[line[:split_at].strip()] = line[split_at + 1 :].strip()
    return config


def rewrite_standard_library_sys_path():
    """Once this site file is loaded the standard library paths have already been set, fix them up"""
    exe, prefix, exec_prefix = get_exe_prefixes(base=False)
    base_exe, base_prefix, base_exec = get_exe_prefixes(base=True)
    exe_dir = exe[: exe.rfind(sep)]
    for at, path in enumerate(sys.path):
        path = abs_path(path)  # replace old sys prefix path starts with new
        skip_rewrite = path == exe_dir  # don't fix the current executable location, notably on Windows this gets added
        skip_rewrite = skip_rewrite  # ___SKIP_REWRITE____
        if not skip_rewrite:
            sys.path[at] = map_path(path, base_exe, exe_dir, exec_prefix, base_prefix, prefix, base_exec)

    # the rewrite above may have changed elements from PYTHONPATH, revert these if on
    if sys.flags.ignore_environment:
        return
    import os

    python_paths = []
    if "PYTHONPATH" in os.environ and os.environ["PYTHONPATH"]:
        for path in os.environ["PYTHONPATH"].split(os.pathsep):
            if path not in python_paths:
                python_paths.append(path)
    sys.path[: len(python_paths)] = python_paths


def get_exe_prefixes(base=False):
    return tuple(abs_path(getattr(sys, ("base_" if base else "") + i)) for i in ("executable", "prefix", "exec_prefix"))


def abs_path(value):
    values, keep = value.split(sep), []
    at = len(values) - 1
    while at >= 0:
        if values[at] == "..":
            at -= 1
        else:
            keep.append(values[at])
        at -= 1
    return sep.join(keep[::-1])


def map_path(path, base_executable, exe_dir, exec_prefix, base_prefix, prefix, base_exec_prefix):
    if path_starts_with(path, exe_dir):
        # content inside the exe folder needs to remap to original executables folder
        orig_exe_folder = base_executable[: base_executable.rfind(sep)]
        return "{}{}".format(orig_exe_folder, path[len(exe_dir) :])
    elif path_starts_with(path, prefix):
        return "{}{}".format(base_prefix, path[len(prefix) :])
    elif path_starts_with(path, exec_prefix):
        return "{}{}".format(base_exec_prefix, path[len(exec_prefix) :])
    return path


def path_starts_with(directory, value):
    return directory.startswith(value if value[-1] == sep else value + sep)


def disable_user_site_package():
    """Flip the switch on enable user site package"""
    # sys.flags is a c-extension type, so we cannot monkeypatch it, replace it with a python class to flip it
    sys.original_flags = sys.flags

    class Flags(object):
        def __init__(self):
            self.__dict__ = {key: getattr(sys.flags, key) for key in dir(sys.flags) if not key.startswith("_")}

    sys.flags = Flags()
    sys.flags.no_user_site = 1


def add_global_site_package():
    """add the global site package"""
    import site

    # add user site package
    sys.flags = sys.original_flags  # restore original
    site.ENABLE_USER_SITE = None  # reset user site check
    # add the global site package to the path - use new prefix and delegate to site.py
    orig_prefixes = None
    try:
        orig_prefixes = site.PREFIXES
        site.PREFIXES = [sys.base_prefix, sys.base_exec_prefix]
        site.main()
    finally:
        site.PREFIXES = orig_prefixes + site.PREFIXES


# Debian and it's derivatives patch this function. We undo the damage
def rewrite_getsitepackages(here):
    site = sys.modules["site"]

    site_package_dirs = get_site_packages_dirs(here)
    orig_getsitepackages = site.getsitepackages

    def getsitepackages():
        sitepackages = orig_getsitepackages()
        if sys.prefix not in site.PREFIXES or sys.exec_prefix not in site.PREFIXES:
            # Someone messed with the prefixes, so we stop patching
            return sitepackages
        for path in site_package_dirs:
            if path not in sitepackages:
                sitepackages.insert(0, path)

        return sitepackages

    site.getsitepackages = getsitepackages


main()
