"""Implement https://www.python.org/dev/peps/pep-0514/ to discover interpreters - Windows only"""
from __future__ import absolute_import, print_function, unicode_literals

import os
import re
from logging import basicConfig, getLogger

import six

if six.PY3:
    import winreg
else:
    # noinspection PyUnresolvedReferences
    import _winreg as winreg

LOGGER = getLogger(__name__)


def enum_keys(key):
    at = 0
    while True:
        try:
            yield winreg.EnumKey(key, at)
        except OSError:
            break
        at += 1


def get_value(key, value_name):
    try:
        return winreg.QueryValueEx(key, value_name)[0]
    except OSError:
        return None


def discover_pythons():
    for hive, hive_name, key, flags, default_arch in [
        (winreg.HKEY_CURRENT_USER, "HKEY_CURRENT_USER", r"Software\Python", 0, 64),
        (winreg.HKEY_LOCAL_MACHINE, "HKEY_LOCAL_MACHINE", r"Software\Python", winreg.KEY_WOW64_64KEY, 64),
        (winreg.HKEY_LOCAL_MACHINE, "HKEY_LOCAL_MACHINE", r"Software\Python", winreg.KEY_WOW64_32KEY, 32),
    ]:
        for spec in process_set(hive, hive_name, key, flags, default_arch):
            yield spec


def process_set(hive, hive_name, key, flags, default_arch):
    try:
        with winreg.OpenKeyEx(hive, key, 0, winreg.KEY_READ | flags) as root_key:
            for company in enum_keys(root_key):
                if company == "PyLauncher":  # reserved
                    continue
                for spec in process_company(hive_name, company, root_key, default_arch):
                    yield spec
    except OSError:
        pass


def process_company(hive_name, company, root_key, default_arch):
    with winreg.OpenKeyEx(root_key, company) as company_key:
        for tag in enum_keys(company_key):
            spec = process_tag(hive_name, company, company_key, tag, default_arch)
            if spec is not None:
                yield spec


def process_tag(hive_name, company, company_key, tag, default_arch):
    with winreg.OpenKeyEx(company_key, tag) as tag_key:
        version = load_version_data(hive_name, company, tag, tag_key)
        if version is not None:  # if failed to get version bail
            major, minor, _ = version
            arch = load_arch_data(hive_name, company, tag, tag_key, default_arch)
            if arch is not None:
                exe_data = load_exe(hive_name, company, company_key, tag)
                if exe_data is not None:
                    exe, args = exe_data
                    return company, major, minor, arch, exe, args


def load_exe(hive_name, company, company_key, tag):
    key_path = "{}/{}/{}".format(hive_name, company, tag)
    try:
        with winreg.OpenKeyEx(company_key, r"{}\InstallPath".format(tag)) as ip_key:
            with ip_key:
                exe = get_value(ip_key, "ExecutablePath")
                if exe is None:
                    ip = get_value(ip_key, None)
                    if ip is None:
                        msg(key_path, "no ExecutablePath or default for it")

                    else:
                        exe = os.path.join(ip, str("python.exe"))
                if exe is not None and os.path.exists(exe):
                    args = get_value(ip_key, "ExecutableArguments")
                    return exe, args
                else:
                    msg(key_path, "could not load exe with value {}".format(exe))
    except OSError:
        msg("{}/{}".format(key_path, "InstallPath"), "missing")
    return None


def load_arch_data(hive_name, company, tag, tag_key, default_arch):
    arch_str = get_value(tag_key, "SysArchitecture")
    if arch_str is not None:
        key_path = "{}/{}/{}/SysArchitecture".format(hive_name, company, tag)
        try:
            return parse_arch(arch_str)
        except ValueError as sys_arch:
            msg(key_path, sys_arch)
    return default_arch


def parse_arch(arch_str):
    if isinstance(arch_str, six.string_types):
        match = re.match(r"^(\d+)bit$", arch_str)
        if match:
            return int(next(iter(match.groups())))
        error = "invalid format {}".format(arch_str)
    else:
        error = "arch is not string: {}".format(repr(arch_str))
    raise ValueError(error)


def load_version_data(hive_name, company, tag, tag_key):
    for candidate, key_path in [
        (get_value(tag_key, "SysVersion"), "{}/{}/{}/SysVersion".format(hive_name, company, tag)),
        (tag, "{}/{}/{}".format(hive_name, company, tag)),
    ]:
        if candidate is not None:
            try:
                return parse_version(candidate)
            except ValueError as sys_version:
                msg(key_path, sys_version)
    return None


def parse_version(version_str):
    if isinstance(version_str, six.string_types):
        match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?$", version_str)
        if match:
            return tuple(int(i) if i is not None else None for i in match.groups())
        error = "invalid format {}".format(version_str)
    else:
        error = "version is not string: {}".format(repr(version_str))
    raise ValueError(error)


def msg(path, what):
    LOGGER.warning("PEP-514 violation in Windows Registry at {} error: {}".format(path, what))


def _run():
    basicConfig()
    interpreters = []
    for spec in discover_pythons():
        interpreters.append(repr(spec))
    print("\n".join(sorted(interpreters)))


if __name__ == "__main__":
    _run()
