# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys
import shutil
import zipfile
from paddle.utils.download import get_path_from_url

__all__ = []

DEFAULT_CACHE_DIR = '~/.cache'
VAR_DEPENDENCY = 'dependencies'
MODULE_HUBCONF = 'hubconf.py'
HUB_DIR = os.path.expanduser(os.path.join('~', '.cache', 'paddle', 'hub'))


def _remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def _import_module(name, repo_dir):
    sys.path.insert(0, repo_dir)
    try:
        hub_module = __import__(name)
        sys.modules.pop(name)
    except ImportError:
        sys.path.remove(repo_dir)
        raise RuntimeError(
            'Please make sure config exists or repo error messages above fixed when importing'
        )

    sys.path.remove(repo_dir)

    return hub_module


def _git_archive_link(repo_owner, repo_name, branch, source):
    if source == 'github':
        return 'https://github.com/{}/{}/archive/{}.zip'.format(
            repo_owner, repo_name, branch)
    elif source == 'gitee':
        return 'https://gitee.com/{}/{}/repository/archive/{}.zip'.format(
            repo_owner, repo_name, branch)


def _parse_repo_info(repo, source):
    branch = 'main' if source == 'github' else 'master'
    if ':' in repo:
        repo_info, branch = repo.split(':')
    else:
        repo_info = repo
    repo_owner, repo_name = repo_info.split('/')
    return repo_owner, repo_name, branch


def _make_dirs(dirname):
    try:
        from pathlib import Path
    except ImportError:
        from pathlib2 import Path
    Path(dirname).mkdir(exist_ok=True)


def _get_cache_or_reload(repo, force_reload, verbose=True, source='github'):
    # Setup hub_dir to save downloaded files
    hub_dir = HUB_DIR

    _make_dirs(hub_dir)

    # Parse github/gitee repo information
    repo_owner, repo_name, branch = _parse_repo_info(repo, source)
    # Github allows branch name with slash '/',
    # this causes confusion with path on both Linux and Windows.
    # Backslash is not allowed in Github branch name so no need to
    # to worry about it.
    normalized_br = branch.replace('/', '_')
    # Github renames folder repo/v1.x.x to repo-1.x.x
    # We don't know the repo name before downloading the zip file
    # and inspect name from it.
    # To check if cached repo exists, we need to normalize folder names.
    repo_dir = os.path.join(hub_dir,
                            '_'.join([repo_owner, repo_name, normalized_br]))

    use_cache = (not force_reload) and os.path.exists(repo_dir)

    if use_cache:
        if verbose:
            sys.stderr.write('Using cache found in {}\n'.format(repo_dir))
    else:
        cached_file = os.path.join(hub_dir, normalized_br + '.zip')
        _remove_if_exists(cached_file)

        url = _git_archive_link(repo_owner, repo_name, branch, source=source)

        fpath = get_path_from_url(
            url,
            hub_dir,
            check_exist=not force_reload,
            decompress=False,
            method=('wget' if source == 'gitee' else 'get'))
        shutil.move(fpath, cached_file)

        with zipfile.ZipFile(cached_file) as cached_zipfile:
            extraced_repo_name = cached_zipfile.infolist()[0].filename
            extracted_repo = os.path.join(hub_dir, extraced_repo_name)
            _remove_if_exists(extracted_repo)
            # Unzip the code and rename the base folder
            cached_zipfile.extractall(hub_dir)

        _remove_if_exists(cached_file)
        _remove_if_exists(repo_dir)
        # rename the repo
        shutil.move(extracted_repo, repo_dir)

    return repo_dir


def _load_entry_from_hubconf(m, name):
    '''load entry from hubconf
    '''
    if not isinstance(name, str):
        raise ValueError(
            'Invalid input: model should be a str of function name')

    func = getattr(m, name, None)

    if func is None or not callable(func):
        raise RuntimeError('Cannot find callable {} in hubconf'.format(name))

    return func


def _check_module_exists(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _check_dependencies(m):
    dependencies = getattr(m, VAR_DEPENDENCY, None)

    if dependencies is not None:
        missing_deps = [
            pkg for pkg in dependencies if not _check_module_exists(pkg)
        ]
        if len(missing_deps):
            raise RuntimeError('Missing dependencies: {}'.format(', '.join(
                missing_deps)))


def list(repo_dir, source='github', force_reload=False):
    r"""
    List all entrypoints available in `github` hubconf.

    Args:
        repo_dir(str): github or local path.

            github path (str): a str with format "repo_owner/repo_name[:tag_name]" with an optional
            tag/branch. The default branch is `main` if not specified.
            
            local path (str): local repo path
        
        source (str): `github` | `gitee` | `local`, default is `github`.
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download, default is `False`.
    Returns:
        entrypoints: a list of available entrypoint names

    Example:
        .. code-block:: python

            import paddle

            paddle.hub.list('lyuwenyu/paddlehub_demo:main', source='github', force_reload=False)

    """
    if source not in ('github', 'gitee', 'local'):
        raise ValueError(
            'Unknown source: "{}". Allowed values: "github" | "gitee" | "local".'.
            format(source))

    if source in ('github', 'gitee'):
        repo_dir = _get_cache_or_reload(
            repo_dir, force_reload, True, source=source)

    hub_module = _import_module(MODULE_HUBCONF.split('.')[0], repo_dir)

    entrypoints = [
        f for f in dir(hub_module)
        if callable(getattr(hub_module, f)) and not f.startswith('_')
    ]

    return entrypoints


def help(repo_dir, model, source='github', force_reload=False):
    """
    Show help information of model

    Args:
        repo_dir(str): github or local path.

            github path (str): a str with format "repo_owner/repo_name[:tag_name]" with an optional
            tag/branch. The default branch is `main` if not specified.
            
            local path (str): local repo path.
        
        model (str): model name.
        source (str): `github` | `gitee` | `local`, default is `github`.
        force_reload (bool, optional): default is `False`.
    Return:
        docs

    Example:
        .. code-block:: python

            import paddle

            paddle.hub.help('lyuwenyu/paddlehub_demo:main', model='MM', source='github')

    """
    if source not in ('github', 'gitee', 'local'):
        raise ValueError(
            'Unknown source: "{}". Allowed values: "github" | "gitee" | "local".'.
            format(source))

    if source in ('github', 'gitee'):
        repo_dir = _get_cache_or_reload(
            repo_dir, force_reload, True, source=source)

    hub_module = _import_module(MODULE_HUBCONF.split('.')[0], repo_dir)

    entry = _load_entry_from_hubconf(hub_module, model)

    return entry.__doc__


def load(repo_dir, model, source='github', force_reload=False, **kwargs):
    """
    Load model

    Args:
        repo_dir(str): github or local path.

            github path (str): a str with format "repo_owner/repo_name[:tag_name]" with an optional
            tag/branch. The default branch is `main` if not specified.

            local path (str): local repo path.
        
        model (str): model name.
        source (str): `github` | `gitee` | `local`, default is `github`.
        force_reload (bool, optional): default is `False`.
        **kwargs: parameters using for model
    Return:
        paddle model
    Example:
        .. code-block:: python

            import paddle
            paddle.hub.load('lyuwenyu/paddlehub_demo:main', model='MM', source='github')

    """
    if source not in ('github', 'gitee', 'local'):
        raise ValueError(
            'Unknown source: "{}". Allowed values: "github" | "gitee" | "local".'.
            format(source))

    if source in ('github', 'gitee'):
        repo_dir = _get_cache_or_reload(
            repo_dir, force_reload, True, source=source)

    hub_module = _import_module(MODULE_HUBCONF.split('.')[0], repo_dir)

    _check_dependencies(hub_module)

    entry = _load_entry_from_hubconf(hub_module, model)

    return entry(**kwargs)
