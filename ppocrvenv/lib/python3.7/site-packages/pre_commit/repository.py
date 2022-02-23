import json
import logging
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple

import pre_commit.constants as C
from pre_commit.clientlib import load_manifest
from pre_commit.clientlib import LOCAL
from pre_commit.clientlib import META
from pre_commit.hook import Hook
from pre_commit.languages.all import languages
from pre_commit.languages.helpers import environment_dir
from pre_commit.prefix import Prefix
from pre_commit.store import Store
from pre_commit.util import parse_version
from pre_commit.util import rmtree


logger = logging.getLogger('pre_commit')


def _state(additional_deps: Sequence[str]) -> object:
    return {'additional_dependencies': sorted(additional_deps)}


def _state_filename(prefix: Prefix, venv: str) -> str:
    return prefix.path(venv, f'.install_state_v{C.INSTALLED_STATE_VERSION}')


def _read_state(prefix: Prefix, venv: str) -> Optional[object]:
    filename = _state_filename(prefix, venv)
    if not os.path.exists(filename):
        return None
    else:
        with open(filename) as f:
            return json.load(f)


def _write_state(prefix: Prefix, venv: str, state: object) -> None:
    state_filename = _state_filename(prefix, venv)
    staging = f'{state_filename}staging'
    with open(staging, 'w') as state_file:
        state_file.write(json.dumps(state))
    # Move the file into place atomically to indicate we've installed
    os.replace(staging, state_filename)


def _hook_installed(hook: Hook) -> bool:
    lang = languages[hook.language]
    venv = environment_dir(lang.ENVIRONMENT_DIR, hook.language_version)
    return (
        venv is None or (
            (
                _read_state(hook.prefix, venv) ==
                _state(hook.additional_dependencies)
            ) and
            lang.healthy(hook.prefix, hook.language_version)
        )
    )


def _hook_install(hook: Hook) -> None:
    logger.info(f'Installing environment for {hook.src}.')
    logger.info('Once installed this environment will be reused.')
    logger.info('This may take a few minutes...')

    lang = languages[hook.language]
    assert lang.ENVIRONMENT_DIR is not None
    venv = environment_dir(lang.ENVIRONMENT_DIR, hook.language_version)

    # There's potentially incomplete cleanup from previous runs
    # Clean it up!
    if hook.prefix.exists(venv):
        rmtree(hook.prefix.path(venv))

    lang.install_environment(
        hook.prefix, hook.language_version, hook.additional_dependencies,
    )
    if not lang.healthy(hook.prefix, hook.language_version):
        raise AssertionError(
            f'BUG: expected environment for {hook.language} to be healthy() '
            f'immediately after install, please open an issue describing '
            f'your environment',
        )
    # Write our state to indicate we're installed
    _write_state(hook.prefix, venv, _state(hook.additional_dependencies))


def _hook(
        *hook_dicts: Dict[str, Any],
        root_config: Dict[str, Any],
) -> Dict[str, Any]:
    ret, rest = dict(hook_dicts[0]), hook_dicts[1:]
    for dct in rest:
        ret.update(dct)

    version = ret['minimum_pre_commit_version']
    if parse_version(version) > parse_version(C.VERSION):
        logger.error(
            f'The hook `{ret["id"]}` requires pre-commit version {version} '
            f'but version {C.VERSION} is installed.  '
            f'Perhaps run `pip install --upgrade pre-commit`.',
        )
        exit(1)

    lang = ret['language']
    if ret['language_version'] == C.DEFAULT:
        ret['language_version'] = root_config['default_language_version'][lang]
    if ret['language_version'] == C.DEFAULT:
        ret['language_version'] = languages[lang].get_default_version()

    if not ret['stages']:
        ret['stages'] = root_config['default_stages']

    if languages[lang].ENVIRONMENT_DIR is None:
        if ret['language_version'] != C.DEFAULT:
            logger.error(
                f'The hook `{ret["id"]}` specifies `language_version` but is '
                f'using language `{lang}` which does not install an '
                f'environment.  '
                f'Perhaps you meant to use a specific language?',
            )
            exit(1)
        if ret['additional_dependencies']:
            logger.error(
                f'The hook `{ret["id"]}` specifies `additional_dependencies` '
                f'but is using language `{lang}` which does not install an '
                f'environment.  '
                f'Perhaps you meant to use a specific language?',
            )
            exit(1)

    return ret


def _non_cloned_repository_hooks(
        repo_config: Dict[str, Any],
        store: Store,
        root_config: Dict[str, Any],
) -> Tuple[Hook, ...]:
    def _prefix(language_name: str, deps: Sequence[str]) -> Prefix:
        language = languages[language_name]
        # pygrep / script / system / docker_image do not have
        # environments so they work out of the current directory
        if language.ENVIRONMENT_DIR is None:
            return Prefix(os.getcwd())
        else:
            return Prefix(store.make_local(deps))

    return tuple(
        Hook.create(
            repo_config['repo'],
            _prefix(hook['language'], hook['additional_dependencies']),
            _hook(hook, root_config=root_config),
        )
        for hook in repo_config['hooks']
    )


def _cloned_repository_hooks(
        repo_config: Dict[str, Any],
        store: Store,
        root_config: Dict[str, Any],
) -> Tuple[Hook, ...]:
    repo, rev = repo_config['repo'], repo_config['rev']
    manifest_path = os.path.join(store.clone(repo, rev), C.MANIFEST_FILE)
    by_id = {hook['id']: hook for hook in load_manifest(manifest_path)}

    for hook in repo_config['hooks']:
        if hook['id'] not in by_id:
            logger.error(
                f'`{hook["id"]}` is not present in repository {repo}.  '
                f'Typo? Perhaps it is introduced in a newer version?  '
                f'Often `pre-commit autoupdate` fixes this.',
            )
            exit(1)

    hook_dcts = [
        _hook(by_id[hook['id']], hook, root_config=root_config)
        for hook in repo_config['hooks']
    ]
    return tuple(
        Hook.create(
            repo_config['repo'],
            Prefix(store.clone(repo, rev, hook['additional_dependencies'])),
            hook,
        )
        for hook in hook_dcts
    )


def _repository_hooks(
        repo_config: Dict[str, Any],
        store: Store,
        root_config: Dict[str, Any],
) -> Tuple[Hook, ...]:
    if repo_config['repo'] in {LOCAL, META}:
        return _non_cloned_repository_hooks(repo_config, store, root_config)
    else:
        return _cloned_repository_hooks(repo_config, store, root_config)


def install_hook_envs(hooks: Sequence[Hook], store: Store) -> None:
    def _need_installed() -> List[Hook]:
        seen: Set[Tuple[Prefix, str, str, Tuple[str, ...]]] = set()
        ret = []
        for hook in hooks:
            if hook.install_key not in seen and not _hook_installed(hook):
                ret.append(hook)
            seen.add(hook.install_key)
        return ret

    if not _need_installed():
        return
    with store.exclusive_lock():
        # Another process may have already completed this work
        for hook in _need_installed():
            _hook_install(hook)


def all_hooks(root_config: Dict[str, Any], store: Store) -> Tuple[Hook, ...]:
    return tuple(
        hook
        for repo in root_config['repos']
        for hook in _repository_hooks(repo, store, root_config)
    )
