import argparse
import functools
import logging
import re
import shlex
import sys
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

import cfgv
from identify.identify import ALL_TAGS

import pre_commit.constants as C
from pre_commit.color import add_color_option
from pre_commit.errors import FatalError
from pre_commit.languages.all import all_languages
from pre_commit.logging_handler import logging_handler
from pre_commit.util import parse_version
from pre_commit.util import yaml_load

logger = logging.getLogger('pre_commit')

check_string_regex = cfgv.check_and(cfgv.check_string, cfgv.check_regex)


def check_type_tag(tag: str) -> None:
    if tag not in ALL_TAGS:
        raise cfgv.ValidationError(
            f'Type tag {tag!r} is not recognized.  '
            f'Try upgrading identify and pre-commit?',
        )


def check_min_version(version: str) -> None:
    if parse_version(version) > parse_version(C.VERSION):
        raise cfgv.ValidationError(
            f'pre-commit version {version} is required but version '
            f'{C.VERSION} is installed.  '
            f'Perhaps run `pip install --upgrade pre-commit`.',
        )


def _make_argparser(filenames_help: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*', help=filenames_help)
    parser.add_argument('-V', '--version', action='version', version=C.VERSION)
    add_color_option(parser)
    return parser


MANIFEST_HOOK_DICT = cfgv.Map(
    'Hook', 'id',

    cfgv.Required('id', cfgv.check_string),
    cfgv.Required('name', cfgv.check_string),
    cfgv.Required('entry', cfgv.check_string),
    cfgv.Required('language', cfgv.check_one_of(all_languages)),
    cfgv.Optional('alias', cfgv.check_string, ''),

    cfgv.Optional('files', check_string_regex, ''),
    cfgv.Optional('exclude', check_string_regex, '^$'),
    cfgv.Optional('types', cfgv.check_array(check_type_tag), ['file']),
    cfgv.Optional('types_or', cfgv.check_array(check_type_tag), []),
    cfgv.Optional('exclude_types', cfgv.check_array(check_type_tag), []),

    cfgv.Optional(
        'additional_dependencies', cfgv.check_array(cfgv.check_string), [],
    ),
    cfgv.Optional('args', cfgv.check_array(cfgv.check_string), []),
    cfgv.Optional('always_run', cfgv.check_bool, False),
    cfgv.Optional('fail_fast', cfgv.check_bool, False),
    cfgv.Optional('pass_filenames', cfgv.check_bool, True),
    cfgv.Optional('description', cfgv.check_string, ''),
    cfgv.Optional('language_version', cfgv.check_string, C.DEFAULT),
    cfgv.Optional('log_file', cfgv.check_string, ''),
    cfgv.Optional('minimum_pre_commit_version', cfgv.check_string, '0'),
    cfgv.Optional('require_serial', cfgv.check_bool, False),
    cfgv.Optional('stages', cfgv.check_array(cfgv.check_one_of(C.STAGES)), []),
    cfgv.Optional('verbose', cfgv.check_bool, False),
)
MANIFEST_SCHEMA = cfgv.Array(MANIFEST_HOOK_DICT)


class InvalidManifestError(FatalError):
    pass


load_manifest = functools.partial(
    cfgv.load_from_filename,
    schema=MANIFEST_SCHEMA,
    load_strategy=yaml_load,
    exc_tp=InvalidManifestError,
)


def validate_manifest_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _make_argparser('Manifest filenames.')
    args = parser.parse_args(argv)

    with logging_handler(args.color):
        ret = 0
        for filename in args.filenames:
            try:
                load_manifest(filename)
            except InvalidManifestError as e:
                print(e)
                ret = 1
        return ret


LOCAL = 'local'
META = 'meta'


# should inherit from cfgv.Conditional if sha support is dropped
class WarnMutableRev(cfgv.ConditionalOptional):
    def check(self, dct: Dict[str, Any]) -> None:
        super().check(dct)

        if self.key in dct:
            rev = dct[self.key]

            if '.' not in rev and not re.match(r'^[a-fA-F0-9]+$', rev):
                logger.warning(
                    f'The {self.key!r} field of repo {dct["repo"]!r} '
                    f'appears to be a mutable reference '
                    f'(moving tag / branch).  Mutable references are never '
                    f'updated after first install and are not supported.  '
                    f'See https://pre-commit.com/#using-the-latest-version-for-a-repository '  # noqa: E501
                    f'for more details.  '
                    f'Hint: `pre-commit autoupdate` often fixes this.',
                )


class OptionalSensibleRegexAtHook(cfgv.OptionalNoDefault):
    def check(self, dct: Dict[str, Any]) -> None:
        super().check(dct)

        if '/*' in dct.get(self.key, ''):
            logger.warning(
                f'The {self.key!r} field in hook {dct.get("id")!r} is a '
                f"regex, not a glob -- matching '/*' probably isn't what you "
                f'want here',
            )
        for fwd_slash_re in (r'[\\/]', r'[\/]', r'[/\\]'):
            if fwd_slash_re in dct.get(self.key, ''):
                logger.warning(
                    fr'pre-commit normalizes slashes in the {self.key!r} '
                    fr'field in hook {dct.get("id")!r} to forward slashes, '
                    fr'so you can use / instead of {fwd_slash_re}',
                )


class OptionalSensibleRegexAtTop(cfgv.OptionalNoDefault):
    def check(self, dct: Dict[str, Any]) -> None:
        super().check(dct)

        if '/*' in dct.get(self.key, ''):
            logger.warning(
                f'The top-level {self.key!r} field is a regex, not a glob -- '
                f"matching '/*' probably isn't what you want here",
            )
        for fwd_slash_re in (r'[\\/]', r'[\/]', r'[/\\]'):
            if fwd_slash_re in dct.get(self.key, ''):
                logger.warning(
                    fr'pre-commit normalizes the slashes in the top-level '
                    fr'{self.key!r} field to forward slashes, so you '
                    fr'can use / instead of {fwd_slash_re}',
                )


class MigrateShaToRev:
    key = 'rev'

    @staticmethod
    def _cond(key: str) -> cfgv.Conditional:
        return cfgv.Conditional(
            key, cfgv.check_string,
            condition_key='repo',
            condition_value=cfgv.NotIn(LOCAL, META),
            ensure_absent=True,
        )

    def check(self, dct: Dict[str, Any]) -> None:
        if dct.get('repo') in {LOCAL, META}:
            self._cond('rev').check(dct)
            self._cond('sha').check(dct)
        elif 'sha' in dct and 'rev' in dct:
            raise cfgv.ValidationError('Cannot specify both sha and rev')
        elif 'sha' in dct:
            self._cond('sha').check(dct)
        else:
            self._cond('rev').check(dct)

    def apply_default(self, dct: Dict[str, Any]) -> None:
        if 'sha' in dct:
            dct['rev'] = dct.pop('sha')

    remove_default = cfgv.Required.remove_default


def _entry(modname: str) -> str:
    """the hook `entry` is passed through `shlex.split()` by the command
    runner, so to prevent issues with spaces and backslashes (on Windows)
    it must be quoted here.
    """
    return f'{shlex.quote(sys.executable)} -m pre_commit.meta_hooks.{modname}'


def warn_unknown_keys_root(
        extra: Sequence[str],
        orig_keys: Sequence[str],
        dct: Dict[str, str],
) -> None:
    logger.warning(f'Unexpected key(s) present at root: {", ".join(extra)}')


def warn_unknown_keys_repo(
        extra: Sequence[str],
        orig_keys: Sequence[str],
        dct: Dict[str, str],
) -> None:
    logger.warning(
        f'Unexpected key(s) present on {dct["repo"]}: {", ".join(extra)}',
    )


_meta = (
    (
        'check-hooks-apply', (
            ('name', 'Check hooks apply to the repository'),
            ('files', f'^{re.escape(C.CONFIG_FILE)}$'),
            ('entry', _entry('check_hooks_apply')),
        ),
    ),
    (
        'check-useless-excludes', (
            ('name', 'Check for useless excludes'),
            ('files', f'^{re.escape(C.CONFIG_FILE)}$'),
            ('entry', _entry('check_useless_excludes')),
        ),
    ),
    (
        'identity', (
            ('name', 'identity'),
            ('verbose', True),
            ('entry', _entry('identity')),
        ),
    ),
)


class NotAllowed(cfgv.OptionalNoDefault):
    def check(self, dct: Dict[str, Any]) -> None:
        if self.key in dct:
            raise cfgv.ValidationError(f'{self.key!r} cannot be overridden')


META_HOOK_DICT = cfgv.Map(
    'Hook', 'id',
    cfgv.Required('id', cfgv.check_string),
    cfgv.Required('id', cfgv.check_one_of(tuple(k for k, _ in _meta))),
    # language must be system
    cfgv.Optional('language', cfgv.check_one_of({'system'}), 'system'),
    # entry cannot be overridden
    NotAllowed('entry', cfgv.check_any),
    *(
        # default to the hook definition for the meta hooks
        cfgv.ConditionalOptional(key, cfgv.check_any, value, 'id', hook_id)
        for hook_id, values in _meta
        for key, value in values
    ),
    *(
        # default to the "manifest" parsing
        cfgv.OptionalNoDefault(item.key, item.check_fn)
        # these will always be defaulted above
        if item.key in {'name', 'language', 'entry'} else
        item
        for item in MANIFEST_HOOK_DICT.items
    ),
)
CONFIG_HOOK_DICT = cfgv.Map(
    'Hook', 'id',

    cfgv.Required('id', cfgv.check_string),

    # All keys in manifest hook dict are valid in a config hook dict, but
    # are optional.
    # No defaults are provided here as the config is merged on top of the
    # manifest.
    *(
        cfgv.OptionalNoDefault(item.key, item.check_fn)
        for item in MANIFEST_HOOK_DICT.items
        if item.key != 'id'
    ),
    OptionalSensibleRegexAtHook('files', cfgv.check_string),
    OptionalSensibleRegexAtHook('exclude', cfgv.check_string),
)
CONFIG_REPO_DICT = cfgv.Map(
    'Repository', 'repo',

    cfgv.Required('repo', cfgv.check_string),

    cfgv.ConditionalRecurse(
        'hooks', cfgv.Array(CONFIG_HOOK_DICT),
        'repo', cfgv.NotIn(LOCAL, META),
    ),
    cfgv.ConditionalRecurse(
        'hooks', cfgv.Array(MANIFEST_HOOK_DICT),
        'repo', LOCAL,
    ),
    cfgv.ConditionalRecurse(
        'hooks', cfgv.Array(META_HOOK_DICT),
        'repo', META,
    ),

    MigrateShaToRev(),
    WarnMutableRev(
        'rev',
        cfgv.check_string,
        '',
        'repo',
        cfgv.NotIn(LOCAL, META),
        True,
    ),
    cfgv.WarnAdditionalKeys(('repo', 'rev', 'hooks'), warn_unknown_keys_repo),
)
DEFAULT_LANGUAGE_VERSION = cfgv.Map(
    'DefaultLanguageVersion', None,
    cfgv.NoAdditionalKeys(all_languages),
    *(cfgv.Optional(x, cfgv.check_string, C.DEFAULT) for x in all_languages),
)
CONFIG_SCHEMA = cfgv.Map(
    'Config', None,

    cfgv.RequiredRecurse('repos', cfgv.Array(CONFIG_REPO_DICT)),
    cfgv.OptionalRecurse(
        'default_language_version', DEFAULT_LANGUAGE_VERSION, {},
    ),
    cfgv.Optional(
        'default_stages',
        cfgv.check_array(cfgv.check_one_of(C.STAGES)),
        C.STAGES,
    ),
    cfgv.Optional('files', check_string_regex, ''),
    cfgv.Optional('exclude', check_string_regex, '^$'),
    cfgv.Optional('fail_fast', cfgv.check_bool, False),
    cfgv.Optional(
        'minimum_pre_commit_version',
        cfgv.check_and(cfgv.check_string, check_min_version),
        '0',
    ),
    cfgv.WarnAdditionalKeys(
        (
            'repos',
            'default_language_version',
            'default_stages',
            'files',
            'exclude',
            'fail_fast',
            'minimum_pre_commit_version',
            'ci',
        ),
        warn_unknown_keys_root,
    ),
    OptionalSensibleRegexAtTop('files', cfgv.check_string),
    OptionalSensibleRegexAtTop('exclude', cfgv.check_string),

    # do not warn about configuration for pre-commit.ci
    cfgv.OptionalNoDefault('ci', cfgv.check_type(dict)),
)


class InvalidConfigError(FatalError):
    pass


def ordered_load_normalize_legacy_config(contents: str) -> Dict[str, Any]:
    data = yaml_load(contents)
    if isinstance(data, list):
        logger.warning(
            'normalizing pre-commit configuration to a top-level map.  '
            'support for top level list will be removed in a future version.  '
            'run: `pre-commit migrate-config` to automatically fix this.',
        )
        return {'repos': data}
    else:
        return data


load_config = functools.partial(
    cfgv.load_from_filename,
    schema=CONFIG_SCHEMA,
    load_strategy=ordered_load_normalize_legacy_config,
    exc_tp=InvalidConfigError,
)


def validate_config_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _make_argparser('Config filenames.')
    args = parser.parse_args(argv)

    with logging_handler(args.color):
        ret = 0
        for filename in args.filenames:
            try:
                load_config(filename)
            except InvalidConfigError as e:
                print(e)
                ret = 1
        return ret
