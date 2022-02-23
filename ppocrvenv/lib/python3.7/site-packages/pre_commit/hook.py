import logging
import shlex
from typing import Any
from typing import Dict
from typing import NamedTuple
from typing import Sequence
from typing import Tuple

from pre_commit.prefix import Prefix

logger = logging.getLogger('pre_commit')


class Hook(NamedTuple):
    src: str
    prefix: Prefix
    id: str
    name: str
    entry: str
    language: str
    alias: str
    files: str
    exclude: str
    types: Sequence[str]
    types_or: Sequence[str]
    exclude_types: Sequence[str]
    additional_dependencies: Sequence[str]
    args: Sequence[str]
    always_run: bool
    fail_fast: bool
    pass_filenames: bool
    description: str
    language_version: str
    log_file: str
    minimum_pre_commit_version: str
    require_serial: bool
    stages: Sequence[str]
    verbose: bool

    @property
    def cmd(self) -> Tuple[str, ...]:
        return (*shlex.split(self.entry), *self.args)

    @property
    def install_key(self) -> Tuple[Prefix, str, str, Tuple[str, ...]]:
        return (
            self.prefix,
            self.language,
            self.language_version,
            tuple(self.additional_dependencies),
        )

    @classmethod
    def create(cls, src: str, prefix: Prefix, dct: Dict[str, Any]) -> 'Hook':
        # TODO: have cfgv do this (?)
        extra_keys = set(dct) - _KEYS
        if extra_keys:
            logger.warning(
                f'Unexpected key(s) present on {src} => {dct["id"]}: '
                f'{", ".join(sorted(extra_keys))}',
            )
        return cls(src=src, prefix=prefix, **{k: dct[k] for k in _KEYS})


_KEYS = frozenset(set(Hook._fields) - {'src', 'prefix'})
