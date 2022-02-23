import re
import textwrap

import yaml

from pre_commit.util import yaml_load


def _is_header_line(line: str) -> bool:
    return line.startswith(('#', '---')) or not line.strip()


def _migrate_map(contents: str) -> str:
    if isinstance(yaml_load(contents), list):
        # Find the first non-header line
        lines = contents.splitlines(True)
        i = 0
        # Only loop on non empty configuration file
        while i < len(lines) and _is_header_line(lines[i]):
            i += 1

        header = ''.join(lines[:i])
        rest = ''.join(lines[i:])

        # If they are using the "default" flow style of yaml, this operation
        # will yield a valid configuration
        try:
            trial_contents = f'{header}repos:\n{rest}'
            yaml_load(trial_contents)
            contents = trial_contents
        except yaml.YAMLError:
            contents = f'{header}repos:\n{textwrap.indent(rest, " " * 4)}'

    return contents


def _migrate_sha_to_rev(contents: str) -> str:
    return re.sub(r'(\n\s+)sha:', r'\1rev:', contents)


def migrate_config(config_file: str, quiet: bool = False) -> int:
    with open(config_file) as f:
        orig_contents = contents = f.read()

    contents = _migrate_map(contents)
    contents = _migrate_sha_to_rev(contents)

    if contents != orig_contents:
        with open(config_file, 'w') as f:
            f.write(contents)

        print('Configuration has been migrated.')
    elif not quiet:
        print('Configuration is already migrated.')
    return 0
