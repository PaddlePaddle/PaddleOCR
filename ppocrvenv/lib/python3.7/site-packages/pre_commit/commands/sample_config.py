# TODO: maybe `git ls-remote git://github.com/pre-commit/pre-commit-hooks` to
# determine the latest revision?  This adds ~200ms from my tests (and is
# significantly faster than https:// or http://).  For now, periodically
# manually updating the revision is fine.
SAMPLE_CONFIG = '''\
# See https://pre-commit.com for more information
# See https://pre-commit.com/hooks.html for more hooks
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v3.2.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
'''


def sample_config() -> int:
    print(SAMPLE_CONFIG, end='')
    return 0
