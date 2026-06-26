#!/usr/bin/env bash
# Fixture-based test for the docs-only matcher.
# Extracts the matcher function from action.yml and runs it against fixtures.
set -euo pipefail

ACTION_FILE="$(dirname "$0")/../action.yml"

# Extract the matcher script block delimited by sentinel comments.
matcher_script=$(awk '/# >>> matcher-begin/{flag=1;next} /# <<< matcher-end/{flag=0} flag' "$ACTION_FILE")

if [ -z "$matcher_script" ]; then
  echo "FAIL: could not extract matcher script from $ACTION_FILE"
  exit 1
fi

run_case() {
  local name="$1"
  local expected="$2"
  shift 2
  local changed_files
  changed_files=$(printf '%s\n' "$@")
  local actual
  actual=$(CHANGED_FILES="$changed_files" GITHUB_OUTPUT=/dev/null bash -c "$matcher_script"; echo "::$?")
  local code="${actual##*::}"
  local out="${actual%::*}"
  out="${out//$'\n'/}"
  if [ "$out" = "$expected" ] && [ "$code" = "0" ]; then
    echo "PASS: $name"
  else
    echo "FAIL: $name (expected '$expected', got '$out', exit $code)"
    exit 1
  fi
}

run_case "docs-only: single md"          true  "README.md"
run_case "docs-only: docs dir"           true  "docs/api/foo.md" "docs/index.md"
run_case "docs-only: .github only"       true  ".github/workflows/foo.yml" ".github/ISSUE_TEMPLATE.md"
run_case "docs-only: mixed md and code"  false "README.md" "src/foo.py"
run_case "code-only"                     false "src/foo.py" "tests/test_foo.py"
run_case "nested md outside docs"        true  "paddleocr-js/README.md"
run_case "yml outside .github is code"   false "pyproject.toml" "config.yml"
run_case "empty diff is not docs-only"   false ""
run_case "deploy is code"                false "deploy/Dockerfile"

echo "All matcher tests passed."
