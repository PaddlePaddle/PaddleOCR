.PHONY: all format lint test tests integration_tests help

# Default target executed when no arguments are given to make.
all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

integration_test integration_tests: TEST_FILE = tests/integration_tests/

test tests:
	uv run --group test pytest --disable-socket --allow-unix-socket $(TEST_FILE)

integration_test integration_tests:
	uv run --group test --group test_integration pytest --retries 3 --retry-delay 1 $(TEST_FILE)

check_imports: $(shell find langchain_paddleocr -name '*.py')
	uv run --all-groups python ./scripts/check_imports.py $^

test_watch:
	uv run ptw --snapshot-update --now . -- -vv $(TEST_FILE)

# Run unit tests and generate a coverage report.
coverage:
	uv run --with pytest-cov pytest --cov \
		--cov-report xml \
		--cov-report html \
		--cov-report term-missing:skip-covered \
		$(TEST_FILE)
	open htmlcov/index.html

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_paddleocr
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	./scripts/lint_imports.sh
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run --all-groups mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --fix $(PYTHON_FILES)

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports                - check imports'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'integration_test             - run integration tests (NOTE: "export PADDLEOCR_ACCESS_TOKEN=..." is needed.)'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
