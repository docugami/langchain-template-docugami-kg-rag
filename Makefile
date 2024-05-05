.PHONY: all format lint test tests integration_tests docker_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/ docugami_kg_rag/

test:
	poetry run pytest --doctest-modules $(TEST_FILE)

tests:
	poetry run pytest --doctest-modules $(TEST_FILE)

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_package: PYTHON_FILES=docugami_kg_rag
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	poetry run ruff check .
	poetry run ruff check $(PYTHON_FILES) --diff
	poetry run ruff check --select I $(PYTHON_FILES)
	mkdir -p $(MYPY_CACHE); poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	poetry run ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	poetry run codespell --skip "./tests/testdata/*,*.yaml,*.lock,*.pdf,*.csv" --toml pyproject.toml

spell_fix:
	poetry run codespell --skip "./tests/testdata/*" --toml pyproject.toml -w

check_imports: $(shell find docugami_kg_rag -name '*.py')
	poetry run python ./scripts/check_imports.py $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports				- check imports'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'spell_check                  - run spell checker'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
