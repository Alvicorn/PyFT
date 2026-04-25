.PHONY: unit_tests*


MIN_COV = 80
PYTEST_COV_OPTS = --cov=. --cov-fail-under=$(MIN_COV) --cov-report term-missing

unit_tests:
	uv run pytest tests/unit/

unit_tests_cov:
	uv run pytest $(PYTEST_COV_OPTS) tests/unit

unit_tests_debug:
	uv run pytest -o log_cli=true -o log_cli_level=DEBUG -s tests/unit/
