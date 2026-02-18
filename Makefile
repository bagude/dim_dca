.PHONY: install test lint run

install:
	python -m pip install -e .[dev]

test:
	pytest

lint:
	ruff check src tests

run:
	python -m dim_dca.cli
