.PHONY: install run debug clean lint

run:
	uv run python -m src

install:
	uv sync


debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache

lint:
	flake8 . --exclude .venv
	mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs .
