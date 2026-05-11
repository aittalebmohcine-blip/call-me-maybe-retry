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
	flake8 src --exclude .venv
	mypy src
