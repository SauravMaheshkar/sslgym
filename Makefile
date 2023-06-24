## Install Dependencies
requirements:
	pip install -U pip setuptools wheel
	pip install -r .devcontainer/requirements.txt
	pre-commit install

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "__pycache__" -delete
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	pre-commit gc
	pre-commit clean

## Testing
test:
	pytest --durations=0 -v .

## Basic linting
lint:
	black src
	ruff check src
	mypy src
