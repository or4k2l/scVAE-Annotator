.PHONY: help install install-dev test test-quick test-full lint lint-fix format clean docs

help:
	@echo "scVAE-Annotator - Makefile Commands"
	@echo "===================================="
	@echo ""
	@echo "Installation:"
	@echo "  make install        Install package and dependencies"
	@echo "  make install-dev    Install package with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run standard tests"
	@echo "  make test-quick     Run only fast tests"
	@echo "  make test-full      Run all tests with coverage"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run all linters"
	@echo "  make lint-fix       Auto-fix linting issues"
	@echo "  make format         Format code with black and isort"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          Clean build artifacts and cache"
	@echo "  make docs           Build documentation"
	@echo ""

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-xdist pytest-timeout coverage[toml]
	pip install black flake8 mypy pylint isort bandit safety

test:
	pytest tests/ -v --cov=. --cov-report=term-missing -m "not slow"

test-quick:
	pytest tests/ -v -m "not slow"

test-full:
	pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html --cov-report=xml

lint:
	black --check --diff .
	isort --check-only --diff .
	flake8 .
	pylint scvae_annotator.py src/ --exit-zero
	bandit -r . || true

lint-fix:
	black .
	isort .

format: lint-fix

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + || true
	rm -rf build dist .coverage htmlcov .pytest_cache .mypy_cache
	rm -f coverage.xml .coverage.*

docs:
	@echo "Documentation build not yet configured"
	@echo "See docs/README.md for manual build instructions"
