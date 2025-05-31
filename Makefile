.PHONY: test test-unit test-integration test-functional test-all test-coverage clean

# Default target
all: test-all

# Install dependencies
setup:
	pip install -r requirements.txt
	pip install -r test_requirements.txt

# Run tests
test:
	pytest tests/unit/

test-unit:
	pytest tests/unit/

test-integration:
	pytest tests/integration/

test-functional:
	pytest tests/functional/

test-all:
	pytest

# Run tests with coverage
test-coverage:
	pytest --cov=src --cov-report=html

# Clean build artifacts
clean:
	rm -rf .coverage
	rm -rf coverage_html_report
	rm -rf .pytest_cache
	rm -rf **/__pycache__
	rm -rf **/*.pyc

# Linting
lint:
	pylint src/ tests/

# Run application
run:
	python paper_revision.py