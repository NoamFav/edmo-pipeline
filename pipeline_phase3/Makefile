.PHONY: help install install-dev setup test clean run docs

help:
	@echo "EDMO Pipeline - Available targets:"
	@echo "  install      - Install Python dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  setup        - Complete setup (dependencies, models, build)"
	@echo "  test         - Run all tests"
	@echo "  clean        - Clean build artifacts and cache"
	@echo "  run          - Start all services"
	@echo "  docs         - Generate documentation"
	@echo "  format       - Format code (Python, Go)"

install:
	pip install -r requirements.txt
	python -m spacy download nl_core_news_sm

install-dev:
	pip install -r requirements-dev.txt

setup: install
	@echo "Building Go core..."
	cd src/go_core && go build -o ../../bin/edmo-pipeline
	@echo "Building C++ modules..."
	mkdir -p src/cpp_native/build
	cd src/cpp_native/build && cmake .. && make

test:
	./scripts/setup/run_tests.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage
	rm -rf src/cpp_native/build
	rm -rf bin

run:
	./scripts/deployment/start_services.sh

docs:
	cd docs && sphinx-build -b html . _build/html

format:
	black src/python_services/
	isort src/python_services/
	cd src/go_core && go fmt ./...
