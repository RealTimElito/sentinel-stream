.PHONY: help setup install train test lint clean docker-up docker-down

help:
	@echo "Sentinel-Stream Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup       - Set up development environment"
	@echo "  install     - Install Python dependencies"
	@echo "  train       - Train the TGN model"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linters"
	@echo "  clean       - Clean generated files"
	@echo "  docker-up   - Start Docker services"
	@echo "  docker-down - Stop Docker services"

setup:
	@bash scripts/setup.sh

install:
	@pip install -r requirements.txt

train:
	@python scripts/train_model.py --config configs/tgn_config.yaml

test:
	@pytest tests/ -v

lint:
	@black --check src/ scripts/
	@flake8 src/ scripts/ --max-line-length=100 --ignore=E203,W503

clean:
	@rm -rf __pycache__ */__pycache__ */*/__pycache__
	@rm -rf *.pyc */*.pyc */*/*.pyc
	@rm -rf .pytest_cache
	@rm -rf .coverage htmlcov

docker-up:
	@docker-compose up -d

docker-down:
	@docker-compose down

