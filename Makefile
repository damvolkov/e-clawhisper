.PHONY: install sync lint type test test-slow check infra down deb clean-dist help

install: ## Install all dependencies
	uv sync --dev

sync: ## Sync dependencies
	uv sync --dev

lint: ## Lint and format
	uv run ruff check --fix .
	uv run ruff format .

type: ## Type check
	uv run ty check src/

test: ## Run unit tests
	uv run pytest -v

test-slow: ## Run integration tests (requires running e-voice)
	uv run pytest -m slow -v

check: lint type test ## Full check: lint + type + test

infra: ## Start e-voice infrastructure
	docker compose -f compose.infra.yml up -d

down: ## Stop infrastructure
	docker compose -f compose.infra.yml down

deb: ## Build .deb package
	@./packaging/build.sh

clean-dist: ## Clean build artifacts
	@rm -rf dist/

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
