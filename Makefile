.PHONY: install sync lint type test dev console infra down help

install: ## Install all dependencies
	uv sync --dev

sync: ## Sync dependencies
	uv sync --dev

lint: ## Lint and format
	uv run ruff check --fix .
	uv run ruff format .

type: ## Type check
	uv run ty check src/

test: ## Run tests
	uv run pytest -v

check: lint type test ## Full check: lint + type + test

dev: ## Run in dev mode (requires LiveKit server)
	uv run clawhisper run --mode dev

console: ## Run in console mode (local mic/speaker)
	uv run clawhisper run --mode console

infra: ## Start STT + TTS infrastructure
	docker compose -f compose.infra.yml up -d

down: ## Stop infrastructure
	docker compose -f compose.infra.yml down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
