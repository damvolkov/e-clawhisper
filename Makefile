.PHONY: install sync lint type test check start stop status infra down script help

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

start: ## Start the voice daemon (foreground)
	uv run eclaw start

stop: ## Stop the voice daemon
	uv run eclaw stop

status: ## Show daemon status
	uv run eclaw status

info: ## Show current configuration
	uv run eclaw info

infra: ## Start STT + TTS infrastructure
	docker compose -f compose.infra.yml up -d

down: ## Stop infrastructure
	docker compose -f compose.infra.yml down

script: ## Run a test script (e.g. make script vad_streaming)
	@uv run python tests/scripts/$(filter-out $@,$(MAKECMDGOALS)).py

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

%: ## Catch-all for script arguments
	@:

