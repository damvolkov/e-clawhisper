.PHONY: install sync lint type test check run start stop status infra down script help _ensure-stt _ensure-tts

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

run: ## Ensure infra (STT+TTS) is up, then start a session
	@$(MAKE) -j2 _ensure-stt _ensure-tts
	@ENVIRONMENT=DEV LOG_LEVEL=debug uv run python -m e_clawhisper.main session start

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

_ensure-stt:
	@nc -z localhost $${STT_PORT:-45120} 2>/dev/null \
		&& echo '✓ STT already up' \
		|| { echo '⏳ Starting STT...'; \
		     docker compose -f compose.infra.yml up -d stt; \
		     for i in $$(seq 1 90); do \
		       nc -z localhost $${STT_PORT:-45120} 2>/dev/null && echo '✓ STT ready' && exit 0; \
		       sleep 1; \
		     done; \
		     echo '✗ STT timeout'; exit 1; }

_ensure-tts:
	@nc -z localhost $${TTS_PORT:-45130} 2>/dev/null \
		&& echo '✓ TTS already up' \
		|| { echo '⏳ Starting TTS...'; \
		     docker compose -f compose.infra.yml up -d tts; \
		     for i in $$(seq 1 60); do \
		       nc -z localhost $${TTS_PORT:-45130} 2>/dev/null && echo '✓ TTS ready' && exit 0; \
		       sleep 1; \
		     done; \
		     echo '✗ TTS timeout'; exit 1; }

script: ## Run a test script (e.g. make script sentinel [args])
	$(eval _ARGS := $(filter-out $@ script,$(MAKECMDGOALS)))
	$(eval _NAME := $(word 1,$(_ARGS)))
	$(eval _REST := $(wordlist 2,$(words $(_ARGS)),$(_ARGS)))
	@uv run python tests/scripts/$(_NAME).py $(_REST)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

%: ## Catch-all for script arguments
	@:

