<p align="center">
  <img src="logo.jpg" alt="e-heed" width="200" />
</p>

<h1 align="center">e-heed</h1>

<p align="center">
  <strong>Always-on voice daemon — wake-word activated desktop bridge for Agent backends, powered by <a href="https://github.com/damvolkov/e-voice">e-voice</a>.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-%3E%3D3.13-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://ubuntu.com/"><img src="https://img.shields.io/badge/platform-Ubuntu%2024.04+-E95420?logo=ubuntu&logoColor=white" alt="Ubuntu"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/pkg-uv-blueviolet?logo=astral" alt="uv"></a>
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/lint-ruff-orange?logo=astral" alt="Ruff"></a>
</p>

---

Two decoupled pipelines managed by a state-machine orchestrator:

```
Mic -> [SENTINEL: Energy -> Silero VAD || OpenWakeWord] -> wakeword!
    -> [TURN: STT || VAD -> LLM -> TTS -> Speaker] -> back to sentinel
                                                    -> or LOOP (continue conversation)
```

## Architecture

```
                        +------------------------------------------------------+
                        |                DAEMON (background)                    |
                        |                                                      |
  +----------+   IPC    |  +------------------------------------------------+  |
  |   CLI    |<-------->|  |             Unix Socket Server                  |  |
  |  eheed   | (socket) |  +---------------------+-------------------+------+  |
  +----------+          |               +---------v--------+                   |
                        |               |   Orchestrator   |                   |
                        |               | SENTINEL | TURN  |                   |
                        |               |   | LOOP         |                   |
                        |               +---------+--------+                   |
                        |         +---------------+----------------+           |
                        |   +-----v------+   wakeword   +----------v---+       |
                        |   |  SENTINEL  | -----------> |    TURN      |       |
                        |   |  Pipeline  | <----------- |    Pipeline  |       |
                        |   |            |   complete    |              |       |
                        |   +-----+------+      |       +------+-------+       |
                        |         |             |              |               |
                        |         |        +----v----+         |               |
                        |         |        |  LOOP   |         |               |
                        |         |        | timeout |         |               |
                        |         |        +----+----+         |               |
                        |         |             |              |               |
                        |   +-----v-----------+ |  +-----------v-----------+   |
                        |   | Silero VAD(ONNX)| |  | STT (e-voice WS)     |   |
                        |   | OpenWakeWord    | |  | LLM (Generic/OpenFang)|  |
                        |   | Energy gate     | |  | TTS (e-voice WS)     |   |
                        |   +-----------------+ |  | VAD (end-of-speech)  |   |
                        |                       |  +-----------------------+   |
                        |         +-------------v--------------+               |
                        |         | Audio Adapter (shared)     |               |
                        |         | Mic -> asyncio.Queue       |               |
                        |         | Speaker <- play_pcm_queue  |               |
                        |         +----------------------------+               |
                        +------------------------------------------------------+
                                           |
                                    +------v-------+
                                    |   e-voice    |
                                    |  STT + TTS   |
                                    |   :45140     |
                                    +--------------+
```

## Pipelines

### SENTINEL -- passive listening

Runs continuously. Classifies each 512-sample chunk in parallel:

1. **Energy gate** -- RMS below `energy_floor` -> SILENCE (skip VAD)
2. **Silero VAD** (ONNX) -- speech probability -> NOISE or VOICE
3. **OpenWakeWord** (ONNX) -- always fed (maintains mel spectrogram state)

Both ONNX models run via `ThreadPoolExecutor(2)` -- true parallelism since ONNX releases GIL. Latency per chunk: `max(vad, oww)` ~1.5ms.

When wakeword confidence exceeds threshold -> signal orchestrator -> transition to TURN.

### TURN -- active conversation

Executes one complete turn cycle:

1. **STT streaming || VAD end-of-speech** -- audio flows to STT while Silero VAD tracks silence duration
2. **Transcript** -- STT finalizes when VAD detects end-of-speech
3. **3-stage streaming pipeline** -- three concurrent `create_task` stages connected by `asyncio.Queue`:

```
LLM text_delta -> sentence_queue -> TTS.synthesize -> pcm_queue -> sd.RawOutputStream
     stage 1                          stage 2                      stage 3
```

While sentence N plays, TTS synthesizes N+1, and LLM generates N+2.

Returns `TurnComplete` or `TurnError` -> orchestrator decides next phase.

### LOOP -- continuous conversation

When `conversation.enabled: true`, the orchestrator enters LOOP after a completed TURN instead of returning to SENTINEL. A brief listening window (`loop_timeout` seconds) waits for follow-up speech. If the user speaks within the window, a new TURN begins immediately without requiring a wakeword. After `max_turns` consecutive turns or a timeout with no speech, the orchestrator returns to SENTINEL.

### VAD dual usage

| Pipeline | Role | Behavior |
|----------|------|----------|
| SENTINEL | Voice classification | Probability + threshold -> SILENCE/NOISE/VOICE |
| TURN | End-of-speech detection | Silence-duration tracking with min recording time |

Same `SileroVAD` class, different wrappers: `is_voice()` vs `EndOfSpeechDetector`.

## Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| Ubuntu | 24.04+ | Primary target platform |
| Python | >= 3.13 | Required |
| [uv](https://docs.astral.sh/uv/) | >= 0.8 | Package manager |
| Docker + Compose | latest | e-voice backend |
| PortAudio | system | `sudo apt install libportaudio2` |
| ONNX Runtime | >= 1.24 | VAD + wakeword inference |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/eager-dev/e-heed.git
cd e-heed

# 2. Install
make install

# 3. Start e-voice (STT + TTS unified service)
make infra

# 4. Run the daemon
make run generic
```

## Installation

### From source (development)

```bash
# System dependencies (Ubuntu/Debian)
sudo apt update && sudo apt install -y libportaudio2 libsndfile1

# Python dependencies
make install        # uv sync --dev
```

### From .deb package (production)

```bash
# Build the package
make deb

# Install
sudo dpkg -i dist/e-heed_*.deb
sudo apt-get install -f   # resolve system deps

# Manage via systemd
sudo systemctl enable --now e-heed
sudo systemctl status e-heed
journalctl -u e-heed -f
```

## CLI

```
eheed                         Root command
├── session                   Daemon lifecycle
│   ├── start                 Start voice daemon (foreground)
│   ├── stop                  Stop running daemon via IPC
│   ├── status                Show daemon state, pipeline, agent info
│   ├── health                Check health of dependent services
│   └── logs                  Follow daemon logs (journalctl)
└── config                    Configuration inspection
    ├── info                  Display current config
    ├── backends              List available backends
    └── init                  Copy default config to ~/.config/e-heed/
```

## Backends

### STT

| Backend | Protocol | Default URL | Notes |
|---------|----------|-------------|-------|
| **e-voice** (default) | WebSocket | `http://localhost:45140` | Binary PCM16, VAD segmentation, LocalAgreement |
| WhisperLive | WebSocket | `ws://localhost:45120` | Legacy, requires separate container |

### TTS

| Backend | Protocol | Default URL | Notes |
|---------|----------|-------------|-------|
| **e-voice** (default) | WebSocket | `http://localhost:45140` | Base64 PCM16 chunks, persistent connection |
| Kokoro | HTTP streaming | `http://localhost:45130` | Legacy, requires separate container |
| Piper | Wyoming (TCP) | `tcp://localhost:45130` | Legacy |

### Agent / LLM

| Backend | Protocol | Notes |
|---------|----------|-------|
| **Generic** (default) | PydanticAI | Multi-provider: Gemini, OpenAI, Anthropic, vLLM |
| OpenFang | WebSocket+REST | Full agent OS |

## Configuration

### Environment (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| ENVIRONMENT | DEV | Runtime environment |
| LOG_LEVEL | debug | Logging verbosity |
| SOCKET_PATH | /tmp/eheed.sock | Unix socket for IPC |
| CONFIG_PATH | config.yaml | App config file path |

### Application (`config.yaml`)

```yaml
language: es

agent:
  name: damien
  backend: generic        # openfang | generic

stt:
  backend: evoice          # evoice | whisperlive
  evoice:
    url: http://localhost:45140
    language: es

tts:
  backend: evoice          # evoice | kokoro | piper
  evoice:
    url: http://localhost:45140
    voice: af_heart
    speed: 1.0

backends:
  generic:
    provider: gemini       # gemini | openai | anthropic | vllm
    model: gemini-2.0-flash
```

See [config.yaml](config.yaml) for the full reference.

## Infrastructure

e-voice provides unified STT + TTS as a single Docker container:

```bash
make infra    # Start e-voice
make down     # Stop containers
```

Legacy backends (WhisperLive, Kokoro) available via `docker compose --profile legacy`.

## Development

```bash
make install  # Install all dependencies
make lint     # Lint and format (ruff)
make type     # Type check (ty)
make test     # Run tests (pytest, excludes slow)
make check    # lint + type + test

make run generic  # Start daemon with generic LLM agent
make status       # Check daemon status

# Integration tests (requires running e-voice)
uv run pytest -m slow -v
```

## Project Structure

```
src/e_heed/
├── main.py                          Entry point
├── health.py                        Health probes
├── cli/
│   ├── main.py                      Cyclopts app
│   └── commands/
│       ├── session.py               start / stop / status / health / logs
│       └── config.py                info / backends / init
├── daemon/
│   ├── server.py                    Unix socket IPC server
│   ├── orchestrator.py              State machine SENTINEL <-> TURN <-> LOOP
│   ├── sentinel/
│   │   ├── pipeline.py              Passive listening loop
│   │   ├── vad.py                   Silero VAD (pure numpy ONNX)
│   │   └── wakeword.py              OpenWakeWord ONNX wrapper
│   ├── turn/
│   │   ├── pipeline.py              Active conversation cycle
│   │   └── vad.py                   End-of-speech detector
│   └── adapters/
│       ├── audio.py                 Mic/Speaker (sounddevice)
│       ├── base.py                  Adapter protocols (STTPort, TTSPort, AgentPort)
│       ├── stt/
│       │   ├── base.py              STTAdapter ABC
│       │   ├── evoice.py            e-voice WebSocket STT
│       │   └── whisperlive.py       WhisperLive (legacy)
│       ├── tts/
│       │   ├── base.py              TTSAdapter ABC
│       │   ├── evoice.py            e-voice WebSocket TTS
│       │   ├── kokoro.py            Kokoro HTTP (legacy)
│       │   └── piper.py             Piper Wyoming TCP (legacy)
│       └── agent/
│           ├── base.py              AgentAdapter ABC
│           ├── generic.py           PydanticAI multi-provider LLM
│           └── openfang.py          OpenFang WebSocket + REST
└── shared/
    ├── settings.py                  Env (Settings) + YAML (AppConfig)
    ├── logger.py                    Pipeline-prefixed colored logging
    └── operational/
        ├── events.py                WakewordEvent, TurnComplete, TurnError
        └── exceptions.py            AppError hierarchy
```

## License

[MIT](LICENSE)
