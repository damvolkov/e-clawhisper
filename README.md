<p align="center">
  <img src="logo.jpg" alt="e-clawhisper" width="200" />
</p>

<h1 align="center">e-clawhisper</h1>

<p align="center">
  <strong>Always-on voice daemon — wake-word activated desktop bridge for AgentOS backends.</strong>
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
  |  eclaw   | (socket) |  +---------------------+-------------------+------+  |
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
                        |   | Silero VAD(ONNX)| |  | STT (WhisperLive)    |   |
                        |   | OpenWakeWord    | |  | LLM (OpenFang/GenAI) |   |
                        |   | Energy gate     | |  | TTS (Piper/Kokoro)   |   |
                        |   +-----------------+ |  | VAD (end-of-speech)  |   |
                        |                       |  +-----------------------+   |
                        |         +-------------v--------------+               |
                        |         | Audio Adapter (shared)     |               |
                        |         | Mic -> asyncio.Queue       |               |
                        |         | Speaker <- play_pcm_queue  |               |
                        |         +----------------------------+               |
                        +------------------------------------------------------+
                                  |              |             |
                            +-----v----+  +------v-----+  +---v--------+
                            | Whisper  |  |   Kokoro   |  |  OpenFang  |
                            |  Live    |  |   (HTTP)   |  |   (WS)     |
                            | :45120   |  |  :45130    |  |  :4200     |
                            +----------+  +------------+  +------------+
```

## Pipelines

### SENTINEL — passive listening

Runs continuously. Classifies each 512-sample chunk in parallel:

1. **Energy gate** — RMS below `energy_floor` -> SILENCE (skip VAD)
2. **Silero VAD** (ONNX) — speech probability -> NOISE or VOICE
3. **OpenWakeWord** (ONNX) — always fed (maintains mel spectrogram state)

Both ONNX models run via `ThreadPoolExecutor(2)` — true parallelism since ONNX releases GIL. Latency per chunk: `max(vad, oww)` ~1.5ms.

When wakeword confidence exceeds threshold -> signal orchestrator -> transition to TURN.

### TURN — active conversation

Executes one complete turn cycle:

1. **STT streaming || VAD end-of-speech** — audio flows to WhisperLive while Silero VAD tracks silence duration
2. **Transcript** — STT finalizes when VAD detects end-of-speech
3. **3-stage streaming pipeline** — three concurrent `create_task` stages connected by `asyncio.Queue`:

```
LLM text_delta -> sentence_queue -> TTS.synthesize -> pcm_queue -> sd.RawOutputStream
     stage 1                          stage 2                      stage 3
```

While sentence N plays, TTS synthesizes N+1, and LLM generates N+2.

Returns `TurnComplete` or `TurnError` -> orchestrator decides next phase.

### LOOP — continuous conversation

When `conversation.enabled: true`, the orchestrator enters LOOP after a completed TURN instead of returning to SENTINEL. A brief listening window (`loop_timeout` seconds) waits for follow-up speech. If the user speaks within the window, a new TURN begins immediately without requiring a wakeword. After `max_turns` consecutive turns or a timeout with no speech, the orchestrator returns to SENTINEL.

```
SENTINEL --wakeword--> TURN --complete--> LOOP --speech--> TURN
                                           |
                                           +--timeout/max_turns--> SENTINEL
```

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
| Docker + Compose | latest | STT/TTS backends |
| PortAudio | system | `sudo apt install libportaudio2` |
| ONNX Runtime | >= 1.24 | VAD + wakeword inference |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/eager-dev/e-clawhisper.git
cd e-clawhisper

# 2. Install
make install

# 3. Start backend services (WhisperLive + Kokoro)
make infra

# 4. Run the daemon
make start
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
sudo dpkg -i dist/e-clawhisper_*.deb
sudo apt-get install -f   # resolve system deps

# Manage via systemd
sudo systemctl enable --now e-clawhisper
sudo systemctl status e-clawhisper
journalctl -u e-clawhisper -f
```

The `.deb` bundles a self-contained Python venv, ONNX models, and a systemd unit. Configuration lives in `/etc/e-clawhisper/config.yaml` (survives upgrades).

## CLI

```
eclaw                         Root command
├── session                   Daemon lifecycle
│   ├── start                 Start voice daemon (foreground)
│   ├── stop                  Stop running daemon via IPC
│   ├── status                Show daemon state, pipeline, agent info
│   └── health                Check health of dependent services
└── config                    Configuration inspection
    ├── info                  Display current config
    └── backends              List available backends
```

### Health check

```bash
eclaw health
# [HEALTH] [OK]   stt ws://localhost:45120 reachable
# [HEALTH] [OK]   tts http://localhost:45130 reachable
# [HEALTH] [OK]   agent http://127.0.0.1:4200 reachable
```

The daemon also runs a startup health check — if any backend is unreachable, it exits with a clear error before entering the main loop.

## Backends

| Component | Backend | Protocol | Default URL |
|-----------|---------|----------|-------------|
| STT | WhisperLive | WebSocket | `ws://localhost:45120` |
| TTS | Kokoro | HTTP streaming | `http://localhost:45130` |
| TTS | Piper | Wyoming (TCP) | `tcp://localhost:45130` |
| Agent | OpenFang | WebSocket+REST | `http://127.0.0.1:4200` |
| Agent | Generic | PydanticAI | configurable per provider |
| VAD | Silero | Local (ONNX) | — |
| Wakeword | OpenWakeWord | Local (ONNX) | — |

### Generic LLM providers

The `generic` agent backend uses PydanticAI and supports multiple providers:

| Provider | Model (default) | Notes |
|----------|----------------|-------|
| `gemini` | `gemini-2.0-flash` | Google AI |
| `openai` | `gpt-4o-mini` | OpenAI |
| `anthropic` | `claude-sonnet-4-20250514` | Anthropic |
| `vllm` | user-specified | Self-hosted, requires `url` |

## Configuration

### Environment (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| ENVIRONMENT | DEV | Runtime environment |
| LOG_LEVEL | debug | Logging verbosity |
| SOCKET_PATH | /tmp/e-claw.sock | Unix socket for IPC |
| CONFIG_PATH | config.yaml | App config file path |

### Application (`config.yaml`)

All service connections use URL format. Validation is strict (`extra="forbid"`) — unknown keys are rejected.

```yaml
language: es

agent:
  name: damien
  backend: openfang        # openfang | generic

sentinel:
  energy_floor: 0.01       # RMS below this -> SILENCE
  vad_threshold: 0.5       # Silero probability threshold
  wakeword:
    model: alexa           # OWW model name
    threshold: 0.5

vad:
  threshold: 0.5           # End-of-speech threshold (TURN)
  silence_duration: 1.5    # Seconds of silence -> end utterance
  min_recording_time: 1.0  # Minimum before allowing end-of-speech

conversation:
  enabled: true            # Enable LOOP phase after TURN
  loop_timeout: 2.0        # Seconds to wait for follow-up speech
  max_turns: 10            # Max consecutive turns before SENTINEL

backends:
  openfang:
    url: http://127.0.0.1:4200
    timeout: 30.0
  generic:
    provider: gemini       # gemini | openai | anthropic | vllm
    model: gemini-2.0-flash
    timeout: 60.0
    url: http://localhost:45100  # vllm only

stt:
  backend: whisperlive
  whisperlive:
    url: ws://localhost:45120
    model: large-v3-turbo

tts:
  backend: kokoro
  kokoro:
    url: http://localhost:45130
    voice: em_alex
    sample_rate: 24000

audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 512          # Must be 512 for Silero VAD

logging:
  idle_interval: 0.25
  turn_interval: 0.25
```

## Models

ONNX models stored in `models/` (gitignored):

```
models/
├── vad/
│   └── silero_vad.onnx    # Auto-copied from silero-vad package
└── ww/
    └── alexa.onnx          # Custom or pretrained OWW model
```

Pretrained OWW models are resolved automatically. Custom models placed in `models/ww/` take priority.

## Infrastructure

STT and TTS run as Docker containers:

```bash
make infra    # Start WhisperLive + Kokoro
make down     # Stop containers
```

## Development

```bash
make install  # Install all dependencies
make lint     # Lint and format (ruff)
make type     # Type check (ty)
make test     # Run tests (pytest)
make check    # lint + type + test

make start    # Start daemon (foreground)
make status   # Check daemon status
make deb      # Build .deb package
```

## Packaging

The `.deb` package is built with [nfpm](https://nfpm.goreleaser.com/) and bundles everything needed to run on a bare Debian/Ubuntu system:

| Contents | Location |
|----------|----------|
| Python venv (self-contained) | `/opt/e-clawhisper/.venv/` |
| Source code | `/opt/e-clawhisper/src/e_clawhisper/` |
| ONNX models | `/opt/e-clawhisper/models/` |
| Compose file (infra) | `/opt/e-clawhisper/compose.infra.yml` |
| CLI wrapper | `/usr/bin/eclaw` |
| Config | `/etc/e-clawhisper/config.yaml` |
| Environment | `/etc/e-clawhisper/.env` |
| Systemd unit | `/usr/lib/systemd/system/e-clawhisper.service` |

Dependencies: `libportaudio2`, `libasound2`. Recommends: `pulseaudio` or `pipewire-pulse`, `docker.io`.

```bash
make deb                          # Build
sudo dpkg -i dist/e-clawhisper_*.deb
sudo systemctl enable --now e-clawhisper
```

## Logging

Pipeline-prefixed ANSI colored output:

```
14:32:01 [SENTINEL] [SILENCE] e=0.003
14:32:01 [SENTINEL] [NOISE]   vad=0.12 e=0.018
14:32:02 [SENTINEL] [VOICE]   vad=0.99 e=0.045
14:32:02 [SENTINEL] [WAKEWORD] score=0.87 model=alexa
14:32:03 [TURN]     [STT]     streaming audio...
14:32:05 [TURN]     [AGENT]   sending transcript
14:32:07 [TURN]     [TTS]     synthesizing response
14:32:08 [LOOP]     [LISTEN]  waiting for follow-up...
14:32:10 [LOOP]     [TIMEOUT] returning to sentinel
14:32:10 [SYSTEM]   [START]   ready — agent='damien' wakeword='alexa'
```

## Project Structure

```
src/e_clawhisper/
├── main.py                          Entry point
├── health.py                        Health probes (HTTP, TCP, WebSocket)
├── cli/
│   ├── main.py                      Cyclopts app
│   └── commands/
│       ├── session.py               start / stop / status / health
│       └── config.py                info / backends
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
│       ├── stt/
│       │   └── whisperlive.py       WhisperLive WebSocket
│       ├── tts/
│       │   ├── piper.py             Piper Wyoming TCP
│       │   └── kokoro.py            Kokoro HTTP streaming
│       ├── agent/
│       │   ├── openfang.py          OpenFang WebSocket + REST
│       │   └── generic.py           PydanticAI multi-provider LLM
│       └── base.py                  Adapter protocols (STTPort, TTSPort, AgentPort)
└── shared/
    ├── settings.py                  Env (Settings) + YAML (AppConfig)
    ├── logger.py                    Pipeline-prefixed colored logging
    └── operational/
        ├── events.py                WakewordEvent, TurnComplete, TurnError
        └── exceptions.py            AppError hierarchy
```

## License

[MIT](LICENSE)
