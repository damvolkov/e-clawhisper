# e-clawhisper

Always-on voice daemon — wake-word activated desktop bridge for AgentOS backends.

Two decoupled pipelines managed by a state machine orchestrator:

```
Mic → [SENTINEL: Energy → Silero VAD ‖ OpenWakeWord] → wakeword!
    → [TURN: STT ‖ VAD → LLM → TTS → Speaker] → back to sentinel
```

## Architecture

```
                        ┌──────────────────────────────────────────────────────┐
                        │                DAEMON (background)                    │
                        │                                                      │
  ┌──────────┐   IPC    │  ┌────────────────────────────────────────────────┐  │
  │   CLI    │◄────────►│  │             Unix Socket Server                │  │
  │  eclaw   │ (socket) │  └───────────────────┬────────────────────────────┘  │
  └──────────┘          │               ┌──────▼──────┐                        │
                        │               │ Orchestrator │ state: SENTINEL | TURN │
                        │               └──────┬──────┘                        │
                        │         ┌────────────┴────────────┐                  │
                        │   ┌─────▼──────┐           ┌──────▼─────┐            │
                        │   │  SENTINEL  │  wakeword  │    TURN    │            │
                        │   │  Pipeline  │──────────►│  Pipeline  │            │
                        │   │            │◄──────────│            │            │
                        │   └─────┬──────┘  complete  └──────┬─────┘            │
                        │         │                          │                  │
                        │   ┌─────▼──────────────┐   ┌──────▼──────────────┐   │
                        │   │ Silero VAD (ONNX)  │   │ STT (WhisperLive)  │   │
                        │   │ OpenWakeWord (ONNX)│   │ LLM (OpenFang)     │   │
                        │   │ Energy gate        │   │ TTS (Piper)        │   │
                        │   └────────────────────┘   │ VAD (end-of-speech)│   │
                        │                            └─────────────────────┘   │
                        │         ┌──────────────────────────┐                 │
                        │         │ Audio Adapter (shared)   │                 │
                        │         │ Mic → asyncio.Queue      │                 │
                        │         │ Speaker ← play_audio()   │                 │
                        │         └──────────────────────────┘                 │
                        └──────────────────────────────────────────────────────┘
                                  │              │             │
                            ┌─────▼────┐  ┌─────▼─────┐  ┌───▼────────┐
                            │ Whisper  │  │   Piper   │  │  OpenFang  │
                            │  Live    │  │   (TCP)   │  │   (WS)     │
                            │  :9090   │  │  :10200   │  │  :4200     │
                            └──────────┘  └───────────┘  └────────────┘
```

## Pipelines

### SENTINEL — passive listening

Runs continuously. Classifies each 512-sample chunk in parallel:

1. **Energy gate** — RMS below `energy_floor` → SILENCE (skip VAD)
2. **Silero VAD** (ONNX) — speech probability → NOISE or VOICE
3. **OpenWakeWord** (ONNX) — always fed (maintains mel spectrogram state)

Both ONNX models run via `ThreadPoolExecutor(2)` — true parallelism since ONNX releases GIL. Latency per chunk: `max(vad, oww)` ~1.5ms.

When wakeword confidence exceeds threshold → signal orchestrator → transition to TURN.

### TURN — active conversation

Executes one complete turn cycle:

1. **STT streaming ‖ VAD end-of-speech** — audio flows to WhisperLive while Silero VAD tracks silence duration
2. **Transcript** — STT finalizes when VAD detects end-of-speech
3. **LLM** — transcript sent to OpenFang agent, streaming response collected
4. **TTS → Speaker** — Piper synthesizes response, audio played back

Returns `TurnComplete` or `TurnError` → orchestrator transitions back to SENTINEL.

### VAD dual usage

| Pipeline | Role | Behavior |
|----------|------|----------|
| SENTINEL | Voice classification | Probability + threshold → SILENCE/NOISE/VOICE |
| TURN | End-of-speech detection | Silence-duration tracking with min recording time |

Same `SileroVAD` class, different wrappers: `is_voice()` vs `EndOfSpeechDetector`.

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
14:32:08 [SYSTEM]   [START]   ready — agent='damien' wakeword='alexa'
```

## CLI

```
eclaw                         Root command
├── session                   Daemon lifecycle
│   ├── start                 Start voice daemon (foreground)
│   ├── stop                  Stop running daemon via IPC
│   └── status                Show daemon state, pipeline, agent info
└── config                    Configuration inspection
    ├── info                  Display current config
    └── backends              List available backends
```

## Backends

| Component | Backend       | Protocol       | Default Port |
|-----------|---------------|----------------|--------------|
| STT       | WhisperLive   | WebSocket      | 9090         |
| TTS       | Piper         | Wyoming (TCP)  | 10200        |
| Agent     | OpenFang      | WebSocket+REST | 4200         |
| VAD       | Silero        | Local (ONNX)   | —            |
| Wakeword  | OpenWakeWord  | Local (ONNX)   | —            |

## Configuration

### Environment (`.env`)

| Variable      | Default              | Description            |
|---------------|----------------------|------------------------|
| ENVIRONMENT   | DEV                  | Runtime environment    |
| LOG_LEVEL     | debug                | Logging verbosity      |
| SOCKET_PATH   | /tmp/e-claw.sock     | Unix socket for IPC    |
| CONFIG_PATH   | config.yaml          | App config file path   |

### Application (`config.yaml`)

```yaml
agent:
  name: damien             # Agent name on OpenFang
  backend: openfang

sentinel:
  energy_floor: 0.01       # RMS below this → SILENCE
  vad_threshold: 0.5       # Silero probability threshold
  wakeword:
    model: alexa           # OWW model name (separate from agent)
    threshold: 0.5

vad:
  threshold: 0.5           # End-of-speech threshold (TURN)
  silence_duration: 1.5    # Seconds of silence → end utterance
  min_recording_time: 1.0  # Minimum before allowing end-of-speech

audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 512          # Must be 512 for Silero VAD
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
make infra    # Start WhisperLive + Piper
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
make script vad_streaming  # Run test scripts
```

## Project Structure

```
src/e_clawhisper/
├── main.py                          Entry point
├── cli/
│   ├── main.py                      Cyclopts app
│   └── commands/
│       ├── session.py               start / stop / status
│       └── config.py                info / backends
├── daemon/
│   ├── server.py                    Unix socket IPC server
│   ├── orchestrator.py              State machine SENTINEL ↔ TURN
│   ├── sentinel/
│   │   ├── pipeline.py              Passive listening loop
│   │   ├── vad.py                   Silero VAD (pure numpy ONNX)
│   │   └── wakeword.py              OpenWakeWord ONNX wrapper
│   ├── turn/
│   │   ├── pipeline.py              Active conversation cycle
│   │   └── vad.py                   End-of-speech detector
│   └── adapters/
│       ├── audio.py                 Mic/Speaker (sounddevice)
│       ├── stt.py                   WhisperLive WebSocket
│       ├── tts.py                   Piper Wyoming TCP
│       └── llm.py                   OpenFang WebSocket + REST
└── shared/
    ├── settings.py                  Env (Settings) + YAML (AppConfig)
    ├── logger.py                    Pipeline-prefixed colored logging
    └── operational/
        ├── events.py                WakewordEvent, TurnComplete, Turn
        └── exceptions.py            AppError hierarchy
```
