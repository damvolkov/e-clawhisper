# e-clawhisper

Always-on voice daemon — wake-word activated desktop bridge for AgentOS backends.

Mic → VAD → STT → Agent → TTS → Speaker

## Architecture

Hexagonal architecture with three layers: CLI control, daemon core, and pluggable adapters.

```
                          ┌─────────────────────────────────────────────────┐
                          │               DAEMON (background)               │
                          │                                                 │
  ┌──────────┐    IPC     │  ┌───────────────────────────────────────────┐  │
  │   CLI    │◄──────────►│  │            Unix Socket Server             │  │
  │  eclaw   │  (socket)  │  └───────────────────────────────────────────┘  │
  └──────────┘            │                      │                          │
                          │               ┌──────▼──────┐                   │
                          │               │ Orchestrator │                   │
                          │               └──────┬──────┘                   │
                          │                      │                          │
                          │  ┌───────────────────▼───────────────────────┐  │
                          │  │            Pipeline Runner                │  │
                          │  │                                           │  │
                          │  │   ┌─────┐   ┌──────────┐   ┌─────────┐  │  │
                          │  │   │ Mic │──►│   VAD    │──►│  Wake   │  │  │
                          │  │   └─────┘   │ (TenVAD) │   │  Word   │  │  │
                          │  │             └──────────┘   └────┬────┘  │  │
                          │  │                                 │       │  │
                          │  │            ┌────────────────────▼────┐  │  │
                          │  │            │     Turn Manager        │  │  │
                          │  │            │  (barge-in + timeout)   │  │  │
                          │  │            └────────────┬───────────┘  │  │
                          │  │                         │              │  │
                          │  │   ┌─────────┐    ┌──────▼──────┐      │  │
                          │  │   │ Speaker │◄───│    TTS      │      │  │
                          │  │   └─────────┘    └─────────────┘      │  │
                          │  │                         ▲              │  │
                          │  │   ┌─────────┐    ┌──────┴──────┐      │  │
                          │  │   │   STT   │───►│   Agent     │      │  │
                          │  │   │  (WS)   │    │   (WS)      │      │  │
                          │  │   └─────────┘    └─────────────┘      │  │
                          │  └───────────────────────────────────────┘  │
                          └─────────────────────────────────────────────┘
                                    │              │             │
                              ┌─────▼────┐  ┌─────▼─────┐  ┌───▼────────┐
                              │ Whisper  │  │   Piper   │  │  OpenFang  │
                              │  Live    │  │   (TCP)   │  │   (WS)    │
                              │  :9090   │  │  :10200   │  │  :4200    │
                              └──────────┘  └───────────┘  └────────────┘
```

## Pipeline Flow

All adapters (STT, Agent, TTS) connect at daemon startup and remain warm.
VAD runs continuously; audio only flows to STT when speech is detected.

```
┌─────────────────────────────────────────────────────────────────────┐
│ STARTUP                                                             │
│  Agent WS connects (persistent) ── keep-alive pings                │
│  STT WS connects (warm-up model) ── reconnects per utterance       │
│  TTS TCP ready (stateless per synthesis)                            │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ IDLE — Passive Listening                                            │
│                                                                     │
│  Mic ──► VAD ──► speech? ──► STT (stream audio)                    │
│                  no → discard                                       │
│                                                                     │
│  VAD silence timeout (1.5s) ──► STT finalize ──► transcript        │
│     └── contains wake word? ──► YES → ACTIVATE conversation        │
│                                  NO → reset STT, continue IDLE     │
│                                                                     │
│  Pre-roll buffer keeps last 2s of audio (context retention)         │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STREAMING — Active Conversation                                     │
│                                                                     │
│  Mic ──► VAD ──► speech? ──► STT (stream audio)                    │
│                                                                     │
│  VAD silence timeout ──► STT finalize ──► transcript ──► Agent WS  │
│                                                                     │
│  Agent streams text_delta chunks back                               │
│  Full response ──► TTS synthesize ──► Speaker (non-blocking)        │
│  → transition to SPEAKING                                           │
│                                                                     │
│  No activity 30s ──► conversation timeout ──► IDLE                  │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ SPEAKING — TTS Playback                                             │
│                                                                     │
│  Audio playing through speakers (non-blocking)                      │
│  VAD still monitoring microphone                                    │
│                                                                     │
│  Speech detected? ──► BARGE-IN: stop TTS → back to STREAMING       │
│  Playback finished? ──► back to STREAMING (or IDLE if timed out)    │
└─────────────────────────────────────────────────────────────────────┘
```

### Sequence Diagram

```
    Mic          VAD        WakeWord      STT(WS)     Agent(WS)    TTS(TCP)    Speaker
     │            │            │            │            │            │            │
     │─── chunk ─►│            │            │            │            │            │
     │            │── speech ─►│            │            │            │            │
     │            │            │        ┌───┤            │            │            │
     │            │            │        │audio           │            │            │
     │            │            │        │streaming       │            │            │
     │            │            │        └───┤            │            │            │
     │            │── silence ─┼──────────►│            │            │            │
     │            │  (1.5s)    │  finalize  │            │            │            │
     │            │            │◄─ transcript            │            │            │
     │            │            │── "damien" found ──────►│            │            │
     │            │            │            │  ┌── query ┤            │            │
     │            │            │            │  │streaming│            │            │
     │            │            │            │  └────────►│            │            │
     │            │            │            │            │── text ───►│            │
     │            │            │            │            │            │── PCM ────►│
     │            │            │            │            │            │            │
```

## CLI

```
eclaw                         Root command
├── session                   Daemon lifecycle
│   ├── start                 Start voice daemon (foreground)
│   ├── stop                  Stop running daemon via IPC
│   └── status                Show daemon state, pipeline, agent info
└── config                    Configuration inspection
    ├── info                  Display current config (agent, STT, TTS, VAD)
    └── backends              List available backend implementations
```

## Backends

| Component | Backend       | Protocol       | Default Port |
|-----------|---------------|----------------|--------------|
| STT       | WhisperLive   | WebSocket      | 9090         |
| TTS       | Piper         | Wyoming (TCP)  | 10200        |
| Agent     | OpenFang      | WebSocket+REST | 4200         |
| VAD       | TEN VAD       | Local (native) | —            |

## Infrastructure

STT and TTS run as Docker containers defined in `compose.infra.yml`:

```bash
# Start backend services
make infra

# Stop backend services
make down
```

## Development

```bash
# Install dependencies
make install

# Run tests
make test

# Lint
make lint

# Start daemon (foreground)
make start

# Check daemon status
make status
```

## Configuration

Environment variables (`.env`):

| Variable      | Default              | Description            |
|---------------|----------------------|------------------------|
| ENVIRONMENT   | DEV                  | Runtime environment    |
| LOG_LEVEL     | debug                | Logging verbosity      |
| SOCKET_PATH   | /tmp/e-claw.sock     | Unix socket for IPC    |

Application config (`config.yaml`): agent name, backend selection, STT/TTS/VAD parameters.

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
│   ├── orchestrator.py              Pipeline assembly + lifecycle
│   ├── pipeline/
│   │   ├── states.py                PipelineState / ConversationMode
│   │   ├── manager.py               Component factory
│   │   └── runner.py                Core async audio loop
│   ├── core/
│   │   ├── models.py                VADResult, TranscriptChunk, etc.
│   │   ├── interfaces/              ABCs: stt, tts, agent, processor
│   │   └── processors/
│   │       ├── vad.py               TEN VAD (hop-based framing)
│   │       ├── wake_word.py         Substring wake-word detection
│   │       └── turn_manager.py      Barge-in + conversation timeout
│   └── adapters/
│       ├── stt/whisper_live.py      WhisperLive WebSocket
│       ├── tts/piper.py             Piper Wyoming TCP
│       └── agents/openfang.py       OpenFang WebSocket + REST
└── shared/
    ├── settings.py                  Env (Settings) + YAML (AppConfig)
    ├── logger.py                    Structured logging with icons
    └── operational/
        ├── audio_device.py          sounddevice mic/speaker
        └── buffer.py                Lock-free ring buffer
```
