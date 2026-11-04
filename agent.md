# Agent Guide

## Project
Earshot: a Meta AI agent for any Bluetooth glasses or earbuds, built for the Meta Global AI Developer Hackathon.
Submissions are due Nov 4, 2026, 9:00 PM ET (6:00 PM PT); the team target is 2:00 PM PT.
Muse Voice Transcribe hears (realtime ASR with turn detection) and Muse Spark thinks (Responses API, tools, web search).
The phone or browser speaks each answer into the headset. Architecture and the client ⇄ server protocol: `README.md`.

## Setup
- Install: `make setup` (uv, Python 3.12), then `cp server/.env.example server/.env` and add `META_API_KEY`.
  Add `FIREWORKS_API_KEY` for the Muse Spark Open fallback brain (open weights on Fireworks).
- Run: `make serve`, then open http://localhost:8000 in a real browser (the Claude app's browser pane blocks the mic).
  The iPhone app connects to `ws://<mac-lan-ip>:8000/ws`.
- Test: `make test` (fakes only: no network, no keys). Spikes: `make probe` (model latency), `make probe-asr` (ASR round trip).
- iOS: `make ios-build` compiles for the simulator without signing. Run on a phone from Xcode (team `WL2YZ3LL6N` is set).
- Deploy: `make deploy` (Fly.io app `earshot-yuansshi`, config in `fly.toml`), live at https://earshot-yuansshi.fly.dev.
  The Fly CLI needs Docker Desktop's socket (`DOCKER_HOST`, set by the Makefile). Secrets were imported from `server/.env`.
- Live check: `make smoke` speaks to the deployed server; `make smoke URL=ws://localhost:8000/ws` checks the local one.
  `uv --directory server run python scripts/smoke.py --photo` sends a test image.
  `scripts/vision_check.py` asks both brains about a test image directly.

## Structure
- `server/src/earshot/`:
  - `app.py`: FastAPI with `/ws`, `/healthz`, and static `web/`.
  - `session.py`: the turn gate (wake word, follow-up window, echo filter) and per-connection orchestration.
  - `relay.py`: Meta realtime ASR with an 80 ms pacing clock and reconnects.
  - `agent.py`: the streaming tool loop, over the Responses API or Chat Completions for open-weight hosts.
  - `tools.py`: server and device tools.
  - `memory.py`: per-device `remember`/`recall` notes.
  - `rewind.py`: the Rewind buffer, the last `rewind_s` of speech not meant for the agent. It's held in memory only, and its receipts keep a line count, never the words.
  - `research.py`: Muse Wonder background jobs. Reports are spoken into the device's open session, or held until it reconnects.
  - `actions.py`: receipts.
  - `budget.py`: the daily demo spend cap.
- `server/config/models.yaml`: the single source of truth for model ids, prices, and ASR, turn and budget settings.
- `server/scripts/probe.py`: latency spike S1; writes `server/results/probe-*.json`.
- `server/tests/`: pytest; doubles in `fakes.py`.
- `web/`: the judge-facing web client (no build step).
- `ios/`: the SwiftUI iPhone app in `Earshot.xcodeproj`. Files in `ios/Earshot/` are picked up automatically through a synchronized folder.
  - `AudioPipeline`: HFP mic → 16 kHz PCM, a silent Now Playing loop so headset taps arrive, and the chime.
  - `AgentSocket`, `SpeechOut`.
  - `DeviceTools`: Reminders and location.
  - `RemoteControls`: headset taps.
  - `AgentSession`: orchestration.
  - `SessionView`: the UI, including the spike diagnostics.

## Conventions
- Keep changes small and focused.
- Match existing code style.
- Never commit `.env`, API keys, or raw data. Keys stay on the server; clients never see them.
- Anything tunable goes in `server/config/models.yaml`, not in code.
- The protocol in `README.md` is shared by `web/app.js`, the iOS app, and `session.py`: change all sides together.

## Notes
- Meta's Llama API shut down July 6, 2026. Use the Meta Model API: `https://api.meta.ai/v1`, `META_API_KEY`.
- Realtime ASR:
  - It needs continuous real-time PCM, silence included, at 16 or 24 kHz only.
  - Set `mode: ENDPOINTING` explicitly; the API default is PUSH_TO_TALK.
- Muse Spark doesn't stream its reasoning, so the first word waits for all the thinking. Keep voice turns at `reasoning_effort: minimal`.
- iOS can't start the mic from the background. Start the session in the foreground and keep the audio engine running.
- The corporate network re-signs TLS; `config.py` injects `truststore` so outbound calls trust the OS keychain.
- From this Mac's network, Meta's model endpoints return 404 (speech-to-text works). Locally the agent falls back to Muse Spark Open.
  Measure Meta from the Fly machine: `fly machine exec <id> "..." -a earshot-yuansshi`. The exec API times out after about 60 s, so run long jobs in the background there.
- The Fly account is on a trial: machines stop after 5 minutes until a card is added at https://fly.io/trial.
