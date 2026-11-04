# Agent Guide

## Project
EveryMuse: a Meta AI agent for any Bluetooth glasses or earbuds, built for the Meta Global AI Developer Hackathon.
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
- `server/src/everymuse/`:
  - `app.py`: FastAPI with `/ws`, `/healthz`, and static `web/`.
  - `session.py`: the turn gate (wake word, follow-up window, echo filter) and per-connection orchestration.
  - `relay.py`: Meta realtime ASR with an 80 ms pacing clock and reconnects.
  - `agent.py`: the streaming tool loop, over the Responses API or Chat Completions for open-weight hosts.
  - `tools.py`: server and device tools.
  - `memory.py`: per-device `remember`/`recall` notes.
  - `rewind.py`: the Rewind buffer, the last `rewind_s` of speech not meant for the agent. It's held in memory only, and its receipts keep a line count, never the words.
  - `headsup.py`: Heads-up. A fast screener (Muse Spark 1.2, chosen by `scripts/eval_headsup.py`) reads the Rewind buffer and offers one yes/no; a yes runs the agent with that approval already given.
  - `find.py`: Find it. SAM 3.1 locates what the user asked about in their photo; the result is spoken as a position and sent as `highlight` boxes.
  - `languages.py`: the 25 languages Muse Voice Transcribe is validated for. From the hello's `language` tag it sets the ASR `languageBias` (the language plus English, for "EveryMuse"), the reply language, the fixed phrases, yes/no words and the reminder question. `spaced()` splits CJK and Thai transcripts so the wake word is found.
  - `research.py`: Muse Wonder background jobs. Reports are spoken into the device's open session, or held until it reconnects.
  - `actions.py`: hash-chained receipts (`verify` finds the first broken line).
  - `budget.py`: the daily demo spend cap.
- `server/eval/`: the labeled Heads-up set and committed results. Scripts:
  - `eval_headsup.py`: the screener comparison. Run it on the server, since Meta's API 404s from this Mac.
  - `eval_live.py`: the judge prompts against a live server.
  - `privacy_check.py`: Muse Image plus SAM 3.1 faces.
  - `eval_languages.py`: ASR language bias in 23 languages. ASR works from this Mac.
  - `eval_live_languages.py`: spoken weather and reminder in each language, against the live server.
  - `eval_headsup_languages.py`: Heads-up on translated snippets. Run it on the server.
  - `eval_find.py`: Find it on Muse Image scenes. Run it on the server.
  - `eval_load.py`: 1, 4 and 8 sessions at once.
  - `eval_cold_start.py`: stops the machine, then times a first visit.
  - `eval_research.py`: 20 Muse Wonder research jobs. Run it on the server.
  - `eval_headsup_more.py`: Muse Wonder writes 120 harder snippets, and the screener prompt before and after is compared. Run it on the server.
  - `eval_soak.py`: one session held past the ASR's 60-minute cap.
  - `eval_headsup_hard.py`: 480 Wonder-written snippets, judged by a second Wonder pass, then 4 screeners compared. Run it on the server; it resumes across restarts. Data is in `server/eval/hard/`.
  - Trial machines stop every 5 minutes, and their rootfs resets. Server-side scripts write to `$EVERYMUSE_DATA_DIR` (the volume), or the results are copied there before the stop.
- `EVALUATION.md`: every measured number, with the trade-offs.
- `server/config/models.yaml`: the single source of truth for model ids, prices, and ASR, turn and budget settings.
- `server/scripts/probe.py`: latency spike S1; writes `server/results/probe-*.json`.
- `server/tests/`: pytest; doubles in `fakes.py`.
- `web/`: the judge-facing web client (no build step). `web/media/everymuse-demo.mp4` is the 82 s demo video (720p, about 2 MB), served by the app and linked from the page and README.
- `ios/`: the SwiftUI iPhone app in `EveryMuse.xcodeproj`. Files in `ios/EveryMuse/` are picked up automatically through a synchronized folder.
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
  Measure Meta from the Fly machine instead.
  - Short commands: `fly machine exec <id> "..." -a earshot-yuansshi`. It has a short deadline, and background jobs it starts get killed.
  - Long jobs: `fly ssh console -a earshot-yuansshi -C "sh -c '...'"`. Upload a script by piping it through `base64 -d`.
  - The machine has 512 MB: create one API client per provider, never one per call, or the job is killed for lack of memory.
- The Fly account is on a trial: machines stop after 5 minutes until a card is added at https://fly.io/trial.
- `EVERYMUSE_DATA_DIR` is `/data/everymuse` on the `earshot_data` volume (1 GB, sjc), so the daily spend, notes and receipts survive restarts and deploys. The machine is pinned to that volume's host.
- Restarts are expected (the trial stops the machine every 5 minutes). So the volume also holds:
  - each device's conversation (`history/`, without photos). It's deleted on a clean close (codes 1000/1001), and ignored and deleted after 15 minutes.
  - research in flight (`research/pending/`, the Wonder response id), which `RESEARCHER.resume()` polls again at startup.
  - undelivered reports (`research/inbox/`).
  - models the API refused, with when to try them again (`refused.json`, written by `watch_brain` after each check). The app restores it before serving, so the first turn and `/healthz` already know about a refusal.
- The web page resends a question that got no reply at all before a reconnect.
- The relay reports `relayDown` after 3 refused attempts in a row, and the page shows "can't hear you".
