# Earshot

**Meta's models on every ear.** Earshot turns the Bluetooth glasses or earbuds you already own into a hands-free AI agent. The phone stays in your pocket. You say "Hey Earshot…", Meta's Muse models listen, think, look things up and act, and the answer plays in your ears.

**Try it:** https://earshot-yuansshi.fly.dev. Hold to talk with any mic or Bluetooth headset, or type.

Most "smart" audio glasses are just a Bluetooth headset: a microphone, two speakers and a button. Nothing runs on them, and nothing needs to. Earshot puts the intelligence where it can live:

- **The phone (or a browser)** holds the Bluetooth audio session, streams your voice, speaks each answer a sentence at a time, and runs actions that belong on your device, such as reminders and location.
- **The agent server** hears with Muse Voice Transcribe and thinks with Muse Spark: tools, web search, and adjustable reasoning.
- **Look:** take a photo (tap Look on iPhone, or the camera button on the web), then ask about it out loud: "what does this sign say?" Muse Spark answers from the image. No wake word is needed right after a photo.
- **Get back to me:** "look into the best e-bikes under $1,500 and get back to me" hands the job to **Muse Wonder**, which researches in the background with web search. A minute or so later it speaks up with the answer, or tells you when you next connect.
- **Memory:** "remember I parked on level 3, spot 42", then later "where did I park?"
- **Rewind:** "what was that address she just said?" In hands-free mode Earshot keeps the last 3 minutes of speech that wasn't meant for it, in memory only, and answers from it. Nothing it overhears is written to disk.
- **Asks first:** before it changes anything, like adding a reminder, it asks "Should I remind you to…?" and waits for a spoken or tapped yes.

## How it works

```
glasses / earbuds ──Bluetooth (HFP mic + speaker)──▶ iPhone app ──┐   PCM16 16 kHz + JSON events
any browser + any headset ──▶ web client ─────────────────────────┤   over one WebSocket (/ws)
                                                                  ▼
            Agent server (FastAPI): holds the keys, never the audio
              ├─ ASR relay ── own 80 ms clock, silence fill, reconnects ──▶ Muse Voice Transcribe (realtime)
              ├─ turn gate ── wake word · 8 s follow-up window · echo filter · 3-minute Rewind buffer (memory only)
              ├─ agent loop ── streaming tool calls ──▶ Muse Spark (Responses API + web_search)
              ├─ server tools: weather, memory, Rewind, research    device tools (run on your phone): location, reminders
              └─ receipts ── every tool call with arguments, result and timing, shown live
```

| Job | Meta model | Why this one |
|---|---|---|
| Ears | `muse-voice-transcribe-1.0`, realtime, ENDPOINTING mode | The server marks where each turn ends, about 0.16 s after you stop talking, with keyword bias for the wake word. |
| Brain | `muse-spark-1.3` on the Responses API at `reasoning_effort: minimal` | Tool calling plus built-in web search, fast enough for voice. "Think about it" escalates the effort. |
| Long tasks | `muse-wonder-1.0` as a background response, polled | "Look into it and get back to me". Measured live: a 2-search job reported back in 23 s for $0.04. |
| Fallback brain | Muse Spark Open (open weights) on Fireworks | Takes the turn automatically if the hosted brain fails before speaking, so the demo survives an outage. Any OpenAI-compatible model is one line in `server/config/models.yaml`. |

Meta has no text-to-speech model, so speech is synthesized on the device. That's also the lowest-latency option.

## Why not just use Siri?

- On iPhone, "Hey Siri" on AirPods and the glasses' assistant button are hard-wired to Siri.
- iOS 27 lets Siri hand off to ChatGPT, and to nothing else.
- Earshot doesn't replace Siri. It runs its own always-ready session over the same Bluetooth audio route, so a frontier agent answers instead.
- The same app works with AirPods exactly as it does with glasses.

## Run it

```bash
git clone https://github.com/yuansshi/meta_hackthon.git && cd meta_hackthon
make setup                                   # uv + Python 3.12
cp server/.env.example server/.env           # add META_API_KEY
make serve                                   # http://localhost:8000
make test                                    # unit tests with fakes: no network, no keys
make probe                                   # latency spike: time to first word per model and effort
```

Open http://localhost:8000 in Chrome or Safari, and pick your Bluetooth headset as the microphone. Hold the button (or Space) to talk, or switch to **Hands-free** and start with "Hey Earshot".

Deploy: `docker build -t earshot .` at the repo root, then run it on any host that supports WebSockets, with `META_API_KEY` set as a secret.

### iPhone app (`ios/`)

1. Run `open ios/Earshot.xcodeproj`, choose your iPhone as the run destination, and press Run.
   - Signing uses your Apple Development team.
   - On first install, trust the developer in Settings → General → VPN & Device Management, and turn on Developer Mode if iOS asks.
2. Pair the glasses (or AirPods) with the phone. In Earshot → Server, enter one of:
   - `ws://<mac-ip>:8000/ws`, with the phone on the same Wi-Fi as the Mac running `make serve`.
   - The deployed `wss://…/ws` URL.
3. Tap **Start listening** once with the app open (iOS only lets an app *start* the mic in the foreground). Then lock the phone and put it away.
4. Say "Hey Earshot, …", or tap the glasses and just talk.

**Diagnostics** shows the audio route and mic sample rate, mic buffers per second, headset taps, interruptions and restarts, and how often the agent heard its own voice. It also has an echo test and a test line every 30 s for checking background audio.

## Client ⇄ server protocol (`/ws`)

Client → server:
- `hello {client: ios|web, device_id, tz, sample_rate: 16000, tools: [...], mode: wake|ptt}`
- binary PCM16LE mono 16 kHz frames of any size
- `tts {state: start|end, text}`
- `tool_result {id, ok, result | error}`
- `text {text}`
- `mode {mode}`
- `mute {on}`
- `ptt {state}`
- `listen {}` (a headset tap: the next sentence needs no wake word)
- `image {data, text?}`: a JPEG/PNG/WebP data URL, at most 1024 px. It waits up to 2 minutes for the next question, or is asked about right away when `text` is given. `data: null` removes it.

Server → client:
- `ready {brain, asr, mode, wake_word, tools}`
- `state {listening|thinking}`
- `transcript {final, text, accepted?, reason?}`
- `say {turn, text, final}` (speak each non-empty `text` in order)
- `tool_call {id, name, args}`
- `action {turn, entry}` (receipts)
- `photo_used {turn}` (the staged photo went with this turn)
- `confirm {turn, question}` (a yes/no question; the next utterance or `text` is the answer)
- `report {turn, goal, details, model}` (a background research job finished; its summary follows as `say` events, and `turn_done` has `background: true`)
- `turn_done {turn, model, first_say_ms, total_ms, cost_usd, tools, searches}`
- `error {message}`

## Privacy and cost

- Audio streams to Meta's speech recognition only while a session is open. No audio is stored.
- Transcripts go to Muse Spark.
- Rewind works only in hands-free mode. It holds the last 3 minutes of speech that wasn't meant for Earshot in server memory, drops it when the session ends, and sends it to Muse Spark only when you ask about it. Its receipts record how many lines were heard, never the words.
- Receipts (tool calls) are kept per device in `server/data/` so you can see exactly what the agent did.
- The public demo has a daily spend cap, a 10-minute session limit and a per-address session limit.

Everything in this repository was built during the Meta Global AI Developer Hackathon (Oct 26 to Nov 4, 2026).
