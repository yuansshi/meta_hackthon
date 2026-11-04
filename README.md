# Earshot

**Meta's models on every ear.** Earshot turns the Bluetooth glasses or earbuds you already own into a Meta AI agent you use without taking out your phone. It also notices what you'll want before you ask.

**Built for your world, as the hackathon theme asks: for people on the move with their hands full.**
- The walk home with a bag in each hand.
- The school run with a toddler.
- The platform where the announcement just changed.

Today a question means stopping, finding the phone, unlocking it and typing, which takes both hands and your eyes. With Earshot you ask out loud and keep walking: the first word comes back in 1.2 to 3.5 s. And when a friend says "dinner's at my place Friday at seven", Earshot offers the reminder on its own.

**Try it now:** https://earshot-yuansshi.fly.dev. It works in any browser, with any headset, or by typing.

**Watch it (82 s):** https://earshot-yuansshi.fly.dev/media/earshot-demo.mp4 shows one take of the live app on the five core prompts, with each Meta model named on screen as it works. **In Spanish, Japanese and Hindi (77 s):** https://earshot-yuansshi.fly.dev/media/earshot-languages.mp4

## Try it in 60 seconds

1. **Type** "Remember I parked on level 3, spot 42", then "Where did I park?"
2. **Tap** *Sample photo: which platform is the 19:05?*
   - Muse Spark reads the board, and SAM 3.1 outlines where the answer is.
3. **Tap** *Look into the best e-bikes under $1,500 and get back to me*, then keep using it.
   - Muse Wonder researches in the background and speaks up 30 to 60 s later.
4. **Switch to Hands-free** (allow the mic) and say "Hey Earshot, will it rain this afternoon?"
   - Watch the receipts: location, then weather.
5. **Turn on Heads-up.** Read this aloud without the wake word: "Dinner's at my place Friday at seven, 1420 Pine Street."
   - Earshot asks whether to remind you. Say "yes".
6. **Then ask** "Hey Earshot, what was that address?" That's Rewind, the last 3 minutes, which is never stored.
7. **Pick another language** under Language, below the conversation, and ask in it. Earshot hears it and answers in it.

## What makes it different

- **Heads-up: it notices, then asks.**
  - In hands-free mode, Muse Spark screens speech that wasn't said to Earshot, using your own notes for context.
  - When it finds a plan, a request or an announcement you'd want, it offers in one short question.
  - Only a yes lets it act. It's opt-in, and nothing it overhears is stored.
- **Rewind: "what did she just say?"** Earshot answers from the last 3 minutes of speech it heard nearby, held in memory only.
- **Find it: "where is it?"** SAM 3.1 locates what you asked about in a photo. Earshot says where it is, and the app outlines it.
- **It speaks your language.** Earshot hears and answers in all 25 languages Muse Voice Transcribe is validated for, including Heads-up offers and reminder questions. Pick one on the page; the iPhone app uses the phone's language.
- **It gets back to you.** Muse Wonder takes long research jobs in the background, then reports back, even after you've disconnected.
- **It asks before it changes anything, and leaves receipts.** Every tool call shows its arguments, result and timing. The receipt log is hash-chained, so tampering is detectable.

## Results

These were measured live on Meta's models; the full study is in **[EVALUATION.md](EVALUATION.md)**.

| What | Result |
|---|---|
| The 6 prompts above, 3 live trials each, run three times | **54/54 passed** |
| End of speech to transcript | 0.17 to 0.30 s |
| End of speech to first word | 1.2 to 3.5 s (p50, by prompt) |
| Cold start, server fully stopped, 3 trials | The page loads in 5.7 s; the first word comes 2.6 s after asking |
| Several judges at once, 1 to 7 sessions | Every turn passed; first word 1.7 to 1.9 s (p50), 3.8 s (p90) at 7 |
| Find it for low vision, 60 Muse Image scenes | The correct side named in 60/60; SAM 3.1 outlined the target in 55, and never claimed an absent one (60/60) |
| Find it in clutter and dim light, 60 more | The correct side named in 58/58; SAM 3.1 found 52, never an absent one |
| Heads-up: friend stops talking to the offer | 3.7 s (p50) |
| Heads-up end to end by voice, 96 hard snippets through the live server | 92.9% precision, 97.5% recall; an offer 2.3 s after the speaker stops |
| Heads-up screener, 40 labeled snippets, 2 runs | 100% precision, 100% recall, no false offers (Muse Spark 1.2) |
| Heads-up on 120 harder snippets written by Muse Wonder | 90% precision, 99% recall, up from 81% precision before a prompt fix |
| Heads-up on 2,155 more, written and cross-checked by Muse Wonder, 3 Meta screeners | Shipped Muse Spark 1.2: 92.9% precision, 99.4% recall, 1.3 s. The deeper models tie on precision (95% intervals overlap), with lower recall or 7× the cost |
| Heads-up in the other 24 languages, 1,920 screenings | 99.7% precision, 98.4% recall; every offer in the user's language |
| Heads-up on hard snippets in Spanish, Hindi, Japanese and Arabic, 1,000 screenings | 100% recall in each; precision 85 to 91%, against 90.5% in English on the same snippets |
| Speech recognition biased to the user's language, 23 languages × 3 runs | Requests within 15% character error in 68 of 69, against 61 of 69 with English only |
| Spoken end to end in 23 languages, live, 3 trials each | Weather 68/69 answered in the language; a reminder confirmed by a spoken yes 62/69 |
| An hour in a pocket: one session for 70 minutes | 13 of 14 answered, across speech recognition's 60-minute session cap and 3 dropped connections |
| A server restart mid-session (the Fly trial forces one every 5 minutes), 3 trials | Reconnected in 5.4 to 6.1 s; the conversation and a research job in flight both survived, 3/3 |
| Muse Wonder research report | 100 of 100 varied jobs reported back: 29 s (41 s p90), 7 web searches and $0.13 at the median |
| Those 100 reports, fact-checked claim by claim by Muse Spark 1.3 with web search | 541 of 554 claims supported, 6 contradicted (near misses, like a bus that's 75 minutes, not an hour) |
| Cost per question | About half a cent; Heads-up adds about $0.25 an hour of nearby conversation |
| Cost of all six steps above | About $0.10 to $0.32, mostly the research job ([details](EVALUATION.md#7-what-the-public-demo-costs-to-run)) |
| Unit tests | 99, all using fake models and fake speech recognition |

## How it works: one cascade over Meta's models

```
glasses / earbuds ──Bluetooth (HFP mic + speaker)──▶ iPhone app ──┐   PCM16 16 kHz + JSON events
any browser + any headset ──▶ web client ─────────────────────────┤   over one WebSocket (/ws)
                                                                  ▼
            Agent server (FastAPI): holds the keys, never the audio
              ├─ ASR relay ── own 80 ms clock, silence fill, reconnects ──▶ Muse Voice Transcribe (realtime)
              ├─ turn gate ── wake word · 8 s follow-up window · echo filter
              │     ├─ said to Earshot ──▶ agent loop ──▶ Muse Spark 1.3 (tools, web search, vision)
              │     │                         ├─ "where is it?" ──▶ SAM 3.1 (Find it)
              │     │                         └─ "look into it" ──▶ Muse Wonder (background, reports back)
              │     └─ not said to Earshot ──▶ 3-minute Rewind buffer (memory only)
              │                                 └─ Heads-up ──▶ Muse Spark 1.2 screens ──▶ "Want me to…?" ──▶ agent
              ├─ fallback ── Muse Spark Open on Fireworks takes over if Meta's API fails
              └─ receipts ── every tool call, hash-chained; never the words overheard
```

| Job | Meta model | Why this one (measured) |
|---|---|---|
| Ears | `muse-voice-transcribe-1.0`, realtime, endpointing mode | It marks where each turn ends, 0.2 s after you stop talking, with keyword bias for "Earshot". Its language bias is set to the user's language: in 23 languages, requests were transcribed within 15% character error in 68 of 69 trials, against 61 of 69 with English only. |
| Brain | `muse-spark-1.3`, Responses API, `minimal` effort | Tools, built-in web search and vision, with a first word in about 2 s. "Think about it" raises the effort. |
| Heads-up screener | `muse-spark-1.2`, `minimal` effort | Best of 7 setups we compared: 100% recall, 2.0 s median, $0.07 an hour. |
| Find it | `sam-3.1` | Finds things in photos in 1 to 3 s, for a quarter of a cent. |
| Long tasks | `muse-wonder-1.0`, background mode | Research that reports back in under a minute. |
| Fallback | Muse Spark Open (open weights) on Fireworks | Takes the turn automatically if the hosted brain fails, so the demo survives an outage. |

- **Muse Image** made the synthetic test photos: 60 everyday scenes for the Find it test, and the privacy check, both in [EVALUATION.md](EVALUATION.md).
- **Speech output:** Meta has no text-to-speech model, so the phone or browser speaks each answer, which is also the fastest option.
- **Muse Glimmer** is self-hosted only, so it isn't used; it's the next step for an on-device wake word.

## Privacy

- **Audio** streams to Meta's speech recognition only while a session is open. No audio, video or photo is stored.
- **Overheard speech:** what wasn't said to Earshot stays in server memory for 3 minutes, hands-free mode only. It reaches Muse Spark only when you ask about it, or when you've turned Heads-up on.
- **Your conversation** is copied to the server's disk, without photos, so a reconnect after a server restart picks it up. Closing the page deletes it; otherwise it's ignored after 15 minutes and deleted.
- **Receipts** record what the agent did, never what it overheard. They're kept per device in `server/data/` and hash-chained, so editing one breaks the chain.
- **Spend limits:** the public demo has a daily spend cap, a 10-minute session limit and a per-address session limit. The day's spend is kept on a volume, so restarts don't reset the cap.
- **More detail:** the full data lifecycle and storage analysis are in [EVALUATION.md](EVALUATION.md#4-privacy-and-storage-store-meaning-not-media).

## Why not just use Siri?

- **Siri is hard-wired.** On iPhone, "Hey Siri" on AirPods and the glasses' assistant button only reach Siri, and iOS 27 lets Siri hand off to ChatGPT and nothing else.
- **Earshot answers instead, over the same route.** It doesn't replace Siri; it runs its own always-ready session over the same Bluetooth audio route.
- **AirPods work too,** exactly as glasses do.

## Run it

```bash
git clone https://github.com/yuansshi/meta_hackthon.git && cd meta_hackthon
make setup                                   # uv + Python 3.12
cp server/.env.example server/.env           # add META_API_KEY
make serve                                   # http://localhost:8000
make test                                    # unit tests with fakes: no network, no keys
make probe                                   # latency spike: time to first word per model and effort
uv --directory server run python scripts/eval_live.py     # the live evaluation above
uv --directory server run python scripts/eval_headsup.py  # the screener comparison (needs Meta API access)
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
3. Tap **Start listening** once with the app open (iOS only lets an app *start* the mic in the foreground). Then lock the phone and put it away. Turn on **Heads-up** if you want offers.
4. Say "Hey Earshot, …", or tap the glasses and just talk.

**Diagnostics** shows the audio route and mic sample rate, mic buffers per second, headset taps, interruptions and restarts, and how often the agent heard its own voice. It also has an echo test and a test line every 30 s for checking background audio.

## Client ⇄ server protocol (`/ws`)

Client → server:
- `hello {client: ios|web, device_id, tz, sample_rate: 16000, tools: [...], mode: wake|ptt, heads_up?}`
- binary PCM16LE mono 16 kHz frames of any size
- `tts {state: start|end, text}`
- `tool_result {id, ok, result | error}`
- `text {text}`
- `mode {mode}`
- `mute {on}`
- `ptt {state}`
- `heads_up {on}`
- `listen {}` (a headset tap: the next sentence needs no wake word)
- `image {data, text?}`: a JPEG/PNG/WebP data URL, at most 1024 px. It waits up to 2 minutes for the next question, or is asked about right away when `text` is given. `data: null` removes it.

Server → client:
- `ready {brain, asr, mode, wake_word, tools, heads_up}`
- `state {listening|thinking}`
- `transcript {final, text, accepted?, reason?}`
- `say {turn, text, final}` (speak each non-empty `text` in order; turns starting with `h` are Heads-up offers)
- `tool_call {id, name, args}`
- `action {turn, entry}` (receipts)
- `photo_used {turn}` (the staged photo went with this turn)
- `highlight {turn, boxes: [{x1, y1, x2, y2}]}` (Find it: where SAM 3.1 found the thing, as fractions of the photo)
- `confirm {turn, question}` (a yes/no question; the next short yes or no, spoken or typed, answers it)
- `report {turn, goal, details, model}` (a background research job finished; its summary follows as `say` events, and `turn_done` has `background: true`)
- `heads_up {on}`
- `asr {ok, message?}` (speech recognition stopped answering, say for want of credit, or came back; typing still works)
- `turn_done {turn, model, first_say_ms, total_ms, cost_usd, tools, searches, heads_up?}`
- `error {message}`

Everything in this repository was built during the Meta Global AI Developer Hackathon (Oct 26 to Nov 4, 2026).
