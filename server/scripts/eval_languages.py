"""How well Muse Voice Transcribe hears a spoken request in each of its validated languages, for a voice agent.

    uv run python scripts/eval_languages.py            # all languages, three language-bias settings each
    uv run python scripts/eval_languages.py --diarize  # also: who said what, in a two-person conversation

For each language, a macOS voice speaks "Hey Earshot, will it rain this afternoon?" in that language, and the clip
streams to the realtime API at real speed, as a phone would. Measured per language and language-bias setting:
- wake: whether Earshot's own wake-word gate finds "Earshot" in the transcript;
- cer: character error rate of the rest of the sentence against what was spoken;
- latency: end of speech to the final transcript.
Writes eval/results/languages-<date>.json. Speech recognition costs $0.18 an hour; a full run costs about 2 cents.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import wave
from pathlib import Path

from earshot.config import api_key, load_config
from earshot.languages import spaced
from earshot.relay import AsrRelay
from earshot.session import Gate

RATE = 16000
LEAD_S, TAIL_S = 0.5, 2.5
# (Meta language name, macOS voice, the request as spoken). Marathi and Tagalog have no macOS voice.
LANGS = [
    ("English", "Samantha", "Hey Earshot, will it rain this afternoon?"),
    ("Spanish", "Paulina", "Hey Earshot, ¿va a llover esta tarde?"),
    ("French", "Thomas", "Hey Earshot, est-ce qu'il va pleuvoir cet après-midi ?"),
    ("German", "Anna", "Hey Earshot, wird es heute Nachmittag regnen?"),
    ("Italian", "Alice", "Hey Earshot, pioverà oggi pomeriggio?"),
    ("Portuguese", "Luciana", "Hey Earshot, vai chover hoje à tarde?"),
    ("Dutch", "Xander", "Hey Earshot, gaat het vanmiddag regenen?"),
    ("Polish", "Zosia", "Hey Earshot, czy dziś po południu będzie padać?"),
    ("Turkish", "Yelda", "Hey Earshot, bu öğleden sonra yağmur yağacak mı?"),
    ("Indonesian", "Damayanti", "Hey Earshot, apakah sore ini akan hujan?"),
    ("Malay", "Amira", "Hey Earshot, adakah hujan akan turun petang ini?"),
    ("Vietnamese", "Linh", "Hey Earshot, chiều nay trời có mưa không?"),
    ("Japanese", "Kyoko", "Hey Earshot、今日の午後は雨が降りますか？"),
    ("Mandarin Chinese", "Tingting", "Hey Earshot，今天下午会下雨吗？"),
    ("Korean", "Yuna", "Hey Earshot, 오늘 오후에 비가 올까요?"),
    ("Hindi", "Lekha", "Hey Earshot, क्या आज दोपहर बारिश होगी?"),
    ("Bengali", "Piya", "Hey Earshot, আজ বিকেলে কি বৃষ্টি হবে?"),
    ("Tamil", "Vani", "Hey Earshot, இன்று மதியம் மழை பெய்யுமா?"),
    ("Telugu", "Geeta", "Hey Earshot, ఈ మధ్యాహ్నం వర్షం పడుతుందా?"),
    ("Kannada", "Soumya", "Hey Earshot, ಇಂದು ಮಧ್ಯಾಹ್ನ ಮಳೆ ಬರುತ್ತದೆಯೇ?"),
    ("Thai", "Kanya", "Hey Earshot บ่ายนี้ฝนจะตกไหม"),
    ("Hebrew", "Carmit", "Hey Earshot, האם ירד גשם אחר הצהריים?"),
    ("Arabic", "Majed", "Hey Earshot، هل ستمطر بعد الظهر؟"),
]
CONVERSATION = [  # (who, macOS voice, line): a friend, the user, and Earshot's own voice from the speaker
    ("friend", "Daniel", "Don't forget, dinner's at my place on Friday at seven."),
    ("user", "Alex", "Great. What's the address again?"),
    ("friend", "Daniel", "It's 1420 Pine Street, unit 3."),
    ("user", "Alex", "Hey Earshot, remind me about dinner on Friday."),
    ("earshot", "Samantha", "Should I remind you about dinner on Friday at seven?"),
    ("user", "Alex", "Yes please."),
]
GAP_S = 0.7


def pcm_of(voice: str, text: str, work: Path, name: str) -> bytes:
    aiff, wav = work / f"{name}.aiff", work / f"{name}.wav"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", str(RATE), "-ac", "1",
                    "-c:a", "pcm_s16le", str(wav)], check=True)
    with wave.open(str(wav)) as w:
        return w.readframes(w.getnframes())


def silence(seconds: float) -> bytes:
    return b"\0\0" * int(RATE * seconds)


def letters(text: str) -> str:
    """For comparing transcripts: NFKC, lowercase, no punctuation or spaces (so it works for CJK and Thai)."""
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if not unicodedata.category(c).startswith(("P", "Z", "S")))


def strip_wake(text: str) -> str:
    return re.sub(r"^\W*(hey|hi|ok|okay)?\W*earshot\W*", "", text, flags=re.I)


def request_cer(said: str, heard: str) -> float:
    """CER of the request alone: the wake word is scored separately, so skip up to 3 leading words of what was heard."""
    ref, words = strip_wake(said), spaced(heard).split()
    return min(cer(ref, " ".join(words[k:])) for k in range(min(4, len(words) + 1)))


def cer(ref: str, hyp: str) -> float:
    a, b = letters(ref), letters(hyp)
    if not a:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


async def stream(audio: bytes, mode: str, bias: list[str] | None) -> tuple[float, list[tuple[float, dict]]]:
    """Streams the audio at real speed; returns the time the first sample went out and every event with its time."""
    cfg = dict(load_config()["asr"])
    cfg["mode"] = mode
    cfg["languages"] = bias or []
    events: list[tuple[float, dict]] = []

    async def on_event(evt: dict) -> None:
        events.append((time.monotonic(), evt))

    relay = AsrRelay(cfg, api_key(cfg["api_key_env"]), on_event)
    if not bias:  # no hint at all: let the model detect the language
        relay.handshake = (lambda h=relay.handshake: {k: v for k, v in h().items() if k != "languageBias"})
    task = asyncio.create_task(relay.run())
    for _ in range(200):
        if relay.session_id:
            break
        await asyncio.sleep(0.05)
    t0 = time.monotonic()
    for i in range(0, len(audio), relay.frame_bytes):
        relay.feed(audio[i:i + relay.frame_bytes])
        await asyncio.sleep(cfg["frame_ms"] / 1000)
    await asyncio.sleep(1.0)
    await relay.close()
    task.cancel()
    return t0, events


async def one_language(lang, voice, text, setting, pcm, gate, sem, wake_voice="native") -> dict:
    bias = {"english": ["English"], "own": [lang] + ([] if lang == "English" else ["English"]), "none": None}[setting]
    async with sem:
        t0, events = await stream(silence(LEAD_S) + pcm + silence(TAIL_S), "ENDPOINTING", bias)
    done = [(t, e) for t, e in events if e.get("type") == "speechComplete" and e.get("transcript")]
    heard = " ".join(e["transcript"] for _, e in done)
    speech_end = t0 + LEAD_S + len(pcm) / 2 / RATE
    return {"language": lang, "setting": setting, "wake_voice": wake_voice, "said": text, "heard": heard,
            "wake": gate.find_wake(heard) is not None,
            "cer": round(request_cer(text, heard), 3) if heard else 1.0,
            "latency_s": round(done[-1][0] - speech_end, 2) if done else None}


async def diarize(work: Path) -> dict:
    """The same conversation in DIARIZATION and in ENDPOINTING mode: speaker labels, and how fast each turn lands."""
    parts, ends, at = [silence(LEAD_S)], [], LEAD_S
    for k, (who, voice, line) in enumerate(CONVERSATION):
        pcm = pcm_of(voice, line, work, f"conv{k}")
        parts += [pcm, silence(GAP_S)]
        at += len(pcm) / 2 / RATE
        ends.append((who, line, at))
        at += GAP_S
    audio = b"".join(parts) + silence(TAIL_S)
    out = {}
    for mode in ("DIARIZATION", "ENDPOINTING"):
        t0, events = await stream(audio, mode, ["English"])
        turns, label = [], None
        for t, e in events:
            if e.get("type") == "speaker":
                label = e.get("label")
            elif e.get("type") == "speechComplete" and e.get("transcript"):
                turns.append({"heard": e["transcript"], "speaker": label, "at_s": round(t - t0, 2)})
            elif e.get("type") == "speechStart":
                label = None
        for tr in turns:  # match each finished turn to the spoken line that ended just before it
            before = [x for x in ends if x[2] <= tr["at_s"] + 0.05]
            if before:
                who, line, end = before[-1]
                tr.update(who=who, line=line, latency_s=round(tr["at_s"] - end, 2))
        out[mode] = {"turns": turns, "event_types": sorted({e.get("type") for _, e in events})}
    return out


async def main() -> None:
    gate = Gate(load_config()["turns"])
    work = Path(tempfile.mkdtemp(prefix="langs-"))
    pcms = {lang: pcm_of(voice, text, work, f"l{i}") for i, (lang, voice, text) in enumerate(LANGS)}
    # The brand name said the English way, then the request in the language: closer to how people say a product name
    # than a voice reading "Hey Earshot" with its own language's spelling rules.
    hey = pcm_of("Samantha", "Hey Earshot,", work, "hey")
    mixed = {lang: hey + silence(0.15) + pcm_of(voice, strip_wake(text), work, f"m{i}")
             for i, (lang, voice, text) in enumerate(LANGS)}
    sem = asyncio.Semaphore(20)
    runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 1
    rows = []
    for run in range(runs):  # streaming timing varies, so repeat and pool
        batch = await asyncio.gather(
            *(one_language(lang, voice, text, s, pcms[lang], gate, sem) for lang, voice, text in LANGS
              for s in ("english", "own", "none")),
            *(one_language(lang, voice, text, s, mixed[lang], gate, sem, "english") for lang, voice, text in LANGS
              for s in ("english", "own")))
        rows += [{**r, "run": run} for r in batch]
    summary = {}
    for s, wv in (("english", "native"), ("own", "native"), ("none", "native"), ("english", "english"),
                  ("own", "english")):
        rs = [r for r in rows if r["setting"] == s and r["wake_voice"] == wv]
        s = f"{s}/wake said in {wv}"
        lat = sorted(r["latency_s"] for r in rs if r["latency_s"] is not None)
        summary[s] = {"wake": f"{sum(r['wake'] for r in rs)}/{len(rs)}",
                      "median_cer": round(sorted(r["cer"] for r in rs)[len(rs) // 2], 3),
                      "under_15pct_cer": f"{sum(r['cer'] < 0.15 for r in rs)}/{len(rs)}",
                      "median_latency_s": lat[len(lat) // 2] if lat else None}
    result = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "summary": summary, "rows": rows}
    if "--diarize" in sys.argv:
        result["diarization"] = await diarize(work)
    out = Path(__file__).resolve().parents[1] / "eval" / "results" / f"languages-{time.strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(summary, indent=1))
    for r in sorted(rows, key=lambda r: (r["language"], r["wake_voice"], r["setting"])):
        print(f"{r['language']:17s} {r['setting']:8s} {r['wake_voice']:7s} wake={'Y' if r['wake'] else '-'} "
              f"cer={r['cer']:.2f} lat={r['latency_s']}  {r['heard'][:60]}")
    if "--diarize" in sys.argv:
        for mode, d in result["diarization"].items():
            print(mode, d["event_types"])
            for t in d["turns"]:
                print(f"   {t.get('who', '?'):8s} speaker={t['speaker']} lat={t.get('latency_s')}  {t['heard'][:60]}")
    print("wrote", out)


if __name__ == "__main__":
    asyncio.run(main())
