"""EveryMuse end to end in each language, against the live server: spoken question, tools, spoken answer, spoken yes.

    uv run python scripts/eval_live_languages.py --url wss://earshot-yuansshi.fly.dev/ws --trials 2

Each session's hello carries the language. Per language and trial, two spoken scenarios:
- weather: "Hey EveryMuse," (said the English way, like a product name) then "will it rain this afternoon?" in the
  language. Passes if the wake word was heard, the forecast tool ran, and the reply is in that language.
- reminder: "Hey EveryMuse," then "remind me to buy coffee at six" in the language; EveryMuse asks first, and the answer
  is a spoken yes in the language. Passes if the reminder was created.
Device tools get canned answers, as in eval_live.py. Writes eval/results/live-languages-<date>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
import uuid
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_live import Trial, done  # noqa: E402
from websockets.asyncio.client import connect  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RATE = 16000
# code: (macOS voice, "will it rain this afternoon?", "remind me to buy coffee at six", a spoken yes)
LANGS = {  # code: (macOS voice, "will it rain this afternoon?", "remind me to buy coffee at 6 PM", a spoken yes)
    # Japanese says 夕方六時: Muse Voice Transcribe hears 午後 (gogo, "afternoon") as 五五 ("five five"). Even so, it
    # often loses the middle of this sentence; see EVALUATION.md section 8.
    "en": ("Samantha", "will it rain this afternoon?", "remind me to buy coffee at 6 PM.", "Yes, please."),
    "es": ("Paulina", "¿va a llover esta tarde?", "recuérdame comprar café a las seis de la tarde.", "Sí, por favor."),
    "fr": ("Thomas", "est-ce qu'il va pleuvoir cet après-midi ?", "rappelle-moi d'acheter du café à dix-huit heures.", "Oui, s'il te plaît."),
    "de": ("Anna", "wird es heute Nachmittag regnen?", "erinnere mich, um achtzehn Uhr Kaffee zu kaufen.", "Ja, bitte."),
    "it": ("Alice", "pioverà oggi pomeriggio?", "ricordami di comprare il caffè alle diciotto.", "Sì, grazie."),
    "pt": ("Luciana", "vai chover hoje à tarde?", "me lembre de comprar café às seis da tarde.", "Sim, por favor."),
    "nl": ("Xander", "gaat het vanmiddag regenen?", "herinner me eraan om zes uur 's avonds koffie te kopen.", "Ja, graag."),
    "pl": ("Zosia", "czy dziś po południu będzie padać?", "przypomnij mi, żeby kupić kawę o osiemnastej.", "Tak, proszę."),
    "tr": ("Yelda", "bu öğleden sonra yağmur yağacak mı?", "bana akşam altıda kahve almamı hatırlat.", "Evet, lütfen."),
    "id": ("Damayanti", "apakah sore ini akan hujan?", "ingatkan saya untuk membeli kopi jam enam sore.", "Ya, boleh."),
    "ms": ("Amira", "adakah hujan akan turun petang ini?", "ingatkan saya untuk membeli kopi pukul enam petang.", "Ya, boleh."),
    "vi": ("Linh", "chiều nay trời có mưa không?", "nhắc tôi mua cà phê lúc sáu giờ chiều.", "Vâng, được."),
    "ja": ("Kyoko", "今日の午後は雨が降りますか？", "夕方六時にコーヒーを買うようにリマインドして。", "はい、お願いします。"),
    "zh": ("Tingting", "今天下午会下雨吗？", "提醒我下午六点买咖啡。", "好的。"),
    "ko": ("Yuna", "오늘 오후에 비가 올까요?", "오후 여섯 시에 커피 사라고 알려줘.", "네, 좋아요."),
    "hi": ("Lekha", "क्या आज दोपहर बारिश होगी?", "मुझे शाम छह बजे कॉफ़ी खरीदने की याद दिलाना।", "हाँ, ठीक है।"),
    "bn": ("Piya", "আজ বিকেলে কি বৃষ্টি হবে?", "আমাকে সন্ধ্যা ছটায় কফি কিনতে মনে করিয়ে দিও।", "হ্যাঁ, ঠিক আছে।"),
    "ta": ("Vani", "இன்று மதியம் மழை பெய்யுமா?", "மாலை ஆறு மணிக்கு காபி வாங்க நினைவூட்டு.", "ஆமாம், சரி."),
    "te": ("Geeta", "ఈ మధ్యాహ్నం వర్షం పడుతుందా?", "సాయంత్రం ఆరు గంటలకు కాఫీ కొనమని గుర్తు చేయి.", "అవును, సరే."),
    "kn": ("Soumya", "ಇಂದು ಮಧ್ಯಾಹ್ನ ಮಳೆ ಬರುತ್ತದೆಯೇ?", "ಸಂಜೆ ಆರು ಗಂಟೆಗೆ ಕಾಫಿ ಕೊಳ್ಳಲು ನೆನಪಿಸು.", "ಹೌದು, ಸರಿ."),
    "th": ("Kanya", "บ่ายนี้ฝนจะตกไหม", "เตือนฉันให้ซื้อกาแฟตอนหกโมงเย็น", "ใช่ ได้เลย"),
    "he": ("Carmit", "האם ירד גשם אחר הצהריים?", "תזכיר לי לקנות קפה בשש בערב.", "כן, בבקשה."),
    "ar": ("Majed", "هل ستمطر بعد الظهر؟", "ذكرني بشراء القهوة في السادسة مساءً.", "نعم، من فضلك."),
}
SCRIPTS = {"ar": ("ARABIC",), "bn": ("BENGALI",), "he": ("HEBREW",), "hi": ("DEVANAGARI",), "ja": ("HIRAGANA", "KATAKANA"),
           "zh": ("CJK",), "ko": ("HANGUL",), "ta": ("TAMIL",), "te": ("TELUGU",), "kn": ("KANNADA",), "th": ("THAI",)}
WORDS = {  # common words, for telling Latin-script languages apart in a short spoken reply
    "en": "the is it will and to of in this rain afternoon be should no not today but with",
    "es": "el la es de que y en no lluvia tarde va esta hoy se con por un una muy sí",
    "fr": "le la les est de et pas il cet après-midi pluie va un une en du ne sera",
    "de": "der die das ist und nicht es heute regen nachmittag wird ein eine zu mit kein",
    "it": "il la è di e non oggi pioggia pomeriggio che un una per con sarà",
    "pt": "o a é de e não hoje chuva tarde que um uma com para vai está",
    "nl": "de het is een en niet vandaag regen middag van op te er geen wordt",
    "pl": "jest nie i w na dziś deszcz po się będzie z że dzisiaj padać",
    "tr": "ve bir bu değil yağmur bugün öğleden sonra için ile mı çok yok",
    "id": "dan yang tidak hujan hari ini sore akan di ada dengan cerah",
    "ms": "dan yang tidak hujan hari ini petang akan di ada dengan cerah",
    "vi": "không có trời mưa chiều nay là và của hôm sẽ",
}
ENGLISH = set("the it is will be and to of in this that rain afternoon should stay dry today degrees cloudy "
              "around just but with your you at for".split())


def pcm_of(voice: str, text: str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        aiff, wav = Path(d) / "a.aiff", Path(d) / "a.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", str(RATE), "-ac", "1",
                        "-c:a", "pcm_s16le", str(wav)], check=True)
        with wave.open(str(wav)) as w:
            return w.readframes(w.getnframes())


def in_language(code: str, text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if code in SCRIPTS:
        names = [unicodedata.name(c, "") for c in letters]
        share = sum(any(s in n for s in SCRIPTS[code] + (("CJK",) if code == "ja" else ())) for n in names) / len(names)
        kana = any(("HIRAGANA" in n or "KATAKANA" in n) for n in names)
        return share >= 0.5 and (kana if code == "ja" else not kana if code == "zh" else True)
    words = [w.strip(".,!?¿¡;:\"'()«»“”").lower() for w in text.split()]
    english = sum(w in ENGLISH for w in words)
    if code == "en":
        return english >= 2
    # The failure that matters is answering in English; any other Latin-script reply is read as the language.
    return english < 3 or sum(w in set(WORDS[code].split()) for w in words) > english


async def weather(tr: Trial, code: str, pcm: dict) -> dict:
    end = await tr.speak(pcm["weather"])
    t, _ = await tr.wait(done, 45, end)
    r = tr.turn(end, t)
    heard = [m for _, m in tr.events if m.get("type") == "transcript" and m.get("final")]
    woke = any(m.get("accepted") for m in heard)
    lang_ok = in_language(code, r["said"])
    return {**r, "heard": heard[-1]["text"] if heard else "", "wake": woke, "in_language": lang_ok,
            "ok": woke and "get_weather" in r["tools"] and lang_ok}


async def reminder(tr: Trial, code: str, pcm: dict) -> dict:
    end = await tr.speak(pcm["reminder"])
    asked_at, asked = await tr.wait(lambda m: m.get("type") == "confirm", 40, end)
    await asyncio.sleep(0.6)
    await tr.speak(pcm["yes"])
    t, _ = await tr.wait(done, 40, asked_at)
    r = tr.turn(end, t)
    made = any(m.get("type") == "tool_call" and m.get("name") == "create_reminder" for _, m in tr.events)
    return {**r, "question": asked.get("question", ""), "question_in_language": in_language(code, asked.get("question", "")),
            "ok": made}


async def run_one(url: str, code: str, scenario: str, pcm: dict, gate: asyncio.Semaphore) -> dict:
    async with gate:
        for attempt in (1, 2, 3):
            if attempt > 1:
                await asyncio.sleep(4 * attempt)
            try:
                async with connect(url, open_timeout=30, max_size=None) as ws:
                    await ws.send(json.dumps({"type": "hello", "client": "web", "mode": "wake", "language": code,
                                              "device_id": f"eval-{uuid.uuid4().hex[:8]}", "tz": "America/Los_Angeles",
                                              "sample_rate": RATE, "tools": ["get_location", "create_reminder"]}))
                    tr = Trial(ws, time.perf_counter())
                    tr.yes = None  # this harness answers by voice, in the language
                    reader = asyncio.create_task(read(tr))
                    await tr.wait(lambda m: m.get("type") == "ready", 30)
                    await asyncio.sleep(1.5)
                    try:
                        result = await (weather if scenario == "weather" else reminder)(tr, code, pcm)
                    finally:
                        reader.cancel()
                    return {"language": code, "scenario": scenario, "attempt": attempt, **result}
            except Exception as e:  # keep what happened, so a failure can be told apart from a dropped session
                seen = [m for _, m in getattr(locals().get("tr"), "events", [])]
                error = (f"{type(e).__name__}: {e}"[:120] + " | heard: " +
                         " / ".join(m["text"] for m in seen if m.get("type") == "transcript" and m.get("final"))[:120] +
                         " | said: " + " ".join(m["text"] for m in seen if m.get("type") == "say" and m.get("text"))[:160])
        return {"language": code, "scenario": scenario, "ok": False, "error": error, "first_s": None, "cost": 0}


async def read(tr: Trial) -> None:
    """Like Trial.read, but leaves yes/no questions to the spoken answer."""
    async for raw in tr.ws:
        msg = json.loads(raw)
        tr.events.append((time.perf_counter(), msg))
        if msg.get("type") == "tool_call":
            result = ({"latitude": 47.6062, "longitude": -122.3321, "accuracy_m": 30, "place": "Downtown, Seattle"}
                      if msg["name"] == "get_location" else {"ok": True, "list": "Reminders"})
            await tr.ws.send(json.dumps({"type": "tool_result", "id": msg["id"], "ok": True, "result": result}))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="wss://earshot-yuansshi.fly.dev/ws")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    codes = [c for c in LANGS if not a.only or c in a.only.split(",")]
    hey = pcm_of("Samantha", "Hey EveryMuse,")
    gap = b"\0\0" * int(RATE * 0.15)
    pcm = {c: {"weather": hey + gap + pcm_of(LANGS[c][0], LANGS[c][1]),
               "reminder": hey + gap + pcm_of(LANGS[c][0], LANGS[c][2]),
               "yes": pcm_of(LANGS[c][0], LANGS[c][3])} for c in codes}
    gate = asyncio.Semaphore(4)
    started = time.perf_counter()
    rows = await asyncio.gather(*(run_one(a.url, c, s, pcm[c], gate) for _ in range(a.trials) for c in codes
                                  for s in ("weather", "reminder")))
    by = {}
    for r in rows:
        by.setdefault(r["language"], []).append(r)
    table = {}
    for c, rs in by.items():
        w = [r for r in rs if r["scenario"] == "weather"]
        m = [r for r in rs if r["scenario"] == "reminder"]
        firsts = [r["first_s"] for r in w if r.get("first_s") is not None]
        table[c] = {"weather": f"{sum(r['ok'] for r in w)}/{len(w)}", "wake": f"{sum(r.get('wake', False) for r in w)}/{len(w)}",
                    "answered_in_language": f"{sum(r.get('in_language', False) for r in w)}/{len(w)}",
                    "reminder_with_spoken_yes": f"{sum(r['ok'] for r in m)}/{len(m)}",
                    "first_word_p50_s": statistics.median(firsts) if firsts else None}
    total = {"weather": f"{sum(r['ok'] for r in rows if r['scenario'] == 'weather')}/{sum(r['scenario'] == 'weather' for r in rows)}",
             "reminder": f"{sum(r['ok'] for r in rows if r['scenario'] == 'reminder')}/{sum(r['scenario'] == 'reminder' for r in rows)}",
             "cost_usd": round(sum(r.get("cost") or 0 for r in rows), 3), "minutes": round((time.perf_counter() - started) / 60, 1)}
    out = ROOT / "eval" / "results" / f"live-languages-{time.strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "url": a.url, "total": total, "by_language": table,
                               "rows": rows}, ensure_ascii=False, indent=1))
    print(json.dumps(total))
    for c, t in table.items():
        print(f"{c}  weather {t['weather']}  wake {t['wake']}  in-language {t['answered_in_language']}  "
              f"reminder {t['reminder_with_spoken_yes']}  first {t['first_word_p50_s']}")
    for r in rows:
        if not r["ok"]:
            print("FAIL", r["language"], r["scenario"], r.get("error") or "", "| heard:", r.get("heard", "")[:60],
                  "| said:", r.get("said", "")[:90], "| q:", r.get("question", "")[:60])
    print("wrote", out)


if __name__ == "__main__":
    asyncio.run(main())
