"""Does Heads-up work in other languages? The labeled snippets in eval/overheard.json, translated, then screened.

    uv run python scripts/eval_headsup_languages.py --languages es,fr,de,pt,ja,zh,ko,hi,ar --repeats 2

Run it on the server (fly ssh console), which reaches Meta's model API.
1. Muse Spark 1.3 translates each snippet, and the user's notes, into the language. Names, numbers and addresses
   stay as they'd be said there. Translations are cached in eval/overheard-<code>.json.
2. The shipped screener (models.yaml heads_up.model, same prompt, told the user's language) screens each one.
3. Reports precision and recall per language, and whether each offer is written in that language.
Writes eval/results/headsup-languages-<date>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import unicodedata

from openai import AsyncOpenAI

from earshot.config import DATA_DIR, ROOT, api_key, load_config, model_cfg
from earshot.headsup import screen
from earshot.languages import resolve

TZ = "America/Los_Angeles"
SCRIPTS = {"ja": ("HIRAGANA", "KATAKANA", "CJK"), "zh": ("CJK",), "ko": ("HANGUL",), "hi": ("DEVANAGARI",),
           "mr": ("DEVANAGARI",), "ar": ("ARABIC",), "he": ("HEBREW",), "th": ("THAI",), "bn": ("BENGALI",),
           "ta": ("TAMIL",), "te": ("TELUGU",), "kn": ("KANNADA",)}
WORDS = {"es": "el la de que y en un una por para con los las del se te tu", "fr": "le la de et un une les des du pour à te",
         "de": "der die das und ein eine zu den dich mit für am um", "pt": "o a de que e um uma para com os as do da te",
         "nl": "de het een en van ik je dat niet op te is om voor wil", "it": "il la di e un una per che non mi ti vuoi alle",
         "id": "di dan yang untuk ini itu saya kamu ke akan ada dengan mau",
         "ms": "di dan yang untuk ini itu saya anda ke akan ada dengan nak", "pl": "i w na z do nie się że to czy o jest",
         "tl": "ang ng sa mga na ay at ko mo ba", "tr": "ve bir bu için de da mı mi ile ne",
         "vi": "và của là có không cho tôi bạn này một lúc"}


def in_language(code: str, text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if code in SCRIPTS:
        share = sum(any(s in unicodedata.name(c, "") for s in SCRIPTS[code]) for c in letters) / len(letters)
        return share >= 0.4
    words = [w.strip(".,!?¿¡;:\"'()«»“”").lower() for w in text.split()]
    english = sum(w in {"the", "to", "you", "your", "at", "on", "a", "for", "want", "me", "remind"} for w in words)
    return sum(w in set(WORDS.get(code, "").split()) for w in words) > english


async def translate(client: AsyncOpenAI, item: dict, name: str) -> dict:
    src = {"heard": item["heard"], "notes": item.get("notes", [])}
    r = await client.responses.create(
        model="muse-spark-1.3", reasoning={"effort": "low"}, max_output_tokens=2000,
        instructions=f"Translate the JSON's strings into natural, spoken {name}, as people nearby would actually say "
                     "them. Keep names, numbers, times and street addresses as a speaker of that language would say "
                     "them. Reply with the same JSON shape only.",
        input=[{"role": "user", "content": json.dumps(src, ensure_ascii=False)}])
    text = r.output_text or ""
    out = json.loads(text[text.index("{"):text.rindex("}") + 1])
    return {**item, "heard": out["heard"], "notes": out.get("notes", [])}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", default="es,fr,de,pt,ja,zh,ko,hi,ar")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--set", default="overheard", help="eval/<set>.json; translations go to eval/<set>-<code>.json")
    args = ap.parse_args()
    cfg = load_config()
    items = json.loads((ROOT / "eval" / f"{args.set}.json").read_text())
    brain = model_cfg(cfg, "muse-spark")
    client = AsyncOpenAI(api_key=api_key(brain["api_key_env"]), base_url=brain["base_url"], timeout=120, max_retries=2)
    m = model_cfg(cfg, cfg["heads_up"]["model"])
    gate = asyncio.Semaphore(args.concurrency)
    codes = args.languages.split(",")

    async def translated(code: str) -> list[dict]:
        path = (ROOT / "eval" / f"{args.set}-{code}.json") if args.set == "overheard" else (DATA_DIR / "eval-lang" / f"{args.set}-{code}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return json.loads(path.read_text())

        async def one(item):
            async with gate:
                return await translate(client, item, resolve(code).name)
        out = await asyncio.gather(*(one(it) for it in items))
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        return out

    sets = dict(zip(codes, await asyncio.gather(*(translated(c) for c in codes))))

    async def screen_one(code: str, item: dict, rep: int) -> dict:
        heard = [f"{3 * (len(item['heard']) - i)} s ago: {line}" for i, line in enumerate(item["heard"])]
        async with gate:
            t = time.perf_counter()
            try:
                offer, cost, _ = await screen(client, m, heard, item.get("notes", []), [], TZ,
                                              cfg["heads_up"]["max_output_tokens"], m.get("reasoning_effort", "minimal"),
                                              resolve(code).name)
                error = None
            except Exception as e:
                offer, cost, error = None, 0.0, f"{type(e).__name__}: {e}"[:160]
        return {"language": code, "id": item["id"], "rep": rep, "expect": item["expect"], "offered": offer is not None,
                "offer": offer.offer if offer else None, "request": offer.request if offer else None,
                "offer_in_language": in_language(code, offer.offer) if offer else None,
                "s": round(time.perf_counter() - t, 3), "cost": cost, "error": error}

    t0 = time.perf_counter()
    rows = await asyncio.gather(*(screen_one(c, it, rep) for c in codes for it in sets[c] for rep in range(args.repeats)))
    summary = {}
    for c in codes:
        rs = [r for r in rows if r["language"] == c]
        tp = sum(r["offered"] and r["expect"] == "offer" for r in rs)
        fp = sum(r["offered"] and r["expect"] == "none" for r in rs)
        pos = sum(r["expect"] == "offer" for r in rs)
        offers = [r for r in rs if r["offered"]]
        summary[c] = {"precision": round(tp / (tp + fp), 3) if tp + fp else 0.0, "recall": round(tp / pos, 3) if pos else 0.0,
                      "false_offers": fp, "offers_in_language": f"{sum(bool(r['offer_in_language']) for r in offers)}/{len(offers)}",
                      "p50_s": statistics.median(r["s"] for r in rs if not r["error"]), "errors": sum(bool(r["error"]) for r in rs),
                      "cost_per_screen_usd": round(statistics.mean(r["cost"] for r in rs), 5)}
    out = (ROOT / "eval" / "results" / f"headsup-languages-{time.strftime('%Y-%m-%d')}.json") if args.set == "overheard" else (DATA_DIR / "eval-lang" / f"headsup-{args.set}-{time.strftime('%Y-%m-%d')}.json")
    if out.exists():  # an earlier run's other languages stay in the file
        old = json.loads(out.read_text())
        rows = [r for r in old["rows"] if r["language"] not in codes] + rows
        summary = {**{c: s for c, s in old["summary"].items() if c not in codes}, **summary}
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": m["id"], "items": len(items),
                               "repeats": args.repeats, "wall_s": round(time.perf_counter() - t0, 1), "summary": summary,
                               "rows": rows}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        if r["offered"] != (r["expect"] == "offer") or r["offer_in_language"] is False:
            print("MISS", r["language"], r["id"], r["expect"], "|", r["offer"])
    print("wrote", out)
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
