"""A bigger, harder test set for the Heads-up screener: Muse Wonder writes new labeled snippets, then the shipped
screener (Muse Spark 1.2) is scored on them.

    uv run python scripts/eval_headsup_more.py --per-category 10 --repeats 2   # on the server (fly ssh console)

The 40 hand-written snippets in eval/overheard.json all passed, so this asks a different, deeper model for new ones,
including near-misses meant to trip the screener: plans with no time, other people's trains, a store announcement.
Labels come from the category each snippet was written for. Writes eval/overheard-generated.json and
eval/results/headsup-generated-<date>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

from openai import AsyncOpenAI

from earshot import headsup
from earshot.config import DATA_DIR, ROOT, api_key, load_config, model_cfg
from earshot.headsup import screen

TZ = "America/Los_Angeles"
OUT = DATA_DIR / "eval-headsup"  # on the server's volume, so a trial restart can't lose it
# v2 is shipped (headsup.PROMPT): it says other people's plans and staff announcements aren't the listener's.
# v1 is the prompt before that change, rebuilt here so the comparison can be rerun.
V2 = headsup.PROMPT
V1 = (V2.replace("- a plan they are part of, with a time,", "- a plan with a time,")
      .replace("- a detail someone gave them that's worth keeping,", "- a detail worth keeping,")
      .replace("- an announcement that changes a plan in their notes", "- an announcement that changes their plans, judging by their notes")
      .replace("plans with no time or place yet, other people's plans, trains or appointments, and announcements for staff "
               "or for other travelers.", "plans with no time or place yet."))


CATEGORIES = {  # name: (expect, kinds, what to write)
    "plan": ("offer", ["reminder"], "a friend or family member mentions a plan the listener is part of, with a day or time and a place"),
    "ask": ("offer", ["reminder", "note"], "someone asks the listener to do, bring, buy or remember something specific"),
    "detail": ("offer", ["note"], "someone tells the listener a detail worth keeping: a phone number, a code, a name, a gate, a locker, where something is"),
    "announce": ("offer", ["alert"], "a station, airport or venue announcement that changes a plan in the listener's notes (write the notes too)"),
    "lookup": ("offer", ["lookup", "research"], "someone asks the listener to find something out or look into something concrete"),
    "smalltalk": ("none", [], "ordinary small talk with nothing to act on"),
    "media": ("none", [], "a TV show, radio ad or video playing nearby"),
    "vague": ("none", [], "a vague plan with no day, time or place yet, like 'we should get dinner sometime'"),
    "past": ("none", [], "people talking about something that already happened"),
    "others": ("none", [], "a plan, train or appointment that belongs to other people, not the listener (give the listener's notes, which don't match)"),
    "irrelevant": ("none", [], "an announcement that doesn't affect anything in the listener's notes (write the notes too)"),
    "store": ("none", [], "a store, office or public announcement meant for staff, like 'cleanup on aisle five'"),
}
PROMPT = """Write {n} different snippets of speech that a person's microphone might overhear nearby, for testing an \
assistant that offers help only when there is clearly something the listener would want.
Category: {what}.
Vary the people, places, wording and length (one to three lines each). Make some of them subtle. Use made-up names, \
numbers and places. Reply with JSON only: {{"snippets": [{{"heard": ["line", ...], "notes": ["the listener's own \
notes, if any"]}}, ...]}}"""


async def generate(client: AsyncOpenAI, name: str, n: int) -> list[dict]:
    expect, kinds, what = CATEGORIES[name]
    r = await client.responses.create(model="muse-wonder-1.0", reasoning={"effort": "low"}, max_output_tokens=6000,
                                      input=[{"role": "user", "content": PROMPT.format(n=n, what=what)}])
    text = r.output_text or ""
    items = json.loads(text[text.index("{"):text.rindex("}") + 1])["snippets"]
    return [{"id": f"{name}-{k}", "expect": expect, "kinds": kinds, "heard": it["heard"], "notes": it.get("notes") or []}
            for k, it in enumerate(items)]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--prompts", default="v1,v2")
    a = ap.parse_args()
    cfg = load_config()
    brain = model_cfg(cfg, "muse-spark")
    client = AsyncOpenAI(api_key=api_key(brain["api_key_env"]), base_url=brain["base_url"], timeout=240, max_retries=2)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "overheard-generated.json"
    if path.exists():
        items = json.loads(path.read_text())
    else:
        batches = await asyncio.gather(*(generate(client, c, a.per_category) for c in CATEGORIES), return_exceptions=True)
        items = [it for b in batches if isinstance(b, list) for it in b]
        path.write_text(json.dumps(items, indent=1))
    m = model_cfg(cfg, cfg["heads_up"]["model"])
    gate = asyncio.Semaphore(24)
    sets = {"original": json.loads((ROOT / "eval" / "overheard.json").read_text()), "generated": items}
    prompts = {"v1": V1, "v2": V2}

    async def one(set_name: str, item: dict, rep: int) -> dict:
        heard = [f"{3 * (len(item['heard']) - i)} s ago: {line}" for i, line in enumerate(item["heard"])]
        async with gate:
            t = time.perf_counter()
            try:
                offer, cost, _ = await screen(client, m, heard, item.get("notes") or [], [], TZ,
                                              cfg["heads_up"]["max_output_tokens"], m.get("reasoning_effort", "minimal"))
                error = None
            except Exception as e:
                offer, cost, error = None, 0.0, f"{type(e).__name__}: {e}"[:160]
        return {"set": set_name, "id": item["id"], "rep": rep, "expect": item["expect"], "offered": offer is not None,
                "kind": offer.kind if offer else None, "offer": offer.offer if offer else None,
                "s": round(time.perf_counter() - t, 3), "cost": cost, "error": error}

    results, summary = {}, {}
    for version in a.prompts.split(","):
        headsup.PROMPT = prompts[version]  # screen() reads the module's prompt; one version at a time
        rows = await asyncio.gather(*(one(sn, it, rep) for sn, its in sets.items() for it in its for rep in range(a.repeats)))
        results[version] = rows
        for sn in sets:
            rs = [r for r in rows if r["set"] == sn]
            tp = sum(r["offered"] and r["expect"] == "offer" for r in rs)
            fp = sum(r["offered"] and r["expect"] == "none" for r in rs)
            pos, neg = sum(r["expect"] == "offer" for r in rs), sum(r["expect"] == "none" for r in rs)
            by = {c: f"{sum(r['offered'] for r in rs if r['id'].startswith(c + '-'))}/"
                     f"{sum(1 for r in rs if r['id'].startswith(c + '-'))}" for c in CATEGORIES} if sn == "generated" else {}
            summary[f"{version}/{sn}"] = {"screenings": len(rs), "precision": round(tp / (tp + fp), 3) if tp + fp else 0,
                                          "recall": round(tp / pos, 3) if pos else 0,
                                          "false_offer_rate": round(fp / neg, 3) if neg else 0,
                                          "p50_s": statistics.median(r["s"] for r in rs),
                                          "errors": sum(bool(r["error"]) for r in rs), "offered_by_category": by}
    headsup.PROMPT = V2
    out = OUT / f"headsup-generated-{time.strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": m["id"], "v2_prompt": V2,
                               "summary": summary, "rows": results}, indent=1))
    print(json.dumps(summary, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    asyncio.run(main())
