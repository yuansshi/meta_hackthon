"""Heads-up on a large, hard, cross-checked test set: does a deeper model screen better than the shipped one?

    uv run python scripts/eval_headsup_hard.py --per-category 40    # on the server; rerun until it says "done"

1. Muse Wonder writes snippets in 12 categories (eval_headsup_more.CATEGORIES), including near-misses.
2. A second Muse Wonder pass judges each one on its own: would the listener clearly want an offer? Only snippets
   where the judge agrees with the category's label are kept, so a mislabeled snippet can't count against a screener.
3. Four screeners, all with the shipped prompt, screen the kept snippets once each: Muse Spark 1.2 minimal (shipped),
   Muse Spark 1.3 minimal, Muse Wonder low, and Muse Spark Open on Fireworks.
Every stage saves to $EARSHOT_DATA_DIR/eval-hard/ as it goes, so a Fly trial restart loses nothing: rerun the
same command and it picks up where it stopped. Writes summary.json there.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent))
from eval_headsup_more import CATEGORIES, PROMPT  # noqa: E402

from earshot.config import DATA_DIR, api_key, load_config, model_cfg  # noqa: E402
from earshot.headsup import screen  # noqa: E402

TZ = "America/Los_Angeles"
OUT = DATA_DIR / "eval-hard"  # --out picks another folder under the data dir
JUDGE = """Someone's microphone overheard the speech below; none of it was said to their assistant. Their own notes \
are listed too. Would this person clearly want their assistant to offer help right now (a reminder, a note, a \
lookup, or an alert about their own plans)? Count only things that are theirs: said to them, asked of them, or \
changing a plan in their notes. Other people's plans, staff announcements, TV, small talk and vague or past plans \
are not. Reply with one word: offer or none."""
SCREENERS = [  # label, models.yaml key, overrides, effort
    ("Muse Spark 1.2, minimal (shipped)", "muse-spark-screen", {}, "minimal"),
    ("Muse Spark 1.3, minimal", "muse-spark", {}, "minimal"),
    ("Muse Wonder 1.0, low", "muse-wonder", {}, "low"),
    ("Muse Spark Open on Fireworks, low", "muse-spark-open", {}, "low"),
]


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def append(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=40)
    ap.add_argument("--out", default="eval-hard")
    a = ap.parse_args()
    global OUT
    OUT = DATA_DIR / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    clients: dict[str, AsyncOpenAI] = {}

    def client(m: dict) -> AsyncOpenAI:
        if m["provider"] not in clients:
            clients[m["provider"]] = AsyncOpenAI(api_key=api_key(m["api_key_env"]), base_url=m["base_url"],
                                                 timeout=240, max_retries=2)
        return clients[m["provider"]]

    meta = client(model_cfg(cfg, "muse-spark"))
    gate = asyncio.Semaphore(24)

    # 1. generate, 20 at a time per call
    gen = OUT / "snippets.jsonl"
    have = lines(gen)
    counts = {c: sum(r["category"] == c for r in have) for c in CATEGORIES}

    async def write(c: str, batch: int) -> None:
        async with gate:
            r = await meta.responses.create(model="muse-wonder-1.0", reasoning={"effort": "low"}, max_output_tokens=8000,
                                            input=[{"role": "user", "content": PROMPT.format(n=20, what=CATEGORIES[c][2])
                                                    + f"\nThis is batch {batch}: make every snippet different from "
                                                      "earlier batches."}])
        text = r.output_text or ""
        for k, it in enumerate(json.loads(text[text.index("{"):text.rindex("}") + 1])["snippets"]):
            key = " ".join(str(it["heard"][0]).lower().split())[:80] if it.get("heard") else ""
            if not key or key in seen:
                continue  # a repeat of an earlier snippet
            seen.add(key)
            append(gen, {"id": f"{c}-{batch}-{k}", "category": c, "expect": CATEGORIES[c][0], "heard": it["heard"],
                         "notes": it.get("notes") or []})

    seen = {" ".join(str(r["heard"][0]).lower().split())[:80] for r in have if r.get("heard")}
    batches_done = {c: len({r["id"].split("-")[1] for r in have if r["category"] == c}) for c in CATEGORIES}
    need = [(c, b) for c in CATEGORIES for b in range(batches_done[c], a.per_category // 20)]
    if need:
        await asyncio.gather(*(write(c, b) for c, b in need), return_exceptions=True)
    items = lines(gen)

    # 2. judge each snippet on its own
    jpath = OUT / "judged.jsonl"
    judged = {r["id"]: r for r in lines(jpath)}

    async def judge(it: dict) -> None:
        text = "\n".join(it["heard"]) + "\n\nTheir notes: " + ("; ".join(it["notes"]) or "none")
        async with gate:
            try:
                r = await meta.responses.create(model="muse-wonder-1.0", reasoning={"effort": "medium"},
                                                max_output_tokens=2000, instructions=JUDGE,
                                                input=[{"role": "user", "content": text}])
                word = (r.output_text or "").strip().lower().strip(".")
            except Exception as e:
                word = f"error: {type(e).__name__}"
        append(jpath, {"id": it["id"], "judge": word})

    await asyncio.gather(*(judge(it) for it in items if it["id"] not in judged))
    judged = {r["id"]: r["judge"] for r in lines(jpath)}
    kept = [it for it in items if judged.get(it["id"]) == it["expect"]]

    # 3. screen the kept snippets with each screener, once
    spath = OUT / "screened.jsonl"
    done = {(r["config"], r["id"]) for r in lines(spath)}

    async def one(label: str, m: dict, effort: str, it: dict) -> None:
        heard = [f"{3 * (len(it['heard']) - i)} s ago: {line}" for i, line in enumerate(it["heard"])]
        async with gate:
            t = time.perf_counter()
            try:
                offer, cost, _ = await asyncio.wait_for(screen(client(m), m, heard, it["notes"], [], TZ,
                                                               cfg["heads_up"]["max_output_tokens"], effort), 60)
                error = None
            except Exception as e:
                offer, cost, error = None, 0.0, f"{type(e).__name__}: {e}"[:120]
        append(spath, {"config": label, "id": it["id"], "category": it["category"], "expect": it["expect"],
                       "offered": offer is not None, "offer": offer.offer if offer else None,
                       "s": round(time.perf_counter() - t, 3), "cost": cost, "error": error})

    jobs = []
    for label, key, over, effort in SCREENERS:
        m = {**model_cfg(cfg, key), **over}
        jobs += [one(label, m, effort, it) for it in kept if (label, it["id"]) not in done]
    await asyncio.gather(*jobs)

    rows = lines(spath)
    summary = {"snippets": len(items), "judge_agreed": len(kept),
               "judge_disagreed_by_category": {c: sum(1 for it in items if it["category"] == c and
                                                      judged.get(it["id"]) not in (None, it["expect"])) for c in CATEGORIES},
               "screeners": {}}
    for label, *_ in SCREENERS:
        rs = [r for r in rows if r["config"] == label and not r["error"]]
        tp = sum(r["offered"] and r["expect"] == "offer" for r in rs)
        fp = sum(r["offered"] and r["expect"] == "none" for r in rs)
        pos, neg = sum(r["expect"] == "offer" for r in rs), sum(r["expect"] == "none" for r in rs)
        summary["screeners"][label] = {
            "screened": len(rs), "precision": round(tp / (tp + fp), 3) if tp + fp else 0, "recall": round(tp / pos, 3) if pos else 0,
            "false_offer_rate": round(fp / neg, 3) if neg else 0, "p50_s": statistics.median(r["s"] for r in rs) if rs else None,
            "cost_per_screen": round(statistics.mean(r["cost"] for r in rs), 5) if rs else None,
            "errors": sum(1 for r in rows if r["config"] == label and r["error"]),
            "false_offers_by_category": {c: sum(1 for r in rs if r["category"] == c and r["offered"] and r["expect"] == "none")
                                         for c in CATEGORIES if CATEGORIES[c][0] == "none"}}
    complete = all(v["screened"] == len(kept) for v in summary["screeners"].values())
    (OUT / "summary.json").write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "complete": complete,
                                                  **summary}, indent=1))
    print(json.dumps(summary, indent=1))
    print("done" if complete else "not finished: run again")


if __name__ == "__main__":
    asyncio.run(main())
