"""Which model should screen overheard speech for Heads-up? A labeled comparison.

Runs the shipped screener (`earshot.headsup.screen`, same prompt) over `eval/overheard.json`: 20 snippets that
deserve an offer and 20 that don't. It tries several Meta models and settings, then reports precision, recall,
latency and cost.

    uv run python scripts/eval_headsup.py --repeats 2 --out results/headsup-eval.json

Needs a network that reaches Meta's model API (run it on the server: `fly machine exec`). Rows are appended to
`<out>.rows.jsonl` as they finish, so a stopped machine loses nothing already measured.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI

from earshot.config import ROOT, api_key, load_config, model_cfg
from earshot.headsup import screen

TZ = "America/Los_Angeles"
CONTRIBUTOR = {"price_in": 0.10, "price_cached_in": 0.002, "price_out": 0.20}
CONFIGS = [  # label, model key in models.yaml, overrides, reasoning effort
    ("Muse Spark 1.3, minimal (shipped)", "muse-spark", {}, "minimal"),
    ("Muse Spark 1.3, low", "muse-spark", {}, "low"),
    ("Muse Spark 1.2, minimal", "muse-spark", {"id": "muse-spark-1.2"}, "minimal"),
    ("Muse Spark 1.1, minimal", "muse-spark", {"id": "muse-spark-1.1"}, "minimal"),
    ("Muse Spark 1.3 Contributor, minimal", "muse-spark", {"id": "muse-spark-1.3-contributor", **CONTRIBUTOR}, "minimal"),
    ("Muse Spark Open on Fireworks, low", "muse-spark-open", {}, "low"),
    ("Muse Wonder 1.0, low", "muse-wonder", {}, "low"),
]


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=24)  # the 512 MB server runs this beside the app
    ap.add_argument("--max-output-tokens", type=int, default=1200)
    ap.add_argument("--only", help="run configs whose label contains this text")
    ap.add_argument("--out", default=str(ROOT / "results" / "headsup-eval.json"))
    args = ap.parse_args()
    cfg = load_config()
    items = json.loads((ROOT / "eval" / "overheard.json").read_text())
    configs = [c for c in CONFIGS if not args.only or args.only.lower() in c[0].lower()]
    clients: dict[str, AsyncOpenAI] = {}
    gate = asyncio.Semaphore(args.concurrency)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_path = out.with_suffix(".rows.jsonl")
    rows_path.write_text("")

    def client_for(m: dict) -> AsyncOpenAI:  # one client per provider: each carries its own connection pool
        if m["provider"] not in clients:
            clients[m["provider"]] = AsyncOpenAI(api_key=api_key(m["api_key_env"]), base_url=m["base_url"],
                                                 timeout=120, max_retries=1)
        return clients[m["provider"]]

    async def one(label: str, m: dict, effort: str, item: dict, rep: int) -> dict:
        client = client_for(m)
        heard = [f"{3 * (len(item['heard']) - i)} s ago: {line}" for i, line in enumerate(item["heard"])]
        async with gate:
            t = time.perf_counter()
            try:
                offer, cost, raw = await screen(client, m, heard, item.get("notes", []), [], TZ,
                                                args.max_output_tokens, effort)
                error = None
            except Exception as e:  # recorded, not fatal: an error is a missed offer in production too
                offer, cost, raw, error = None, 0.0, "", f"{type(e).__name__}: {e}"[:160]
            row = {"config": label, "id": item["id"], "rep": rep, "expect": item["expect"],
                   "kinds": item.get("kinds", []), "offered": offer is not None,
                   "kind": offer.kind if offer else None, "offer": offer.offer if offer else None,
                   "request": offer.request if offer else None, "empty": not raw.strip(), "error": error,
                   "s": round(time.perf_counter() - t, 3), "cost": cost}
        with rows_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    jobs = []
    for label, key, overrides, effort in configs:
        m = {**model_cfg(cfg, key), **overrides}
        jobs += [one(label, m, effort, item, rep) for rep in range(args.repeats) for item in items]
    t0 = time.perf_counter()
    rows = await asyncio.gather(*jobs)
    summary = []
    for label, *_ in configs:
        rs = [r for r in rows if r["config"] == label]
        tp = sum(r["offered"] and r["expect"] == "offer" for r in rs)
        fp = sum(r["offered"] and r["expect"] == "none" for r in rs)
        fn = sum(not r["offered"] and r["expect"] == "offer" for r in rs)
        positives, negatives = tp + fn, sum(r["expect"] == "none" for r in rs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        kind_ok = sum(r["offered"] and r["expect"] == "offer" and r["kind"] in r["kinds"] for r in rs)
        lat = [r["s"] for r in rs if not r["error"]]
        mean_cost = statistics.mean(r["cost"] for r in rs)
        summary.append({
            "config": label, "calls": len(rs), "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
            "false_offer_rate": round(fp / negatives, 3) if negatives else 0.0,
            "kind_accuracy": round(kind_ok / tp, 3) if tp else 0.0,
            "p50_s": round(pct(lat, 0.5), 2), "p90_s": round(pct(lat, 0.9), 2),
            "cost_per_screen_usd": round(mean_cost, 5), "cost_per_hour_usd": round(mean_cost * 60, 3),
            "empty": sum(r["empty"] and not r["error"] for r in rs), "errors": sum(bool(r["error"]) for r in rs),
        })
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "items": len(items),
                               "repeats": args.repeats, "wall_s": round(time.perf_counter() - t0, 1),
                               "summary": summary}, indent=1))
    print("| Screener | Precision | Recall | False offers | Kind right | p50 | p90 | $ per screen | $ per hour* |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in summary:
        print(f"| {s['config']} | {s['precision']:.0%} | {s['recall']:.0%} | {s['false_offer_rate']:.0%} | "
              f"{s['kind_accuracy']:.0%} | {s['p50_s']} s | {s['p90_s']} s | ${s['cost_per_screen_usd']:.4f} | "
              f"${s['cost_per_hour_usd']:.2f} |  (errors {s['errors']}, empty {s['empty']})")
    print(f"* at 60 screenings an hour of nearby conversation. {len(rows)} calls in {time.perf_counter() - t0:.0f} s.")


if __name__ == "__main__":
    asyncio.run(main())
