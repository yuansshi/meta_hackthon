"""Were the research reports right? An independent fact-check of the 20 Muse Wonder summaries from eval_research.py.

    uv run python scripts/eval_research_check.py    # on the server (it needs Meta's API and web search)

Muse Spark 1.3, at high effort and with web search, checks each spoken summary claim by claim: prices, places,
rules, dates, products. Each claim is supported, contradicted or unverifiable. Writes
$EARSHOT_DATA_DIR/eval-research-check.json and prints "done".
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from openai import AsyncOpenAI

from earshot.config import DATA_DIR, ROOT, api_key, load_config, model_cfg

CHECK = """You are fact-checking a short research answer that an assistant gave someone in Seattle on {date}. Use web \
search to check each concrete claim in it: prices, product names, places, opening hours, rules, dates, fares. Reply \
with JSON only: {{"claims": [{{"claim": "...", "verdict": "supported" | "contradicted" | "unverifiable", "note": "one \
short line"}}]}}. Count a price or time as supported when it's close (within about 10%) or was true very recently."""


async def main() -> None:
    cfg = load_config()
    m = model_cfg(cfg, "muse-spark")
    client = AsyncOpenAI(api_key=api_key(m["api_key_env"]), base_url=m["base_url"], timeout=240, max_retries=2)
    src = next(p for p in (ROOT / "eval" / "research-2026-11-04.json", ROOT / "eval" / "results" / "research-2026-11-04.json")
               if p.exists())
    reports = [r for r in json.loads(src.read_text())["rows"] if r.get("ok")]
    gate = asyncio.Semaphore(10)

    async def check(r: dict) -> dict:
        async with gate:
            t = time.perf_counter()
            try:
                resp = await client.responses.create(
                    model="muse-spark-1.3", reasoning={"effort": "high"}, max_output_tokens=6000,
                    tools=[{"type": "web_search"}], instructions=CHECK.format(date="November 4, 2026"),
                    input=[{"role": "user", "content": f"Task: {r['goal']}\nAnswer: {r['summary']}"}])
                text = resp.output_text or ""
                claims = json.loads(text[text.index("{"):text.rindex("}") + 1])["claims"]
                searches = sum(1 for o in resp.output if o.type == "web_search_call")
                u = resp.usage
                cost = searches * 0.0025 + ((u.input_tokens * 1.25 + u.output_tokens * 4.25) / 1e6 if u else 0)
                return {"goal": r["goal"], "claims": claims, "searches": searches, "cost": round(cost, 4),
                        "s": round(time.perf_counter() - t, 1)}
            except Exception as e:
                return {"goal": r["goal"], "error": f"{type(e).__name__}: {e}"[:160]}

    rows = await asyncio.wait_for(asyncio.gather(*(check(r) for r in reports)), 260)
    claims = [c for r in rows for c in r.get("claims", [])]
    verdicts = {v: sum(c.get("verdict") == v for c in claims) for v in ("supported", "contradicted", "unverifiable")}
    summary = {"reports": len(reports), "checked": sum("claims" in r for r in rows), "claims": len(claims), **verdicts,
               "reports_with_a_contradiction": sum(any(c.get("verdict") == "contradicted" for c in r.get("claims", []))
                                                   for r in rows),
               "cost_usd": round(sum(r.get("cost", 0) for r in rows), 3)}
    (DATA_DIR / "eval-research-check.json").write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                                   "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
