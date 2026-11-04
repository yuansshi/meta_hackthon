"""How trustworthy are the hard set's labels? A second, independent judge on the 2,400 snippets.

    uv run python scripts/eval_judge2.py    # on the server; rerun until it prints "done"

`eval_headsup_hard.py` kept the snippets where Muse Wonder's judgment matched the category a snippet was written for.
Here a different model, Muse Spark 1.3 at medium effort, judges every snippet with the same question. Agreement
between the two judges (and with the category) says how much the screener results can be trusted. Saves as it goes
to $EVERYMUSE_DATA_DIR/eval-hard-2/judged2.jsonl and writes judge-agreement.json there.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent))
from eval_headsup_hard import JUDGE, append, lines  # noqa: E402

from everymuse.config import DATA_DIR, api_key, load_config, model_cfg  # noqa: E402

OUT = DATA_DIR / "eval-hard-2"


async def main() -> None:
    cfg = load_config()
    m = model_cfg(cfg, "muse-spark")
    client = AsyncOpenAI(api_key=api_key(m["api_key_env"]), base_url=m["base_url"], timeout=120, max_retries=2)
    items = lines(OUT / "snippets.jsonl")
    path = OUT / "judged2.jsonl"
    done = {r["id"] for r in lines(path)}
    gate = asyncio.Semaphore(24)

    async def judge(it: dict) -> None:
        text = "\n".join(it["heard"]) + "\n\nTheir notes: " + ("; ".join(it["notes"]) or "none")
        async with gate:
            try:
                r = await client.responses.create(model="muse-spark-1.3", reasoning={"effort": "medium"},
                                                  max_output_tokens=3000, instructions=JUDGE,
                                                  input=[{"role": "user", "content": text}])
                word = (r.output_text or "").strip().lower().strip(".")
                u = r.usage
                cost = (u.input_tokens * 1.25 + u.output_tokens * 4.25) / 1e6 if u else 0.0
            except Exception as e:
                word, cost = f"error: {type(e).__name__}", 0.0
        append(path, {"id": it["id"], "judge": word, "cost": cost})

    await asyncio.wait_for(asyncio.gather(*(judge(it) for it in items if it["id"] not in done)), 250)
    first = {r["id"]: r["judge"] for r in lines(OUT / "judged.jsonl")}
    second = {r["id"]: r["judge"] for r in lines(path)}
    expect = {it["id"]: it["expect"] for it in items}
    both = [i for i in expect if i in first and i in second]
    summary = {"snippets": len(items), "judged_by_both": len(both),
               "judges_agree": sum(first[i] == second[i] for i in both),
               "second_judge_agrees_with_category": sum(second[i] == expect[i] for i in both),
               "all_three_agree": sum(first[i] == second[i] == expect[i] for i in both),
               "second_judge_cost_usd": round(sum(r.get("cost", 0) for r in lines(path)), 2)}
    (OUT / "judge-agreement.json").write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **summary}, indent=1))
    print(json.dumps(summary, indent=1))
    print("done" if len(both) == len(items) else "not finished: run again")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except TimeoutError:
        print("not finished: run again")
