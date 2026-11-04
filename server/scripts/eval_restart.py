"""A judge's session across a server restart, as the Fly trial forces every 5 minutes.

    uv run python scripts/eval_restart.py --trials 3

Each trial, on one device id, like a browser tab:
1. "My name is Priya, and I'm in Seattle for the weekend."
2. "Look into the best umbrellas for windy cities and get back to me." (a Muse Wonder job starts)
3. The machine is stopped (`fly machine stop`), dropping the connection mid-job.
4. The client reconnects, as the page does; the request starts the machine again.
5. "What's my name?" It passes if the answer says Priya: the conversation outlived the restart.
6. The research report must still arrive: the job ran on at Meta, and the restarted server collected it.
Writes eval/results/restart-<date>.json. Needs flyctl, logged in.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
import uuid
from pathlib import Path

from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]
APP = "earshot-yuansshi"
URL = f"wss://{APP}.fly.dev/ws"


def machine_ids() -> list[str]:
    out = subprocess.run(["fly", "machines", "list", "--app", APP, "--json"], capture_output=True, text=True, check=True)
    return [m["id"] for m in json.loads(out.stdout)]


def stop_machines() -> None:
    """Stops every machine and waits until they really have: `fly machine stop` returns while it's still stopping."""
    for mid in machine_ids():
        subprocess.run(["fly", "machine", "stop", mid, "--app", APP], capture_output=True, text=True)
    for _ in range(60):
        out = subprocess.run(["fly", "machines", "list", "--app", APP, "--json"], capture_output=True, text=True)
        if all(m["state"] in ("stopped", "suspended") for m in json.loads(out.stdout or "[]")):
            return
        time.sleep(2)


class Conn:
    def __init__(self, ws):
        self.ws, self.events = ws, []

    async def read(self) -> None:
        try:
            async for raw in self.ws:
                self.events.append((time.perf_counter(), json.loads(raw)))
        except Exception:
            pass

    async def wait(self, test, timeout: float, after: float = 0.0):
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            for t, m in self.events:
                if t >= after and test(m):
                    return t, m
            await asyncio.sleep(0.05)
        raise TimeoutError

    def said(self, after: float, kind=None) -> str:
        return " ".join(m["text"] for t, m in self.events if t >= after and m.get("type") == "say" and m.get("text"))


async def open_session(device: str) -> tuple[object, Conn, asyncio.Task, float]:
    t = time.perf_counter()
    for attempt in range(15):  # like the page: a restarting machine can turn the first tries away (503)
        try:
            ws = await connect(URL, open_timeout=90, max_size=None)
            break
        except Exception:
            if attempt == 14:
                raise
            await asyncio.sleep(1.0)
    await ws.send(json.dumps({"type": "hello", "client": "web", "mode": "ptt", "device_id": device,
                              "tz": "America/Los_Angeles", "sample_rate": 16000, "tools": []}))
    c = Conn(ws)
    task = asyncio.create_task(c.read())
    await c.wait(lambda m: m.get("type") == "ready", 90)
    return ws, c, task, time.perf_counter() - t


def done(m: dict, background: bool = False) -> bool:
    return m.get("type") == "turn_done" and bool(m.get("background")) == background


async def trial(k: int) -> dict:
    device = f"restart-{uuid.uuid4().hex[:8]}"
    row: dict = {"trial": k}
    ws, c, task, _ = await open_session(device)
    sent = time.perf_counter()
    await ws.send(json.dumps({"type": "text", "text": "My name is Priya, and I'm in Seattle for the weekend."}))
    await c.wait(lambda m: done(m), 40, sent)
    sent = time.perf_counter()
    await ws.send(json.dumps({"type": "text", "text": "Look into the best umbrellas for windy cities and get back to me."}))
    await c.wait(lambda m: done(m), 40, sent)
    await asyncio.sleep(2)
    stopped = time.perf_counter()
    stop_machines()  # what the trial does every 5 minutes
    row["stop_s"] = round(time.perf_counter() - stopped, 1)
    task.cancel()
    ws.transport.abort()  # the connection just dies, as a browser sees it: no clean close from the client
    ws, c, task, row["reconnect_s"] = await open_session(device)  # the page reconnects; this starts the machine
    sent = time.perf_counter()
    await ws.send(json.dumps({"type": "text", "text": "What's my name?"}))
    t, _ = await c.wait(lambda m: done(m), 40, sent)
    row["answer"] = c.said(sent)
    row["remembered"] = "priya" in row["answer"].lower()
    try:
        t, m = await c.wait(lambda m: done(m, background=True), 240, 0)
        row["report"] = c.said(t - 120)[-200:]
        row["report_after_restart"] = True
        row["report_s_after_stop"] = round(t - stopped, 1)
    except TimeoutError:
        row["report_after_restart"] = False
    task.cancel()
    await ws.close()
    return row


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    a = ap.parse_args()
    rows = []
    for k in range(a.trials):
        try:
            rows.append(await trial(k))
        except Exception as e:
            rows.append({"trial": k, "error": f"{type(e).__name__}: {e}"[:200]})
        print(json.dumps(rows[-1]), flush=True)
    summary = {"trials": len(rows), "remembered": sum(bool(r.get("remembered")) for r in rows),
               "report_after_restart": sum(bool(r.get("report_after_restart")) for r in rows),
               "reconnect_s": sorted(r["reconnect_s"] for r in rows if "reconnect_s" in r)}
    out = ROOT / "eval" / "results" / f"restart-{time.strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary), "\nwrote", out)


if __name__ == "__main__":
    asyncio.run(main())
