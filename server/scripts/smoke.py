"""End-to-end check of a running server, local or deployed: talk to it the way a device does.

    uv run python scripts/smoke.py --url wss://earshot-yuansshi.fly.dev/ws      # spoken: say → PCM → server
    uv run python scripts/smoke.py --url ws://localhost:8000/ws --text "Will it rain in Seattle?"
    uv run python scripts/smoke.py --photo                                      # a test image plus a typed question

Streams speech synthesized by `say` as real-time 16 kHz PCM (then silence, like an open mic), answers
device tool calls with canned results, and prints every event with timings. Exits 1 if no answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import base64

from probe import synth_pcm  # same directory
from vision_check import two_color_png
from websockets.asyncio.client import connect

CANNED = {
    "get_location": {"latitude": 47.6062, "longitude": -122.3321, "accuracy_m": 30, "place": "Downtown, Seattle"},
    "create_reminder": {"ok": True, "list": "Reminders (smoke test)"},
}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="wss://earshot-yuansshi.fly.dev/ws")
    ap.add_argument("--say", default="Hey EveryMuse, will it rain in Seattle this afternoon?")
    ap.add_argument("--text", help="send typed text instead of speech")
    ap.add_argument("--photo", action="store_true", help="attach a left-red, right-blue test image to the question")
    ap.add_argument("--wait-report", action="store_true", help="keep listening for a background research report")
    ap.add_argument("--timeout", type=float, default=45)
    args = ap.parse_args()
    if args.photo and not args.text:
        args.text = "What colors are the left and right halves of this image?"
    pcm = None if args.text else synth_pcm(args.say, 16000)
    t0 = time.perf_counter()
    marks: dict[str, float] = {}
    done = asyncio.Event()
    answered = False

    def stamp() -> str:
        return f"{time.perf_counter() - t0:6.2f}s"

    async with connect(args.url, open_timeout=15, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "client": "web", "device_id": "smoke", "mode": "wake",
                                  "tz": "America/Los_Angeles", "sample_rate": 16000,
                                  "tools": ["get_location", "create_reminder"]}))

        async def reader() -> None:
            nonlocal answered
            async for raw in ws:
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "transcript" and not msg.get("final"):
                    continue  # partials are noisy; the final one is printed
                if kind == "transcript":
                    marks.setdefault("transcript", time.perf_counter())
                if kind == "say" and msg.get("text"):
                    marks.setdefault("first_say", time.perf_counter())
                    answered = True
                print(stamp(), json.dumps(msg)[:240])
                if kind == "tool_call":
                    await ws.send(json.dumps({"type": "tool_result", "id": msg["id"], "ok": True,
                                              "result": CANNED.get(msg["name"], {"error": "unknown tool"})}))
                if kind == "confirm":  # answer "add … to your reminders?" the way a person would
                    await ws.send(json.dumps({"type": "text", "text": "yes"}))
                finished = kind == "turn_done" and (msg.get("background") or not args.wait_report)
                if finished or (kind == "error" and "not set" in msg.get("message", "")):
                    done.set()

        async def writer() -> None:
            if args.photo:
                image = "data:image/png;base64," + base64.b64encode(two_color_png()).decode()
                await ws.send(json.dumps({"type": "image", "data": image, "text": args.text}))
                return
            if args.text:
                await ws.send(json.dumps({"type": "text", "text": args.text}))
                return
            await asyncio.sleep(2.0)  # let the server open its ASR session
            step = 640  # 20 ms
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i:i + step])
                await asyncio.sleep(0.02)
            marks["speech_end"] = time.perf_counter()
            print(stamp(), f"(finished speaking {len(pcm) / 32000:.1f} s of audio)")
            while not done.is_set():  # an open mic keeps sending silence
                await ws.send(bytes(step))
                await asyncio.sleep(0.02)

        tasks = [asyncio.create_task(reader()), asyncio.create_task(writer())]
        try:
            await asyncio.wait_for(done.wait(), args.timeout)
        except TimeoutError:
            print(stamp(), "timed out")
        for t in tasks:
            t.cancel()

    if "speech_end" in marks and "transcript" in marks:
        print(f"speech end → final transcript: {marks['transcript'] - marks['speech_end']:.2f} s")
    if "speech_end" in marks and "first_say" in marks:
        print(f"speech end → first spoken word: {marks['first_say'] - marks['speech_end']:.2f} s")
    return 0 if answered else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
