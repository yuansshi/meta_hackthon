"""Receipts: every tool call the agent makes, with arguments, result, status and timing (JSONL per device)."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


class ActionLog:
    def __init__(self, path: Path | None):
        self.path = path
        self.entries: list[dict] = []
        self._t0: dict[str, float] = {}

    def start(self, tool: str, args: dict, where: str) -> dict:
        entry = {"id": uuid.uuid4().hex[:8], "at": time.strftime("%H:%M:%S"), "tool": tool, "args": args,
                 "where": where, "status": "running"}
        self._t0[entry["id"]] = time.perf_counter()
        self.entries.append(entry)
        return dict(entry)

    def finish(self, entry: dict, ok: bool, result: dict) -> dict:
        ms = round((time.perf_counter() - self._t0.pop(entry["id"], time.perf_counter())) * 1000)
        status = "cancelled" if result.get("cancelled") else "done" if ok else "failed"
        done = {**entry, "status": status, "result": result, "ms": ms}
        self.entries = [done if e["id"] == entry["id"] else e for e in self.entries]
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({**done, "date": time.strftime("%Y-%m-%d")}) + "\n")
        return done
