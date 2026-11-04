"""Receipts: every tool call the agent makes, with arguments, result, status and timing (JSONL per device).

The log is tamper-evident: each line carries the hash of the line before it (`prev`) and its own SHA-256 (`hash`),
so editing, deleting or reordering a past receipt breaks the chain, and `verify` says where.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

GENESIS = "0" * 64


def digest(record: dict) -> str:
    """SHA-256 of a receipt without its own hash, over a canonical encoding."""
    body = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tail_hash(path: Path) -> str:
    """The hash of the last receipt, read from the end of the file, so a long log costs nothing to continue."""
    if not path.exists() or path.stat().st_size == 0:
        return GENESIS
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 8192))
        last = f.read().splitlines()[-1]
    try:
        return json.loads(last).get("hash") or GENESIS
    except json.JSONDecodeError:
        return GENESIS


def verify(path: Path) -> tuple[bool, int]:
    """(True, receipts) if the chain is intact, else (False, the 1-based line where it first breaks)."""
    prev, n = GENESIS, 0
    for n, line in enumerate(path.read_text().splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return False, n
        if record.get("prev", GENESIS) != prev or record.get("hash") != digest(record):
            return False, n
        prev = record["hash"]
    return True, n


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
            record = {**done, "date": time.strftime("%Y-%m-%d"), "prev": tail_hash(self.path)}
            record["hash"] = digest(record)
            with self.path.open("a") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        return done
