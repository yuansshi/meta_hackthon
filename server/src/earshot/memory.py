"""Per-device notes the agent saves and looks up: "remember I parked on level 3, spot 42"."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .tools import Tool, obj

MAX_NOTES = 200


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 2}


class Memory:
    def __init__(self, path: Path | None):
        self.path = path
        self.notes: list[dict] = json.loads(path.read_text()) if path and path.exists() else []

    async def remember(self, note: str) -> dict:
        note = note.strip()
        if not note:
            return {"error": "nothing to remember"}
        self.notes = [*self.notes, {"note": note[:500], "saved": time.strftime("%Y-%m-%dT%H:%M")}][-MAX_NOTES:]
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.notes))
        return {"ok": True, "saved": note, "notes": len(self.notes)}

    async def recall(self, query: str | None = None) -> dict:
        """Notes matching any keyword, newest first; with no match, the latest notes so the model can judge."""
        latest = list(reversed(self.notes))
        wanted = words(query or "")
        hits = [n for n in latest if wanted & words(n["note"])] if wanted else latest
        return {"notes": (hits or latest)[:20], "matched": bool(hits) if wanted else None}


def memory_tools(memory: Memory) -> dict[str, Tool]:
    return {
        "remember": Tool(
            "remember",
            "Save a note when the user asks you to remember something: where they parked, a name, an item for "
            "a list. Keep their words and the details that matter.",
            obj({"note": {"type": "string"}}, ("note",)), "server", run=memory.remember),
        "recall": Tool(
            "recall",
            "Look up notes the user asked you to remember earlier. Pass a few keywords, or nothing for the "
            "latest notes.",
            obj({"query": {"type": "string"}}), "server", run=memory.recall),
    }
