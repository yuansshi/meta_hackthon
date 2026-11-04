"""Long "look into it and get back to me" jobs on Muse Wonder, spoken back when they finish.

The brain calls `start_background_task` and tells the user it will get back to them. The job then
outlives that turn, and even the connection: Muse Wonder works as a background response (polled),
and the finished report is spoken into whatever session the device has open, or held until it next
connects. If Wonder is unavailable, Muse Spark Open does the job (without web search).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from openai import AsyncOpenAI

from .actions import ActionLog
from .agent import FALLBACK_ERRORS, AgentError
from .budget import Budget, token_cost
from .config import api_key, model_cfg
from .tools import Tool, obj

log = logging.getLogger(__name__)

PROMPT = """You are a research assistant working in the background for someone who will hear your answer \
through earbuds while out and about. It is {now} for them ({tz}).
Task: {goal}
Context from the conversation: {context}
Research as needed{search}, then reply in exactly this shape:
SUMMARY: two or three short sentences meant to be spoken aloud, answer first, no lists or URLs.
DETAILS: concise notes with the key facts, options, prices or times{sources}."""

TOOL_DESCRIPTION = (
    "Hand a job that needs real research or planning (comparing options, planning a trip or a day, digging "
    "through several sources) to Muse Wonder, a deeper model that works in the background for a minute or "
    "two and then reports back on its own. Use it when the user says 'look into', 'research', 'get back to "
    "me' or 'figure out', or when a good answer clearly needs more than one quick search. After calling it, "
    "tell the user in a few words that you'll get back to them; don't answer the question yourself.")
TOOL_PARAMETERS = obj({
    "goal": {"type": "string", "description": "What to find out, as a short phrase that reads naturally after "
                                              "'I looked into', e.g. 'the best e-bikes under $1,500 for a hilly "
                                              "commute' or 'whether Pike Place Market is open late tonight'."},
    "context": {"type": "string", "description": "Anything useful from the conversation: where the user "
                                                 "is, preferences, constraints, dates."},
}, ("goal",))


def research_tool(run) -> Tool:
    return Tool("start_background_task", TOOL_DESCRIPTION, TOOL_PARAMETERS, "server", run=run)


def split_report(text: str) -> tuple[str, str]:
    """SUMMARY (spoken) and DETAILS (shown); a reply that ignored the format is spoken as is."""
    m = re.search(r"SUMMARY:\s*(.*?)\s*(?:DETAILS:\s*(.*))?$", text.strip(), re.S | re.I)
    if not m:
        return text.strip()[:500], ""
    return m.group(1).strip(), (m.group(2) or "").strip()


@dataclass
class Job:
    id: str
    device_id: str
    goal: str
    context: str
    tz: str
    actions: ActionLog
    started: float = field(default_factory=time.monotonic)


class Researcher:
    def __init__(self, cfg: dict, budget: Budget, clients: dict | None = None, poll_s: float | None = None):
        self.cfg = cfg
        self.budget = budget
        self.clients = clients if clients is not None else {}  # provider -> AsyncOpenAI (tests inject fakes)
        rc = cfg.get("research", {})
        self.poll_s = poll_s if poll_s is not None else rc.get("poll_s", 3)
        self.timeout_s = rc.get("timeout_s", 600)
        self.per_day = rc.get("per_device_per_day", 5)
        self.live: dict[str, object] = {}  # device id -> its open session (the newest wins)
        self.inbox: dict[str, list[dict]] = {}  # reports waiting for a device to reconnect
        self.started: Counter[str] = Counter()
        self.day = time.strftime("%Y-%m-%d")
        self.tasks: set[asyncio.Task] = set()

    def attach(self, device_id: str, session) -> list[dict]:
        """Registers a session and returns the reports that finished while the device was away."""
        self.live[device_id] = session
        return self.inbox.pop(device_id, [])

    def detach(self, device_id: str, session) -> None:
        if self.live.get(device_id) is session:
            del self.live[device_id]

    def start(self, device_id: str, goal: str, context: str, tz: str, actions: ActionLog) -> dict:
        if time.strftime("%Y-%m-%d") != self.day:
            self.day, self.started = time.strftime("%Y-%m-%d"), Counter()
        if not goal.strip():
            return {"error": "say what to look into"}
        if self.started[device_id] >= self.per_day:
            return {"error": f"this device already started {self.per_day} research jobs today"}
        self.started[device_id] += 1
        job = Job(uuid.uuid4().hex[:8], device_id, goal.strip(), context.strip(), tz or "UTC", actions)
        task = asyncio.create_task(self._run(job))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return {"started": True, "job": job.id, "model": model_cfg(self.cfg, self.cfg["background"])["id"],
                "expect": "a minute or two",
                "tell_user": "only that you'll get back to them, in under ten words. The app already said 'On it'."}

    async def _run(self, job: Job) -> None:
        entry = job.actions.start("research", {"goal": job.goal}, "server")
        report: dict = {"job": job.id, "goal": job.goal, "searches": 0, "cost_usd": 0.0, "model": ""}
        try:
            text, report["model"], report["searches"], report["cost_usd"] = await asyncio.wait_for(
                self._ask(job), self.timeout_s)
            report["summary"], report["details"] = split_report(text)
            report["ok"] = bool(report["summary"])
            result = {"summary": report["summary"], "model": report["model"], "searches": report["searches"]}
        except Exception as e:  # the user still hears that the job didn't work out
            log.warning("research %s failed: %r", job.id, e)
            report["ok"] = False
            result = {"error": f"{type(e).__name__}: {e}"[:200]}
        report["seconds"] = round(time.monotonic() - job.started)
        report["entry"] = job.actions.finish(entry, report["ok"], result)
        await self.deliver(job.device_id, report)

    async def deliver(self, device_id: str, report: dict) -> None:
        session = self.live.get(device_id)
        if session is not None:
            try:
                await session.deliver_report(report)
                return
            except Exception:
                log.exception("delivering report %s failed; keeping it for later", report["job"])
        self.inbox.setdefault(device_id, []).append(report)

    def client(self, m: dict) -> AsyncOpenAI:
        if m["provider"] not in self.clients:
            self.clients[m["provider"]] = AsyncOpenAI(api_key=api_key(m["api_key_env"]), base_url=m["base_url"],
                                                      timeout=90, max_retries=1)
        return self.clients[m["provider"]]

    def prompt(self, job: Job, m: dict) -> str:
        try:
            now = datetime.now(ZoneInfo(job.tz)).strftime("%A %B %-d, %Y, %-I:%M %p")
        except (KeyError, ValueError):
            now = datetime.now(ZoneInfo("UTC")).strftime("%A %B %-d, %Y, %-I:%M %p")
        searching = m.get("web_search")
        return PROMPT.format(now=now, tz=job.tz, goal=job.goal, context=job.context or "none",
                             search=" with web search" if searching else " (you can't browse the web now; say so "
                             "briefly if live information matters)",
                             sources=", with sources" if searching else "")

    async def _ask(self, job: Job) -> tuple[str, str, int, float]:
        """(text, model id, web searches, $). Wonder runs as a background response; Spark Open is the fallback."""
        m = model_cfg(self.cfg, self.cfg["background"])
        try:
            return await self._background(m, job)
        except FALLBACK_ERRORS as e:
            fb = self.cfg.get("background_fallback")
            if not fb:
                raise
            log.warning("research on %s failed (%s); using %s", m["id"], type(e).__name__, fb)
            return await self._chat(model_cfg(self.cfg, fb), job)

    async def _background(self, m: dict, job: Job) -> tuple[str, str, int, float]:
        client = self.client(m)
        r = await client.responses.create(
            model=m["id"], input=self.prompt(job, m), background=True,
            tools=[{"type": "web_search"}] if m.get("web_search") else [],
            reasoning={"effort": m["reasoning_effort"]}, max_output_tokens=m["max_output_tokens"])
        try:
            while r.status in ("queued", "in_progress"):
                await asyncio.sleep(self.poll_s)
                r = await client.responses.retrieve(r.id)
        except asyncio.CancelledError:  # timed out: don't leave the job burning credits
            try:
                await client.responses.cancel(r.id)
            except Exception:
                pass
            raise
        if r.status != "completed":
            raise AgentError(f"research ended {r.status}: {getattr(r, 'error', None)}")
        searches = sum(1 for o in r.output if o.type == "web_search_call")
        cost = searches * self.cfg["search_price_per_1k"] / 1000
        if r.usage:
            cached = getattr(r.usage.input_tokens_details, "cached_tokens", 0) or 0
            cost += token_cost(m, r.usage.input_tokens, cached, r.usage.output_tokens)
        self.budget.add(cost)
        return r.output_text, m["id"], searches, cost

    async def _chat(self, m: dict, job: Job) -> tuple[str, str, int, float]:
        extra = {"reasoning_effort": m["reasoning_effort"]} if m.get("reasoning_effort") else {}
        r = await self.client(m).chat.completions.create(
            model=m["id"], messages=[{"role": "user", "content": self.prompt(job, m)}],
            max_tokens=max(m["max_output_tokens"], 3000), **extra)
        cost = 0.0
        if r.usage:
            cost = token_cost(m, r.usage.prompt_tokens, 0, r.usage.completion_tokens)
        self.budget.add(cost)
        return r.choices[0].message.content or "", m["id"], 0, cost
