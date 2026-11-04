"""The agent loop: stream a model reply, speak it a sentence at a time, run tools, repeat.

Muse Spark goes through the Responses API (built-in web_search, `previous_response_id` chaining);
open-weight hosts such as Fireworks go through Chat Completions. Either way the caller sees the same
events: `say` (a sentence to speak) and `action` (a receipt), and gets the turn's stats back.
If the brain fails before saying anything, the turn is retried once on the `fallback` model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import openai
from openai import AsyncOpenAI

from .actions import ActionLog
from .budget import Budget, token_cost
from .config import api_key, model_cfg
from .tools import Tool

log = logging.getLogger(__name__)

Emit = Callable[[str, dict], Awaitable[None]]
DeviceCall = Callable[[str, dict], Awaitable[dict]]
Confirm = Callable[[str], Awaitable[bool]]

SYSTEM = """You are a voice assistant people hear through Bluetooth glasses or earbuds while their \
phone stays in a pocket. Everything you write is spoken aloud.
- Reply in one to three short sentences. No lists, markdown, emoji or URLs. Write numbers and times \
the way people say them.
- Never say your own name; the microphone would take it as a new request.
- Use tools instead of guessing: get_location for where the user is, get_weather for weather, \
create_reminder for reminders.{search}
- The app already says "On it" when a tool starts, so never announce that you're about to use one; \
after it runs, say what happened in a few words, like "Done, coffee filters at six."
- When the user shares a photo, answer about what's in it, and read any text in it that helps.
- If a request is ambiguous, ask one short question.
It is {now} for the user ({tz}). They are using the {client} app."""
SEARCH = " Use web search for news, opening hours, scores and anything else that changes."
NO_SEARCH = " You can't browse the web right now; say so briefly if a question needs live information."

THINK = re.compile(r"\b(think (about|hard|carefully|it through)|take your time)\b", re.I)
ACK = {"function_call": "On it.", "web_search_call": "Let me check."}
FRESH_AFTER_S = 15 * 60  # start a new conversation after this much idle time (keeps chained context cheap)
SKIP_AFTER_404_S = 600  # a model the API says doesn't exist for this key is skipped this long
FALLBACK_ERRORS = (openai.NotFoundError, openai.InternalServerError, openai.APIConnectionError,
                   openai.APITimeoutError, openai.RateLimitError, openai.PermissionDeniedError)
UNAVAILABLE: dict[str, float] = {}  # model key -> monotonic time until which we go straight to the fallback
MAX_HISTORY = 40  # resent items per conversation; older whole turns are dropped


def user_now(tz: str | None) -> datetime:
    try:
        return datetime.now(ZoneInfo(tz or "UTC"))
    except (KeyError, ValueError):
        return datetime.now(ZoneInfo("UTC"))


def say_when(due: str | None, tz: str | None) -> str:
    """'2026-11-03T18:00' → ' at 6 PM' (today), ' tomorrow at 9 AM', or ' on Friday at 9 AM'."""
    if not due:
        return ""
    try:
        when = datetime.fromisoformat(due)
    except ValueError:
        return f" for {due}"
    clock = when.strftime("%-I:%M %p").replace(":00 ", " ")
    today = user_now(tz).date()
    if when.date() == today:
        return f" at {clock}"
    if when.date() == today + timedelta(days=1):
        return f" tomorrow at {clock}"
    return f" on {when.strftime('%A')} at {clock}"


ACTION_VERBS = {"buy", "call", "pick", "take", "send", "email", "pay", "book", "get", "bring", "check", "water",
                "feed", "renew", "cancel", "schedule", "return", "clean", "finish", "submit", "order", "text", "meet",
                "go", "make", "read", "write", "watch", "start", "stop", "charge", "drop", "grab", "refill", "print",
                "sign", "file", "visit", "walk", "leave", "ask", "tell", "update", "move", "put", "wash", "cook",
                "pack", "plan", "reply", "follow"}


def confirm_question(name: str, args: dict, tz: str | None) -> str:
    """The yes/no question spoken before a side-effect tool runs."""
    if name == "create_reminder":
        title = str(args.get("title") or "that").strip().rstrip(".")
        first = title.split()[0].lower() if title.split() else ""
        about = f"to {title}" if first in ACTION_VERBS else f"about {title}"
        return f"Should I remind you {about}{say_when(args.get('due'), tz)}?"
    return f"Should I {name.replace('_', ' ')}?"


def strip_images(items: list[dict]) -> None:
    """Earlier photos become a short note, so each request resends only the newest one."""
    for item in items:
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = [
                p if p.get("type") not in ("input_image", "image_url")
                else {"type": "input_text" if p["type"] == "input_image" else "text", "text": "[an earlier photo]"}
                for p in content]


def user_content(text: str, image: str | None, api: str) -> str | list[dict]:
    if not image:
        return text
    if api == "chat":
        return [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": image}}]
    return [{"type": "input_text", "text": text}, {"type": "input_image", "image_url": image}]


def trim_history(items: list[dict], limit: int = MAX_HISTORY) -> list[dict]:
    """Drops the oldest whole turns (cut at a user message, so a tool call never loses its result)."""
    if len(items) <= limit:
        return items
    starts = [i for i, item in enumerate(items) if item.get("role") == "user"]
    keep = next((i for i in starts if len(items) - i <= limit), starts[-1] if starts else len(items) - limit)
    return items[keep:]


class AgentError(RuntimeError):
    pass


async def check_model(cfg: dict, key: str) -> bool | None:
    """One tiny real request. True if the model answers, False if the API refuses it, None if unsure."""
    m = model_cfg(cfg, key)
    client = AsyncOpenAI(api_key=api_key(m["api_key_env"]), base_url=m["base_url"], timeout=30, max_retries=0)
    try:
        if m["api"] == "chat":
            await client.chat.completions.create(model=m["id"], messages=[{"role": "user", "content": "ping"}],
                                                 max_tokens=16)
        else:
            await client.responses.create(model=m["id"], input="ping", max_output_tokens=16,
                                          reasoning={"effort": m.get("reasoning_effort", "minimal")})
        return True
    except (openai.NotFoundError, openai.PermissionDeniedError):
        return False
    except Exception:  # network trouble or a 5xx says nothing about access
        return None


class Sentences:
    """Cuts streamed text into speakable sentences so the device can start talking early."""

    END = re.compile(r"[.!?…](?=\s)|\n")

    def __init__(self, min_chars: int = 12):
        self.buf = ""
        self.min_chars = min_chars

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out = []
        while (m := self.END.search(self.buf, self.min_chars - 1 if len(self.buf) >= self.min_chars else len(self.buf))):
            sentence, self.buf = self.buf[: m.end()].strip(), self.buf[m.end():].lstrip()
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> str:
        rest, self.buf = self.buf.strip(), ""
        return rest


@dataclass
class TurnStats:
    model: str
    first_say_s: float | None = None
    total_s: float = 0.0
    cost_usd: float = 0.0
    rounds: int = 0
    says: int = 0
    searches: int = 0
    fallback_from: str | None = None
    tools: list[str] = field(default_factory=list)


class Agent:
    def __init__(self, cfg: dict, budget: Budget, tools: dict[str, Tool], actions: ActionLog,
                 context: dict, clients: dict | None = None):
        self.cfg = cfg
        self.budget = budget
        self.tools = tools
        self.actions = actions
        self.context = context  # tz, client
        self.clients = clients if clients is not None else {}  # provider -> AsyncOpenAI (tests inject fakes)
        self.prev_id: str | None = None  # Responses chaining
        self.items: list[dict] = []  # stateless Responses input items
        self.messages: list[dict] = []  # Chat Completions history
        self.last_turn_at = 0.0
        self.confirm: Confirm | None = None  # the session asks the user before side-effect tools run

    def remember_report(self, text: str) -> None:
        """A finished background report joins the conversation, so follow-up questions about it work."""
        self.items.append({"role": "assistant", "content": text})
        self.messages.append({"role": "assistant", "content": text})

    def client(self, m: dict) -> AsyncOpenAI:
        if m["provider"] not in self.clients:
            self.clients[m["provider"]] = AsyncOpenAI(api_key=api_key(m["api_key_env"]), base_url=m["base_url"],
                                                      timeout=60, max_retries=1)
        return self.clients[m["provider"]]

    def instructions(self, m: dict) -> str:
        tz = self.context.get("tz") or "UTC"
        try:
            now = datetime.now(ZoneInfo(tz))
        except (KeyError, ValueError):
            tz, now = "UTC", datetime.now(ZoneInfo("UTC"))
        return SYSTEM.format(now=now.strftime("%A %B %-d, %Y, %-I:%M %p"), tz=tz,
                             search=SEARCH if m.get("web_search") else NO_SEARCH,
                             client={"ios": "iPhone", "web": "web"}.get(self.context.get("client"), "web"))

    def model_for(self, text: str) -> dict:
        m = model_cfg(self.cfg, self.cfg["think"] if THINK.search(text) else self.cfg["brain"])
        fb = self.cfg.get("fallback")
        if fb and UNAVAILABLE.get(m["key"], 0) > time.monotonic():
            return model_cfg(self.cfg, fb)
        return m

    async def respond(self, text: str, emit: Emit, device_call: DeviceCall, image: str | None = None) -> TurnStats:
        if time.monotonic() - self.last_turn_at > FRESH_AFTER_S:
            self.prev_id, self.items, self.messages = None, [], []
        self.last_turn_at = time.monotonic()
        m = self.model_for(text)
        stats = TurnStats(model=m["id"])
        t0 = time.perf_counter()

        async def say(sentence: str) -> None:
            if stats.first_say_s is None:
                stats.first_say_s = time.perf_counter() - t0
            stats.says += 1
            await emit("say", {"text": sentence})

        if m["key"] == self.cfg.get("think"):
            await say("Let me think about that.")
        said_before = stats.says
        saved = list(self.items), list(self.messages)
        try:
            await self._turn(m, text, image, say, emit, device_call, stats)
        except FALLBACK_ERRORS as e:
            fb = self.cfg.get("fallback")
            if not fb or fb == m["key"] or stats.says > said_before:
                raise  # nowhere to go, or the user already heard part of this answer
            if isinstance(e, openai.NotFoundError | openai.PermissionDeniedError):
                UNAVAILABLE[m["key"]] = time.monotonic() + SKIP_AFTER_404_S
            log.warning("%s failed (%s: %s); retrying the turn on %s", m["id"], type(e).__name__, e, fb)
            self.items, self.messages = saved  # drop the failed attempt's half-turn from the history
            stats.fallback_from, m = m["id"], model_cfg(self.cfg, fb)
            stats.model = m["id"]
            await self._turn(m, text, image, say, emit, device_call, stats)
        stats.total_s = time.perf_counter() - t0
        return stats

    async def _turn(self, m, text, image, say, emit, device_call, stats) -> None:
        turn = self._chat_turn if m["api"] == "chat" else self._responses_turn
        await turn(m, text, image, say, emit, device_call, stats)

    async def _responses_turn(self, m, text, image, say, emit, device_call, stats) -> None:
        tools = [t.spec("responses") for t in self.tools.values()]
        if m.get("web_search"):
            tools.append({"type": "web_search"})
        chained = m.get("chain") == "previous_response_id"
        user = {"role": "user", "content": user_content(text, image, "responses")}
        if chained:
            pending: list[dict] = [user]
        else:
            if image:
                strip_images(self.items)
            self.items.append(user)
            pending = self.items
        spoke = stats.says > 0
        for _ in range(self.cfg["turns"]["max_tool_rounds"]):
            stats.rounds += 1
            if not chained:
                self.items = pending = trim_history(self.items)
            kwargs = dict(model=m["id"], instructions=self.instructions(m), input=list(pending), tools=tools,
                          reasoning={"effort": m["reasoning_effort"]}, max_output_tokens=m["max_output_tokens"],
                          stream=True)
            if chained and self.prev_id:
                kwargs["previous_response_id"] = self.prev_id
            splitter, final, called = Sentences(), None, False
            stream = await self.client(m).responses.create(**kwargs)
            async for ev in stream:
                if ev.type == "response.output_text.delta":
                    if called:
                        continue  # narration beside a tool call; the next round has the real answer
                    for sentence in splitter.feed(ev.delta):
                        spoke = True
                        await say(sentence)
                elif ev.type == "response.output_item.added":
                    called = called or ev.item.type == "function_call"
                    if ev.item.type in ACK and not spoke:
                        spoke = True
                        await say(ACK[ev.item.type])
                elif ev.type in ("response.completed", "response.incomplete"):
                    final = ev.response
                elif ev.type == "response.failed":
                    raise AgentError(f"model failed: {getattr(ev.response, 'error', None)}")
                elif ev.type == "error":
                    raise AgentError(f"model error: {getattr(ev, 'message', ev)}")
            if (tail := splitter.flush()) and not called:
                spoke = True
                await say(tail)
            if final is None:
                raise AgentError("stream ended without a final response")
            self._account(m, final, stats)
            calls = [o for o in final.output if o.type == "function_call"]
            if chained:
                self.prev_id = final.id
            else:
                self.items += [o.model_dump(exclude_none=True, by_alias=True) for o in final.output
                               if o.type != "reasoning"]
            if not calls:
                return
            outputs = await asyncio.gather(*(self._run_tool(c.name, c.arguments, emit, device_call, stats)
                                             for c in calls))
            results = [{"type": "function_call_output", "call_id": c.call_id, "output": out}
                       for c, out in zip(calls, outputs)]
            if chained:
                pending = results
            else:
                self.items += results
                pending = self.items
        await say("Sorry, that took too many steps.")

    async def _chat_turn(self, m, text, image, say, emit, device_call, stats) -> None:
        tools = [t.spec("chat") for t in self.tools.values()]
        if image:
            strip_images(self.messages)
        self.messages.append({"role": "user", "content": user_content(text, image, "chat")})
        spoke = stats.says > 0
        for _ in range(self.cfg["turns"]["max_tool_rounds"]):
            stats.rounds += 1
            self.messages = trim_history(self.messages)
            extra = {"reasoning_effort": m["reasoning_effort"]} if m.get("reasoning_effort") else {}
            stream = await self.client(m).chat.completions.create(
                model=m["id"], messages=[{"role": "system", "content": self.instructions(m)}, *self.messages],
                tools=tools, max_tokens=m["max_output_tokens"], stream=True,
                stream_options={"include_usage": True}, **extra)
            splitter, content, calls, usage = Sentences(), "", {}, None
            async for ch in stream:
                usage = ch.usage or usage
                if not ch.choices:
                    continue
                d = ch.choices[0].delta
                if d.content:
                    content += d.content
                    for sentence in [] if calls else splitter.feed(d.content):  # silent once a tool call starts
                        spoke = True
                        await say(sentence)
                for tc in d.tool_calls or []:
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    slot["id"] = tc.id or slot["id"]
                    if tc.function:
                        slot["name"] += tc.function.name or ""
                        slot["arguments"] += tc.function.arguments or ""
                    if not spoke:
                        spoke = True
                        await say(ACK["function_call"])
            if (tail := splitter.flush()) and not calls:
                spoke = True
                await say(tail)
            if usage:
                cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                cost = token_cost(m, usage.prompt_tokens, cached, usage.completion_tokens)
                stats.cost_usd += cost
                self.budget.add(cost)
            msg: dict = {"role": "assistant", "content": content or None}
            if calls:
                msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": c["arguments"]}}
                                     for c in calls.values()]
            self.messages.append(msg)
            if not calls:
                return
            for c in calls.values():
                out = await self._run_tool(c["name"], c["arguments"], emit, device_call, stats)
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
        await say("Sorry, that took too many steps.")

    def _account(self, m: dict, final, stats: TurnStats) -> None:
        u = final.usage
        searches = sum(1 for o in final.output if o.type == "web_search_call")
        cost = searches * self.cfg["search_price_per_1k"] / 1000
        if u:
            cached = getattr(u.input_tokens_details, "cached_tokens", 0) or 0
            cost += token_cost(m, u.input_tokens, cached, u.output_tokens)
        stats.searches += searches
        stats.cost_usd += cost
        self.budget.add(cost)

    async def _run_tool(self, name: str, arguments: str, emit: Emit, device_call: DeviceCall,
                        stats: TurnStats) -> str:
        """Runs one tool call with a receipt; returns the JSON string the model sees."""
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        tool = self.tools.get(name)
        entry = self.actions.start(name, args, tool.where if tool else "unknown")
        await emit("action", entry)
        try:
            if tool is None:
                result = {"error": f"unknown tool {name!r}"}
            elif (tool.side_effect and self.confirm and self.cfg["turns"].get("confirm_side_effects", True)
                  and not await self.confirm(confirm_question(name, args, self.context.get("tz")))):
                result = {"cancelled": True, "reason": "the user didn't say yes, so nothing was changed"}
            elif tool.where == "server":
                result = await tool.run(**args)
            else:
                result = await device_call(name, args)
        except Exception as e:  # tool failures go back to the model, which can explain or retry
            result = {"error": f"{type(e).__name__}: {e}"[:300]}
        await emit("action", self.actions.finish(entry, "error" not in result, result))
        stats.tools.append(name)
        return json.dumps(result)
