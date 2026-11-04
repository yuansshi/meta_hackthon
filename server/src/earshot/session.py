"""One device connection: audio → ASR relay → turn gate → agent → `say` events back to the device."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from difflib import SequenceMatcher

from .actions import ActionLog
from .agent import Agent, Sentences
from .budget import Budget
from .config import DATA_DIR, MissingKey, api_key
from .memory import Memory, memory_tools
from .relay import AsrRelay
from .research import Researcher, research_tool
from .rewind import Heard, rewind_tool
from .tools import available

log = logging.getLogger(__name__)

Send = Callable[[dict], Awaitable[None]]
GREETINGS = {"hey", "hi", "hello", "ok", "okay", "yo", "oh"}
EDGE = ".,!?;:\"'()[]…-–—"
IMAGE_RE = re.compile(r"data:image/(?:jpeg|png|webp);base64,[A-Za-z0-9+/]+=*")
MAX_IMAGE_CHARS = 2_000_000  # ~1.5 MB of image as base64; clients send at most 1024 px JPEGs
IMAGE_TTL_S = 120  # a staged photo goes with the next question asked within this window
PHOTO_QUESTION = "What's in this photo?"
YES = re.compile(r"^(yes|yeah|yep|yup|sure|ok|okay|do it|go ahead|please|correct|right|confirm|sounds good|absolutely)\b")
NO = re.compile(r"\b(no|nope|nah|don't|do not|cancel|stop|never ?mind|wait)\b")


def speakable(text: str) -> list[str]:
    s = Sentences()
    return [*s.feed(text + " "), *([rest] if (rest := s.flush()) else [])]


def clean(word: str) -> str:
    return word.strip(EDGE).lower()


def normalize(text: str) -> str:
    return " ".join(c for c in map(clean, text.split()) if c)


@dataclass
class Decision:
    accept: bool
    text: str
    reason: str


class Gate:
    """Decides which finished utterances are meant for the agent. Pure logic; callers pass the clock."""

    def __init__(self, cfg: dict, mode: str = "wake"):
        self.cfg = cfg
        self.mode = mode  # "wake" (hands-free, needs the wake word) or "ptt" (push-to-talk)
        self.wake = cfg["wake_word"].lower()
        self.variants = [v.lower().split() for v in [cfg["wake_word"], *cfg["wake_variants"]]]
        self.speaking_since: float | None = None
        self.quiet_after = 0.0
        self.follow_up_until = 0.0
        self.recent_said: deque[str] = deque(maxlen=8)

    def device_speaking(self, started: bool, text: str | None, now: float) -> None:
        if started:
            self.speaking_since = now
            if text:
                self.recent_said.append(normalize(text))
        else:
            self.speaking_since = None
            self.quiet_after = now + self.cfg["echo_tail_s"]
            self.follow_up_until = now + self.cfg["follow_up_s"]

    def open_follow_up(self, now: float) -> None:
        self.follow_up_until = now + self.cfg["follow_up_s"]

    def speaking(self, now: float) -> bool:
        if self.speaking_since is not None and now - self.speaking_since > self.cfg["speaking_timeout_s"]:
            self.speaking_since = None  # the device never said `tts end`; don't go deaf forever
        return self.speaking_since is not None

    def find_wake(self, text: str) -> str | None:
        """Whatever follows the wake word if it appears within the first few words, else None."""
        words = [(w, c) for w in text.split() if (c := clean(w))]
        i = 0
        while i < min(2, len(words)) and words[i][1] in GREETINGS:
            i += 1
        for start in range(i, min(i + 3, len(words))):
            for v in self.variants:
                if [c for _, c in words[start:start + len(v)]] == v:
                    return " ".join(w for w, _ in words[start + len(v):])
            for span in (1, 2):
                if start + span <= len(words):
                    joined = "".join(c for _, c in words[start:start + span])
                    if SequenceMatcher(None, joined, self.wake).ratio() >= self.cfg["wake_fuzzy"]:
                        return " ".join(w for w, _ in words[start + span:])
        return None

    def is_echo(self, norm: str, now: float) -> bool:
        if now > self.quiet_after + self.cfg.get("echo_match_window_s", 3.0):
            return False  # long after the device spoke, a similar sentence is the user's own
        for said in self.recent_said:
            if len(norm) >= 8 and norm in said:
                return True
            if SequenceMatcher(None, norm, said).ratio() >= self.cfg["echo_similarity"]:
                return True
        return False

    def decide(self, text: str, now: float) -> Decision:
        norm = normalize(text)
        if not norm:
            return Decision(False, "", "empty")
        if self.speaking(now) or now < self.quiet_after:
            return Decision(False, text, "echo:speaking")
        if self.is_echo(norm, now):
            return Decision(False, text, "echo:match")
        rest = self.find_wake(text)
        if rest is not None:
            return Decision(True, rest, "wake") if normalize(rest) else Decision(True, "", "wake_only")
        if self.mode == "ptt":
            return Decision(True, text, "ptt")
        if now < self.follow_up_until:
            return Decision(True, text, "follow_up")
        return Decision(False, text, "no_wake")


def safe_id(raw: object) -> str:
    s = str(raw or "")
    return s if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", s) else "anon"


class Session:
    def __init__(self, *, cfg: dict, budget: Budget, send: Send, relay_factory=AsrRelay, agent_factory=Agent,
                 researcher: Researcher | None = None, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.budget = budget
        self.send = send
        self.relay_factory = relay_factory
        self.agent_factory = agent_factory
        self.researcher = researcher
        self.clock = clock
        self.client = "web"
        self.device_id = "anon"
        self.tz = "UTC"
        self.actions = ActionLog(None)
        self.awaiting: asyncio.Future | None = None  # a yes/no question waiting for its answer
        self.current_turn = ""
        self.reports = 0
        self.state = ""
        self.muted = False
        self.relay: AsrRelay | None = None
        self.relay_task: asyncio.Task | None = None
        self.turn: asyncio.Task | None = None
        self.turns = 0
        self.pending: dict[str, asyncio.Future] = {}
        self.pending_image: tuple[str, float] | None = None  # (data URL, staged at)
        self.asr_billed_s = 0.0
        self.gate = Gate(cfg["turns"])
        self.heard = Heard(cfg["turns"].get("rewind_s", 0), clock)  # speech not meant for the agent, for Rewind
        self.agent: Agent | None = None

    async def start(self, hello: dict) -> None:
        self.client = hello.get("client") if hello.get("client") in ("ios", "web") else "web"
        self.device_id = device_id = safe_id(hello.get("device_id"))
        self.tz = str(hello.get("tz") or "UTC")
        self.gate = Gate(self.cfg["turns"], "ptt" if hello.get("mode") == "ptt" else "wake")
        self.actions = ActionLog(DATA_DIR / "actions" / f"{device_id}.jsonl")
        tools = available([t for t in hello.get("tools") or [] if isinstance(t, str)])
        tools.update(memory_tools(Memory(DATA_DIR / "memory" / f"{device_id}.json")))
        if self.heard.window_s > 0:
            tools["rewind"] = rewind_tool(self.heard, lambda: self.gate.mode)
        if self.researcher and self.cfg.get("background"):
            tools["start_background_task"] = research_tool(self.start_research)
        self.agent = self.agent_factory(self.cfg, self.budget, tools, self.actions,
                                        {"tz": self.tz, "client": self.client})
        self.agent.confirm = self.confirm
        asr = self.cfg["asr"]
        try:
            self.relay = self.relay_factory(asr, api_key(asr["api_key_env"]), self.on_asr_event)
            self.relay_task = asyncio.create_task(self.relay.run())
        except MissingKey as e:
            await self.send({"type": "error", "message": str(e)})
        await self.send({"type": "ready", "brain": self.agent.model_for("")["id"],
                         "asr": asr["model"] if self.relay else None, "mode": self.gate.mode,
                         "wake_word": self.cfg["turns"]["wake_word"], "tools": sorted(tools)})
        await self.set_state("listening")
        for report in self.researcher.attach(device_id, self) if self.researcher else []:
            await self.deliver_report(report, away=True)

    async def set_state(self, value: str) -> None:
        if value != self.state:
            self.state = value
            await self.send({"type": "state", "value": value})

    def handle_audio(self, pcm: bytes) -> None:
        if self.relay and not self.muted:
            self.relay.feed(pcm)

    async def handle_text(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "tts":
            self.gate.device_speaking(msg.get("state") == "start", msg.get("text"), self.clock())
        elif kind == "tool_result":
            fut = self.pending.get(str(msg.get("id")))
            if fut and not fut.done():
                ok = msg.get("ok", True)
                fut.set_result((msg.get("result") or {}) if ok else {"error": msg.get("error") or "failed on device"})
        elif kind == "text" and str(msg.get("text", "")).strip():
            text = str(msg["text"]).strip()[:2000]
            if self.awaiting and not self.awaiting.done():
                self.awaiting.set_result(text)
            else:
                self.start_turn(Decision(True, text, "typed"))
        elif kind == "listen":  # a headset tap: the next utterance needs no wake word
            self.gate.open_follow_up(self.clock())
        elif kind == "image":
            await self.stage_image(msg)
        elif kind == "mute":
            self.muted = bool(msg.get("on"))
        elif kind == "mode" and msg.get("mode") in ("wake", "ptt"):
            self.gate.mode = msg["mode"]

    async def on_asr_event(self, evt: dict) -> None:
        kind = evt.get("type")
        now = self.clock()
        if kind == "transcript" and not self.gate.speaking(now):
            await self.send({"type": "transcript", "final": False, "text": evt.get("transcript", "")})
        elif kind == "speechComplete":
            text = (evt.get("transcript") or "").strip()
            d = self.gate.decide(text, now)
            if self.awaiting and not self.awaiting.done() and text and not d.reason.startswith("echo"):
                await self.send({"type": "transcript", "final": True, "text": text, "accepted": True,
                                 "reason": "answer"})
                self.awaiting.set_result(text)  # the answer to "add … to your reminders?", no wake word needed
                return
            if d.reason == "no_wake" and self.heard.window_s > 0:
                self.heard.add(text)  # not for the agent, but "what did she just say?" may ask about it
            if d.accept and self.turn and not self.turn.done():
                d = Decision(False, text, "busy")
            await self.send({"type": "transcript", "final": True, "text": text, "accepted": d.accept,
                             "reason": d.reason})
            if d.accept:
                self.start_turn(d)

    async def stage_image(self, msg: dict) -> None:
        """A photo waits for the next question, typed or spoken; with `text` it is asked about right away."""
        data = msg.get("data")
        if not data:
            self.pending_image = None
            return
        if not isinstance(data, str) or len(data) > MAX_IMAGE_CHARS or not IMAGE_RE.fullmatch(data):
            await self.send({"type": "error", "message": "That photo couldn't be used. Send a JPEG, PNG or WebP under 1.5 MB."})
            return
        self.pending_image = (data, self.clock())
        self.gate.open_follow_up(self.clock())  # "what's this?" right after a photo needs no wake word
        if question := str(msg.get("text") or "").strip():
            self.start_turn(Decision(True, question[:2000], "typed"))

    def take_image(self) -> str | None:
        """The staged photo, if it is still fresh. Each photo goes with exactly one turn."""
        image, self.pending_image = self.pending_image, None
        if image and self.clock() - image[1] <= IMAGE_TTL_S:
            return image[0]
        return None

    def start_turn(self, d: Decision) -> None:
        if self.turn and not self.turn.done():
            return
        self.turn = asyncio.create_task(self.run_turn(d))

    async def confirm(self, question: str) -> bool:
        """Asks a yes/no question inside the current turn and waits for the spoken or typed answer."""
        self.awaiting = asyncio.get_running_loop().create_future()
        await self.say(self.current_turn, question)
        await self.send({"type": "confirm", "turn": self.current_turn, "question": question})
        try:
            answer = await asyncio.wait_for(self.awaiting, self.cfg["turns"].get("confirm_timeout_s", 20))
        except TimeoutError:
            answer = ""
        finally:
            self.awaiting = None
        said = normalize(answer)
        return bool(YES.match(said)) and not NO.search(said)

    async def start_research(self, goal: str, context: str | None = None) -> dict:
        return self.researcher.start(self.device_id, goal, context or "", self.tz, self.actions)

    async def deliver_report(self, report: dict, away: bool = False) -> None:
        """Speaks a finished research job, after any answer that's in progress."""
        while self.turn and not self.turn.done():
            await asyncio.wait({self.turn})
        self.reports += 1
        turn = f"r{self.reports}"
        await self.send({"type": "action", "turn": turn, "entry": report["entry"]})
        topic = report["goal"].rstrip(".?! ")
        if report["ok"]:
            await self.send({"type": "report", "turn": turn, "goal": report["goal"],
                             "details": report.get("details", ""), "model": report["model"]})
            lead = "While you were away, I finished looking into" if away else "I looked into"
            text = f"{lead} {topic}. {report['summary']}"
            self.agent.remember_report(f"Background research on {report['goal']}: {report['summary']}\n"
                                       f"{report.get('details', '')}")
        else:
            text = f"Sorry, I couldn't finish looking into {topic}."
        for sentence in speakable(text):
            await self.say(turn, sentence)
        await self.say(turn, "", final=True)
        await self.send({"type": "turn_done", "turn": turn, "model": report["model"], "first_say_ms": None,
                         "total_ms": report["seconds"] * 1000, "cost_usd": round(report["cost_usd"], 5),
                         "tools": ["research"], "searches": report["searches"], "fallback_from": None,
                         "background": True})

    async def say(self, turn: str, text: str, final: bool = False) -> None:
        await self.send({"type": "say", "turn": turn, "text": text, "final": final})

    async def run_turn(self, d: Decision) -> None:
        self.turns += 1
        turn = self.current_turn = f"t{self.turns}"
        self.bill_asr()
        await self.set_state("thinking")
        try:
            if d.reason == "wake_only":
                await self.say(turn, "Yes?", final=True)
                return
            if self.budget.exceeded:
                await self.say(turn, "I've used up today's demo budget. Please try again tomorrow.", final=True)
                return

            async def emit(kind: str, payload: dict) -> None:
                if kind == "say":
                    await self.say(turn, payload["text"])
                else:
                    await self.send({"type": "action", "turn": turn, "entry": payload})

            image = self.take_image()
            if image:
                await self.send({"type": "photo_used", "turn": turn})
            stats = await self.agent.respond(d.text or PHOTO_QUESTION, emit, self.device_call, image=image)
            await self.say(turn, "", final=True)
            await self.send({"type": "turn_done", "turn": turn, "model": stats.model,
                             "first_say_ms": round(stats.first_say_s * 1000) if stats.first_say_s else None,
                             "total_ms": round(stats.total_s * 1000), "cost_usd": round(stats.cost_usd, 5),
                             "tools": stats.tools, "searches": stats.searches,
                             "fallback_from": stats.fallback_from})
        except Exception as e:  # the device should always hear something back
            log.exception("turn %s failed", turn)
            await self.say(turn, "Sorry, something went wrong.", final=True)
            await self.send({"type": "error", "message": f"{type(e).__name__}: {e}"[:300]})
        finally:
            await self.set_state("listening")

    async def device_call(self, name: str, args: dict) -> dict:
        call_id = uuid.uuid4().hex[:12]
        fut = asyncio.get_running_loop().create_future()
        self.pending[call_id] = fut
        await self.send({"type": "tool_call", "id": call_id, "name": name, "args": args})
        try:
            return await asyncio.wait_for(fut, self.cfg["turns"]["device_tool_timeout_s"])
        except TimeoutError:
            return {"error": "the device didn't answer in time"}
        finally:
            self.pending.pop(call_id, None)

    def bill_asr(self) -> None:
        if self.relay:
            new_s = self.relay.sent_s - self.asr_billed_s
            self.asr_billed_s = self.relay.sent_s
            self.budget.add(new_s / 3600 * self.cfg["asr"]["price_per_hour"])

    async def expire(self) -> None:
        await self.say("limit", "This demo session has reached its time limit. Reload the page to start again.",
                       final=True)
        await self.send({"type": "error", "message": "session time limit reached"})

    async def close(self) -> None:
        self.heard.clear()
        if self.researcher:
            self.researcher.detach(self.device_id, self)  # jobs keep running; reports wait in the inbox
        if self.awaiting and not self.awaiting.done():
            self.awaiting.cancel()
        if self.turn and not self.turn.done():
            self.turn.cancel()
        for fut in self.pending.values():
            if not fut.done():
                fut.cancel()
        if self.relay:
            await self.relay.close()
            self.bill_asr()
        if self.relay_task:
            self.relay_task.cancel()
