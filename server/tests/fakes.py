"""Test doubles: a scripted model client (Responses and Chat Completions) and a fake ASR socket."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS

CLOSE = object()  # put on a FakeWS inbox to make the "server" close the stream normally


def item(kind: str, **fields) -> NS:
    obj = NS(type=kind, **fields)
    obj.model_dump = lambda **_: {"type": kind, **fields}
    return obj


def completed(rid: str, output: list, inp: int = 100, out: int = 20) -> NS:
    usage = NS(input_tokens=inp, output_tokens=out, input_tokens_details=NS(cached_tokens=0),
               output_tokens_details=NS(reasoning_tokens=0))
    return NS(type="response.completed", response=NS(id=rid, output=output, usage=usage, output_text=""))


def text_reply(rid: str, text: str) -> list:
    words = text.split(" ")
    deltas = [w + (" " if i < len(words) - 1 else "") for i, w in enumerate(words)]
    return [NS(type="response.output_item.added", item=item("message")),
            *[NS(type="response.output_text.delta", delta=d) for d in deltas],
            completed(rid, [item("message")])]


def tool_reply(rid: str, name: str, args: dict, call_id: str) -> list:
    fc = item("function_call", name=name, arguments=json.dumps(args), call_id=call_id)
    return [NS(type="response.output_item.added", item=fc), completed(rid, [fc])]


class FakeStream:
    def __init__(self, events: list):
        self.events = events

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for e in self.events:
            yield e


class FakeEndpoint:
    def __init__(self, scripts: list[list]):
        self.scripts = list(scripts)
        self.calls: list[dict] = []
        self.delay = 0.0  # seconds before the reply starts, like a model thinking

    async def create(self, **kw):
        self.calls.append(kw)
        if self.delay:
            await asyncio.sleep(self.delay)
        script = self.scripts.pop(0)
        if isinstance(script, Exception):
            raise script
        return FakeStream(script)


class FakeModel:
    """Stands in for AsyncOpenAI: `.responses.create` and `.chat.completions.create` replay scripts."""

    def __init__(self, scripts: list[list]):
        self.responses = FakeEndpoint(scripts)
        self.chat = NS(completions=self.responses)


class FakeResearchModel:
    """Muse Wonder as background Responses (create → queued, retrieve → done) plus plain Chat Completions."""

    def __init__(self, text: str = "", fail: Exception | None = None, polls: int = 1):
        self.text, self.fail, self.polls = text, fail, polls
        self.created: list[dict] = []
        self.retrieved = 0
        self.responses = self
        self.chat = NS(completions=NS(create=self._chat))

    async def create(self, **kw):
        self.created.append(kw)
        if self.fail:
            raise self.fail
        return NS(id="w1", status="queued")

    async def retrieve(self, rid):
        self.retrieved += 1
        usage = NS(input_tokens=500, output_tokens=1500, input_tokens_details=NS(cached_tokens=0))
        return NS(id=rid, status="completed" if self.retrieved >= self.polls else "in_progress",
                  output_text=self.text, output=[NS(type="web_search_call"), NS(type="message")], usage=usage)

    async def cancel(self, rid):
        pass

    async def _chat(self, **kw):
        self.created.append(kw)
        return NS(choices=[NS(message=NS(content=self.text))], usage=NS(prompt_tokens=300, completion_tokens=400))


def chat_chunk(content: str | None = None, tool_calls: list | None = None, usage=None) -> NS:
    return NS(usage=usage, choices=[NS(delta=NS(content=content, tool_calls=tool_calls))])


def chat_tool(index: int, call_id: str | None, name: str | None, arguments: str | None) -> NS:
    return NS(index=index, id=call_id, function=NS(name=name, arguments=arguments))


class FakeRelay:
    def __init__(self, cfg: dict, key: str, on_event):
        self.cfg = cfg
        self.on_event = on_event
        self.fed = bytearray()
        self.sent_s = 0.0

    async def run(self) -> None:
        await asyncio.Event().wait()

    def feed(self, pcm: bytes) -> None:
        self.fed += pcm

    async def close(self) -> None:
        pass


class FakeWS:
    def __init__(self, server_msgs: list):
        self.sent: list = []
        self.inbox: asyncio.Queue = asyncio.Queue()
        for m in server_msgs:
            self.inbox.put_nowait(m)
        self.close_code: int | None = None
        self.close_reason = ""

    async def send(self, data) -> None:
        self.sent.append(data)

    async def recv(self):
        m = await self.inbox.get()
        return json.dumps(m) if isinstance(m, dict) else m

    def __aiter__(self):
        return self

    async def __anext__(self):
        m = await self.inbox.get()
        if m is CLOSE:
            self.close_code = self.close_code or 1000
            raise StopAsyncIteration
        return json.dumps(m) if isinstance(m, dict) else m

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_code = self.close_code or code
        self.inbox.put_nowait(CLOSE)  # like a real socket, closing ends the read loop

    async def wait_closed(self) -> None:
        pass


class FakeConnect:
    def __init__(self, sockets: list[FakeWS]):
        self.sockets = list(sockets)
        self.urls: list[str] = []

    def __call__(self, url: str, **kw):
        self.urls.append(url)
        ws = self.sockets.pop(0)

        class Ctx:
            async def __aenter__(self):
                return ws

            async def __aexit__(self, *exc):
                return False

        return Ctx()
