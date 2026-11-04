"""Heads-up: EveryMuse notices something you'll want in speech that wasn't said to it, and offers to act.

A cascade over Meta's models, cheapest and fastest first:
1. Muse Voice Transcribe hears everything (the turn gate sets aside what wasn't said to the agent).
2. Muse Spark, at minimal effort, screens that overheard speech, with the user's own notes for context, and
   proposes at most one short offer: "Want me to remind you about dinner Friday at seven?"
3. Only a spoken or tapped yes lets the agent act. It then acts without asking twice, and a research offer goes
   on to Muse Wonder like any other "look into it" request.

Off unless the client turns it on, hands-free mode only, and nothing it screens is stored: the screener sees the
Rewind buffer (memory only), and receipts record the offer's kind and outcome, never the words heard.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .agent import user_now
from .budget import token_cost

KINDS = ("reminder", "note", "lookup", "research", "alert")

PROMPT = """You quietly watch what a person's microphone overhears around them. None of it was said to you, and \
they can't see you. Decide whether there is one thing they would clearly want help with right now:
- a plan they are part of, with a time, a date or a place they will need, like dinner on Friday at seven at an address
- something someone asked them to do, bring or remember
- a detail someone gave them that's worth keeping, like a name, a phone number, a gate, a platform or where something is
- an announcement that changes a plan in their notes
Most of the time there is nothing: small talk, background chatter, a TV or a radio, plans with no time or place \
yet, other people's plans, trains or appointments, and announcements for staff or for other travelers. \
Offer nothing then. Never offer the same thing twice.
You can't tell who is speaking: it may be them or someone near them. So use concrete details, never pronouns: \
"dinner at 1420 Pine Street, unit 3", not "dinner at my place".

They speak {language}: write "offer" and "request" in {language}.
Their notes: {notes}
Offers already made: {made}
It is {now} for them ({tz}).

Reply with JSON only:
{{"kind": "none"}}
or
{{"kind": "reminder" | "note" | "lookup" | "research" | "alert",
  "offer": "one short yes-or-no question to say to them, under 15 words, in their language",
  "request": "what they would say to an assistant if they said yes, with every detail that was heard"}}
An "alert" is for an announcement that clearly affects the plans in their notes; its "offer" is a short \
statement to tell them, not a question, and it needs no "request"."""


@dataclass
class Offer:
    kind: str
    offer: str
    request: str


def parse_offer(text: str) -> Offer | None:
    """The screener's JSON, or None for no offer or anything malformed: when in doubt, stay quiet."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    kind = str(data.get("kind") or "none").lower()
    offer = " ".join(str(data.get("offer") or "").split())
    request = " ".join(str(data.get("request") or "").split())
    if kind not in KINDS or not offer or len(offer) > 200 or (kind != "alert" and not request):
        return None
    return Offer(kind, offer, request[:600])


async def screen(client, m: dict, heard: list[str], notes: list[str], made: list[str], tz: str,
                 max_output_tokens: int, effort: str = "minimal",
                 language: str = "English") -> tuple[Offer | None, float, str]:
    """One screening call: (offer or None, cost in $, raw reply). Streams like every turn, so tests share fakes."""
    prompt = PROMPT.format(notes="; ".join(notes) or "none", made="; ".join(made) or "none", language=language,
                           now=user_now(tz).strftime("%A %B %-d, %-I:%M %p"), tz=tz)
    overheard = "Overheard, oldest first:\n" + "\n".join(heard)
    if m["api"] == "responses":
        stream = await client.responses.create(
            model=m["id"], instructions=prompt, input=[{"role": "user", "content": overheard}],
            reasoning={"effort": effort}, max_output_tokens=max_output_tokens, stream=True)
        text, final = "", None
        async for ev in stream:
            if ev.type == "response.output_text.delta":
                text += ev.delta
            elif ev.type in ("response.completed", "response.incomplete"):
                final = ev.response
        u = getattr(final, "usage", None)
        cost = token_cost(m, u.input_tokens, getattr(u.input_tokens_details, "cached_tokens", 0) or 0,
                          u.output_tokens) if u else 0.0
    else:  # an open-weights host on Chat Completions
        extra = {"reasoning_effort": m["reasoning_effort"]} if m.get("reasoning_effort") else {}
        stream = await client.chat.completions.create(
            model=m["id"], messages=[{"role": "system", "content": prompt}, {"role": "user", "content": overheard}],
            max_tokens=max_output_tokens, stream=True, stream_options={"include_usage": True}, **extra)
        text, usage = "", None
        async for ch in stream:
            usage = ch.usage or usage
            if ch.choices and ch.choices[0].delta.content:
                text += ch.choices[0].delta.content
        cost = token_cost(m, usage.prompt_tokens, 0, usage.completion_tokens) if usage else 0.0
    return parse_offer(text), cost, text


def is_answer(said: str, lang) -> bool:
    """While an offer waits, only a short yes or no answers it; the conversation around the user carries on.
    `lang` is the user's Language (see languages.py); `said` is normalized text."""
    return lang.short(said) and (lang.is_yes(said) or lang.is_no(said))
