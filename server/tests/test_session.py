import asyncio
import json
import time

import httpx
import openai
import pytest
from fakes import FakeModel, FakeRelay, FakeResearchModel, chat_chunk, chat_tool, text_reply, tool_reply

from everymuse import agent as agent_mod
from everymuse import session as session_mod
from everymuse import tools as tools_mod
from everymuse.agent import Agent, Sentences
from everymuse.budget import Budget
from everymuse.config import load_config
from everymuse.research import Researcher
from everymuse.session import Session

FORECAST = {
    "current": {"time": "2026-11-03T14:15", "temperature_2m": 61.2, "apparent_temperature": 59.8,
                "weather_code": 61, "wind_speed_10m": 9.4},
    "hourly": {"time": [f"2026-11-03T{h:02d}:00" for h in range(24)],
               "temperature_2m": [55 + h * 0.3 for h in range(24)],
               "precipitation_probability": [min(100, h * 4) for h in range(24)],
               "weather_code": [61] * 24},
    "daily": {"time": ["2026-11-03", "2026-11-04"], "weather_code": [61, 2],
              "temperature_2m_max": [63.4, 66.0], "temperature_2m_min": [50.1, 49.0],
              "precipitation_probability_max": [80, 10]},
}
HELLO = {"type": "hello", "client": "ios", "device_id": "test", "tz": "America/Los_Angeles",
         "tools": ["get_location", "create_reminder"]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setattr(session_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(agent_mod, "UNAVAILABLE", {})

    def handler(request: httpx.Request) -> httpx.Response:
        if "geocoding" in request.url.host:
            return httpx.Response(200, json={"results": [{"name": "Seattle", "admin1": "Washington",
                                                          "latitude": 47.6, "longitude": -122.3}]})
        return httpx.Response(200, json=FORECAST)

    monkeypatch.setattr(tools_mod, "http_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return tmp_path


async def make_session(meta=(), fireworks=(), hello=None, clock=time.monotonic, researcher=None):
    cfg = load_config()
    sent: list[dict] = []
    models = {"meta": FakeModel(list(meta)), "fireworks": FakeModel(list(fireworks))}

    async def send(msg):
        sent.append(msg)

    def agent_factory(cfg, budget, tools, actions, context):
        return Agent(cfg, budget, tools, actions, context, clients=models)

    s = Session(cfg=cfg, budget=Budget(100), send=send, relay_factory=FakeRelay, agent_factory=agent_factory,
                clock=clock, researcher=researcher)
    await s.start(hello or HELLO)
    return s, sent, models


async def next_msg(sent: list, kind: str, timeout: float = 2.0) -> dict:
    for _ in range(int(timeout / 0.01)):
        for m in sent:
            if m["type"] == kind:
                return m
        await asyncio.sleep(0.01)
    raise AssertionError(f"no {kind!r} message in {sent}")


def says(sent: list) -> list[str]:
    return [m["text"] for m in sent if m["type"] == "say" and m["text"]]


def not_found() -> openai.NotFoundError:
    response = httpx.Response(404, request=httpx.Request("POST", "https://api.meta.ai/v1/responses"))
    return openai.NotFoundError("The requested model was not found.", response=response, body=None)


async def test_wake_word_turn_runs_a_device_tool_and_resends_history(env):
    s, sent, models = await make_session(meta=[
        tool_reply("r1", "create_reminder", {"title": "buy coffee filters", "due": "2026-11-03T18:00"}, "c1"),
        text_reply("r2", "Done, coffee filters at six."),
    ])
    await s.on_asr_event({"type": "speechComplete", "transcript": "Hey EveryMuse, remind me to buy coffee filters at 6"})
    question = await next_msg(sent, "confirm")
    assert question["question"].startswith("Should I remind you to buy coffee filters")
    await s.on_asr_event({"type": "speechComplete", "transcript": "Yes please."})  # no wake word for the answer
    call = await next_msg(sent, "tool_call")
    assert call["name"] == "create_reminder" and call["args"]["title"] == "buy coffee filters"
    await s.handle_text({"type": "tool_result", "id": call["id"], "ok": True, "result": {"ok": True, "id": "R1"}})
    await s.turn
    assert says(sent) == ["Okay.", question["question"], "Done, coffee filters at six."]
    first, second = models["meta"].responses.calls
    assert first["input"] == [{"role": "user", "content": "remind me to buy coffee filters at 6"}]
    assert first["reasoning"] == {"effort": "minimal"} and {"type": "web_search"} in first["tools"]
    assert "previous_response_id" not in second  # stateless: the whole turn is resent
    user, fc, out = second["input"]
    assert user["role"] == "user" and fc["type"] == "function_call" and fc["call_id"] == "c1"
    assert out["type"] == "function_call_output" and json.loads(out["output"]) == {"ok": True, "id": "R1"}
    assert [m["entry"]["status"] for m in sent if m["type"] == "action"] == ["running", "done"]
    done = await next_msg(sent, "turn_done")
    assert done["tools"] == ["create_reminder"] and done["first_say_ms"] is not None
    assert (env / "actions" / "test.jsonl").exists()  # receipts persisted


async def test_chained_mode_sends_only_the_tool_result(env):
    s, sent, models = await make_session(meta=[
        tool_reply("r1", "get_weather", {"place": "Seattle"}, "c1"),
        text_reply("r2", "Rain later."),
    ])
    s.cfg["models"]["muse-spark"]["chain"] = "previous_response_id"
    await s.handle_text({"type": "text", "text": "rain in seattle?"})
    await s.turn
    second = models["meta"].responses.calls[1]
    assert second["previous_response_id"] == "r1"
    assert [i["type"] for i in second["input"]] == ["function_call_output"]


def test_history_is_trimmed_at_turn_boundaries():
    turn = lambda n: [{"role": "user", "content": f"q{n}"}, {"type": "function_call", "call_id": f"c{n}"},
                      {"type": "function_call_output", "call_id": f"c{n}"}, {"role": "assistant", "content": "a"}]
    items = [item for n in range(5) for item in turn(n)]  # 20 items
    kept = agent_mod.trim_history(items, limit=9)
    assert kept[0] == {"role": "user", "content": "q3"} and len(kept) == 8  # two whole turns, nothing orphaned
    assert agent_mod.trim_history(items, limit=40) is items


async def test_server_tool_runs_here_and_never_reaches_the_device(env):
    s, sent, models = await make_session(meta=[
        tool_reply("r1", "get_weather", {"place": "Seattle"}, "c1"),
        text_reply("r2", "Light rain now, and it should stay wet this afternoon."),
    ])
    await s.handle_text({"type": "text", "text": "Will it rain in Seattle this afternoon?"})
    await s.turn
    assert not [m for m in sent if m["type"] == "tool_call"]
    weather = json.loads(models["meta"].responses.calls[1]["input"][-1]["output"])
    assert weather["place"] == "Seattle, Washington" and weather["now"]["sky"] == "light rain"
    assert weather["today"]["rain_chance_pct"] == 80 and weather["next_12h"][0]["time"] == "14:00"


async def test_device_timeout_comes_back_as_a_tool_error(env):
    s, sent, models = await make_session(meta=[
        tool_reply("r1", "get_location", {}, "c1"),
        text_reply("r2", "I couldn't get your location."),
    ])
    s.cfg = {**s.cfg, "turns": {**s.cfg["turns"], "device_tool_timeout_s": 0.05}}
    await s.handle_text({"type": "text", "text": "where am I"})
    await s.turn
    assert "error" in json.loads(models["meta"].responses.calls[1]["input"][-1]["output"])
    assert [m["entry"]["status"] for m in sent if m["type"] == "action"] == ["running", "failed"]


async def test_ignored_utterances_and_wake_only(env):
    s, sent, models = await make_session()
    await s.on_asr_event({"type": "speechComplete", "transcript": "so anyway I told him"})
    final = await next_msg(sent, "transcript")
    assert final["accepted"] is False and final["reason"] == "no_wake"
    assert s.turn is None
    await s.on_asr_event({"type": "speechComplete", "transcript": "EveryMuse?"})
    await s.turn
    assert says(sent) == ["Yes?"] and models["meta"].responses.calls == []


async def test_headset_tap_lets_the_next_sentence_skip_the_wake_word(env):
    s, sent, models = await make_session(meta=[text_reply("r1", "Sunny all afternoon.")])
    await s.handle_text({"type": "listen"})
    await s.on_asr_event({"type": "speechComplete", "transcript": "what's the weather like"})
    await s.turn
    final = next(m for m in sent if m["type"] == "transcript")
    assert final["accepted"] and final["reason"] == "follow_up"
    assert says(sent) == ["Sunny all afternoon."]


async def test_budget_stop(env):
    s, sent, models = await make_session()
    s.budget.add(1000)
    await s.handle_text({"type": "text", "text": "hello"})
    await s.turn
    assert "budget" in says(sent)[0] and models["meta"].responses.calls == []


async def test_think_requests_escalate_reasoning(env):
    s, sent, models = await make_session(meta=[text_reply("r1", "Here's a plan.")])
    await s.handle_text({"type": "text", "text": "think about how to plan my Saturday"})
    await s.turn
    assert says(sent)[0] == "Let me think about that."
    assert models["meta"].responses.calls[0]["reasoning"] == {"effort": "medium"}


async def test_chat_completions_path_for_open_weight_hosts(env, monkeypatch):
    monkeypatch.setenv("EVERYMUSE_BRAIN", "muse-spark-open")
    s, sent, models = await make_session(fireworks=[
        [chat_chunk(tool_calls=[chat_tool(0, "c1", "get_weather", '{"place": ')]),
         chat_chunk(tool_calls=[chat_tool(0, None, None, '"Seattle"}')])],
        [chat_chunk("Rain this afternoon. "), chat_chunk("Bring a jacket.")],
    ])
    await s.handle_text({"type": "text", "text": "will it rain in seattle"})
    await s.turn
    assert says(sent) == ["On it.", "Rain this afternoon.", "Bring a jacket."]
    call = models["fireworks"].responses.calls[1]
    assert call["reasoning_effort"] == "low" and "can't browse the web" in call["messages"][0]["content"]
    tool_msg = call["messages"][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["place"] == "Seattle, Washington"


async def test_brain_404_falls_back_and_later_turns_skip_it(env):
    s, sent, models = await make_session(
        meta=[not_found()],
        fireworks=[[chat_chunk("Sunny, around seventy.")], [chat_chunk("Still sunny.")]])
    await s.handle_text({"type": "text", "text": "how's the weather"})
    await s.turn
    assert says(sent) == ["Sunny, around seventy."]
    done = await next_msg(sent, "turn_done")
    assert done["fallback_from"] == "muse-spark-1.3" and done["model"].endswith("muse-spark-open")
    await s.handle_text({"type": "text", "text": "and later?"})
    await s.turn
    assert len(models["meta"].responses.calls) == 1  # the 404 is remembered; no second wasted call
    assert says(sent)[-1] == "Still sunny."


async def test_no_fallback_once_the_user_heard_part_of_the_answer(env):
    s, sent, models = await make_session(meta=[[*text_reply("r1", "Here is the first part. ")[:-1]]],
                                         fireworks=[[chat_chunk("should not be used")]])
    stream = models["meta"].responses.scripts[0]
    stream.append(type("E", (), {"type": "response.failed", "response": None})())
    await s.handle_text({"type": "text", "text": "tell me a story"})
    await s.turn
    assert models["fireworks"].responses.calls == []
    assert "Sorry, something went wrong." in says(sent)


PHOTO = "data:image/jpeg;base64," + "A" * 400


async def test_staged_photo_goes_with_the_next_spoken_question(env):
    s, sent, models = await make_session(meta=[text_reply("r1", "It says Pike Place Market.")])
    await s.handle_text({"type": "image", "data": PHOTO})
    await s.on_asr_event({"type": "speechComplete", "transcript": "what does that sign say"})  # no wake word needed
    await s.turn
    content = models["meta"].responses.calls[0]["input"][-1]["content"]
    assert content == [{"type": "input_text", "text": "what does that sign say"},
                       {"type": "input_image", "image_url": PHOTO}]
    assert any(m["type"] == "photo_used" for m in sent) and s.pending_image is None


async def test_photo_with_a_question_is_asked_right_away(env):
    s, sent, models = await make_session(meta=[text_reply("r1", "A golden retriever.")])
    await s.handle_text({"type": "image", "data": PHOTO, "text": "what breed is this?"})
    await s.turn
    assert models["meta"].responses.calls[0]["input"][-1]["content"][0]["text"] == "what breed is this?"
    assert says(sent) == ["A golden retriever."]


async def test_bad_or_stale_photos_are_not_sent(env):
    now = [0.0]
    s, sent, models = await make_session(meta=[text_reply("r1", "Hello.")], clock=lambda: now[0])
    await s.handle_text({"type": "image", "data": "data:text/html;base64,AAAA"})
    assert s.pending_image is None and any(m["type"] == "error" for m in sent)
    await s.handle_text({"type": "image", "data": PHOTO})
    now[0] = 500.0  # long past the two-minute window
    await s.handle_text({"type": "text", "text": "hello"})
    await s.turn
    assert models["meta"].responses.calls[0]["input"][-1]["content"] == "hello"


async def test_only_the_newest_photo_is_resent(env):
    s, sent, models = await make_session(meta=[text_reply("r1", "A cat."), text_reply("r2", "A dog.")])
    await s.handle_text({"type": "image", "data": PHOTO, "text": "what is this?"})
    await s.turn
    newer = PHOTO.replace("A", "B")
    await s.handle_text({"type": "image", "data": newer, "text": "and this?"})
    await s.turn
    second = models["meta"].responses.calls[1]["input"]
    images = [p for item in second if isinstance(item.get("content"), list)
              for p in item["content"] if p["type"] == "input_image"]
    assert [p["image_url"] for p in images] == [newer]
    assert second[0]["content"][1] == {"type": "input_text", "text": "[an earlier photo]"}


async def test_photo_on_the_chat_path(env, monkeypatch):
    monkeypatch.setenv("EVERYMUSE_BRAIN", "muse-spark-open")
    s, sent, models = await make_session(fireworks=[[chat_chunk("A red and blue square.")]])
    await s.handle_text({"type": "image", "data": PHOTO, "text": "colors?"})
    await s.turn
    content = models["fireworks"].responses.calls[0]["messages"][-1]["content"]
    assert content == [{"type": "text", "text": "colors?"}, {"type": "image_url", "image_url": {"url": PHOTO}}]


async def test_fallback_rolls_back_the_failed_attempt(env):
    s, sent, models = await make_session(meta=[not_found()], fireworks=[[chat_chunk("Hi there.")]])
    await s.handle_text({"type": "text", "text": "hello"})
    await s.turn
    assert s.agent.items == []  # no unanswered user item left for the brain's next turn
    assert [m["role"] for m in s.agent.messages] == ["user", "assistant"]


async def test_no_or_silence_cancels_the_reminder(env):
    for answer in ("no thanks", None):
        s, sent, models = await make_session(meta=[
            tool_reply("r1", "create_reminder", {"title": "call mom"}, "c1"), text_reply("r2", "Okay, I didn't add it."),
        ])
        s.cfg = {**s.cfg, "turns": {**s.cfg["turns"], "confirm_timeout_s": 0.05}}
        await s.handle_text({"type": "text", "text": "remind me to call mom"})
        await next_msg(sent, "confirm")
        if answer:
            await s.handle_text({"type": "text", "text": answer})  # typed answers work too
        await s.turn
        assert not [m for m in sent if m["type"] == "tool_call"]  # nothing reached the phone
        assert json.loads(models["meta"].responses.calls[1]["input"][-1]["output"])["cancelled"] is True
        assert [m["entry"]["status"] for m in sent if m["type"] == "action"] == ["running", "cancelled"]


async def test_memory_round_trip(env):
    s, sent, models = await make_session(meta=[
        tool_reply("r1", "remember", {"note": "parked on level 3, spot 42"}, "c1"), text_reply("r2", "Got it."),
        tool_reply("r3", "recall", {"query": "where parked"}, "c2"), text_reply("r4", "Level 3, spot 42."),
    ])
    await s.handle_text({"type": "text", "text": "remember I parked on level 3 spot 42"})
    await s.turn
    await s.handle_text({"type": "text", "text": "where did I park?"})
    await s.turn
    recalled = json.loads(models["meta"].responses.calls[3]["input"][-1]["output"])
    assert recalled["matched"] and recalled["notes"][0]["note"] == "parked on level 3, spot 42"
    assert (env / "memory" / "test.json").exists()


def research_scripts(goal: str) -> list:
    return [tool_reply("r1", "start_background_task", {"goal": goal}, "c1"),
            text_reply("r2", "I'll look into it and get back to you.")]


async def test_research_runs_in_the_background_and_reports_back(env):
    wonder = FakeResearchModel("SUMMARY: The Aventon Level 3 is the best pick for a hilly commute. "
                               "DETAILS: - Aventon Level 3, $1,399, torque sensor\n- Ride1Up 700, $1,295")
    researcher = Researcher(load_config(), Budget(100), clients={"meta": wonder}, poll_s=0.01)
    s, sent, models = await make_session(meta=research_scripts("the best e-bikes under $1,500"), researcher=researcher)
    await s.handle_text({"type": "text", "text": "look into the best e-bikes under 1500 and get back to me"})
    await s.turn
    report = await next_msg(sent, "report")
    assert "Ride1Up" in report["details"] and report["model"] == "muse-wonder-1.0"
    spoken = [m["text"] for m in sent if m["type"] == "say" and m["turn"] == report["turn"] and m["text"]]
    assert spoken == ["I looked into the best e-bikes under $1,500.",
                      "The Aventon Level 3 is the best pick for a hilly commute."]
    job = wonder.created[0]
    assert job["background"] is True and job["model"] == "muse-wonder-1.0" and {"type": "web_search"} in job["tools"]
    done = next(m for m in sent if m["type"] == "turn_done" and m.get("background"))
    assert done["searches"] == 1 and done["cost_usd"] > 0
    receipts = [(m["entry"]["tool"], m["entry"]["status"]) for m in sent if m["type"] == "action"]
    assert ("start_background_task", "done") in receipts and ("research", "done") in receipts
    assert "Aventon" in s.agent.items[-1]["content"]  # follow-up questions can use the report


async def test_report_waits_for_the_device_to_reconnect(env):
    wonder = FakeResearchModel("SUMMARY: Pike Place Market opens at nine. DETAILS: Daily 9 to 6.", polls=5)
    researcher = Researcher(load_config(), Budget(100), clients={"meta": wonder}, poll_s=0.01)
    s, sent, _ = await make_session(meta=research_scripts("Pike Place Market's hours"), researcher=researcher)
    await s.handle_text({"type": "text", "text": "find out when pike place opens and get back to me"})
    await s.turn
    await s.close()  # walked away before the job finished
    await asyncio.gather(*list(researcher.tasks))
    assert researcher.inbox["test"]
    s2, sent2, _ = await make_session(researcher=researcher)  # same device, later
    assert any(m["type"] == "say" and m["text"].startswith("While you were away") for m in sent2)
    assert "test" not in researcher.inbox


async def test_research_falls_back_when_wonder_is_unavailable(env):
    meta = FakeResearchModel(fail=not_found())
    fireworks = FakeResearchModel("SUMMARY: Probably the Aventon Level 3. DETAILS: From memory; no live prices.")
    researcher = Researcher(load_config(), Budget(100), clients={"meta": meta, "fireworks": fireworks}, poll_s=0.01)
    s, sent, _ = await make_session(meta=research_scripts("the best e-bikes under $1,500"), researcher=researcher)
    await s.handle_text({"type": "text", "text": "look into e-bikes and get back to me"})
    await s.turn
    report = await next_msg(sent, "report")
    assert report["model"].endswith("muse-spark-open")
    assert "can't browse the web" in fireworks.created[0]["messages"][0]["content"]


async def test_narration_beside_a_tool_call_is_not_spoken(env):
    from types import SimpleNamespace as NS

    from fakes import completed, item
    fc = item("function_call", name="get_weather", arguments='{"place": "Seattle"}', call_id="c1")
    narrated = [NS(type="response.output_item.added", item=fc),
                NS(type="response.output_text.delta", delta="I'll check Seattle's weather for you. "),
                completed("r1", [fc])]
    s, sent, models = await make_session(meta=[narrated, text_reply("r2", "Dry and overcast all afternoon.")])
    await s.handle_text({"type": "text", "text": "rain in seattle?"})
    await s.turn
    assert says(sent) == ["On it.", "Dry and overcast all afternoon."]


async def test_narration_before_a_search_after_a_tool_is_not_spoken(env):
    from types import SimpleNamespace as NS

    from fakes import completed, item
    search = item("web_search_call", id="ws1", status="completed")
    after_tool = [NS(type="response.output_item.added", item=item("message")),
                  NS(type="response.output_text.delta", delta="Got the weather. "),
                  NS(type="response.output_text.delta", delta="Now I'll check the hours."),
                  NS(type="response.output_item.added", item=search),
                  NS(type="response.output_item.added", item=item("message")),
                  NS(type="response.output_text.delta", delta="Rain after three, "),
                  NS(type="response.output_text.delta", delta="and they close at nine."),
                  completed("r2", [item("message"), search, item("message")])]
    s, sent, models = await make_session(meta=[
        tool_reply("r1", "get_weather", {"place": "Seattle"}, "c1"), after_tool])
    await s.handle_text({"type": "text", "text": "rain, and is the store open till nine?"})
    await s.turn
    assert says(sent) == ["On it.", "Rain after three, and they close at nine."]


async def test_text_cut_off_by_a_search_does_not_run_into_the_answer(env):
    from types import SimpleNamespace as NS

    from fakes import completed, item
    search = item("web_search_call", id="ws1", status="completed")
    events = [NS(type="response.output_item.added", item=item("message")),
              NS(type="response.output_text.delta", delta="Let me look that up"),
              NS(type="response.output_item.added", item=search),
              NS(type="response.output_item.added", item=item("message")),
              NS(type="response.output_text.delta", delta="They close at nine tonight."),
              completed("r1", [item("message"), search, item("message")])]
    s, sent, models = await make_session(meta=[events])
    await s.handle_text({"type": "text", "text": "are they open till nine?"})
    await s.turn
    assert says(sent) == ["Let me check.", "They close at nine tonight."]


def test_confirm_question_reads_a_capitalized_title_as_a_verb():
    from everymuse.agent import confirm_question
    assert (confirm_question("create_reminder", {"title": "Go to Clement Street Night Market"}, None)
            == "Should I remind you to go to Clement Street Night Market?")
    assert confirm_question("create_reminder", {"title": "Dentist"}, None) == "Should I remind you about Dentist?"


async def test_rewind_answers_from_speech_that_was_not_for_it(env):
    s, sent, models = await make_session(meta=[
        tool_reply("r1", "rewind", {}, "c1"), text_reply("r2", "1420 Pine Street, unit 3, at seven.")])
    await s.on_asr_event({"type": "speechComplete", "transcript": "Meet me at 1420 Pine Street, unit 3, at seven."})
    assert s.turn is None  # not said to the agent
    await s.on_asr_event({"type": "speechComplete", "transcript": "Hey EveryMuse, what was that address?"})
    await s.turn
    out = next(i for i in models["meta"].responses.calls[1]["input"] if i.get("type") == "function_call_output")
    heard = json.loads(out["output"])["heard"]
    assert len(heard) == 1 and heard[0].endswith("Meet me at 1420 Pine Street, unit 3, at seven.")
    assert says(sent) == ["On it.", "1420 Pine Street, unit 3, at seven."]
    receipt = [m["entry"] for m in sent if m["type"] == "action" and m["entry"]["status"] == "done"][-1]
    assert receipt["tool"] == "rewind" and receipt["result"] == {"lines": 1}
    assert "Pine" not in (session_mod.DATA_DIR / "actions" / "test.jsonl").read_text()  # words never hit disk


async def test_rewind_keeps_only_recent_speech_that_was_not_for_it(env):
    now = [1000.0]
    s, sent, models = await make_session(meta=[text_reply("r1", "Sure.")], clock=lambda: now[0])
    await s.on_asr_event({"type": "speechComplete", "transcript": "The train to Fremont leaves from platform two."})
    now[0] += 100
    await s.handle_text({"type": "tts", "state": "start", "text": "Your coffee is ready."})
    await s.on_asr_event({"type": "speechComplete", "transcript": "Your coffee is ready."})  # its own voice
    await s.handle_text({"type": "tts", "state": "end"})
    now[0] += 5
    await s.on_asr_event({"type": "speechComplete", "transcript": "Hey EveryMuse, thanks"})  # said to it
    await s.turn
    assert [line.split(": ", 1)[1] for line in s.heard.recent()] == ["The train to Fremont leaves from platform two."]
    now[0] += 90  # 195 s after the first line, past the 180 s window
    assert s.heard.recent() == []
    assert (await s.agent.tools["rewind"].run())["note"].startswith("nothing was heard")
    await s.close()
    assert not s.heard.lines


async def test_rewind_explains_push_to_talk(env):
    s, sent, models = await make_session(hello={**HELLO, "mode": "ptt"})
    result = await s.agent.tools["rewind"].run()
    assert result["heard"] == [] and "push-to-talk" in result["note"]


OFFER = ('{"kind": "reminder", "offer": "Want me to remind you about dinner Friday at seven?", '
         '"request": "Remind me about dinner at 1420 Pine Street on Friday at 7 PM."}')


async def heads_up_session(meta, clock=time.monotonic):
    s, sent, models = await make_session(meta=meta, hello={**HELLO, "heads_up": True}, clock=clock)
    s.cfg["heads_up"]["debounce_s"] = 0
    return s, sent, models


async def test_heads_up_offers_and_acts_on_a_yes_without_asking_twice(env):
    s, sent, models = await heads_up_session([
        text_reply("s1", OFFER),
        tool_reply("r1", "create_reminder", {"title": "Dinner at 1420 Pine Street", "due": "2026-11-06T19:00"}, "c1"),
        text_reply("r2", "Done, dinner Friday at seven."),
    ])
    await s.on_asr_event({"type": "speechComplete", "transcript": "Dinner's at my place Friday at seven, 1420 Pine."})
    question = await next_msg(sent, "confirm")
    assert question == {"type": "confirm", "turn": "h1", "question": "Want me to remind you about dinner Friday at seven?"}
    await s.on_asr_event({"type": "speechComplete", "transcript": "And bring that wine you like."})  # the friend again
    assert s.awaiting and not s.awaiting.done()  # only a short yes or no answers an offer
    await s.on_asr_event({"type": "speechComplete", "transcript": "Yes please."})
    call = await next_msg(sent, "tool_call")
    assert call["name"] == "create_reminder"
    await s.handle_text({"type": "tool_result", "id": call["id"], "ok": True, "result": {"ok": True}})
    await s.turn
    assert says(sent) == ["Want me to remind you about dinner Friday at seven?", "On it.", "Done, dinner Friday at seven."]
    assert sum(m["type"] == "confirm" for m in sent) == 1  # the yes to the offer was the only question
    screened = models["meta"].responses.calls[0]
    assert "Dinner's at my place" in screened["input"][0]["content"] and screened["reasoning"] == {"effort": "minimal"}
    receipt = [m["entry"] for m in sent if m["type"] == "action" and m["entry"]["tool"] == "heads_up"][-1]
    assert receipt["args"] == {"kind": "reminder"} and receipt["result"] == {"accepted": True}
    assert s.agent.preapproved == 0


async def test_heads_up_does_nothing_without_a_yes(env):
    s, sent, models = await heads_up_session([text_reply("s1", OFFER)])
    await s.on_asr_event({"type": "speechComplete", "transcript": "Dinner's at my place Friday at seven."})
    await next_msg(sent, "confirm")
    await s.on_asr_event({"type": "speechComplete", "transcript": "No thanks."})
    await s.turn
    assert not [m for m in sent if m["type"] == "tool_call"]
    receipt = [m["entry"] for m in sent if m["type"] == "action" and m["entry"]["tool"] == "heads_up"][-1]
    assert receipt["status"] == "cancelled" and "Dinner" not in json.dumps(receipt)


async def test_heads_up_is_off_by_default_quiet_when_nothing_matters_and_spaced_out(env):
    s, sent, models = await make_session(meta=[])
    await s.on_asr_event({"type": "speechComplete", "transcript": "Dinner's at my place Friday at seven."})
    await asyncio.sleep(0.05)
    assert not models["meta"].responses.calls  # off unless the client asks for it
    now = [1000.0]
    s, sent, models = await heads_up_session([text_reply("s1", '{"kind": "none"}'), text_reply("s2", OFFER)],
                                             clock=lambda: now[0])
    await s.on_asr_event({"type": "speechComplete", "transcript": "Nice weather today, huh."})
    await asyncio.sleep(0.05)
    assert len(models["meta"].responses.calls) == 1 and not [m for m in sent if m["type"] == "confirm"]
    await s.on_asr_event({"type": "speechComplete", "transcript": "Dinner's at my place Friday at seven."})
    await next_msg(sent, "confirm")
    await s.handle_text({"type": "text", "text": "no"})
    await s.turn
    now[0] += 10
    await s.on_asr_event({"type": "speechComplete", "transcript": "Oh, and lunch on Saturday at noon."})
    await asyncio.sleep(0.05)
    assert len(models["meta"].responses.calls) == 2  # too soon after an offer: not even screened


async def test_heads_up_alert_is_spoken_without_a_question(env):
    alert = '{"kind": "alert", "offer": "Heads-up: your 5:15 to Fremont now leaves from platform 5."}'
    s, sent, models = await heads_up_session([text_reply("s1", alert)])
    await s.memory.remember("I take the 5:15 to Fremont after work")
    await s.on_asr_event({"type": "speechComplete", "transcript": "The 5:15 to Fremont will now depart from platform 5."})
    done = await next_msg(sent, "turn_done")
    assert done["heads_up"] and says(sent) == ["Heads-up: your 5:15 to Fremont now leaves from platform 5."]
    assert not [m for m in sent if m["type"] == "confirm"]
    assert "5:15 to Fremont after work" in models["meta"].responses.calls[0]["instructions"]  # its notes give context


async def test_heads_up_screens_on_the_fallback_when_meta_is_unreachable(env):
    s, sent, models = await make_session(meta=[not_found()], fireworks=[[chat_chunk(OFFER)]],
                                         hello={**HELLO, "heads_up": True})
    s.cfg["heads_up"]["debounce_s"] = 0
    await s.on_asr_event({"type": "speechComplete", "transcript": "Dinner's at my place Friday at seven."})
    question = await next_msg(sent, "confirm")
    assert question["question"] == "Want me to remind you about dinner Friday at seven?"
    assert models["meta"].responses.calls[0]["model"] == "muse-spark-1.2"  # the screener the eval picked
    assert models["fireworks"].responses.calls[0]["model"].endswith("muse-spark-open")


SAM_RECORDS = "<0f>0<|box;x1=1500;y1=100;x2=1799;y2=299;w=1920;h=1280;c=0.91|><|mask;x=0;y=0;data=2,2,!AAA|>"


async def test_find_it_locates_with_sam_and_outlines_the_photo(env):
    s, sent, models = await make_session(meta=[
        tool_reply("r1", "find_in_photo", {"thing": "exit sign"}, "c1"),
        text_reply("sam", SAM_RECORDS),
        text_reply("r2", "The exit sign is at the top right."),
    ])
    await s.handle_text({"type": "image", "data": PHOTO, "text": "Where's the exit?"})
    await s.turn
    sam = models["meta"].responses.calls[1]
    assert sam["model"] == "sam-3.1" and sam["input"][0]["content"][0] == {"type": "input_text", "text": "exit sign"}
    highlight = await next_msg(sent, "highlight")
    assert highlight["boxes"] == [{"x1": 0.7812, "y1": 0.0781, "x2": 0.9375, "y2": 0.2344}]
    out = next(i for i in models["meta"].responses.calls[2]["input"] if i.get("type") == "function_call_output")
    assert json.loads(out["output"])["where"] == ["top right"]
    assert says(sent) == ["On it.", "The exit sign is at the top right."]


async def test_find_it_needs_a_recent_photo(env):
    s, sent, models = await make_session()
    assert "no photo" in (await s.agent.tools["find_in_photo"].run(thing="exit sign"))["error"]


def test_find_it_says_where_in_words():
    from everymuse.find import where
    assert where({"x1": 0.8, "y1": 0.05, "x2": 0.95, "y2": 0.25}) == "top right"
    assert where({"x1": 0.4, "y1": 0.4, "x2": 0.6, "y2": 0.6}) == "in the middle"
    assert where({"x1": 0.0, "y1": 0.4, "x2": 0.2, "y2": 0.6}) == "on the left"
    assert where({"x1": 0.4, "y1": 0.8, "x2": 0.6, "y2": 0.95}) == "at the bottom"
    assert where({"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.9}) == "filling most of the photo"


async def test_out_of_credit_falls_back_but_our_own_bad_request_does_not(env):
    def status(code: int) -> openai.APIStatusError:
        response = httpx.Response(code, request=httpx.Request("POST", "https://api.meta.ai/v1/responses"))
        return openai.APIStatusError(f"HTTP {code}", response=response, body=None)

    s, sent, models = await make_session(meta=[status(402)], fireworks=[[chat_chunk("Still here.")]])
    await s.handle_text({"type": "text", "text": "hello"})
    await s.turn
    done = await next_msg(sent, "turn_done")
    assert says(sent) == ["Still here."] and done["fallback_from"] == "muse-spark-1.3"
    assert "muse-spark" in agent_mod.UNAVAILABLE  # later turns skip the refused brain for a while

    agent_mod.UNAVAILABLE.clear()
    s, sent, models = await make_session(meta=[status(400)], fireworks=[[chat_chunk("should not be used")]])
    await s.handle_text({"type": "text", "text": "hello"})
    await s.turn
    assert not models["fireworks"].responses.calls and says(sent) == ["Sorry, something went wrong."]


def test_parse_offer_stays_quiet_when_unsure():
    from everymuse.headsup import parse_offer
    assert parse_offer('{"kind": "none"}') is None
    assert parse_offer("not json at all") is None
    assert parse_offer('{"kind": "reminder", "offer": "Remind you?"}') is None  # nothing to act on
    o = parse_offer('Sure: {"kind": "note", "offer": "Want me to save her number?", "request": "Remember Dana is 555 0134."}')
    assert o and o.kind == "note" and o.request.startswith("Remember")


def test_split_report_skips_wondering_out_loud():
    from everymuse.research import split_report
    text = ("Searching for Seattle's three tallest buildings to build your cited summary.I'll open that source to "
            "verify the figures.SUMMARY: Seattle's three tallest are Columbia Center at 933 feet, Rainier Square "
            "at 847 feet, and 1201 Third Avenue at 772 feet.\n\nDETAILS:\n- Columbia Center — 933 ft\n- Rainier Square — 847 ft")
    summary, details = split_report(text)
    assert summary.startswith("Seattle's three tallest") and summary.endswith("772 feet.")
    assert details.startswith("- Columbia Center")
    assert split_report("No format at all.") == ("No format at all.", "")


async def test_research_daily_limit(env):
    researcher = Researcher(load_config(), Budget(100), clients={"meta": FakeResearchModel("SUMMARY: ok.")}, poll_s=0.01)
    researcher.per_day = 1
    from everymuse.actions import ActionLog
    assert researcher.start("dev", "one thing", "", "UTC", ActionLog(None))["started"]
    assert "error" in researcher.start("dev", "another thing", "", "UTC", ActionLog(None))
    await asyncio.gather(*list(researcher.tasks))


def test_sentence_splitter_waits_for_whole_sentences():
    sp = Sentences()
    assert sp.feed("It's 3.5 degrees") == []
    assert sp.feed(" colder today. Bring a") == ["It's 3.5 degrees colder today."]
    assert sp.feed(" coat") == [] and sp.flush() == "Bring a coat"


async def test_research_in_flight_survives_a_server_restart(env):
    import json as _json
    store = env / "research"
    wonder = FakeResearchModel("SUMMARY: The Aventon Level 3 is the pick. DETAILS: $1,399.", polls=1000)
    researcher = Researcher(load_config(), Budget(100), clients={"meta": wonder}, poll_s=0.01, store=store)
    s, sent, _ = await make_session(meta=research_scripts("e-bikes under $1,500"), researcher=researcher)
    await s.handle_text({"type": "text", "text": "look into e-bikes under 1500 and get back to me"})
    await s.turn
    await asyncio.sleep(0.05)
    pending = list((store / "pending").glob("*.json"))
    assert len(pending) == 1 and _json.loads(pending[0].read_text())["response_id"] == "w1"
    researcher.shutting_down = True  # the server stops: its tasks are cancelled, the job runs on at Meta
    for t in list(researcher.tasks):
        t.cancel()
    await asyncio.gather(*list(researcher.tasks), return_exceptions=True)
    await s.close()
    assert pending[0].exists()  # still on disk for the next start
    done = FakeResearchModel("SUMMARY: The Aventon Level 3 is the pick. DETAILS: $1,399.", polls=1)
    after = Researcher(load_config(), Budget(100), clients={"meta": done}, poll_s=0.01, store=store)  # restarted
    assert after.resume() == 1
    await asyncio.gather(*list(after.tasks))
    assert not pending[0].exists() and (store / "inbox" / "test.json").exists()
    s2, sent2, _ = await make_session(researcher=after)  # the device reconnects
    spoken = [m["text"] for m in sent2 if m["type"] == "say" and m["text"]]
    assert spoken[0].startswith("While you were away") and "Aventon" in spoken[-1]
    assert not (store / "inbox" / "test.json").exists()


async def test_the_conversation_continues_after_a_server_restart(env):
    import json as _json
    s, sent, _ = await make_session(meta=[text_reply("r1", "Paris.")])
    await s.handle_text({"type": "text", "text": "What's the capital of France?"})
    await s.turn
    await s.close(clean=False)  # the server went away mid-session
    s2, sent2, models2 = await make_session(meta=[text_reply("r2", "Madrid.")])  # the page reconnects
    await s2.handle_text({"type": "text", "text": "And Spain?"})
    await s2.turn
    sent_input = models2["meta"].responses.calls[0]["input"]
    assert sent_input[0]["content"] == "What's the capital of France?"  # the turn before the restart
    assert sent_input[1]["type"] == "message" and sent_input[-1]["content"] == "And Spain?"  # its reply, then now


async def test_a_clean_close_forgets_the_conversation_and_photos_are_never_saved(env):
    import json as _json
    photo = "data:image/jpeg;base64," + "A" * 400
    s, sent, _ = await make_session(meta=[text_reply("r1", "A red bike.")])
    await s.stage_image({"type": "image", "data": photo, "text": "What's this?"})
    await s.turn
    saved = (env / "history" / "test.json").read_text()
    assert "base64" not in saved and "a photo the user shared" in saved
    await s.close(clean=True)
    assert not (env / "history" / "test.json").exists()
    s2, _, models2 = await make_session(meta=[text_reply("r2", "Hi.")])
    await s2.handle_text({"type": "text", "text": "hello"})
    await s2.turn
    assert "red bike" not in _json.dumps(models2["meta"].responses.calls[0]["input"])
