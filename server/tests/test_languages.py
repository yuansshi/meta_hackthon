"""Earshot in the user's language: speech recognition bias, replies, fixed phrases, wake word and yes/no."""

import pytest
from fakes import text_reply, tool_reply
from test_session import HELLO, OFFER, env, make_session, next_msg, says  # noqa: F401  (env is a fixture)

from earshot.config import load_config
from earshot.headsup import is_answer
from earshot.languages import ENGLISH, LANGUAGES, PHRASES, resolve, spaced
from earshot.session import Gate, normalize


def test_resolve_reads_client_language_tags():
    assert resolve("es-MX").name == "Spanish"
    assert resolve("zh-Hans-CN").name == "Mandarin Chinese"
    assert resolve("pt_BR").code == "pt" and resolve("fil").code == "tl" and resolve("iw").code == "he"
    assert resolve("ru") is ENGLISH and resolve(None) is ENGLISH and resolve("") is ENGLISH
    assert len(LANGUAGES) == 25  # every language Muse Voice Transcribe is validated for
    assert all(len(lang.phrases) == len(PHRASES) and all(lang.phrases) for lang in LANGUAGES.values())


def test_bias_keeps_english_for_the_wake_word():
    assert ENGLISH.bias == ["English"]
    assert resolve("ja").bias == ["Japanese", "English"]


@pytest.mark.parametrize("heard, rest", [
    ("Hey earshot,今天下午会下雨吗?", "今天下午会下雨吗?"),  # no space after the comma, as transcribed
    ("Hey Earshot、今日の午後は雨が降りますか？", "今日の午後は雨が降りますか？"),
    ("Hey earshot 오늘 오후에 비가 올까요?", "오늘 오후에 비가 올까요?"),
    ("Hey Earshot, ¿va a llover esta tarde?", "¿va a llover esta tarde?"),
    ("Hey earshot, क्या आज दोपहर बारिश होगी?", "क्या आज दोपहर बारिश होगी?"),
    ("ให้ Earshot บ่ายนี้ฝนจะตกไหม", "บ่ายนี้ฝนจะตกไหม"),
])
def test_wake_word_inside_other_scripts(heard, rest):
    assert Gate(load_config()["turns"]).find_wake(heard) == rest


def test_spacing_leaves_accented_latin_words_alone():
    assert spaced("Earshot, pioverà oggi pomeriggio?") == "Earshot, pioverà oggi pomeriggio?"
    assert spaced("chiều nay trời có mưa không") == "chiều nay trời có mưa không"
    assert spaced("It's 3.5 miles") == "It's 3.5 miles"


@pytest.mark.parametrize("code, yes, no", [
    ("es", ["sí, por favor", "claro", "yes"], ["no, gracias", "mejor no", "sí... no, espera"]),
    ("fr", ["oui", "d'accord"], ["non merci", "attends"]),
    ("de", ["ja bitte", "gerne"], ["nein danke", "lieber nicht"]),
    ("ja", ["はい、お願いします", "うん"], ["いいえ", "やめて"]),
    ("zh", ["好的", "可以", "是的"], ["不要", "算了", "不用了"]),
    ("hi", ["हाँ ठीक है", "जी"], ["नहीं", "मत करो"]),
    ("ar", ["نعم", "تمام"], ["لا شكرا"]),
    ("th", ["ใช่", "ได้เลย"], ["ไม่เอา", "ไม่ได้"]),
])
def test_yes_and_no_in_each_language(code, yes, no):
    lang = resolve(code)
    for said in yes:
        s = normalize(said)
        assert lang.is_yes(s) and not lang.is_no(s), said
    for said in no:
        assert lang.is_no(normalize(said)), said


def test_spanish_words_that_are_not_answers():
    es = resolve("es")
    assert not es.is_no(normalize("sí, para mañana"))  # "para" means "for" too, so it isn't a no
    assert not es.is_yes(normalize("siempre llego tarde"))  # "si" only counts as a whole word


def test_a_long_sentence_is_not_an_answer_to_an_offer():
    ja, zh = resolve("ja"), resolve("zh")
    assert is_answer(normalize("はい"), ja)
    assert not is_answer(normalize("明日の会議は十時からですが、資料はまだ要りません"), ja)
    assert not is_answer(normalize("我们明天晚上七点在我家吃饭，不要迟到"), zh)


async def test_a_spanish_session_hears_and_answers_in_spanish(env):
    s, sent, models = await make_session(hello={**HELLO, "language": "es-MX"}, meta=[
        tool_reply("r1", "get_weather", {"place": "Madrid"}, "c1"),
        text_reply("r2", "No, esta tarde no va a llover."),
    ])
    assert s.relay.cfg["languages"] == ["Spanish", "English"]
    assert (await next_msg(sent, "ready"))["language"] == "es"
    await s.on_asr_event({"type": "speechComplete", "transcript": "Hey Earshot, ¿va a llover esta tarde en Madrid?"})
    await s.turn
    assert says(sent) == ["Enseguida.", "No, esta tarde no va a llover."]
    instructions = models["meta"].responses.calls[0]["instructions"]
    assert "Always reply in Spanish" in instructions


async def test_a_spanish_reminder_is_confirmed_with_si(env):
    s, sent, _ = await make_session(hello={**HELLO, "language": "es"}, meta=[
        tool_reply("r1", "create_reminder", {"title": "Comprar café", "due": "2026-11-03T18:00"}, "c1"),
        text_reply("r2", "Listo."),
    ])
    await s.on_asr_event({"type": "speechComplete", "transcript": "Hey Earshot, recuérdame comprar café a las seis"})
    await next_msg(sent, "confirm")
    assert says(sent)[0] == "Vale."
    await s.on_asr_event({"type": "speechComplete", "transcript": "Sí, por favor."})
    call = await next_msg(sent, "tool_call")
    assert call["name"] == "create_reminder"


async def test_english_sessions_are_unchanged(env):
    s, sent, models = await make_session(meta=[text_reply("r1", "Hi.")])
    assert s.relay.cfg["languages"] == ["English"]
    await s.handle_text({"type": "text", "text": "hello"})
    await s.turn
    assert "Always reply in" not in models["meta"].responses.calls[0]["instructions"]


async def test_heads_up_offers_in_the_users_language(env):
    s, sent, models = await make_session(hello={**HELLO, "heads_up": True, "language": "fr"},
                                         meta=[text_reply("s1", OFFER)])
    s.cfg["heads_up"]["debounce_s"] = 0
    await s.on_asr_event({"type": "speechComplete", "transcript": "Le dîner est vendredi à sept heures, chez moi."})
    await next_msg(sent, "confirm")
    assert "write \"offer\" and \"request\" in French" in models["meta"].responses.calls[0]["instructions"]


def test_reminder_question_in_the_users_language():
    from datetime import date, datetime

    from earshot.agent import confirm_question
    from earshot.languages import REMIND, remind_question

    assert set(REMIND) == set(LANGUAGES) - {"en"}
    today = date(2026, 11, 4)
    six = datetime(2026, 11, 4, 18, 0)
    assert remind_question(resolve("es"), "Comprar café", six, today) == "¿Te recuerdo «Comprar café» hoy a las 18:00?"
    assert remind_question(resolve("ja"), "コーヒーを買う", datetime(2026, 11, 5, 6, 0), today) == \
        "「コーヒーを買う」を明日の6:00にリマインドしますか？"
    assert remind_question(resolve("zh"), "买咖啡", None, today) == "要我提醒你「买咖啡」吗？"
    assert remind_question(resolve("de"), "Kaffee kaufen", datetime(2026, 11, 9, 9, 30), today) == \
        "Soll ich dich an „Kaffee kaufen“ erinnern, am 9.11. um 9:30?"
    for code in REMIND:  # every template fills in without leftover braces
        q = remind_question(resolve(code), "X", six, today)
        assert "{" not in q and "X" in q and "18:00" in q, code
    assert confirm_question("create_reminder", {"title": "Dentist"}, None) == "Should I remind you about Dentist?"


async def test_a_pause_after_the_wake_word_keeps_the_question(env):
    import json as _json
    s, sent, models = await make_session(meta=[text_reply("r1", "No rain this afternoon.")])
    await s.on_asr_event({"type": "speechComplete", "transcript": "Hey Earshot."})
    await __import__("asyncio").sleep(0.2)  # the user paused, so speech recognition ended the turn there
    await s.on_asr_event({"type": "speechComplete", "transcript": "Will it rain this afternoon?"})
    await s.turn
    assert says(sent) == ["No rain this afternoon."]  # no "Yes?" in between
    assert "Will it rain this afternoon?" in _json.dumps(models["meta"].responses.calls[0]["input"])


async def test_a_sentence_split_at_a_pause_is_answered_whole(env):
    import json as _json
    s, sent, models = await make_session(meta=[text_reply("r1", "Platform three.")])
    models["meta"].responses.delay = 0.4
    await s.on_asr_event({"type": "speechComplete", "transcript": "Hey Earshot, which platform"})
    await __import__("asyncio").sleep(0.1)
    await s.on_asr_event({"type": "speechComplete", "transcript": "is the 19:05 to Harbor Quarter?"})
    await s.turn
    assert says(sent) == ["Platform three."]
    last = _json.dumps(models["meta"].responses.calls[-1]["input"])
    assert "which platform is the 19:05 to Harbor Quarter?" in last
    assert last.count("which platform") == 1  # the cut-off attempt left nothing in the history
    finals = [m for m in sent if m["type"] == "transcript" and m.get("final")]
    assert finals[-1]["reason"] == "merged" and finals[-1]["accepted"]


async def test_a_turn_that_has_spoken_is_not_restarted(env):
    s, sent, models = await make_session(meta=[text_reply("r1", "Sure.")])
    await s.on_asr_event({"type": "speechComplete", "transcript": "Hey Earshot, hello"})
    await s.turn
    await s.on_asr_event({"type": "speechComplete", "transcript": "so anyway I told him"})
    final = [m for m in sent if m["type"] == "transcript" and m.get("final")][-1]
    assert final["reason"] == "no_wake" and says(sent) == ["Sure."]


async def test_a_bare_wake_word_is_answered_in_the_users_language(env):
    s, sent, _ = await make_session(hello={**HELLO, "language": "es"})
    s.cfg["turns"]["wake_wait_s"] = 0.05
    await s.on_asr_event({"type": "speechComplete", "transcript": "Earshot?"})
    await s.turn
    assert says(sent) == ["¿Sí?"]


def test_romanized_short_answers_in_indic_languages():
    for code, said in (("te", "Avunu, సరే."), ("ta", "Amam, sari."), ("hi", "Haan ji"), ("kn", "houdu")):
        lang = resolve(code)
        assert lang.is_yes(normalize(said)) and not lang.is_no(normalize(said)), said


async def test_the_rest_after_a_pause_is_awaited_while_speech_goes_on(env):
    import asyncio as _asyncio
    s, sent, models = await make_session(meta=[text_reply("r1", "No rain today.")])
    s.cfg["turns"]["wake_wait_s"] = 0.1
    await s.on_asr_event({"type": "speechComplete", "transcript": "Hey Earshot."})
    await s.on_asr_event({"type": "speechStart"})  # they go on talking, longer than the wait
    await _asyncio.sleep(0.5)
    assert says(sent) == []  # no "Yes?" while they're mid-sentence
    await s.on_asr_event({"type": "speechComplete", "transcript": "Will it rain this afternoon?"})
    await s.turn
    assert says(sent) == ["No rain today."]


def test_leaked_planning_is_recognized():
    from earshot.agent import leaked
    assert leaked("Need location then weather - but must not announce tool? Developer says never announce that.", "he")
    assert leaked("Need location to check rain, but only one-three short sentences Hebrew, no announcement?", "he")
    assert leaked("The user wants the weather, so call get_location first, then the tool calls.", "en")
    assert not leaked("No, it should stay dry this afternoon in Seattle.", "en")
    assert not leaked("לא, אחר הצהריים יהיה יבש בסיאטל.", "he")
    assert not leaked("On it.", "he") and not leaked("Seattle", "ja")


async def test_a_reply_written_to_itself_is_not_spoken_and_retried(env):
    s, sent, models = await make_session(hello={**HELLO, "language": "he"}, meta=[
        text_reply("r1", "Need location then weather, but must not announce tool? Developer says never announce that."),
        text_reply("r2", "לא, אחר הצהריים יהיה יבש."),
    ])
    await s.handle_text({"type": "text", "text": "האם ירד גשם אחר הצהריים?"})
    await s.turn
    assert says(sent) == ["לא, אחר הצהריים יהיה יבש."]
    calls = models["meta"].responses.calls
    assert len(calls) == 2 and calls[1]["reasoning"] == {"effort": "low"}
