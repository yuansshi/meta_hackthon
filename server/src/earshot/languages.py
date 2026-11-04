"""Earshot in the 25 languages Muse Voice Transcribe is validated for.

The client says which language its user speaks, as a tag like "es-MX". The server then:
- biases speech recognition toward that language, plus English for the name "Earshot";
- tells Muse Spark, the Heads-up screener and Muse Wonder to answer in it;
- speaks its few fixed phrases in it, and hears yes and no in it.

scripts/eval_languages.py measured the effect of the bias in 23 of the languages (all with a macOS voice):
with an English-only bias the wake word was found in 13, and with the user's own language in all 23.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

PHRASES = ("on_it", "okay", "check", "budget_out", "went_wrong", "research_failed")


@dataclass(frozen=True)
class Language:
    code: str  # BCP-47 primary subtag
    name: str  # Meta's name for it: what languageBias and the prompts use
    native: str  # what the language menu shows
    phrases: tuple[str, ...]  # PHRASES, in order
    yes: tuple[str, ...]
    no: tuple[str, ...]
    spaced: bool = True  # words are separated by spaces (not in Japanese, Chinese or Thai)
    _re: dict = field(default_factory=dict, compare=False, repr=False)

    def say(self, key: str) -> str:
        if key == "listening":  # the answer to a bare "Hey Earshot"
            return LISTENING.get(self.code, "Yes?")
        return self.phrases[PHRASES.index(key)]

    @property
    def bias(self) -> list[str]:
        """Speech recognition's languageBias: the user's language, plus English for "Earshot" and mixed speech."""
        return [self.name] if self.code == "en" else [self.name, "English"]

    def _words(self, kind: str, start: bool) -> re.Pattern:
        key = (kind, start)
        if key not in self._re:
            words = sorted(map(re.escape, getattr(self, kind)), key=len, reverse=True)
            alt = "|".join(words)
            if self.spaced:
                pattern = ("^" if start else "(?:^| )") + f"(?:{alt})(?= |$)"
            else:
                pattern = ("^" if start else "") + f"(?:{alt})"
            self._re[key] = re.compile(pattern)
        return self._re[key]

    def is_yes(self, said: str) -> bool:
        """`said` is normalized text. A yes starts the answer, in this language or in English."""
        return bool(EN_YES.match(said) or (self.code != "en" and self._words("yes", True).search(said)))

    def is_no(self, said: str) -> bool:
        """A no anywhere in the answer wins over a yes ("yes, actually no")."""
        return bool(EN_NO.search(said) or (self.code != "en" and self._words("no", False).search(said)))

    def short(self, said: str) -> bool:
        """Short enough to be an answer rather than a sentence of conversation."""
        return len(said.split()) <= 4 if self.spaced else len(said.replace(" ", "")) <= 12


EN_YES = re.compile(r"^(yes|yeah|yep|yup|sure|ok|okay|do it|go ahead|please|correct|right|confirm|sounds good|"
                    r"absolutely)\b")
EN_NO = re.compile(r"\b(no|nope|nah|don't|do not|cancel|stop|never ?mind|wait)\b")


def _lang(code, name, native, phrases, yes, no, spaced=True) -> Language:
    assert len(phrases) == len(PHRASES), code
    return Language(code, name, native, tuple(phrases), tuple(yes), tuple(no), spaced)


LANGUAGES: dict[str, Language] = {lang.code: lang for lang in [
    _lang("en", "English", "English",
          ["On it.", "Okay.", "Let me check.", "I've used up today's demo budget. Please try again tomorrow.",
           "Sorry, something went wrong.", "Sorry, I couldn't finish that research."],
          [], []),
    _lang("ar", "Arabic", "العربية",
          ["حاضر.", "حسنًا.", "لحظة.", "نفدت ميزانية العرض التجريبي لليوم. حاول مرة أخرى غدًا.",
           "عذرًا، حدث خطأ ما.", "عذرًا، لم أتمكن من إنهاء هذا البحث."],
          ["نعم", "أجل", "ايوه", "أيوه", "اي", "حسنا", "حسنًا", "تمام", "طيب", "موافق"],
          ["لا", "كلا", "ألغ", "الغاء", "إلغاء", "توقف", "انتظر"]),
    _lang("bn", "Bengali", "বাংলা",
          ["এখনই দেখছি।", "ঠিক আছে।", "একটু দেখি।", "আজকের ডেমো বাজেট শেষ হয়ে গেছে। দয়া করে কাল আবার চেষ্টা করুন।",
           "দুঃখিত, কিছু একটা সমস্যা হয়েছে।", "দুঃখিত, ওই খোঁজটা শেষ করা গেল না।"],
          ["হ্যাঁ", "হাঁ", "হ্যা", "ঠিক আছে", "আচ্ছা", "অবশ্যই", "hya", "haan"],
          ["না", "বাতিল", "থামো", "দাঁড়াও", "na"]),
    _lang("nl", "Dutch", "Nederlands",
          ["Komt eraan.", "Oké.", "Even kijken.", "Het demobudget voor vandaag is op. Probeer het morgen opnieuw.",
           "Sorry, er ging iets mis.", "Sorry, ik kon dat onderzoek niet afmaken."],
          ["ja", "jawel", "graag", "prima", "oké", "goed", "doe maar", "zeker"],
          ["nee", "annuleer", "stop", "wacht", "liever niet"]),
    _lang("fr", "French", "Français",
          ["Je m'en occupe.", "D'accord.", "Je vérifie.", "J'ai épuisé le budget de démo du jour. Réessayez demain.",
           "Désolé, un problème est survenu.", "Désolé, je n'ai pas pu terminer cette recherche."],
          ["oui", "ouais", "d'accord", "bien sûr", "vas-y", "allez-y", "volontiers", "s'il te plaît",
           "s'il vous plaît", "parfait"],
          ["non", "annule", "annuler", "arrête", "attends", "pas maintenant"]),
    _lang("de", "German", "Deutsch",
          ["Mach ich.", "Okay.", "Ich schaue nach.",
           "Das Demo-Budget für heute ist aufgebraucht. Bitte versuch es morgen wieder.",
           "Entschuldigung, da ist etwas schiefgegangen.", "Entschuldigung, ich konnte die Recherche nicht abschließen."],
          ["ja", "jawohl", "klar", "genau", "gerne", "gern", "bitte", "mach das", "in ordnung", "sicher"],
          ["nein", "nee", "abbrechen", "stopp", "warte", "lieber nicht"]),
    _lang("he", "Hebrew", "עברית",
          ["אני על זה.", "בסדר.", "רגע אחד.", "נגמר תקציב ההדגמה של היום. נסו שוב מחר.", "סליחה, משהו השתבש.",
           "סליחה, לא הצלחתי לסיים את המחקר הזה."],
          ["כן", "בטח", "בסדר", "אוקיי", "יאללה", "בהחלט"],
          ["לא", "בטל", "עצור", "חכה"]),
    _lang("hi", "Hindi", "हिन्दी",
          ["अभी देखते हैं।", "ठीक है।", "एक पल।", "आज का डेमो बजट खत्म हो गया है। कृपया कल फिर कोशिश करें।",
           "माफ़ कीजिए, कुछ गड़बड़ हो गई।", "माफ़ कीजिए, वह खोज पूरी नहीं हो पाई।"],
          ["हाँ", "हां", "जी", "जी हाँ", "ठीक है", "बिल्कुल", "हाँजी", "haan", "ha", "ji"],
          ["नहीं", "नही", "मत", "रुको", "रद्द", "nahi", "nahin"]),
    _lang("id", "Indonesian", "Bahasa Indonesia",
          ["Siap.", "Oke.", "Saya cek dulu.", "Anggaran demo hari ini sudah habis. Coba lagi besok.",
           "Maaf, ada yang salah.", "Maaf, saya tidak bisa menyelesaikan riset itu."],
          ["ya", "iya", "boleh", "oke", "tentu", "silakan", "betul"],
          ["tidak", "nggak", "enggak", "gak", "jangan", "batal", "tunggu"]),
    _lang("it", "Italian", "Italiano",
          ["Ci penso io.", "Va bene.", "Controllo.", "Ho esaurito il budget demo di oggi. Riprova domani.",
           "Scusa, qualcosa è andato storto.", "Scusa, non sono riuscito a finire la ricerca."],
          ["sì", "si", "certo", "va bene", "d'accordo", "per favore", "vai", "volentieri"],
          ["no", "annulla", "fermati", "aspetta"]),
    _lang("ja", "Japanese", "日本語",
          ["了解です。", "わかりました。", "確認します。", "今日のデモ予算を使い切りました。また明日お試しください。",
           "すみません、問題が発生しました。", "すみません、その調査を完了できませんでした。"],
          ["はい", "うん", "ええ", "お願い", "そう", "いいよ", "いいですよ", "オッケー"],
          ["いいえ", "いや", "ううん", "やめて", "キャンセル", "待って", "だめ", "結構です"], spaced=False),
    _lang("kn", "Kannada", "ಕನ್ನಡ",
          ["ಈಗಲೇ ನೋಡುತ್ತೇನೆ.", "ಸರಿ.", "ಒಂದು ನಿಮಿಷ.", "ಇಂದಿನ ಡೆಮೊ ಬಜೆಟ್ ಮುಗಿದಿದೆ. ದಯವಿಟ್ಟು ನಾಳೆ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
           "ಕ್ಷಮಿಸಿ, ಏನೋ ತಪ್ಪಾಗಿದೆ.", "ಕ್ಷಮಿಸಿ, ಆ ಸಂಶೋಧನೆಯನ್ನು ಮುಗಿಸಲಾಗಲಿಲ್ಲ."],
          ["ಹೌದು", "ಸರಿ", "ಆಯ್ತು", "ಓಕೆ", "houdu", "haudu", "sari"],
          ["ಇಲ್ಲ", "ಬೇಡ", "ರದ್ದು", "ನಿಲ್ಲಿಸು", "illa", "beda"]),
    _lang("ko", "Korean", "한국어",
          ["알겠어요.", "네.", "확인해 볼게요.", "오늘 데모 예산을 다 썼어요. 내일 다시 시도해 주세요.",
           "죄송해요, 문제가 생겼어요.", "죄송해요, 그 조사를 끝내지 못했어요."],
          ["네", "응", "예", "좋아", "좋아요", "그래", "그래요", "부탁해"],
          ["아니", "아니요", "아뇨", "취소", "잠깐", "됐어"]),
    _lang("ms", "Malay", "Bahasa Melayu",
          ["Baik.", "Okey.", "Saya semak dulu.", "Bajet demo hari ini sudah habis. Sila cuba lagi esok.",
           "Maaf, ada sesuatu yang tidak kena.", "Maaf, saya tidak dapat menyiapkan kajian itu."],
          ["ya", "boleh", "okey", "baik", "tentu", "betul"],
          ["tidak", "tak", "jangan", "batal", "tunggu"]),
    _lang("zh", "Mandarin Chinese", "中文",
          ["好的。", "好。", "我查一下。", "今天的演示预算已经用完了，请明天再试。", "抱歉，出了点问题。",
           "抱歉，我没能完成这项调研。"],
          ["是", "好", "对", "可以", "行", "嗯", "要"],
          ["不要", "不用", "不行", "不是", "别", "取消", "算了", "等等"], spaced=False),
    _lang("mr", "Marathi", "मराठी",
          ["लगेच बघूया.", "ठीक आहे.", "एक मिनिट.", "आजचे डेमो बजेट संपले आहे. कृपया उद्या पुन्हा प्रयत्न करा.",
           "माफ करा, काहीतरी चुकले.", "माफ करा, ते संशोधन पूर्ण होऊ शकले नाही."],
          ["हो", "होय", "ठीक आहे", "चालेल", "नक्की", "ho", "hoy"],
          ["नाही", "नको", "रद्द", "थांब", "nahi", "nako"]),
    _lang("pl", "Polish", "Polski",
          ["Już się robi.", "Dobrze.", "Sprawdzam.", "Wykorzystałem dzisiejszy budżet demo. Spróbuj jutro.",
           "Przepraszam, coś poszło nie tak.", "Przepraszam, nie udało mi się dokończyć tego wyszukiwania."],
          ["tak", "jasne", "dobrze", "okej", "proszę", "pewnie", "oczywiście"],
          ["nie", "anuluj", "stop", "czekaj"]),
    _lang("pt", "Portuguese", "Português",
          ["Deixa comigo.", "Tá bom.", "Vou verificar.", "Usei todo o orçamento de demonstração de hoje. "
           "Tente de novo amanhã.", "Desculpe, algo deu errado.", "Desculpe, não consegui terminar essa pesquisa."],
          ["sim", "claro", "pode", "isso", "por favor", "beleza", "certo", "com certeza"],
          ["não", "nao", "cancela", "espera"]),
    _lang("es", "Spanish", "Español",
          ["Enseguida.", "Vale.", "Déjame ver.", "Ya usé el presupuesto de demostración de hoy. "
           "Vuelve a intentarlo mañana.", "Perdona, algo salió mal.", "Perdona, no pude terminar esa investigación."],
          ["sí", "si", "claro", "vale", "dale", "por favor", "de acuerdo", "correcto", "adelante", "perfecto"],
          ["no", "cancela", "cancelar", "espera", "mejor no"]),
    _lang("tl", "Tagalog", "Tagalog",
          ["Sige, titingnan ko.", "Sige.", "Sandali lang.", "Naubos na ang demo budget ngayong araw. "
           "Pakisubukan ulit bukas.", "Pasensya, may nagkaproblema.",
           "Pasensya, hindi ko natapos ang pananaliksik na iyon."],
          ["oo", "opo", "sige", "oke", "tama"],
          ["hindi", "huwag", "wag", "kanselahin", "teka", "ayaw"]),
    _lang("ta", "Tamil", "தமிழ்",
          ["இதோ பார்க்கிறேன்.", "சரி.", "ஒரு நிமிடம்.", "இன்றைய டெமோ பட்ஜெட் தீர்ந்துவிட்டது. நாளை மீண்டும் முயற்சிக்கவும்.",
           "மன்னிக்கவும், ஏதோ தவறு நடந்தது.", "மன்னிக்கவும், அந்த ஆய்வை முடிக்க முடியவில்லை."],
          ["ஆம்", "ஆமா", "ஆமாம்", "சரி", "ஓகே", "aam", "aama", "amam", "aamaam", "sari"],
          ["இல்லை", "வேண்டாம்", "ரத்து", "நிறுத்து", "illai", "vendam", "venam"]),
    _lang("te", "Telugu", "తెలుగు",
          ["ఇప్పుడే చూస్తాను.", "సరే.", "ఒక్క నిమిషం.", "ఈరోజు డెమో బడ్జెట్ అయిపోయింది. దయచేసి రేపు మళ్ళీ ప్రయత్నించండి.",
           "క్షమించండి, ఏదో పొరపాటు జరిగింది.", "క్షమించండి, ఆ పరిశోధనను పూర్తి చేయలేకపోయాను."],
          ["అవును", "సరే", "అలాగే", "ఓకే", "avunu", "sare"],
          ["వద్దు", "లేదు", "కాదు", "రద్దు", "ఆగు", "vaddu", "ledu"]),
    _lang("th", "Thai", "ไทย",
          ["รับทราบ", "ได้เลย", "ขอเช็กก่อนนะ", "งบเดโมของวันนี้หมดแล้ว กรุณาลองใหม่พรุ่งนี้", "ขออภัย มีบางอย่างผิดพลาด",
           "ขออภัย ทำการค้นคว้านั้นไม่เสร็จ"],
          ["ใช่", "ได้", "ครับ", "ค่ะ", "โอเค", "เอา", "ตกลง"],
          ["ไม่", "ยกเลิก", "เดี๋ยว", "หยุด"], spaced=False),
    _lang("tr", "Turkish", "Türkçe",
          ["Hemen bakıyorum.", "Tamam.", "Kontrol ediyorum.", "Bugünkü demo bütçesi bitti. Lütfen yarın tekrar deneyin.",
           "Üzgünüm, bir şeyler ters gitti.", "Üzgünüm, o araştırmayı bitiremedim."],
          ["evet", "tamam", "olur", "tabii", "peki", "lütfen"],
          ["hayır", "hayir", "iptal", "dur", "bekle", "yok"]),
    _lang("vi", "Vietnamese", "Tiếng Việt",
          ["Để tôi lo.", "Được.", "Để tôi xem.", "Hôm nay đã hết ngân sách demo. Vui lòng thử lại vào ngày mai.",
           "Xin lỗi, đã có lỗi xảy ra.", "Xin lỗi, tôi chưa thể hoàn thành việc tìm hiểu đó."],
          ["có", "vâng", "được", "ừ", "đúng", "dạ"],
          ["không", "đừng", "hủy", "khoan", "thôi"]),
]}
ALIASES = {"fil": "tl", "iw": "he", "in": "id", "cmn": "zh", "yue": "zh"}
# What Earshot says to a bare "Hey Earshot", so the user goes on.
LISTENING = {"en": "Yes?", "ar": "نعم؟", "bn": "হ্যাঁ, বলুন?", "nl": "Ja?", "fr": "Oui ?", "de": "Ja?", "he": "כן?",
             "hi": "जी, बोलिए?", "id": "Ya?", "it": "Sì?", "ja": "はい？", "kn": "ಹೇಳಿ?", "ko": "네?", "ms": "Ya?", "zh": "嗯？",
             "mr": "बोला?", "pl": "Tak?", "pt": "Sim?", "es": "¿Sí?", "tl": "Ano iyon?", "ta": "சொல்லுங்கள்?", "te": "చెప్పండి?",
             "th": "มีอะไรให้ช่วยไหม", "tr": "Evet?", "vi": "Vâng?"}
ENGLISH = LANGUAGES["en"]


def resolve(tag: object) -> Language:
    """A client's language tag ("es-MX", "zh-Hans-CN", "fil") as a supported Language; English otherwise."""
    code = str(tag or "").strip().replace("_", "-").split("-")[0].lower()
    return LANGUAGES.get(ALIASES.get(code, code), ENGLISH)


CJK_PUNCT = "、，。！？：；「」『』（）《》【】،؟"
_GLUED = re.compile(rf"([{CJK_PUNCT}])(?=\S)|([,.!?;:])(?=[^\x00-\x7F\s])")


def _latin(c: str) -> bool:
    return c.isascii() or "LATIN" in unicodedata.name(c, "")


def spaced(text: str) -> str:
    """Splits "Hey earshot,今天下午会下雨吗" into words: a space after punctuation glued to the next word, and between
    a Latin letter and a letter of another script. Japanese, Chinese and Thai don't put spaces between words."""
    t = _GLUED.sub(lambda m: (m.group(1) or m.group(2)) + " ", text)
    out: list[str] = []
    for prev, c in zip(" " + t, t):
        if prev.isalpha() and c.isalpha() and _latin(prev) != _latin(c):
            out.append(" ")
        out.append(c)
    return "".join(out)


# "Should I remind you…?" in each language: (question, today, tomorrow, another day, date). The server writes this
# question from the reminder's own arguments, so what the user says yes to is exactly what gets saved.
# {t} is a 24-hour time like 18:00; {d} and {m} are the day and month.
REMIND = {
    "ar": ("هل أذكّرك بـ«{title}» {when}؟", "اليوم الساعة {t}", "غدًا الساعة {t}", "يوم {date} الساعة {t}", "{d}/{m}"),
    "bn": ("আপনাকে কি {when} “{title}” মনে করিয়ে দেব?", "আজ {t}-এ", "কাল {t}-এ", "{date} তারিখে {t}-এ", "{d}/{m}"),
    "nl": ("Zal ik je herinneren aan ‘{title}’, {when}?", "vandaag om {t}", "morgen om {t}", "op {date} om {t}", "{d}-{m}"),
    "fr": ("Je te rappelle « {title} » {when} ?", "aujourd'hui à {t}", "demain à {t}", "le {date} à {t}", "{d}/{m}"),
    "de": ("Soll ich dich an „{title}“ erinnern, {when}?", "heute um {t}", "morgen um {t}", "am {date} um {t}", "{d}.{m}."),
    "he": ("להזכיר לך „{title}” {when}?", "היום ב-{t}", "מחר ב-{t}", "ב-{date} ב-{t}", "{d}/{m}"),
    "hi": ("क्या मैं आपको {when} “{title}” की याद दिलाऊँ?", "आज {t} बजे", "कल {t} बजे", "{date} को {t} बजे", "{d}/{m}"),
    "id": ("Mau saya ingatkan “{title}” {when}?", "hari ini jam {t}", "besok jam {t}", "tanggal {date} jam {t}", "{d}/{m}"),
    "it": ("Ti ricordo «{title}» {when}?", "oggi alle {t}", "domani alle {t}", "il {date} alle {t}", "{d}/{m}"),
    "ja": ("「{title}」を{when}リマインドしますか？", "今日の{t}に", "明日の{t}に", "{date}の{t}に", "{m}月{d}日"),
    "kn": ("{when} “{title}” ಎಂದು ನೆನಪಿಸಲೇ?", "ಇಂದು {t}ಕ್ಕೆ", "ನಾಳೆ {t}ಕ್ಕೆ", "{date}ರಂದು {t}ಕ್ಕೆ", "{d}/{m}"),
    "ko": ("{when}‘{title}’ 알려 드릴까요?", "오늘 {t}에 ", "내일 {t}에 ", "{date} {t}에 ", "{m}월 {d}일"),
    "ms": ("Nak saya ingatkan “{title}” {when}?", "hari ini pukul {t}", "esok pukul {t}", "pada {date} pukul {t}", "{d}/{m}"),
    "zh": ("要我{when}提醒你「{title}」吗？", "今天{t}", "明天{t}", "{date}{t}", "{m}月{d}日"),
    "mr": ("मी तुम्हाला {when} “{title}” ची आठवण करून देऊ का?", "आज {t} वाजता", "उद्या {t} वाजता", "{date} रोजी {t} वाजता",
           "{d}/{m}"),
    "pl": ("Przypomnieć ci o „{title}” {when}?", "dziś o {t}", "jutro o {t}", "{date} o {t}", "{d}.{m}"),
    "pt": ("Quer que eu te lembre de “{title}” {when}?", "hoje às {t}", "amanhã às {t}", "em {date} às {t}", "{d}/{m}"),
    "es": ("¿Te recuerdo «{title}» {when}?", "hoy a las {t}", "mañana a las {t}", "el {date} a las {t}", "{d}/{m}"),
    "tl": ("Ipapaalala ko ba ang “{title}” {when}?", "ngayong araw nang {t}", "bukas nang {t}", "sa {date} nang {t}",
           "{d}/{m}"),
    "ta": ("{when} “{title}” என்று நினைவூட்டட்டுமா?", "இன்று {t} மணிக்கு", "நாளை {t} மணிக்கு", "{date} அன்று {t} மணிக்கு",
           "{d}/{m}"),
    "te": ("{when} “{title}” గుర్తు చేయనా?", "ఈరోజు {t}కి", "రేపు {t}కి", "{date}న {t}కి", "{d}/{m}"),
    "th": ("ให้เตือนเรื่อง “{title}” {when} ไหม", "วันนี้ เวลา {t}", "พรุ่งนี้ เวลา {t}", "วันที่ {date} เวลา {t}", "{d}/{m}"),
    "tr": ("“{title}” için {when} hatırlatayım mı?", "bugün saat {t}", "yarın saat {t}", "{date} saat {t}", "{d}.{m}"),
    "vi": ("Tôi nhắc bạn “{title}” {when} nhé?", "lúc {t} hôm nay", "lúc {t} ngày mai", "lúc {t} ngày {date}", "{d}/{m}"),
}


def remind_question(lang: Language, title: str, when, today) -> str:
    """The reminder question in `lang` (not English, which agent.confirm_question words). `when` is the due datetime
    or None; `today` is the user's date."""
    q, on_today, on_tomorrow, on_day, date_fmt = REMIND[lang.code]
    phrase = ""
    if when is not None:
        t = f"{when.hour}:{when.minute:02d}"
        days = (when.date() - today).days
        phrase = (on_today if days == 0 else on_tomorrow if days == 1 else
                  on_day.replace("{date}", date_fmt.format(d=when.day, m=when.month))).format(t=t)
    text = q.format(title=title, when=phrase)
    return re.sub(r"\s+([?？])", r"\1", re.sub(r"\s{2,}", " ", text)).strip()
