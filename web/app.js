// Earshot web client: mic → 16 kHz PCM over a WebSocket; server events → speech, bubbles and receipts.
const $ = (s) => document.querySelector(s);
const state = {
  ws: null, mode: "ptt", ptt: false, tailUntil: 0, muted: false, audio: null, retries: 0,
  server: "offline", bubbles: {}, thinking: null, turns: [], cost: 0, photo: null, photoTimer: null, quick: null,
  confirmTurn: null,
};
const STATUS = { listening: "Listening", thinking: "Thinking", speaking: "Speaking", offline: "Offline", connecting: "Connecting", muted: "Muted" };

const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};
const DEVICE_ID = store.get("earshot.device", null) || (() => {
  const id = "web-" + Math.random().toString(36).slice(2, 14);
  store.set("earshot.device", id);
  return id;
})();

function toast(text) {
  const t = $("#toast");
  t.textContent = text;
  t.classList.add("on");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove("on"), 4000);
}

function notice(text) {
  const n = $("#notice");
  n.textContent = text || "";
  n.classList.toggle("on", Boolean(text));
}

// One place decides what the pill says: speech wins, then the server's state, then mute.
function renderStatus() {
  let s = state.server;
  if (speech.speaking) s = "speaking";
  else if (s === "listening" && state.muted) s = "muted";
  $("#status").dataset.s = s === "muted" ? "offline" : s;
  $("#status-text").textContent = STATUS[s] || s;
}

function send(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(obj));
}

// Two soft notes the moment a turn is accepted, so the model's thinking time isn't dead air.
let earcon = null;
function chime() {
  if (!earcon) return;
  const t = earcon.currentTime;
  [660, 880].forEach((freq, i) => {
    const o = earcon.createOscillator();
    const g = earcon.createGain();
    const start = t + i * 0.09;
    o.frequency.value = freq;
    g.gain.setValueAtTime(0, start);
    g.gain.linearRampToValueAtTime(0.07, start + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, start + 0.12);
    o.connect(g).connect(earcon.destination);
    o.start(start);
    o.stop(start + 0.14);
  });
}

// ---------- speech out: one sentence at a time, reported to the server so it can ignore the echo ----------
const speech = {
  queue: [], speaking: false, voice: null,
  push(text) { this.queue.push(text); if (!this.speaking) this.next(); },
  next() {
    const text = this.queue.shift();
    if (text === undefined) { this.speaking = false; renderStatus(); return; }
    this.speaking = true;
    renderStatus();
    const u = new SpeechSynthesisUtterance(text);
    if (this.voice) u.voice = this.voice;
    u.rate = 1.05;
    let ended = false;
    const end = () => { if (ended) return; ended = true; send({ type: "tts", state: "end", text }); this.next(); };
    u.onend = end;
    u.onerror = end;
    send({ type: "tts", state: "start", text });
    speechSynthesis.speak(u);
  },
  stop() {
    this.queue = [];
    if (this.speaking) { speechSynthesis.cancel(); send({ type: "tts", state: "end" }); }
    this.speaking = false;
    renderStatus();
  },
};

// macOS ships joke voices ("Bubbles", "Zarvox") that would sink a demo; keep them out of the menu.
const NOVELTY = new Set(["Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Fred", "Good News",
  "Jester", "Junior", "Kathy", "Organ", "Ralph", "Superstar", "Trinoids", "Whisper", "Wobble", "Zarvox"]);

function loadVoices() {
  const voices = speechSynthesis.getVoices().filter((v) => v.lang && v.lang.startsWith("en") && !NOVELTY.has(v.name));
  if (!voices.length) return;
  const preferred = ["Samantha", "Ava", "Google US English", "Microsoft Aria", "Allison", "Karen", "Daniel"];
  const saved = store.get("earshot.voice", null);
  const pick = voices.find((v) => v.name === saved) || preferred.map((p) => voices.find((v) => v.name.includes(p))).find(Boolean) || voices[0];
  const sel = $("#voice");
  sel.innerHTML = "";
  for (const v of voices) sel.add(new Option(`${v.name} (${v.lang})`, v.name, false, v === pick));
  speech.voice = pick;
}
speechSynthesis.onvoiceschanged = loadVoices;
loadVoices();
$("#voice").addEventListener("change", (e) => {
  speech.voice = speechSynthesis.getVoices().find((v) => v.name === e.target.value) || null;
  store.set("earshot.voice", e.target.value);
});

// ---------- mic in ----------
function shouldSend() {
  if (state.muted) return false;
  if (state.mode === "wake") return true;
  return state.ptt || performance.now() < state.tailUntil;
}

async function startMic(deviceId) {
  stopMic();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { deviceId: deviceId ? { exact: deviceId } : undefined, channelCount: 1,
             echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule("pcm-worklet.js");
  const node = new AudioWorkletNode(ctx, "pcm16k");
  const sink = ctx.createGain();
  sink.gain.value = 0; // keeps the graph pulled without playing the mic back
  ctx.createMediaStreamSource(stream).connect(node).connect(sink).connect(ctx.destination);
  node.port.onmessage = ({ data }) => {
    $("#talk").style.setProperty("--level", shouldSend() ? Math.min(1, data.level * 6).toFixed(3) : 0);
    if (shouldSend() && state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(data.pcm);
  };
  state.audio = { stream, ctx };
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
  const sel = $("#mic");
  const current = stream.getAudioTracks()[0].getSettings().deviceId;
  sel.innerHTML = "";
  for (const d of devices) sel.add(new Option(d.label || "Microphone", d.deviceId, false, d.deviceId === current));
  notice("");
}

function stopMic() {
  if (!state.audio) return;
  state.audio.stream.getTracks().forEach((t) => t.stop());
  state.audio.ctx.close();
  state.audio = null;
}

$("#mic").addEventListener("change", (e) => startMic(e.target.value).catch((err) => toast(`Microphone: ${err.message}`)));

// ---------- conversation ----------
function addMsg(cls, text) {
  $("#empty")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  el.textContent = text;
  $("#log").append(el);
  $("#log").scrollTop = $("#log").scrollHeight;
  return el;
}

function showThinking() {
  if (state.thinking) return;
  state.thinking = addMsg("agent", "");
  state.thinking.setAttribute("aria-label", "Thinking");
  state.thinking.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
}

function clearThinking() {
  state.thinking?.remove();
  state.thinking = null;
}

function bubbleFor(turn) {
  let b = state.bubbles[turn];
  if (!b) {
    if (state.thinking && !turn.startsWith("r")) { // the dots become this turn's bubble, where the dots were
      b = state.thinking;
      state.thinking = null;
      b.removeAttribute("aria-label");
      b.innerHTML = "";
    } else {
      b = addMsg("agent", "");
    }
    b.append(document.createElement("span"));
    state.bubbles[turn] = b;
  }
  return b;
}

function agentSay(turn, text) {
  const span = bubbleFor(turn).firstElementChild;
  span.textContent = span.textContent ? `${span.textContent} ${text}` : text;
  $("#log").scrollTop = $("#log").scrollHeight;
}

// A finished Muse Wonder job: labelled bubble, details one click away, spoken summary to follow.
function showReport(msg) {
  const b = bubbleFor(msg.turn);
  b.classList.add("report");
  b.dataset.label = `${prettyModel(msg.model)} report`;
  if (msg.details) {
    const d = document.createElement("details");
    d.append(Object.assign(document.createElement("summary"), { textContent: "Details" }),
             Object.assign(document.createElement("div"), { textContent: msg.details }));
    b.append(d);
  }
  chime();
}

// "Add … to your reminders?": answer out loud, or tap.
function showConfirm(msg) {
  clearConfirm();
  const quick = document.createElement("div");
  quick.className = "quick";
  for (const answer of ["Yes", "No"]) {
    const btn = Object.assign(document.createElement("button"), { type: "button", textContent: answer });
    btn.addEventListener("click", () => {
      answered();
      speech.stop();
      addMsg("user", answer);
      send({ type: "text", text: answer.toLowerCase() });
    });
    quick.append(btn);
  }
  $("#log").append(quick);
  $("#log").scrollTop = $("#log").scrollHeight;
  state.quick = quick;
  state.confirmTurn = msg.turn;
}

// After the answer, the rest of the reply starts a new bubble below it, so the conversation reads in order.
function answered() {
  clearConfirm();
  if (state.confirmTurn) delete state.bubbles[state.confirmTurn];
  state.confirmTurn = null;
}

function clearConfirm() {
  state.quick?.remove();
  state.quick = null;
}

const shortModel = (id) => (id || "").split("/").pop();
const prettyModel = (id) => shortModel(id).replace(/^muse-spark-open$/, "Muse Spark Open")
  .replace(/^muse-spark-([\d.]+).*$/, "Muse Spark $1").replace(/^muse-wonder-([\d.]+).*$/, "Muse Wonder $1");

function setBrain(model, fallback) {
  $("#brain").textContent = prettyModel(model) + (fallback ? " (fallback)" : "");
  $("#brain-badge").classList.toggle("fallback", Boolean(fallback) || shortModel(model) === "muse-spark-open");
}

function turnDone(msg) {
  state.turns.push(msg.first_say_ms);
  state.cost += msg.cost_usd || 0;
  const firsts = state.turns.filter((x) => x != null).sort((a, b) => a - b);
  $("#stat-turns").textContent = state.turns.length;
  $("#stat-first").textContent = firsts.length ? `${(firsts[Math.floor((firsts.length - 1) / 2)] / 1000).toFixed(1)} s` : "–";
  $("#stat-cost").textContent = `$${state.cost.toFixed(state.cost < 0.1 ? 4 : 2)}`;
  setBrain(msg.model, msg.fallback_from);
  clearConfirm();
  const b = state.bubbles[msg.turn];
  if (!b) return;
  const meta = document.createElement("span");
  meta.className = "meta";
  const first = msg.background ? `${Math.round(msg.total_ms / 1000)} s in the background`
    : msg.first_say_ms != null ? `${(msg.first_say_ms / 1000).toFixed(1)} s to first word` : "no reply";
  const extra = msg.searches ? ` · ${msg.searches} web search${msg.searches > 1 ? "es" : ""}` : "";
  meta.textContent = `${first} · ${prettyModel(msg.model)}${msg.fallback_from ? " (fallback)" : ""}${extra} · $${msg.cost_usd.toFixed(4)}`;
  b.append(meta);
}

// ---------- receipts: say what happened in words, keep the JSON one click away ----------
function fmtDue(due) {
  const d = new Date(due);
  return isNaN(d) ? due : d.toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" });
}

function describe(e) {
  const a = e.args || {};
  const r = e.result || {};
  const fail = r.error ? `failed: ${r.error}` : r.cancelled ? "Not done: you didn't say yes." : "";
  switch (e.tool) {
    case "start_background_task":
      return { title: "Deep research", what: `“${a.goal || ""}”`,
               out: fail || (r.started ? "Muse Wonder is working on it in the background" : "") };
    case "research":
      return { title: "Research report", what: `“${a.goal || ""}”`, out: fail || r.summary || "" };
    case "remember":
      return { title: "Memory", what: "saved", out: fail || (r.saved ? `“${r.saved}”` : "") };
    case "recall": {
      const notes = (r.notes || []).slice(0, 3).map((n) => n.note).join(" · ");
      return { title: "Memory", what: a.query ? `looked up “${a.query}”` : "looked up notes",
               out: fail || notes || "Nothing saved yet" };
    }
    case "get_weather": {
      const day = (label, d) => (d ? ` · ${label} ${d.high_f}°/${d.low_f}°, ${d.rain_chance_pct}% rain` : "");
      return { title: "Weather", what: a.place || (a.latitude != null ? "your location" : ""),
               out: fail || (r.now ? `${r.now.temp_f}°F ${r.now.sky} now${day("today", r.today)}${day("tomorrow", r.tomorrow)}` : "") };
    }
    case "get_location":
      return { title: "Location", what: "", out: fail || r.place || (r.latitude != null ? `${r.latitude}, ${r.longitude} (±${r.accuracy_m} m)` : "") };
    case "create_reminder":
      return { title: "Reminder", what: `“${a.title || ""}”${a.due ? ` · ${fmtDue(a.due)}` : ""}`,
               out: fail || (r.ok ? `Saved to ${r.saved_to || r.list || "Reminders"}` : "") };
    case "rewind": {  // the receipt keeps a count, never the words heard
      const n = r.lines || 0;
      return { title: "Rewind", what: a.minutes ? `last ${a.minutes} min` : "last few minutes",
               out: fail || (n ? `${n} line${n === 1 ? "" : "s"} heard nearby` : r.note || "Nothing heard") };
    }
    default:
      return { title: e.tool, what: "", out: fail || "" };
  }
}

function upsertReceipt(e) {
  const box = $("#receipts");
  box.querySelector(".none")?.remove();
  let el = box.querySelector(`[data-id="${e.id}"]`);
  if (!el) {
    el = document.createElement("div");
    el.className = "receipt";
    el.dataset.id = e.id;
    box.prepend(el);
  }
  const d = describe(e);
  const took = e.ms >= 10000 ? `${Math.round(e.ms / 1000)} s` : `${e.ms} ms`;
  const label = e.status === "running" ? "running" : `${e.status} · ${took}`;
  el.innerHTML = "";
  const row = document.createElement("div");
  row.className = "row";
  row.append(Object.assign(document.createElement("b"), { textContent: d.title }),
             Object.assign(document.createElement("span"), { className: "what", textContent: d.what }),
             Object.assign(document.createElement("span"), { className: `st ${e.status}`, textContent: label }));
  el.append(row);
  if (d.out) el.append(Object.assign(document.createElement("div"), { className: "out", textContent: d.out }));
  el.append(Object.assign(document.createElement("div"), { className: "where",
    textContent: `${e.where === "device" ? "Ran on your device" : "Ran on the server"} at ${e.at}` }));
  const raw = document.createElement("details");
  raw.append(Object.assign(document.createElement("summary"), { textContent: "Raw call" }),
             Object.assign(document.createElement("code"), { textContent: JSON.stringify({ args: e.args, result: e.result }, null, 1) }));
  el.append(raw);
}

function renderReminders() {
  const list = store.get("earshot.reminders", []);
  const box = $("#reminders");
  if (!list.length) return;
  box.innerHTML = "";
  for (const r of list.slice().reverse()) {
    const el = Object.assign(document.createElement("div"), { className: "receipt" });
    const row = Object.assign(document.createElement("div"), { className: "row" });
    row.append(Object.assign(document.createElement("b"), { textContent: r.title }),
               Object.assign(document.createElement("span"), { className: "what", textContent: r.due ? fmtDue(r.due) : "" }));
    el.append(row);
    box.append(el);
  }
}

// ---------- device tools: what "on your device" means in a browser ----------
async function runTool(name, args) {
  if (name === "get_location") {
    const pos = await new Promise((resolve, reject) =>
      navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 8000, maximumAge: 300000 }));
    return { latitude: +pos.coords.latitude.toFixed(4), longitude: +pos.coords.longitude.toFixed(4),
             accuracy_m: Math.round(pos.coords.accuracy) };
  }
  if (name === "create_reminder") {
    const list = store.get("earshot.reminders", []);
    list.push({ title: args.title, due: args.due || null, notes: args.notes || null, created: new Date().toISOString() });
    store.set("earshot.reminders", list);
    renderReminders();
    return { ok: true, saved_to: "this browser's demo reminder list" };
  }
  return { error: `this device can't run ${name}` };
}

// ---------- server events ----------
function onEvent(msg) {
  switch (msg.type) {
    case "ready":
      setBrain(msg.brain, false);
      break;
    case "state":
      state.server = msg.value;
      if (msg.value === "thinking") { chime(); showThinking(); }
      if (msg.value === "listening") clearThinking();
      renderStatus();
      break;
    case "transcript":
      if (!msg.final) { $("#caption").textContent = msg.text ? `“${msg.text}”` : ""; break; }
      $("#caption").textContent = "";
      if (msg.accepted) { if (msg.reason === "answer") answered(); addMsg("user", msg.text); }
      else if (state.mode === "wake" && msg.reason === "no_wake" && msg.text) {
        $("#caption").textContent = `Heard “${msg.text}”. Start with “Hey Earshot” to ask something.`;
      }
      break;
    case "say":
      if (msg.text) { agentSay(msg.turn, msg.text); speech.push(msg.text); }
      break;
    case "tool_call":
      runTool(msg.name, msg.args || {})
        .then((result) => send({ type: "tool_result", id: msg.id, ok: !result.error, result, error: result.error }))
        .catch((err) => send({ type: "tool_result", id: msg.id, ok: false, error: err.message || String(err) }));
      break;
    case "action":
      upsertReceipt(msg.entry);
      break;
    case "photo_used":
      photoUsed();
      break;
    case "report":
      showReport(msg);
      break;
    case "confirm":
      showConfirm(msg);
      break;
    case "turn_done":
      turnDone(msg);
      break;
    case "error":
      toast(msg.message);
      break;
  }
}

function connect() {
  state.server = "connecting";
  renderStatus();
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    state.retries = 0;
    send({ type: "hello", client: "web", device_id: DEVICE_ID, mode: state.mode, sample_rate: 16000,
           tz: Intl.DateTimeFormat().resolvedOptions().timeZone, tools: ["get_location", "create_reminder"] });
  };
  ws.onmessage = (ev) => onEvent(JSON.parse(ev.data));
  ws.onclose = (ev) => {
    state.server = "offline";
    clearThinking();
    renderStatus();
    if (ev.code === 1013) { notice(ev.reason || "Too many open sessions. Close another tab and reload."); return; }
    if (state.retries++ < 5) setTimeout(connect, 1000 * state.retries);
    else notice("Lost the connection to the server. Reload the page to try again.");
  };
  state.ws = ws;
}

// ---------- controls ----------
const talk = $("#talk");
function pttDown() {
  if (state.mode !== "ptt") return;
  speech.stop();
  state.ptt = true;
  talk.classList.add("held");
  send({ type: "ptt", state: "down" });
}
function pttUp() {
  if (!state.ptt) return;
  state.ptt = false;
  state.tailUntil = performance.now() + 400; // keep the last word
  talk.classList.remove("held");
  send({ type: "ptt", state: "up" });
}
talk.addEventListener("pointerdown", (e) => {
  if (state.mode === "wake") {
    speech.stop();
    state.muted = !state.muted;
    talk.classList.toggle("muted", state.muted);
    send({ type: "mute", on: state.muted });
    renderStatus();
    return;
  }
  e.preventDefault();
  talk.setPointerCapture(e.pointerId);
  pttDown();
});
talk.addEventListener("pointerup", pttUp);
talk.addEventListener("pointercancel", pttUp);
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat && document.activeElement !== $("#text")) { e.preventDefault(); pttDown(); }
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space" && state.ptt) { e.preventDefault(); pttUp(); }
});

const HINTS = {
  ptt: ["Hold to talk", 'Hold the button<span class="kb"> or <kbd>Space</kbd></span> and speak. Pick your Bluetooth glasses or earbuds as the microphone below to hear answers in your ears.'],
  wake: ["Say “Hey Earshot…”", "Hands-free, like the glasses: start with “Hey Earshot”. For eight seconds after each answer, just keep talking. Tap the button to mute."],
};
function setMode(mode) {
  state.mode = mode;
  state.muted = false;
  talk.classList.remove("muted");
  for (const b of document.querySelectorAll(".modes button")) b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
  $("#hint-title").textContent = HINTS[mode][0];
  $("#hint-text").innerHTML = HINTS[mode][1];
  talk.setAttribute("aria-label", mode === "ptt" ? "Hold to talk" : "Mute or unmute");
  send({ type: "mode", mode });
  renderStatus();
}
for (const b of document.querySelectorAll(".modes button")) b.addEventListener("click", () => setMode(b.dataset.mode));

function sendText(text) {
  text = text.trim();
  if (!text) return;
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) { toast("Not connected yet. One moment."); return; }
  speech.stop();
  addMsg("user", text);
  send({ type: "text", text });
}
$("#typed").addEventListener("submit", (e) => { e.preventDefault(); sendText($("#text").value); $("#text").value = ""; });
for (const c of document.querySelectorAll(".chip")) {
  c.addEventListener("click", () => (c.dataset.action === "photo" ? $("#photo-file").click() : sendText(c.textContent)));
}

// ---------- photos: staged on the server, then sent with the next question, spoken or typed ----------
const PHOTO_TTL_MS = 115000; // the server keeps a staged photo for two minutes

async function downscale(file, max = 1024) {
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  const scale = Math.min(1, max / Math.max(bmp.width, bmp.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bmp.width * scale);
  canvas.height = Math.round(bmp.height * scale);
  canvas.getContext("2d").drawImage(bmp, 0, 0, canvas.width, canvas.height);
  bmp.close?.();
  return canvas.toDataURL("image/jpeg", 0.82);
}

function stagePhoto(dataUrl) {
  state.photo = dataUrl;
  $("#staged-img").src = dataUrl || "";
  $("#staged").hidden = !dataUrl;
  $("#text").placeholder = dataUrl ? "Ask about the photo" : "Or type a request";
  clearTimeout(state.photoTimer);
  if (dataUrl) state.photoTimer = setTimeout(() => stagePhoto(null), PHOTO_TTL_MS);
}

function photoUsed() {
  const users = document.querySelectorAll("#log .msg.user");
  const last = users[users.length - 1];
  if (last && state.photo) {
    last.prepend(Object.assign(document.createElement("img"), { className: "thumb", src: state.photo, alt: "Photo you asked about" }));
  }
  stagePhoto(null);
}

$("#photo-btn").addEventListener("click", () => $("#photo-file").click());
$("#photo-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = ""; // picking the same photo again should still fire
  if (!file) return;
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) { toast("Not connected yet. One moment."); return; }
  try {
    const dataUrl = await downscale(file);
    stagePhoto(dataUrl);
    send({ type: "image", data: dataUrl });
    $("#text").focus();
  } catch (err) {
    toast(`Couldn't read that photo (${err.message}).`);
  }
});
$("#staged-x").addEventListener("click", () => { stagePhoto(null); send({ type: "image", data: null }); });

$("#go").addEventListener("click", async () => {
  $("#start").remove();
  speechSynthesis.speak(new SpeechSynthesisUtterance("")); // unlocks speech on Safari, which needs a user gesture
  try { earcon = new AudioContext(); } catch {}
  connect();
  try {
    await startMic();
  } catch (err) {
    notice(`No microphone (${err.message}). You can still type, or use the sample prompts.`);
  }
});

renderReminders();
