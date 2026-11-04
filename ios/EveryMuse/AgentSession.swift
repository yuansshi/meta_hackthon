import AVFoundation
import Foundation
import Observation
import SwiftUI
import UserNotifications

/// One live conversation: headset mic → agent server; server events → speech, bubbles, receipts and
/// device tools. Everything here runs on the main actor; the audio and socket helpers are thread-safe.
@MainActor
@Observable
final class AgentSession {
    enum Mode: String, CaseIterable, Identifiable {
        case wake, ptt
        var id: String { rawValue }
        var label: String { self == .wake ? "Hands-free" : "Push to talk" }
    }

    struct Message: Identifiable {
        let id = UUID()
        let fromUser: Bool
        var text: String
        var meta = ""
        var photo: UIImage?
        var found: [CGRect] = []  // Find it: where SAM 3.1 found the thing, as fractions of the photo
    }

    struct Receipt: Identifiable {
        let id: String
        var tool: String
        var status: String
        var detail: String
    }

    struct LogLine: Identifiable {
        let id = UUID()
        let at = Date()
        let text: String
    }

    static let defaultURL = "wss://earshot-yuansshi.fly.dev/ws"

    var running = false
    var status = "Not started"
    var caption = ""
    var messages: [Message] = []
    var receipts: [Receipt] = []
    var log: [LogLine] = []
    var brain = ""
    var lastError = ""
    var level: Float = 0
    var buffersPerSecond = 0
    var route = "—"
    var format = "—"
    var voice = ""
    var echoTranscripts = 0
    var stagedPhoto: UIImage?
    var mode: Mode = .wake {
        didSet { if running { socket.send(["type": "mode", "mode": mode.rawValue]); updateGate() } }
    }
    var serverURL = UserDefaults.standard.string(forKey: "serverURL") ?? AgentSession.defaultURL {
        didSet { UserDefaults.standard.set(serverURL, forKey: "serverURL") }
    }
    var voiceProcessing = true
    var testSpeech = false
    /// Heads-up: EveryMuse speaks up when it overhears something you'll want, and asks first. Hands-free only.
    var headsUp = UserDefaults.standard.bool(forKey: "headsUp") {
        didSet {
            UserDefaults.standard.set(headsUp, forKey: "headsUp")
            if running { socket.send(["type": "heads_up", "on": headsUp]) }
        }
    }
    var muted = false {
        didSet { if running { socket.send(["type": "mute", "on": muted]); updateGate() } }
    }
    private(set) var pttHeld = false

    private let audio = AudioPipeline()
    private let socket = AgentSocket()
    private let speech = SpeechOut()
    private let tools = DeviceTools()
    private let remote = RemoteControls()
    private var bubbleForTurn: [String: UUID] = [:]
    private var pttTail: Task<Void, Never>?
    private var ticker: Task<Void, Never>?
    private var reconnect: Task<Void, Never>?
    private var silentSeconds = 0
    private var ticks = 0
    private var testCount = 0
    private var audioBroken = false
    private var serverState = "listening"
    private let deviceID: String = {
        if let id = UserDefaults.standard.string(forKey: "deviceID") { return id }
        let id = "ios-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12)
        UserDefaults.standard.set(id, forKey: "deviceID")
        return id
    }()

    init() {
        audio.sink = { [socket] data in socket.sendAudio(data) }
        audio.onLog = { [weak self] text in Task { @MainActor in self?.note(text) } }
        audio.onRestart = { [weak self] ok in Task { @MainActor in self?.audioRestarted(ok) } }
        socket.onOpen = { [weak self] in Task { @MainActor in self?.socketOpened() } }
        socket.onClose = { [weak self] why in Task { @MainActor in self?.socketClosed(why) } }
        socket.onMessage = { [weak self] msg in
            nonisolated(unsafe) let msg = msg
            Task { @MainActor in self?.handle(msg) }
        }
        speech.onStart = { [weak self] text in Task { @MainActor in self?.speechStarted(text) } }
        speech.onEnd = { [weak self] text in Task { @MainActor in self?.speechEnded(text) } }
        remote.onCommand = { [weak self] name in self?.headsetTap(name) }
        voice = speech.voiceName
        let nc = NotificationCenter.default
        nc.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.note("app → background") }
        }
        nc.addObserver(forName: UIApplication.willEnterForegroundNotification, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.note("app → foreground") }
        }
    }

    // MARK: - control

    func start() async {
        guard !running else { return }
        lastError = ""
        guard let url = URL(string: serverURL.trimmingCharacters(in: .whitespaces)),
              ["ws", "wss"].contains(url.scheme ?? "") else {
            lastError = "The server URL must start with ws:// or wss://"
            return
        }
        guard await AVAudioApplication.requestRecordPermission() else {
            lastError = "Microphone access is off. Turn it on in Settings → EveryMuse."
            return
        }
        _ = try? await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound])
        do {
            try audio.start(voiceProcessing: voiceProcessing)
        } catch {
            lastError = "Audio: \(error.localizedDescription)"
            note("audio start failed: \(error.localizedDescription)")
            return
        }
        running = true
        audioBroken = false
        status = "Connecting"
        route = audio.routeDescription()
        format = audio.formatDescription()
        tools.startLocation()
        remote.enable()
        socket.connect(url)
        startTicker()
    }

    func stop() {
        running = false
        ticker?.cancel()
        reconnect?.cancel()
        socket.disconnect()
        speech.stop()
        audio.stop()
        remote.disable()
        tools.stopLocation()
        audio.gateOpen = false
        status = "Stopped"
        level = 0
    }

    func sendText(_ text: String) {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard running, !text.isEmpty else { return }
        speech.stop()
        messages.append(Message(fromUser: true, text: text))
        socket.send(["type": "text", "text": text])
    }

    /// "Look": the photo waits on the server for the next question, spoken through the headset or typed.
    func sendPhoto(_ image: UIImage) {
        guard running else {
            lastError = "Start listening first, then tap Look."
            return
        }
        guard let jpeg = Self.jpegForUpload(image) else {
            lastError = "Couldn't prepare that photo."
            return
        }
        stagedPhoto = image
        socket.send(["type": "image", "data": "data:image/jpeg;base64," + jpeg.base64EncodedString()])
        caption = "Photo ready. Ask about it."
        note("photo staged (\(jpeg.count / 1024) KB)")
    }

    func clearPhoto() {
        stagedPhoto = nil
        socket.send(["type": "image", "data": NSNull()])
    }

    /// At most 1024 px on the long side, upright, JPEG: small enough to send in a second on cellular.
    static func jpegForUpload(_ image: UIImage, maxSide: CGFloat = 1024) -> Data? {
        let scale = min(1, maxSide / max(image.size.width, image.size.height))
        let size = CGSize(width: (image.size.width * scale).rounded(), height: (image.size.height * scale).rounded())
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let resized = UIGraphicsImageRenderer(size: size, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: size))
        }
        return resized.jpegData(compressionQuality: 0.8)
    }

    func pttDown() {
        guard running, mode == .ptt else { return }
        speech.stop()
        pttTail?.cancel()
        pttHeld = true
        updateGate()
    }

    func pttUp() {
        guard pttHeld else { return }
        pttHeld = false
        pttTail = Task { [weak self] in  // keep the last word
            try? await Task.sleep(for: .milliseconds(400))
            guard !Task.isCancelled else { return }
            self?.updateGate()
        }
    }

    /// Spike S4: say ten sentences with the mic open and count how many the server hears back.
    func runEchoTest() {
        guard running else { return }
        echoTranscripts = 0
        note("echo test: speaking 10 sentences")
        for i in 1...10 {
            speech.speak("Echo test sentence \(i). The quick brown fox jumps over the lazy dog.")
        }
    }

    func scenePhaseChanged(_ phase: ScenePhase) {
        guard phase == .active, running, audioBroken else { return }
        note("back in the foreground: restarting audio")
        audio.restart(reason: "user returned", reactivate: true)
    }

    // MARK: - events

    private func updateGate() {
        let tailOpen = pttTail.map { !$0.isCancelled } ?? false
        let wantAudio = mode == .wake || pttHeld || tailOpen
        audio.gateOpen = running && !muted && wantAudio
    }

    private func socketOpened() {
        guard running else { return }
        note("connected to \(serverURL)")
        socket.send([
            "type": "hello", "client": "ios", "device_id": deviceID, "tz": TimeZone.current.identifier,
            "sample_rate": 16000, "tools": ["get_location", "create_reminder"], "mode": mode.rawValue,
            "heads_up": headsUp, "language": Locale.preferredLanguages.first ?? "en-US",
        ])
        updateGate()
    }

    private func socketClosed(_ why: String) {
        guard running else { return }
        status = "Offline"
        note("server connection lost: \(why)")
        reconnect?.cancel()
        reconnect = Task { [weak self] in
            try? await Task.sleep(for: .seconds(2))
            guard let self, self.running, !Task.isCancelled, let url = URL(string: self.serverURL) else { return }
            self.status = "Connecting"
            self.socket.connect(url)
        }
    }

    private func handle(_ msg: [String: Any]) {
        switch msg["type"] as? String {
        case "ready":
            brain = msg["brain"] as? String ?? ""
            status = "Listening"
        case "state":
            serverState = msg["value"] as? String ?? ""
            if serverState == "thinking" { audio.chime() }
            refreshStatus()
        case "transcript":
            let text = msg["text"] as? String ?? ""
            guard msg["final"] as? Bool == true else {
                caption = text
                return
            }
            caption = ""
            let reason = msg["reason"] as? String ?? ""
            if msg["accepted"] as? Bool == true {
                messages.append(Message(fromUser: true, text: text))
            } else if reason.hasPrefix("echo") {
                echoTranscripts += 1
                note("ignored own voice (\(reason)): \(text)")
            } else if !text.isEmpty {
                caption = "Heard “\(text)”. Start with “Hey EveryMuse”."
            }
        case "say":
            guard let text = msg["text"] as? String, !text.isEmpty, let turn = msg["turn"] as? String else { return }
            if turn.hasPrefix("h"), bubbleForTurn[turn] == nil { audio.chime() }  // a Heads-up offer starts
            appendAgent(turn: turn, text: text)
            speech.speak(text)
        case "tool_call":
            guard let id = msg["id"] as? String, let name = msg["name"] as? String else { return }
            let args = msg["args"] as? [String: Any] ?? [:]
            Task {
                let result = await tools.run(name, args)
                let error = result["error"] as? String
                socket.send(["type": "tool_result", "id": id, "ok": error == nil, "result": result,
                             "error": error ?? NSNull()])
            }
        case "action":
            guard let entry = msg["entry"] as? [String: Any], let id = entry["id"] as? String else { return }
            upsertReceipt(id: id, entry: entry)
        case "photo_used":
            if let i = messages.lastIndex(where: { $0.fromUser }) { messages[i].photo = stagedPhoto }
            stagedPhoto = nil
        case "highlight":
            guard let boxes = msg["boxes"] as? [[String: Any]],
                  let i = messages.lastIndex(where: { $0.photo != nil }) else { return }
            messages[i].found = boxes.compactMap { b in
                guard let x1 = b["x1"] as? Double, let y1 = b["y1"] as? Double,
                      let x2 = b["x2"] as? Double, let y2 = b["y2"] as? Double else { return nil }
                return CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
            }
        case "report":  // a Muse Wonder job finished; its summary follows as `say` events
            audio.chime()
            note("research report: \(msg["goal"] as? String ?? "")")
        case "confirm":
            note("asked: \(msg["question"] as? String ?? "")")
        case "turn_done":
            guard let turn = msg["turn"] as? String, let bubble = bubbleForTurn[turn],
                  let i = messages.firstIndex(where: { $0.id == bubble }) else { return }
            let background = (msg["total_ms"] as? Double).map { String(format: "%.0f s in the background", $0 / 1000) }
            let first = msg["heads_up"] as? Bool == true ? "Heads-up"
                : msg["background"] as? Bool == true ? background ?? "in the background"
                : (msg["first_say_ms"] as? Double).map { String(format: "%.1f s to first word", $0 / 1000) } ?? "no reply"
            let model = (msg["model"] as? String ?? "").split(separator: "/").last.map(String.init) ?? ""
            let fallback = msg["fallback_from"] is String ? " (fallback)" : ""
            let cost = String(format: "$%.4f", msg["cost_usd"] as? Double ?? 0)
            messages[i].meta = "\(first) · \(model)\(fallback) · \(cost)"
        case "error":
            lastError = msg["message"] as? String ?? "error"
            note("server: \(lastError)")
        case "asr":  // speech recognition stopped answering (say, no credit), or came back
            let ok = msg["ok"] as? Bool ?? true
            lastError = ok ? "" : (msg["message"] as? String ?? "EveryMuse can't hear you right now.")
            note(ok ? "speech recognition is back" : "speech recognition unavailable")
        default:
            break
        }
    }

    private func appendAgent(turn: String, text: String) {
        if let bubble = bubbleForTurn[turn], let i = messages.firstIndex(where: { $0.id == bubble }) {
            messages[i].text += " " + text
        } else {
            let message = Message(fromUser: false, text: text)
            bubbleForTurn[turn] = message.id
            messages.append(message)
        }
        if messages.count > 60 { messages.removeFirst(messages.count - 60) }
    }

    private func upsertReceipt(id: String, entry: [String: Any]) {
        let tool = entry["tool"] as? String ?? "?"
        let status = entry["status"] as? String ?? ""
        let ms = (entry["ms"] as? Int).map { " · \($0) ms" } ?? ""
        let args = (try? JSONSerialization.data(withJSONObject: entry["args"] ?? [:])).flatMap { String(data: $0, encoding: .utf8) } ?? ""
        let receipt = Receipt(id: id, tool: tool, status: status + ms, detail: args)
        if let i = receipts.firstIndex(where: { $0.id == id }) { receipts[i] = receipt } else { receipts.insert(receipt, at: 0) }
    }

    private func speechStarted(_ text: String) {
        socket.send(["type": "tts", "state": "start", "text": text])
        refreshStatus()
    }

    private func speechEnded(_ text: String) {
        socket.send(["type": "tts", "state": "end", "text": text])
        refreshStatus()
    }

    private func refreshStatus() {
        guard running else { return }
        if speech.isSpeaking { status = "Speaking" } else if serverState == "thinking" { status = "Thinking" } else { status = muted ? "Muted" : "Listening" }
    }

    private func headsetTap(_ name: String) {
        note("headset tap: \(name)")
        guard running else { return }
        if speech.isSpeaking { speech.stop() }
        switch mode {
        case .wake:  // a tap means "I'm about to talk": the next sentence needs no wake word
            socket.send(["type": "listen"])
            audio.chime()
            caption = "Listening…"
        case .ptt:
            if pttHeld { pttUp() } else { pttDown() }
        }
    }

    private func audioRestarted(_ ok: Bool) {
        route = audio.routeDescription()
        format = audio.formatDescription()
        audioBroken = !ok
        guard !ok else { return }
        lastError = "Audio stopped. Open EveryMuse to resume."
        let content = UNMutableNotificationContent()
        content.title = "EveryMuse paused"
        content.body = "Tap to resume listening."
        UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: "resume", content: content, trigger: nil))
    }

    private func startTicker() {
        ticker?.cancel()
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                self?.tick()
            }
        }
    }

    private func tick() {
        guard running else { return }
        ticks += 1
        let stats = audio.takeStats()
        level = stats.level
        buffersPerSecond = stats.buffers
        if stats.buffers == 0 {  // spike S3: capture silently stopped (background, interruption, route)
            silentSeconds += 1
            if silentSeconds == 2 { note("no mic buffers for 2 s") }
        } else {
            if silentSeconds >= 2 { note("mic buffers back after \(silentSeconds) s") }
            silentSeconds = 0
        }
        if testSpeech, ticks % 30 == 0 {
            testCount += 1
            speech.speak("Test line \(testCount). Still listening.")
        }
    }

    private func note(_ text: String) {
        log.append(LogLine(text: text))
        if log.count > 200 { log.removeFirst(log.count - 200) }
        if text.hasPrefix("started") || text.hasPrefix("restarted") || text.hasPrefix("route") {
            route = audio.routeDescription()
            format = audio.formatDescription()
        }
    }
}
