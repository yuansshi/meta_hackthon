import AVFoundation
import os

/// Owns the audio session and engine: the headset's Bluetooth (HFP) mic in, 16 kHz mono PCM16 out.
///
/// iOS won't let an app *start* recording from the background, so `start()` runs once in the foreground
/// and the engine then keeps running; turns are gated in software (`gateOpen`), never by stopping capture.
/// A silent looping player keeps us the Now Playing app, which is how headset taps reach the app, and
/// a second player plays the "thinking" chime into the same route.
final class AudioPipeline: @unchecked Sendable {
    enum PipelineError: LocalizedError {
        case noInput
        var errorDescription: String? { "No microphone input is available." }
    }

    /// Called on the audio thread with each converted chunk of 16 kHz PCM16.
    var sink: (@Sendable (Data) -> Void)?
    /// Diagnostics, interruptions and restarts; called on arbitrary threads.
    var onLog: (@Sendable (String) -> Void)?
    var onRestart: (@Sendable (Bool) -> Void)?

    private struct Meter { var gateOpen = false; var level: Float = 0; var buffers = 0 }
    private let meter = OSAllocatedUnfairLock(initialState: Meter())
    private let engine = AVAudioEngine()
    private let silence = AVAudioPlayerNode()
    private let chimePlayer = AVAudioPlayerNode()
    private var converter: AVAudioConverter?
    private var observers: [NSObjectProtocol] = []
    private let outFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: true)!
    private(set) var voiceProcessing = true
    private(set) var running = false

    var gateOpen: Bool {
        get { meter.withLock { $0.gateOpen } }
        set { meter.withLock { $0.gateOpen = newValue } }
    }

    /// Peak level and buffer count since the last call (the UI polls this once a second).
    func takeStats() -> (level: Float, buffers: Int) {
        meter.withLock { m in
            let out = (m.level, m.buffers)
            m.level = 0
            m.buffers = 0
            return out
        }
    }

    func start(voiceProcessing: Bool) throws {
        self.voiceProcessing = voiceProcessing
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.allowBluetoothHFP, .defaultToSpeaker])
        try session.setActive(true)
        preferHeadsetMic()
        try engine.inputNode.setVoiceProcessingEnabled(voiceProcessing)
        if !engine.attachedNodes.contains(silence) {
            engine.attach(silence)
            engine.attach(chimePlayer)
        }
        try rebuild()
        observe()
        running = true
        log("started · \(routeDescription()) · \(formatDescription())")
    }

    func stop() {
        running = false
        observers.forEach(NotificationCenter.default.removeObserver)
        observers = []
        engine.inputNode.removeTap(onBus: 0)
        silence.stop()
        chimePlayer.stop()
        engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    /// Re-reads the hardware format, reinstalls the tap and restarts the engine (route changes, interruptions).
    func restart(reason: String, reactivate: Bool = false) {
        guard running else { return }
        do {
            if reactivate {
                try AVAudioSession.sharedInstance().setActive(true)
                preferHeadsetMic()
            }
            engine.stop()
            try rebuild()
            log("restarted (\(reason)) · \(routeDescription()) · \(formatDescription())")
            onRestart?(true)
        } catch {
            log("restart FAILED (\(reason)): \(error.localizedDescription)")
            onRestart?(false)
        }
    }

    func routeDescription() -> String {
        let route = AVAudioSession.sharedInstance().currentRoute
        let ports = { (ps: [AVAudioSessionPortDescription]) in
            ps.isEmpty ? "none" : ps.map { "\($0.portName) [\($0.portType.rawValue)]" }.joined(separator: ", ")
        }
        return "in: \(ports(route.inputs)) · out: \(ports(route.outputs))"
    }

    func formatDescription() -> String {
        let session = AVAudioSession.sharedInstance()
        let f = engine.inputNode.outputFormat(forBus: 0)
        let band = session.sampleRate <= 8000 ? " (narrowband)" : ""
        return String(format: "hw %.0f Hz%@ · mic %.0f Hz × %d ch · io %.0f ms · voice processing %@",
                      session.sampleRate, band, f.sampleRate, Int(f.channelCount),
                      session.ioBufferDuration * 1000, voiceProcessing ? "on" : "off")
    }

    /// Two short rising notes, played into the headset the moment a turn is accepted.
    func chime() {
        guard running, let buffer = Self.chimeBuffer(format: chimePlayer.outputFormat(forBus: 0)) else { return }
        chimePlayer.scheduleBuffer(buffer, at: nil, options: .interrupts)
        if !chimePlayer.isPlaying { chimePlayer.play() }
    }

    // MARK: - internals

    private func preferHeadsetMic() {
        let session = AVAudioSession.sharedInstance()
        if let hfp = session.availableInputs?.first(where: { $0.portType == .bluetoothHFP }) {
            try? session.setPreferredInput(hfp)
        }
    }

    private func rebuild() throws {
        let input = engine.inputNode
        let inFormat = input.outputFormat(forBus: 0)
        guard inFormat.sampleRate > 0, inFormat.channelCount > 0 else { throw PipelineError.noInput }
        input.removeTap(onBus: 0)
        converter = AVAudioConverter(from: inFormat, to: outFormat)
        input.installTap(onBus: 0, bufferSize: 1024, format: inFormat) { [weak self] buffer, _ in
            self?.process(buffer)
        }
        let mixerRate = engine.mainMixerNode.outputFormat(forBus: 0).sampleRate
        let playFormat = AVAudioFormat(standardFormatWithSampleRate: mixerRate > 0 ? mixerRate : 48000, channels: 1)!
        engine.connect(silence, to: engine.mainMixerNode, format: playFormat)
        engine.connect(chimePlayer, to: engine.mainMixerNode, format: playFormat)
        engine.prepare()
        try engine.start()
        if let quiet = AVAudioPCMBuffer(pcmFormat: playFormat, frameCapacity: AVAudioFrameCount(playFormat.sampleRate)) {
            quiet.frameLength = quiet.frameCapacity  // zero-filled: one second of silence, looped
            silence.scheduleBuffer(quiet, at: nil, options: .loops)
            silence.play()
        }
    }

    private func process(_ buffer: AVAudioPCMBuffer) {
        let level = Self.rms(buffer)
        let open = meter.withLock { m -> Bool in
            m.level = max(m.level, level)
            m.buffers += 1
            return m.gateOpen
        }
        guard open, let converter, let sink else { return }
        let ratio = outFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
        guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return }
        var supplied = false
        var error: NSError?
        _ = converter.convert(to: out, error: &error) { _, status in
            if supplied {
                status.pointee = .noDataNow  // keeps the resampler's state for the next buffer
                return nil
            }
            supplied = true
            status.pointee = .haveData
            return buffer
        }
        guard error == nil, out.frameLength > 0, let samples = out.int16ChannelData else { return }
        sink(Data(bytes: samples[0], count: Int(out.frameLength) * 2))
    }

    private func observe() {
        let nc = NotificationCenter.default
        observers = [
            nc.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: nil) { [weak self] n in
                self?.interrupted(n)
            },
            nc.addObserver(forName: AVAudioSession.routeChangeNotification, object: nil, queue: nil) { [weak self] n in
                let raw = n.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt ?? 0
                let reason = AVAudioSession.RouteChangeReason(rawValue: raw).map(Self.describe) ?? "unknown"
                self?.log("route changed (\(reason)) · \(self?.routeDescription() ?? "")")
            },
            nc.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil) { [weak self] _ in
                self?.restart(reason: "engine configuration changed")
            },
            nc.addObserver(forName: AVAudioSession.mediaServicesWereResetNotification, object: nil, queue: nil) { [weak self] _ in
                self?.restart(reason: "media services reset", reactivate: true)
            },
        ]
    }

    private func interrupted(_ n: Notification) {
        guard let raw = n.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: raw) else { return }
        if type == .began {
            log("interruption began (Siri, a call, or another app took the audio)")
            return
        }
        let opts = AVAudioSession.InterruptionOptions(rawValue: n.userInfo?[AVAudioSessionInterruptionOptionKey] as? UInt ?? 0)
        log("interruption ended (shouldResume: \(opts.contains(.shouldResume)))")
        restart(reason: "after interruption", reactivate: true)
    }

    private func log(_ text: String) {
        onLog?(text)
    }

    private static func describe(_ r: AVAudioSession.RouteChangeReason) -> String {
        switch r {
        case .newDeviceAvailable: "new device"
        case .oldDeviceUnavailable: "device gone"
        case .categoryChange: "category change"
        case .override: "override"
        case .wakeFromSleep: "wake from sleep"
        case .noSuitableRouteForCategory: "no suitable route"
        case .routeConfigurationChange: "configuration change"
        default: "other"
        }
    }

    private static func rms(_ b: AVAudioPCMBuffer) -> Float {
        guard let d = b.floatChannelData, b.frameLength > 0 else { return 0 }
        let n = Int(b.frameLength)
        var sum: Float = 0
        for i in 0..<n { sum += d[0][i] * d[0][i] }
        return (sum / Float(n)).squareRoot()
    }

    private static func chimeBuffer(format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let rate = format.sampleRate > 0 ? format.sampleRate : 48000
        let note = Int(rate * 0.11)
        guard let fmt = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1),
              let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(note * 2)),
              let ch = buf.floatChannelData else { return nil }
        buf.frameLength = buf.frameCapacity
        for (k, freq) in [660.0, 880.0].enumerated() {
            for i in 0..<note {
                let t = Double(i) / rate
                let envelope = min(1, Double(i) / (rate * 0.01)) * exp(-t * 18)
                ch[0][k * note + i] = Float(sin(2 * .pi * freq * t) * envelope * 0.18)
            }
        }
        return buf
    }
}
