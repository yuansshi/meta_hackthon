import AVFoundation

/// Speaks the agent's sentences through the app's audio session, so they play in the headset.
/// Start and end are reported to the server, which ignores what the mic hears meanwhile (echo).
final class SpeechOut: NSObject, @unchecked Sendable, AVSpeechSynthesizerDelegate {
    var onStart: (@Sendable (String) -> Void)?
    var onEnd: (@Sendable (String) -> Void)?

    private let synth = AVSpeechSynthesizer()
    /// The best installed voice for the phone's language, which is also the language Earshot hears and answers in.
    private lazy var voice: AVSpeechSynthesisVoice? = {
        let tag = Locale.preferredLanguages.first ?? "en-US"
        let code = String(tag.prefix { $0 != "-" })
        let voices = AVSpeechSynthesisVoice.speechVoices().filter {
            $0.language.hasPrefix(code) && !$0.voiceTraits.contains(.isNoveltyVoice)
        }
        let exact = voices.filter { $0.language == tag }
        return (exact.isEmpty ? voices : exact).max { $0.quality.rawValue < $1.quality.rawValue }
            ?? AVSpeechSynthesisVoice(language: tag) ?? AVSpeechSynthesisVoice(language: "en-US")
    }()

    override init() {
        super.init()
        synth.delegate = self
        synth.usesApplicationAudioSession = true
    }

    var isSpeaking: Bool { synth.isSpeaking }
    var voiceName: String { voice.map { "\($0.name) (\(Self.quality($0.quality)))" } ?? "system default" }

    func speak(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = voice
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 1.04
        synth.speak(utterance)
    }

    func stop() {
        synth.stopSpeaking(at: .immediate)
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        onStart?(utterance.speechString)
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        onEnd?(utterance.speechString)
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        onEnd?(utterance.speechString)
    }

    private static func quality(_ q: AVSpeechSynthesisVoiceQuality) -> String {
        switch q {
        case .premium: "premium"
        case .enhanced: "enhanced"
        default: "default"
        }
    }
}
