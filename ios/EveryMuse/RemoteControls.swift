import MediaPlayer

/// Headset taps. iOS delivers play/pause/next/previous to the Now Playing app; the session decides
/// what a tap means. Whether taps still arrive while the mic's HFP link is open is spike S2.
@MainActor
final class RemoteControls {
    var onCommand: ((String) -> Void)?
    private var targets: [(MPRemoteCommand, Any)] = []

    func enable() {
        let center = MPRemoteCommandCenter.shared()
        let commands: [(MPRemoteCommand, String)] = [
            (center.togglePlayPauseCommand, "play/pause"), (center.playCommand, "play"),
            (center.pauseCommand, "pause"), (center.nextTrackCommand, "next"),
            (center.previousTrackCommand, "previous"),
        ]
        for (command, name) in commands {
            command.isEnabled = true
            let token = command.addTarget { [weak self] _ in
                Task { @MainActor in self?.onCommand?(name) }
                return .success
            }
            targets.append((command, token))
        }
        MPNowPlayingInfoCenter.default().nowPlayingInfo = [
            MPMediaItemPropertyTitle: "EveryMuse",
            MPMediaItemPropertyArtist: "Listening",
            MPNowPlayingInfoPropertyIsLiveStream: true,
            MPNowPlayingInfoPropertyPlaybackRate: 1.0,
        ]
    }

    func disable() {
        for (command, token) in targets { command.removeTarget(token) }
        targets = []
        MPNowPlayingInfoCenter.default().nowPlayingInfo = nil
    }
}
