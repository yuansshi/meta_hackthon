import Foundation
import os

/// The WebSocket to the agent server: JSON events both ways, binary 16 kHz PCM frames up.
/// Thread-safe: audio frames are sent straight from the audio thread.
final class AgentSocket: NSObject, @unchecked Sendable, URLSessionWebSocketDelegate {
    var onOpen: (@Sendable () -> Void)?
    var onMessage: (@Sendable ([String: Any]) -> Void)?
    var onClose: (@Sendable (String) -> Void)?

    private struct State { var task: URLSessionWebSocketTask?; var open = false; var inFlight = 0 }
    private let state = OSAllocatedUnfairLock(initialState: State())
    private lazy var session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    private let maxInFlightAudio = 50  // ~2-3 s of frames; beyond that the network is stalled and we drop audio

    var isOpen: Bool { state.withLock { $0.open } }

    func connect(_ url: URL) {
        let task = session.webSocketTask(with: url)
        task.maximumMessageSize = 4 << 20
        let old = state.withLock { s -> URLSessionWebSocketTask? in
            let old = s.task
            s = State(task: task)
            return old
        }
        old?.cancel(with: .goingAway, reason: nil)
        task.resume()
        receive(task)
        keepAlive(task)
    }

    func disconnect() {
        let task = state.withLock { s -> URLSessionWebSocketTask? in
            let t = s.task
            s = State()
            return t
        }
        task?.cancel(with: .normalClosure, reason: nil)
    }

    func send(_ object: [String: Any]) {
        guard let task = state.withLock({ $0.task }),
              let data = try? JSONSerialization.data(withJSONObject: object),
              let text = String(data: data, encoding: .utf8) else { return }
        task.send(.string(text)) { _ in }
    }

    func sendAudio(_ data: Data) {
        let task = state.withLock { s -> URLSessionWebSocketTask? in
            guard s.open, s.inFlight < maxInFlightAudio else { return nil }
            s.inFlight += 1
            return s.task
        }
        guard let task else { return }
        task.send(.data(data)) { [weak self] _ in
            self?.state.withLock { $0.inFlight = max(0, $0.inFlight - 1) }
        }
    }

    private func current(_ task: URLSessionWebSocketTask) -> Bool {
        state.withLock { $0.task === task }
    }

    private func receive(_ task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            guard let self, self.current(task) else { return }
            switch result {
            case .success(.string(let text)):
                if let data = text.data(using: .utf8),
                   let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    self.onMessage?(object)
                }
                self.receive(task)
            case .success:
                self.receive(task)
            case .failure(let error):
                self.state.withLock { $0.open = false }
                self.onClose?(error.localizedDescription)
            }
        }
    }

    private func keepAlive(_ task: URLSessionWebSocketTask) {
        Task { [weak self] in
            while let self, self.current(task) {
                try? await Task.sleep(for: .seconds(20))
                task.sendPing { _ in }
            }
        }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        guard current(webSocketTask) else { return }
        state.withLock { $0.open = true }
        onOpen?()
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        guard current(webSocketTask) else { return }
        state.withLock { $0.open = false }
        let why = reason.flatMap { String(data: $0, encoding: .utf8) } ?? ""
        onClose?("closed (\(closeCode.rawValue)) \(why)")
    }
}
