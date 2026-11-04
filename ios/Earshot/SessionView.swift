import SwiftUI

struct SessionView: View {
    @Bindable var session: AgentSession
    @Environment(\.scenePhase) private var phase
    @State private var typed = ""
    @State private var holding = false
    @State private var showCamera = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    header
                    controls
                    if !session.caption.isEmpty {
                        Text(session.caption).italic().foregroundStyle(.secondary)
                    }
                    if !session.lastError.isEmpty {
                        Label(session.lastError, systemImage: "exclamationmark.triangle")
                            .font(.callout).foregroundStyle(.orange)
                    }
                    conversation
                    typedRow
                    receipts
                    diagnostics
                    settings
                }
                .padding()
            }
            .navigationTitle("Earshot")
        }
        .onChange(of: phase) { _, newPhase in session.scenePhaseChanged(newPhase) }
    }

    private var statusColor: Color {
        switch session.status {
        case "Listening": .green
        case "Thinking": .orange
        case "Speaking": .blue
        case "Offline": .red
        default: .gray
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Circle().fill(statusColor).frame(width: 10, height: 10)
            Text(session.status).font(.headline)
            Spacer()
            if !session.brain.isEmpty {
                Text(session.brain).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var controls: some View {
        VStack(spacing: 12) {
            Button {
                if session.running { session.stop() } else { Task { await session.start() } }
            } label: {
                Label(session.running ? "Stop" : "Start listening", systemImage: session.running ? "stop.fill" : "waveform")
                    .font(.title3.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
            .tint(session.running ? .gray : .blue)

            Picker("Mode", selection: $session.mode) {
                ForEach(AgentSession.Mode.allCases) { Text($0.label).tag($0) }
            }
            .pickerStyle(.segmented)

            if session.running {
                if session.mode == .ptt {
                    Text(holding ? "Listening…" : "Hold to talk")
                        .font(.headline)
                        .frame(maxWidth: .infinity, minHeight: 64)
                        .background(holding ? Color.blue : Color.blue.opacity(0.15), in: Capsule())
                        .foregroundStyle(holding ? .white : .blue)
                        .gesture(DragGesture(minimumDistance: 0)
                            .onChanged { _ in if !holding { holding = true; session.pttDown() } }
                            .onEnded { _ in holding = false; session.pttUp() })
                } else {
                    Toggle(session.muted ? "Muted" : "Say “Hey Earshot…”, or tap your glasses", isOn: Binding(
                        get: { !session.muted }, set: { session.muted = !$0 }))
                        .font(.callout)
                }
                ProgressView(value: min(1, Double(session.level) * 6))
                    .tint(session.level > 0.02 ? .green : .gray)
            }
        }
    }

    private var conversation: some View {
        VStack(spacing: 10) {
            if session.messages.isEmpty {
                Text("Pair your glasses or earbuds, tap Start, then put the phone in your pocket and say “Hey Earshot, will it rain this afternoon?”")
                    .font(.callout).foregroundStyle(.secondary)
            }
            ForEach(session.messages) { m in
                HStack {
                    if m.fromUser { Spacer(minLength: 40) }
                    VStack(alignment: .leading, spacing: 4) {
                        if let photo = m.photo {
                            Image(uiImage: photo).resizable().scaledToFit()
                                .frame(maxWidth: 160, maxHeight: 120)
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                        }
                        Text(m.text)
                        if !m.meta.isEmpty {
                            Text(m.meta).font(.caption2).foregroundStyle(m.fromUser ? .white.opacity(0.8) : .secondary)
                        }
                    }
                    .padding(10)
                    .background(m.fromUser ? Color.blue : Color(.secondarySystemBackground),
                                in: RoundedRectangle(cornerRadius: 14))
                    .foregroundStyle(m.fromUser ? .white : .primary)
                    if !m.fromUser { Spacer(minLength: 40) }
                }
            }
        }
    }

    private var typedRow: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let photo = session.stagedPhoto {
                HStack(spacing: 10) {
                    Image(uiImage: photo).resizable().scaledToFill()
                        .frame(width: 44, height: 44)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    Text("Photo attached. Ask about it out loud or type a question.").font(.callout)
                    Spacer()
                    Button(action: session.clearPhoto) {
                        Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                    }
                    .accessibilityLabel("Remove photo")
                }
                .padding(8)
                .background(Color.blue.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
            }
            HStack {
                TextField(session.stagedPhoto == nil ? "Or type a request" : "Ask about the photo", text: $typed)
                    .textFieldStyle(.roundedBorder)
                    .submitLabel(.send)
                    .onSubmit(send)
                Button { showCamera = true } label: { Label("Look", systemImage: "camera") }
                    .disabled(!session.running)
                Button("Send", action: send).disabled(!session.running || typed.isEmpty)
            }
        }
        .sheet(isPresented: $showCamera) {
            CameraPicker { session.sendPhoto($0) }.ignoresSafeArea()
        }
    }

    private func send() {
        session.sendText(typed)
        typed = ""
    }

    private var receipts: some View {
        DisclosureGroup("Receipts (\(session.receipts.count))") {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(session.receipts) { r in
                    VStack(alignment: .leading, spacing: 2) {
                        HStack {
                            Text(r.tool).font(.callout.weight(.semibold))
                            Spacer()
                            Text(r.status).font(.caption)
                                .foregroundStyle(r.status.hasPrefix("done") ? .green : r.status.hasPrefix("failed") ? .red : .orange)
                        }
                        Text(r.detail).font(.caption.monospaced()).foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.top, 6)
        }
    }

    private var diagnostics: some View {
        DisclosureGroup("Diagnostics") {
            VStack(alignment: .leading, spacing: 8) {
                Text(session.route).font(.caption.monospaced())
                Text(session.format).font(.caption.monospaced())
                Text("\(session.buffersPerSecond) mic buffers/s · echo heard \(session.echoTranscripts) · voice \(session.voice)")
                    .font(.caption.monospaced())
                Toggle("Voice processing (applies on next start)", isOn: $session.voiceProcessing).font(.callout)
                Toggle("Speak a test line every 30 s", isOn: $session.testSpeech).font(.callout)
                Button("Echo test: speak 10 sentences", action: session.runEchoTest).disabled(!session.running)
                Divider()
                ForEach(session.log.suffix(40).reversed()) { line in
                    Text("\(line.at.formatted(date: .omitted, time: .standard))  \(line.text)")
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.top, 6)
        }
    }

    private var settings: some View {
        DisclosureGroup("Server") {
            VStack(alignment: .leading, spacing: 6) {
                TextField("ws://<mac-ip>:8000/ws", text: $session.serverURL)
                    .textFieldStyle(.roundedBorder)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .keyboardType(.URL)
                    .disabled(session.running)
                Text("Same Wi-Fi as the Mac running `make serve`, or the deployed wss:// URL.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(.top, 6)
        }
    }
}
