import SwiftUI

@main
struct EarshotApp: App {
    @State private var session = AgentSession()

    var body: some Scene {
        WindowGroup {
            SessionView(session: session)
        }
    }
}
