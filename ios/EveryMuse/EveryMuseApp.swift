import SwiftUI

@main
struct EveryMuseApp: App {
    @State private var session = AgentSession()

    var body: some Scene {
        WindowGroup {
            SessionView(session: session)
        }
    }
}
