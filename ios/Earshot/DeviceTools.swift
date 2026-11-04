import CoreLocation
import EventKit
import Foundation

/// What "on your device" means on iPhone: Reminders through EventKit, location through Core Location.
/// Location updates start with the session (in the foreground) so they keep coming in the pocket.
@MainActor
final class DeviceTools: NSObject, CLLocationManagerDelegate {
    private let store = EKEventStore()
    private let manager = CLLocationManager()
    private var latest: CLLocation?
    private var cachedPlace: (location: CLLocation, name: String)?

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyHundredMeters
        manager.distanceFilter = 50
    }

    func startLocation() {
        manager.requestWhenInUseAuthorization()
        manager.allowsBackgroundLocationUpdates = true
        manager.pausesLocationUpdatesAutomatically = false
        manager.showsBackgroundLocationIndicator = true
        manager.startUpdatingLocation()
    }

    func stopLocation() {
        manager.stopUpdatingLocation()
    }

    func run(_ name: String, _ args: [String: Any]) async -> [String: Any] {
        switch name {
        case "create_reminder": await createReminder(args)
        case "get_location": await location()
        default: ["error": "this device can't run \(name)"]
        }
    }

    private func createReminder(_ args: [String: Any]) async -> [String: Any] {
        guard let title = args["title"] as? String, !title.isEmpty else { return ["error": "a reminder needs a title"] }
        do {
            guard try await store.requestFullAccessToReminders() else {
                return ["error": "Reminders access is off; turn it on in Settings"]
            }
            let reminder = EKReminder(eventStore: store)
            reminder.title = title
            reminder.notes = args["notes"] as? String
            reminder.calendar = store.defaultCalendarForNewReminders()
            if let due = (args["due"] as? String).flatMap(Self.parseLocalDate) {
                reminder.dueDateComponents = Calendar.current.dateComponents([.year, .month, .day, .hour, .minute], from: due)
                reminder.addAlarm(EKAlarm(absoluteDate: due))
            }
            try store.save(reminder, commit: true)
            return ["ok": true, "list": reminder.calendar?.title ?? "Reminders", "due": args["due"] as? String ?? NSNull()]
        } catch {
            return ["error": error.localizedDescription]
        }
    }

    /// The model sends local times like "2026-11-03T18:00"; accept a few nearby shapes too.
    static func parseLocalDate(_ text: String) -> Date? {
        let iso = ISO8601DateFormatter()
        if let date = iso.date(from: text) { return date }
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = .current
        for format in ["yyyy-MM-dd'T'HH:mm:ss", "yyyy-MM-dd'T'HH:mm", "yyyy-MM-dd HH:mm", "yyyy-MM-dd"] {
            f.dateFormat = format
            if let date = f.date(from: text) { return date }
        }
        return nil
    }

    private func location() async -> [String: Any] {
        guard let here = latest ?? manager.location else {
            return ["error": "location isn't available yet; check that Earshot may use your location"]
        }
        var out: [String: Any] = [
            "latitude": (here.coordinate.latitude * 1e4).rounded() / 1e4,
            "longitude": (here.coordinate.longitude * 1e4).rounded() / 1e4,
            "accuracy_m": Int(here.horizontalAccuracy),
        ]
        if let name = await placeName(here) { out["place"] = name }
        return out
    }

    private func placeName(_ here: CLLocation) async -> String? {
        if let cached = cachedPlace, cached.location.distance(from: here) < 200 { return cached.name }
        guard let mark = try? await CLGeocoder().reverseGeocodeLocation(here).first else { return nil }
        let name = [mark.subLocality ?? mark.name, mark.locality].compactMap { $0 }.joined(separator: ", ")
        guard !name.isEmpty else { return nil }
        cachedPlace = (here, name)
        return name
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let last = locations.last else { return }
        Task { @MainActor in self.latest = last }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {}
}
