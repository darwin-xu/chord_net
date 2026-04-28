import XCTest
@testable import ChordNet

@MainActor
final class NoteQueueTests: XCTestCase {
    func testEnqueueAddsSingleNoteEvent() {
        let queue = NoteQueue()
        let c4 = PianoNote(chordNetName: "C4")!

        queue.enqueue(c4)

        XCTAssertEqual(queue.events.count, 1)
        XCTAssertEqual(queue.events.first?.notes.map(\.displayName), ["C4"])
    }

    func testEnqueueChordDeduplicatesAndSortsByMidiNumber() {
        let queue = NoteQueue()
        let notes = [
            PianoNote(chordNetName: "G4")!,
            PianoNote(chordNetName: "C4")!,
            PianoNote(chordNetName: "E4")!,
            PianoNote(chordNetName: "C4")!,
        ]

        queue.enqueueChord(notes)

        XCTAssertEqual(queue.events.count, 1)
        XCTAssertEqual(queue.events.first?.notes.map(\.displayName), ["C4", "E4", "G4"])
    }

    func testEnqueueChordIgnoresEmptyInput() {
        let queue = NoteQueue()

        queue.enqueueChord([])

        XCTAssertTrue(queue.events.isEmpty)
    }
}
