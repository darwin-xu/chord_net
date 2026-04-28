import XCTest
@testable import ChordNet

final class AudioEngineTests: XCTestCase {
    func testModelOutputNoteNamesMatchPianoMidiRange() {
        XCTAssertEqual(AudioEngine.noteNames.count, 88)
        XCTAssertEqual(AudioEngine.noteNames.first, "A0")
        XCTAssertEqual(AudioEngine.noteNames[60 - PianoNote.minimumMIDINote], "C4")
        XCTAssertEqual(AudioEngine.noteNames[69 - PianoNote.minimumMIDINote], "A4")
        XCTAssertEqual(AudioEngine.noteNames.last, "C8")
    }
}
