import XCTest
@testable import ChordNet

final class KeyboardLayoutTests: XCTestCase {
    func testFullKeyboardRangeMatchesAnEightyEightKeyPiano() {
        XCTAssertEqual(KeyboardLayout.noteRange.count, 88)
        XCTAssertEqual(KeyboardLayout.noteRange.first?.midiNumber, PianoNote.minimumMIDINote)
        XCTAssertEqual(KeyboardLayout.noteRange.first?.displayName, "A0")
        XCTAssertEqual(KeyboardLayout.noteRange.first?.frequency ?? .nan, 27.5, accuracy: 0.0001)

        XCTAssertEqual(KeyboardLayout.noteRange.last?.midiNumber, PianoNote.maximumMIDINote)
        XCTAssertEqual(KeyboardLayout.noteRange.last?.displayName, "C8")
        XCTAssertEqual(KeyboardLayout.noteRange.last?.frequency ?? .nan, 4186.009, accuracy: 0.001)
    }

    func testVisibleRangeIsOneChromaticOctaveFromMiddleC() {
        XCTAssertEqual(KeyboardLayout.visibleNotes.map(\.displayName), [
            "C4", "C#4", "D4", "D#4", "E4", "F4",
            "F#4", "G4", "G#4", "A4", "A#4", "B4",
        ])
        XCTAssertEqual(KeyboardLayout.visibleWhiteKeys.map(\.displayName), [
            "C4", "D4", "E4", "F4", "G4", "A4", "B4",
        ])
        XCTAssertEqual(KeyboardLayout.visibleBlackKeys.map(\.displayName), [
            "C#4", "D#4", "F#4", "G#4", "A#4",
        ])
    }

    func testWhiteKeyOffsetsAreStableForLayoutMath() {
        XCTAssertEqual(KeyboardLayout.whiteKeyOffset(for: 21), 0)
        XCTAssertEqual(KeyboardLayout.whiteKeyOffset(for: 60), 23)
        XCTAssertEqual(KeyboardLayout.visibleWhiteKeyOffset(for: 60), 0)
        XCTAssertEqual(KeyboardLayout.visibleWhiteKeyOffset(for: 61), 1)
        XCTAssertEqual(KeyboardLayout.visibleWhiteKeyOffset(for: 63), 2)
        XCTAssertEqual(KeyboardLayout.visibleWhiteKeyOffset(for: 71), 6)
    }
}
