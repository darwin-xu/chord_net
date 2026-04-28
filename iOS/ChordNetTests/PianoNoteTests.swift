import XCTest
@testable import ChordNet

final class PianoNoteTests: XCTestCase {
    func testFrequencyConversionUsesEqualTemperament() {
        let a4 = PianoNote.from(frequency: 440)

        XCTAssertEqual(a4?.midiNumber, 69)
        XCTAssertEqual(a4?.displayName, "A4")
        XCTAssertEqual(a4?.pianoKeyIndex, 48)
        XCTAssertEqual(a4?.centsOffset ?? .nan, 0, accuracy: 0.0001)

        let c4 = PianoNote.from(frequency: 261.625565)
        XCTAssertEqual(c4?.midiNumber, 60)
        XCTAssertEqual(c4?.displayName, "C4")
    }

    func testFrequencyConversionRejectsInvalidAndOutOfRangeValues() {
        XCTAssertNil(PianoNote.from(frequency: 0))
        XCTAssertNil(PianoNote.from(frequency: -440))
        XCTAssertNil(PianoNote.from(frequency: .nan))
        XCTAssertNil(PianoNote.from(frequency: 20))
        XCTAssertNil(PianoNote.from(frequency: 5000))
    }

    func testChordNetNoteNameParsingCoversPianoRangeAndSharps() {
        XCTAssertEqual(PianoNote.midiNumber(from: "A0"), 21)
        XCTAssertEqual(PianoNote.midiNumber(from: "C4"), 60)
        XCTAssertEqual(PianoNote.midiNumber(from: "C#4"), 61)
        XCTAssertEqual(PianoNote.midiNumber(from: "A4"), 69)
        XCTAssertEqual(PianoNote.midiNumber(from: "C8"), 108)

        let note = PianoNote(chordNetName: "F#5")
        XCTAssertEqual(note?.midiNumber, 78)
        XCTAssertEqual(note?.displayName, "F#5")
        XCTAssertTrue(note?.isBlackKey == true)
    }

    func testChordNetNoteNameParsingRejectsUnsupportedNames() {
        XCTAssertNil(PianoNote.midiNumber(from: "G#8"))
        XCTAssertNil(PianoNote.midiNumber(from: "G#-1"))
        XCTAssertNil(PianoNote.midiNumber(from: "Bb4"))
        XCTAssertNil(PianoNote.midiNumber(from: "c4"))
        XCTAssertNil(PianoNote.midiNumber(from: ""))
    }
}
