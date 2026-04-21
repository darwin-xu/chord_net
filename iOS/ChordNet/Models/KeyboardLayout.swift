import Foundation

struct KeyboardLayout {
    static let noteRange: [PianoNote] = (PianoNote.minimumMIDINote...PianoNote.maximumMIDINote).compactMap { midiNumber in
        let frequency = 440.0 * pow(2.0, Double(midiNumber - 69) / 12.0)
        return PianoNote(midiNumber: midiNumber, frequency: frequency, centsOffset: 0)
    }

    static let visibleRange = 60...71

    static var visibleNotes: [PianoNote] {
        noteRange.filter { visibleRange.contains($0.midiNumber) }
    }

    static var visibleWhiteKeys: [PianoNote] {
        visibleNotes.filter { !$0.isBlackKey }
    }

    static var visibleBlackKeys: [PianoNote] {
        visibleNotes.filter(\.isBlackKey)
    }

    static var whiteKeys: [PianoNote] {
        noteRange.filter { !$0.isBlackKey }
    }

    static var blackKeys: [PianoNote] {
        noteRange.filter(\.isBlackKey)
    }

    static func whiteKeyOffset(for midiNumber: Int) -> Int {
        noteRange.filter { !$0.isBlackKey && $0.midiNumber < midiNumber }.count
    }

    static func visibleWhiteKeyOffset(for midiNumber: Int) -> Int {
        visibleNotes.filter { !$0.isBlackKey && $0.midiNumber < midiNumber }.count
    }
}