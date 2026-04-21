import Foundation

struct PianoNote: Equatable, Identifiable {
    static let minimumMIDINote = 21
    static let maximumMIDINote = 108
    private static let pitchClasses = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    let midiNumber: Int
    let frequency: Double
    let centsOffset: Double

    var id: Int { midiNumber }

    var pitchClass: String {
        Self.pitchClasses[(midiNumber % 12 + 12) % 12]
    }

    var octave: Int {
        (midiNumber / 12) - 1
    }

    var displayName: String {
        "\(pitchClass)\(octave)"
    }

    var pianoKeyIndex: Int {
        midiNumber - Self.minimumMIDINote
    }

    var isBlackKey: Bool {
        [1, 3, 6, 8, 10].contains(midiNumber % 12)
    }

    static func from(frequency: Double) -> PianoNote? {
        guard frequency.isFinite, frequency > 0 else {
            return nil
        }

        let noteNumber = 69 + 12 * log2(frequency / 440.0)
        let roundedMIDI = Int(noteNumber.rounded())
        guard (minimumMIDINote...maximumMIDINote).contains(roundedMIDI) else {
            return nil
        }

        let equalTemperedFrequency = 440.0 * pow(2.0, Double(roundedMIDI - 69) / 12.0)
        let centsOffset = 1200.0 * log2(frequency / equalTemperedFrequency)
        return PianoNote(midiNumber: roundedMIDI, frequency: frequency, centsOffset: centsOffset)
    }
}

struct DetectionResult: Equatable {
    let note: PianoNote
    let confidence: Double
    let amplitude: Float
}