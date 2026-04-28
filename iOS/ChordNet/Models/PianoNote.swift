import Foundation

struct PianoNote: Equatable, Identifiable {
    static let minimumMIDINote = 21
    static let maximumMIDINote = 108
    private static let pitchClasses = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    let midiNumber: Int
    let frequency: Double
    let centsOffset: Double

    init(midiNumber: Int, frequency: Double, centsOffset: Double) {
        self.midiNumber = midiNumber
        self.frequency = frequency
        self.centsOffset = centsOffset
    }

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

    init?(chordNetName: String) {
        guard let midi = Self.midiNumber(from: chordNetName) else { return nil }
        let frequency = 440.0 * pow(2.0, Double(midi - 69) / 12.0)
        self.init(midiNumber: midi, frequency: frequency, centsOffset: 0)
    }

    static func midiNumber(from noteName: String) -> Int? {
        let semitones = [
            "C": 0,
            "C#": 1,
            "D": 2,
            "D#": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "G": 7,
            "G#": 8,
            "A": 9,
            "A#": 10,
            "B": 11,
        ]

        let pitchClass: String
        let octaveText: Substring

        if noteName.count >= 2, noteName.dropFirst().first == "#" {
            pitchClass = String(noteName.prefix(2))
            octaveText = noteName.dropFirst(2)
        } else {
            pitchClass = String(noteName.prefix(1))
            octaveText = noteName.dropFirst()
        }

        guard let semitone = semitones[pitchClass],
              let octave = Int(octaveText)
        else {
            return nil
        }

        let midi = (octave + 1) * 12 + semitone
        guard (minimumMIDINote...maximumMIDINote).contains(midi) else {
            return nil
        }
        return midi
    }
}

struct DetectionResult: Equatable {
    let note: PianoNote
    let confidence: Double
    let amplitude: Float
}
