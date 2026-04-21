import SwiftUI

struct ContentView: View {
    @StateObject private var engine = AudioEngine()
    @StateObject private var staffNoteQueue = NoteQueue()

    private let waveformStyle: WaveformEnvelopeView.Style = .neonThreads
    private let staffHeightRatio: CGFloat = 0.35

    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                statusBar

                WaveformEnvelopeView(
                    envelope: waveformEnvelope,
                    isListening: engine.isRunning,
                    headline: waveformHeadline,
                    style: waveformStyle
                )

                StaffNoteView(queue: staffNoteQueue)
                    .frame(height: geometry.size.height * staffHeightRatio)

                PianoKeyboardView(activeMIDINotes: activeMIDINotes)
                    .frame(height: geometry.size.height * 0.24)

                controlsPanel
            }
            .frame(width: geometry.size.width, height: geometry.size.height, alignment: .top)
        }
        .background(
            LinearGradient(
                colors: [
                    Color(red: 0.95, green: 0.96, blue: 0.98),
                    Color(red: 0.89, green: 0.91, blue: 0.95),
                ],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()
        )
        .onChange(of: engine.detectedNotes) { _, newNotes in
            for note in newNotes.compactMap(PianoNote.init(chordNetName:)) {
                staffNoteQueue.enqueue(note)
            }
        }
    }

    private var statusBar: some View {
        VStack(spacing: 0) {
            HStack {
                Spacer()

                Text(statusTitle)
                    .font(.system(size: 20, weight: .semibold, design: .rounded))
                    .foregroundStyle(statusColor)

                Spacer()
            }
        }
        .frame(height: 64)
        .background(Color.white.opacity(0.92))
    }

    private var controlsPanel: some View {
        VStack(spacing: 10) {
            Spacer(minLength: 4)

            Button {
                if engine.isRunning {
                    engine.stop()
                } else {
                    engine.start()
                }
            } label: {
                circularControl(
                    colors: engine.isRunning
                        ? [
                            Color(red: 0.93, green: 0.14, blue: 0.20),
                            Color(red: 0.62, green: 0.04, blue: 0.08),
                        ]
                        : [
                            Color(red: 0.25, green: 0.60, blue: 0.97),
                            Color(red: 0.04, green: 0.24, blue: 0.74),
                        ],
                    shadowColor: engine.isRunning ? Color.red.opacity(0.18) : Color.blue.opacity(0.18),
                    systemImage: engine.isRunning ? "stop.fill" : "mic.fill"
                )
            }
            .buttonStyle(.plain)
            .frame(width: 92, height: 92)
            .accessibilityLabel(engine.isRunning ? "Stop listening" : "Start listening")

            Text(controlCaption)
                .font(.system(size: 12, weight: .medium, design: .rounded))
                .foregroundStyle(Color.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)

            Spacer(minLength: 18)
        }
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .fill(Color.white.opacity(0.55))
                .shadow(color: Color.black.opacity(0.08), radius: 14, y: -4)
                .ignoresSafeArea(edges: .bottom)
        )
    }

    private func circularControl(colors: [Color], shadowColor: Color, systemImage: String) -> some View {
        ZStack {
            Circle()
                .fill(
                    LinearGradient(
                        colors: colors,
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .overlay(
                    Circle()
                        .stroke(Color.white.opacity(0.35), lineWidth: 2)
                        .padding(4)
                )
                .shadow(color: shadowColor, radius: 10, y: 6)

            Image(systemName: systemImage)
                .font(.system(size: 40, weight: .regular))
                .foregroundStyle(.white)
        }
    }

    private var activeMIDINotes: Set<Int> {
        Set(engine.detectedNotes.compactMap { PianoNote(chordNetName: $0)?.midiNumber })
    }

    private var waveformEnvelope: [CGFloat] {
        let values = engine.waveformLevels.map { CGFloat($0) }
        return values.isEmpty ? Array(repeating: 0, count: 32) : values
    }

    private var waveformHeadline: String {
        guard !engine.detectedNotes.isEmpty else { return engine.statusMessage }
        return engine.detectedNotes.joined(separator: " · ")
    }

    private var statusTitle: String {
        engine.isRunning ? "Listening..." : "Ready"
    }

    private var statusColor: Color {
        engine.isRunning ? Color(red: 0.24, green: 0.42, blue: 0.67) : Color.secondary
    }

    private var controlCaption: String {
        if engine.isRunning {
            return "ChordNet is analyzing the microphone input"
        }
        return engine.statusMessage
    }
}

private extension PianoNote {
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

#Preview {
    ContentView()
}
