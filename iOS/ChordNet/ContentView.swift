import SwiftUI
#if STAFF_INPUT_MODE
import UIKit
#endif

struct ContentView: View {
    @StateObject private var engine = AudioEngine()
    @StateObject private var staffNoteQueue = NoteQueue()
#if STAFF_INPUT_MODE
    @State private var staffInput = StaffKeyboardInputState()
#endif

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

#if STAFF_INPUT_MODE
                staffInputPanel
#endif

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
            staffNoteQueue.enqueueChord(newNotes.compactMap(PianoNote.init(chordNetName:)))
        }
#if STAFF_INPUT_MODE
        .background(
            StaffKeyboardInputCapture { key in
                handleStaffInput(key)
            }
        )
#endif
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
        var notes = Set(engine.detectedNotes.compactMap { PianoNote(chordNetName: $0)?.midiNumber })
#if STAFF_INPUT_MODE
        notes.formUnion(staffInput.pendingNotes.map(\.midiNumber))
#endif
        return notes
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

#if STAFF_INPUT_MODE
    private var staffInputPanel: some View {
        HStack(spacing: 12) {
            Text("Staff Input")
                .font(.system(size: 12, weight: .semibold, design: .rounded))
                .foregroundStyle(Color(red: 0.24, green: 0.42, blue: 0.67))

            Text(staffInput.displayText)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundStyle(Color(red: 0.13, green: 0.18, blue: 0.27))
                .lineLimit(1)
                .minimumScaleFactor(0.75)

            Spacer(minLength: 8)

            ForEach([3, 4, 5, 6], id: \.self) { octave in
                Button {
                    staffInput.selectOctave(octave)
                } label: {
                    Text("\(octave)")
                        .font(.system(size: 12, weight: .semibold, design: .rounded))
                        .frame(width: 18)
                }
                .buttonStyle(.bordered)
                .tint(staffInput.selectedOctave == octave ? Color(red: 0.24, green: 0.42, blue: 0.67) : Color.secondary)
                .accessibilityLabel("Select octave \(octave)")
            }

            Button {
                commitStaffInput()
            } label: {
                Image(systemName: "music.note.list")
                    .font(.system(size: 16, weight: .semibold))
            }
            .buttonStyle(.bordered)
            .disabled(staffInput.pendingNotes.isEmpty)
            .accessibilityLabel("Draw keyboard chord")

            Button {
                staffInput.clear()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 16, weight: .semibold))
            }
            .buttonStyle(.bordered)
            .accessibilityLabel("Clear keyboard chord")
        }
        .padding(.horizontal, 14)
        .frame(height: 44)
        .background(Color.white.opacity(0.86))
    }

    private func handleStaffInput(_ key: StaffInputKey) {
        switch key {
        case .note(let letter):
            staffInput.append(letter)
        case .octave(let octave):
            staffInput.selectOctave(octave)
        case .commit:
            commitStaffInput()
        case .delete:
            staffInput.removeLast()
        case .clear:
            staffInput.clear()
        }
    }

    private func commitStaffInput() {
        staffNoteQueue.enqueueChord(staffInput.pendingNotes)
        staffInput.clear()
    }
#endif
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

#if STAFF_INPUT_MODE
private enum StaffInputKey {
    case note(Character)
    case octave(Int)
    case commit
    case delete
    case clear
}

private struct StaffKeyboardInputState {
    private(set) var pendingNotes: [PianoNote] = []
    private(set) var selectedOctave = 4

    var displayText: String {
        let notes = pendingNotes.map(\.displayName).joined(separator: " ")
        let guide = "Oct \(selectedOctave) | c d e f g a b | C D F G A = sharp | Space draws"
        return notes.isEmpty ? guide : "\(guide) | \(notes)"
    }

    mutating func append(_ letter: Character) {
        guard let midiNumber = Self.midiNumber(for: letter, octave: selectedOctave) else { return }
        let frequency = 440.0 * pow(2.0, Double(midiNumber - 69) / 12.0)
        let note = PianoNote(midiNumber: midiNumber, frequency: frequency, centsOffset: 0)

        if !pendingNotes.contains(where: { $0.midiNumber == note.midiNumber }) {
            pendingNotes.append(note)
        }
    }

    mutating func selectOctave(_ octave: Int) {
        selectedOctave = octave
    }

    mutating func removeLast() {
        guard !pendingNotes.isEmpty else { return }
        pendingNotes.removeLast()
    }

    mutating func clear() {
        pendingNotes.removeAll()
    }

    private static func midiNumber(for letter: Character, octave: Int) -> Int? {
        let semitone: Int
        switch String(letter) {
        case "c": semitone = 0
        case "C": semitone = 1
        case "d": semitone = 2
        case "D": semitone = 3
        case "e": semitone = 4
        case "f": semitone = 5
        case "F": semitone = 6
        case "g": semitone = 7
        case "G": semitone = 8
        case "a": semitone = 9
        case "A": semitone = 10
        case "b": semitone = 11
        default: return nil
        }

        return (octave + 1) * 12 + semitone
    }
}

private struct StaffKeyboardInputCapture: UIViewRepresentable {
    let onInput: (StaffInputKey) -> Void

    func makeUIView(context: Context) -> KeyboardInputView {
        let view = KeyboardInputView()
        view.onInput = onInput
        return view
    }

    func updateUIView(_ uiView: KeyboardInputView, context: Context) {
        uiView.onInput = onInput
        DispatchQueue.main.async {
            uiView.window?.makeKey()
            uiView.becomeFirstResponder()
        }
    }

    final class KeyboardInputView: UIView {
        var onInput: ((StaffInputKey) -> Void)?

        override var canBecomeFirstResponder: Bool { true }

        override func didMoveToWindow() {
            super.didMoveToWindow()
            becomeFirstResponder()
        }

        override func pressesBegan(_ presses: Set<UIPress>, with event: UIPressesEvent?) {
            var handled = false
            for press in presses {
                guard let characters = press.key?.characters else { continue }
                for character in characters {
                    if let key = inputKey(for: character, keyCode: press.key?.keyCode) {
                        onInput?(key)
                        handled = true
                    }
                }
            }

            if !handled {
                super.pressesBegan(presses, with: event)
            }
        }

        private func inputKey(for character: Character, keyCode: UIKeyboardHIDUsage?) -> StaffInputKey? {
            switch character {
            case "c", "C", "d", "D", "e", "f", "F", "g", "G", "a", "A", "b":
                return .note(character)
            case "3", "4", "5", "6":
                return .octave(Int(String(character)) ?? 4)
            case " ", "\r", "\n":
                return .commit
            case "\u{8}", "\u{7f}":
                return .delete
            case "\u{1b}":
                return .clear
            default:
                if keyCode == .keyboardDeleteOrBackspace {
                    return .delete
                }
                if keyCode == .keyboardReturnOrEnter || keyCode == .keypadEnter {
                    return .commit
                }
                if keyCode == .keyboardEscape {
                    return .clear
                }
                return nil
            }
        }
    }
}
#endif
