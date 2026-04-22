import SwiftUI

// Grand staff reference using diatonic steps relative to middle C (C4 = step 0).
// Treble lines:  E4 G4 B4 D5 F5  -> steps 2, 4, 6, 8, 10
// Bass lines:    G2 B2 D3 F3 A3  -> steps -10, -8, -6, -4, -2
// Middle C sits between the staves on ledger line step 0.

// MARK: - NoteQueue

/// Public interface for injecting notes into StaffNoteView.
/// Call `enqueue(_:)` to add a note or `enqueueChord(_:)` for simultaneous notes.
@MainActor
final class NoteQueue: ObservableObject {
    struct QueuedEvent: Identifiable {
        let id = UUID()
        let notes: [PianoNote]
    }

    @Published private(set) var events: [QueuedEvent] = []

    func enqueue(_ note: PianoNote) {
        events.append(QueuedEvent(notes: [note]))
    }

    func enqueueChord(_ notes: [PianoNote]) {
        let uniqueNotes = Dictionary(grouping: notes, by: \.midiNumber)
            .compactMap { $0.value.first }
            .sorted { $0.midiNumber < $1.midiNumber }

        guard !uniqueNotes.isEmpty else { return }
        events.append(QueuedEvent(notes: uniqueNotes))
    }

    fileprivate func trim(keepingAtMost maxCount: Int) {
        guard events.count > maxCount else { return }
        events.removeFirst(events.count - maxCount)
    }
}

// Chromatic pitch class (0-11) to diatonic step within one octave.
// Accidentals share the same diatonic position as their natural note.
private let pitchClassStep: [Int] = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6]

private struct StaffChordNote: Identifiable {
    let id: Int
    let note: PianoNote
    let step: Int
    let isDisplaced: Bool
}

private func diatonicStep(for midiNumber: Int) -> Int {
    let delta = midiNumber - 60
    let octaves = delta >= 0 ? delta / 12 : (delta - 11) / 12
    let pitchClass = ((delta % 12) + 12) % 12
    return octaves * 7 + pitchClassStep[pitchClass]
}

struct StaffNoteView: View {
    @ObservedObject var queue: NoteQueue

    var body: some View {
        StaffCanvas(queue: queue)
    }
}

private struct StaffCanvas: View {
    @ObservedObject var queue: NoteQueue

    private let trebleLineSteps = [2, 4, 6, 8, 10]
    private let bassLineSteps = [-10, -8, -6, -4, -2]

    private func yFor(step: Int, top: CGFloat, stepSize: CGFloat) -> CGFloat {
        top + CGFloat(12 - step) * stepSize
    }

    private func ledgerSteps(for step: Int) -> [Int] {
        if step == 0 {
            return [0]
        }

        if step > 10 {
            let highestLine = step.isMultiple(of: 2) ? step : step - 1
            return Array(stride(from: 12, through: highestLine, by: 2))
        }

        if step < -10 {
            let lowestLine = step.isMultiple(of: 2) ? step : step + 1
            return Array(stride(from: -12, through: lowestLine, by: -2))
        }

        return []
    }

    private func chordNotes(for notes: [PianoNote]) -> [StaffChordNote] {
        let sorted = notes
            .map { note in (note: note, step: diatonicStep(for: note.midiNumber)) }
            .sorted { lhs, rhs in
                if lhs.step == rhs.step {
                    return lhs.note.midiNumber < rhs.note.midiNumber
                }
                return lhs.step < rhs.step
            }

        var displacedSteps = Set<Int>()
        var cluster: [(note: PianoNote, step: Int)] = []

        func markCluster(_ cluster: [(note: PianoNote, step: Int)]) {
            guard cluster.count > 1 else { return }
            for (index, item) in cluster.enumerated() where index.isMultiple(of: 2) == false {
                displacedSteps.insert(item.step)
            }
        }

        for item in sorted {
            if let previous = cluster.last, item.step - previous.step <= 1 {
                cluster.append(item)
            } else {
                markCluster(cluster)
                cluster = [item]
            }
        }
        markCluster(cluster)

        return sorted.map { item in
            StaffChordNote(
                id: item.note.midiNumber,
                note: item.note,
                step: item.step,
                isDisplaced: displacedSteps.contains(item.step)
            )
        }
    }

    private func stemGoesUp(for notes: [StaffChordNote]) -> Bool {
        let averageStep = notes.reduce(0) { $0 + $1.step } / max(notes.count, 1)
        return averageStep < 2
    }

    var body: some View {
        GeometryReader { geo in
            let width = geo.size.width
            let height = geo.size.height
            let topPad: CGFloat = 34
            let bottomPad: CGFloat = 34
            let usableHeight = max(height - topPad - bottomPad, 120)
            let stepSize = usableHeight / 24
            let lineGap = stepSize * 2
            let smuflBrace = String(UnicodeScalar(0xE000)!)
            let smuflTrebleClef = String(UnicodeScalar(0xE050)!)
            let smuflBassClef = String(UnicodeScalar(0xE062)!)
            
            // Left and right margin of the grand sheet
            let margin: CGFloat = 10

            let trebleTopY = yFor(step: 10, top: topPad, stepSize: stepSize)
            let bassBottomY = yFor(step: -10, top: topPad, stepSize: stepSize)
            
            let braceCenterY = (trebleTopY + bassBottomY) / 2
            let braceSpan = bassBottomY - trebleTopY
            let braceBaseSize = lineGap * 10
            let braceVerticalScale = max(braceSpan / (braceBaseSize * 1.45), 1)
            let braceFont = UIFont(name: "Bravura", size: braceBaseSize) ?? .systemFont(ofSize: braceBaseSize)
            let braceWidth = glyphMetrics(for: smuflBrace, font: braceFont).bounds.width
            let braceCenterX = margin + braceWidth / 2
            
            // X position for the sheet bar
            let barLineX = margin + braceWidth
            
            let clefLeftX: CGFloat = barLineX + 10
            let trebleClefSize = lineGap * 4
            let trebleClefFont = UIFont(name: "Bravura", size: trebleClefSize) ?? .systemFont(ofSize: trebleClefSize)
            let trebleClefWidth = glyphMetrics(for: smuflTrebleClef, font: trebleClefFont).bounds.width
            let trebleClefCenterX = clefLeftX + trebleClefWidth / 2

            let bassClefSize = lineGap * 4
            let bassClefFont = UIFont(name: "Bravura", size: bassClefSize) ?? .systemFont(ofSize: bassClefSize)
            let bassClefWidth = glyphMetrics(for: smuflBassClef, font: bassClefFont).bounds.width
            let bassClefCenterX = clefLeftX + bassClefWidth / 2
            let noteHeadWidth = lineGap * 1.08
            let noteHeadHeight = lineGap * 0.74
            let stemHeight = lineGap * 3.4
            let stemWidth: CGFloat = 1.5
            
            let noteStartX = max(clefLeftX + trebleClefWidth, clefLeftX + bassClefWidth) + 10
            let noteEndX = width - margin
            let noteSpacing = lineGap * 2
            let eventCount = queue.events.count
            let maxVisible = max(Int((noteEndX - noteStartX) / noteSpacing) + 2, 1)
            // Anchor the newest note one half-spacing from the right edge so it is
            // fully visible as soon as it enters, rather than being half-clipped.
            let newestNoteX = noteEndX - noteSpacing * 0.5

            let staffStartX = barLineX
            ZStack {
                Color.white

                Text(smuflBrace)
                    .font(.custom("Bravura", size: braceBaseSize))
                    .foregroundStyle(Color.black.opacity(0.88))
                    .scaleEffect(x: 1.0, y: braceVerticalScale, anchor: .center)
                    .position(x: braceCenterX, y: bassBottomY)

                Rectangle()
                    .fill(Color.black.opacity(0.82))
                    .frame(width: 2.2, height: braceSpan)
                    .position(x: barLineX, y: braceCenterY)

                ForEach(trebleLineSteps, id: \.self) { step in
                    staffLine(
                        at: yFor(step: step, top: topPad, stepSize: stepSize),
                        startX: staffStartX,
                        width: width,
                        rightMargin: margin
                    )
                }

                ForEach(bassLineSteps, id: \.self) { step in
                    staffLine(
                        at: yFor(step: step, top: topPad, stepSize: stepSize),
                        startX: staffStartX,
                        width: width,
                        rightMargin: margin
                    )
                }

                Text(smuflTrebleClef)
                    .font(.custom("Bravura", size: trebleClefSize))
                    .foregroundStyle(Color.black.opacity(0.92))
                    .position(
                        x: trebleClefCenterX,
                        y: yFor(step: 4, top: topPad, stepSize: stepSize) - lineGap * 0
                    )

                Text(smuflBassClef)
                    .font(.custom("Bravura", size: bassClefSize))
                    .foregroundStyle(Color.black.opacity(0.92))
                    .position(
                        x: bassClefCenterX,
                        y: yFor(step: -4, top: topPad, stepSize: stepSize) - lineGap * 0
                    )

                ForEach(Array(queue.events.enumerated()), id: \.element.id) { (index, event) in
                    let noteX = newestNoteX - CGFloat(eventCount - 1 - index) * noteSpacing
                    let chord = chordNotes(for: event.notes)
                    let stemUp = stemGoesUp(for: chord)
                    let ledgerLineSteps = Array(Set(chord.flatMap { ledgerSteps(for: $0.step) })).sorted()
                    let hasDisplacedNote = chord.contains { $0.isDisplaced }
                    let blackKeyNotes = chord.filter { $0.note.isBlackKey }
                    let ledgerWidth = noteHeadWidth * (hasDisplacedNote ? 3.0 : 2.15)
                    let accidentalWidth = blackKeyNotes.isEmpty ? 0 : lineGap * (1.2 + CGFloat(blackKeyNotes.count - 1) * 0.32)
                    let leftEdge = noteX - ledgerWidth / 2 - accidentalWidth - lineGap * 0.35

                    if leftEdge >= noteStartX {
                        ForEach(ledgerLineSteps, id: \.self) { ledgerStep in
                            Rectangle()
                                .fill(Color.black.opacity(0.60))
                                .frame(width: ledgerWidth, height: 1.4)
                                .position(x: noteX, y: yFor(step: ledgerStep, top: topPad, stepSize: stepSize))
                        }

                        if let lowest = chord.first, let highest = chord.last {
                            let stemAnchorStep = stemUp ? lowest.step : highest.step
                            let stemStartY = yFor(step: stemAnchorStep, top: topPad, stepSize: stepSize)
                            let stemEndY = stemStartY + (stemUp ? -stemHeight : stemHeight)
                            let stemX = noteX + (stemUp ? noteHeadWidth * 0.45 : -noteHeadWidth * 0.45)

                            Rectangle()
                                .fill(Color(red: 0.13, green: 0.18, blue: 0.27))
                                .frame(width: stemWidth, height: abs(stemEndY - stemStartY))
                                .position(x: stemX, y: (stemStartY + stemEndY) / 2)
                        }

                        ForEach(chord) { chordNote in
                            let offsetDirection: CGFloat = stemUp ? 1 : -1
                            let displacedX = chordNote.isDisplaced ? noteHeadWidth * 0.82 * offsetDirection : 0
                            let noteY = yFor(step: chordNote.step, top: topPad, stepSize: stepSize)

                            Ellipse()
                                .fill(Color(red: 0.13, green: 0.18, blue: 0.27))
                                .frame(width: noteHeadWidth, height: noteHeadHeight)
                                .rotationEffect(.degrees(-20))
                                .position(x: noteX + displacedX, y: noteY)
                        }

                        ForEach(Array(blackKeyNotes.enumerated()), id: \.element.id) { accidentalIndex, chordNote in
                            Text("♯")
                                .font(.system(size: lineGap * 1.2, weight: .semibold, design: .serif))
                                .foregroundStyle(Color(red: 0.13, green: 0.18, blue: 0.27))
                                .position(
                                    x: noteX - lineGap * (1.15 + CGFloat(accidentalIndex) * 0.34),
                                    y: yFor(step: chordNote.step, top: topPad, stepSize: stepSize) - 2
                                )
                        }
                    }
                }
            }
            .animation(.easeInOut(duration: 0.3), value: eventCount)
            .onChange(of: eventCount) { _, newCount in
                if newCount > maxVisible {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                        queue.trim(keepingAtMost: maxVisible)
                    }
                }
            }
        }
    }

    private func staffLine(at y: CGFloat, startX: CGFloat, width: CGFloat, rightMargin: CGFloat) -> some View {
        Rectangle()
            .fill(Color.black.opacity(0.72))
            .frame(height: 1)
            .frame(width: width - startX - rightMargin)
            .position(x: startX + (width - startX - rightMargin) / 2, y: y)
    }
}

private func glyphMetrics(for text: String, font: UIFont) -> (bounds: CGRect, lineSize: CGSize, offset: CGSize) {
    guard !text.isEmpty else {
        return (.zero, .zero, .zero)
    }

    let attributes: [NSAttributedString.Key: Any] = [
        .font: font,
    ]
    let attributedString = NSAttributedString(string: text, attributes: attributes)
    let line = CTLineCreateWithAttributedString(attributedString)

    var ascent: CGFloat = 0
    var descent: CGFloat = 0
    var leading: CGFloat = 0
    let lineWidth = CGFloat(CTLineGetTypographicBounds(line, &ascent, &descent, &leading))
    let lineHeight = ascent + descent
    let glyphBounds = CTLineGetBoundsWithOptions(line, [.useGlyphPathBounds, .excludeTypographicLeading]).standardized.integral

    let lineCenterX = lineWidth * 0.5
    let lineCenterY = (ascent - descent) * 0.5
    let glyphCenterX = glyphBounds.midX
    let glyphCenterY = glyphBounds.midY

    return (
        glyphBounds,
        CGSize(width: lineWidth, height: lineHeight),
        CGSize(
            width: glyphCenterX - lineCenterX,
            height: -(glyphCenterY - lineCenterY)
        )
    )
}

#Preview {
    let queue = NoteQueue()
    queue.enqueueChord([
        PianoNote(midiNumber: 60, frequency: 261.63, centsOffset: 0),
        PianoNote(midiNumber: 64, frequency: 329.63, centsOffset: 0),
        PianoNote(midiNumber: 67, frequency: 392.00, centsOffset: 0),
    ])
    queue.enqueueChord([
        PianoNote(midiNumber: 61, frequency: 277.18, centsOffset: 0),
        PianoNote(midiNumber: 62, frequency: 293.66, centsOffset: 0),
        PianoNote(midiNumber: 65, frequency: 349.23, centsOffset: 0),
    ])
    return StaffNoteView(queue: queue)
        .frame(height: 320)
        .padding()
        .background(Color(.systemGroupedBackground))
}
