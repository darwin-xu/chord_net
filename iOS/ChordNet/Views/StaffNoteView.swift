import SwiftUI

// Grand staff reference using diatonic steps relative to middle C (C4 = step 0).
// Treble lines:  E4 G4 B4 D5 F5  -> steps 2, 4, 6, 8, 10
// Bass lines:    G2 B2 D3 F3 A3  -> steps -10, -8, -6, -4, -2
// Middle C sits between the staves on ledger line step 0.

// MARK: - NoteQueue

/// Public interface for injecting notes into StaffNoteView.
/// Call `enqueue(_:)` to add a note; StaffNoteView observes this object.
@MainActor
final class NoteQueue: ObservableObject {
    struct QueuedNote: Identifiable {
        let id = UUID()
        let note: PianoNote
    }

    @Published private(set) var notes: [QueuedNote] = []

    func enqueue(_ note: PianoNote) {
        notes.append(QueuedNote(note: note))
    }

    fileprivate func trim(keepingAtMost maxCount: Int) {
        guard notes.count > maxCount else { return }
        notes.removeFirst(notes.count - maxCount)
    }
}

// Chromatic pitch class (0-11) to diatonic step within one octave.
// Accidentals share the same diatonic position as their natural note.
private let pitchClassStep: [Int] = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6]

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

    private func noteGlyph(for step: Int) -> String {
        let scalar: UInt32
        if step >= 0 {
            // Treble staff: middle line is step 6 (B4).
            scalar = step >= 6 ? 0xE1D6 : 0xE1D5
        } else {
            // Bass staff: middle line is step -6 (D3).
            scalar = step >= -6 ? 0xE1D6 : 0xE1D5
        }
        return String(UnicodeScalar(scalar)!)
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
            let noteGlyphSize = lineGap * 3.2
            
            let noteStartX = max(clefLeftX + trebleClefWidth, clefLeftX + bassClefWidth) + 10
            let noteEndX = width - margin
            let noteSpacing = lineGap * 2
            let noteCount = queue.notes.count
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

                ForEach(Array(queue.notes.enumerated()), id: \.element.id) { (index, entry) in
                    let noteX = newestNoteX - CGFloat(noteCount - 1 - index) * noteSpacing
                    let step = diatonicStep(for: entry.note.midiNumber)
                    let noteY = yFor(step: step, top: topPad, stepSize: stepSize)
                    let glyph = noteGlyph(for: step)
                    let ledgers = ledgerSteps(for: step)

                    let noteHeadHalfWidth = max(noteGlyphSize * 0.25, 1)
                    let ledgerHalfWidth: CGFloat = ledgers.isEmpty ? noteHeadHalfWidth : max(noteHeadHalfWidth, 18)
                    let accidentalOffsetX: CGFloat = entry.note.isBlackKey ? lineGap * 1.1 : 0
                    let accidentalHalfWidth: CGFloat = entry.note.isBlackKey ? lineGap * 0.35 : 0

                    let leftEdge = noteX - ledgerHalfWidth - accidentalOffsetX - accidentalHalfWidth

                    if leftEdge >= noteStartX {
                        ForEach(ledgers, id: \.self) { ledgerStep in
                            Rectangle()
                                .fill(Color.black.opacity(0.60))
                                .frame(width: 36, height: 1.4)
                                .position(x: noteX, y: yFor(step: ledgerStep, top: topPad, stepSize: stepSize))
                        }

                        Text(glyph)
                            .font(.custom("Bravura", size: noteGlyphSize))
                            .foregroundStyle(Color(red: 0.13, green: 0.18, blue: 0.27))
                            .position(x: noteX, y: noteY)

                        if entry.note.isBlackKey {
                            Text("♯")
                                .font(.system(size: lineGap * 1.2, weight: .semibold, design: .serif))
                                .foregroundStyle(Color(red: 0.13, green: 0.18, blue: 0.27))
                                .position(x: noteX - accidentalOffsetX, y: noteY - 2)
                        }
                    }
                }
            }
            .animation(.easeInOut(duration: 0.3), value: noteCount)
            .onChange(of: noteCount) { _, newCount in
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
    queue.enqueue(PianoNote(midiNumber: 60, frequency: 261.63, centsOffset: 0))
    queue.enqueue(PianoNote(midiNumber: 64, frequency: 329.63, centsOffset: 0))
    queue.enqueue(PianoNote(midiNumber: 67, frequency: 392.00, centsOffset: 0))
    return StaffNoteView(queue: queue)
        .frame(height: 320)
        .padding()
        .background(Color(.systemGroupedBackground))
}
