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

private struct EventLayoutMetrics {
    let leftExtent: CGFloat
    let rightExtent: CGFloat
    let totalWidth: CGFloat
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
    @State private var renderedEvents: [NoteQueue.QueuedEvent] = []
    @State private var slideOffset: CGFloat = 0

    private let trebleLineSteps = [2, 4, 6, 8, 10]
    private let bassLineSteps = [-10, -8, -6, -4, -2]
    private let slideDuration = 0.3

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
        guard !notes.isEmpty else { return true }

        let averageStep = notes.reduce(0) { $0 + $1.step } / notes.count
        let staffMiddleLineStep = averageStep >= 0 ? 6 : -6
        return averageStep < staffMiddleLineStep
    }

    private func accidentalWidth(
        for accidentalCount: Int,
        symbolWidth: CGFloat,
        symbolSpacing: CGFloat
    ) -> CGFloat {
        guard accidentalCount > 0 else { return 0 }
        return CGFloat(accidentalCount) * symbolWidth + CGFloat(accidentalCount - 1) * symbolSpacing
    }

    private func accidentalReservedWidth(
        for event: NoteQueue.QueuedEvent,
        symbolWidth: CGFloat,
        symbolSpacing: CGFloat,
        noteGap: CGFloat
    ) -> CGFloat {
        let accidentalCount = event.notes.filter(\.isBlackKey).count
        guard accidentalCount > 0 else { return 0 }
        return accidentalWidth(for: accidentalCount, symbolWidth: symbolWidth, symbolSpacing: symbolSpacing) + noteGap
    }

    private func horizontalSpacing(
        for metrics: EventLayoutMetrics,
        baseSpacing: CGFloat
    ) -> CGFloat {
        max(baseSpacing, metrics.totalWidth)
    }

    private func eventLayoutMetrics(
        for event: NoteQueue.QueuedEvent,
        baseSpacing: CGFloat,
        noteHeadWidth: CGFloat,
        lineGap: CGFloat,
        accidentalSymbolWidth: CGFloat,
        accidentalSymbolSpacing: CGFloat,
        accidentalNoteGap: CGFloat
    ) -> EventLayoutMetrics {
        let chord = chordNotes(for: event.notes)
        let stemUp = stemGoesUp(for: chord)
        let hasDisplacedNote = chord.contains { $0.isDisplaced }
        let accidentalCount = chord.filter { $0.note.isBlackKey }.count
        let accidentalSpan = accidentalWidth(
            for: accidentalCount,
            symbolWidth: accidentalSymbolWidth,
            symbolSpacing: accidentalSymbolSpacing
        )
        let accidentalReservedWidth = accidentalCount == 0 ? 0 : accidentalSpan + accidentalNoteGap
        let ledgerWidth = noteHeadWidth * (hasDisplacedNote ? 3.0 : 2.15)
        let offsetDirection: CGFloat = stemUp ? 1 : -1

        var minDrawingX = -ledgerWidth / 2
        var maxDrawingX = ledgerWidth / 2

        for chordNote in chord {
            let displacedX = chordNote.isDisplaced ? noteHeadWidth * 0.82 * offsetDirection : 0
            minDrawingX = min(minDrawingX, displacedX - noteHeadWidth / 2)
            maxDrawingX = max(maxDrawingX, displacedX + noteHeadWidth / 2)
        }

        let leftExtent = -minDrawingX + accidentalReservedWidth
        let rightExtent = maxDrawingX + lineGap * 0.45

        return EventLayoutMetrics(
            leftExtent: leftExtent,
            rightExtent: rightExtent,
            totalWidth: leftExtent + rightExtent
        )
    }

    private func maxVisibleEventCount(
        for events: [NoteQueue.QueuedEvent],
        availableWidth: CGFloat,
        baseSpacing: CGFloat,
        noteHeadWidth: CGFloat,
        lineGap: CGFloat,
        accidentalSymbolWidth: CGFloat,
        accidentalSymbolSpacing: CGFloat,
        accidentalNoteGap: CGFloat
    ) -> Int {
        var usedWidth: CGFloat = 0
        var count = 0

        for event in events.reversed() {
            let metrics = eventLayoutMetrics(
                for: event,
                baseSpacing: baseSpacing,
                noteHeadWidth: noteHeadWidth,
                lineGap: lineGap,
                accidentalSymbolWidth: accidentalSymbolWidth,
                accidentalSymbolSpacing: accidentalSymbolSpacing,
                accidentalNoteGap: accidentalNoteGap
            )
            let spacing = horizontalSpacing(for: metrics, baseSpacing: baseSpacing)
            guard usedWidth + spacing <= availableWidth || count == 0 else { break }
            usedWidth += spacing
            count += 1
        }

        return max(count + 1, 1)
    }

    private func incomingSpacing(
        from oldEvents: [NoteQueue.QueuedEvent],
        to newEvents: [NoteQueue.QueuedEvent],
        baseSpacing: CGFloat,
        noteHeadWidth: CGFloat,
        lineGap: CGFloat,
        accidentalSymbolWidth: CGFloat,
        accidentalSymbolSpacing: CGFloat,
        accidentalNoteGap: CGFloat
    ) -> CGFloat {
        let oldIDs = Set(oldEvents.map(\.id))
        let incomingEvents = newEvents.filter { oldIDs.contains($0.id) == false }

        return incomingEvents.reduce(CGFloat(0)) { partialResult, event in
            let metrics = eventLayoutMetrics(
                for: event,
                baseSpacing: baseSpacing,
                noteHeadWidth: noteHeadWidth,
                lineGap: lineGap,
                accidentalSymbolWidth: accidentalSymbolWidth,
                accidentalSymbolSpacing: accidentalSymbolSpacing,
                accidentalNoteGap: accidentalNoteGap
            )
            return partialResult + horizontalSpacing(for: metrics, baseSpacing: baseSpacing)
        }
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
            let sharpBaseFont = UIFont.systemFont(ofSize: lineGap * 1.2, weight: .semibold)
            let sharpDescriptor = sharpBaseFont.fontDescriptor.withDesign(.serif) ?? sharpBaseFont.fontDescriptor
            let sharpFont = UIFont(descriptor: sharpDescriptor, size: lineGap * 1.2)
            let sharpMetrics = glyphMetrics(for: "♯", font: sharpFont)
            let sharpWidth = max(sharpMetrics.bounds.width, sharpMetrics.lineSize.width)
            let sharpSpacing = max(1, sharpWidth * 0.08)
            let sharpNoteGap = max(0, lineGap * 0.1)
            
            let noteStartX = max(clefLeftX + trebleClefWidth, clefLeftX + bassClefWidth) + 10
            let noteEndX = width - margin
            let baseNoteSpacing = lineGap * 2
            let events = renderedEvents.isEmpty ? queue.events : renderedEvents
            let eventIDs = queue.events.map(\.id)
            let eventMetrics = events.map {
                eventLayoutMetrics(
                    for: $0,
                    baseSpacing: baseNoteSpacing,
                    noteHeadWidth: noteHeadWidth,
                    lineGap: lineGap,
                    accidentalSymbolWidth: sharpWidth,
                    accidentalSymbolSpacing: sharpSpacing,
                    accidentalNoteGap: sharpNoteGap
                )
            }
            let eventSpacings = eventMetrics.map {
                horizontalSpacing(for: $0, baseSpacing: baseNoteSpacing)
            }
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

                ZStack {
                    ForEach(Array(events.enumerated()), id: \.element.id) { (index, event) in
                        let newerEventSpacing = eventSpacings.dropFirst(index + 1).reduce(CGFloat(0), +)
                        let layoutMetrics = eventMetrics[index]
                        let slotRight = noteEndX + slideOffset - newerEventSpacing
                        let noteX = slotRight - layoutMetrics.rightExtent
                        let chord = chordNotes(for: event.notes)
                        let stemUp = stemGoesUp(for: chord)
                        let ledgerLineSteps = Array(Set(chord.flatMap { ledgerSteps(for: $0.step) })).sorted()
                        let blackKeyNotes = chord.filter { $0.note.isBlackKey }
                        let ledgerWidth = noteHeadWidth * 2.15
                        let leftEdge = noteX - layoutMetrics.leftExtent
                        let rightEdge = noteX + layoutMetrics.rightExtent

                        if rightEdge >= noteStartX && leftEdge <= noteEndX {
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
                                        x: noteX - noteHeadWidth / 2 - sharpNoteGap - sharpWidth / 2 - CGFloat(accidentalIndex) * (sharpWidth + sharpSpacing),
                                        y: yFor(step: chordNote.step, top: topPad, stepSize: stepSize) + sharpMetrics.offset.height
                                    )
                            }
                        }
                    }
                }
                .mask {
                    Rectangle()
                        .frame(width: max(noteEndX - noteStartX, 0), height: height)
                        .position(x: noteStartX + max(noteEndX - noteStartX, 0) / 2, y: height / 2)
                }
            }
            .onAppear {
                if renderedEvents.isEmpty {
                    renderedEvents = queue.events
                }
            }
            .onChange(of: eventIDs) { _, _ in
                let incomingWidth = incomingSpacing(
                    from: renderedEvents,
                    to: queue.events,
                    baseSpacing: baseNoteSpacing,
                    noteHeadWidth: noteHeadWidth,
                    lineGap: lineGap,
                    accidentalSymbolWidth: sharpWidth,
                    accidentalSymbolSpacing: sharpSpacing,
                    accidentalNoteGap: sharpNoteGap
                )

                if incomingWidth > 0 {
                    var transaction = Transaction(animation: nil)
                    transaction.disablesAnimations = true

                    withTransaction(transaction) {
                        renderedEvents = queue.events
                        slideOffset = incomingWidth
                    }

                    DispatchQueue.main.async {
                        withAnimation(.easeInOut(duration: slideDuration)) {
                            slideOffset = 0
                        }
                    }
                } else {
                    renderedEvents = queue.events
                    slideOffset = 0
                }

                let queueMaxVisible = maxVisibleEventCount(
                    for: queue.events,
                    availableWidth: noteEndX - noteStartX,
                    baseSpacing: baseNoteSpacing,
                    noteHeadWidth: noteHeadWidth,
                    lineGap: lineGap,
                    accidentalSymbolWidth: sharpWidth,
                    accidentalSymbolSpacing: sharpSpacing,
                    accidentalNoteGap: sharpNoteGap
                )

                if queue.events.count > queueMaxVisible {
                    DispatchQueue.main.asyncAfter(deadline: .now() + slideDuration + 0.05) {
                        queue.trim(keepingAtMost: queueMaxVisible)
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
