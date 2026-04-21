import SwiftUI

struct WaveformEnvelopeView: View {
    enum Style: String {
        case envelopeBars
        case neonThreads
        case layeredRibbons
    }

    let envelope: [CGFloat]
    let isListening: Bool
    let noteLabel: String
    let detailLabel: String
    let style: Style

    init(
        envelope: [CGFloat],
        isListening: Bool,
        noteLabel: String,
        detailLabel: String,
        style: Style = .envelopeBars
    ) {
        self.envelope = envelope
        self.isListening = isListening
        self.noteLabel = noteLabel
        self.detailLabel = detailLabel
        self.style = style
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Live Envelope")
                        .font(.system(size: 16, weight: .semibold, design: .rounded))
                        .foregroundStyle(Color.white.opacity(0.78))

                    Text(noteLabel)
                        .font(.system(size: 34, weight: .heavy, design: .rounded))
                        .foregroundStyle(.white)
                        .lineLimit(1)
                        .minimumScaleFactor(0.6)
                }

                Spacer()

                Text(detailLabel)
                    .font(.system(size: 15, weight: .medium, design: .rounded))
                    .foregroundStyle(isListening ? Color(red: 0.60, green: 0.94, blue: 0.77) : Color.white.opacity(0.70))
            }

            GeometryReader { geometry in
                if isFlatEnvelope {
                    flatLine(in: geometry.size)
                } else {
                    switch style {
                    case .envelopeBars:
                        envelopeBars(in: geometry.size)
                    case .neonThreads:
                        neonThreads(in: geometry.size)
                    case .layeredRibbons:
                        layeredRibbons(in: geometry.size)
                    }
                }
            }
            .frame(height: 92)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 18)
        .frame(maxWidth: .infinity)
        .frame(height: 175)
        .background(background)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(Color.white.opacity(0.08))
                .frame(height: 1)
        }
    }

    private var isFlatEnvelope: Bool {
        envelope.allSatisfy { $0 <= 0.01 }
    }

    private func flatLine(in size: CGSize) -> some View {
        ZStack {
            Rectangle()
                .fill(Color.white.opacity(0.14))
                .frame(height: 1)

            Capsule(style: .continuous)
                .fill(Color.white.opacity(0.58))
                .frame(height: 2)
                .padding(.horizontal, 2)
        }
        .frame(width: size.width, height: size.height)
    }

    @ViewBuilder
    private func envelopeBars(in size: CGSize) -> some View {
        let barCount = max(envelope.count, 1)
        let spacing: CGFloat = 3
        let totalSpacing = spacing * CGFloat(max(barCount - 1, 0))
        let barWidth = max((size.width - totalSpacing) / CGFloat(barCount), 2)

        HStack(alignment: .center, spacing: spacing) {
            ForEach(Array(envelope.enumerated()), id: \.offset) { _, value in
                Capsule(style: .continuous)
                    .fill(barGradient(for: value))
                    .frame(width: barWidth, height: max(2, size.height * value))
                    .frame(maxHeight: .infinity, alignment: .center)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
    }

    private func neonThreads(in size: CGSize) -> some View {
        Canvas { context, canvasSize in
            let points = normalizedPoints(in: canvasSize)
            let centerY = canvasSize.height * 0.5

            if points.count > 1 {
                let glowColors: [Color] = [
                    Color(red: 0.16, green: 0.42, blue: 0.98),
                    Color(red: 0.26, green: 0.62, blue: 1.0),
                    Color(red: 0.96, green: 0.82, blue: 0.20),
                ]

                for (index, color) in glowColors.enumerated() {
                    let amplitudeScale = 0.98 - CGFloat(index) * 0.16
                    let phaseShift = CGFloat(index - 1) * 0.22
                    let path = oscillatingPath(
                        points: points,
                        width: canvasSize.width,
                        centerY: centerY,
                        amplitudeScale: amplitudeScale,
                        phaseShift: phaseShift
                    )

                    context.stroke(
                        path,
                        with: .color(color.opacity(index == 2 ? 0.82 : 0.90)),
                        style: StrokeStyle(lineWidth: index == 2 ? 1.8 : 3.0, lineCap: .round, lineJoin: .round)
                    )

                    context.addFilter(.blur(radius: index == 2 ? 1.2 : 6))
                    context.stroke(
                        path,
                        with: .color(color.opacity(index == 2 ? 0.34 : 0.42)),
                        style: StrokeStyle(lineWidth: index == 2 ? 3.5 : 8.5, lineCap: .round, lineJoin: .round)
                    )
                    context.drawLayer { _ in }
                }
            }
        }
        .frame(width: size.width, height: size.height)
    }

    private func layeredRibbons(in size: CGSize) -> some View {
        Canvas { context, canvasSize in
            let points = normalizedPoints(in: canvasSize)
            let centerY = canvasSize.height * 0.5
            let layers: [(offset: CGFloat, opacity: Double, lineWidth: CGFloat)] = [
                (-0.22, 0.22, 1.0),
                (-0.10, 0.28, 1.2),
                (0.00, 0.36, 1.4),
                (0.10, 0.28, 1.2),
                (0.22, 0.22, 1.0),
            ]

            for layer in layers {
                let path = oscillatingPath(
                    points: points,
                    width: canvasSize.width,
                    centerY: centerY,
                    amplitudeScale: 0.78,
                    phaseShift: layer.offset
                )
                let shifted = path.offsetBy(dx: 0, dy: canvasSize.height * layer.offset * 0.22)

                context.stroke(
                    shifted,
                    with: .color(.white.opacity(layer.opacity)),
                    style: StrokeStyle(lineWidth: layer.lineWidth, lineCap: .round, lineJoin: .round)
                )
            }
        }
        .frame(width: size.width, height: size.height)
    }

    private func barGradient(for value: CGFloat) -> LinearGradient {
        let intensity = max(0.15, value)
        return LinearGradient(
            colors: [
                Color.white.opacity(0.22 + intensity * 0.20),
                Color(red: 0.43, green: 0.74, blue: 0.98).opacity(0.55 + intensity * 0.25),
                Color(red: 0.18, green: 0.52, blue: 0.96).opacity(0.78 + intensity * 0.16),
            ],
            startPoint: .bottom,
            endPoint: .top
        )
    }

    private var background: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.11, green: 0.16, blue: 0.24),
                    Color(red: 0.07, green: 0.10, blue: 0.18),
                    Color(red: 0.12, green: 0.18, blue: 0.28),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            RadialGradient(
                colors: [
                    Color(red: 0.34, green: 0.58, blue: 0.95).opacity(0.22),
                    Color.clear,
                ],
                center: .topTrailing,
                startRadius: 10,
                endRadius: 220
            )

            VStack {
                Spacer()
                HStack(spacing: 0) {
                    ForEach(0..<5, id: \.self) { index in
                        Rectangle()
                            .fill(Color.white.opacity(index == 0 ? 0.10 : 0.05))
                            .frame(height: 1)
                        Spacer(minLength: 0)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 28)
            }
        }
    }

    private func normalizedPoints(in size: CGSize) -> [CGPoint] {
        let safeEnvelope = envelope.isEmpty ? [0] : envelope
        let count = safeEnvelope.count
        guard count > 1 else {
            return [CGPoint(x: 0, y: size.height * 0.5), CGPoint(x: size.width, y: size.height * 0.5)]
        }

        return safeEnvelope.enumerated().map { index, value in
            let progress = CGFloat(index) / CGFloat(count - 1)
            return CGPoint(x: progress * size.width, y: min(value, 1.0))
        }
    }

    private func oscillatingPath(
        points: [CGPoint],
        width: CGFloat,
        centerY: CGFloat,
        amplitudeScale: CGFloat,
        phaseShift: CGFloat
    ) -> Path {
        var path = Path()
        guard let first = points.first else { return path }

        let firstY = centerY + waveOffset(for: first, width: width, centerY: centerY, amplitudeScale: amplitudeScale, phaseShift: phaseShift)
        path.move(to: CGPoint(x: first.x, y: firstY))

        for point in points.dropFirst() {
            let y = centerY + waveOffset(for: point, width: width, centerY: centerY, amplitudeScale: amplitudeScale, phaseShift: phaseShift)
            path.addLine(to: CGPoint(x: point.x, y: y))
        }

        return path
    }

    private func waveOffset(
        for point: CGPoint,
        width: CGFloat,
        centerY: CGFloat,
        amplitudeScale: CGFloat,
        phaseShift: CGFloat
    ) -> CGFloat {
        let normalizedX = point.x / max(width, 1)
        let wave = sin((normalizedX * 10.5 + phaseShift) * .pi)
        let amplitude = point.y * centerY * amplitudeScale
        return wave * amplitude
    }
}

#Preview {
    WaveformEnvelopeView(
        envelope: [0.10, 0.16, 0.28, 0.52, 0.76, 0.91, 0.74, 0.48, 0.26, 0.18, 0.24, 0.38, 0.62, 0.84, 0.70, 0.44, 0.22, 0.12],
        isListening: true,
        noteLabel: "C4 · E4 · G4",
        detailLabel: "261.6 Hz",
        style: .neonThreads
    )
}
