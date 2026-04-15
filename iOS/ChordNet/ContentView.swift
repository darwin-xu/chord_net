import SwiftUI

struct ContentView: View {
    @StateObject private var engine = AudioEngine()

    var body: some View {
        VStack(spacing: 0) {
            Group {
                if engine.detectedNotes.isEmpty {
                    Spacer()
                    Text(engine.statusMessage)
                        .font(.title2)
                        .multilineTextAlignment(.center)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal)
                    Spacer()
                } else {
                    ScrollView {
                        LazyVGrid(
                            columns: [GridItem(.adaptive(minimum: 72))],
                            spacing: 12
                        ) {
                            ForEach(engine.detectedNotes, id: \.self) { note in
                                Text(note)
                                    .font(.title)
                                    .fontWeight(.bold)
                                    .padding(.horizontal, 14)
                                    .padding(.vertical, 10)
                                    .background(Color.accentColor.opacity(0.15))
                                    .clipShape(RoundedRectangle(cornerRadius: 8))
                            }
                        }
                        .padding()
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            WaveformView(levels: engine.waveformLevels, isActive: engine.isRunning)
                .frame(height: 72)
                .padding(.horizontal)
                .padding(.bottom, 12)

            Divider()

            Button {
                if engine.isRunning {
                    engine.stop()
                } else {
                    engine.start()
                }
            } label: {
                Text(engine.isRunning ? "Stop" : "Start")
                    .font(.title2)
                    .fontWeight(.semibold)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
            }
            .buttonStyle(.borderedProminent)
            .tint(engine.isRunning ? .red : .blue)
            .padding()
        }
    }
}

private struct WaveformView: View {
    let levels: [Float]
    let isActive: Bool

    var body: some View {
        GeometryReader { geometry in
            let barCount = max(levels.count, 1)
            let spacing: CGFloat = 3
            let barWidth = max(2, (geometry.size.width - spacing * CGFloat(barCount - 1)) / CGFloat(barCount))
            let maxHeight = geometry.size.height

            HStack(alignment: .center, spacing: spacing) {
                ForEach(levels.indices, id: \.self) { index in
                    let level = CGFloat(levels[index])
                    RoundedRectangle(cornerRadius: 3)
                        .fill(isActive ? Color.accentColor : Color.secondary.opacity(0.35))
                        .frame(
                            width: barWidth,
                            height: max(4, maxHeight * level)
                        )
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
        }
        .accessibilityLabel("Microphone waveform")
    }
}

#Preview {
    ContentView()
}
