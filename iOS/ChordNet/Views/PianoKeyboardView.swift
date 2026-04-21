import SwiftUI

struct PianoKeyboardView: View {
    let activeMIDINotes: Set<Int>

    private let whiteKeys = KeyboardLayout.visibleWhiteKeys
    private let blackKeys = KeyboardLayout.visibleBlackKeys

    var body: some View {
        GeometryReader { geometry in
            let whiteKeyWidth = geometry.size.width / CGFloat(whiteKeys.count)
            let blackKeyWidth = whiteKeyWidth * 0.62
            let blackKeyHeight = geometry.size.height * 0.52

            ZStack(alignment: .topLeading) {
                LinearGradient(
                    colors: [
                        Color(red: 0.20, green: 0.24, blue: 0.31),
                        Color(red: 0.10, green: 0.13, blue: 0.19)
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )

                HStack(spacing: 0) {
                    ForEach(whiteKeys) { note in
                        RoundedRectangle(cornerRadius: 5, style: .continuous)
                            .fill(fillColor(for: note, defaultColor: Color.white.opacity(0.98)))
                            .overlay(
                                RoundedRectangle(cornerRadius: 5, style: .continuous)
                                    .stroke(Color.black.opacity(0.22), lineWidth: 1)
                            )
                            .frame(width: whiteKeyWidth)
                            .overlay(alignment: .bottom) {
                                Rectangle()
                                    .fill(Color.black.opacity(0.10))
                                    .frame(height: 14)
                            }
                    }
                }
                .padding(.top, 22)
                .padding(.horizontal, 8)
                .padding(.bottom, 16)

                ForEach(blackKeys) { note in
                    let offset = CGFloat(KeyboardLayout.visibleWhiteKeyOffset(for: note.midiNumber)) * whiteKeyWidth - (blackKeyWidth / 2) + 8
                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .fill(fillColor(for: note, defaultColor: Color.black.opacity(0.84)))
                        .frame(width: blackKeyWidth, height: blackKeyHeight)
                        .overlay(alignment: .bottom) {
                            RoundedRectangle(cornerRadius: 2, style: .continuous)
                                .fill(Color.white.opacity(0.08))
                                .frame(height: 10)
                        }
                        .offset(x: offset, y: 22)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
            .shadow(color: Color.black.opacity(0.14), radius: 12, y: 6)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
    }

    private func fillColor(for note: PianoNote, defaultColor: Color) -> Color {
        guard activeMIDINotes.contains(note.midiNumber) else {
            return defaultColor
        }

        return note.isBlackKey
            ? Color(red: 0.16, green: 0.41, blue: 0.95)
            : Color(red: 0.15, green: 0.44, blue: 0.98)
    }
}