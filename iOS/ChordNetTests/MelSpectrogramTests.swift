import XCTest
@testable import ChordNet

final class MelSpectrogramTests: XCTestCase {
    func testSilenceProducesExpectedShapeAndFiniteLogFloor() {
        let spectrogram = MelSpectrogram()
        let samples = [Float](repeating: 0, count: 15_872)

        let result = spectrogram.compute(samples: samples)

        XCTAssertEqual(result?.nMels, 229)
        XCTAssertEqual(result?.nFrames, 32)
        XCTAssertEqual(result?.data.count, 229 * 32)
        XCTAssertTrue(result?.data.allSatisfy(\.isFinite) == true)
        XCTAssertTrue(result?.data.allSatisfy { abs($0 - logf(1e-6)) < 0.0001 } == true)
    }

    func testSineWaveProducesFiniteNonSilentEnergy() {
        let spectrogram = MelSpectrogram()
        let sampleRate: Float = 22_050
        let samples = (0..<15_872).map { index -> Float in
            let phase = 2 * Float.pi * 440 * Float(index) / sampleRate
            return 0.1 * sinf(phase)
        }

        let result = spectrogram.compute(samples: samples)

        XCTAssertEqual(result?.nFrames, 32)
        XCTAssertTrue(result?.data.allSatisfy(\.isFinite) == true)
        XCTAssertGreaterThan(result?.data.max() ?? -.infinity, logf(1e-6) + 1)
    }
}
