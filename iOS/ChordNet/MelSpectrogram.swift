import Accelerate

/// Compute a log-Mel spectrogram that matches the Python training
/// pipeline (`preprocess.py`):
///
///     librosa.feature.melspectrogram(y, sr, n_fft, hop_length, n_mels)
///     np.log(mel + 1e-6)
///
/// Uses vDSP for FFT and vectorised operations.
final class MelSpectrogram: @unchecked Sendable {

    struct Result {
        /// Flat row-major array: `data[m * nFrames + t]`.
        let data: [Float]
        let nMels: Int
        let nFrames: Int
    }

    // MARK: - Parameters

    let sampleRate: Float
    let nFFT: Int
    let hopLength: Int
    let nMels: Int
    let nFreqBins: Int          // nFFT / 2 + 1

    // MARK: - Precomputed tables

    /// Periodic Hann window (matches librosa's default).
    private let window: [Float]
    /// Mel filterbank, flat row-major: `[nMels * nFreqBins]`.
    private let filterbank: [Float]
    /// vDSP FFT plan.
    private let fftSetup: FFTSetup
    private let log2n: vDSP_Length

    // MARK: - Init / deinit

    init(sampleRate: Float = 22050,
         nFFT: Int = 2048,
         hopLength: Int = 512,
         nMels: Int = 229) {
        self.sampleRate = sampleRate
        self.nFFT       = nFFT
        self.hopLength  = hopLength
        self.nMels      = nMels
        self.nFreqBins  = nFFT / 2 + 1

        // Periodic Hann (same as librosa fftbins=True).
        window = vDSP.window(
            ofType: Float.self,
            usingSequence: .hanningDenormalized,
            count: nFFT,
            isHalfWindow: false
        )

        // FFT setup.
        log2n = vDSP_Length(log2(Float(nFFT)))
        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create FFT setup")
        }
        fftSetup = setup

        // Mel filterbank (Slaney scale + normalisation to match librosa defaults).
        filterbank = Self.buildMelFilterbank(
            sampleRate: sampleRate, nFFT: nFFT, nMels: nMels
        )
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    // MARK: - Public API

    /// Compute the log-Mel spectrogram for `samples` (center=false).
    ///
    /// Returns `nil` if the input is too short for even one frame.
    func compute(samples: [Float]) -> Result? {
        let nFrames = 1 + (samples.count - nFFT) / hopLength
        guard nFrames > 0 else { return nil }

        // Scratch buffers (reused across frames).
        var windowed = [Float](repeating: 0, count: nFFT)
        var realp    = [Float](repeating: 0, count: nFFT / 2)
        var imagp    = [Float](repeating: 0, count: nFFT / 2)
        var power    = [Float](repeating: 0, count: nFreqBins)

        // Output: flat [nMels * nFrames].
        var output   = [Float](repeating: 0, count: nMels * nFrames)

        for f in 0 ..< nFrames {
            let start = f * hopLength

            // 1. Apply Hann window.
            samples.withUnsafeBufferPointer { samplesPtr in
                vDSP_vmul(
                    samplesPtr.baseAddress! + start, 1,
                    window, 1,
                    &windowed, 1,
                    vDSP_Length(nFFT)
                )
            }

            // 2. Pack into split-complex and run real FFT.
            windowed.withUnsafeBufferPointer { wPtr in
                wPtr.baseAddress!.withMemoryRebound(
                    to: DSPComplex.self, capacity: nFFT / 2
                ) { complexPtr in
                    realp.withUnsafeMutableBufferPointer { rBuf in
                        imagp.withUnsafeMutableBufferPointer { iBuf in
                            var split = DSPSplitComplex(realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)
                            vDSP_ctoz(complexPtr, 2, &split, 1, vDSP_Length(nFFT / 2))
                        }
                    }
                }
            }

            realp.withUnsafeMutableBufferPointer { rBuf in
                imagp.withUnsafeMutableBufferPointer { iBuf in
                    var split = DSPSplitComplex(realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)
                    vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                }
            }

            // 3. Power spectrum.
            //    DC  = realp[0]^2, Nyquist = imagp[0]^2.
            power[0] = realp[0] * realp[0]
            for k in 1 ..< nFFT / 2 {
                power[k] = realp[k] * realp[k] + imagp[k] * imagp[k]
            }
            power[nFFT / 2] = imagp[0] * imagp[0]

            // vDSP forward FFT scales output by 2, so power is 4× the
            // true |DFT|².  Divide by 4 to match librosa's |STFT|².
            var scale: Float = 1.0 / 4.0
            vDSP_vsmul(power, 1, &scale, &power, 1, vDSP_Length(nFreqBins))

            // 4. Apply mel filterbank (matrix × vector).
            //    filterbank is [nMels × nFreqBins], power is [nFreqBins].
            filterbank.withUnsafeBufferPointer { fbPtr in
                for m in 0 ..< nMels {
                    var dot: Float = 0
                    vDSP_dotpr(
                        fbPtr.baseAddress! + m * nFreqBins, 1,
                        power, 1,
                        &dot,
                        vDSP_Length(nFreqBins)
                    )
                    output[m * nFrames + f] = dot
                }
            }
        }

        // 5. log(mel + 1e-6)  — matches preprocess.py.
        let count = nMels * nFrames
        let logOffset: Float = 1e-6
        for i in 0 ..< count {
            output[i] = logf(output[i] + logOffset)
        }

        return Result(data: output, nMels: nMels, nFrames: nFrames)
    }

    // MARK: - Mel filterbank construction (Slaney, matching librosa defaults)

    private static func hzToMel(_ hz: Float) -> Float {
        let fSp: Float  = 200.0 / 3.0
        var mel = hz / fSp

        let minLogHz: Float = 1000.0
        let minLogMel       = minLogHz / fSp
        let logStep: Float  = logf(6.4) / 27.0

        if hz >= minLogHz {
            mel = minLogMel + logf(hz / minLogHz) / logStep
        }
        return mel
    }

    private static func melToHz(_ mel: Float) -> Float {
        let fSp: Float  = 200.0 / 3.0
        var hz = fSp * mel

        let minLogHz: Float = 1000.0
        let minLogMel       = minLogHz / fSp
        let logStep: Float  = logf(6.4) / 27.0

        if mel >= minLogMel {
            hz = minLogHz * expf(logStep * (mel - minLogMel))
        }
        return hz
    }

    /// Build a Slaney-normalised mel filterbank identical to
    /// `librosa.filters.mel(sr, n_fft, n_mels, norm='slaney')`.
    private static func buildMelFilterbank(
        sampleRate: Float,
        nFFT: Int,
        nMels: Int,
        fMin: Float = 0,
        fMax: Float? = nil
    ) -> [Float] {
        let fMax = fMax ?? sampleRate / 2
        let nFreqBins = nFFT / 2 + 1

        // FFT bin centre frequencies.
        var fftFreqs = [Float](repeating: 0, count: nFreqBins)
        for i in 0 ..< nFreqBins {
            fftFreqs[i] = Float(i) * sampleRate / Float(nFFT)
        }

        // nMels + 2 mel-spaced points converted back to Hz.
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        var melCentres = [Float](repeating: 0, count: nMels + 2)
        for i in 0 ..< nMels + 2 {
            let melVal = melMin + Float(i) * (melMax - melMin) / Float(nMels + 1)
            melCentres[i] = melToHz(melVal)
        }

        // Differences between consecutive mel points.
        var fdiff = [Float](repeating: 0, count: nMels + 1)
        for i in 0 ..< nMels + 1 {
            fdiff[i] = melCentres[i + 1] - melCentres[i]
        }

        // Build triangular filters + Slaney area normalisation.
        var fb = [Float](repeating: 0, count: nMels * nFreqBins)
        for m in 0 ..< nMels {
            let enorm = 2.0 / (melCentres[m + 2] - melCentres[m])
            for k in 0 ..< nFreqBins {
                let lower = (fftFreqs[k] - melCentres[m])     / fdiff[m]
                let upper = (melCentres[m + 2] - fftFreqs[k]) / fdiff[m + 1]
                let val   = max(0, min(lower, upper))
                fb[m * nFreqBins + k] = val * enorm
            }
        }
        return fb
    }
}
