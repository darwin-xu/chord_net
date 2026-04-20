@preconcurrency import AVFoundation
import CoreML
import Accelerate

/// Captures microphone audio, computes a log-Mel spectrogram, runs the
/// CoreML ChordNet model, and publishes detected note names.
///
/// Thread safety: `sampleBuffer` and `lastInferenceTime` are only
/// accessed from `processingQueue`.  Published properties are only
/// written from the main queue.
final class AudioEngine: ObservableObject, @unchecked Sendable {

    private final class ConverterInput: @unchecked Sendable {
        private var buffer: AVAudioPCMBuffer?

        init(buffer: AVAudioPCMBuffer) {
            self.buffer = buffer
        }

        func take() -> AVAudioPCMBuffer? {
            defer { buffer = nil }
            return buffer
        }
    }

    // MARK: - Published state

    @Published var detectedNotes: [String] = []
    @Published var isRunning: Bool = false
    @Published var waveformLevels: [Float] = Array(repeating: 0, count: 32)
    @Published var statusMessage: String = "Tap Start to begin"

    // MARK: - Constants

    private let targetSR: Double = 22050
    private let nFFT      = 2048
    private let hopLength  = 512
    private let nMels      = 229
    private let nTimeFrames = 32
    private let nNotes     = 88
    private let threshold: Float = 0.95
    /// Minimum RMS below which inference is skipped (silence / ambient noise).
    private let energyGateRMS: Float = 0.008
    /// Minimum seconds between consecutive inferences.
    private let inferenceInterval: CFAbsoluteTime = 0.30

    /// Number of audio samples needed for one inference patch
    /// (center=true):  (nTimeFrames - 1) * hopLength
    private var requiredSamples: Int { (nTimeFrames - 1) * hopLength }  // 15872

    // MARK: - Note names (MIDI 21–108, matching inference.py)

    static let noteNames: [String] = {
        let semitone = ["C", "C#", "D", "D#", "E",
                        "F", "F#", "G", "G#", "A", "A#", "B"]
        return (21 ... 108).map { midi in
            "\(semitone[midi % 12])\(midi / 12 - 1)"
        }
    }()

    // MARK: - Private audio state

    private let engine = AVAudioEngine()
    private let targetFormat: AVAudioFormat
    private var audioConverter: AVAudioConverter?
    private var converterInputFormat: AVAudioFormat?
    private let melSpec: MelSpectrogram
    private var model: MLModel?

    /// Serial queue that owns `sampleBuffer` and `lastInferenceTime`.
    private let processingQueue = DispatchQueue(
        label: "com.chordnet.processing", qos: .userInteractive
    )
    private var sampleBuffer: [Float] = []
    private var lastInferenceTime: CFAbsoluteTime = 0

    // MARK: - Init

    init() {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSR,
            channels: 1,
            interleaved: false
        ) else {
            fatalError("Failed to create target audio format")
        }
        targetFormat = format
        melSpec = MelSpectrogram(
            sampleRate: Float(targetSR),
            nFFT: nFFT,
            hopLength: hopLength,
            nMels: nMels
        )
        loadModel()
    }

    // MARK: - Public API

    func start() {
        guard !isRunning else { return }

        Task {
            let granted = await AVAudioApplication.requestRecordPermission()
            guard granted else {
                await MainActor.run {
                    self.statusMessage = "Microphone permission denied."
                }
                print("[ChordNet] Microphone permission denied")
                return
            }
            await MainActor.run {
                self.startEngine()
            }
        }
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        audioConverter = nil
        converterInputFormat = nil
        isRunning = false
        detectedNotes = []
        waveformLevels = Array(repeating: 0, count: waveformLevels.count)
        statusMessage = model == nil ? "Model not found. Add ChordNetModel.mlmodelc to the app bundle." : "Tap Start to begin"
        processingQueue.async { [weak self] in
            self?.sampleBuffer.removeAll()
        }
    }

    // MARK: - Private helpers

    private func loadModel() {
        guard let url = Bundle.main.url(
            forResource: "ChordNetModel", withExtension: "mlmodelc"
        ) else {
            statusMessage = "Model not found. Add ChordNetModel.mlmodelc to the app bundle."
            print("[ChordNet] CoreML model not found in bundle - run transfer_to_ios.py first")
            return
        }
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            model = try MLModel(contentsOf: url, configuration: config)
            statusMessage = "Tap Start to begin"
        } catch {
            statusMessage = "Model failed to load."
            print("[ChordNet] Failed to load model: \(error)")
        }
    }

    private func startEngine() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement)
            try session.setPreferredSampleRate(targetSR)
            try session.setActive(true)

            let inputNode = engine.inputNode
            let inputFormat = inputNode.outputFormat(forBus: 0)
            guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else {
                statusMessage = "Invalid microphone format."
                print("[ChordNet] Invalid microphone format: \(inputFormat)")
                return
            }

            audioConverter = nil
            converterInputFormat = nil

            inputNode.installTap(
                onBus: 0,
                bufferSize: AVAudioFrameCount(nFFT),
                format: inputFormat
            ) { [weak self] buffer, _ in
                self?.handleBuffer(buffer)
            }

            try engine.start()
            isRunning = true
            statusMessage = model == nil ? "Mic on. Model not found." : "Listening at \(Int(targetSR)) Hz"
        } catch {
            statusMessage = "Audio engine failed to start."
            print("[ChordNet] Engine start error: \(error)")
        }
    }

    /// Called on the audio I/O thread – converts to the model sample rate and dispatches.
    private func handleBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let samples = samplesAtModelRate(from: buffer), !samples.isEmpty else { return }

        processingQueue.async { [weak self] in
            self?.appendAndProcess(samples)
        }
    }

    private func samplesAtModelRate(from buffer: AVAudioPCMBuffer) -> [Float]? {
        if buffer.format.sampleRate == targetSR,
           buffer.format.channelCount == 1,
           let channelData = buffer.floatChannelData {
            let count = Int(buffer.frameLength)
            return Array(UnsafeBufferPointer(start: channelData[0], count: count))
        }

        if audioConverter == nil || converterInputFormat != buffer.format {
            guard let converter = AVAudioConverter(from: buffer.format, to: targetFormat) else {
                print("[ChordNet] Cannot convert microphone format \(buffer.format) to \(targetFormat)")
                return nil
            }
            converter.downmix = true
            converter.sampleRateConverterQuality = AVAudioQuality.high.rawValue
            audioConverter = converter
            converterInputFormat = buffer.format
        }

        guard let converter = audioConverter else { return nil }

        let ratio = targetFormat.sampleRate / buffer.format.sampleRate
        let capacity = max(1, AVAudioFrameCount(ceil(Double(buffer.frameLength) * ratio)) + 1)
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: targetFormat,
            frameCapacity: capacity
        ) else { return nil }

        let converterInput = ConverterInput(buffer: buffer)
        var conversionError: NSError?
        let status = converter.convert(to: outputBuffer, error: &conversionError) { _, inputStatus in
            guard let inputBuffer = converterInput.take() else {
                inputStatus.pointee = .noDataNow
                return nil
            }

            inputStatus.pointee = .haveData
            return inputBuffer
        }

        if let conversionError {
            print("[ChordNet] Audio conversion error: \(conversionError)")
            return nil
        }

        guard status == .haveData || status == .inputRanDry || status == .endOfStream,
              let channelData = outputBuffer.floatChannelData else {
            print("[ChordNet] Audio conversion failed with status: \(status)")
            return nil
        }

        let count = Int(outputBuffer.frameLength)
        return Array(UnsafeBufferPointer(start: channelData[0], count: count))
    }

    private func publishWaveform(from samples: [Float]) {
        let barCount = 32
        let chunkSize = max(1, samples.count / barCount)
        var levels: [Float] = []
        levels.reserveCapacity(barCount)

        for barIndex in 0 ..< barCount {
            let start = barIndex * chunkSize
            guard start < samples.count else {
                levels.append(0)
                continue
            }

            let end = min(samples.count, start + chunkSize)
            var sum: Float = 0
            for sample in samples[start ..< end] {
                sum += sample * sample
            }

            let rms = sqrt(sum / Float(end - start))
            levels.append(min(1, rms * 12))
        }

        DispatchQueue.main.async { [weak self] in
            self?.waveformLevels = levels
        }
    }

    /// Runs on `processingQueue`.
    private func appendAndProcess(_ samples: [Float]) {
        sampleBuffer.append(contentsOf: samples)
        publishWaveform(from: samples)

        let needed = requiredSamples
        guard sampleBuffer.count >= needed else { return }

        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastInferenceTime >= inferenceInterval else { return }
        lastInferenceTime = now

        // Take the most recent window.
        let audio = Array(sampleBuffer.suffix(needed))
        // Keep buffer trimmed.
        if sampleBuffer.count > needed * 2 {
            sampleBuffer = Array(sampleBuffer.suffix(needed))
        }

        let notes = detect(audio: audio)
        if !notes.isEmpty {
            DispatchQueue.main.async { [weak self] in
                self?.detectedNotes = notes
            }
        }
    }

    /// Full pipeline: DC removal → energy gate → RMS normalise → mel spectrogram → CoreML.
    private func detect(audio: [Float]) -> [String] {
        guard let model else { return [] }

        var buf = audio

        // 1. DC removal (subtract mean).
        var mean: Float = 0
        vDSP_meanv(buf, 1, &mean, vDSP_Length(buf.count))
        var negMean = -mean
        vDSP_vsadd(buf, 1, &negMean, &buf, 1, vDSP_Length(buf.count))

        // 2. Energy gate: skip inference if the signal is too quiet.
        //    Typical ambient noise RMS is ~0.001–0.005; a soft piano
        //    note from a phone mic is ~0.01+.
        var rms: Float = 0
        vDSP_rmsqv(buf, 1, &rms, vDSP_Length(buf.count))
        guard rms >= energyGateRMS else { return [] }

        // 3. RMS normalise to 0.1 (matches training pipeline).
        if rms > 1e-6 {
            var scale = Float(0.1) / rms
            vDSP_vsmul(buf, 1, &scale, &buf, 1, vDSP_Length(buf.count))
        }

        // 3. Log-Mel spectrogram.
        guard let spec = melSpec.compute(samples: buf) else { return [] }
        let nFrames = spec.nFrames

        // 4. CoreML inference.
        guard let mlInput = try? MLMultiArray(
            shape: [1, 1, NSNumber(value: nMels), NSNumber(value: nFrames)],
            dataType: .float32
        ) else { return [] }

        let ptr = mlInput.dataPointer.bindMemory(
            to: Float32.self, capacity: nMels * nFrames
        )
        spec.data.withUnsafeBufferPointer { src in
            ptr.update(from: src.baseAddress!, count: nMels * nFrames)
        }

        let provider = try? MLDictionaryFeatureProvider(
            dictionary: ["mel_spectrogram": MLFeatureValue(multiArray: mlInput)]
        )
        guard let provider,
              let prediction = try? model.prediction(from: provider) else { return [] }

        // Find the first output feature (name varies by export).
        guard let outputName = model.modelDescription.outputDescriptionsByName.keys.first,
              let outputArray = prediction.featureValue(for: outputName)?.multiArrayValue
        else { return [] }

        // 5. Threshold and map to note names.
        let outPtr = outputArray.dataPointer.bindMemory(
            to: Float32.self, capacity: nNotes
        )
        var notes: [String] = []
        for i in 0 ..< nNotes {
            if outPtr[i] >= threshold {
                notes.append(Self.noteNames[i])
            }
        }
        return notes
    }
}
