#ifndef WHISPER_TRANSCRIBER_H
#define WHISPER_TRANSCRIBER_H

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <chrono>
#include <stdexcept>
#include <atomic>
#include <algorithm>
#include <queue>

#include "ITranscriber.h"
#include "whisper.h"
#include "voice-activity-detector.h"

/**
 * WhisperTranscriber - Wrapper for Silero VAD + Whisper.cpp integration
 *
 * Implementation details:
 * - Uses Silero VAD for voice activity detection
 * - Whisper.cpp for speech recognition (statically linked)
 * - Token probability-based confidence scoring
 * - Thread-safe processing with internal buffering
 * - GPU acceleration support via CUDA (optional)
 */
class WhisperTranscriber : public ITranscriber {
private:
    // Whisper context
    std::unique_ptr<whisper_context, decltype(&whisper_free)> ctx_;

    // VAD components (TEN VAD)
    std::unique_ptr<VoiceActivityDetector> vad_;
    size_t lastProcessedSegmentIndex_;

    // Transcription queue
    std::queue<ITranscriber::TranscriptionResult> completedTranscriptions_;

    // Thread safety
    mutable std::mutex vadMutex_;
    mutable std::mutex transcriptionMutex_;
    mutable std::mutex ctxMutex_;

    // Configuration
    int sampleRate_;
    float confidenceThreshold_;

    // Performance metrics
    struct PerformanceMetrics {
        std::atomic<uint64_t> totalTranscriptions{0};
        std::atomic<uint64_t> totalInferenceTimeMs{0};
        std::atomic<uint64_t> minInferenceTimeMs{UINT64_MAX};
        std::atomic<uint64_t> maxInferenceTimeMs{0};
        std::atomic<uint64_t> totalAudioDurationMs{0};

        void recordInference(uint64_t durationMs, float audioDurationSec) {
            totalTranscriptions++;
            totalInferenceTimeMs += durationMs;
            totalAudioDurationMs += static_cast<uint64_t>(audioDurationSec * 1000);

            // Update min
            uint64_t currentMin = minInferenceTimeMs.load();
            while (durationMs < currentMin &&
                   !minInferenceTimeMs.compare_exchange_weak(currentMin, durationMs));

            // Update max
            uint64_t currentMax = maxInferenceTimeMs.load();
            while (durationMs > currentMax &&
                   !maxInferenceTimeMs.compare_exchange_weak(currentMax, durationMs));
        }

        void printStats() const {
            uint64_t count = totalTranscriptions.load();
            if (count == 0) {
                std::cout << "\n=== No transcriptions performed yet ===" << std::endl;
                return;
            }

            uint64_t totalTime = totalInferenceTimeMs.load();
            uint64_t totalAudio = totalAudioDurationMs.load();
            uint64_t avg = totalTime / count;
            float realTimeFactorMs = static_cast<float>(totalTime) / static_cast<float>(totalAudio);

            std::cout << "\n=== Transcription Performance Statistics ===" << std::endl;
            std::cout << "  Total Transcriptions: " << count << std::endl;
            std::cout << "  Average Inference Time: " << avg << "ms" << std::endl;
            std::cout << "  Min Inference Time: " << minInferenceTimeMs.load() << "ms" << std::endl;
            std::cout << "  Max Inference Time: " << maxInferenceTimeMs.load() << "ms" << std::endl;
            std::cout << "  Total Audio Processed: " << (totalAudio / 1000.0f) << "s" << std::endl;
            std::cout << "  Real-Time Factor: " << realTimeFactorMs << "x" << std::endl;
            std::cout << "  (Lower is better; <1.0 means faster than real-time)" << std::endl;
            std::cout << "===========================================" << std::endl;
        }

        void reset() {
            totalTranscriptions = 0;
            totalInferenceTimeMs = 0;
            minInferenceTimeMs = UINT64_MAX;
            maxInferenceTimeMs = 0;
            totalAudioDurationMs = 0;
        }
    };

    mutable PerformanceMetrics metrics_;

    // Internal helper methods
    float computeConfidence(whisper_context* ctx, int segmentIndex) const;
    ITranscriber::TranscriptionResult transcribeSegment(const VoiceActivitySegment& segment);

public:
    /**
     * Constructor - Initialize Whisper model and TEN VAD
     *
     * @param modelPath Path to Whisper GGML model file (e.g., "models/ggml-model.bin")
     * @param vadThreshold VAD detection threshold (0.0-1.0, default: 0.5)
     * @param sampleRate Audio sample rate (default: 16000 Hz)
     * @param confidenceThreshold Minimum confidence for accepting transcriptions (default: 0.65)
     * @param useGPU Enable CUDA GPU acceleration (default: true)
     */
    explicit WhisperTranscriber(
        const std::string& modelPath,
        float vadThreshold = 0.5f,
        int sampleRate = 16000,
        float confidenceThreshold = 0.65f,
        bool useGPU = true
    );

    /**
     * Destructor
     */
    ~WhisperTranscriber();

    // ITranscriber interface implementation
    void processAudio(const float* audioData, size_t numFrames) override;
    std::vector<ITranscriber::TranscriptionResult> getCompletedTranscriptions() override;
    bool shouldAcceptTranscription(const ITranscriber::TranscriptionResult& result) const override;
    void printPerformanceStats() const override;
    std::string getModelName() const override;

    /**
     * Reset performance statistics
     */
    void resetPerformanceStats() {
        metrics_.reset();
    }
};

#endif // WHISPER_TRANSCRIBER_H
