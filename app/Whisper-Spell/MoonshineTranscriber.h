#ifndef MOONSHINE_TRANSCRIBER_H
#define MOONSHINE_TRANSCRIBER_H

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <chrono>
#include <stdexcept>
#include <atomic>
#include <algorithm>

// Undefine Windows macros that conflict with Moonshine enum values
#ifdef ERROR
#undef ERROR
#endif

#include "ITranscriber.h"
#include "moonshine-cpp.h"
#include "voice-activity-detector.h"

namespace moonshine {

/**
 * Transcription result with metadata
 */
struct TranscriptionResult {
    std::string text;
    float duration;          // Duration of the audio segment in seconds
    std::chrono::milliseconds latency;  // Time taken to transcribe
    bool is_complete;        // Whether the segment is complete

    TranscriptionResult()
        : duration(0.0f), latency(0), is_complete(false) {}
};

/**
 * Performance metrics for transcription monitoring
 */
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

/**
 * MoonshineTranscriber - Wrapper for TEN VAD + Moonshine integration
 *
 * This implementation matches the working MoonshineInference code exactly:
 * - Uses C++ API (moonshine::Transcriber) instead of C API
 * - VAD runs incrementally in audio callback
 * - Transcription uses createStream() per segment
 * - Thread-safe segment processing
 */
class MoonshineTranscriber : public ITranscriber {
private:
    std::unique_ptr<moonshine::Transcriber> transcriber;
    std::unique_ptr<VoiceActivityDetector> vad;
    std::mutex vadMutex;
    size_t lastProcessedSegmentIndex;

public:
    // Public performance metrics for monitoring
    PerformanceMetrics metrics;
    /**
     * Constructor - Aligns with MoonshineInference implementation
     * @param modelPath Path to the Moonshine model directory (containing encoder_model.ort, decoder_model_merged.ort, tokenizer.bin)
     * @param vadThreshold VAD detection threshold (0.0-1.0, default: 0.5)
     */
    explicit MoonshineTranscriber(const std::string& modelPath, float vadThreshold = 0.5f)
        : lastProcessedSegmentIndex(0)
    {
        try {
            // Initialize Moonshine Transcriber (C++ API)
            transcriber = std::make_unique<moonshine::Transcriber>(
                modelPath,
                moonshine::ModelArch::TINY,
                0.5f  // Update interval
            );

            // Initialize TEN VAD (same parameters as working code)
            vad = std::make_unique<VoiceActivityDetector>(
                vadThreshold,  // threshold
                256,           // hop_size (256 samples = 16ms at 16kHz)
                32,            // window_size (32 frames)
                8192,          // look_behind_sample_count (512ms at 16kHz)
                15 * 16000     // max_segment_sample_count (15 seconds)
            );

            // Start VAD
            vad->start();

            std::cout << "MoonshineTranscriber initialized successfully" << std::endl;
            std::cout << "  Model: TINY" << std::endl;
            std::cout << "  VAD Threshold: " << vadThreshold << std::endl;

        } catch (const std::exception& e) {
            throw MoonshineException(std::string("Failed to initialize MoonshineTranscriber: ") + e.what());
        }
    }

    /**
     * Destructor
     */
    ~MoonshineTranscriber() {
        if (vad) {
            vad->stop();
        }
    }

    /**
     * Process audio incrementally (called from audio callback)
     * This feeds audio to TEN VAD for incremental processing.
     *
     * @param audioData Float audio samples (mono, 16kHz)
     * @param numFrames Number of samples
     */
    void processAudio(const float* audioData, size_t numFrames) override {
        if (!vad) {
            throw MoonshineException("VAD not initialized");
        }

        std::lock_guard<std::mutex> lock(vadMutex);

        try {
            // Feed NEW audio to VAD incrementally (VAD maintains internal state)
            vad->process_audio(audioData, numFrames, 16000);
        } catch (const std::exception& e) {
            throw MoonshineException(std::string("VAD processing error: ") + e.what());
        }
    }

    /**
     * Get completed speech segments (non-blocking)
     * Returns segments that have been detected and are ready for transcription.
     * Matches the working MoonshineInference pattern.
     *
     * @return Vector of completed voice activity segments
     */
    std::vector<VoiceActivitySegment> getCompletedSegments() {
        std::lock_guard<std::mutex> lock(vadMutex);

        std::vector<VoiceActivitySegment> completedSegments;

        if (!vad) {
            return completedSegments;
        }

        const auto* segments = vad->get_segments();
        if (!segments) {
            return completedSegments;
        }

        // Process only NEW completed segments (matching working code logic)
        for (size_t i = lastProcessedSegmentIndex; i < segments->size(); i++) {
            const auto& segment = (*segments)[i];
            if (segment.is_complete) {
                completedSegments.push_back(segment);
                lastProcessedSegmentIndex = i + 1;  // Track what we've processed
            }
        }

        return completedSegments;
    }

    /**
     * Transcribe a voice activity segment using Moonshine
     * Uses the SAME approach as the working MoonshineInference code:
     * - Creates a temporary stream for each segment
     * - Uses streaming API to avoid double-VAD issue
     *
     * @param segment Voice activity segment to transcribe
     * @return Transcription result with text and metadata
     */
    TranscriptionResult transcribeSegment(const VoiceActivitySegment& segment) {
        TranscriptionResult result;

        auto startTime = std::chrono::high_resolution_clock::now();

        try {
            if (segment.audio_data.empty()) {
                return result;
            }

            float duration = segment.end_time - segment.start_time;

            // Moonshine requires at least 1.0 second of audio
            constexpr float MIN_DURATION_SECONDS = 1.0f;
            constexpr size_t MIN_SAMPLES = static_cast<size_t>(16000 * MIN_DURATION_SECONDS);

            std::vector<float> audio_to_transcribe = segment.audio_data;

            // Pad short segments with silence to meet minimum length
            if (audio_to_transcribe.size() < MIN_SAMPLES) {
                audio_to_transcribe.resize(MIN_SAMPLES, 0.0f);  // Pad with zeros (silence)
            }

            // Transcribe using streaming API (same as working code)
            // Create temporary stream for this segment
            auto stream = transcriber->createStream(0.1f);  // Short update interval
            stream.start();

            // Feed the complete VAD segment
            stream.addAudio(audio_to_transcribe, 16000);

            // Force immediate transcription and stop
            auto transcript = stream.updateTranscription(moonshine::Transcriber::FLAG_FORCE_UPDATE);
            stream.stop();

            // Find the longest/best line in the transcript
            std::string best_text;
            for (const auto& line : transcript.lines) {
                if (!line.text.empty() && line.text.length() > best_text.length()) {
                    best_text = line.text;
                }
            }

            if (!best_text.empty()) {
                result.text = trimWhitespace(toLowerCase(best_text));
                result.duration = duration;
                result.is_complete = segment.is_complete;
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            result.latency = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

            // Record performance metrics
            metrics.recordInference(result.latency.count(), duration);

        } catch (const std::exception& e) {
            std::cerr << "Transcription error: " << e.what() << std::endl;
            result.text = "";
        }

        return result;
    }

    /**
     * Get completed transcriptions (ITranscriber interface)
     * Wraps getCompletedSegments() + transcribeSegment() to return unified results.
     *
     * @return Vector of completed transcription results
     */
    std::vector<ITranscriber::TranscriptionResult> getCompletedTranscriptions() override {
        auto segments = getCompletedSegments();
        std::vector<ITranscriber::TranscriptionResult> results;

        for (auto& segment : segments) {
            auto moonshineResult = transcribeSegment(segment);

            if (moonshineResult.text.empty()) {
                continue;
            }

            // Convert Moonshine result to ITranscriber format
            ITranscriber::TranscriptionResult result;
            result.text = moonshineResult.text;
            result.duration = moonshineResult.duration;
            result.latency = moonshineResult.latency;
            result.is_complete = moonshineResult.is_complete;

            // Pseudo-confidence based on duration (Moonshine doesn't provide confidence scores)
            // Map duration to 0.0-1.0: 0.5s=0.25, 1.0s=0.5, 2.0s+=1.0
            result.confidence = (std::min)(1.0f, moonshineResult.duration / 2.0f);

            results.push_back(std::move(result));
        }

        return results;
    }

    /**
     * Filter transcription results (ITranscriber interface)
     * Uses duration-based filtering since Moonshine doesn't provide confidence.
     *
     * @param result Transcription result to check
     * @return true if the result should be accepted
     */
    bool shouldAcceptTranscription(const ITranscriber::TranscriptionResult& result) const override {
        // Reject if text is empty
        if (result.text.empty()) {
            return false;
        }

        // Reject very short utterances (likely noise) - use 0.3s threshold
        if (result.duration < 0.3f) {
            return false;
        }

        // Reject very short text (single character, likely misrecognition)
        if (result.text.length() < 2) {
            return false;
        }

        // Accept valid transcriptions
        return true;
    }

    /**
     * Print performance statistics (ITranscriber interface)
     */
    void printPerformanceStats() const override {
        metrics.printStats();
    }

    /**
     * Get model name for logging (ITranscriber interface)
     *
     * @return Model name string
     */
    std::string getModelName() const override {
        return "Moonshine TINY";
    }

    /**
     * Reset performance statistics
     */
    void resetPerformanceStats() {
        metrics.reset();
    }

private:
    /**
     * Trim leading and trailing whitespace
     */
    static std::string trimWhitespace(const std::string& str) {
        size_t start = 0;
        size_t end = str.length();

        while (start < end && std::isspace(static_cast<unsigned char>(str[start]))) {
            ++start;
        }

        while (end > start && std::isspace(static_cast<unsigned char>(str[end - 1]))) {
            --end;
        }

        return str.substr(start, end - start);
    }

    /**
     * Convert string to lowercase
     */
    static std::string toLowerCase(const std::string& str) {
        std::string result = str;
        for (char& c : result) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return result;
    }
};

} // namespace moonshine

#endif // MOONSHINE_TRANSCRIBER_H
