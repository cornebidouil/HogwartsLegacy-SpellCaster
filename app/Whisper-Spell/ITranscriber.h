#ifndef ITRANSCRIBER_H
#define ITRANSCRIBER_H

#include <string>
#include <vector>
#include <chrono>

/**
 * ITranscriber - Abstract interface for speech recognition engines
 *
 * This interface provides a unified API for different transcription models
 * (Whisper, Moonshine, etc.) allowing runtime selection via configuration.
 *
 * Implementation Pattern:
 * - processAudio(): Feed audio incrementally (called from audio callback)
 * - getCompletedTranscriptions(): Poll for completed transcriptions (non-blocking)
 * - shouldAcceptTranscription(): Filter low-quality results
 * - printPerformanceStats(): Display model-specific metrics
 */
class ITranscriber {
public:
    /**
     * Unified transcription result format
     * All implementations must return results in this format
     */
    struct TranscriptionResult {
        std::string text;                      // Transcribed text (lowercase, trimmed)
        float confidence;                      // Confidence score 0.0-1.0
                                               // - Whisper: Token probability average
                                               // - Moonshine: Duration-based pseudo-confidence
        float duration;                        // Audio duration in seconds
        std::chrono::milliseconds latency;     // Transcription time (ms)
        bool is_complete;                      // Whether segment is complete

        TranscriptionResult()
            : confidence(0.0f), duration(0.0f), latency(0), is_complete(false) {}
    };

    virtual ~ITranscriber() = default;

    /**
     * Process audio incrementally (called from audio callback)
     *
     * Feed audio samples to the transcriber's internal VAD and buffer.
     * This method should be non-blocking and thread-safe.
     *
     * @param audioData Float audio samples (mono, 16kHz)
     * @param numFrames Number of samples
     */
    virtual void processAudio(const float* audioData, size_t numFrames) = 0;

    /**
     * Get completed transcriptions (non-blocking poll)
     *
     * Returns all transcriptions that have completed since the last call.
     * Results are removed from internal queue after being returned.
     *
     * @return Vector of completed transcription results
     */
    virtual std::vector<TranscriptionResult> getCompletedTranscriptions() = 0;

    /**
     * Filter transcription results based on quality metrics
     *
     * Implementations should check model-specific quality indicators:
     * - Whisper: Confidence threshold, text length
     * - Moonshine: Duration threshold, text length
     *
     * @param result Transcription result to evaluate
     * @return true if the result should be accepted
     */
    virtual bool shouldAcceptTranscription(const TranscriptionResult& result) const = 0;

    /**
     * Print performance statistics
     *
     * Display model-specific metrics such as:
     * - Average/min/max inference time
     * - Real-time factor
     * - Total transcriptions processed
     */
    virtual void printPerformanceStats() const = 0;

    /**
     * Get model name for logging
     *
     * @return Human-readable model name (e.g., "Whisper TINY", "Moonshine TINY")
     */
    virtual std::string getModelName() const = 0;
};

#endif // ITRANSCRIBER_H
