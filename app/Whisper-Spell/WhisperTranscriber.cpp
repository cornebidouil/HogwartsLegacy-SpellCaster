#include "WhisperTranscriber.h"
#include <algorithm>
#include <numeric>
#include <cmath>

/**
 * WhisperTranscriber Implementation
 *
 * Uses Whisper.cpp for speech recognition and TEN VAD for voice activity detection
 */

WhisperTranscriber::WhisperTranscriber(
    const std::string& modelPath,
    float vadThreshold,
    int sampleRate,
    float confidenceThreshold,
    bool useGPU
) : ctx_(nullptr, whisper_free),
    lastProcessedSegmentIndex_(0),
    sampleRate_(sampleRate),
    confidenceThreshold_(confidenceThreshold)
{
    try {
        // Initialize Whisper context with GPU support
        struct whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = useGPU;

        ctx_.reset(whisper_init_from_file_with_params(modelPath.c_str(), cparams));

        if (!ctx_) {
            throw std::runtime_error("Failed to load Whisper model from: " + modelPath);
        }

        // Initialize TEN VAD (same as Moonshine)
        vad_ = std::make_unique<VoiceActivityDetector>(
            vadThreshold,   // threshold
            256,            // hop_size (256 samples = 16ms at 16kHz)
            32,             // window_size (32 frames)
            8192,           // look_behind_sample_count (512ms at 16kHz)
            15 * 16000      // max_segment_sample_count (15 seconds)
        );

        if (!vad_) {
            throw std::runtime_error("Failed to initialize TEN VAD");
        }

        // Start VAD
        vad_->start();

        std::cout << "WhisperTranscriber initialized successfully" << std::endl;
        std::cout << "  Model: Whisper" << std::endl;
        std::cout << "  VAD: TEN (threshold: " << vadThreshold << ")" << std::endl;
        std::cout << "  GPU Acceleration: " << (useGPU ? "Enabled" : "Disabled") << std::endl;
        std::cout << "  Sample Rate: " << sampleRate << " Hz" << std::endl;
        std::cout << "  Confidence Threshold: " << confidenceThreshold << std::endl;

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to initialize WhisperTranscriber: ") + e.what());
    }
}

WhisperTranscriber::~WhisperTranscriber() {
    if (vad_) {
        vad_->stop();
    }
    // Other cleanup handled by smart pointers
}

void WhisperTranscriber::processAudio(const float* audioData, size_t numFrames) {
    if (!audioData || !vad_) {
        return;
    }

    std::lock_guard<std::mutex> lock(vadMutex_);

    try {
        // Feed audio incrementally to TEN VAD (VAD maintains internal state)
        vad_->process_audio(audioData, numFrames, sampleRate_);

        // Get newly completed speech segments
        const auto* segments = vad_->get_segments();
        if (!segments) {
            return;
        }

        // Process only NEW completed segments
        for (size_t i = lastProcessedSegmentIndex_; i < segments->size(); i++) {
            const auto& segment = (*segments)[i];
            if (segment.is_complete) {
                // Transcribe the segment asynchronously
                auto result = transcribeSegment(segment);

                // Add to completed queue if non-empty
                if (!result.text.empty()) {
                    std::lock_guard<std::mutex> lock(transcriptionMutex_);
                    completedTranscriptions_.push(std::move(result));
                }

                lastProcessedSegmentIndex_ = i + 1;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[WhisperTranscriber] VAD error: " << e.what() << std::endl;
    }
}

ITranscriber::TranscriptionResult WhisperTranscriber::transcribeSegment(const VoiceActivitySegment& segment) {
    ITranscriber::TranscriptionResult result;

    auto startTime = std::chrono::high_resolution_clock::now();

    try {
        if (segment.audio_data.empty()) {
            return result;
        }

        float duration = segment.end_time - segment.start_time;
        result.duration = duration;

        const auto& audioData = segment.audio_data;

        std::lock_guard<std::mutex> lock(ctxMutex_);

        // Prepare Whisper parameters
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.print_realtime   = false;
        wparams.print_progress   = false;
        wparams.print_timestamps = false;
        wparams.print_special    = false;
        wparams.translate        = false;
        wparams.single_segment   = true;
        wparams.language         = "en";
        wparams.n_threads        = (std::min)(4, (int)std::thread::hardware_concurrency());

        // Whisper requires minimum 1 second of audio
        const int MIN_AUDIO_LENGTH = sampleRate_ + 512;
        std::vector<float> processedAudio = audioData;

        if (processedAudio.size() < MIN_AUDIO_LENGTH) {
            // Pad with silence
            processedAudio.resize(MIN_AUDIO_LENGTH, 0.0f);
        }

        // Run Whisper transcription
        if (whisper_full(ctx_.get(), wparams, processedAudio.data(), processedAudio.size()) != 0) {
            std::cerr << "Whisper transcription error" << std::endl;
            return result;
        }

        // Extract transcription and compute confidence
        const int n_segments = whisper_full_n_segments(ctx_.get());
        std::string transcription;
        float totalConfidence = 0.0f;

        for (int i = 0; i < n_segments; ++i) {
            const char* text = whisper_full_get_segment_text(ctx_.get(), i);
            transcription += text;

            // Compute confidence from token probabilities
            float segmentConfidence = computeConfidence(ctx_.get(), i);
            totalConfidence += segmentConfidence;
        }

        // Average confidence across segments
        if (n_segments > 0) {
            result.confidence = totalConfidence / n_segments;
        }

        // Clean up transcription text
        transcription.erase(std::remove(transcription.begin(), transcription.end(), '.'), transcription.end());

        // Trim whitespace
        size_t start = 0;
        size_t end = transcription.length();
        while (start < end && std::isspace(static_cast<unsigned char>(transcription[start]))) {
            ++start;
        }
        while (end > start && std::isspace(static_cast<unsigned char>(transcription[end - 1]))) {
            --end;
        }
        transcription = transcription.substr(start, end - start);

        // Convert to lowercase
        std::transform(transcription.begin(), transcription.end(), transcription.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        result.text = transcription;
        result.is_complete = true;

        auto endTime = std::chrono::high_resolution_clock::now();
        result.latency = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        // Record performance metrics
        metrics_.recordInference(result.latency.count(), duration);

    } catch (const std::exception& e) {
        std::cerr << "Transcription error: " << e.what() << std::endl;
        result.text = "";
    }

    return result;
}

float WhisperTranscriber::computeConfidence(whisper_context* ctx, int segmentIndex) const {
    const int n_tokens = whisper_full_n_tokens(ctx, segmentIndex);

    if (n_tokens == 0) {
        return 0.0f;
    }

    // Compute average token probability
    // whisper_full_get_token_p returns log probability
    float sum_exp_probs = 0.0f;

    for (int i = 0; i < n_tokens; ++i) {
        float log_prob = whisper_full_get_token_p(ctx, segmentIndex, i);
        sum_exp_probs += std::exp(log_prob);
    }

    // Normalize by number of tokens
    return sum_exp_probs / n_tokens;
}

std::vector<ITranscriber::TranscriptionResult> WhisperTranscriber::getCompletedTranscriptions() {
    std::lock_guard<std::mutex> lock(transcriptionMutex_);

    std::vector<ITranscriber::TranscriptionResult> results;

    // Drain the queue
    while (!completedTranscriptions_.empty()) {
        results.push_back(std::move(completedTranscriptions_.front()));
        completedTranscriptions_.pop();
    }

    return results;
}

bool WhisperTranscriber::shouldAcceptTranscription(const ITranscriber::TranscriptionResult& result) const {
    // Reject empty text
    if (result.text.empty()) {
        return false;
    }

    // Reject very short text (likely noise)
    if (result.text.length() < 2) {
        return false;
    }

    // Reject low confidence transcriptions
    if (result.confidence < confidenceThreshold_) {
        return false;
    }

    // Accept valid transcriptions
    return true;
}

void WhisperTranscriber::printPerformanceStats() const {
    metrics_.printStats();
}

std::string WhisperTranscriber::getModelName() const {
    return "Whisper";
}
