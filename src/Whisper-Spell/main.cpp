#include <iostream>
#include <vector>
#include <portaudio.h>
#include <whisper.h>
//#include <torch/script.h> // One of the headers that include PyTorch C++ API
#include <algorithm>
#include <regex>
#include <numeric>
#include <cmath>
#include <limits>
#include <mutex>
#include <thread>
#include <chrono>

#include "SileroVAD.h"
#include "Keybinder.h"
#include "Tools.h"
#include "cuda_tools.h"



#define SAMPLE_RATE 16000
#define FRAMES_PER_BUFFER 512
#define SPEECH_THRESHOLD 0.5


// Define the structure for token data
/*struct whisper_token_data {
    int n_logits;
    float* logits;
};*/

// Global variables for audio processing
//std::vector<float> audioBuffer;
std::vector<float> speechBuffer;
std::vector<float> vadBuffer; //(7 * FRAMES_PER_BUFFER, 0.0f);
bool isSpeaking = false;
whisper_context* ctx;
VadIterator* vad;
Keybinder* keys;
std::mutex bufferMutex;


HMODULE whisperLib = nullptr;
typedef whisper_context* (*whisper_init_from_file_with_params_func)(const char* path_model, whisper_context_params params);
typedef int (*whisper_full_func)(struct whisper_context* ctx, struct whisper_full_params params, const float* samples, int n_samples);



// Load the .ggml model using Whisper.cpp
bool loadWhisperModel(const std::string& modelPath) {
    
    auto init_func = (whisper_init_from_file_with_params_func)GetProcAddress(whisperLib, "whisper_init_from_file_with_params");
    if (!init_func) {
        std::cerr << "Failed to get whisper_init_from_file_with_params function" << std::endl;
        FreeLibrary(whisperLib);
        return false;
    }

    struct whisper_context_params cparams = whisper_context_default_params();
    //cparams.use_gpu = false;
    ctx = init_func(modelPath.c_str(), cparams); // whisper_init_from_file_with_params 
    //ctx = whisper_init_from_file_with_params(modelPath.c_str(), cparams);

    //ctx = whisper_init_from_file(modelPath.c_str());
    return ctx != nullptr;
}

// Function to compute confidence
float compute_confidence(const whisper_token_data* tokens, int n_tokens) {

    std::vector<float> log_probs;

    std::cout << "Log probabilities:" << std::endl;
    for (int i = 0; i < n_tokens; ++i) {
        const auto& token = tokens[i];
        log_probs.push_back(token.plog);
        std::cout << "Token " << i << ": " << token.plog << std::endl;
    }

    float sum_exp_probs = 0.0f;
    std::cout << "\nExponentiated probabilities:" << std::endl;
    for (int i = 0; i < n_tokens; ++i) {
        float exp_prob = std::exp(log_probs[i]);
        sum_exp_probs += exp_prob;
        std::cout << "Token " << i << ": " << exp_prob << std::endl;
    }

    std::cout << "\nSum of exponentiated probabilities: " << sum_exp_probs << std::endl;

    float confidence = sum_exp_probs / n_tokens;
    std::cout << "Final confidence score: " << confidence << std::endl;

    return confidence;

    /*float sum_exp_probs = std::accumulate(log_probs.begin(), log_probs.end(), 0.0f,
        [](float acc, float val) { return acc + std::exp(val); });

    return sum_exp_probs / n_tokens;*/
}

// Transcribe function using the Whisper model
std::pair<std::string, float> transcribeAudio(const std::vector<float>& audioData, int sampleRate) {
    auto full_func = (whisper_full_func)GetProcAddress(whisperLib, "whisper_full");
    if (!full_func) {
        return { "Failed to get whisper_full function" , 0.0f };
    }
    
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = false;
    wparams.print_special    = false;
    wparams.translate        = false;
    wparams.single_segment   = true;
    wparams.language         = "en";
    wparams.n_threads        = min(4, (int32_t)std::thread::hardware_concurrency());  //12;


    const int MIN_AUDIO_LENGTH = SAMPLE_RATE + 512;

    if (audioData.size() < MIN_AUDIO_LENGTH) {
        // Calculate how many samples of silence to add
        int padding_samples = MIN_AUDIO_LENGTH - audioData.size();
        
        // Create a padded version of the audio data
        std::vector<float> paddedAudio = audioData;
        paddedAudio.resize(MIN_AUDIO_LENGTH, 0.0f); // Pad with zeros (silence)

        if (full_func(ctx, wparams, paddedAudio.data(), paddedAudio.size()) != 0) { //whisper_full
            return {"Error in transcription", 0.0f};
        }
    } else {
        if (full_func(ctx, wparams, audioData.data(), audioData.size()) != 0) { //whisper_full
            return {"Error in transcription", 0.0f};
        }
    }


    const int n_segments = whisper_full_n_segments(ctx);
    std::string transcription;
    float confidence = 0.0f;

    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        transcription += text;

        const int n_tokens = whisper_full_n_tokens(ctx, i);
        /*std::vector<whisper_token_data> tokens(n_tokens);
        for (int j = 0; j < n_tokens; ++j) {
            whisper_token_data token = whisper_full_get_token_data(ctx, i, j);
            tokens[j] = token;
        }
        confidence += compute_confidence(tokens.data(), n_tokens);*/
        
        //std::cout << "Segment " << i << std::endl;

        float proba = 0.0f;
        for (int j = 0; j < n_tokens; ++j) {
            proba += whisper_full_get_token_p(ctx, i, j);

            /*whisper_token token_id = whisper_full_get_token_id(ctx, i, j);
            const char* token_text = whisper_token_to_str(ctx, token_id);
            std::cout << "Token ID: " << token_id << ", Text: " << token_text << std::endl;*/
        }
        confidence += proba / n_tokens;

    }

    confidence /= n_segments;
    
    transcription.erase(std::remove(transcription.begin(), transcription.end(), '.'), transcription.end());

    return {transcription, confidence};
}

// Callback function for PortAudio
static int paCallback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData) {
    
    const float* in = static_cast<const float*>(inputBuffer);
    {
        // Append audio data to the buffer
        std::lock_guard<std::mutex> lock(bufferMutex);
        //audioBuffer.insert(audioBuffer.end(), in, in + framesPerBuffer);
        vadBuffer.insert(vadBuffer.end(), in, in + framesPerBuffer);
    }
    

    const size_t maxVadBufferSize = 7 * FRAMES_PER_BUFFER;
    if (vadBuffer.size() > maxVadBufferSize) {
        vadBuffer.erase(vadBuffer.begin(), vadBuffer.end() - maxVadBufferSize);
    }

    // Process VAD and other logic
    // Assuming you have VAD function and logic here
    // std::vector<SpeechTimestamp> speechTimestamps = getSpeechTimestamps(audioBuffer, SAMPLE_RATE);
    

    //float speech_level = 0.0f;
    bool is_speech = false;
    try {
        vad->process(vadBuffer); // std::vector<float>(in, in + framesPerBuffer)
        //vad->process(std::vector<float>(in, in + framesPerBuffer));

        const auto& speech_timestamps = vad->get_speech_timestamps();
        is_speech = !speech_timestamps.empty();

        /*std::vector<timestamp_t> stamps = vad->get_speech_timestamps();
        for (int i = 0; i < stamps.size(); i++) {
            std::cout << stamps[i].c_str() << std::endl;
        }*/

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }

    //std::cout << "Frame per buffer: " << framesPerBuffer << std::endl;
    //std::cout << "Speech level: " << vad->get_speech_prob() << " (" << is_speech << ")" << std::endl;
    

    if (is_speech) { // speech_level > SPEECH_THRESHOLD
        if (!isSpeaking) {
            isSpeaking = true;
            speechBuffer.clear();
            speechBuffer.insert(speechBuffer.end(), vadBuffer.begin(), vadBuffer.end());
            keys->start();
        }
        else {
            speechBuffer.insert(speechBuffer.end(), in, in + framesPerBuffer);
        }
    } else {
        if (isSpeaking) {
            isSpeaking = false;

            // Transcribe the collected speech
            std::vector<float> audioChunk; //(speechBuffer.begin(), speechBuffer.end());
            {
                std::lock_guard<std::mutex> lock(bufferMutex);
                audioChunk = std::move(speechBuffer);
                speechBuffer.clear();
            }
            

            // Transcribe using Whisper model
            std::async(std::launch::async, [audioChunk = std::move(audioChunk)]() { //[this, ...]
                auto start = std::chrono::high_resolution_clock::now();
                auto [transcription, confidence] = transcribeAudio(audioChunk, SAMPLE_RATE);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;

                //if (transcription.size() != 0)
                //std::cout << "Transcription: " << transcription << " (Confidence: " << confidence << ")" << " | t: " << elapsed.count() << "s" << std::endl;

                if (confidence > 0.7f && !transcription.empty()) {
                    std::cout << "Transcription: " << transcription << " (Confidence: " << confidence << ")" << " | t: " << elapsed.count() << "s" << std::endl;
                    keys->decode(transcription, true);
                    auto full_end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> full_elapsed = full_end - end; //start
                    std::cout << "Input performance, t: " << full_elapsed.count() << "s" << std::endl;
                } else {
                    keys->stop();
                }
            });
            
            // Implement any post-transcription logic (e.g., command execution)

            //speechBuffer.clear();
            
        }
    }

    return paContinue;
}

int main() {

    const ORTCHAR_T* model_path = ORT_TSTR("models/silero_vad.onnx");
    vad = new VadIterator(
        model_path,         // Model Path
        SAMPLE_RATE,        // Sample rate
        32,                 // Frame in ms (adjust as needed) = 20   
        SPEECH_THRESHOLD,   // Detection threshold = 0.5f     
        0,                  // Min Silence Duration
        32                  // Speech pad ms
    );

    std::cout << std::endl << std::endl;
    std::cout << "\t*********************************************" << std::endl;
    std::cout << "\t***                                       ***" << std::endl;
    std::cout << "\t***    Program created by Cornebidouil    ***" << std::endl;
    std::cout << "\t***    https://github.com/cornebidouil    ***" << std::endl;
    std::cout << "\t***                                       ***" << std::endl;
    std::cout << "\t*********************************************" << std::endl << std::endl << std::endl;

    std::cout << "GPU drivers updating is highly recommended" << std::endl;
    //std::cout << "NVIDIA card: " << hasNvidiaGPU() << std::endl;
    //whisperLib = hasNvidiaGPU() ? LoadLibraryA("whisper_cuda.dll") : LoadLibraryA("whisper.dll");
    switch (detectGPU()) {
        case GPUVendor::NVIDIA_CUDA :
            if (isCudaCompatible()) {
                whisperLib = LoadLibraryA("whisper_cuda.dll");
                std::cout << "NVIDIA card detected with CUDA compatibility.";
            }
            else {
                whisperLib = LoadLibraryA("whisper_clblast.dll");
                std::cout << "NVIDIA card detected without CUDA compatibility. Using CLBlast.";
            }
            break;
        case GPUVendor::AMD_ROCM :
            whisperLib = LoadLibraryA("whisper_hipblas.dll");
            std::cout << "AMD card detected with hipBLAS compatibility.";
            break;
        case GPUVendor::DEFAULT_GPU :
            whisperLib = LoadLibraryA("whisper_clblast.dll");
            std::cout << "Graphic card detected. Using CLBlast.";
            break;
        case GPUVendor::AMD_DEFAULT :
            whisperLib = LoadLibraryA("whisper_clblast.dll");
            std::cout << "AMD card detected without hipBLAS compatibility. Using CLBlast.";
            break;
        case GPUVendor::NVIDIA_DEFAULT :
            whisperLib = LoadLibraryA("whisper_clblast.dll");
            std::cout << "NVIDIA card detected without CUDA compatibility. Using CLBlast.";
            break;
        default :
            whisperLib = LoadLibraryA("whisper_openblas.dll");
            std::cout << "No compatible graphic card detected."; 
            break;
    }
    std::cout << std::endl << std::endl;
    //whisperLib = LoadLibraryA("whisper-openblas.dll");

    PaError err;

    keys = new Keybinder("keybinding.txt");

    // Load the Whisper model
    std::string modelPath = "models/ggml-model.bin";
    if (!loadWhisperModel(modelPath)) {
        std::cerr << "Failed to load Whisper model from " << modelPath << std::endl;
        return 1;
    }

    //std::cout << "Vocab size: " << whisper_model_n_vocab(ctx) << " " << whisper_is_multilingual(ctx) << std::endl;
    //whisper_token token_test = 51864;
    //std::cout << whisper_token_to_str(ctx, token_test) << std::endl;

    // Initialize PortAudio
    err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return 1;
    }

    /*
    // Find ASIO host API
    PaHostApiIndex asioApiIndex = Pa_HostApiTypeIdToHostApiIndex(paASIO);
    if (asioApiIndex == paHostApiNotFound) {
        std::cerr << "ASIO host API not found!" << std::endl;
        Pa_Terminate();
        return -1;
    }

    const PaHostApiInfo* asioApiInfo = Pa_GetHostApiInfo(asioApiIndex);
    std::cout << "Using ASIO host API: " << asioApiInfo->name << std::endl;

    const PaDeviceInfo* defaultDeviceInfo = Pa_GetDeviceInfo(Pa_GetDefaultInputDevice());
    std::cout << "Default: " << defaultDeviceInfo->name << " -> " << defaultDeviceInfo->hostApi << std::endl;

    // List all available ASIO devices
    std::cout << std::endl << "Available ASIO devices:" << std::endl;
    int numDevices = Pa_GetDeviceCount();
    std::vector<int> asioDeviceIndexes;

    for (int i = 0; i < numDevices; ++i) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        std::cout << "\t" << i << ": " << deviceInfo->name; //<< std::endl;
        if (deviceInfo->maxInputChannels > 0) { //deviceInfo->hostApi == asioApiIndex
            //std::cout << "\t" << asioDeviceIndexes.size() << ": " << deviceInfo->name << std::endl;
            std::cout << " -> " << deviceInfo->hostApi;
            asioDeviceIndexes.push_back(i);
        }
        std::cout << std::endl;
    }

    // Let the user select an ASIO device
    int userSelection = -1;
    std::cout << std::endl << "Enter the index of the ASIO device you want to use: ";
    std::cin >> userSelection;

    // Validate the device index
    if (userSelection < 0 || userSelection >= asioDeviceIndexes.size()) {
        std::cerr << "Invalid device index." << std::endl;
        Pa_Terminate();
        return 1;
    }

    int asioDeviceIndex = asioDeviceIndexes[userSelection];
    const PaDeviceInfo* selectedDeviceInfo = Pa_GetDeviceInfo(asioDeviceIndex);
    if (selectedDeviceInfo->hostApi != asioApiIndex) {
        std::cerr << "Selected device is not an ASIO device." << std::endl;
        Pa_Terminate();
        return 1;
    }*/

   // List all available input devices
    int numDevices = Pa_GetDeviceCount();
    std::cout << std::endl << std::endl << "Available input devices:" << std::endl << std::endl;
    std::vector<int> inputDeviceIndexes;

    for (int i = 0; i < numDevices; ++i) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
        if (deviceInfo->maxInputChannels > 0) {
            /*std::cout << inputDeviceIndexes.size() << ": " << deviceInfo->name 
                      << " (API: " << Pa_GetHostApiInfo(deviceInfo->hostApi)->name << ")" << std::endl;
            inputDeviceIndexes.push_back(i);*/

            // Set up test parameters
            PaStreamParameters testParams;
            testParams.device = i;
            testParams.channelCount = 1;  // mono
            testParams.sampleFormat = paFloat32;
            testParams.suggestedLatency = deviceInfo->defaultLowInputLatency;
            testParams.hostApiSpecificStreamInfo = NULL;

            // Test if we can open a stream with this device
            err = Pa_IsFormatSupported(&testParams, NULL, SAMPLE_RATE);
            if (err == paFormatIsSupported) {
                std::cout << inputDeviceIndexes.size() << ": " << deviceInfo->name
                    << " (API: " << Pa_GetHostApiInfo(deviceInfo->hostApi)->name << ")" << std::endl;
                inputDeviceIndexes.push_back(i);
            }
        }
    }


    /*std::cout << std::endl;
    for (int i = 0; i < inputDeviceIndexes.size(); ++i) {
        std::cout << "inputDeviceIndexes[" << i << "] = " << inputDeviceIndexes[i] << std::endl;
    }*/

    // Let the user select an input device
    int userSelection = -1;
    std::cout << std::endl << "Enter the index of the input device you want to use (latency WASAPI > MME > DirectSound): ";
    std::cin >> userSelection;

    // Validate the device index
    if (userSelection < 0 || userSelection >= inputDeviceIndexes.size()) {
        std::cerr << "Invalid device index." << std::endl;
        Pa_Terminate();
        return 1;
    }

    int selectedDeviceIndex = inputDeviceIndexes[userSelection];
    const PaDeviceInfo* selectedDeviceInfo = Pa_GetDeviceInfo(selectedDeviceIndex);

    std::cout << std::endl << "Selected device information:" << std::endl;
    std::cout << "Name: " << selectedDeviceInfo->name << std::endl;
    std::cout << "Max input channels: " << selectedDeviceInfo->maxInputChannels << std::endl;
    std::cout << "Default sample rate: " << selectedDeviceInfo->defaultSampleRate << " Hz" << std::endl;
    std::cout << "Default low input latency: " << selectedDeviceInfo->defaultLowInputLatency << " seconds" << std::endl;
    std::cout << "Default high input latency: " << selectedDeviceInfo->defaultHighInputLatency << " seconds" << std::endl;
    std::cout << "Host API: " << Pa_GetHostApiInfo(selectedDeviceInfo->hostApi)->name << std::endl << std::endl;



    // Use the ASIO host API to open a stream or query devices
    PaStreamParameters inputParameters;
    inputParameters.device = selectedDeviceIndex;                                  //Pa_GetDefaultInputDevice()
    inputParameters.channelCount = 1;                                              // mono input
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = selectedDeviceInfo->defaultLowInputLatency; //Pa_GetDeviceInfo(inputParameters.device)->defaultLowOutputLatency;
    inputParameters.hostApiSpecificStreamInfo = NULL;



    // Open an audio I/O stream
    PaStream* stream;
    /*err = Pa_OpenDefaultStream(&stream, 1, 0, paFloat32, SAMPLE_RATE, FRAMES_PER_BUFFER, paCallback, nullptr);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return 1;
    }*/
    err = Pa_OpenStream(&stream, &inputParameters, NULL, SAMPLE_RATE, FRAMES_PER_BUFFER, paClipOff, paCallback, nullptr);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        std::cerr << "Failed to open stream for device: " << selectedDeviceInfo->name << std::endl;
        Pa_Terminate();
        return -1;
    }

    // Start the stream
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return 1;
    }

    const PaStreamInfo* streamInfo = Pa_GetStreamInfo(stream);
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(inputParameters.device);

    std::cout << std::endl << "Input device: " << deviceInfo->name << std::endl;
    std::cout << "Input latency: " << streamInfo->inputLatency << " seconds" << std::endl;
    std::cout << "Sample rate: " << streamInfo->sampleRate << " Hz" << std::endl << std::endl;

    // --- Find Whisper-Spell.exe path
    //
    TCHAR buffer[MAX_PATH];
    GetCurrentDirectory(MAX_PATH, buffer);
    #ifdef UNICODE
    std::wstring wPath(buffer);
    std::string exePath(wPath.begin(), wPath.end());
    #else
    std::string exePath(buffer);
    #endif

    // --- Write keybinding recommendations
    //
    std::cout << std::endl << "Binding Recommendations : " << std::endl << std::endl;
    std::cout << "\tYour binding file is located here : " << exePath+"\\keybinding.txt"<< std::endl << std::endl;
    std::cout << "\tDefault configuration :" << std::endl << std::endl;
    std::cout << "\t\tnone;none;none;none" << std::endl;
    std::cout << "\t\tnone;none;none;none" << std::endl;
    std::cout << "\t\tnone;none;none;none" << std::endl;
    std::cout << "\t\tnone;none;none;none" << std::endl << std::endl;
    std::cout << "\tFill the \"none\" with the spell name associated to the spellbar (without uppercase)." << std::endl; 
    std::cout << "\tExample : you have lumos and accio as the first 2 spells of your 1rst spellbar loadout" << std::endl << std::endl;
    std::cout << "\t\tlumos;accio;none;none" << std::endl;
    std::cout << "\t\tnone;none;none;none" << std::endl;
    std::cout << "\t\tnone;none;none;none" << std::endl;
    std::cout << "\t\tnone;none;none;none" << std::endl << std::endl;

    std::cout << std::endl << std::endl;

    std::cout << "Listening... Press Ctrl+C to stop." << std::endl << std::endl;

    // Keep the stream active
    while (true) {
        Pa_Sleep(1000);

        // Potential processing here
    }

    // Stop the stream
    err = Pa_StopStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return 1;
    }

    // Close the stream
    err = Pa_CloseStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return 1;
    }

    // Terminate PortAudio
    Pa_Terminate();

    // Free whisper library
    whisper_free(ctx);
    FreeLibrary(whisperLib);

    // Clear vad model
    delete vad;
    delete keys;

    return 0;
}
