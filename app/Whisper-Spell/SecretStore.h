#pragma once

// Protects small secrets at rest with the Windows Data Protection API (DPAPI).
//
// A value protected here can only be decrypted by the same Windows user on the
// same machine, so a config.ini copied elsewhere, or read by another account,
// does not reveal the secret. No key management is needed: Windows derives the
// key from the user's logon credentials.

#include <windows.h>
#include <wincrypt.h>
#include <string>
#include <vector>

#pragma comment(lib, "crypt32.lib")

namespace SecretStore {

namespace detail {

    // Extra entropy mixed into the key so that other DPAPI users running under
    // the same Windows account cannot decrypt our blobs by accident.
    inline DATA_BLOB entropy() {
        static const char kEntropy[] = "HogwartsLegacy-SpellCaster/crowdsourcing/v1";
        DATA_BLOB blob;
        blob.pbData = reinterpret_cast<BYTE*>(const_cast<char*>(kEntropy));
        blob.cbData = static_cast<DWORD>(sizeof(kEntropy) - 1);
        return blob;
    }

    inline std::string toBase64(const BYTE* data, DWORD size) {
        const DWORD flags = CRYPT_STRING_BASE64 | CRYPT_STRING_NOCRLF;
        DWORD length = 0;  // includes the terminating NUL on the sizing call
        if (!CryptBinaryToStringA(data, size, flags, nullptr, &length) || length == 0) {
            return {};
        }
        std::string out(length, '\0');
        if (!CryptBinaryToStringA(data, size, flags, &out[0], &length)) {
            return {};
        }
        out.resize(std::char_traits<char>::length(out.c_str()));  // drop the NUL, whatever the API counted
        return out;
    }

    inline std::vector<BYTE> fromBase64(const std::string& text) {
        DWORD size = 0;
        if (!CryptStringToBinaryA(text.c_str(), static_cast<DWORD>(text.size()),
                                  CRYPT_STRING_BASE64, nullptr, &size, nullptr, nullptr) || size == 0) {
            return {};
        }
        std::vector<BYTE> out(size);
        if (!CryptStringToBinaryA(text.c_str(), static_cast<DWORD>(text.size()),
                                  CRYPT_STRING_BASE64, out.data(), &size, nullptr, nullptr)) {
            return {};
        }
        out.resize(size);
        return out;
    }

} // namespace detail

/// Overwrites the contents of a string before releasing it.
inline void wipe(std::string& secret) {
    if (!secret.empty()) {
        SecureZeroMemory(&secret[0], secret.size());
    }
    secret.clear();
}

/// Encrypts `plaintext` for the current Windows user and returns it as base64.
/// Returns an empty string for an empty input or if DPAPI is unavailable.
inline std::string protect(const std::string& plaintext) {
    if (plaintext.empty()) {
        return {};
    }

    DATA_BLOB in;
    in.pbData = reinterpret_cast<BYTE*>(const_cast<char*>(plaintext.data()));
    in.cbData = static_cast<DWORD>(plaintext.size());
    DATA_BLOB entropy = detail::entropy();
    DATA_BLOB out = {};

    if (!CryptProtectData(&in, L"SpellCaster crowdsourcing credential", &entropy,
                          nullptr, nullptr, CRYPTPROTECT_UI_FORBIDDEN, &out)) {
        return {};
    }

    std::string encoded = detail::toBase64(out.pbData, out.cbData);
    SecureZeroMemory(out.pbData, out.cbData);
    LocalFree(out.pbData);
    return encoded;
}

/// Decrypts a value produced by protect(). Returns false if the value is
/// malformed or was protected by another Windows user or on another machine.
inline bool unprotect(const std::string& encoded, std::string& outPlaintext) {
    wipe(outPlaintext);

    std::vector<BYTE> blob = detail::fromBase64(encoded);
    if (blob.empty()) {
        return false;
    }

    DATA_BLOB in;
    in.pbData = blob.data();
    in.cbData = static_cast<DWORD>(blob.size());
    DATA_BLOB entropy = detail::entropy();
    DATA_BLOB out = {};

    if (!CryptUnprotectData(&in, nullptr, &entropy, nullptr, nullptr,
                            CRYPTPROTECT_UI_FORBIDDEN, &out)) {
        return false;
    }

    outPlaintext.assign(reinterpret_cast<const char*>(out.pbData), out.cbData);
    SecureZeroMemory(out.pbData, out.cbData);
    LocalFree(out.pbData);
    return true;
}

} // namespace SecretStore
