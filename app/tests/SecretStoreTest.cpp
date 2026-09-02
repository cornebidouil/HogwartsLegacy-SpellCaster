// Round-trip test for SecretStore (DPAPI protection of the crowdsourcing password).
//
// Build and run from a Visual Studio developer prompt:
//   cl /nologo /EHsc /W4 /std:c++20 /utf-8 /I..\Whisper-Spell SecretStoreTest.cpp && SecretStoreTest.exe
//
// Exit code is the number of failed checks.

#include "SecretStore.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        ++failures;
    }
}

} // namespace

int main() {
    const std::string secret = "expelliarmus 123 \xC3\xA9 !\"\\";  // spaces, UTF-8, quotes, backslash

    const std::string encoded = SecretStore::protect(secret);
    check(!encoded.empty(), "protect returns a value");
    check(encoded.find_first_of("\r\n \t;") == std::string::npos,
          "value fits on one INI line without whitespace or comment characters");
    check(encoded.find('\0') == std::string::npos, "no embedded or trailing NUL");
    std::printf("      protected value length: %zu characters\n", encoded.size());

    std::string decoded;
    check(SecretStore::unprotect(encoded, decoded) && decoded == secret,
          "round trip restores the exact bytes");

    std::string tampered = encoded;
    tampered[tampered.size() / 2] = (tampered[tampered.size() / 2] == 'A') ? 'B' : 'A';
    std::string out = "stale";
    check(!SecretStore::unprotect(tampered, out) && out.empty(),
          "tampered value is rejected and the output cleared");
    check(!SecretStore::unprotect("not base64 at all!!", out), "garbage is rejected");
    check(!SecretStore::unprotect("", out), "empty value is rejected");
    check(SecretStore::protect("").empty(), "empty secret yields an empty value");

    const std::string encodedAgain = SecretStore::protect(secret);
    check(encodedAgain != encoded, "protecting twice gives different values");
    check(SecretStore::unprotect(encodedAgain, decoded) && decoded == secret,
          "second value also round trips");

    std::string toWipe = "wipe me";
    SecretStore::wipe(toWipe);
    check(toWipe.empty(), "wipe clears the string");

    std::printf("%s\n", failures ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED");
    return failures;
}
