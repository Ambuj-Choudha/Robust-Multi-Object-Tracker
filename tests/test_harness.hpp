// Minimal check/report harness shared by the tests.
//
// Deliberately not a framework: a test is a main() that returns 0 or 1, so it
// runs under ctest, under gdb, or straight from a shell with no setup. If the
// suite ever outgrows this, that is the point to bring in Catch2 - not before.
#pragma once

#include <iostream>
#include <string>
#include <string_view>
#include <variant>

#include "common/status.hpp"

namespace test {

// Accumulates failures instead of returning on the first one: when a change
// breaks an invariant it is useful to see every consequence at once, rather
// than fixing and re-running to find the next.
class Checks {
    public:
        explicit Checks(std::string_view suite) : suite_{suite} {}

        void check(bool passed, std::string_view what) {
            if (passed) {
                std::cout << "  ok   " << what << '\n';
                return;
            }
            std::cout << "  FAIL " << what << '\n';
            ++failures_;
        }

        // Reports the values on mismatch; a bare "expected X" is rarely enough
        // to tell what went wrong.
        template <typename T, typename U>
        void check_eq(const T& actual, const U& expected, std::string_view what) {
            if (actual == expected) {
                std::cout << "  ok   " << what << '\n';
                return;
            }
            std::cout << "  FAIL " << what << " (got " << actual << ", expected " << expected << ")\n";
            ++failures_;
        }

        // Return this from main().
        int report() const {
            if (failures_ == 0) {
                std::cout << suite_ << ": PASS\n";
                return 0;
            }
            std::cout << suite_ << ": FAIL (" << failures_ << " check(s))\n";
            return 1;
        }

    private:
        std::string suite_;
        int failures_{0};
};

// Renders a Failure or an Error for test output only. Intentionally separate
// from ConsoleReporter: the reporter rate-limits and writes to cerr, both of
// which would lose failures here.
inline std::string describe(const Status::Failure& failure) {
    return "Failure: " + std::string{failure.cause};
}

inline std::string describe(const Status::Error& error) {
    if (const auto* fatal = std::get_if<Status::Fatal>(&error)) {
        return "Fatal: " + fatal->cause;
    }
    const auto& recoverable = std::get<Status::Recoverable>(error);
    return "Recoverable: " + std::string{recoverable.failure.cause};
}

}  // namespace test
