// Tests ConsoleReporter's rate limiting.
//
// The reporter deliberately drops output, which makes it the one component
// where a bug is invisible in normal use: too aggressive and a real fault goes
// unreported, too lax and the console floods and hides everything else. Both
// failure modes look like "it printed something".
//
// The reporter writes to cerr, so these tests redirect cerr and read back what
// landed there.
//
// Run from repo root: ./build-ninja/test_reporting

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "reporting.hpp"
#include "test_harness.hpp"

namespace {

constexpr const char* kCause = "reporter test cause";

// Swaps cerr's buffer for the duration of a scope, so a test can inspect what
// the reporter wrote. Restores it in the destructor to keep a failing check
// from silencing the harness's own output.
class CapturedCerr {
    public:
        CapturedCerr() : original_{std::cerr.rdbuf(buffer_.rdbuf())} {}
        ~CapturedCerr() { std::cerr.rdbuf(original_); }

        CapturedCerr(const CapturedCerr&) = delete;
        CapturedCerr& operator=(const CapturedCerr&) = delete;

        std::string text() const { return buffer_.str(); }

        // Number of lines written, i.e. how many reports actually got through.
        std::size_t lines() const {
            const std::string s = buffer_.str();
            return static_cast<std::size_t>(std::count(s.begin(), s.end(), '\n'));
        }

    private:
        std::ostringstream buffer_;
        std::streambuf* original_;
};

Status::Error recoverable(Status::Stage stage, int attempt) {
    return Status::Recoverable{Status::Failure{stage, kCause}, attempt};
}

void fatal_is_never_suppressed(test::Checks& checks) {
    ConsoleReporter reporter;
    CapturedCerr captured;

    // A Fatal explains why the process is stopping; rate-limiting it could drop
    // the only message that matters.
    for (int i = 0; i < 3; ++i) {
        reporter.report(Status::Fatal{Status::Stage::Inference, "model load failed"});
    }

    checks.check_eq(captured.lines(), std::size_t{3}, "every Fatal is printed");
    checks.check(captured.text().find("[FATAL]") != std::string::npos, "Fatal is tagged [FATAL]");
    checks.check(captured.text().find("Inference") != std::string::npos, "Fatal names its stage");
}

void repeated_recoverables_are_collapsed(test::Checks& checks) {
    ConsoleReporter reporter;
    CapturedCerr captured;

    for (int i = 1; i <= 100; ++i) {
        reporter.report(recoverable(Status::Stage::Source, i));
    }

    // A camera failing every frame at 30fps would otherwise bury the console.
    checks.check_eq(captured.lines(), std::size_t{1}, "a burst from one stage prints once");
    checks.check(captured.text().find("[WARN]") != std::string::npos, "Recoverable is tagged [WARN]");
    checks.check(captured.text().find("attempt 1") != std::string::npos,
                 "the first of the burst is the one printed");
}

void stages_are_limited_independently(test::Checks& checks) {
    ConsoleReporter reporter;
    CapturedCerr captured;

    // The whole reason the limiter is per-stage: a noisy camera must not
    // suppress a tracker warning that happens in the same second.
    reporter.report(recoverable(Status::Stage::Source, 1));
    reporter.report(recoverable(Status::Stage::Source, 2));
    reporter.report(recoverable(Status::Stage::Tracking, 1));
    reporter.report(recoverable(Status::Stage::Inference, 1));

    checks.check_eq(captured.lines(), std::size_t{3}, "each stage gets its own first warning");
    checks.check(captured.text().find("Tracking") != std::string::npos,
                 "a Tracking warning survives a Source burst");
    checks.check(captured.text().find("Inference") != std::string::npos,
                 "an Inference warning survives a Source burst");
}

void a_fatal_does_not_consume_the_recoverable_interval(test::Checks& checks) {
    ConsoleReporter reporter;
    CapturedCerr captured;

    // Fatal bypasses the limiter, so it must not touch the per-stage timestamp
    // either - otherwise a Fatal would mute the following warning.
    reporter.report(Status::Fatal{Status::Stage::Source, "source died"});
    reporter.report(recoverable(Status::Stage::Source, 1));

    checks.check_eq(captured.lines(), std::size_t{2},
                    "a Fatal does not suppress a later Recoverable from the same stage");
}

void the_interval_reopens(test::Checks& checks) {
    ConsoleReporter reporter;
    CapturedCerr captured;

    reporter.report(recoverable(Status::Stage::Postprocess, 1));
    reporter.report(recoverable(Status::Stage::Postprocess, 2));
    checks.check_eq(captured.lines(), std::size_t{1}, "the second warning is inside the interval");

    // Sleeping past the window is the only way to observe this; the interval is
    // a compile-time constant, so the test reads it rather than hard-coding a
    // duration that would drift if the default changes.
    std::this_thread::sleep_for(ReporterDefaults::recoverable_min_interval +
                                std::chrono::milliseconds{50});

    reporter.report(recoverable(Status::Stage::Postprocess, 3));
    checks.check_eq(captured.lines(), std::size_t{2}, "a warning prints again once the interval elapses");
    checks.check(captured.text().find("attempt 3") != std::string::npos,
                 "the warning after the gap reports the current attempt count");
}

}  // namespace

int main() {
    test::Checks checks{"test_reporting"};

    fatal_is_never_suppressed(checks);
    repeated_recoverables_are_collapsed(checks);
    stages_are_limited_independently(checks);
    a_fatal_does_not_consume_the_recoverable_interval(checks);
    the_interval_reopens(checks);

    return checks.report();
}
