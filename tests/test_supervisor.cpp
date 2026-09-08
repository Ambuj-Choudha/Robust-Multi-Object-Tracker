// Tests Supervisor: the Recoverable -> Fatal boundary, and the
// per-stage bookkeeping it rests on.
//
// This is the one component that decides whether the run ends, and every stage
// now defers to it, so an off-by-one here changes how long the camera, the
// tracker and both detection stages keep going. It needs no hardware and no
// model, because a Status::Failure is just a stage and a literal.
//
// Run from repo root: ./build-ninja/test_supervisor

#include "supervisor.hpp"

#include <array>

#include "test_harness.hpp"

namespace {

constexpr const char* kCause = "test failure";
constexpr int kThreshold = 3;

// Same threshold everywhere, so a test names the stage it cares about without
// having to remember the shipped table.
Supervisor make_supervisor(int threshold = kThreshold) {
    std::array<int, Status::stage_count> thresholds{};
    thresholds.fill(threshold);
    return Supervisor{thresholds};
}

Status::Error fail_once(Supervisor& supervisor, Status::Stage stage) {
    return supervisor.classify(Status::Failure{.origin = stage, .cause = kCause});
}

void escalates_after_the_threshold(test::Checks& checks) {
    auto supervisor = make_supervisor();

    // Failures 1..threshold stay Recoverable and count up.
    for (int attempt = 1; attempt <= kThreshold; ++attempt) {
        const auto error = fail_once(supervisor, Status::Stage::Source);
        checks.check(Status::is_recoverable(error),
                     "failure " + std::to_string(attempt) + " of " + std::to_string(kThreshold) + " is Recoverable");
        if (Status::is_recoverable(error)) {
            checks.check_eq(std::get<Status::Recoverable>(error).attempt_count, attempt,
                            "attempt_count tracks the streak");
        }
    }

    // The first failure past the threshold escalates.
    checks.check(Status::is_fatal(fail_once(supervisor, Status::Stage::Source)),
                 "the failure after the threshold is Fatal");
}

// A threshold of 0 means no tolerance at all: the first failure ends the run.
// Worth pinning because it is the value most likely to be configured by
// accident, and > vs >= in classify() decides it.
void a_zero_threshold_is_immediately_fatal(test::Checks& checks) {
    auto supervisor = make_supervisor(0);
    checks.check(Status::is_fatal(fail_once(supervisor, Status::Stage::Tracking)),
                 "a threshold of 0 makes the first failure Fatal");
}

void success_ends_the_streak(test::Checks& checks) {
    auto supervisor = make_supervisor();

    fail_once(supervisor, Status::Stage::Source);
    fail_once(supervisor, Status::Stage::Source);
    supervisor.record_success(Status::Stage::Source);

    // The point of "consecutive": an intermittent fault must not accumulate
    // across recoveries into a spurious Fatal.
    const auto error = fail_once(supervisor, Status::Stage::Source);
    checks.check(Status::is_recoverable(error), "a failure after recovery is Recoverable again");
    if (Status::is_recoverable(error)) {
        checks.check_eq(std::get<Status::Recoverable>(error).attempt_count, 1,
                        "the streak restarts at 1, it does not resume");
    }
}

// The reason the table is per-stage rather than one counter: a camera that
// drops every frame must not spend the tracker's tolerance.
void stages_are_counted_independently(test::Checks& checks) {
    auto supervisor = make_supervisor();

    for (int i = 0; i <= kThreshold; ++i) fail_once(supervisor, Status::Stage::Source);

    const auto tracking = fail_once(supervisor, Status::Stage::Tracking);
    checks.check(Status::is_recoverable(tracking),
                 "Tracking still has its full tolerance after Source exhausted its own");
    if (Status::is_recoverable(tracking)) {
        checks.check_eq(std::get<Status::Recoverable>(tracking).attempt_count, 1,
                        "Tracking's streak starts at 1, not at Source's count");
    }
}

void a_success_elsewhere_does_not_clear_a_streak(test::Checks& checks) {
    auto supervisor = make_supervisor();

    fail_once(supervisor, Status::Stage::Inference);
    supervisor.record_success(Status::Stage::Source);

    const auto error = fail_once(supervisor, Status::Stage::Inference);
    checks.check(Status::is_recoverable(error) &&
                     std::get<Status::Recoverable>(error).attempt_count == 2,
                 "recording Source success leaves Inference's streak alone");
}

void the_judgment_carries_the_stage_and_cause(test::Checks& checks) {
    auto supervisor = make_supervisor();

    const auto recoverable = fail_once(supervisor, Status::Stage::Postprocess);
    checks.check(std::get<Status::Recoverable>(recoverable).failure.origin == Status::Stage::Postprocess,
                 "Recoverable carries the failing stage");

    // Every cause is a literal, so the view it holds outlives the error. If
    // that regresses the view dangles, and this is the cheapest place to notice.
    checks.check(std::get<Status::Recoverable>(recoverable).failure.cause == kCause,
                 "the tolerated report still points at the literal the stage gave");

    for (int i = 0; i < kThreshold; ++i) fail_once(supervisor, Status::Stage::Postprocess);
    const auto fatal = fail_once(supervisor, Status::Stage::Postprocess);
    checks.check(std::get<Status::Fatal>(fatal).origin == Status::Stage::Postprocess,
                 "Fatal carries the same stage");

    // The Fatal message is the last thing printed before the process exits, so
    // it has to say what actually kept failing.
    const auto& cause = std::get<Status::Fatal>(fatal).cause;
    checks.check(cause.find(kCause) != std::string::npos,
                 "Fatal quotes the underlying cause, not just the count");
}

// The regression this design exists to prevent. main() reports success for the
// detector stages that ran before a failing one, so two Preprocess failures
// separated by a frame where preprocessing succeeded are not consecutive.
//
// Without that reporting the streak inflates: sporadic failures interleaved
// across the three detector stages would cross a threshold and stop a pipeline
// in which no stage ever failed its threshold number of times in a row.
void an_interleaved_failure_does_not_inflate_an_earlier_streak(test::Checks& checks) {
    auto supervisor = make_supervisor();

    // Frame N: preprocessing fails.
    const auto first = fail_once(supervisor, Status::Stage::Preprocess);
    checks.check_eq(std::get<Status::Recoverable>(first).attempt_count, 1,
                    "frame N: the Preprocess streak opens at 1");

    // Frame N+1: preprocessing succeeds, inference fails. main() reports the
    // Preprocess success because Preprocess runs before Inference.
    supervisor.record_success(Status::Stage::Preprocess);
    fail_once(supervisor, Status::Stage::Inference);

    // Frame N+2: preprocessing fails again. The two failures were not
    // consecutive, so this is a fresh streak.
    const auto third = fail_once(supervisor, Status::Stage::Preprocess);
    checks.check_eq(std::get<Status::Recoverable>(third).attempt_count, 1,
                    "frame N+2: the Preprocess streak restarts, it does not resume at 2");
}

// The shipped table has to line up with the enum, or every stage is classified
// against its neighbour's tolerance.
void the_shipped_table_covers_every_stage(test::Checks& checks) {
    checks.check_eq(SupervisorConfig::failure_thresholds.size(), Status::stage_count,
                    "one threshold per pipeline stage");

    Supervisor supervisor{SupervisorConfig::failure_thresholds};
    const auto error = supervisor.classify(
        Status::Failure{.origin = Status::Stage::Visualization, .cause = kCause});
    checks.check(Status::is_fatal(error),
                 "the Visualization entry is the last one, and it tolerates nothing");
}

}  // namespace

int main() {
    test::Checks checks{"test_supervisor"};

    escalates_after_the_threshold(checks);
    a_zero_threshold_is_immediately_fatal(checks);
    success_ends_the_streak(checks);
    stages_are_counted_independently(checks);
    a_success_elsewhere_does_not_clear_a_streak(checks);
    the_judgment_carries_the_stage_and_cause(checks);
    an_interleaved_failure_does_not_inflate_an_earlier_streak(checks);
    the_shipped_table_covers_every_stage(checks);

    return checks.report();
}
