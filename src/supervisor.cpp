#include "supervisor.hpp"

#include <format>

Supervisor::Supervisor(std::array<int, Status::stage_count> failure_thresholds) noexcept : failure_thresholds_{failure_thresholds} {}

Status::Error Supervisor::classify(const Status::Failure& failure) {
    const std::size_t stage = Status::stage_index(failure.origin);
    const int streak = ++consecutive_failures_.at(stage);

    if (streak > failure_thresholds_.at(stage)) {
      return Status::Fatal{.origin = failure.origin,
          .cause = std::format("unrecoverable after {} consecutive failures: {}", streak, failure.cause)};
    }

    return Status::Recoverable{.failure = failure, .attempt_count = streak};
}

void Supervisor::record_success(Status::Stage stage) {
    consecutive_failures_.at(Status::stage_index(stage)) = 0;
}
