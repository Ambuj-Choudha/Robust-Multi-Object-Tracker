#include "reporting.hpp"

#include <iostream>
#include <variant>

constexpr std::string_view stage_name(Status::Stage stage) {
    switch (stage) {
        case Status::Stage::Source:        return "Source";
        case Status::Stage::Preprocess:    return "Preprocess";
        case Status::Stage::Inference:     return "Inference";
        case Status::Stage::Postprocess:   return "Postprocess";
        case Status::Stage::Tracking:      return "Tracking";
        case Status::Stage::Visualization: return "Visualization";
    }
    return "Unknown";
}

bool ConsoleReporter::start_new_warning_interval_(Status::Stage origin) {
    const auto now = Clock::now();
    auto& last_warned = last_recoverable_.at(static_cast<std::size_t>(origin));

    const bool first_warning = (last_warned == Clock::time_point{});
    const bool interval_elapsed = ((now - last_warned) >= ReporterDefaults::recoverable_min_interval);

    if (!first_warning && !interval_elapsed) {
        return false;
    }

    last_warned = now;
    return true;
}

void ConsoleReporter::report(const Status::Error& error) {
    if (const auto* fatal = std::get_if<Status::Fatal>(&error)) {  // get_if() returns a nullptr if the error is not Fatal
        std::cerr << "[FATAL] [" << stage_name(fatal->origin) << "] " << fatal->cause << '\n';
        return;
    }

    // Dropped error report if the minimal interval is not elapsed, otherwise print
    const auto& recoverable = std::get<Status::Recoverable>(error);
    if (!start_new_warning_interval_(recoverable.failure.origin)) {
        return;
    }
    std::cerr << "[WARN] [" << stage_name(recoverable.failure.origin) << "] "
              << recoverable.failure.cause << " (attempt " << recoverable.attempt_count << ")\n";
}
