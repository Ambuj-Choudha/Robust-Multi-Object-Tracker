#pragma once
#include <cstddef>
#include <expected>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace Status {

// One entry per pipeline stage that can fail on its own.
enum class Stage { Source, Preprocess, Inference, Postprocess, Tracking, Visualization };

// Update the last enumerator here if you add a stage after Visualization.
inline constexpr std::size_t stage_count = static_cast<std::size_t>(Stage::Visualization) + 1;


[[nodiscard]] inline constexpr std::size_t stage_index(Stage stage) noexcept {
    return static_cast<std::size_t>(stage);
}

struct Failure {
    Stage origin;
    std::string_view cause; 
};

struct Fatal {
    Stage origin;
    std::string cause;  // a Fatal happens once, so the allocation is fine
};

struct Recoverable {
    Failure failure;
    int attempt_count;  // length of the failure streak this one belongs to
};

using Error = std::variant<Fatal, Recoverable>;

template <typename T>
using Result = std::expected<T, Failure>;

[[nodiscard]] inline bool is_fatal(const Error& error) noexcept {
    return std::holds_alternative<Fatal>(error);
}

[[nodiscard]] inline bool is_recoverable(const Error& error) noexcept {
    return std::holds_alternative<Recoverable>(error);
}

// Carries a Fatal, just thrown instead of returned: constructors throw
// (nothing to return from), the loop returns.
class FatalException : public std::runtime_error {
public:
    explicit FatalException(Fatal fatal)
        : std::runtime_error{fatal.cause}, error_{std::move(fatal)} {}

    [[nodiscard]] const Fatal& error() const noexcept { return error_; }

   private:
    Fatal error_;
};

enum class SourceState { Streaming, EndOfStream };

}
