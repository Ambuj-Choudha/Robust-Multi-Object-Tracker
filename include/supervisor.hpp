#pragma once

#include <array>

#include "common/status.hpp"

struct SupervisorConfig {
    static constexpr std::array<int, Status::stage_count> failure_thresholds{
        150,  // Source: device starts dropping frames(could be disconnection), try roughly for 5s (at 30 FPS)
        10,   // Preprocess
        10,   // Inference
        10,   // Postprocess
        10,   // Tracking
        0,    // Visualization: it draws or it throws, it reports no per-frame failure
    };

    static constexpr int video_file_source_threshold = 10;
};

class Supervisor {
    public:
        explicit Supervisor(std::array<int, Status::stage_count> failure_thresholds) noexcept;

        [[nodiscard]] Status::Error classify(const Status::Failure& failure);

        void record_success(Status::Stage stage);

    private:
        std::array<int, Status::stage_count> failure_thresholds_;
        std::array<int, Status::stage_count> consecutive_failures_{};
};
