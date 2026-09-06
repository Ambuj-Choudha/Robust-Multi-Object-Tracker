#pragma once

#include <opencv2/core.hpp>

namespace FpsFixedParams {
    constexpr double smoothing_alpha = 0.1;

    // Seeds the smoothed reading so it doesn't ramp up from 0 at startup.
    constexpr double start_fps = 30.0;
}

class Fps {
    public:
        double tick() noexcept {
            int64_t now = cv::getTickCount();
            double time_delta = static_cast<double>(now - last_tick_) / cv::getTickFrequency();
            last_tick_ = now;

            if (time_delta <= 0.0) {
                return smoothed_fps_;
            }

            double instant_fps = 1.0 / time_delta;
            smoothed_fps_ =
                (FpsFixedParams::smoothing_alpha * instant_fps) +
                ((1.0 - FpsFixedParams::smoothing_alpha) * smoothed_fps_);

            return smoothed_fps_;
        }

    private:
        int64_t last_tick_{cv::getTickCount()};
        double smoothed_fps_{FpsFixedParams::start_fps};
};
