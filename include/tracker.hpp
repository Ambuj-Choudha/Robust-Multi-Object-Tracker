#pragma once

#include <memory>
#include <vector>

#include "common/status.hpp"
#include "common/types.hpp"

// Forward declaration so users of tracker.hpp don't have to pull in
// ByteTrack's Eigen-heavy headers.
namespace byte_track { class BYTETracker; }

// Mirrors byte_track::BYTETracker's own constructor defaults
struct TrackerConfig {
    static constexpr int frame_rate    = 30;
    static constexpr int track_buffer  = 30;
    static constexpr float track_thresh = 0.5F;
    static constexpr float high_thresh = 0.6F;
    static constexpr float match_thresh = 0.8F;
};

namespace TrackerFixedParams {
    // IoU tuning threshold before a track's class_id is recovered from a detection
constexpr float class_recovery_min_iou = 0.5F;
}

class ByteTrackerAdapter {
    public:
        ByteTrackerAdapter(int frame_rate    = TrackerConfig::frame_rate,
                           int track_buffer  = TrackerConfig::track_buffer,
                           float track_thresh = TrackerConfig::track_thresh,
                           float high_thresh  = TrackerConfig::high_thresh,
                           float match_thresh = TrackerConfig::match_thresh);
        ByteTrackerAdapter(const ByteTrackerAdapter&) = delete;
        ByteTrackerAdapter& operator=(const ByteTrackerAdapter&) = delete;
        ByteTrackerAdapter(ByteTrackerAdapter&&) = delete;
        ByteTrackerAdapter& operator=(ByteTrackerAdapter&&) = delete;
        ~ByteTrackerAdapter();

        // Per-frame, so a failed solve drops the frame and continues
        Status::Result<std::vector<Data::TrackedDetection>> update(const std::vector<Data::Detection>& detections);

    private:
        std::unique_ptr<byte_track::BYTETracker> impl_;
};
