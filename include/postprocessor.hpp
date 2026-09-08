#pragma once

#include <vector>

#include "common/status.hpp"
#include "common/types.hpp"
#include "engine.hpp"

// Configurable params
struct PostprocessorConfig {
    static constexpr double ConfThreshold = 0.5;
};

namespace PostprocessorFixedParams {
    // YOLOv10 emits one row per detection as [x1, y1, x2, y2, score, class_id].
    // process() indexes row[0]..row[5], declare default for verifying o/p shape
    constexpr int64_t OutputFieldsPerRow = 6;
}

// Turns the model's raw output rows into confidence-filtered, image-space Detections.
class YOLOv10Postprocessor {
    public:
        explicit YOLOv10Postprocessor(double confidence_threshold = PostprocessorConfig::ConfThreshold);

        Status::Result<std::vector<Data::Detection>> process(InferenceEngine::Output raw_outputs,
                                                              const Data::LetterboxTransform& transform,
                                                              int img_w, int img_h);

    private:
        double confidence_threshold_;
};
