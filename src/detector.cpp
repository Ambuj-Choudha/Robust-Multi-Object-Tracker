#include "detector.hpp"

#include <array>

#include "common/types.hpp"

namespace {
    constexpr const char* kInferenceFailedCause = "inference run failed for this frame";

    // Must match the order of the steps in detect(), below.
    constexpr std::array kdetectionPipeline{Status::Stage::Preprocess, Status::Stage::Inference, Status::Stage::Postprocess};
}

YOLOv10DetectorONNX::YOLOv10DetectorONNX(const std::string& model_path, double confidence_threshold)
    : engine_{model_path}, preprocessor_{static_cast<int>(engine_.input_shape()[2])},
      postprocessor_{confidence_threshold} {}

Status::Result<std::vector<Data::Detection>> YOLOv10DetectorONNX::detect(const Data::Frame& frame) {
    // Step 1: Letterbox transformation
    auto preprocessed_frame = preprocessor_.process(frame);
    if (!preprocessed_frame) {
        return std::unexpected(preprocessed_frame.error());  // drop this frame, keep the loop alive
    }
    
    // Step 2: Inference through ONNX model
    auto raw_outputs = engine_.infer(preprocessed_frame->blob.ptr<float>(), preprocessed_frame->blob.total());
    if (!raw_outputs) {
        return std::unexpected(Status::Failure{.origin = Status::Stage::Inference, .cause = kInferenceFailedCause});
    }

    // Step 3: Undo letterbox, wrap in the DS and return
    return postprocessor_.process(*raw_outputs, preprocessed_frame->transform, frame.mat.cols, frame.mat.rows);
}

std::span<const Status::Stage> YOLOv10DetectorONNX::detection_stages() const noexcept {
    return kdetectionPipeline;
}
