#include "postprocessor.hpp"
#include "transforms.hpp"

#include <format>
#include <new>

namespace {
    constexpr const char* kDetectionsOutOfMemoryCause = "out of memory storing detections";
}

YOLOv10Postprocessor::YOLOv10Postprocessor(double confidence_threshold)
    : confidence_threshold_{confidence_threshold},
      retry_monitor_{Status::Stage::Postprocess, PostprocessorFixedParams::RetryBudget} {}

Status::Result<std::vector<Data::Detection>> YOLOv10Postprocessor::process(
    InferenceEngine::Output raw_outputs, const Data::LetterboxTransform& transform,
    int img_w, int img_h) {
    if (raw_outputs.cols != PostprocessorFixedParams::OutputFieldsPerRow) {
      throw Status::FatalException(Status::Fatal{
          .origin = Status::Stage::Postprocess,
          .cause = std::format(
              "model output has {} fields per row, expected {}",
              raw_outputs.cols, PostprocessorFixedParams::OutputFieldsPerRow)});
    }

    std::vector<Data::Detection> detections;

    try {
        detections.reserve(raw_outputs.rows);

        // YOLOv10 output rows are sorted by score descending; break early below threshold.
        for (int64_t i = 0; i < raw_outputs.rows; ++i) {
            const float* row = raw_outputs.row(i);  // i-th detection
            const double conf = row[4];

            // rows are score-sorted, so once we drop below threshold we're done
            if (conf < confidence_threshold_) {
              break;
            }

            Data::BBox bbox_orig = postprocess::undo_letter_box_transform(row[0], row[1], row[2], row[3],
                                                                        transform, img_w, img_h);
            const int class_id = static_cast<int>(row[5]);

            detections.push_back(Data::Detection{.bbox = bbox_orig,
                                                 .class_id = class_id,
                                                 .confidence_score = conf});
        }
    } catch (const std::bad_alloc&) {
        return std::unexpected(retry_monitor_.record_failure(kDetectionsOutOfMemoryCause, "postprocess"));
    }

    retry_monitor_.record_success();
    return detections;
}
