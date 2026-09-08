// Tests YOLOv10Postprocessor: threshold filtering and the letterbox inversion
// that maps model-space boxes back to image pixels.
//
// InferenceEngine::Output is a plain view over a float buffer, so the model's
// output can be synthesised here. That is the point of postprocessing being its
// own class: these boundaries are pinned exactly, with no 9.4MB model loaded.
//
// Run from repo root: ./build-ninja/test_postprocessor

#include "postprocessor.hpp"

#include <vector>

#include "test_harness.hpp"

namespace {

constexpr double kThreshold = 0.5;
constexpr int64_t kFieldsPerRow = 6;

// One YOLOv10 output row: [x1, y1, x2, y2, score, class_id] in model space.
struct Row {
    float x1, y1, x2, y2, score, class_id;
};

std::vector<float> flatten(const std::vector<Row>& rows) {
    std::vector<float> buffer;
    for (const auto& row : rows) {
        buffer.insert(buffer.end(), {row.x1, row.y1, row.x2, row.y2, row.score, row.class_id});
    }
    return buffer;
}

InferenceEngine::Output view_of(const std::vector<float>& buffer) {
    return InferenceEngine::Output{.data_ptr = buffer.data(),
                                   .rows = static_cast<int64_t>(buffer.size()) / kFieldsPerRow,
                                   .cols = kFieldsPerRow};
}

void filtering_stops_at_the_first_row_below_threshold(test::Checks& checks) {
    YOLOv10Postprocessor postprocessor{kThreshold};
    const auto buffer = flatten({{10, 10, 20, 20, 0.90F, 0},
                                 {10, 10, 20, 20, 0.60F, 1},
                                 {10, 10, 20, 20, 0.30F, 2},
                                 {10, 10, 20, 20, 0.95F, 3}});

    auto result = postprocessor.process(view_of(buffer), {.scale = 1.0, .dw = 0, .dh = 0}, 640, 640);
    checks.check(result.has_value(), "a well-formed output postprocesses");
    if (!result) return;

    checks.check_eq(result->size(), std::size_t{2}, "only the rows above the threshold are kept");
    if (result->size() == 2) {
        checks.check_eq((*result)[0].class_id, 0, "the first kept row is the highest scoring one");
        checks.check_eq((*result)[1].class_id, 1, "the second kept row is the next one down");
    }
}

// The inverse of the wide-frame letterbox in test_preprocessor: scale 0.5 with
// the pad on the short axis.
void boxes_come_back_in_image_space(test::Checks& checks) {
    YOLOv10Postprocessor postprocessor{kThreshold};
    const auto buffer = flatten({{100, 200, 300, 400, 0.90F, 17}});

    auto result = postprocessor.process(view_of(buffer), {.scale = 0.5, .dw = 0, .dh = 160}, 1280, 640);
    checks.check(result.has_value(), "a letterboxed box postprocesses");
    if (!result || result->empty()) return;

    const auto& bbox = result->front().bbox;
    checks.check_eq(bbox.x1, 200, "x1 undoes the scale");
    checks.check_eq(bbox.y1, 80, "y1 undoes the pad, then the scale");
    checks.check_eq(bbox.x2, 600, "x2 undoes the scale");
    checks.check_eq(bbox.y2, 480, "y2 undoes the pad, then the scale");
    checks.check_eq(result->front().class_id, 17, "class_id is read from the last field");
}

// A box the model puts partly outside the frame must not escape the image
// bounds, downstream indexes pixels with these coordinates.
void out_of_frame_boxes_are_clamped(test::Checks& checks) {
    YOLOv10Postprocessor postprocessor{kThreshold};
    const auto buffer = flatten({{-40, -40, 400, 400, 0.90F, 0}});

    auto result = postprocessor.process(view_of(buffer), {.scale = 0.5, .dw = 0, .dh = 0}, 100, 100);
    checks.check(result.has_value(), "an out-of-frame box postprocesses");
    if (!result || result->empty()) return;

    const auto& bbox = result->front().bbox;
    checks.check_eq(bbox.x1, 0, "x1 clamps at the left edge");
    checks.check_eq(bbox.y1, 0, "y1 clamps at the top edge");
    checks.check_eq(bbox.x2, 99, "x2 clamps at img_w - 1");
    checks.check_eq(bbox.y2, 99, "y2 clamps at img_h - 1");
}

void confidence_is_carried_through(test::Checks& checks) {
    YOLOv10Postprocessor postprocessor{kThreshold};
    const auto buffer = flatten({{10, 10, 20, 20, 0.75F, 0}});

    auto result = postprocessor.process(view_of(buffer), {.scale = 1.0, .dw = 0, .dh = 0}, 640, 640);
    checks.check(result.has_value(), "a single detection postprocesses");
    if (!result || result->empty()) return;

    checks.check(std::abs(result->front().confidence_score - 0.75) < 1e-6,
                 "confidence_score is read from the score field");
}

// A different field count means the loaded model is not the one this decoder
// understands, which no retry can fix.
void wrong_row_width_is_fatal(test::Checks& checks) {
    YOLOv10Postprocessor postprocessor{kThreshold};
    const std::vector<float> buffer(10, 0.0F);
    const auto malformed = InferenceEngine::Output{.data_ptr = buffer.data(), .rows = 2, .cols = 5};

    try {
        auto result = postprocessor.process(malformed, {.scale = 1.0, .dw = 0, .dh = 0}, 640, 640);
        checks.check(false, "a wrong row width throws instead of returning an error");
    } catch (const Status::FatalException& fatal) {
        checks.check(fatal.error().origin == Status::Stage::Postprocess,
                     "a wrong row width is Fatal[Postprocess]");
    }
}

}  // namespace

int main() {
    test::Checks checks{"test_postprocessor"};

    filtering_stops_at_the_first_row_below_threshold(checks);
    boxes_come_back_in_image_space(checks);
    out_of_frame_boxes_are_clamped(checks);
    confidence_is_carried_through(checks);
    wrong_row_width_is_fatal(checks);

    return checks.report();
}
