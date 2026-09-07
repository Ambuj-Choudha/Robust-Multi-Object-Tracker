// Tests YOLOPreprocessor: the letterbox geometry and the blob shape handed to the model.
//
// These assertions are the reason preprocessing is its own class. They need no
// model file and no inference, so the letterbox maths can be pinned exactly
// rather than inferred from whether a detection happened to come out right.
//
// Run from repo root: ./build-ninja/test_preprocessor

#include "preprocessor.hpp"

#include <cmath>
#include <string>

#include "test_harness.hpp"

namespace {

constexpr int kTargetSize = 640;

bool close_to(double actual, double expected) {
    return std::abs(actual - expected) < 1e-9;
}

// A square source needs no padding: it scales straight onto the square input.
void square_input_needs_no_padding(test::Checks& checks) {
    YOLOPreprocessor preprocessor{kTargetSize};
    Data::Frame frame{cv::Mat::zeros(320, 320, CV_8UC3)};

    auto result = preprocessor.process(frame);
    checks.check(result.has_value(), "a square frame preprocesses");
    if (!result) return;

    checks.check(close_to(result->transform.scale, 2.0), "scale is target / side length");
    checks.check_eq(result->transform.dw, 0, "no horizontal padding");
    checks.check_eq(result->transform.dh, 0, "no vertical padding");
}

// 2:1 source: the pad lands on the short axis and is split evenly, which is
// exactly what postprocessing subtracts to get back to image space.
void wide_input_pads_the_short_axis(test::Checks& checks) {
    YOLOPreprocessor preprocessor{kTargetSize};
    Data::Frame frame{cv::Mat::zeros(640, 1280, CV_8UC3)};

    auto result = preprocessor.process(frame);
    checks.check(result.has_value(), "a wide frame preprocesses");
    if (!result) return;

    checks.check(close_to(result->transform.scale, 0.5), "scale fits the long axis");
    checks.check_eq(result->transform.dw, 0, "the long axis needs no padding");
    checks.check_eq(result->transform.dh, 160, "the short axis pad is centred");
}

// The model's input tensor is fixed regardless of what came in.
void blob_is_nchw_at_target_size(test::Checks& checks, int rows, int cols, const std::string& what) {
    YOLOPreprocessor preprocessor{kTargetSize};
    Data::Frame frame{cv::Mat::zeros(rows, cols, CV_8UC3)};

    auto result = preprocessor.process(frame);
    checks.check(result.has_value(), what + " frame preprocesses");
    if (!result) return;

    const cv::Mat& blob = result->blob;
    checks.check_eq(blob.dims, 4, what + " blob is 4-dimensional");
    checks.check_eq(blob.size[0], 1, what + " blob holds one image");
    checks.check_eq(blob.size[1], 3, what + " blob has three channels");
    checks.check_eq(blob.size[2], kTargetSize, what + " blob height is the target size");
    checks.check_eq(blob.size[3], kTargetSize, what + " blob width is the target size");
}

// Webcams and decoders hand over grayscale and BGRA too, not only BGR.
void non_bgr_input_is_converted(test::Checks& checks, int type, const std::string& what) {
    YOLOPreprocessor preprocessor{kTargetSize};
    Data::Frame frame{cv::Mat::zeros(320, 320, type)};

    auto result = preprocessor.process(frame);
    checks.check(result.has_value(), what + " input is accepted");
    if (!result) return;

    checks.check_eq(result->blob.size[1], 3, what + " input still yields three channels");
}

// EOF hands out a default-constructed Frame. Reaching preprocessing with one
// means the source reported success without decoding, so it is Fatal, not a retry.
void empty_frame_is_fatal(test::Checks& checks) {
    YOLOPreprocessor preprocessor{kTargetSize};
    Data::Frame empty{};

    try {
        auto result = preprocessor.process(empty);
        checks.check(false, "an empty frame throws instead of returning an error");
    } catch (const Status::FatalException& fatal) {
        checks.check(fatal.error().origin == Status::Stage::Preprocess,
                     "an empty frame is Fatal[Preprocess]");
    }
}

void unsupported_channel_count_is_fatal(test::Checks& checks) {
    YOLOPreprocessor preprocessor{kTargetSize};
    Data::Frame frame{cv::Mat::zeros(320, 320, CV_8UC2)};

    try {
        auto result = preprocessor.process(frame);
        checks.check(false, "a 2-channel frame throws instead of returning an error");
    } catch (const Status::FatalException& fatal) {
        checks.check(fatal.error().origin == Status::Stage::Preprocess,
                     "an unsupported channel count is Fatal[Preprocess]");
    }
}

}  // namespace

int main() {
    test::Checks checks{"test_preprocessor"};

    square_input_needs_no_padding(checks);
    wide_input_pads_the_short_axis(checks);

    blob_is_nchw_at_target_size(checks, 320, 320, "square");
    blob_is_nchw_at_target_size(checks, 640, 1280, "wide");
    blob_is_nchw_at_target_size(checks, 1280, 640, "tall");

    non_bgr_input_is_converted(checks, CV_8UC1, "grayscale");
    non_bgr_input_is_converted(checks, CV_8UC4, "BGRA");

    empty_frame_is_fatal(checks);
    unsupported_channel_count_is_fatal(checks);

    return checks.report();
}
