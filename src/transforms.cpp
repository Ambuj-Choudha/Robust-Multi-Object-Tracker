#include <format>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "common/status.hpp"
#include "common/types.hpp"
#include "transforms.hpp"
#include "preprocessor.hpp"

std::tuple<cv::Mat, double, int, int> preprocess::apply_letterbox_transform(const Data::Frame& src, int target_size) {

    int target_size_width = target_size;
    int target_size_height = target_size;
    auto pad_colour = PreprocessorFixedParams::LetterboxPaddingColour;

    // EOF gives out a default-constructed Frame, so check the state before proceeding
    if (src.mat.empty()) {
      throw Status::FatalException(Status::Fatal{
          .origin = Status::Stage::Preprocess,
          .cause = "empty frame reached preprocessing: the source reported "
                   "success without decoding one"});
    }

    cv::Mat img;
    switch (src.mat.channels()) {
        case 3: img = src.mat;                                     break;
        case 1: cv::cvtColor(src.mat, img, cv::COLOR_GRAY2BGR);    break;
        case 4: cv::cvtColor(src.mat, img, cv::COLOR_BGRA2BGR);    break;
        default:
          throw Status::FatalException(Status::Fatal{
              .origin = Status::Stage::Preprocess,
              .cause = std::format("unsupported channel count {}: expected 1 "
                                   "(gray), 3 (BGR) or 4 (BGRA)",
                                   src.mat.channels())});
    }

    int img_w = img.cols;
    int img_h = img.rows;

    // 1. Uniform scale that fits the image inside the square, preserving aspect ratio.
    double scale = std::min(static_cast<double>(target_size_width) / img_w,
                            static_cast<double>(target_size_height) / img_h);

    // 2. Resized (un-padded) dimensions.
    int new_w = static_cast<int>(std::round(img_w * scale));
    int new_h = static_cast<int>(std::round(img_h * scale));

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    // 3. Padding needed to reach the square, split evenly (centered).
    int pad_w = target_size_width - new_w;
    int pad_h = target_size_height - new_h;
    int pad_left = pad_w / 2;
    int pad_right = pad_w - pad_left;
    int pad_top = pad_h / 2;
    int pad_bottom = pad_h - pad_top;

    cv::Mat output;
    cv::copyMakeBorder(resized, output, pad_top, pad_bottom, pad_left, pad_right,
                       cv::BORDER_CONSTANT, pad_colour);

    return std::tuple(output, scale, pad_left, pad_top);
}

Data::BBox postprocess::undo_letter_box_transform(float x1, float y1, float x2, float y2,
                                                  const Data::LetterboxTransform& transform, int img_w, int img_h) {
    // Inverse of apply_letterbox: strip the pad, then undo the uniform scale.
    Data::BBox bbox{};
    bbox.x1 = static_cast<int>(std::round((x1 - transform.dw) / transform.scale));
    bbox.y1 = static_cast<int>(std::round((y1 - transform.dh) / transform.scale));
    bbox.x2 = static_cast<int>(std::round((x2 - transform.dw) / transform.scale));
    bbox.y2 = static_cast<int>(std::round((y2 - transform.dh) / transform.scale));

    // clip the coordinates to valid image dim
    bbox.x1 = std::clamp(bbox.x1, 0, img_w - 1);
    bbox.y1 = std::clamp(bbox.y1, 0, img_h - 1);
    bbox.x2 = std::clamp(bbox.x2, 0, img_w - 1);
    bbox.y2 = std::clamp(bbox.y2, 0, img_h - 1);

    return bbox;
}
