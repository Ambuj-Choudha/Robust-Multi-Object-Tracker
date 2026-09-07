#include "preprocessor.hpp"
#include "transforms.hpp"

#include <opencv2/dnn.hpp>

namespace {
    constexpr const char* kBlobOutOfMemoryCause = "out of memory building preprocess blob";

    bool is_out_of_memory(const cv::Exception& preprocess_error) noexcept {
        return preprocess_error.code == cv::Error::StsNoMem;
    }
}

YOLOPreprocessor::YOLOPreprocessor(int target_size)
    : target_size_{target_size}, retry_monitor_{Status::Stage::Preprocess, PreprocessorFixedParams::RetryBudget} {}

Status::Result<Data::LetterboxedBlob> YOLOPreprocessor::process(const Data::Frame& frame) {
    try {
        auto [letterboxed_frame, scale, dw, dh] = preprocess::apply_letterbox_transform(frame, target_size_);

        cv::dnn::Image2BlobParams imgParams(
            PreprocessorFixedParams::rescale_factor,
            cv::Size(target_size_, target_size_),
            PreprocessorFixedParams::mean,
            PreprocessorFixedParams::swapRB,
            CV_32F,
            cv::dnn::DNN_LAYOUT_NCHW,
            cv::dnn::DNN_PMODE_NULL,           // letterbox already padded to target size
            PreprocessorFixedParams::LetterboxPaddingColour  // unused when mode is NULL
        );

        auto blob = cv::dnn::blobFromImageWithParams(letterboxed_frame, imgParams);
        retry_monitor_.record_success();
        return Data::LetterboxedBlob{.blob = blob, .transform = {.scale = scale, .dw = dw, .dh = dh}};
        
    } catch (const cv::Exception& preprocess_error) {
        if (is_out_of_memory(preprocess_error)) {
            return std::unexpected(retry_monitor_.record_failure(kBlobOutOfMemoryCause, "preprocess"));
        }
        throw;  // any other cv::Exception is not a modelled in this stage
    }
}
