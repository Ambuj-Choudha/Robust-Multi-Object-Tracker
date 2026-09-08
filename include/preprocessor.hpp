#pragma once

#include <opencv2/core.hpp>

#include "common/status.hpp"
#include "common/types.hpp"

namespace PreprocessorFixedParams {
    constexpr double rescale_factor = 1.0/255.0;
    inline const cv::Scalar mean{0, 0, 0};
    constexpr bool swapRB = true;
    inline const cv::Scalar LetterboxPaddingColour{114, 114, 114};
}

// Turns a raw frame into a model-ready blob: letterbox to a square, then
// pack into the NCHW tensor layout the ONNX model expects.
class YOLOPreprocessor {
    public:
        explicit YOLOPreprocessor(int target_size);

        Status::Result<Data::LetterboxedBlob> process(const Data::Frame& frame);

    private:
        int target_size_;
};
