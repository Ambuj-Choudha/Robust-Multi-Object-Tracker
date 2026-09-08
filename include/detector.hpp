#pragma once

#include <string>
#include <vector>

#include "common/status.hpp"
#include "common/types.hpp"
#include "engine.hpp"
#include "postprocessor.hpp"
#include "preprocessor.hpp"

namespace DetectorFixedParams {
    constexpr int num_classes = 80;
}

class DetectorBase{
    public:
        virtual ~DetectorBase() = default;
        virtual Status::Result<std::vector<Data::Detection>> detect(const Data::Frame& frame) = 0;

        [[nodiscard]] virtual int num_classes() const noexcept = 0;
};


class YOLOv10DetectorONNX : public DetectorBase{
    public:
        YOLOv10DetectorONNX(const std::string& model_path, double confidence_threshold = PostprocessorConfig::ConfThreshold);
        
        YOLOv10DetectorONNX(const YOLOv10DetectorONNX&) = delete;
        YOLOv10DetectorONNX& operator=(const YOLOv10DetectorONNX&) = delete;
        
        YOLOv10DetectorONNX(YOLOv10DetectorONNX&&) = delete;
        YOLOv10DetectorONNX& operator=(YOLOv10DetectorONNX&&) = delete;
        
        ~YOLOv10DetectorONNX() override = default;

        Status::Result<std::vector<Data::Detection>> detect(const Data::Frame& frame) override;

        [[nodiscard]] int num_classes() const noexcept override {
            return DetectorFixedParams::num_classes;
        }

    private:
        InferenceEngine engine_;
        YOLOPreprocessor preprocessor_;
        YOLOv10Postprocessor postprocessor_;
};
