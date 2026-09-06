#pragma once

#include<optional>
#include<string>
#include<tuple>
#include<unordered_map>
#include<vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "common/types.hpp"

struct VisualizerConfig {
    static inline std::tuple<int,int,int> text_colour{0, 0, 0};
    static inline double font_scale = 0.5;
    static inline std::string class_labels_file_path = "assets/labels/coco.names";
};

// Fixed internal constants for Visualizer; not caller-configurable.
namespace VisualizerFixedParams {
    constexpr int font = cv::FONT_HERSHEY_SIMPLEX;
    constexpr int colour_seed = 42;
    inline const cv::Point fps_text_origin{10, 20};
}


class Visualizer{
    public:
        Visualizer(int border_thickness, std::optional<std::tuple<int, int, int>> text_colour = std::nullopt);

        void draw_detections(Data::Frame& frame, const std::vector<Data::Detection>& detections);
        void draw_tracked_detections(Data::Frame& frame, const std::vector<Data::TrackedDetection>& tracked);
        void draw_fps(Data::Frame& frame, double fps);
        void set_font_scale(double new_font_scale);
    private:
        int border_thickness_;
        std::tuple<int, int, int> text_colour_;
        double font_scale_{VisualizerConfig::font_scale};

        // define from the asset/coco.names
        const std::unordered_map<int, std::string> class_labels_dict_;

        const std::unordered_map<int, cv::Scalar> colour_map_;
        void draw_bbox_w_labels_(cv::Mat& img, const Data::Detection& detected_obj);
        void draw_bbox_w_track_id_(cv::Mat& img, const Data::TrackedDetection& tracked_obj);
};
