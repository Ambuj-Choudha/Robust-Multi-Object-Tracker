#include "visualization.hpp"

#include <format>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "common/status.hpp"
#include "common/types.hpp"

namespace {
    std::unordered_map<int, std::string> load_class_labels(const std::string& labels_file_path) {
        std::unordered_map<int, std::string> labels;
        std::ifstream in(labels_file_path);

        if (!in) {
          throw Status::FatalException(Status::Fatal{
              .origin = Status::Stage::Visualization,
              .cause = std::format("failed to open class labels file '{}'",
                                   labels_file_path)});
        }

        std::string line;
        int idx = 0;
        while (std::getline(in, line)) {
          if (!line.empty() && line.back() == '\r') {
            line.pop_back();
          }
            labels.emplace(idx++, std::move(line));
        }
        return labels;
    }

    std::unordered_map<int, cv::Scalar> generate_colour_map(int num_classes) {
        std::mt19937 rng(VisualizerFixedParams::colour_seed);
        std::uniform_int_distribution<int> dist(0, 255);
        std::unordered_map<int, cv::Scalar> colours;
        for (int i = 0; i < num_classes; ++i) {
            int r = dist(rng);
            int g = dist(rng);
            int b = dist(rng);
            colours.emplace(i, cv::Scalar(b, g, r));
        }
        return colours;
    }
}

Visualizer::Visualizer(int border_thickness, int num_classes, std::optional<std::tuple<int, int, int>> text_colour)
    : border_thickness_{border_thickness}, text_colour_{text_colour.value_or(VisualizerConfig::text_colour)},
      class_labels_dict_{load_class_labels(VisualizerConfig::class_labels_file_path)}, colour_map_{generate_colour_map(num_classes)} {
  if (std::cmp_not_equal(class_labels_dict_.size(), num_classes)) {
    throw Status::FatalException(Status::Fatal{
        .origin = Status::Stage::Visualization,
        .cause = std::format(
            "class labels file '{}' has {} entries but this build is "
            "configured for "
            "{} classes - the labels file does not match the model",
            VisualizerConfig::class_labels_file_path, class_labels_dict_.size(), num_classes)});
  }
}

void Visualizer::draw_detections(Data::Frame& frame, const std::vector<Data::Detection>& detections) {
    for(const auto& detection : detections) {
        this->draw_bbox_w_labels_(frame.mat, detection);
    }
}

void Visualizer::draw_fps(Data::Frame& frame, double fps) {
    std::string label = std::format("FPS: {:.1f}", fps);
    const auto& [tr, tg, tb] = text_colour_;
    cv::putText(frame.mat, label, VisualizerFixedParams::fps_text_origin,
                VisualizerFixedParams::font, font_scale_, cv::Scalar(tb, tg, tr), 1);
}

void Visualizer::set_font_scale(double new_font_scale) {
    this->font_scale_ = new_font_scale;
}

void Visualizer::draw_bbox_w_labels_(cv::Mat& img, const Data::Detection& detected_obj) {
    // extract the right coordinates from the Detection struct
    int x1 = detected_obj.bbox.x1;
    int y1 = detected_obj.bbox.y1;
    int x2 = detected_obj.bbox.x2;
    int y2 = detected_obj.bbox.y2;
    int class_id = detected_obj.class_id;
    // get the label for the object based on the class ID using the undodered_map 'class_labels_dict_'created earlier, also mention the confidence score. "<class_name> : <conf_score>"
    const auto& box_colour = this->colour_map_.at(class_id);
    const auto& class_name = this->class_labels_dict_.at(class_id);

    std::ostringstream oss;
    oss << class_name << " : " << detected_obj.confidence_score;
    std::string label = oss.str();

    // Draw the Rectangle
    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), box_colour, border_thickness_);

    // Write the text
    const auto& [tr, tg, tb] = text_colour_;
    cv::putText(img, label, cv::Point(x1, y1 - 5), VisualizerFixedParams::font, this->font_scale_, cv::Scalar(tb, tg, tr), 1);
}

void Visualizer::draw_tracked_detections(Data::Frame& frame, const std::vector<Data::TrackedDetection>& tracked) {
    for (const auto& t : tracked) {
        this->draw_bbox_w_track_id_(frame.mat, t);
    }
}

void Visualizer::draw_bbox_w_track_id_(cv::Mat& img, const Data::TrackedDetection& tracked_obj) {
    const auto& d = tracked_obj.detection;

    const bool has_class = d.class_id >= 0;

    const cv::Scalar box_colour =
        has_class ? colour_map_.at(d.class_id)
                  : colour_map_.at(static_cast<int>(tracked_obj.track_id % colour_map_.size()));

    std::ostringstream oss;
    oss << "ID " << tracked_obj.track_id;
    if (has_class) {
        oss << " " << class_labels_dict_.at(d.class_id);
    }
    std::string label = oss.str();

    cv::rectangle(img, cv::Point(d.bbox.x1, d.bbox.y1), cv::Point(d.bbox.x2, d.bbox.y2),
                  box_colour, border_thickness_);
    const auto& [tr, tg, tb] = text_colour_;
    cv::putText(img, label, cv::Point(d.bbox.x1, d.bbox.y1 - 5),
                VisualizerFixedParams::font, this->font_scale_, cv::Scalar(tb, tg, tr), 1);
}
