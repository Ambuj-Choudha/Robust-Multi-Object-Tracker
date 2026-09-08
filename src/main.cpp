#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <variant>

#include "camera.hpp"
#include "common/fps.hpp"
#include "common/status.hpp"
#include "common/types.hpp"
#include "detector.hpp"
#include "reporting.hpp"
#include "supervisor.hpp"
#include "tracker.hpp"
#include "visualization.hpp"

int main(int argc, char* argv[]) {
    std::unique_ptr<IInputSource> source;  // use pointer from base class
    ConsoleReporter reporter;
    
    if (argc > 2) {
        std::cout << "Wrong number of arguments passed!" << "\n";
        std::cout << "Use <program name> <source_video_file> OR simply <program name> for webcam..." << "\n";
        return -1;
    }

    try {
        using namespace std::string_literals;
        std::string model_path = "assets/model/yolov10n/yolov10n.onnx"s;

        auto failure_thresholds = SupervisorConfig::failure_thresholds;

        if (argc < 2) {
            source = std::make_unique<WebcamCamera>();
        } else {
            source = std::make_unique<VideoFile>(argv[1]);
            failure_thresholds[Status::stage_index(Status::Stage::Source)] = SupervisorConfig::video_file_source_threshold;
        }

        Supervisor supervisor{failure_thresholds};

        auto detector = YOLOv10DetectorONNX(model_path);
        auto tracker = ByteTrackerAdapter{};
        auto visualizer_obj = Visualizer(2, detector.num_classes());
        Fps fps;

        auto should_stop = [&](const Status::Failure& failure) {
            const auto error = supervisor.classify(failure);
            reporter.report(error);
            return Status::is_fatal(error);
        };

        while (true) {
            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) {
                break;
            }

            auto frame = source->getNextFrame();

            // EOF: Check before unwrapping
            if (source->getSourceState() == Status::SourceState::EndOfStream) {
                std::cout << "End of stream reached.\n";
                break;
            }

            if (!frame) {
              if (should_stop(frame.error())) {
                return -1;
              }
                continue;  // inside the threshold: drop this frame, keep looping
            }
            supervisor.record_success(Status::Stage::Source);

            Data::Frame input_frame = *frame;

            auto detections_in_current_frame = detector.detect(input_frame);
            if (!detections_in_current_frame) {
              if (should_stop(detections_in_current_frame.error())) {
                return -1;
              }
                continue;
            }
            supervisor.record_success(Status::Stage::Preprocess);
            supervisor.record_success(Status::Stage::Inference);
            supervisor.record_success(Status::Stage::Postprocess);

            auto tracked_in_current_frame = tracker.update(*detections_in_current_frame);
            if (!tracked_in_current_frame) {
              if (should_stop(tracked_in_current_frame.error())) {
                return -1;
              }
                continue;  // one bad solve: drop this frame's tracks, keep looping
            }
            supervisor.record_success(Status::Stage::Tracking);

            visualizer_obj.draw_tracked_detections(input_frame, *tracked_in_current_frame);
            visualizer_obj.draw_fps(input_frame, fps.tick());
            cv::imshow("Detected Objects", input_frame.mat);
        }
    } catch (const Status::FatalException& e) {
        reporter.report(e.error());
        return -1;
    } catch (const std::exception& e) {
        // catches third-party (ByteTrack) and stdlib exceptions
        std::cout << "Fatal exception: " << e.what() << "\n";
        return -1;
    }
    return 0;
}
