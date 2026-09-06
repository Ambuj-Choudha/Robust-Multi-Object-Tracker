#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <format>
#include <iostream>
#include <string>
#include <string_view>

#include "common/retry_monitor.hpp"
#include "common/status.hpp"
#include "common/types.hpp"

namespace CameraDefaults {
    inline constexpr int FrameDefaultWidth  = 1280;
    inline constexpr int FrameDefaultHeight = 720;

    // Consecutive failed reads allowed before it becomes a fatal error
    inline constexpr int WebcamRetryBudget    = 150;
    inline constexpr int VideoFileRetryBudget = 10;
}

// Pure interface
class IInputSource {
    public:
        virtual ~IInputSource() = default;
        virtual Status::Result<Data::Frame> getNextFrame() = 0;
        [[nodiscard]] virtual Status::SourceState getSourceState()
            const noexcept = 0;
};

// Abstract base class
class VideoCaptureBase : public IInputSource {
    public:
     [[nodiscard]] Status::SourceState getSourceState()
         const noexcept override {
       return source_state_;
     }

    protected:
        explicit VideoCaptureBase(int retry_budget)
            : retry_monitor_{Status::Stage::Source, retry_budget} {}

        cv::VideoCapture cap;

        RetryMonitor retry_monitor_;
        Status::SourceState source_state_{Status::SourceState::Streaming};

        // message is const char* rather than string_view on purpose:
        // it has to be a literal (it ends up in a non-owning view)
        Status::Error record_failure(const char* message, std::string_view operation) {
            auto error = retry_monitor_.record_failure(message, operation);

            source_state_ = retry_monitor_.exhausted() ? Status::SourceState::Failed
                                                      : Status::SourceState::DisconnectedRetrying;
            return error;
        }

        void record_success() {
            retry_monitor_.record_success();
            source_state_ = Status::SourceState::Streaming;
        }
};

class WebcamCamera : public VideoCaptureBase {
    public:
        WebcamCamera(int deviceID = 0, int apiID = cv::CAP_ANY,
                     int retry_budget = CameraDefaults::WebcamRetryBudget);
        WebcamCamera(const WebcamCamera&) = delete;
        WebcamCamera& operator=(const WebcamCamera&) = delete;
        WebcamCamera(WebcamCamera&&) = default;
        ~WebcamCamera() override = default;

        Status::Result<Data::Frame> getNextFrame() override;

    private:
        int deviceID;
        int apiID;
};

class VideoFile : public VideoCaptureBase {
    public:
        VideoFile(const std::string& source_file, int apiID = cv::CAP_ANY,
                  int retry_budget = CameraDefaults::VideoFileRetryBudget);
        VideoFile(const VideoFile&) = delete;
        VideoFile& operator=(const VideoFile&) = delete;
        VideoFile(VideoFile&&) = default;
        ~VideoFile() override = default;

        Status::Result<Data::Frame> getNextFrame() override;

    private:
        std::string source_file;
        int apiID;

        // cv::VideoCapture::read() returns false for a clean end of stream and
        // for a decode that died mid-file, with nothing to tell them apart, 
        // so count how many frames were read against the expected number of frames
        long long expected_frame_count_{0};
        long long frames_read_{0};
};
