#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <string>

#include "common/status.hpp"
#include "common/types.hpp"

namespace CameraDefaults {
    inline constexpr int FrameDefaultWidth  = 1280;
    inline constexpr int FrameDefaultHeight = 720;
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
        VideoCaptureBase() = default;

        cv::VideoCapture cap;

        Status::SourceState source_state_{Status::SourceState::Streaming};

        // virtual only so a test can substitute a decoder that fails on demand
        virtual bool read_frame(cv::Mat& frame) { return cap.read(frame); }
};

class WebcamCamera : public VideoCaptureBase {
    public:
        WebcamCamera(int deviceID = 0, int apiID = cv::CAP_ANY);
        WebcamCamera(const WebcamCamera&) = delete;
        WebcamCamera& operator=(const WebcamCamera&) = delete;
        WebcamCamera(WebcamCamera&&) = default;
        WebcamCamera& operator=(WebcamCamera&&) = default;
        ~WebcamCamera() override = default;

        Status::Result<Data::Frame> getNextFrame() override;

    private:
        int deviceID;
        int apiID;
};

class VideoFile : public VideoCaptureBase {
    public:
        VideoFile(const std::string& source_file, int apiID = cv::CAP_ANY);
        VideoFile(const VideoFile&) = delete;
        VideoFile& operator=(const VideoFile&) = delete;
        VideoFile(VideoFile&&) = default;
        VideoFile& operator=(VideoFile&&) = default;
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
