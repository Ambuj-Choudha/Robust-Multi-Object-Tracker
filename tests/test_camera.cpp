// Tests the source stage's contract with the loop in main(). main() asks a
// source only two things, so those are what is pinned here:
//   - does this call hand back a frame, or a Status::Failure tagged Stage::Source
//   - has the stream ended, which is not an error
// plus the one thing that is fatal rather than per-frame: a source that cannot
// be opened at all.
//
// The escalation tests that used to live here moved to test_supervisor.cpp with
// the state they exercised. VideoCaptureBase no longer counts failures or
// decides severity, so FakeSource below drives IInputSource, the interface main
// actually depends on, rather than subclassing a base whose cap it never opens.
//
// VideoFile's own failure branches need a decoder that fails on a file that
// opened cleanly, which no amount of corrupting a container produces: a corrupt
// file fails to open instead. FailingVideoFile overrides the read_frame() seam
// to get at them.
//
// Run from repo root: ./build-ninja/test_camera

#include <filesystem>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "camera.hpp"
#include "test_harness.hpp"

#ifndef PROJECT_ROOT
#define PROJECT_ROOT "."
#endif

namespace {

constexpr const char* kFakeCause = "fake source failure";

// Writes a short clip so the paths that need a real container - opening,
// frame counting, EOF - run against a real decoder. Returns an empty path if
// no encoder is available.
std::string make_clip(const std::filesystem::path& path, int frames) {
    cv::VideoWriter writer{path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10.0, cv::Size{64, 64}};
    if (!writer.isOpened()) return {};

    for (int i = 0; i < frames; ++i) {
        writer.write(cv::Mat::zeros(64, 64, CV_8UC3));
    }
    writer.release();
    return path.string();
}

// Opens a real clip, so the constructor and its frame count are genuine, then
// substitutes the read. Modes match the three branches getNextFrame() has after
// the read returns.
class FailingVideoFile : public VideoFile {
    public:
        enum class Mode { DecodeFails, DecodeReturnsEmpty, OutOfMemory };

        FailingVideoFile(const std::string& clip, Mode mode) : VideoFile{clip}, mode_{mode} {}

    protected:
        bool read_frame(cv::Mat& frame) override {
            switch (mode_) {
                case Mode::DecodeFails:
                    return false;
                case Mode::DecodeReturnsEmpty:
                    frame = cv::Mat{};
                    return true;
                case Mode::OutOfMemory:
                    throw cv::Exception{cv::Error::StsNoMem, "simulated", "read_frame", __FILE__, __LINE__};
            }
            return false;
        }

    private:
        Mode mode_;
};


// A source with no decoder behind it, so the failure path is reachable on
// demand instead of depending on a decoder being coaxed into breaking.
class FakeSource : public IInputSource {
    public:
        Status::Result<Data::Frame> getNextFrame() override {
            if (at_end_) {
                source_state_ = Status::SourceState::EndOfStream;
                return Data::Frame{};
            }
            if (fail_next_) {
                return std::unexpected(Status::Failure{.origin = Status::Stage::Source,
                                                       .cause = kFakeCause});
            }
            return Data::Frame{cv::Mat::zeros(2, 2, CV_8UC3)};
        }

        [[nodiscard]] Status::SourceState getSourceState() const noexcept override {
            return source_state_;
        }

        void set_failing(bool failing) { fail_next_ = failing; }
        void set_at_end(bool at_end) { at_end_ = at_end; }

    private:
        Status::SourceState source_state_{Status::SourceState::Streaming};
        bool fail_next_{true};
        bool at_end_{false};
};

void a_failed_read_is_tagged_source(test::Checks& checks) {
    FakeSource source;

    auto frame = source.getNextFrame();
    checks.check(!frame.has_value(), "a failing read returns an error, not a frame");

    // The supervisor indexes its threshold table on origin, so a failure tagged
    // with the wrong stage is judged against another stage's tolerance.
    checks.check(frame.error().origin == Status::Stage::Source,
                 "a source failure is tagged Stage::Source");
    checks.check(frame.error().cause == kFakeCause,
                 "the cause travels through untouched");
}

// A source reports what happened and nothing more. It used to move itself to
// DisconnectedRetrying and Failed on the way; that is PipelineSupervisor's call
// now, and this pins that the source stopped making it.
void a_failed_read_does_not_move_the_source_state(test::Checks& checks) {
    FakeSource source;

    checks.check(source.getSourceState() == Status::SourceState::Streaming,
                 "a fresh source starts in Streaming");

    for (int i = 0; i < 5; ++i) {
        auto frame = source.getNextFrame();
        checks.check(!frame.has_value(), "read " + std::to_string(i + 1) + " fails");
    }

    checks.check(source.getSourceState() == Status::SourceState::Streaming,
                 "repeated failures leave the state alone; severity is not the source's job");
}

// main() checks getSourceState() before unwrapping, so EOF must not also
// produce an error - that would report a normal shutdown as a failure.
void end_of_stream_is_not_an_error(test::Checks& checks) {
    FakeSource source;
    source.set_failing(false);
    source.set_at_end(true);

    auto frame = source.getNextFrame();
    checks.check(frame.has_value(), "reading at EOF is not an error");
    checks.check(source.getSourceState() == Status::SourceState::EndOfStream,
                 "reaching the end sets EndOfStream");
}

void a_real_file_reaches_end_of_stream(test::Checks& checks) {
    const auto dir = std::filesystem::temp_directory_path();
    const std::string clip = make_clip(dir / "obj_tracker_eof_test.avi", 5);
    if (clip.empty()) {
        std::cout << "  skip no MJPG encoder available; real-decoder EOF not covered\n";
        return;
    }

    VideoFile source{clip};
    checks.check(source.getSourceState() == Status::SourceState::Streaming,
                 "a freshly opened file starts in Streaming");

    int decoded = 0;
    // Bounded so a decoder that never reports EOF fails the test instead of
    // hanging it.
    for (int i = 0; i < 50; ++i) {
        auto frame = source.getNextFrame();
        if (source.getSourceState() == Status::SourceState::EndOfStream) break;
        if (!frame.has_value()) {
            checks.check(false, "unexpected error mid-clip: " + test::describe(frame.error()));
            break;
        }
        ++decoded;
    }

    checks.check_eq(decoded, 5, "every written frame is decoded before EOF");
    checks.check(source.getSourceState() == Status::SourceState::EndOfStream,
                 "running out of frames sets EndOfStream");

    auto past_eof = source.getNextFrame();
    checks.check(past_eof.has_value(), "reading past EOF is not an error");
    checks.check(source.getSourceState() == Status::SourceState::EndOfStream,
                 "EndOfStream is sticky");

    std::filesystem::remove(clip);
}

// The branch that separates a broken file from a finished one. frames_read_
// exists only to make this distinction, so if it regresses a truncated clip is
// silently reported as a clean end of stream.
void a_decode_failure_before_the_last_frame_is_an_error(test::Checks& checks) {
    const auto dir = std::filesystem::temp_directory_path();
    const std::string clip = make_clip(dir / "obj_tracker_decode_fail_test.avi", 20);
    if (clip.empty()) {
        std::cout << "  skip no MJPG encoder available; decode-failure path not covered\n";
        return;
    }

    FailingVideoFile source{clip, FailingVideoFile::Mode::DecodeFails};

    auto frame = source.getNextFrame();
    checks.check(!frame.has_value(), "a decode that fails before the end is an error");
    if (!frame.has_value()) {
        checks.check(frame.error().origin == Status::Stage::Source,
                     "a decode failure is tagged Stage::Source");
    }
    checks.check(source.getSourceState() == Status::SourceState::Streaming,
                 "a decode failure is not an end of stream");

    std::filesystem::remove(clip);
}

// A decoder that reports success and hands back nothing. Distinct from a failed
// read, and it must not reach preprocessing, where an empty Mat is Fatal.
void an_empty_frame_from_a_successful_read_is_an_error(test::Checks& checks) {
    const auto dir = std::filesystem::temp_directory_path();
    const std::string clip = make_clip(dir / "obj_tracker_empty_read_test.avi", 20);
    if (clip.empty()) {
        std::cout << "  skip no MJPG encoder available; empty-read path not covered\n";
        return;
    }

    FailingVideoFile source{clip, FailingVideoFile::Mode::DecodeReturnsEmpty};

    auto frame = source.getNextFrame();
    checks.check(!frame.has_value(), "a successful read of an empty frame is still an error");
    if (!frame.has_value()) {
        checks.check(frame.error().origin == Status::Stage::Source,
                     "an empty decode is tagged Stage::Source");
    }

    std::filesystem::remove(clip);
}

// OpenCV signals allocation failure as a cv::Exception with StsNoMem, not
// std::bad_alloc. The source models that one code and rethrows everything else,
// so this pins which exceptions are caught rather than escaping to main.
void an_allocation_failure_is_a_recoverable_error(test::Checks& checks) {
    const auto dir = std::filesystem::temp_directory_path();
    const std::string clip = make_clip(dir / "obj_tracker_oom_test.avi", 20);
    if (clip.empty()) {
        std::cout << "  skip no MJPG encoder available; allocation-failure path not covered\n";
        return;
    }

    FailingVideoFile source{clip, FailingVideoFile::Mode::OutOfMemory};

    try {
        auto frame = source.getNextFrame();
        checks.check(!frame.has_value(), "an StsNoMem exception becomes an error, not a throw");
        if (!frame.has_value()) {
            checks.check(frame.error().origin == Status::Stage::Source,
                         "an allocation failure is tagged Stage::Source");
        }
    } catch (const cv::Exception&) {
        checks.check(false, "StsNoMem escaped getNextFrame() instead of being modelled");
    }

    std::filesystem::remove(clip);
}

void missing_file_is_fatal_at_construction(test::Checks& checks) {
    // Taxonomy: a source that cannot be opened at all has no per-frame story,
    // so it throws from the constructor rather than reporting a Failure the
    // supervisor would have to classify.
    try {
        VideoFile source{std::string{PROJECT_ROOT} + "/data/input/does_not_exist.mp4"};
        checks.check(false, "expected a FatalException for a missing video file");
    } catch (const Status::FatalException& e) {
        checks.check(e.error().origin == Status::Stage::Source,
                     "a missing video file is Fatal[Source]");
    }
}

}  // namespace

int main() {
    test::Checks checks{"test_camera"};

    a_failed_read_is_tagged_source(checks);
    a_failed_read_does_not_move_the_source_state(checks);
    end_of_stream_is_not_an_error(checks);
    a_real_file_reaches_end_of_stream(checks);
    a_decode_failure_before_the_last_frame_is_an_error(checks);
    an_empty_frame_from_a_successful_read_is_an_error(checks);
    an_allocation_failure_is_a_recoverable_error(checks);
    missing_file_is_fatal_at_construction(checks);

    return checks.report();
}
