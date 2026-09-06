#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "common/status.hpp"

// Thin wrapper around an ONNX Runtime session for a single-input,
// single-output model. Detector talks to the model through this.
class InferenceEngine {
public:
    // View onto the model's last output tensor.
    // `data_ptr` points into a buffer owned by the engine and is valid only
    // until the next call to infer() — the underlying Ort::Value is held
    // as a member and overwritten on each infer().
    struct Output {
        const float* data_ptr;
        int64_t rows;   // e.g. 300 detection slots
        int64_t cols;   // e.g. 6 fields per row

        // Pointer to the start of row `i` (row-major layout).
        [[nodiscard]] const float* row(int64_t i) const {
          return data_ptr + (i * cols);
        }
    };

    explicit InferenceEngine(const std::string& model_path);
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) = delete;
    InferenceEngine& operator=(InferenceEngine&&) = delete;
    ~InferenceEngine() = default;

    std::optional<Output> infer(const float* input, size_t input_len);

    // Shape the model declares at its single input tensor, e.g. {1, 3, 640, 640}.
    // Read from the ONNX at load time — no hardcoded values in this class.
    [[nodiscard]] const std::vector<int64_t>& input_shape() const {
      return input_shape_;
    }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_{nullptr};
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    Ort::Value last_output_{nullptr};  // keeps most recent Run() result alive
};
