#include "engine.hpp"

#include <format>
#include <stdexcept>

#include "common/status.hpp"

InferenceEngine::InferenceEngine(const std::string& model_path)
    : env_{ORT_LOGGING_LEVEL_WARNING, "InferenceEngine"} {
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Covers the whole constructor body, as it can fail at GetInputTypeInfo if it
  // has no input 0.
  try {
    // ORT session ctor takes ORTCHAR_T*, which is wchar_t on Windows and char
    // elsewhere.
#ifdef _WIN32
        std::wstring wpath(model_path.begin(), model_path.end());
        session_ = Ort::Session{env_, wpath.c_str(), session_options_};
#else
        session_ = Ort::Session{env_, model_path.c_str(), session_options_};
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        input_name_  = std::string{session_.GetInputNameAllocated(0, allocator).get()};
        output_name_ = std::string{session_.GetOutputNameAllocated(0, allocator).get()};

        // Read the input shape the model declares (e.g. {1, 3, 640, 640}).
        input_shape_ = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    } catch (const Ort::Exception& load_error) {
      throw Status::FatalException(Status::Fatal{
          .origin = Status::Stage::Inference,
          .cause =
              std::format("failed to load model '{}': {} (ORT error {})",
                          model_path, load_error.what(),
                          static_cast<int>(load_error.GetOrtErrorCode()))});
    }

    // Some models declare batch dim as -1 (dynamic). Pin it to 1 for our single-frame use.
    if (!input_shape_.empty() && input_shape_[0] < 0) {
        input_shape_[0] = 1;
    }
}

std::optional<InferenceEngine::Output> InferenceEngine::infer(const float* input, size_t input_len) {
    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input),  // ORT C API is non-const; we don't mutate.
            input_len,
            input_shape_.data(),
            input_shape_.size()
        );

        const char* input_names[]  = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};

        auto output_tensors = session_.Run(Ort::RunOptions{nullptr},
                                           input_names, &input_tensor, 1,
                                           output_names, 1);
        last_output_ = std::move(output_tensors[0]);
    } catch (const Ort::Exception&) {
        return std::nullopt;
    }

    auto shape = last_output_.GetTensorTypeAndShapeInfo().GetShape();
    // Expected: [1, N, K] — squeeze the leading batch dim.
    if (shape.size() != 3 || shape[0] != 1) {
        std::string actual;
        for (std::size_t i = 0; i < shape.size(); ++i) {
          actual += ((i != 0U) ? "," : "") + std::to_string(shape[i]);
        }
        throw Status::FatalException(Status::Fatal{
            .origin = Status::Stage::Inference,
            .cause = std::format(
                "unexpected model output shape [{}]: expected [1,N,K]",
                actual)});
    }
    return Output{.data_ptr = last_output_.GetTensorData<float>(),
                  .rows = shape[1],
                  .cols = shape[2]};
}
