#include <algorithm>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "backend.h"
#include "status.h"

namespace mlperf_bench {

Ort::Env env { ORT_LOGGING_LEVEL_WARNING, "mlperf_bench" };

Backend::Backend() {
    OrtGetAllocatorWithDefaultOptions(&allocator_);
};

Status Backend::LoadModel(std::string path, std::vector<std::string> outputs) {
#ifdef _WIN32
    std::wstring widestr = std::wstring(path.begin(), path.end());
    session_ = new Ort::Session(env, widestr.c_str(), opt_);
#else
    session_ = new Ort::Session(env, path.c_str(), opt_);
#endif
    for (size_t i = 0; i < this->session_->GetInputCount(); i++) {
        input_names_.push_back(session_->GetInputName(i, allocator_));
        auto ti = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto input_type = ti.GetElementType();

        // FIXME: ti.GetElementType() returns junk on linux. Hack it for now.
        if (path.find("ssd") != std::string::npos) {
            input_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        }
        else {
            input_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        }
        input_type_.push_back(input_type);
    }
    for (size_t i = 0; i < this->session_->GetOutputCount(); i++) {
        char* name = session_->GetOutputName(i, allocator_);
        if (outputs.size() == 0 ||
            std::find(outputs.begin(), outputs.end(), name) != outputs.end()) {
            auto ti = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
            auto shape = ti.GetShape();
            output_shapes_.push_back(shape);
            output_names_.push_back(name);
        }
    }
    return Status::OK();
}

std::vector<Ort::Value> Backend::Run(Ort::Value* inputs, size_t input_count) {
    std::vector<Ort::Value> results =
        session_->Run(run_options_, input_names_.data(), inputs, 1,
                      output_names_.data(), output_names_.size());
    return results;
}

}
