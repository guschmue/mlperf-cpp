
#include <algorithm>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "backend.h"
#include "status.h"

namespace mlperf_bench {

Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mlperf_bench"};
const Ort::RunOptions run_options(nullptr);
auto allocator_info =
    Ort::AllocatorInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
Ort::Allocator allocator = Ort::Allocator(nullptr).CreateDefault();

Backend::Backend() {}

Status Backend::LoadModel(std::string path, std::vector<std::string> outputs) {
    Ort::SessionOptions opt;
    opt.SetGraphOptimizationLevel(3);

#ifdef _WIN32
    std::wstring widestr = std::wstring(path.begin(), path.end());
    session_ = new Ort::Session(env, widestr.c_str(), opt);
#else
    session_ = new Ort::Session(env, path.c_str(), opt);
#endif
    for (size_t i = 0; i < this->session_->GetInputCount(); i++) {
        input_names_.push_back(session_->GetInputName(i, allocator));
    }
    for (size_t i = 0; i < this->session_->GetOutputCount(); i++) {
        char* name = session_->GetOutputName(i, allocator);
        if (outputs.size() == 0 ||
            std::find(outputs.begin(), outputs.end(), name) != outputs.end()) {
            auto ti =
                session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
            auto shape = ti.GetShape();
            output_shapes_.push_back(shape);
            output_names_.push_back(name);
        }
    }
    return Status::OK();
}

std::vector<Ort::Value> Backend::Run(ptensor_t* inputs, size_t input_count) {
    std::vector<int64_t>& shapes = std::get<0>(*inputs);
    std::vector<float>& data = std::get<1>(*inputs);
    Ort::Value t = Ort::Value::CreateTensor<float>(
        allocator_info, data.data(), data.size(), shapes.data(), shapes.size());

    std::vector<Ort::Value> results =
        session_->Run(run_options, input_names_.data(), &t, 1,
                      output_names_.data(), output_names_.size());

    return results;
}
}
