#include <map>
#include <string>
#include <vector>
#include "mlperf_bench.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "status.h"

namespace mlperf_bench {

class Backend {
   public:
    Backend();
    Status LoadModel(std::string path, std::vector<std::string> outputs);
    std::vector<Ort::Value> Run(Ort::Value* inputs, size_t input_count);

    template <typename T>
    Ort::Value GetTensor(std::vector<int64_t>& shapes, std::vector<T>& data) {
        return Ort::Value::CreateTensor<T>(
            allocator_info_, data.data(), data.size(), shapes.data(), shapes.size());
    };
    ONNXTensorElementDataType GetInputType(size_t idx) { return input_type_[idx]; };
    Ort::SessionOptions& GetOpt() { return opt_; };

   private:
    Ort::Session* session_;
    Ort::SessionOptions opt_;
    std::vector<std::vector<int64_t>> output_shapes_;
    std::vector<char*> output_names_;
    std::vector<char*> input_names_;
    std::vector<ONNXTensorElementDataType> input_type_;
    const Ort::RunOptions run_options_; //(nullptr);
    Ort::AllocatorInfo allocator_info_ =
        Ort::AllocatorInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    OrtAllocator* allocator_;
};
}
