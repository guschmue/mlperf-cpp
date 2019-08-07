#include <map>
#include <string>
#include <vector>
#include "mlperf_bench.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include "status.h"

namespace mlperf_bench {

class Backend {
   public:
    Backend();
    Status LoadModel(std::string path, std::vector<std::string> outputs);
    std::vector<Ort::Value> Run(ptensor_t* inputs, size_t input_count);

   private:
    Ort::Session* session_;
    std::vector<std::vector<int64_t>> output_shapes_;
    std::vector<char*> output_names_;
    std::vector<char*> input_names_;
};
}
