#include <getopt.h>
#include <chrono>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <tuple>
#include <future>
#include <filesystem>

#include "cxxopts.hpp"

#include "mlperf_bench.h"
#include "thread_queue.h"
#include "backend.h"
#include "loadgen.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"
#include "npy.h"

namespace fs = std::filesystem;


namespace mlperf_bench {
 

class Item {
public:
    int64_t query_id;
    int64_t content_id;
    std::chrono::nanoseconds start;
    // data and label are mapped in thread
};

class Runner {

public:
    void handle_tasks() {
    }

    void start_run(std::map<std::string, std::string> result_dict, bool take_accuracy) {

    }

    void run_one_item(Item& qitem) {
    }

    void enqueue(const std::vector<mlperf::QuerySample>& query_samples) {
        #if 0
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        if len(query_samples) < self.max_batchsize:
            data, label = self.ds.get_samples(idx)
            self.run_one_item(Item(query_id, idx, data, label))
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                data, label = self.ds.get_samples(idx[i:i+bs])
                self.run_one_item(Item(query_id[i:i+bs], idx[i:i+bs], data, label))
        #endif
    }

    void finish() {
    }

private:
   // const Dataset& ds;
    bool take_accuracy = false;
    Backend& model;
    std::future<void> post_process;
    int threads = 1;
    int max_batchsize = 32;
};

typedef void(*post_processor_t)(std::vector<Ort::Value>&, std::vector<std::vector<uint8_t>>&);

void PostProcess_Argmax(std::vector<Ort::Value>& val, std::vector<std::vector<uint8_t>>& buf) {
    // for ArgMax we only care about the first result
    Ort::Value& r = val[0];
    auto type_info = r.GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    size_t total_len = type_info.GetElementCount();
    size_t batch_Len = total_len / shape[0];
    float* p = r.GetTensorMutableData<float>();
    for (int batch = 0; batch < shape[0]; batch++) {
        float argmax = -1;
        float last_val = -1;
        for (int i = 0; i < shape[1]; i++, p++) {
            if (*p > last_val) {
                argmax = i - 1;
                last_val = *p;
            }
        }
        std::vector<uint8_t> ele((uint8_t *)&argmax, (uint8_t *)(&argmax + 1));
        buf.push_back(ele);
    }
}

std::map<std::string, post_processor_t> post_processors = {
    {"ArgMax", PostProcess_Argmax},
};


class QuerySampleLibrary : public mlperf::QuerySampleLibrary {
public:
    QuerySampleLibrary(Backend *be) {

    }

    const std::string& Name() const override { return name_; }

    size_t TotalSampleCount() override { return files_.size(); }

    size_t PerformanceSampleCount() override { return files_.size(); }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
        for (auto s : samples) {
            LoadItem(s);
        }
        std::cout << "loaded.\n";
    }

    void UnloadSamplesFromRam(
        const std::vector<mlperf::QuerySampleIndex>& samples) override {
        loaded_.clear();
    }

    int FromDir(std::string path, size_t count) {
        files_.clear();
        for (auto& p : fs::recursive_directory_iterator(path)) {
            if (fs::is_regular_file(p.path())) {
                std::string pp(p.path().string());
                files_.push_back(pp);
                if (count > 0 && files_.size() > count)
                    break;
            }
        }
        std::sort(files_.begin(), files_.end());
        return 0;
    }

    int LoadItem(size_t idx) {
        std::vector<unsigned long> shape;
        std::vector<float> data;
        npy::LoadArrayFromNumpy<float>(files_[idx], shape, data);
        std::vector<int64_t> shapes;
        shapes.push_back(1);
        for (auto i : shape) {
            shapes.push_back(i);
        }
        loaded_[idx] = std::make_tuple(shapes, data);
        return 0;
    }

    ptensor_t GetItem(size_t idx) { return loaded_[idx]; };

private:
    std::string name_{ "QSL" };
    std::vector<std::string> files_;
    std::map<size_t, ptensor_t> loaded_;
};


class SystemUnderTest : public mlperf::SystemUnderTest {

public:
    SystemUnderTest(QuerySampleLibrary *qsl, Backend *be, post_processor_t post_proc, int max_batchsize) {
        qsl_ = qsl;
        be_ = be;
        post_proc_ = post_proc;
        max_batchsize_ = max_batchsize;
    }

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
        std::vector<mlperf::QuerySampleResponse> responses;
        responses.reserve(samples.size());
        std::vector<std::vector<uint8_t>> dummy;
        for (auto& s : samples) {
            ptensor_t q = qsl_->GetItem(s.index);
            std::vector<Ort::Value> results = be_->Run(&q, 1);
            std::vector<std::vector<uint8_t>> buf;
            post_proc_(results, buf);
            for (auto& b : buf) {
                // hold a reference so the buffer doesn't get released before QuerySamplesComplete
                dummy.push_back(b);
                responses.push_back({ s.id, (uintptr_t)dummy.back().data(), b.size() });
            }
        }
        mlperf::QuerySamplesComplete(responses.data(), responses.size());
    }

    void FlushQueries() override {
    }

    void ReportLatencyResults(
        const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override {
    }

    const std::string& Name() const override { return name_; }

protected:
    QuerySampleLibrary* qsl_;
    Backend* be_;
    post_processor_t post_proc_;
    std::string name_{ "QSL" };
    int max_batchsize_ = 32;
};


void run(std::string model, 
    std::string datadir, 
    mlperf::TestScenario scenario, 
    mlperf::TestMode mode,
    std::map<std::string, std::string> profile,
    int count, 
    int time, 
    int threads,
    int max_batchsize)
{
    Backend be;
    be.LoadModel(model, {});

    mlperf::LogSettings default_log_settings;
    mlperf::TestSettings settings;
    settings.scenario = scenario;
    settings.mode = mode;
    settings.server_target_qps = 2000000;

    if (time > 0) {
        settings.min_duration_ms = time * 1000;
        settings.max_duration_ms = time * 1000;
    }

    int entries_to_read = 0;
    if (count > 0) {
        entries_to_read = count;
        settings.min_query_count = count;
        settings.max_query_count = count;
    }
    else {
        if (mode == mlperf::TestMode::PerformanceOnly) {
            entries_to_read = 500;
        }
    }

    QuerySampleLibrary qsl(&be);
    qsl.FromDir(datadir, entries_to_read);

    // warmup
    qsl.LoadItem(0);
    ptensor_t q = qsl.GetItem(0);
    for (int i = 0; i < 10; i++) {
        std::vector<Ort::Value> results = be.Run(&q, 1);
    }

    post_processor_t post_proc = post_processors[profile["post_process"]];
    SystemUnderTest sut(&qsl, &be, post_proc, max_batchsize);

    mlperf::StartTest(&sut, &qsl, settings, default_log_settings);
}


std::map<std::string, std::map<std::string, std::string>> profiles = { {
    "resnet50-onnxruntime",
    {
        {"dataset", "imagenet"},
        {"queries-single", "1024"},
        {"queries-multi", "24576"},
        {"max-latency", "0.010"},
        {"qps", "10"},
        {"backend", "onnxruntime"},
        {"inputs", "input_tensor:0"},
        {"outputs", "ArgMax:0"},
        {"post_process", "ArgMax"},
    },
} };

std::map<std::string, mlperf::TestScenario> scenario_map = {
    {"SingleStream", mlperf::TestScenario::SingleStream},
    {"MultiStream", mlperf::TestScenario::MultiStream},
    {"Server", mlperf::TestScenario::Server},
    {"Offline", mlperf::TestScenario::Offline},
};

std::map<std::string, mlperf::TestMode> mode_map = {
    {"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly},
    {"PerformanceOnly", mlperf::TestMode::PerformanceOnly},
};


int main(int argc, char *argv[]) {

    cxxopts::Options options("mlperf_bench", "mlperf_bench");
    options.add_options()
        ("model", "model to load", cxxopts::value<std::string>()->default_value(""))
        ("scenario", "scenario to load", cxxopts::value<std::string>()->default_value("MultiStream"))
        ("mode", "mode", cxxopts::value<std::string>()->default_value("PerformanceOnly"))
        ("datadir", "datadir to load", cxxopts::value<std::string>()->default_value(""))
        ("profile", "profile to load", cxxopts::value<std::string>()->default_value("resnet50-onnxruntime"))
        ("time", "time to run", cxxopts::value<int32_t>()->default_value("0"))
        ("threads", "threads", cxxopts::value<int32_t>()->default_value("2"))
        ("max-batchsize", "max-batchsize", cxxopts::value<int32_t>()->default_value("32"))
        ("count", "count", cxxopts::value<int32_t>()->default_value("0"))
        ("help", "help");

    try {
        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help({ "", "Group" }) << std::endl;
            exit(0);
        }

        run(result["model"].as<std::string>(),
            result["datadir"].as<std::string>(),
            scenario_map[result["scenario"].as<std::string>()],
            mode_map[result["mode"].as<std::string>()],
            profiles[result["profile"].as<std::string>()],
            result["count"].as<int32_t>(),
            result["time"].as<int32_t>(),
            result["threads"].as<int32_t>(),
            result["max-batchsize"].as<int32_t>());

    }
    catch (const cxxopts::OptionException& e)
    {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }

 

    return 0;
}

}

// out of namespace

int main(int argc, char *argv[]) {
    return mlperf_bench::main(argc, argv);
}

// build on windows: cmake ..  -A x64 -G "Visual Studio 15 2017"
// --model w:\resnet_for_mlperf\mobilenet_v1_1.0_224.onnx --datadir w:\inference\v0.5\classification_and_detection\preprocessed\imagenet_mobilenet\NCHW
