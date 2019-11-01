// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT license.

//
// A C++ implementation for mlperf inference benchmarks using onnxruntime
//

#include <chrono>
#include <exception>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

// loadgen includes
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"

// local includes
#include "cxxopts.hpp"
#include "backend.h"
#include "loadgen.h"
#include "mlperf_bench.h"
#include "npy.h"


namespace mlperf_bench {

class Exception : public std::exception {
public:
    Exception(std::string err) {
        err_ = err;
    }
    virtual const char* what() const throw() { return err_.c_str();  }

private:
    std::string err_;
};

//
// Dump tensors for debugging
//
void Dbg(Ort::Value& t) {
    auto type_info = t.GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    float* p = t.GetTensorMutableData<float>();
    std::cout << "shape: [";
    for (auto s : shape) {
        std::cout << s << ",";
    }
    std::cout << "], type: " << type_info.GetElementType() << ", items: " << type_info.GetElementCount() << " ";
    std::cout << *(p + 0) << " " << *(p + 1) << " ";
    std::cout << "\n";
}

//
// function prototype for post processors
//
typedef void (*post_processor_t)(std::vector<Ort::Value> &,
                                 std::vector<std::vector<uint8_t>> &,
                                 const std::vector<mlperf::QuerySample> &);

//
// Argmax Post processor
//
void PostProcess_Argmax(std::vector<Ort::Value> &val,
    std::vector<std::vector<uint8_t>> &buf,
    const std::vector<mlperf::QuerySample> &samples) {
    Ort::Value &r = val[0];
    auto type_info = r.GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    int64_t *p = r.GetTensorMutableData<int64_t>();
    for (int batch = 0; batch < shape[0]; batch++) {
        float argmax = (float)*p++ - 1;
        std::vector<uint8_t> ele((uint8_t *)&argmax, (uint8_t *)(&argmax + 1));
        buf.push_back(ele);
    }
}

//
// Common Post processor (model ran Argmax)
//
void PostProcess_Softmax(std::vector<Ort::Value> &val,
    std::vector<std::vector<uint8_t>> &buf,
    const std::vector<mlperf::QuerySample> &samples) {
    Ort::Value &r = val[0];
    auto type_info = r.GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    float *p = r.GetTensorMutableData<float>();
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


//
// Post processor for tensorflow object detection models
//
void PostProcess_TF_SSDMobilenet(std::vector<Ort::Value> &val,
    std::vector<std::vector<uint8_t>> &buf,
    const std::vector<mlperf::QuerySample> &samples) {
    Ort::Value &num_detections_val = val[0];
    float* detection_boxes = val[1].GetTensorMutableData<float>();
    float* detection_scores = val[2].GetTensorMutableData<float>();
    float* detection_classes = val[3].GetTensorMutableData<float>();

    auto type_info = num_detections_val.GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    float *num_detections = num_detections_val.GetTensorMutableData<float>();

    for (int batch = 0; batch < shape[0]; batch++) {
        std::vector<uint8_t> ele(int(num_detections[batch])*7*sizeof(float));
        float* result = (float*)ele.data();
        for (int detection = 0; detection < int(num_detections[batch]); detection++) {
            result[0] = float(samples[batch].index);
            result[1] = detection_boxes[4* detection + 0];
            result[2] = detection_boxes[4 * detection + 1];
            result[3] = detection_boxes[4 * detection + 2];
            result[4] = detection_boxes[4 * detection + 3];
            result[5] = detection_scores[detection];
            result[6] = float(detection_classes[detection]);
            result += 7;
        }
        buf.push_back(ele);
    }
}


//
// map string to post processor
//
std::map<std::string, post_processor_t> post_processors = {
    {"ArgMax", PostProcess_Argmax}, 
    {"SoftMax", PostProcess_Softmax},
    {"TF_SSDMobilenet", PostProcess_TF_SSDMobilenet},
};


//
// our flavor of QuerySampleLibrary that lets us request a item by index 
//
class QuerySampleLibrary : public mlperf::QuerySampleLibrary {
   public:
    virtual Ort::Value& GetItem(size_t idx) = 0;
};


//
// implementation of QuerySampleLibrary for real data
//
template <class T>
class Qsl : public QuerySampleLibrary {
   public:
    Qsl(Backend *be, std::string path, size_t count) { 
        be_ = be;
        FromDir(path, count); 
    }

    const std::string &Name() const override { return name_; }

    size_t TotalSampleCount() override { return files_.size(); }

    size_t PerformanceSampleCount() override { return files_.size(); }

    void LoadSamplesToRam(
        const std::vector<mlperf::QuerySampleIndex> &samples) override {
        loaded_.clear();
        dummy_.clear();
        for (auto s : samples) {
            LoadItem(s);
        }
    }

    void UnloadSamplesFromRam(
        const std::vector<mlperf::QuerySampleIndex> &samples) override {
        loaded_.clear();
        dummy_.clear();
    }

    Ort::Value& GetItem(size_t idx) override { return loaded_.at(idx); };

   private:
    int FromDir(std::string path, size_t count) {
        files_.clear();
#if 0
        // recurse through the directory. We are not using this because 
        // it is hard to guarante the same order that the accuracy script would use.
        for (auto &p : fs::recursive_directory_iterator(path)) {
            if (fs::is_regular_file(p.path())) {
                std::string pp(p.path().string());
                files_.push_back(pp);
                if (count > 0 && files_.size() > count) break;
            }
        }
        std::sort(files_.begin(), files_.end());
#else
        // we use this version to guarante order since the QSL index need be the index
        // into the file list so accuracy can be calculated.
        std::ifstream infile(path);
        if (!infile.is_open()) {
            throw Exception("no validation file list found");
        }
        std::string line;
        std::string basepath = path.substr(0, path.find_last_of("/\\"));
        while (std::getline(infile, line)) {
            std::string file_name = line.substr(0, line.find_last_of(" \t")) + ".npy";
            files_.push_back(basepath + "/" + file_name);
            if (count > 0 && files_.size() > count) break;
        }
#endif
        return 0;
    }

    int LoadItem(size_t idx) {
        std::vector<unsigned long> shape;
        std::vector<T> data;
        npy::LoadArrayFromNumpy<T>(files_[idx], shape, data);
        std::vector<int64_t> shapes;
        shapes.push_back(1);
        for (auto i : shape) {
            shapes.push_back(i);
        }
        loaded_.emplace(idx, be_->GetTensor<T>(shapes, data));
        dummy_.emplace_back(std::move(data));
        return 0;
    }

    const std::string name_{"QSL"};
    std::vector<std::string> files_;
    std::map<size_t, Ort::Value> loaded_;
    std::vector <std::vector<T>> dummy_;
    Backend *be_;
};

//
// implementation of QuerySampleLibrary for fake data
//
template <class T>
class FakeQsl : public QuerySampleLibrary {
   public:
    FakeQsl(Backend *be, std::string dataset) {
        // for now its all imagenet
        be_ = be;
        std::vector<int64_t> shape;
        if (dataset == "imagenet" || dataset == "imagenet_mobilenet") {
            shape = { 1, 3, 224, 224 };
        }
        else if (dataset == "coco-300") {
            // comes as NHWC
            shape = { 1, 300, 300, 3 };
        }
        else if (dataset == "coco-1200") {
            shape = { 1, 3, 1200, 1200 };
        }
        else {
            throw Exception("unknown dataset type");
        }
        size_ = 1;
        shape_ = shape;
        for (auto i : shape) {
            size_ *= i;
        }
        for (int i = 0; i < size_; i++) {
            data_.push_back((float)(i & 255));
        }
        Ort::Value t = be->GetTensor<T>(shape_, data_);
        template_ = std::move(t);
    }

    const std::string &Name() const override { return name_; }

    size_t TotalSampleCount() override { return count_; }

    size_t PerformanceSampleCount() override { return count_; }

    void LoadSamplesToRam(
        const std::vector<mlperf::QuerySampleIndex> &samples) override {}

    void UnloadSamplesFromRam(
        const std::vector<mlperf::QuerySampleIndex> &samples) override {}

    Ort::Value& GetItem(size_t idx) { return template_; };

   private:
    const std::string name_{"FakeQSL"};
    std::vector<T> data_;
    const size_t count_ = 500;
    size_t size_;
    std::vector<int64_t> shape_;
    Ort::Value template_{ nullptr };
    Backend *be_;
};

//
// SystemUnderTest single threaded implementation
//
class SystemUnderTest : public mlperf::SystemUnderTest {
   public:
    SystemUnderTest(QuerySampleLibrary *qsl, Backend *be,
                    post_processor_t post_proc, int threads,
                    int max_batchsize) {
        qsl_ = qsl;
        be_ = be;
        post_proc_ = post_proc;
        threads_ = threads;
        max_batchsize_ = max_batchsize;
        input_type_ = be->GetInputType(0);
    }

    void IssueQuery(const std::vector<mlperf::QuerySample> &samples) override {
        if (input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            IssueQueryProc<float>(samples);
        }
        else if (input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            IssueQueryProc<uint8_t>(samples);
        }
    }

    template <typename T>
    void IssueQueryProc(const std::vector<mlperf::QuerySample> &samples) {
        std::vector<mlperf::QuerySampleResponse> responses;
        responses.reserve(samples.size());
        std::vector<std::vector<uint8_t>> dummy;

        std::vector<Ort::Value> results;
        if (samples.size() == 1) {
            // we have 1 query to run
            Ort::Value& q = qsl_->GetItem(samples[0].index);
            results = be_->Run(&q, 1);
        }
        else {
            // FIXME: this is not very efficient. 
            // loadgen will make the samples continues in the near future and once it does
            // change this code to just point into the continues data buffer.
            Ort::Value& qq = qsl_->GetItem(samples[0].index);
            std::vector<int64_t> shapes;
            auto type_info = qq.GetTensorTypeAndShapeInfo();
            auto shape = type_info.GetShape();
            auto elements = type_info.GetElementCount();
            int64_t size = 1;
            shape[0] = samples.size();
            for (auto si : shape) {
                size *= si;
                shapes.push_back(si);
            }
            std::vector<T> data(size);
            size_t idx = 0;
            T *dst = data.data();
            for (auto &s : samples) {
                Ort::Value& r = qsl_->GetItem(s.index);
                T *src = r.GetTensorMutableData<T>();
                ::memcpy(dst, src, sizeof(T) * elements);
                dst += elements;
            }
            Ort::Value q = be_->GetTensor<T>(shapes, data);
            results = be_->Run(&q, 1);
        }

        std::vector<std::vector<uint8_t>> buf;
        post_proc_(results, buf, samples);
        size_t idx = 0;
        for (auto &b : buf) {
            // hold a reference so the buffer doesn't get released before
            // QuerySamplesComplete
            dummy.push_back(b);
            responses.push_back(
                {samples[idx].id, (uintptr_t)dummy.back().data(), b.size()});
            idx++;
        }
        mlperf::QuerySamplesComplete(responses.data(), responses.size());
    }

    void FlushQueries() override {}

    void ReportLatencyResults(
        const std::vector<mlperf::QuerySampleLatency> &latencies_ns) override {}

    const std::string &Name() const override { return name_; }

   protected:
    const std::string name_{"Sut"};
    QuerySampleLibrary *qsl_;
    Backend *be_;
    post_processor_t post_proc_;
    int max_batchsize_ = 32;
    int threads_ = 1;
    ONNXTensorElementDataType input_type_;
};

//
// SystemUnderTest multi threaded implementation
//
class SystemUnderTestPool : public SystemUnderTest {
   public:
    SystemUnderTestPool(QuerySampleLibrary *qsl, Backend *be,
                        post_processor_t post_proc, int threads,
                        int max_batchsize)
        : SystemUnderTest(qsl, be, post_proc, threads, max_batchsize) {
        samples_.reserve(kReserveSampleSize);
        next_poll_time_ =
            std::chrono::high_resolution_clock::now() + poll_period_;
        for (int i = 0; i < threads; i++) {
            thread_pool_.emplace_back(&SystemUnderTestPool::WorkerThread, this);
        }
    }

    ~SystemUnderTestPool() override {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            keep_workers_alive_ = false;
        }
        cv_.notify_all();
        for (auto &thread : thread_pool_) {
            thread.join();
        }
    }

    void IssueQuery(const std::vector<mlperf::QuerySample> &samples) override {
        std::unique_lock<std::mutex> lock(mutex_);
        samples_.insert(samples_.end(), samples.begin(), samples.end());
    }

   protected:
    void WorkerThread() {
        std::vector<mlperf::QuerySample> samples;
        samples.reserve(kReserveSampleSize);
        std::unique_lock<std::mutex> lock(mutex_);
        while (keep_workers_alive_) {
            next_poll_time_ += poll_period_;
            auto my_wakeup_time = next_poll_time_;
            cv_.wait_until(lock, my_wakeup_time,
                           [&]() { return !keep_workers_alive_; });
            if (samples_.size() <= max_batchsize_) {
                // if we can fit in one batch, take all
                samples.swap(samples_);
            } else {
                // take only as much as fits into one batch
                auto it = std::next(samples_.begin(), max_batchsize_);
                std::move(samples_.begin(), it, std::back_inserter(samples));
                samples_.erase(samples_.begin(), it);
            }
            lock.unlock();

            if (samples.size() > 0) {
                if (input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                    IssueQueryProc<float>(samples);
                }
                else if (input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
                    IssueQueryProc<uint8_t>(samples);
                }
            }
            lock.lock();
            samples.clear();
        }
    }

    const std::string name_{"SUTPool"};
    static constexpr size_t kReserveSampleSize = 1 * 512;
    const std::chrono::milliseconds poll_period_{1};
    std::chrono::high_resolution_clock::time_point next_poll_time_;

    std::mutex mutex_;
    std::condition_variable cv_;
    bool keep_workers_alive_ = true;
    std::vector<std::thread> thread_pool_;
    std::vector<mlperf::QuerySample> samples_;
};

//
// setup and start loadgen
//
void run(std::string model, std::string datadir,
         std::map<std::string, std::string> profile,
         mlperf::TestSettings &settings, int count, int threads, int max_batchsize,
         int ort_seq, int ort_interop, int ort_intraop) {
    Backend be;

    // setup backend options
    be.GetOpt().SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    if (ort_interop > 0) {
        be.GetOpt().SetInterOpNumThreads(ort_interop);
    }
    if (ort_intraop > 0) {
        be.GetOpt().SetIntraOpNumThreads(ort_intraop);
    }
    be.LoadModel(model, {});

    mlperf::LogSettings log_settings;
    log_settings.log_output.copy_summary_to_stdout = true;

    ONNXTensorElementDataType input_type = be.GetInputType(0);

    QuerySampleLibrary *qsl = NULL;
    if (!datadir.empty()) {
        // use Fake data Qsl
        if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            qsl = new Qsl<float>(&be, datadir, count);
        }
        else if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            qsl = new Qsl<uint8_t>(&be, datadir, count);
        }
        else {
            throw Exception("unsupported input_type");
        }
    } else {
        if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            qsl = new FakeQsl<float>(&be, profile["dataset"]);
        }
        else if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            qsl = new FakeQsl<uint8_t>(&be, profile["dataset"]);
        }
        else {
            throw Exception("unknown input_type");
            // FIXME
        }
    }

    // warmup
    qsl->LoadSamplesToRam({0});
    for (int i = 0; i < 10; i++) {
        Ort::Value& q = qsl->GetItem(0);
        std::vector<Ort::Value> results = be.Run(&q, 1);
    }

    std::cout << qsl->PerformanceSampleCount() << " items loaded.\n";

    post_processor_t post_proc = post_processors[profile["post_process"]];

    if (settings.scenario == mlperf::TestScenario::SingleStream ||
        settings.scenario == mlperf::TestScenario::MultiStream) {
        SystemUnderTest sut(qsl, &be, post_proc, threads, max_batchsize);
        mlperf::StartTest(&sut, qsl, settings, log_settings);
    } else {
        SystemUnderTestPool sut(qsl, &be, post_proc, threads, max_batchsize);
        mlperf::StartTest(&sut, qsl, settings, log_settings);
    }
}

//
// model specific parameters, like inputs, outputs ...
//
std::map<std::string, std::map<std::string, std::string>> profiles = {
    {"resnet50",
     {
         {"dataset", "imagenet"},
         {"backend", "onnxruntime"},
         {"inputs", "input_tensor:0"},
         {"outputs", "ArgMax:0"},
         {"post_process", "ArgMax"},
     }},
    {"mobilenet",
     {
         {"dataset", "imagenet_mobilenet"},
         {"backend", "onnxruntime"},
         {"inputs", "input:0"},
         {"outputs", "MobilenetV1/Predictions/Reshape_1:0"},
         {"post_process", "SoftMax"},
     }},
    {"ssd-mobilenet",
     {
         {"dataset", "coco-300"},
         {"backend", "onnxruntime"},
         {"inputs", "image_tensor:0"},
         {"outputs", "num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0"},
         {"post_process", "TF_SSDMobilenet"},
     }},
    {"ssd-resnet34",
     {
         {"dataset", "coco-1200"},
         {"backend", "onnxruntime"},
         {"inputs", "image"},
         {"outputs", "bboxes,labels,scores"},
         {"post_process", "SSDResnet34"},
     }},
};


//
// mapping string -> scenario
//
std::map<std::string, mlperf::TestScenario> scenario_map = {
    {"SingleStream", mlperf::TestScenario::SingleStream},
    {"MultiStream", mlperf::TestScenario::MultiStream},
    {"Server", mlperf::TestScenario::Server},
    {"Offline", mlperf::TestScenario::Offline},
};

//
// mapping string -> mode
//
std::map<std::string, mlperf::TestMode> mode_map = {
    {"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly},
    {"PerformanceOnly", mlperf::TestMode::PerformanceOnly},
    {"FindPeakPerformance", mlperf::TestMode::FindPeakPerformance},
};


int main(int argc, char *argv[]) {
    cxxopts::Options options("mlperf_bench", "mlperf_bench");
    options.add_options()
        ("model", "model to load", 
            cxxopts::value<std::string>()->default_value(""))
        ("config", "mlperf rules config file",
            cxxopts::value<std::string>()->default_value("mlperf.conf"))
        ("scenario", "scenario to load (SingleStream,MultiStream,Server,Offline)",
            cxxopts::value<std::string>()->default_value("SingleStream"))
        ("mode", "mode (PerformanceOnly,AccuracyOnly,FindPeakPerformance)",
            cxxopts::value<std::string>()->default_value("PerformanceOnly"))
        ("data", "data file to load",
            cxxopts::value<std::string>()->default_value(""))
        ("fake", "use fake data")
        ("profile", "profile to load",
            cxxopts::value<std::string>()->default_value("resnet50"))
        ("time", "time to run", 
            cxxopts::value<int32_t>()->default_value("0"))
        ("threads", "threads", 
            cxxopts::value<int32_t>()->default_value("2"))
        ("max-batchsize", "max-batchsize",
            cxxopts::value<int32_t>()->default_value("32"))
        ("qps", "qps", 
            cxxopts::value<int32_t>()->default_value("20"))
        ("latency", "latency (ms)",
            cxxopts::value<int32_t>()->default_value("10"))
        ("samples-perf-query", "samples-per-query",
            cxxopts::value<int32_t>()->default_value("2"))
        ("count", "count", 
            cxxopts::value<int32_t>()->default_value("0"))
        ("ort-seq", "onnxruntime use sequential executor",
            cxxopts::value<int32_t>()->default_value("0"))
        ("ort-interop", "onnxruntime interop thread count",
            cxxopts::value<int32_t>()->default_value("0"))
        ("ort-intraop", "onnxruntime intraop thread count",
            cxxopts::value<int32_t>()->default_value("0"))
        ("help", "help");

    try {
        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help({ "", "Group" }) << std::endl;
            exit(0);
        }
        if (result.count("model") == 0) {
            std::cout << "specify model with --model path" << std::endl;
            exit(1);
        }
        if (result.count("data") == 0) {
            std::cout << "no data file given - using fake data" << std::endl;
        }
        std::string data = result["data"].as<std::string>();
        if (result.count("fake") != 0) {
            std::cout << "using fake data" << std::endl;
            data.empty();
        }
        std::map<std::string, std::string> profile = profiles[result["profile"].as<std::string>()];
        if (profile.empty()) {
            std::cout << "invalid profile" << std::endl;
            exit(1);
        }
        std::string scenario = result["scenario"].as<std::string>();
        mlperf::TestSettings settings;
        if (settings.FromConfig(result["config"].as<std::string>(), result["profile"].as<std::string>(), scenario)) {
            std::cout << "issue with config " << result["config"].as<std::string>() << std::endl;
            exit(1);
        }
        settings.scenario = scenario_map[scenario];
        settings.mode = mode_map[result["mode"].as<std::string>()];

        int entries_to_read = 0;
        int count = result["count"].as<int32_t>();
        if (settings.mode == mlperf::TestMode::PerformanceOnly) {
            if (count > 500 || count == 0) {
                // in performance mode we use only 500 unique images
                entries_to_read = 500;
            }
        }
        if (settings.scenario == mlperf::TestScenario::MultiStream) {
            if (result.count("samples-perf-query")) {
                settings.multi_stream_samples_per_query =
                    result["samples-perf-query"].as<int32_t>();
            }
        }
        if (settings.scenario == mlperf::TestScenario::Server) {
            settings.server_target_qps = result["qps"].as<int32_t>();
            if (result.count("latency")) {
                settings.server_target_latency_ns =
                    result["latency"].as<int32_t>() * 1000 * 1000;
            }
        }
        if (count > 0) {
            entries_to_read = count;
            settings.min_query_count = count * 5;
            settings.max_query_count = count * 5;
        }
        int time = result["time"].as<int32_t>();
        if (time > 0) {
            settings.min_duration_ms = time * 1000;
            settings.max_duration_ms = time * 1000;
        }

        int ort_seq = result["ort-seq"].as<int32_t>();
        int ort_interop = result["ort-interop"].as<int32_t>();
        int ort_intraop = result["ort-intraop"].as<int32_t>();

        run(result["model"].as<std::string>(),
            data, profile, settings,
            entries_to_read, result["threads"].as<int32_t>(),
            result["max-batchsize"].as<int32_t>(),
            ort_seq, ort_interop, ort_intraop);
    } catch (const cxxopts::OptionException &e) {
        std::cout << "argument error: " << e.what() << std::endl;
        exit(1);
    } catch (const std::exception &e) {
        std::cout << "error: " << e.what() << std::endl;
        exit(1);
    };

    return 0;
}
}  // namespace mlperf_bench

// out of namespace

int main(int argc, char *argv[]) { 
    return mlperf_bench::main(argc, argv); 
}
