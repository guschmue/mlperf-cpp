#include <chrono>
#include <chrono>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <iostream>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#if defined(_WIN32)
#include <filesystem>
// namespace fs = std::filesystem;
namespace fs = std::experimental::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif
#include "cxxopts.hpp"

#include "backend.h"
#include "loadgen.h"
#include "mlperf_bench.h"
#include "npy.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"

namespace mlperf_bench {

typedef void (*post_processor_t)(std::vector<Ort::Value> &,
                                 std::vector<std::vector<uint8_t>> &);

void PostProcess_Argmax(std::vector<Ort::Value> &val,
                        std::vector<std::vector<uint8_t>> &buf) {
    Ort::Value &r = val[0];
    auto type_info = r.GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    int64_t *p = r.GetTensorMutableData<int64_t>();
    for (int batch = 0; batch < shape[0]; batch++) {
        float argmax = (float)*p++;
        std::vector<uint8_t> ele((uint8_t *)&argmax, (uint8_t *)(&argmax + 1));
        buf.push_back(ele);
    }
}

void PostProcess_Softmax(std::vector<Ort::Value> &val,
                         std::vector<std::vector<uint8_t>> &buf) {
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

std::map<std::string, post_processor_t> post_processors = {
    {"ArgMax", PostProcess_Argmax}, {"SoftMax", PostProcess_Softmax},
};

class QuerySampleLibrary : public mlperf::QuerySampleLibrary {
   public:
    virtual ptensor_t GetItem(size_t idx) = 0;
};

class Qsl : public QuerySampleLibrary {
   public:
    Qsl(Backend *be, std::string path, size_t count) { FromDir(path, count); }

    const std::string &Name() const override { return name_; }

    size_t TotalSampleCount() override { return files_.size(); }

    size_t PerformanceSampleCount() override { return files_.size(); }

    void LoadSamplesToRam(
        const std::vector<mlperf::QuerySampleIndex> &samples) override {
        for (auto s : samples) {
            LoadItem(s);
        }
    }

    void UnloadSamplesFromRam(
        const std::vector<mlperf::QuerySampleIndex> &samples) override {
        loaded_.clear();
    }

    ptensor_t GetItem(size_t idx) override { return loaded_[idx]; };

   private:
    int FromDir(std::string path, size_t count) {
        files_.clear();
        for (auto &p : fs::recursive_directory_iterator(path)) {
            if (fs::is_regular_file(p.path())) {
                std::string pp(p.path().string());
                files_.push_back(pp);
                if (count > 0 && files_.size() > count) break;
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

    const std::string name_{"QSL"};
    std::vector<std::string> files_;
    std::map<size_t, ptensor_t> loaded_;
};

class FakeQsl : public QuerySampleLibrary {
   public:
    FakeQsl(Backend *be, std::string dataset) {
        // for now its all imagenet
        std::vector<int64_t> shape = {1, 3, 224, 224};
        size_ = 1;
        shape_ = shape;
        for (auto i : shape) {
            size_ *= i;
        }
        std::vector<float> data;
        for (int i = 0; i < size_; i++) {
            data.push_back((float)(i & 255));
        }
        template_ = std::make_tuple(shape_, data);
    }

    const std::string &Name() const override { return name_; }

    size_t TotalSampleCount() override { return count_; }

    size_t PerformanceSampleCount() override { return count_; }

    void LoadSamplesToRam(
        const std::vector<mlperf::QuerySampleIndex> &samples) override {}

    void UnloadSamplesFromRam(
        const std::vector<mlperf::QuerySampleIndex> &samples) override {}

    ptensor_t GetItem(size_t idx) { return template_; };

   private:
    const std::string name_{"FakeQSL"};
    const size_t count_ = 500;
    size_t size_;
    std::vector<int64_t> shape_;
    ptensor_t template_;
};

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
    }
    void IssueQuery(const std::vector<mlperf::QuerySample> &samples) override {
        IssueQueryProc(samples);
    }

    void IssueQueryProc(const std::vector<mlperf::QuerySample> &samples) {
        std::vector<mlperf::QuerySampleResponse> responses;
        responses.reserve(samples.size());
        std::vector<std::vector<uint8_t>> dummy;
        ptensor_t q;
        bool got_one = false;
        for (auto &s : samples) {
            if (got_one) {
                ptensor_t q1 = qsl_->GetItem(s.index);
                std::get<0>(q)[0]++;
                std::get<1>(q).insert(std::get<1>(q).end(),
                                      std::get<1>(q1).begin(),
                                      std::get<1>(q1).end());
            } else {
                q = qsl_->GetItem(s.index);
                got_one = true;
            }
        }

        std::vector<Ort::Value> results = be_->Run(&q, 1);
        std::vector<std::vector<uint8_t>> buf;
        post_proc_(results, buf);
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
};

//
// Sut with thread pool
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
        std::vector<mlperf::QuerySample> my_samples;
        my_samples.reserve(kReserveSampleSize);
        std::unique_lock<std::mutex> lock(mutex_);
        while (keep_workers_alive_) {
            next_poll_time_ += poll_period_;
            auto my_wakeup_time = next_poll_time_;
            cv_.wait_until(lock, my_wakeup_time,
                           [&]() { return !keep_workers_alive_; });
            if (samples_.size() <= max_batchsize_) {
                // if we can fit in one batch, take all
                my_samples.swap(samples_);
            } else {
                // take only as much as fits into one batch
                auto it = std::next(samples_.begin(), max_batchsize_);
                std::move(samples_.begin(), it, std::back_inserter(my_samples));
                samples_.erase(samples_.begin(), it);
            }
            lock.unlock();

            if (my_samples.size() > 0) {
                IssueQueryProc(my_samples);
            }
            lock.lock();
            my_samples.clear();
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

void run(std::string model, std::string datadir,
         std::map<std::string, std::string> profile,
         mlperf::TestSettings &settings, int count, int threads,
         int max_batchsize) {
    Backend be;
    be.LoadModel(model, {});

    mlperf::LogSettings log_settings;
    log_settings.log_output.copy_summary_to_stdout = true;

    QuerySampleLibrary *qsl;
    if (!datadir.empty()) {
        qsl = new Qsl(&be, datadir, count);
    } else {
        qsl = new FakeQsl(&be, profile["dataset"]);
    }

    // warmup
    qsl->LoadSamplesToRam({0});
    ptensor_t q = qsl->GetItem(0);
    for (int i = 0; i < 10; i++) {
        std::vector<Ort::Value> results = be.Run(&q, 1);
    }

    std::cout << "data loaded.\n";

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

std::map<std::string, std::map<std::string, std::string>> profiles = {
    {"resnet50",
     {
         {"dataset", "imagenet"},
         {"queries-single", "1024"},
         {"queries-multi", "24576"},
         {"queries-server", "270336"},
         {"queries-offline", "1"},
         {"time-single", "60"},
         {"time-multi", "60"},
         {"time-server", "60"},
         {"time-offline", "60"},
         {"max-latency", "0.015"},
         {"qps", "10"},
         {"backend", "onnxruntime"},
         {"inputs", "input_tensor:0"},
         {"outputs", "ArgMax:0"},
         {"post_process", "ArgMax"},
     }},
    {"mobilenet",
     {
         {"dataset", "imagenet_mobilenet"},
         {"queries-single", "1024"},
         {"queries-multi", "24576"},
         {"queries-server", "270336"},
         {"queries-offline", "1"},
         {"time-single", "60"},
         {"time-multi", "60"},
         {"time-server", "60"},
         {"time-offline", "60"},
         {"max-latency", "0.010"},
         {"qps", "10"},
         {"backend", "onnxruntime"},
         {"inputs", "input:0"},
         {"outputs", "MobilenetV1/Predictions/Reshape_1:0"},
         {"post_process", "SoftMax"},
     }},
};

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
    options.add_options()("model", "model to load",
                          cxxopts::value<std::string>()->default_value(""))(
        "scenario",
        "scenario to load (SingleStream,MultiStream,Server,Offline)",
        cxxopts::value<std::string>()->default_value("SingleStream"))(
        "mode", "mode (PerformanceOnly,AccuracyOnly,SubmissionRun)",
        cxxopts::value<std::string>()->default_value("PerformanceOnly"))(
        "datadir", "datadir to load",
        cxxopts::value<std::string>()->default_value(""))(
        "profile", "profile to load",
        cxxopts::value<std::string>()->default_value("resnet50"))(
        "time", "time to run", cxxopts::value<int32_t>()->default_value("0"))(
        "threads", "threads", cxxopts::value<int32_t>()->default_value("2"))(
        "max-batchsize", "max-batchsize",
        cxxopts::value<int32_t>()->default_value("32"))(
        "qps", "qps", cxxopts::value<int32_t>()->default_value("20"))(
        "latency", "latency (ms)",
        cxxopts::value<int32_t>()->default_value("15"))(
        "samples-perf-query", "samples-per-query",
        cxxopts::value<int32_t>()->default_value("2"))(
        "count", "count", cxxopts::value<int32_t>()->default_value("0"))(
        "help", "help");

    try {
        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(0);
        }
        if (result.count("model") == 0) {
            std::cout << "specify model with --model path" << std::endl;
            exit(0);
        }
        if (result.count("datadir") == 0) {
            std::cout << "no datadir given - using fake data" << std::endl;
        }

        std::map<std::string, std::string> profile =
            profiles[result["profile"].as<std::string>()];
        if (profile.empty()) {
            std::cout << "invalid profile" << std::endl;
            exit(1);
        }

        mlperf::TestSettings settings;
        settings.scenario = scenario_map[result["scenario"].as<std::string>()];
        settings.mode = mode_map[result["mode"].as<std::string>()];
        settings.server_target_qps = 2000000;

        int entries_to_read = 0;
        int count = result["count"].as<int32_t>();
        if (settings.mode == mlperf::TestMode::PerformanceOnly) {
            if (count > 500 || count == 0) {
                // in performance mode we use only 500 unique images
                entries_to_read = 500;
            }
        }
        if (settings.scenario == mlperf::TestScenario::SingleStream) {
            settings.min_query_count = std::stoi(profile["queries-single"]);
            settings.min_duration_ms = std::stoi(profile["time-single"]) * 1000;
        }
        if (settings.scenario == mlperf::TestScenario::MultiStream) {
            settings.min_query_count = std::stoi(profile["queries-multi"]);
            settings.min_duration_ms = std::stoi(profile["time-multi"]) * 1000;
            settings.multi_stream_samples_per_query =
                result["samples-perf-query"].as<int32_t>();
        }
        if (settings.scenario == mlperf::TestScenario::Server) {
            settings.min_query_count = std::stoi(profile["queries-server"]);
            settings.min_duration_ms = std::stoi(profile["time-server"]) * 1000;
            settings.server_target_qps = result["qps"].as<int32_t>();
            settings.server_target_latency_ns =
                result["latency"].as<int32_t>() * 1000 * 1000;
        }
        if (settings.scenario == mlperf::TestScenario::Offline) {
            settings.min_query_count = std::stoi(profile["queries-offline"]);
            settings.min_duration_ms =
                std::stoi(profile["time-offline"]) * 1000;
        }
        settings.max_query_count = settings.min_query_count;
        // settings.max_duration_ms = settings.min_duration_ms;
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

        run(result["model"].as<std::string>(),
            result["datadir"].as<std::string>(), profile, settings,
            entries_to_read, result["threads"].as<int32_t>(),
            result["max-batchsize"].as<int32_t>());
    } catch (const cxxopts::OptionException &e) {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }

    return 0;
}
}  // namespace mlperf_bench

// out of namespace

int main(int argc, char *argv[]) { return mlperf_bench::main(argc, argv); }
