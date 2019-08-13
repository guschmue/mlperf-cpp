# mlperf-cpp
c++ implementation for the mlperf inference benchmark app

# installation
Install mlperf inference benchmark:
```
git clone https://github.com/mlperf/inference.git
cd inference/v0.5/classification_and_detection
```

Add submodule to the source tree:
```
git submodule add git@github.com:guschmue/mlperf-cpp.git
```
or
```
git submodule add https://github.com/guschmue/mlperf-cpp.git
```

Build the tree:
```
mkdir build
cd build
cmake .. -DORT_ROOT=$HOME/onnxruntime -DORT_LIB=$HOME/onnxruntime/build/Linux/RelWithDebInfo
```
or for windows you might want to use:
```
cmake ..  -A x64 -G "Visual Studio 15 2017"  -DORT_ROOT=c:/src/onnxruntime  -DORT_LIB=c:/src/onnxruntime/build/Windows/RelWithDebInfo/RelWithDebInfo
```

# running the benchmark
```
Usage:
  mlperf_bench [OPTION...]

      --model arg               model to load (default: )
      --scenario arg            scenario to load (SingleStream,MultiStream,Server,Offline)
      --mode arg                mode (PerformanceOnly,AccuracyOnly,SubmissionRun)
      --datadir arg             datalist to load (ie. imagenet2012/val_map.txt) (default: )
      --profile arg             profile to load (default: resnet50)
      --time arg                time to run (default: 0)
      --threads arg             threads (default: 2)
      --max-batchsize arg       max-batchsize (default: 32)
      --qps arg                 qps (default: 20)
      --latency arg             latency (ms) (default: 15)
      --samples-perf-query arg  samples-per-query (default: 2)
      --count arg               count (default: 0)
      --help                    help```

Some common examples:
```
mlperf_bench --profile mobilenet --model resnet_for_mlperf/opset10/mobilenet_v1_1.0_224.onnx --datadir preprocessed/imagenet_mobilenet/NCHW/val_map.txt --count 500 --scenario SingleStream

* datadir contains the preprocessed images by inference/v0.5/classification_and_detection/python
```

If no datadir is given, the benchmark will create a fake dataset:
```
mlperf_bench --profile mobilenet --model resnet_for_mlperf/opset10/mobilenet_v1_1.0_224.onnx --count 500 --scenario SingleStream
```

```--time``` and ```--count``` limit the runtime of the benchmark. Absense of those options will use the mlperf compliant settings.
