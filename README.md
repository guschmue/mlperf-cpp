mlperf-cpp
========
C++ implementation for the mlperf inference benchmark app

# Supported models
* mobilenet
* resnet50
* ssd-mobilnet
* ssd-resnet34 (post processor is missing)

# Installation
Install mlperf inference benchmark:
```
git clone https://github.com/mlperf/inference.git
cd inference/v0.5/classification_and_detection
```

Add submodule to the source tree:
```
git submodule add git@github.com:guschmue/mlperf-cpp.git
  or
git submodule add https://github.com/guschmue/mlperf-cpp.git
```

To build the soruce tree for linux:
```
cd mlperf-cpp
mkdir build
cd build
cmake .. -DORT_ROOT=$HOME/onnxruntime -DORT_LIB=$HOME/onnxruntime/build/Linux/RelWithDebInfo
```
For windows:
```
cd mlperf-cpp
mkdir build
cd build
cmake ..  -A x64 -G "Visual Studio 15 2017" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DORT_ROOT=c:/src/onnxruntime  -DORT_LIB=c:/src/onnxruntime/build/Windows/RelWithDebInfo/RelWithDebInfo
```

# Running the benchmark
Without the ```--data``` option we'll use fake data to run the benchmark. Accuracy mode is not possible with fake data.
To use the real dataset we require 2 things:
1. pre-processed data files in numpy format. Those are generated by the [mlperf reference app](https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection) and found under the ```preprocessed``` directory, for example ```preprocessed/imagenet_mobilenet/NCHW```. To create them run the reference benchmark in accuracy mode, for example<br> ```run_local.sh onnxruntime cpu mobilenet --accuracy```.
2. A file with files in the data directoy, for example save them as ```val_map.txt``` into the data directory. For imagenet copy imagenet2012/val_map.txt, for coco use [make-coco-list.py](make-coco-list.py) to generate the list. The list is needed because loadgen uses indices to address the images and those need to be in sync with the labels so we can check accuracy.

You can find an example in [run.sh](run.sh).

mlperf_bench has similar options as the reference benchmark app. It uses mlperf.conf which contains model and benchmark specific paramerts defined by mlperf. mlperf.conf is expected in the current working directory and can be overritten with the ```--config``` option.

```
Usage:
  mlperf_bench [OPTION...]
      --config arg              mlperf.conf to use (default: ./mlperf.conf)
      --model arg               onnx model to load (default: )
      --scenario arg            scenario to load (SingleStream,MultiStream,Server,Offline)
      --mode arg                mode (PerformanceOnly,AccuracyOnly,SubmissionRun)
      --data arg             datalist to load (ie. imagenet2012/val_map.txt) (default: )
      --profile arg             profile to load (default: resnet50)
      --time arg                time to run (default: 0)
      --threads arg             threads (default: 2)
      --max-batchsize arg       max-batchsize (default: 32)
      --qps arg                 qps (default: 20)
      --latency arg             latency (ms) (default: 15)
      --samples-perf-query arg  samples-per-query (default: 2)
      --count arg               count (default: 0)
      --help                    help```
```

Some examples:
```
opt="--profile mobilenet --model resnet_for_mlperf/mobilenet_v1_1.0_224.onnx"

# quick test
mlperf_bench $opt --count 500 --scenario SingleStream

# official performance runs
mlperf_bench $opt --scenario SingleStream
mlperf_bench $opt --scenario MultiStream
mlperf_bench $opt --scenario Server
mlperf_bench $opt --scenario Offline

# official accuracy run
mlperf_bench $opt --datadir preprocessed/imagenet_mobilenet/NCHW/val_map.txt --scenario SingleStream --mode AccuracyOnly
```
```--time``` and ```--count``` limit the runtime of the benchmark and produce none compliant runs but those options are helpful during development.

