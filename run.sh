
#./mlperf_bench --model $model_dir/mobilenet_v1_1.0_224.onnx --profile mobilenet-onnxruntime --datadir $data_dir/imagenet_mobilenet/NCHW --scenario Offline $opt $@
#./mlperf_bench --model $model_dir/resnet50_v1.onnx --profile resnet50-onnxruntime --datadir $data_dir/imagenet/NCHW --scenario Offline $opt $@

export MODEL_DIR=$HOME/resnet_for_mlperf/opset10
export DATA_ROOT=$PWD/../../preprocessed

# CI end to end test for all models

if [ "x$DATA_ROOT" == "x" ]; then
    echo "DATA_ROOT not set" && exit 1
fi
if [ "x$MODEL_DIR" == "x" ]; then
    echo "MODEL_DIR not set" && exit 1
fi


# gloal options for all runs
gopt="--max-batchsize 32"

# quick run
#gopt="$gopt --queries-single 1000 --queries-offline 1000 --queries-multi 1000 --time 30 --count 1000"
gopt="$gopt --time 30 --count 500"
gopt="$gopt $@"

set -x

function one_run {
    ./mlperf_bench $* --scenario SingleStream 
    #./mlperf_bench $* --scenario SingleStream --accuracy
    #./mlperf_bench $* --scenario MultiStream
    #./mlperf_bench $* --scenario Server
    #./mlperf_bench $* --scenario Offline
}

one_run --model $MODEL_DIR/resnet50_v1.onnx --datadir $DATA_ROOT/imagenet/NCHW --profile resnet50 --qps 20 $gopt

one_run --model $MODEL_DIR/mobilenet_v1_1.0_224.onnx --datadir $DATA_ROOT/imagenet_mobilenet/NCHW --profile mobilenet --qps 20 $gopt

#one_run --model $MODEL_DIR/ssd_mobilenet_v1_coco_2018_01_28.onnx --datadir $DATA_ROOT/xx --profile ssd-mobilenet --qps 20 $gopt
#one_run onnxruntime cpu ssd-mobilenet --qps 50 $gopt
#one_run onnxruntime cpu ssd-resnet34 --qps 1 $gopt
