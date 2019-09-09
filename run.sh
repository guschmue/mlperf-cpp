#!/bin/bash

# you can pass in additonal options, for example:
# --time 30 --count 500
# --model AccuracyOnly
#


export MODEL_DIR=$HOME/resnet_for_mlperf/opset10
export DATA_ROOT=$PWD/../../preprocessed

# CI end to end test for all models

if [ "x$MODEL_DIR" == "x" ]; then
    echo "MODEL_DIR not set" && exit 1
fi


# gloal options for all runs
gopt="--max-batchsize 32 --threads 2"

#gopt="$gopt --time 30 --count 500"
gopt="$gopt $@"

if [ "x$DATA_ROOT" == "x" ]; then
    echo "DATA_ROOT not set, using fake data"
    gopt="$gopt --fake"    
fi

function one_run {
    ./mlperf_bench --scenario SingleStream $* 
    ./mlperf_bench --scenario MultiStream $* 
    ./mlperf_bench --scenario Server $* 
    ./mlperf_bench --scenario Offline $* 
}

one_run --model $MODEL_DIR/resnet50_v1.onnx --data $DATA_ROOT/imagenet/NCHW/val_map.txt --profile resnet50 --qps 20 $gopt
one_run --model $MODEL_DIR/mobilenet_v1_1.0_224.onnx --data $DATA_ROOT/imagenet_mobilenet/NCHW/val_map.txt --profile mobilenet --qps 20 $gopt
one_run --model $MODEL_DIR/ssd_mobilenet_v1_coco_2018_01_28.onnx --data $DATA_ROOT/coco-300/NHWC/val_map.txt --profile ssd-mobilenet --qps 20 $gopt
#one_run --model $MODEL_DIR/resnet34-ssd1200.onnx --data $DATA_ROOT/coco-1200/NCHW/val_map.txt --profile ssd-resnet34 --qps 20 $gopt
