
set model_dir=w:\resnet_for_mlperf
set data_dir=w:\inference\v0.5\classification_and_detection\preprocessed
set bench=Debug\mlperf_bench.exe
%bench% --model %model_dir%\mobilenet_v1_1.0_224.onnx --profile mobilenet-onnxruntime --datadir %data_dir%\imagenet_mobilenet\NCHW --scenario Offline --count 1000 --qps 25 --latency 100
%bench% --model %model_dir%\resnet50_v1.onnx --profile resnet50-onnxruntime --datadir %data_dir%\imagenet\NCHW --scenario Offline --count 1000 --qps 25 --latency 100

