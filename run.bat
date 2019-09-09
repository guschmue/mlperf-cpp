
set model_dir=w:\resnet_for_mlperf\opset10
set data_dir=w:\inference\v0.5\classification_and_detection\preprocessed
set bench=RelWithDbgInfo\mlperf_bench.exe
%bench% --model %model_dir%\mobilenet_v1_1.0_224.onnx --profile mobilenet --data %data_dir%\imagenet_mobilenet\NCHW\val_map.txt --scenario Offline --count 1000 --qps 25 --latency 100
%bench% --model %model_dir%\resnet50_v1.onnx --profile resnet50 --data %data_dir%\imagenet\NCHW\val_map.txt --scenario Offline --count 1000 --qps 25 --latency 100

