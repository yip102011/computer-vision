
# quick start

build and start container 
```bash
docker build -f Dockerfile.yolox-trt -t yolox-trt .
docker run -it -d --runtime nvidia --gpus all --volume ./YOLOX_outputs/:/workspace/YOLOX/YOLOX_outputs/ --name yolox-trt yolox-trt
```

run demo script
```bash
docker exec yolox-trt python tools/demo.py image -f exps/default/yolox_s.py -c weights/yolox_s.pth --path assets/dog.jpg --save_result --device cpu
docker exec yolox-trt python tools/demo.py image -f exps/default/yolox_s.py -c weights/yolox_s.pth --path assets/dog.jpg --save_result --device gpu
```

# compare cpu and gpu

This script `yolox-json.py` removed complex code, only detect object from image and output json.  
Run custom script to compare inference time between cpu and gpu.  
```bash
docker cp yolox-json.py yolox-trt:/workspace/YOLOX/tools/
docker exec yolox-trt python tools/yolox-json.py -f exps/default/yolox_s.py -c weights/yolox_s.pth --path assets/dog.jpg --device cpu
docker exec yolox-trt python tools/yolox-json.py -f exps/default/yolox_s.py -c weights/yolox_s.pth --path assets/dog.jpg --device gpu
```

## optimize with TensorRT

torch2trt is not compatible with latest TensorRT, so i convert weight file to ONNX file, then convert ONNX file to trt engine file and use custom script to run it.
```bash
# convert pre-train file to onnx file
docker exec yolox-trt python tools/export_onnx.py -f exps/default/yolox_s.py -c weights/yolox_s.pth --output-name YOLOX_outputs/yolox_s.onnx

# test onnx file
docker exec yolox-trt python demo/ONNXRuntime/onnx_inference.py -m YOLOX_outputs/yolox_s.onnx -i assets/dog.jpg -o YOLOX_outputs/onnx/ 

# convert onnx to TensorRT, this step will take about 5 min.
docker exec yolox-trt trtexec --onnx=YOLOX_outputs/yolox_s.onnx --saveEngine=YOLOX_outputs/yolox_s.engine --stronglyTyped

# test TensorRT file
docker cp yolox-trt-json.py yolox-trt:/workspace/YOLOX/tools/
docker exec yolox-trt python tools/yolox-trt-json.py -f exps/default/yolox_s.py --trt_file YOLOX_outputs/yolox_s.engine --path assets/dog.jpg -o YOLOX_outputs/

docker cp yolox-trt-json-optimized.py yolox-trt:/workspace/YOLOX/tools/
docker exec yolox-trt python tools/yolox-trt-json-optimized.py -f exps/default/yolox_s.py --trt_file YOLOX_outputs/yolox_s.engine --path assets/dog.jpg -o YOLOX_outputs/
```



gpu is about 4.5x faster  
- cpu: 0.0771s
- gpu: 0.0167s