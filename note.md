# YOLOX Inference Scripts

Custom YOLOX demo scripts for single-image inference with JSON output.

## Scripts

### CPU Version: `yolox-cpu-json.py`
Optimized for CPU-only inference.

**Usage:**
```bash
python yolox-cpu-json.py -f exps/default/yolox_s.py -c weights/yolox_s.pth --path assets/dog.jpg -o /workspace/mnt/
```

### GPU Version: `yolox-gpu-json.py`
GPU-accelerated inference with CUDA support.

**Usage:**
```bash
python yolox-gpu-json.py -f exps/default/yolox_s.py -c weights/yolox_s.pth --path assets/dog.jpg -o /workspace/mnt/
```

**Specify device:**
```bash
python yolox-gpu-json.py -f exps/default/yolox_s.py -c weights/yolox_s.pth --path assets/dog.jpg -o /workspace/mnt/ --device cuda
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-f`, `--exp_file` | Experiment description file | `None` |
| `-c`, `--ckpt` | Checkpoint file | `None` |
| `--path` | Path to input image | `./assets/dog.jpg` |
| `--conf` | Confidence threshold | `0.3` |
| `--nms` | NMS threshold | `0.3` |
| `--tsize` | Test image size | `None` |
| `--device` | Device to use (GPU only) | `cuda` |
| `-o`, `--output_dir` | Output directory for JSON results | `./YOLOX_outputs` |

## Output

Results saved as JSON to `<output_dir>/<image_name>.json`:

```json
{
  "image": "dog.jpg",
  "width": 640,
  "height": 480,
  "detections": [
    {
      "bbox": [124.5, 119.0, 560.3, 421.2],
      "score": 0.9545,
      "class_id": 1,
      "class_name": "bicycle"
    }
  ]
}
```

## docker build

```bash
docker build . -f Dockerfile.yolox -t yolox
```

## Docker Usage

start new container
```bash
docker run -it -d --runtime nvidia --gpus all --volume ./mnt/:/workspace/mnt/ --name yolox ghcr.io/yip102011/yolox:latest

docker run -it -d --runtime nvidia --gpus all --volume ./YOLOX_outputs/:/YOLOX_outputs/ --name yolox-trt yolox-trt
```

Copy script to container and execute:

```bash
# Copy script
docker cp yolox-gpu-json.py yolox:/workspace/mnt/yolox-gpu-json.py

# Run inference with custom output directory
docker exec yolox python /workspace/mnt/yolox-gpu-json.py \
  -f exps/default/yolox_s.py \
  -c weights/yolox_s.pth \
  --path assets/dog.jpg \
  -o /workspace/mnt/
```



## Docker Usage trt

start new container
```bash
docker run -it -d --runtime nvidia --gpus all --volume ./mnt/:/workspace/mnt/ --name yolox ghcr.io/yip102011/yolox:latest
```

Copy script to container and execute:

```bash

python3 tools/export_onnx.py --output-name /workspace/mnt/yolox_s.onnx -f exps/default/yolox_s.py -c weights/yolox_s.pth
## trtexec --onnx=/workspace/mnt/yolox_s.onnx --saveEngine=/workspace/mnt/yolox_s.engine --fp16
docker exec -it yolox trtexec --onnx=/workspace/mnt/yolox_s.onnx --saveEngine=/workspace/mnt/yolox_s.engine --fp16 --stronglyTyped

python3 /workspace/YOLOX/demo/ONNXRuntime/onnx_inference.py -m /workspace/mnt/yolox_s.onnx -i /workspace/YOLOX/assets/dog.jpg -o /workspace/mnt/dog.jpg -s 0.3 --input_shape 640,640

docker exec yolox python /workspace/YOLOX/tools/trt.py -f exps/default/yolox_s.py -c weights/yolox_s.pth

# Copy script
docker cp yolox-trt-json.py yolox:/workspace/mnt/yolox-trt-json.py

# Run inference with custom output directory
docker exec yolox python /workspace/mnt/yolox-gpu-json.py \
  -f exps/default/yolox_s.py \
  -c weights/yolox_s.pth \
  --path assets/dog.jpg \
  -o /workspace/mnt/


# Run inference with custom output directory
docker exec yolox python /workspace/mnt/yolox-trt-json.py \
  -f exps/default/yolox_s.py \
  -c weights/yolox_s.pth \
  --path assets/dog.jpg \
  -o /workspace/mnt/ \
  --trt_file /workspace/mnt/yolox_s.engine


docker exec yolox python tools/demo.py image -f exps/default/yolox_s.py -c weights/yolox_s.pth --trt --trt_file /workspace/mnt/yolox_s.engine --save_result
```


# Copy script

docker cp yolox-trt-json.py yolox:/workspace/mnt/yolox-trt-json.py

python tools/demo.py image -f exps/default/yolox_s.py --trt --save_result

python tools/trt.py -f exps/default/yolox_s.py -c weights/yolox_s.pth



./YOLOX_outputs/yolox_s/model_trt.pth


trtexec --loadEngine=model.trt --batch=1 --shapes=images:1x3x640x640

trtexec --onnx=model.onnx \
   --saveEngine=yolox_s.trt    --fp16 \
   --workspace=4096

docker exec -it yolox trtexec --onnx=/workspace/mnt/yolox_s.onnx --saveEngine=/workspace/mnt/yolox_s.engine --fp16 --stronglyTyped





docker exec yolox python /workspace/mnt/yolox-trt-json.py \
 -f exps/default/yolox_s.py \
 -c weights/yolox_s.pth \
 --path assets/horse.jpg \
 -o /workspace/mnt/ \
 --trt_file /workspace/mnt/yolox_s.engine





i can very confuse now, let start over. 
the script yolox-gpu-json.py is working with below command line.
```bash
docker exec yolox python /workspace/mnt/yolox-gpu-json.py \
  -f exps/default/yolox_s.py \
  -c weights/yolox_s.pth \
  --path assets/dog.jpg \
  -o /workspace/mnt/
```

now i want to use trt, i converted 'yolox_s.pth' to 'yolox_s.onnx' with below command
```bash
docker exec yolox python tools/export_onnx.py --output-name /workspace/mnt/yolox_s.onnx -f exps/default/yolox_s.py -c weights/yolox_s.pth
```

then i convert onnx to trt file with below command
```bash
docker exec -it yolox trtexec --onnx=/workspace/mnt/yolox_s.onnx --saveEngine=/workspace/mnt/yolox_s.engine --fp16 --stronglyTyped
```

now the script `yolox-trt-json.py` is not working
```log
2026-03-09 07:29:56.455 | INFO     | __main__:main:189 - Running warm-up inference...
Traceback (most recent call last):
  File "/workspace/mnt/yolox-trt-json.py", line 226, in <module>
    main(exp, args)
  File "/workspace/mnt/yolox-trt-json.py", line 190, in main
    outputs_warmup, _ = predictor.inference(args.path)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/mnt/yolox-trt-json.py", line 121, in inference
    outputs = self.decoder(outputs, dtype=outputs.type())
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/YOLOX/yolox/models/yolo_head.py", line 238, in decode_outputs
    for (hsize, wsize), stride in zip(self.hw, self.strides):
                                      ^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1729, in __getattr__
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
AttributeError: 'YOLOXHead' object has no attribute 'hw'
```

help me fix it




docker run -it -d --runtime nvidia --gpus all --volume ./YOLOX_outputs/:/workspace/YOLOX/YOLOX_outputs/ --name yolox-trt yolox-trt

docker run -it -d --gpus all --volume ./YOLOX_outputs/:/workspace/YOLOX/YOLOX_outputs/ --name yolox-trt yolox-trt

Build & Run

# Build
docker build -f Dockerfile.yolox-trt-tensorrt -t yolox-trt-tensorrt .

# Run
docker run --gpus all -it \
  -v ./YOLOX_outputs/:./ \
  --name yolox \
  yolox-trt-tensorrt

Your Commands (same as before)

# Export ONNX
docker exec yolox python tools/export_onnx.py \
  --output-name /workspace/mnt/yolox_s.onnx \
  -f exps/default/yolox_s.py \
  -c weights/yolox_s.pth

# Convert to TensorRT
docker exec yolox trtexec \
  --onnx=/workspace/mnt/yolox_s.onnx \
  --saveEngine=/workspace/mnt/yolox_s.engine \
  --fp16 \
  --stronglyTyped

# Run inference
docker exec yolox python /workspace/mnt/yolox-trt-json.py \
  -f exps/default/yolox_s.py \
  --path assets/dog.jpg \
  -o /workspace/mnt/ \
  --trt_file /workspace/mnt/yolox_s.engine
