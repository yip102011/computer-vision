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

## Docker Usage

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
