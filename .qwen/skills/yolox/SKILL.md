# YOLOX Expert Skill

## Role
You are a YOLOX (You Only Look Once X) computer vision expert. Help users with object detection tasks using the YOLOX framework.

## Container Setup

### Start YOLOX Container
```bash
docker run -it -d --runtime nvidia --gpus all --volume ./mnt/:/workspace/mnt/ --name yolox ghcr.io/yip102011/yolox:latest
```

### Container Details
- **Name**: `yolox`
- **Image**: `ghcr.io/yip102011/yolox:latest`
- **GPU**: NVIDIA runtime with all GPUs
- **Mount**: `./mnt/` → `/workspace/mnt/`

## Available Scripts

### CPU Inference
```bash
python yolox-cpu-json.py -f exps/default/yolox_s.py -c weights/yolox_s.pth --path <image> -o <output_dir>
```

### GPU Inference
```bash
python yolox-gpu-json.py -f exps/default/yolox_s.py -c weights/yolox_s.pth --path <image> -o <output_dir> --device cuda
```

## Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-f` | Experiment file | `exps/default/yolox_s.py` |
| `-c` | Checkpoint path | `weights/yolox_s.pth` |
| `--path` | Input image path | `assets/dog.jpg` |
| `-o` | Output directory | `./YOLOX_outputs` |
| `--conf` | Confidence threshold | `0.3` |
| `--nms` | NMS threshold | `0.3` |
| `--tsize` | Input image size | `None` (use exp default) |
| `--device` | Device (gpu script) | `cuda` |

## Docker Commands

```bash
# Copy script to container
docker cp <local_script.py> yolox:/workspace/mnt/

# Execute in container
docker exec yolox python /workspace/mnt/<script>.py [args]

# View container logs
docker logs yolox

# Stop container
docker stop yolox

# Remove container
docker rm yolox
```

## JSON Output Format
```json
{
  "image": "dog.jpg",
  "width": 768,
  "height": 576,
  "inference_time_sec": 0.1387,
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "score": 0.9545,
      "class_id": 1,
      "class_name": "bicycle"
    }
  ]
}
```

## COCO Classes (80 categories)
Person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## Model Variants
- **YOLOX-Nano**: Fastest, lowest accuracy
- **YOLOX-Tiny**: Fast, moderate accuracy
- **YOLOX-S**: Balanced (default)
- **YOLOX-M**: Good accuracy/speed tradeoff
- **YOLOX-L**: High accuracy
- **YOLOX-X**: Highest accuracy, slowest

## Best Practices
1. Use GPU for batch processing or video inference
2. Adjust `--conf` threshold based on precision/recall needs
3. Use `--tsize` for consistent input sizing
4. Monitor `inference_time_sec` in JSON output for performance
5. Mount volumes for persistent output storage

## Important Rules
- **NEVER modify any code in the YOLOX git submodule** (`YOLOX/` directory)
- Create custom scripts in the project root, not inside the submodule
- The submodule is a third-party dependency and should remain unchanged
