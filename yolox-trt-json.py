#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
import json
from loguru import logger

import cv2
import torch
import numpy as np

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo - TensorRT JSON Output (Optimized)")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="./assets/dog.jpg", help="path to single image")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("--conf", default=0.3, type=float, help="confidence threshold")
    parser.add_argument("--nms", default=0.3, type=float, help="NMS threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test image size")
    parser.add_argument("-o", "--output_dir", default="./YOLOX_outputs", help="output directory")
    parser.add_argument("--trt_file", type=str, default=None, help="TensorRT engine file path")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs")
    return parser


class TRTPredictor(object):
    def __init__(self, exp, cls_names=COCO_CLASSES, trt_file=None):
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.preproc = ValTransform(legacy=False)
        
        # Load TensorRT engine
        import tensorrt as trt
        logger.info(f"Loading TensorRT engine from {trt_file}")
        
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        with open(trt_file, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Get input/output names
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        
        # Pre-allocate GPU buffers ONCE (optimization!)
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_names[0]))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_names[0]))
        
        self.input_buffer = torch.empty(self.input_shape, dtype=torch.float32, device='cuda')
        self.output_buffer = torch.empty(self.output_shape, dtype=torch.float32, device='cuda')
        
        # Create CUDA stream for async execution
        self.stream = torch.cuda.Stream()
        
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"Output shape: {self.output_shape}")
        logger.info("TensorRT engine loaded and buffers pre-allocated")

    def inference(self, img):
        """Run inference on preprocessed image tensor"""
        # Set input shape for dynamic shapes
        self.context.set_input_shape(self.input_names[0], img.shape)
        
        # Set tensor addresses
        self.context.set_tensor_address(self.input_names[0], self.input_buffer.data_ptr())
        self.context.set_tensor_address(self.output_names[0], self.output_buffer.data_ptr())
        
        # Copy input data to pre-allocated buffer
        self.input_buffer.copy_(img)
        
        # Execute inference asynchronously
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        
        return self.output_buffer

    def preprocess(self, img_path):
        """Preprocess image"""
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        ratio = min(self.test_size[0] / height, self.test_size[1] / width)
        
        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float()
        
        return img, {"file_name": os.path.basename(img_path), "width": width, "height": height, "ratio": ratio}

    def extract_results(self, output, img_info):
        ratio = img_info["ratio"]
        results = []
        if output is None:
            return results
        output = output.cpu()
        bboxes = output[:, 0:4] / ratio
        scores = output[:, 4] * output[:, 5]
        cls = output[:, 6]
        for i in range(len(output)):
            if scores[i] < self.confthre:
                continue
            class_id = int(cls[i])
            results.append({
                "bbox": [float(x) for x in bboxes[i].tolist()],
                "score": float(scores[i]),
                "class_id": class_id,
                "class_name": self.cls_names[class_id] if class_id < len(self.cls_names) else "unknown"
            })
        return results


def main(exp, args):
    # Override exp settings
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    # Load model for decoder initialization
    model = exp.get_model()
    model.eval()
    model.cuda()
    
    # Initialize decoder by running dummy forward
    dummy = torch.zeros(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
    model(dummy)
    decoder = model.head.decode_outputs
    
    trt_file = args.trt_file
    assert os.path.exists(trt_file), f"TensorRT file not found: {trt_file}"
    
    logger.info("TensorRT mode enabled")
    logger.info(f"Model Summary: {get_model_info(model, exp.test_size)}")

    # Create predictor
    predictor = TRTPredictor(exp, COCO_CLASSES, trt_file)

    # Preprocess image once
    logger.info(f"Processing: {args.path}")
    img, img_info = predictor.preprocess(args.path)
    img = img.cuda()

    # Warmup
    logger.info(f"Running {args.warmup} warmup inferences...")
    for _ in range(args.warmup):
        with torch.no_grad():
            outputs = predictor.inference(img)
            outputs = decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, predictor.num_classes, predictor.confthre, predictor.nmsthre, class_agnostic=True)
    torch.cuda.synchronize()

    # Timed runs
    logger.info(f"Running {args.runs} timed inferences...")
    times = []
    for i in range(args.runs):
        start = time.time()
        with torch.no_grad():
            outputs = predictor.inference(img)
            outputs = decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, predictor.num_classes, predictor.confthre, predictor.nmsthre, class_agnostic=True)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    logger.info(f"TensorRT Inference Time:")
    logger.info(f"  Average: {avg_time*1000:.2f}ms")
    logger.info(f"  Min: {min_time*1000:.2f}ms")
    logger.info(f"  Max: {max_time*1000:.2f}ms")

    # Extract and save results
    results = predictor.extract_results(outputs[0], img_info)
    
    output_data = {
        "image": img_info["file_name"],
        "width": int(img_info["width"]),
        "height": int(img_info["height"]),
        "inference_time_sec": round(avg_time, 4),
        "detections": results
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_name = os.path.splitext(os.path.basename(args.path))[0] + ".json"
    output_path = os.path.join(args.output_dir, json_name)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"✓ Saved results to {output_path}")
    logger.info(f"✓ Detected {len(results)} objects:")
    for det in results:
        logger.info(f"    {det['class_name']:12s} score={det['score']:.4f} bbox={det['bbox']}")


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
