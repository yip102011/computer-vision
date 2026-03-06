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

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo - GPU Single Image JSON Output")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="./assets/dog.jpg", help="path to single image")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("--conf", default=0.3, type=float, help="confidence threshold")
    parser.add_argument("--nms", default=0.3, type=float, help="NMS threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test image size")
    parser.add_argument("--device", default="cuda", type=str, help="device to use (cuda or cpu)")
    parser.add_argument("-o", "--output_dir", default="./YOLOX_outputs", help="output directory")
    return parser


class Predictor(object):
    def __init__(self, model, exp, cls_names=COCO_CLASSES, device="cuda"):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.preproc = ValTransform(legacy=False)
        self.device = device

    def inference(self, img_path):
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        ratio = min(self.test_size[0] / height, self.test_size[1] / width)

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        return outputs, {"file_name": os.path.basename(img_path), "width": width, "height": height, "ratio": ratio}

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
    # Override exp settings with CLI args
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    # GPU model setup
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = exp.get_model()
    model.eval()
    model = model.to(device)

    # Load checkpoint
    ckpt_file = args.ckpt or os.path.join(exp.output_dir, "best_ckpt.pth")
    logger.info(f"Loading checkpoint from {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(ckpt["model"])
    logger.info("Checkpoint loaded.")

    logger.info(f"Model Summary: {get_model_info(model, exp.test_size)}")

    predictor = Predictor(model, exp, COCO_CLASSES, device)

    # Process single image
    logger.info(f"Processing: {args.path}")
    outputs, img_info = predictor.inference(args.path)
    results = predictor.extract_results(outputs[0], img_info)

    # Prepare output
    output_data = {
        "image": img_info["file_name"],
        "width": int(img_info["width"]),
        "height": int(img_info["height"]),
        "detections": results
    }

    # Save JSON to output directory
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
