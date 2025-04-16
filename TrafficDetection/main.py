# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 18:35:07 2025

"""

import cv2
import torch
import argparse
from ultralytics import YOLO
import numpy as np
import os

# Static model paths (predefined)
MODEL_PATHS = [
    "./Models/best (10).pt",
    "./Models/best (9).pt"
]

# Parse CLI arguments (only input and output)
parser = argparse.ArgumentParser(description="YOLO Ensemble Detection")
parser.add_argument("--input", required=True, help="Path to input video")
parser.add_argument("--output", required=True, help="Path to save output video")
args = parser.parse_args()

models = [YOLO(path).to('cpu').eval() for path in MODEL_PATHS]

def ensemble_nms(all_boxes, iou_threshold=0.5):
    if len(all_boxes) == 0:
        return []
    boxes_array = torch.tensor(all_boxes)
    boxes_xyxy = boxes_array[:, :4]
    scores = boxes_array[:, 4]
    keep = torch.ops.torchvision.nms(boxes_xyxy, scores, iou_threshold)
    return boxes_array[keep].numpy()

cap = cv2.VideoCapture(args.input)
if not cap.isOpened():
    raise IOError(f"Cannot open video file {args.input}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    all_boxes = []

    for model in models:
        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                if confidence > 0.4:
                    all_boxes.append([x1, y1, x2, y2, confidence, class_id])

    filtered_boxes = ensemble_nms(all_boxes)

    for box in filtered_boxes:
        x1, y1, x2, y2, confidence, class_id = box
        label = f"{models[0].names[int(class_id)]}: {confidence:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    out.write(frame)

cap.release()
out.release()
print(f"✅ Output saved to: {args.output}")
