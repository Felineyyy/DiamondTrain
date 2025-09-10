#!/usr/bin/env python3
import os
import json
import cv2
import time
import torch
import argparse
import numpy as np
from collections import defaultdict

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    intersect_x_min = max(x1_min, x2_min)
    intersect_y_min = max(y1_min, y2_min)
    intersect_x_max = min(x1_max, x2_max)
    intersect_y_max = min(y1_max, y2_max)
    
    if intersect_x_max <= intersect_x_min or intersect_y_max <= intersect_y_min:
        return 0.0
    
    intersect_area = (intersect_x_max - intersect_x_min) * (intersect_y_max - intersect_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - intersect_area
    
    return intersect_area / union_area if union_area > 0 else 0.0

def test_model(model_path, data_file, data_dir, confidence, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Register dataset
    if "test_dataset" in DatasetCatalog:
        DatasetCatalog.remove("test_dataset")
        MetadataCatalog.remove("test_dataset")
    
    register_coco_instances("test_dataset", {}, data_file, data_dir)
    metadata = MetadataCatalog.get("test_dataset")
    metadata.thing_classes = ["square"]
    
    # Setup predictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = DefaultPredictor(cfg)
    
    # Load test data
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Build GT dict
    gt_dict = defaultdict(list)
    for ann in data['annotations']:
        img_id = ann['image_id']
        bbox = ann['bbox']
        gt_box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        gt_dict[img_id].append(gt_box)
    
    # Warmup
    if torch.cuda.is_available():
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(5):
            _ = predictor(dummy_img)
        torch.cuda.synchronize()
    
    # Test
    tp, fp, fn = 0, 0, 0
    iou_scores = []
    inference_times = []
    
    for img_info in data['images']:
        img_path = os.path.join(data_dir, img_info['file_name'].replace("./", ""))
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Measure inference time
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        outputs = predictor(img)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        inference_times.append(end_time - start_time)
        
        # Get predictions
        instances = outputs["instances"]
        pred_boxes = instances.pred_boxes.tensor.cpu().numpy() if len(instances) > 0 else []
        pred_scores = instances.scores.cpu().numpy() if len(instances) > 0 else []
        
        # Filter by confidence
        valid_preds = []
        for j, score in enumerate(pred_scores):
            if score >= confidence:
                valid_preds.append(pred_boxes[j])
        
        # Get GT for this image
        img_id = img_info['id']
        gt_boxes = gt_dict[img_id]
        
        # Match predictions to GT
        matched_gt = set()
        for pred_box in valid_preds:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= 0.5:
                tp += 1
                matched_gt.add(best_gt_idx)
                iou_scores.append(best_iou)
            else:
                fp += 1
        
        # Unmatched GT are FN
        fn += len(gt_boxes) - len(matched_gt)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = np.mean(iou_scores) if iou_scores else 0
    avg_time = np.mean(inference_times) if inference_times else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    # Print results
    print(f"Overall Evaluation Results:")
    print(f"  Test images: {len(data['images'])}")
    print(f"  Total GT: {tp + fn}")
    print(f"  Total predictions: {tp + fp}")
    print(f"  True Positives (TP): {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"Performance Metrics:")
    print(f"  Precision: {precision:.3f} ({tp}/{tp + fp})")
    print(f"  Recall: {recall:.3f} ({tp}/{tp + fn})")
    print(f"  F1-Score: {f1_score:.3f}")
    print(f"  Average IoU: {avg_iou:.3f}")
    print(f"  Detection Rate: {tp/max(tp + fn,1)*100:.1f}%")
    print(f"  False Positive Rate: {fp/max(tp + fp,1)*100:.1f}%")
    print(f"Speed Metrics:")
    print(f"  Average FPS: {fps:.1f}")
    print(f"  Average inference time: {avg_time*1000:.1f} ms")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
        'avg_iou': avg_iou,
        'fps': fps,
        'avg_inference_time': avg_time
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--data', type=str, required=True, help='Data file')
    parser.add_argument('--data-dir', type=str, default='./', help='Data directory')
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='./test_results', help='Output directory')
    
    args = parser.parse_args()
    
    metrics = test_model(args.model, args.data, args.data_dir, args.confidence, args.output)
    
    print(f"\nFinal Evaluation Summary:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")  
    print(f"  F1-Score: {metrics['f1']:.3f}")
    print(f"  Average IoU: {metrics['avg_iou']:.3f}")
    print(f"  Average FPS: {metrics['fps']:.1f}")
    print(f"  Average inference time: {metrics['avg_inference_time']*1000:.1f} ms")
