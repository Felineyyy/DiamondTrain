#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Detectron2 MaskRCNN model to ONNX format compatible with TensorRT
"""

import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo
import onnx
import os
from detectron2.structures import Instances
import torch.nn as nn
from typing import List, Dict

def setup_cfg(model_path):
    """
    Setup config from trained model
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    
    # Model parameters from training config
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.98
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    
    # Input size - critical for proper conversion
    cfg.INPUT.MIN_SIZE_TEST = 576
    cfg.INPUT.MAX_SIZE_TEST = 900
    
    # Set to eval mode
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Important for ONNX export
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 300
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 150
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    
    return cfg

class Detectron2ONNXWrapper(nn.Module):
    def __init__(self, model, image_size=(576, 900)):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.model.eval()
        
        # 禁用可能产生Xor操作的预处理
        if hasattr(self.model, 'preprocess_image'):
            # 保存原始方法
            self.original_preprocess = self.model.preprocess_image
            # 替换为简化版本
            self.model.preprocess_image = self.simple_preprocess
    
    def simple_preprocess(self, batched_inputs):
        """
        简化的预处理方法，避免使用Xor操作
        """
        images = []
        for input in batched_inputs:
            if "image" in input:
                img = input["image"]
            elif "image_tensor" in input:
                img = input["image_tensor"]
            else:
                img = input
            
            # 简单的标准化处理
            img = img.float()
            if img.max() > 1.0:
                img = img / 255.0
            
            # 简单的标准化 (根据你的模型配置调整)
            pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)
            pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device)
            img = (img - pixel_mean) / pixel_std
            
            images.append(img)
        
        return images
    
    def forward(self, batched_images):
        # 使用简化预处理
        inputs = []
        for i in range(batched_images.shape[0]):
            img = batched_images[i]
            inputs.append({
                "image": img,
                "height": self.image_size[0],
                "width": self.image_size[1]
            })
        
        # 运行模型
        with torch.no_grad():
            predictions = self.model(inputs)
            
        # Extract outputs - handle both dict and Instances formats
        batch_boxes = []
        batch_scores = []
        batch_labels = []
        batch_masks = []
        
        for pred in predictions:
            # Handle different output formats
            if isinstance(pred, dict) and "instances" in pred:
                instances = pred["instances"]
            elif hasattr(pred, "pred_boxes"):
                instances = pred
            else:
                # Create empty instances if no detections
                instances = Instances(
                    image_size=self.image_size,
                    pred_boxes=torch.zeros((0, 4), device=batched_images.device),
                    scores=torch.zeros(0, device=batched_images.device),
                    pred_classes=torch.zeros(0, device=batched_images.device, dtype=torch.long)
                )
                if hasattr(self.model, 'roi_heads') and hasattr(self.model.roi_heads, 'mask_head'):
                    instances.pred_masks = torch.zeros((0, 28, 28), device=batched_images.device)
            
            # Extract boxes, scores, labels, masks
            boxes = instances.pred_boxes.tensor if hasattr(instances, 'pred_boxes') else torch.zeros((0, 4), device=batched_images.device)
            scores = instances.scores if hasattr(instances, 'scores') else torch.zeros(0, device=batched_images.device)
            labels = instances.pred_classes if hasattr(instances, 'pred_classes') else torch.zeros(0, device=batched_images.device, dtype=torch.long)
            
            # Handle masks
            if hasattr(instances, 'pred_masks'):
                masks = instances.pred_masks
            else:
                masks = torch.zeros((len(boxes), 28, 28), device=batched_images.device)
            
            batch_boxes.append(boxes)
            batch_scores.append(scores)
            batch_labels.append(labels)
            batch_masks.append(masks)
        
        # Pad outputs to have consistent batch size
        max_detections = max(len(boxes) for boxes in batch_boxes) if batch_boxes else 0
        
        padded_boxes = []
        padded_scores = []
        padded_labels = []
        padded_masks = []
        
        for boxes, scores, labels, masks in zip(batch_boxes, batch_scores, batch_labels, batch_masks):
            n = len(boxes)
            if n < max_detections:
                # Pad with zeros
                pad_size = max_detections - n
                boxes = torch.cat([boxes, torch.zeros(pad_size, 4, device=boxes.device)])
                scores = torch.cat([scores, torch.zeros(pad_size, device=scores.device)])
                labels = torch.cat([labels, torch.zeros(pad_size, device=labels.device, dtype=torch.long)])
                masks = torch.cat([masks, torch.zeros(pad_size, 28, 28, device=masks.device)])
            
            padded_boxes.append(boxes)
            padded_scores.append(scores)
            padded_labels.append(labels)
            padded_masks.append(masks)
        
        # Stack into batch tensors
        if padded_boxes:
            return (
                torch.stack(padded_boxes),
                torch.stack(padded_scores),
                torch.stack(padded_labels),
                torch.stack(padded_masks)
            )
        else:
            # Return empty tensors if no detections
            return (
                torch.zeros((1, 0, 4), device=batched_images.device),
                torch.zeros((1, 0), device=batched_images.device),
                torch.zeros((1, 0), device=batched_images.device, dtype=torch.long),
                torch.zeros((1, 0, 28, 28), device=batched_images.device)
            )

def export_detectron2_to_onnx(model_path, output_path):
    """
    Export Detectron2 model to ONNX format using the recommended approach
    """
    print(f"Loading model from: {model_path}")
    
    # Setup config
    cfg = setup_cfg(model_path)
    
    # Build model
    model = build_model(cfg)
    model.eval()
    
    # Load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(model_path)
    
    print("Model loaded successfully")
    
    # Use fixed input size for stability
    height, width = 576, 900
    
    # Wrap the model
    wrapped_model = Detectron2ONNXWrapper(model, (height, width))
    wrapped_model.eval()
    
    # Create tensor input for ONNX export
    dummy_input = torch.randn(1, 3, height, width, device=cfg.MODEL.DEVICE)
    
    # Run once to initialize and check for errors
    try:
        with torch.no_grad():
            outputs = wrapped_model(dummy_input)
            print("Model forward pass successful")
            print(f"Output shapes: boxes={outputs[0].shape}, scores={outputs[1].shape}, labels={outputs[2].shape}, masks={outputs[3].shape}")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        # Try to debug by checking model output format
        print("Debugging model output format...")
        inputs = [{"image": dummy_input[0], "height": height, "width": width}]
        with torch.no_grad():
            raw_output = model(inputs)
            print(f"Raw output type: {type(raw_output)}")
            if isinstance(raw_output, list):
                print(f"Raw output length: {len(raw_output)}")
                for i, item in enumerate(raw_output):
                    print(f"Item {i} type: {type(item)}")
                    if hasattr(item, '__dict__'):
                        print(f"Item {i} attributes: {item.__dict__.keys()}")
            return
    
    # Export to ONNX
    print("Exporting to ONNX format...")
    
    # Define input and output names
    input_names = ["input"]
    output_names = ["boxes", "scores", "labels", "masks"]
    
    # Export with dynamic axes for variable number of detections
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'boxes': {0: 'batch_size', 1: 'num_detections'},
        'scores': {0: 'batch_size', 1: 'num_detections'},
        'labels': {0: 'batch_size', 1: 'num_detections'},
        'masks': {0: 'batch_size', 1: 'num_detections'}
    }
    
    try:
        # Export the model with higher opset version (16 or 17)
        # Note: grid_sampler requires opset >= 16
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            output_path,
            verbose=True,
            opset_version=16,  # 升级到版本16以支持grid_sampler
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        
        # Verify ONNX model
        print(f"Verifying ONNX model: {output_path}")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification successful")
        
        # Print model info
        print(f"\nModel exported successfully to: {output_path}")
        print(f"Input shape: [batch_size, 3, {height}, {width}]")
        print("Output format: boxes, scores, labels, masks")
        print("\nTo convert to TensorRT engine, use:")
        print(f"trtexec --onnx={output_path} --saveEngine=model.engine --fp16")
        
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        # 尝试使用opset 17
        try:
            print("Trying with opset 17...")
            torch.onnx.export(
                wrapped_model,
                dummy_input,
                output_path,
                verbose=True,
                opset_version=17,  # 尝试更高的opset版本
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )
            print("ONNX export with opset 17 successful!")
        except Exception as e2:
            print(f"Error with opset 17: {e2}")
            return None
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Detectron2 to ONNX')
    parser.add_argument('--model', type=str, default='./output/model_final.pth',
                       help='Path to trained Detectron2 model')
    parser.add_argument('--output', type=str, default='./model.onnx',
                       help='Output ONNX file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        exit(1)
    
    export_detectron2_to_onnx(args.model, args.output)
