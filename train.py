#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

import os
import json
import torch
import argparse
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

class Trainer(DefaultTrainer):
    """Simple trainer with evaluation"""
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

def setup_config(output_dir="./output", confidence_threshold=0.9):
    """Setup training config with smaller input size"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    
    # 数据集配置
    cfg.DATASETS.TRAIN = ("square_train",)
    cfg.DATASETS.TEST = ("square_val",)
    cfg.DATALOADER.NUM_WORKERS = 6
    
    # 预训练模型
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    
    # 训练和输入尺寸配置
    cfg.INPUT.MIN_SIZE_TRAIN = (512, 544, 576, 608, 640)
    cfg.INPUT.MAX_SIZE_TRAIN = 900 
    cfg.INPUT.MIN_SIZE_TEST = 576
    cfg.INPUT.MAX_SIZE_TEST = 900
    
    # 模型配置
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    
    # 训练参数
    cfg.SOLVER.IMS_PER_BATCH = 10 if torch.cuda.is_available() else 2
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.MAX_ITER = 1500      
    cfg.SOLVER.STEPS = (800, 1200)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.WEIGHT_DECAY = 0.0001

    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 400   
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 300     
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 200  
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 150   
    
    # ROI配置  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    
    # 梯度裁剪，防止训练不稳定
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    
    # 验证和保存配置
    cfg.TEST.EVAL_PERIOD = 500      
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.TEST.DETECTIONS_PER_IMAGE = 10  
    
    # 设备配置
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 128
    
    # 输出目录
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # 启用混合精度加速
    cfg.SOLVER.AMP.ENABLED = True
    
    return cfg

def register_datasets(train_json, val_json, data_dir="./"):
    """Register training and validation datasets"""
    # 清除已存在的数据集
    for name in ["square_train", "square_val"]:
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
    
    # 注册数据集
    register_coco_instances("square_train", {}, train_json, data_dir)
    register_coco_instances("square_val", {}, val_json, data_dir)
    
    # 设置元数据
    for name in ["square_train", "square_val"]:
        MetadataCatalog.get(name).thing_classes = ["square"]
    
    print(f"Registered datasets: square_train, square_val")
    return "square_train", "square_val"

def print_config_summary(cfg):
    """Print training configuration summary"""
    print("\n=== Training Configuration ===")
    print(f"Input size: {cfg.INPUT.MIN_SIZE_TEST} ")
    print(f"Max size: {cfg.INPUT.MAX_SIZE_TEST} ")
    print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"ROI batch size: {cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}")
    print(f"Device: {cfg.MODEL.DEVICE}")
    print(f"Mixed precision: {cfg.SOLVER.AMP.ENABLED}")
    print(f"Output dir: {cfg.OUTPUT_DIR}")
    print("=" * 40)

def train(train_json, val_json, data_dir="./", output_dir="./output", confidence_threshold=0.7):
    print("Starting training...")
    
    # 注册数据集
    train_dataset, val_dataset = register_datasets(train_json, val_json, data_dir)
    
    # 配置模型
    cfg = setup_config(output_dir, confidence_threshold)
    
    # 打印配置摘要
    print_config_summary(cfg)
    
    # 开始训练
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    print("\nTraining completed!")
    print(f"Model saved to: {os.path.join(output_dir, 'model_final.pth')}")
    
    # 最终验证
    print("\nRunning final validation...")
    cfg.MODEL.WEIGHTS = os.path.join(output_dir, "model_final.pth")
    
    evaluator = COCOEvaluator(val_dataset, cfg, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, val_dataset)
    trainer.model.eval()
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    # 打印结果
    print(f"\n=== Final Validation Results ===")
    if 'segm' in results:
        print(f"Segmentation AP: {results['segm']['AP']:.3f}")
        print(f"Segmentation AP50: {results['segm']['AP50']:.3f}")
        print(f"Segmentation AP75: {results['segm']['AP75']:.3f}")
    if 'bbox' in results:
        print(f"Detection AP: {results['bbox']['AP']:.3f}")
        print(f"Detection AP50: {results['bbox']['AP50']:.3f}")
        print(f"Detection AP75: {results['bbox']['AP75']:.3f}")
    
    return cfg, results

def main():
    parser = argparse.ArgumentParser(description='Training with smaller input size')
    parser.add_argument('--train-data', type=str, required=True, help='Training JSON file')
    parser.add_argument('--val-data', type=str, required=True, help='Validation JSON file') 
    parser.add_argument('--data-dir', type=str, default='./', help='Data root directory')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.train_data):
        print(f"Error: Training data file not found: {args.train_data}")
        return
        
    if not os.path.exists(args.val_data):
        print(f"Error: Validation data file not found: {args.val_data}")
        return
    
    # 开始训练
    cfg, results = train(
        args.train_data,
        args.val_data,
        args.data_dir,
        args.output_dir,
        args.confidence
    )
    
    print(f"\nTraining completed successfully!")
    print(f"Model path: {os.path.join(args.output_dir, 'model_final.pth')}")


if __name__ == "__main__":
    main()
