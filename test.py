import os
import argparse
import torch
import warnings
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

def calculate_iou(pred_mask, gt_mask):
    """计算IoU"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def evaluate_predictions(predictor, test_images, data_dir, annotations, confidence, output_dir):
    """评估预测结果并保存可视化"""
    total_gt = 0
    total_pred = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    iou_scores = []
    detailed_results = []
    
    # 创建可视化输出目录
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"📊 开始评估 {len(test_images)} 张图片...")
    metadata = MetadataCatalog.get("test_dataset")
    
    for idx, img_info in enumerate(test_images):
        img_path = os.path.join(data_dir, img_info['file_name'].replace("./", ""))
        
        if not os.path.exists(img_path):
            print(f"跳过不存在的图片: {img_path}")
            continue
            
        # 获取ground truth
        img_id = img_info['id']
        gt_annotations = [ann for ann in annotations if ann['image_id'] == img_id]
        
        # 预测
        image = cv2.imread(img_path)
        outputs = predictor(image)
        pred_instances = outputs["instances"].to("cpu")
        
        num_gt = len(gt_annotations)
        num_pred = len(pred_instances)
        
        total_gt += num_gt
        total_pred += num_pred
        
        # 计算IoU和匹配
        matched_pred = set()
        matched_gt = set()
        img_ious = []
        
        if num_pred > 0 and num_gt > 0:
            pred_masks = pred_instances.pred_masks.numpy()
            
            for gt_idx, gt_ann in enumerate(gt_annotations):
                best_iou = 0.0
                best_pred_idx = -1
                
                # 创建GT mask
                if 'segmentation' in gt_ann and gt_ann['segmentation']:
                    try:
                        from pycocotools import mask as maskUtils
                        h, w = img_info['height'], img_info['width']
                        rle = maskUtils.frPyObjects(gt_ann['segmentation'], h, w)
                        gt_mask = maskUtils.decode(rle).squeeze()
                        if gt_mask.ndim > 2:
                            gt_mask = gt_mask.any(axis=2)
                    except:
                        continue
                else:
                    continue
                
                # 找最佳匹配
                for pred_idx, pred_mask in enumerate(pred_masks):
                    if pred_idx in matched_pred:
                        continue
                    iou = calculate_iou(pred_mask, gt_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = pred_idx
                
                # 如果IoU > 0.5认为匹配
                if best_iou > 0.5:
                    matched_pred.add(best_pred_idx)
                    matched_gt.add(gt_idx)
                    img_ious.append(best_iou)
        
        # 统计
        img_tp = len(matched_gt)
        img_fp = num_pred - img_tp
        img_fn = num_gt - img_tp
        
        true_positives += img_tp
        false_positives += img_fp
        false_negatives += img_fn
        
        # 计算单张图片指标
        img_precision = img_tp / max(num_pred, 1) if num_pred > 0 else 0.0
        img_recall = img_tp / max(num_gt, 1) if num_gt > 0 else 0.0
        img_avg_iou = np.mean(img_ious) if img_ious else 0.0
        
        iou_scores.extend(img_ious)
        
        detailed_results.append({
            'image': os.path.basename(img_path),
            'gt_count': num_gt,
            'pred_count': num_pred,
            'tp': img_tp,
            'fp': img_fp,
            'fn': img_fn,
            'precision': img_precision,
            'recall': img_recall,
            'avg_iou': img_avg_iou
        })
        
        # 可视化并保存
        v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
        vis_result = v.draw_instance_predictions(pred_instances)
        
        # 保存可视化图片
        output_path = os.path.join(vis_dir, f"{idx+1:03d}_{os.path.basename(img_path)}")
        
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_result.get_image())
        
        title = (f'{os.path.basename(img_path)}\n'
                f'GT: {num_gt}, Pre: {num_pred}, '
                f'Correct: {img_tp}, Wrong: {img_fp}, Miss: {img_fn}\n'
                f'accurancy: {img_precision:.3f}, recall: {img_recall:.3f}, IoU: {img_avg_iou:.3f}')
        
        plt.title(title, fontsize=10)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()  # 关闭图窗以节省内存
        
        # 打印进度
        if (idx + 1) % 10 == 0:
            print(f"  已处理: {idx+1}/{len(test_images)}")
    
    # 计算总体指标
    overall_precision = true_positives / max(total_pred, 1) if total_pred > 0 else 0.0
    overall_recall = true_positives / max(total_gt, 1) if total_gt > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / max(overall_precision + overall_recall, 1e-6)
    overall_iou = np.mean(iou_scores) if iou_scores else 0.0
    
    # 输出结果
    print(f"\n🎯 总体评估结果:")
    print(f"  测试图片数: {len(test_images)}")
    print(f"  总GT目标数: {total_gt}")
    print(f"  总预测数: {total_pred}")
    print(f"  正确检测(TP): {true_positives}")
    print(f"  误检测(FP): {false_positives}")  
    print(f"  漏检测(FN): {false_negatives}")
    print(f"")
    print(f"📈 性能指标:")
    print(f"  准确率(Precision): {overall_precision:.3f} ({true_positives}/{total_pred})")
    print(f"  召回率(Recall): {overall_recall:.3f} ({true_positives}/{total_gt})")
    print(f"  F1分数: {overall_f1:.3f}")
    print(f"  平均IoU: {overall_iou:.3f}")
    print(f"  检测率: {true_positives/max(total_gt,1)*100:.1f}%")
    print(f"  误检率: {false_positives/max(total_pred,1)*100:.1f}%")
    print(f"")
    print(f"📁 可视化结果保存在: {vis_dir}")
    
    return detailed_results, {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'avg_iou': overall_iou,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives
    }

def test_model(model_path, data_file, data_dir, confidence=0.7, output_dir="./test_results"):
    """测试模型并保存结果"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 注册数据集
    if "test_dataset" in DatasetCatalog:
        DatasetCatalog.remove("test_dataset")
        MetadataCatalog.remove("test_dataset")
    
    register_coco_instances("test_dataset", {}, data_file, data_dir)
    metadata = MetadataCatalog.get("test_dataset")
    metadata.thing_classes = ["square"]
    
    # 加载模型
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = DefaultPredictor(cfg)
    
    # 读取测试数据
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print(f"测试所有 {len(data['images'])} 张图片 (置信度: {confidence})")
    
    # 评估所有图片
    detailed_results, overall_metrics = evaluate_predictions(
        predictor, data['images'], data_dir, data['annotations'], confidence, output_dir
    )
    
    # 保存详细结果到CSV
    import csv
    csv_path = os.path.join(output_dir, "detailed_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'gt_count', 'pred_count', 'tp', 'fp', 'fn', 'precision', 'recall', 'avg_iou'])
        writer.writeheader()
        writer.writerows(detailed_results)
    
    # 保存总体指标
    summary_path = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"模型评估总结\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"数据文件: {data_file}\n")
        f.write(f"置信度阈值: {confidence}\n\n")
        f.write(f"总体指标:\n")
        f.write(f"  准确率: {overall_metrics['precision']:.3f}\n")
        f.write(f"  召回率: {overall_metrics['recall']:.3f}\n")
        f.write(f"  F1分数: {overall_metrics['f1']:.3f}\n")
        f.write(f"  平均IoU: {overall_metrics['avg_iou']:.3f}\n")
        f.write(f"  TP: {overall_metrics['tp']}\n")
        f.write(f"  FP: {overall_metrics['fp']}\n")
        f.write(f"  FN: {overall_metrics['fn']}\n")
    
    print(f"📄 详细结果CSV: {csv_path}")
    print(f"📋 评估摘要: {summary_path}")
    
    return overall_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--data', type=str, required=True, help='数据文件 (val.json或test.json)')
    parser.add_argument('--data-dir', type=str, default='./', help='数据目录')
    parser.add_argument('--confidence', type=float, default=0.7, help='置信度')
    parser.add_argument('--output', type=str, default='./test_results', help='输出目录')
    
    args = parser.parse_args()
    
    metrics = test_model(args.model, args.data, args.data_dir, args.confidence, args.output)
    
    if metrics:
        print(f"\n🏆 Final Evaluation Summary:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")  
        print(f"  F1-Score: {metrics['f1']:.3f}")
        print(f"  Average IoU: {metrics['avg_iou']:.3f}")
    else:
        print("❌ Evaluation failed")