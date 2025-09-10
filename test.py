import os
import argparse
import torch
import warnings
import json
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
    """Calculate IoU"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def evaluate_and_visualize(predictor, test_images, data_dir, annotations, confidence, output_dir):
    """Evaluate predictions and save visualizations"""
    total_gt = 0
    total_pred = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    iou_scores = []
    
    # Create visualization output directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Evaluating {len(test_images)} images...")
    metadata = MetadataCatalog.get("test_dataset")
    
    for idx, img_info in enumerate(test_images):
        img_path = os.path.join(data_dir, img_info['file_name'].replace("./", ""))
        
        if not os.path.exists(img_path):
            print(f"Skip missing image: {img_path}")
            continue
            
        # Get ground truth
        img_id = img_info['id']
        gt_annotations = [ann for ann in annotations if ann['image_id'] == img_id]
        
        # Predict
        image = cv2.imread(img_path)
        outputs = predictor(image)
        pred_instances = outputs["instances"].to("cpu")
        
        num_gt = len(gt_annotations)
        num_pred = len(pred_instances)
        
        total_gt += num_gt
        total_pred += num_pred
        
        # Calculate IoU and matching
        matched_pred = set()
        matched_gt = set()
        img_ious = []
        
        if num_pred > 0 and num_gt > 0:
            pred_masks = pred_instances.pred_masks.numpy()
            
            for gt_idx, gt_ann in enumerate(gt_annotations):
                best_iou = 0.0
                best_pred_idx = -1
                
                # Create GT mask
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
                
                # Find best match
                for pred_idx, pred_mask in enumerate(pred_masks):
                    if pred_idx in matched_pred:
                        continue
                    iou = calculate_iou(pred_mask, gt_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = pred_idx
                
                # Match if IoU > 0.5
                if best_iou > 0.5:
                    matched_pred.add(best_pred_idx)
                    matched_gt.add(gt_idx)
                    img_ious.append(best_iou)
        
        # Statistics
        img_tp = len(matched_gt)
        img_fp = num_pred - img_tp
        img_fn = num_gt - img_tp
        
        true_positives += img_tp
        false_positives += img_fp
        false_negatives += img_fn
        
        # Calculate per-image metrics
        img_precision = img_tp / max(num_pred, 1) if num_pred > 0 else 0.0
        img_recall = img_tp / max(num_gt, 1) if num_gt > 0 else 0.0
        img_avg_iou = np.mean(img_ious) if img_ious else 0.0
        
        iou_scores.extend(img_ious)
        
        # Visualize and save
        v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
        vis_result = v.draw_instance_predictions(pred_instances)
        
        # Save visualization
        output_path = os.path.join(vis_dir, f"{idx+1:03d}_{os.path.basename(img_path)}")
        
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_result.get_image())
        
        title = (f'{os.path.basename(img_path)}\n'
                f'GT: {num_gt}, Pred: {num_pred}, '
                f'TP: {img_tp}, FP: {img_fp}, FN: {img_fn}\n'
                f'Precision: {img_precision:.3f}, Recall: {img_recall:.3f}, IoU: {img_avg_iou:.3f}')
        
        plt.title(title, fontsize=10)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()  # Close figure to save memory
        
        # Print progress
        if (idx + 1) % 20 == 0:
            print(f"  Processed: {idx+1}/{len(test_images)}")
    
    # Calculate overall metrics
    overall_precision = true_positives / max(total_pred, 1) if total_pred > 0 else 0.0
    overall_recall = true_positives / max(total_gt, 1) if total_gt > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / max(overall_precision + overall_recall, 1e-6)
    overall_iou = np.mean(iou_scores) if iou_scores else 0.0
    
    # Output results
    print(f"\nOverall Evaluation Results:")
    print(f"  Test images: {len(test_images)}")
    print(f"  Total GT: {total_gt}")
    print(f"  Total predictions: {total_pred}")
    print(f"  True Positives (TP): {true_positives}")
    print(f"  False Positives (FP): {false_positives}")  
    print(f"  False Negatives (FN): {false_negatives}")
    print(f"")
    print(f"Performance Metrics:")
    print(f"  Precision: {overall_precision:.3f} ({true_positives}/{total_pred})")
    print(f"  Recall: {overall_recall:.3f} ({true_positives}/{total_gt})")
    print(f"  F1-Score: {overall_f1:.3f}")
    print(f"  Average IoU: {overall_iou:.3f}")
    print(f"  Detection Rate: {true_positives/max(total_gt,1)*100:.1f}%")
    print(f"  False Positive Rate: {false_positives/max(total_pred,1)*100:.1f}%")
    print(f"")
    print(f"Visualizations saved to: {vis_dir}")
    
    return {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'avg_iou': overall_iou,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives
    }

def test_model(model_path, data_file, data_dir, confidence=0.7, output_dir="./test_results"):
    """Test model and save results"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Register dataset
    if "test_dataset" in DatasetCatalog:
        DatasetCatalog.remove("test_dataset")
        MetadataCatalog.remove("test_dataset")
    
    register_coco_instances("test_dataset", {}, data_file, data_dir)
    metadata = MetadataCatalog.get("test_dataset")
    metadata.thing_classes = ["square"]
    
    # Load model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = DefaultPredictor(cfg)
    
    # Read test data
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print(f"Evaluating all {len(data['images'])} images (confidence: {confidence})")
    
    # Evaluate all images
    overall_metrics = evaluate_and_visualize(
        predictor, data['images'], data_dir, data['annotations'], confidence, output_dir
    )
    
    return overall_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--data', type=str, required=True, help='Data file (val.json or test.json)')
    parser.add_argument('--data-dir', type=str, default='./', help='Data directory')
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='./test_results', help='Output directory')
    
    args = parser.parse_args()
    
    metrics = test_model(args.model, args.data, args.data_dir, args.confidence, args.output)
    
    if metrics:
        print(f"\nFinal Evaluation Summary:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")  
        print(f"  F1-Score: {metrics['f1']:.3f}")
        print(f"  Average IoU: {metrics['avg_iou']:.3f}")
        if 'fps' in metrics:
            print(f"  Average FPS: {metrics['fps']:.1f}")
        if 'avg_inference_time' in metrics:
            print(f"  Average inference time: {metrics['avg_inference_time']*1000:.1f} ms")
    else:
        print("Evaluation failed")
