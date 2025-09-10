import os
import argparse
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

class ValidationTrainer(DefaultTrainer):
    """支持验证的训练器"""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def register_datasets(train_json, val_json, data_dir):
    """注册训练集和验证集"""
    
    # 清除已有注册
    for name in ["square_train", "square_val"]:
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
    
    # 注册训练集
    register_coco_instances("square_train", {}, train_json, data_dir)
    
    # 注册验证集
    register_coco_instances("square_val", {}, val_json, data_dir)
    
    print("数据集注册成功:")
    print(f"  训练集: {train_json}")
    print(f"  验证集: {val_json}")
    
    return "square_train", "square_val"

def setup_config_with_validation(train_dataset, val_dataset, output_dir="./output", confidence_threshold=0.7):
    """配置支持验证的训练参数"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    
    # 数据集配置
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (val_dataset,)  # 验证集
    
    # 训练参数
    cfg.SOLVER.IMS_PER_BATCH = 10
    cfg.DATALOADER.NUM_WORKERS = 6
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.MAX_ITER = 1500
    cfg.SOLVER.STEPS = (800, 1200)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
    
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    
    # 设备配置
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        print("使用GPU训练")
    else:
        cfg.MODEL.DEVICE = "cpu"
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
        print("使用CPU训练")
    
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # 🔍 验证配置 - 关键设置
    cfg.TEST.EVAL_PERIOD = 300  # 每300步验证一次
    cfg.SOLVER.CHECKPOINT_PERIOD = 500  # 每500步保存一次
    cfg.SOLVER.AMP.ENABLED = True
    
    return cfg

def train_with_validation(train_json, val_json, data_dir, output_dir="./output", confidence_threshold=0.7):
    """支持验证的训练流程"""
    print("开始训练 (包含验证)")
    
    # 注册数据集
    train_dataset, val_dataset = register_datasets(train_json, val_json, data_dir)
    
    # 配置模型
    cfg = setup_config_with_validation(train_dataset, val_dataset, output_dir, confidence_threshold)
    
    print(f"训练配置:")
    print(f"  批大小: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  最大迭代: {cfg.SOLVER.MAX_ITER}")
    print(f"  验证频率: 每{cfg.TEST.EVAL_PERIOD}步")
    print(f"  置信度阈值: {confidence_threshold}")
    
    # 开始训练
    trainer = ValidationTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    print("训练完成")
    
    # 最终验证
    print("进行最终验证...")
    cfg.MODEL.WEIGHTS = os.path.join(output_dir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    
    evaluator = COCOEvaluator(val_dataset, cfg, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, val_dataset)
    trainer.model.eval()
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    print(f"\n📊 最终验证结果:")
    if 'segm' in results:
        print(f"分割AP: {results['segm']['AP']:.3f}")
        print(f"分割AP50: {results['segm']['AP50']:.3f}")
        print(f"分割AP75: {results['segm']['AP75']:.3f}")
        print(f"分割APs: {results['segm']['APs']:.3f}")
        print(f"分割APm: {results['segm']['APm']:.3f}")
        print(f"分割APl: {results['segm']['APl']:.3f}")
    
    if 'bbox' in results:
        print(f"检测AP: {results['bbox']['AP']:.3f}")
        print(f"检测AP50: {results['bbox']['AP50']:.3f}")
        print(f"检测AP75: {results['bbox']['AP75']:.3f}")
    
    print(f"\n模型保存在: {os.path.join(output_dir, 'model_final.pth')}")
    
    return cfg, results

def quick_validation_test(model_path, val_json, data_dir, confidence_threshold=0.7):
    """快速验证测试"""
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer
    import cv2
    import matplotlib.pyplot as plt
    import json
    import random
    
    print(f"快速验证测试 (置信度: {confidence_threshold})")
    
    # 注册验证集
    if "square_val_test" in DatasetCatalog:
        DatasetCatalog.remove("square_val_test")
        MetadataCatalog.remove("square_val_test")
    
    register_coco_instances("square_val_test", {}, val_json, data_dir)
    
    # 配置预测器
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get("square_val_test")
    if not hasattr(metadata, 'thing_classes'):
        metadata.thing_classes = ["square"]
    
    # 测试验证集中的图片
    with open(val_json, 'r') as f:
        val_data = json.load(f)
    
    test_images = random.sample(val_data['images'], min(5, len(val_data['images'])))
    
    print(f"测试 {len(test_images)} 张验证集图片:")
    
    for i, img_info in enumerate(test_images):
        img_path = os.path.join(data_dir, img_info['file_name'].replace("./", ""))
        
        if os.path.exists(img_path):
            im = cv2.imread(img_path)
            outputs = predictor(im)
            
            instances = outputs["instances"]
            scores = instances.scores.cpu().numpy() if len(instances) > 0 else []
            
            print(f"  {i+1}. {os.path.basename(img_path)}")
            print(f"     检测数量: {len(instances)}")
            if len(scores) > 0:
                print(f"     置信度: {[f'{s:.3f}' for s in scores]}")
            
            # 可视化第一张图片
            if i == 0:
                v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
                out = v.draw_instance_predictions(instances.to("cpu"))
                
                plt.figure(figsize=(12, 8))
                plt.imshow(out.get_image())
                plt.title(f"验证集测试样例 (检测到 {len(instances)} 个目标)")
                plt.axis('off')
                plt.savefig(os.path.join(os.path.dirname(model_path), "validation_test.png"))
                plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='支持验证集的训练')
    parser.add_argument('--train-data', type=str, required=True, help='训练集JSON文件')
    parser.add_argument('--val-data', type=str, required=True, help='验证集JSON文件')
    parser.add_argument('--data-dir', type=str, default='./', help='数据根目录')
    parser.add_argument('--output-dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--confidence', type=float, default=0.7, help='置信度阈值')
    parser.add_argument('--test-only', action='store_true', help='仅测试验证集')
    parser.add_argument('--model-path', type=str, help='测试模型路径')
    
    args = parser.parse_args()
    
    if args.test_only and args.model_path:
        quick_validation_test(args.model_path, args.val_data, args.data_dir, args.confidence)
    else:
        cfg, results = train_with_validation(
            args.train_data, 
            args.val_data, 
            args.data_dir, 
            args.output_dir, 
            args.confidence
        )