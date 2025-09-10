import json
import random
import argparse
import os
from collections import defaultdict

def split_dataset(json_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, output_dir="./"):
    """
    分割数据集为训练集、验证集和测试集
    
    Args:
        json_file: 原始COCO格式标注文件
        train_ratio: 训练集比例 (默认0.7)
        val_ratio: 验证集比例 (默认0.2) 
        test_ratio: 测试集比例 (默认0.1)
        output_dir: 输出目录
    """

    print("开始分割数据集...")
    print(f"训练集: {train_ratio*100:.1f}%")
    print(f"验证集: {val_ratio*100:.1f}%") 
    print(f"测试集: {test_ratio*100:.1f}%")
    
    # 读取原始数据
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    print(f"原始数据集: {len(images)} 张图片, {len(annotations)} 个标注")
    
    # 按场景分组，确保每个场景的图片都能分布到各个集合中
    scene_groups = defaultdict(list)
    for img in images:
        # 从文件路径提取场景信息
        path_parts = img['file_name'].split('/')
        if len(path_parts) >= 3:
            scene = path_parts[2]  # ./dataset/water/20210603/xxx.jpg -> water
            date = path_parts[3] if len(path_parts) >= 4 else "default"
            scene_key = f"{scene}_{date}"
        else:
            scene_key = "default"
        
        scene_groups[scene_key].append(img)
    
    print(f"发现 {len(scene_groups)} 个场景组:")
    for scene, imgs in scene_groups.items():
        print(f"  {scene}: {len(imgs)} 张图片")
    
    # 为每个场景分配图片到不同集合
    train_images = []
    val_images = []
    test_images = []
    
    for scene, imgs in scene_groups.items():
        # 随机打乱该场景的图片
        random.shuffle(imgs)
        
        n_total = len(imgs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # 分配图片
        scene_train = imgs[:n_train]
        scene_val = imgs[n_train:n_train + n_val]
        scene_test = imgs[n_train + n_val:]
        
        train_images.extend(scene_train)
        val_images.extend(scene_val)
        test_images.extend(scene_test)
        
        print(f"  {scene} -> 训练:{len(scene_train)}, 验证:{len(scene_val)}, 测试:{len(scene_test)}")
    
    # 获取对应的标注
    def get_annotations_for_images(image_list):
        image_ids = set(img['id'] for img in image_list)
        return [ann for ann in annotations if ann['image_id'] in image_ids]
    
    train_annotations = get_annotations_for_images(train_images)
    val_annotations = get_annotations_for_images(val_images)
    test_annotations = get_annotations_for_images(test_images)
    
    # 创建新的数据集
    def create_dataset(images, annotations, categories):
        return {
            'images': images,
            'annotations': annotations,
            'categories': categories
        }
    
    train_data = create_dataset(train_images, train_annotations, categories)
    val_data = create_dataset(val_images, val_annotations, categories)
    test_data = create_dataset(test_images, test_annotations, categories)
    
    # 保存分割后的数据集
    train_file = os.path.join(output_dir, "train.json")
    val_file = os.path.join(output_dir, "val.json")
    test_file = os.path.join(output_dir, "test.json")
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # 打印分割结果
    print(f"\n✅ 数据集分割完成!")
    print(f"训练集: {len(train_images)} 图片, {len(train_annotations)} 标注 -> {train_file}")
    print(f"验证集: {len(val_images)} 图片, {len(val_annotations)} 标注 -> {val_file}")
    print(f"测试集: {len(test_images)} 图片, {len(test_annotations)} 标注 -> {test_file}")
    
    # 验证没有重复
    all_image_ids = []
    all_image_ids.extend([img['id'] for img in train_images])
    all_image_ids.extend([img['id'] for img in val_images])
    all_image_ids.extend([img['id'] for img in test_images])
    
    if len(all_image_ids) == len(set(all_image_ids)):
        print("✅ 验证通过: 没有重复图片")
    else:
        print("❌ 警告: 发现重复图片!")
    
    return train_file, val_file, test_file

def analyze_split_quality(train_file, val_file, test_file):
    """分析分割质量"""
    print("\n📊 分割质量分析:")
    
    files = [("训练集", train_file), ("验证集", val_file), ("测试集", test_file)]
    
    for name, file_path in files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 统计场景分布
        scene_count = defaultdict(int)
        for img in data['images']:
            path_parts = img['file_name'].split('/')
            if len(path_parts) >= 3:
                scene = path_parts[2]
                scene_count[scene] += 1
        
        print(f"\n{name}:")
        print(f"  总图片: {len(data['images'])}")
        print(f"  总标注: {len(data['annotations'])}")
        print(f"  场景分布: {dict(scene_count)}")
        
        # 计算平均每张图的标注数
        if len(data['images']) > 0:
            avg_annotations = len(data['annotations']) / len(data['images'])
            print(f"  平均标注/图片: {avg_annotations:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分割COCO数据集')
    parser.add_argument('--input', type=str, default='trainval.json', help='输入的标注文件')
    parser.add_argument('--output-dir', type=str, default='./', help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='验证集比例') 
    parser.add_argument('--test-ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 找不到输入文件: {args.input}")
        exit(1)
    
    # 分割数据集
    train_file, val_file, test_file = split_dataset(
        args.input,
        args.train_ratio,
        args.val_ratio, 
        args.test_ratio,
        args.output_dir
    )
    
    # 分析分割质量
    analyze_split_quality(train_file, val_file, test_file)
    
    print(f"\n💡 使用方法:")
    print(f"训练: python train_with_val.py --train-data {train_file} --val-data {val_file}")
    print(f"测试: python test.py --test-data {test_file} --model ./output/model_final.pth")