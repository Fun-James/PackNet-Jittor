"""
测试脚本：验证数据加载器和基本功能
在没有Jittor的环境中测试数据结构和逻辑
"""

import os
import sys
from PIL import Image


def test_data_structure():
    """
    测试数据集结构是否正确
    """
    print("=== 测试数据集结构 ===")
    
    data_root = "data"
    datasets = ['cubs_cropped', 'stanford_cars_cropped', 'flowers']
    
    for dataset in datasets:
        dataset_path = os.path.join(data_root, dataset)
        
        if not os.path.exists(dataset_path):
            print(f"❌ 数据集不存在: {dataset_path}")
            continue
            
        print(f"\n📁 检查数据集: {dataset}")
        
        for split in ['train', 'test']:
            split_path = os.path.join(dataset_path, split)
            
            if not os.path.exists(split_path):
                print(f"  ❌ 分割不存在: {split}")
                continue
                
            # 统计类别数量
            classes = [d for d in os.listdir(split_path) 
                      if os.path.isdir(os.path.join(split_path, d))]
            
            # 统计图像数量
            total_images = 0
            for class_name in classes:
                class_path = os.path.join(split_path, class_name)
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                total_images += len(images)
            
            print(f"  ✅ {split}: {len(classes)} 个类别, {total_images} 张图像")
            
            # 显示前几个类别名称
            if len(classes) > 0:
                print(f"     前5个类别: {classes[:5]}")


def test_image_loading():
    """
    测试图像加载功能
    """
    print("\n=== 测试图像加载 ===")
    
    data_root = "data"
    test_datasets = [
        ('cubs_cropped', 'cubs'),
        ('stanford_cars_cropped', 'cars'), 
        ('flowers', 'flowers')
    ]
    
    for folder_name, dataset_name in test_datasets:
        dataset_path = os.path.join(data_root, folder_name, 'train')
        
        if not os.path.exists(dataset_path):
            print(f"❌ 跳过 {dataset_name}: 路径不存在")
            continue
            
        print(f"\n📸 测试 {dataset_name} 图像加载:")
        
        # 找到第一个类别的第一张图像
        classes = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
        
        if len(classes) == 0:
            print(f"  ❌ 没有找到类别文件夹")
            continue
            
        first_class = classes[0]
        class_path = os.path.join(dataset_path, first_class)
        
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if len(images) == 0:
            print(f"  ❌ 类别 {first_class} 中没有图像")
            continue
            
        # 尝试加载第一张图像
        first_image_path = os.path.join(class_path, images[0])
        
        try:
            with Image.open(first_image_path) as img:
                print(f"  ✅ 成功加载图像: {first_image_path}")
                print(f"     图像尺寸: {img.size}")
                print(f"     图像模式: {img.mode}")
                print(f"     类别: {first_class}")
                
        except Exception as e:
            print(f"  ❌ 加载图像失败: {e}")


def test_packnet_logic():
    """
    测试PackNet算法逻辑（不依赖Jittor）
    """
    print("\n=== 测试PackNet算法逻辑 ===")
    
    # 模拟权重和掩码
    import numpy as np
    
    # 模拟一个简单的权重矩阵
    np.random.seed(42)
    weights = np.random.randn(10, 10)
    
    print("原始权重形状:", weights.shape)
    print("原始权重范围:", f"{weights.min():.3f} ~ {weights.max():.3f}")
    
    # 第一次剪枝：保留25%的权重（剪枝75%）
    pruning_ratio = 0.75
    threshold = np.percentile(np.abs(weights), pruning_ratio * 100)
    
    mask1 = np.abs(weights) > threshold
    pruned_weights1 = weights * mask1
    
    print(f"\n第一次剪枝（剪枝率{pruning_ratio:.0%}）:")
    print("保留的权重数量:", np.sum(mask1))
    print("剪枝的权重数量:", np.sum(~mask1))
    print("实际剪枝率:", f"{np.sum(~mask1) / weights.size:.1%}")
    
    # 第二次剪枝：只对剩余的权重进行剪枝
    available_weights = weights[mask1]  # 只考虑第一次剪枝后剩余的权重
    
    if len(available_weights) > 0:
        threshold2 = np.percentile(np.abs(available_weights), pruning_ratio * 100)
        mask2 = np.abs(weights) > threshold2
        mask2 = mask2 & (~mask1)  # 第二次剪枝不能影响第一次剪枝保留的权重
        
        print(f"\n第二次剪枝:")
        print("新保留的权重数量:", np.sum(mask2))
        print("总保留权重数量:", np.sum(mask1) + np.sum(mask2))
        print("总剪枝率:", f"{(weights.size - np.sum(mask1) - np.sum(mask2)) / weights.size:.1%}")
    
    print("✅ PackNet算法逻辑测试完成")


def generate_requirements():
    """
    生成requirements.txt文件
    """
    print("\n=== 生成requirements.txt ===")
    
    requirements = [
        "# PackNet项目依赖",
        "# 核心框架", 
        "jittor>=1.3.0",
        "",
        "# 数据处理",
        "numpy>=1.19.0",
        "Pillow>=8.0.0",
        "",
        "# 可选依赖",
        "matplotlib>=3.3.0  # 用于可视化",
        "tqdm>=4.60.0       # 进度条",
        ""
    ]
    
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(requirements))
    
    print("✅ requirements.txt 已生成")


def print_project_structure():
    """
    打印项目结构说明
    """
    print("\n=== PackNet项目结构说明 ===")
    
    structure = """
    📦 packnet-jittor/
    ├── 📄 dataset.py          # 数据加载器实现
    ├── 📄 pruning.py          # 剪枝和掩码功能
    ├── 📄 main.py             # 主训练脚本
    ├── 📄 test_basic.py       # 基础测试脚本
    ├── 📄 requirements.txt    # 项目依赖
    ├── 📁 data/               # 数据集目录
    │   ├── 📁 cubs_cropped/
    │   ├── 📁 stanford_cars_cropped/
    │   └── 📁 flowers/
    └── 📁 checkpoints/        # 模型和掩码保存目录
    """
    
    print(structure)
    
    print("\n🚀 使用说明:")
    print("1. 安装依赖: pip install -r requirements.txt")
    print("2. 测试数据: python test_basic.py")
    print("3. 运行实验: python main.py")
    
    print("\n📝 代码特点:")
    print("- ✅ 严格按照论文实现PackNet算法")
    print("- ✅ 支持多任务连续学习")
    print("- ✅ 实现了论文中的剪枝策略")
    print("- ✅ 包含完整的训练和评估流程")
    print("- ✅ 详细的注释和文档")


def main():
    """
    主测试函数
    """
    print("PackNet项目基础功能测试")
    print("=" * 60)
    
    # 测试数据集结构
    test_data_structure()
    
    # 测试图像加载
    test_image_loading()
    
    # 测试算法逻辑
    test_packnet_logic()
    
    # 生成requirements文件
    generate_requirements()
    
    # 打印项目结构
    print_project_structure()
    
    print("\n" + "=" * 60)
    print("✅ 基础测试完成！")
    print("如果所有测试都通过，说明数据集和代码结构都是正确的。")
    print("接下来可以安装Jittor并运行完整的训练脚本。")


if __name__ == "__main__":
    main()
