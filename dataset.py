"""
PackNet数据加载器
实现对CUBS、Stanford Cars和Flowers数据集的统一处理
按照论文第4节的预处理方法
"""

import os
import jittor as jt
from jittor.dataset import Dataset
import numpy as np
from PIL import Image
import jittor.transform as transform


class FinegrainedDataset(Dataset):
    """
    细粒度分类数据集的统一加载器
    支持CUBS、Stanford Cars和Flowers三个数据集
    """
    
    def __init__(self, dataset_name, data_root, split='train', transform=None):
        """
        初始化数据集
        
        Args:
            dataset_name: 数据集名称 ('cubs', 'cars', 'flowers')
            data_root: 数据根目录路径
            split: 数据集划分 ('train' or 'test')
            transform: 数据变换
        """
        super().__init__()
        
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.split = split
        self.transform = transform
        
        # 根据数据集名称确定实际的文件夹名称
        self.dataset_folders = {
            'cubs': 'cubs_cropped',
            'cars': 'stanford_cars_cropped', 
            'flowers': 'flowers'
        }
        
        if dataset_name not in self.dataset_folders:
            raise ValueError(f"不支持的数据集: {dataset_name}")
            
        self.dataset_path = os.path.join(data_root, self.dataset_folders[dataset_name])
        
        # 加载数据路径和标签
        self.data_paths, self.labels, self.class_names = self._load_data()
        self.num_classes = len(self.class_names)
        
        print(f"加载 {dataset_name} 数据集 ({split}): {len(self.data_paths)} 张图像, {self.num_classes} 个类别")
    
    def _load_data(self):
        """
        从目录结构中加载数据路径和标签
        
        Returns:
            data_paths: 图像文件路径列表
            labels: 对应的标签列表
            class_names: 类别名称列表
        """
        split_path = os.path.join(self.dataset_path, self.split)
        
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"路径不存在: {split_path}")
        
        data_paths = []
        labels = []
        class_names = []
        
        # 获取所有类别文件夹并排序（确保标签分配的一致性）
        class_folders = sorted([d for d in os.listdir(split_path) 
                               if os.path.isdir(os.path.join(split_path, d))])
        
        for class_idx, class_folder in enumerate(class_folders):
            class_path = os.path.join(split_path, class_folder)
            class_names.append(class_folder)
            
            # 获取该类别下的所有图像文件
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                data_paths.append(image_path)
                labels.append(class_idx)
        
        return data_paths, labels, class_names
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        """
        获取单个数据样本
        
        Args:
            idx: 数据索引
            
        Returns:
            image: 处理后的图像张量
            label: 对应的标签
        """
        image_path = self.data_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像 {image_path}: {e}")
            # 创建一个黑色的替代图像
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 确保标签是整数
        label = int(label)
        
        return image, label


def get_transforms(dataset_name, split='train'):
    """
    根据论文第4节的要求，为不同数据集获取相应的数据变换
    
    Args:
        dataset_name: 数据集名称
        split: 数据集划分
        
    Returns:
        transform: 数据变换组合
    """
    
    if dataset_name in ['cubs', 'cars']:
        # CUBS和Cars数据集：直接缩放到224x224
        if split == 'train':
            # 训练时：缩放 + 随机水平翻转
            transforms = transform.Compose([
                transform.Resize((224, 224)),
                transform.RandomHorizontalFlip(p=0.5),
                transform.ToTensor(),
                transform.ImageNormalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
            ])
        else:
            # 测试时：仅缩放
            transforms = transform.Compose([
                transform.Resize((224, 224)),
                transform.ToTensor(),
                transform.ImageNormalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
            ])
    
    elif dataset_name == 'flowers':
        # Flowers数据集：短边缩放到256，然后224x224裁剪
        if split == 'train':
            # 训练时：短边缩放 + 随机裁剪 + 随机水平翻转
            transforms = transform.Compose([
                transform.Resize(256),  # 短边缩放到256
                transform.RandomCrop(224),
                transform.RandomHorizontalFlip(p=0.5),
                transform.ToTensor(),
                transform.ImageNormalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
            ])
        else:
            # 测试时：短边缩放 + 中心裁剪
            transforms = transform.Compose([
                transform.Resize(256),  # 短边缩放到256
                transform.CenterCrop(224),
                transform.ToTensor(),
                transform.ImageNormalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
            ])
    
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    return transforms


def create_dataloader(dataset_name, data_root, split='train', batch_size=32, shuffle=True, num_workers=0):
    """
    创建数据加载器
    
    Args:
        dataset_name: 数据集名称
        data_root: 数据根目录
        split: 数据集划分
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 并行工作进程数
        
    Returns:
        dataloader: Jittor数据加载器
        num_classes: 类别数量
    """
    # 获取对应的数据变换
    transforms = get_transforms(dataset_name, split)
    
    # 创建数据集
    dataset = FinegrainedDataset(
        dataset_name=dataset_name,
        data_root=data_root,
        split=split,
        transform=transforms
    )
    
    # 创建数据加载器
    base_dataloader = jt.dataset.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=(split == 'train')  # 训练时丢弃最后不完整的batch
    )
    
    # 包装数据加载器以确保正确的类型转换
    class WrappedDataLoader:
        def __init__(self, base_loader):
            self.base_loader = base_loader
        
        def __iter__(self):
            for images, labels in self.base_loader:
                # 确保 labels 是 Jittor 张量
                if not isinstance(labels, jt.Var):
                    labels = jt.array(labels)
                yield images, labels
        
        def __len__(self):
            return len(self.base_loader)
    
    wrapped_dataloader = WrappedDataLoader(base_dataloader)
    
    return wrapped_dataloader, dataset.num_classes


if __name__ == "__main__":
    """
    测试数据加载器功能
    """
    data_root = "data"
    
    # 测试所有数据集
    for dataset_name in ['cubs', 'cars', 'flowers']:
        print(f"\n=== 测试 {dataset_name} 数据集 ===")
        
        # 测试训练集
        train_loader, num_classes = create_dataloader(
            dataset_name=dataset_name,
            data_root=data_root,
            split='train',
            batch_size=4,
            shuffle=True
        )
        
        print(f"训练集: {len(train_loader)} 个batch, {num_classes} 个类别")
        
        # 测试测试集
        test_loader, _ = create_dataloader(
            dataset_name=dataset_name,
            data_root=data_root,
            split='test',
            batch_size=4,
            shuffle=False
        )
        
        print(f"测试集: {len(test_loader)} 个batch")
        
        # 加载一个batch测试
        for images, labels in train_loader:
            print(f"图像形状: {images.shape}, 标签形状: {labels.shape}")
            print(f"标签范围: {labels.min()} - {labels.max()}")
            break
