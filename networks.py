# networks.py

import jittor as jt
import jittor.nn as nn
from jittor.models import vgg16  # <-- 关键改动：从 jittor.models 导入 vgg16

class ModifiedVGG16(jt.nn.Module):
    """
    修改的VGG16模型，支持多任务学习
    (已修正 Jittor 模型加载方式)
    """
    
    def __init__(self):
        super().__init__()
        self.make_model()
        
    def make_model(self):
        """创建模型"""
        # 关键改动：使用导入的 vgg16 函数加载预训练模型
        vgg16_pretrained = vgg16(pretrained=True)
        
        # 初始化数据集和分类器列表
        self.datasets = []
        self.classifiers = jt.nn.ModuleList()
        
        # 提取特征层和FC层
        features = list(vgg16_pretrained.features.children())
        
        # 提取FC6和FC7层
        fc6 = vgg16_pretrained.classifier[0]  # 第一个Linear层
        fc7 = vgg16_pretrained.classifier[3]  # 第二个Linear层
        
        # 构建共享的特征提取器
        shared_layers = features + [
            jt.nn.Flatten(),
            fc6,
            jt.nn.ReLU(),
            jt.nn.Dropout(0.5),
            fc7,
            jt.nn.ReLU(),
            jt.nn.Dropout(0.5)
        ]
        
        self.shared = jt.nn.Sequential(*shared_layers)
        self.classifier = None
        
    def add_dataset(self, dataset, num_outputs):
        """添加新的数据集分类器"""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(jt.nn.Linear(4096, num_outputs))
            
    def set_dataset(self, dataset):
        """设置当前使用的分类器"""
        if dataset in self.datasets:
            idx = self.datasets.index(dataset)
            self.classifier = self.classifiers[idx]
        else:
            raise ValueError(f"数据集 {dataset} 未找到")
            
    def execute(self, x):
        """前向传播"""
        x = self.shared(x)
        if self.classifier is not None:
            x = self.classifier(x)
        return x