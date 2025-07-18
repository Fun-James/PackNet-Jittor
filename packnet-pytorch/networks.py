import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class ModifiedVGG16(nn.Module):
    """
    修改的VGG16模型，支持多任务学习，每个任务独立分类头
    """
    def __init__(self):
        super().__init__()
        self.make_model()

    def make_model(self):
        vgg16_pretrained = vgg16(weights=VGG16_Weights.DEFAULT)
        self.datasets = []
        self.classifiers = nn.ModuleList()
        # 提取特征层
        features = list(vgg16_pretrained.features.children())
        # 提取FC6和FC7
        fc6 = vgg16_pretrained.classifier[0]  # Linear(25088, 4096)
        fc7 = vgg16_pretrained.classifier[3]  # Linear(4096, 4096)
        shared_layers = features + [
            nn.Flatten(),
            fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            fc7,
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        ]
        self.shared = nn.Sequential(*shared_layers)
        self.classifier = None

    def add_dataset(self, dataset, num_outputs):
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            device = next(self.shared.parameters()).device
            self.classifiers.append(nn.Linear(4096, num_outputs).to(device))

    def set_dataset(self, dataset):
        if dataset in self.datasets:
            idx = self.datasets.index(dataset)
            self.classifier = self.classifiers[idx]
        else:
            raise ValueError(f"数据集 {dataset} 未找到")

    def forward(self, x):
        x = self.shared(x)
        if self.classifier is not None:
            x = self.classifier(x)
        return x

def get_model():
    return ModifiedVGG16() 