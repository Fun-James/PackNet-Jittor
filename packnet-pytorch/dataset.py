import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FinegrainedDataset(Dataset):
    """
    细粒度分类数据集的统一加载器，支持CUBS、Stanford Cars和Flowers
    """
    def __init__(self, dataset_name, data_root, split='train', transform=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.dataset_folders = {
            'cubs': 'cubs_cropped',
            'cars': 'stanford_cars_cropped',
            'flowers': 'flowers'
        }
        if dataset_name not in self.dataset_folders:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        self.dataset_path = os.path.join(data_root, self.dataset_folders[dataset_name])
        self.data_paths, self.labels, self.class_names = self._load_data()
        self.num_classes = len(self.class_names)
        print(f"加载 {dataset_name} 数据集 ({split}): {len(self.data_paths)} 张图像, {self.num_classes} 个类别")

    def _load_data(self):
        split_path = os.path.join(self.dataset_path, self.split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"路径不存在: {split_path}")
        data_paths = []
        labels = []
        class_names = []
        class_folders = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        for class_idx, class_folder in enumerate(class_folders):
            class_path = os.path.join(split_path, class_folder)
            class_names.append(class_folder)
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                data_paths.append(image_path)
                labels.append(class_idx)
        return data_paths, labels, class_names

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image_path = self.data_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像 {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        label = int(label)
        return image, label

def get_transforms(dataset_name, split='train'):
    if dataset_name in ['cubs', 'cars']:
        if split == 'train':
            transform_ops = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform_ops = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif dataset_name == 'flowers':
        if split == 'train':
            transform_ops = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform_ops = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    return transform_ops

def create_dataloader(dataset_name, data_root, split='train', batch_size=32, shuffle=True, num_workers=8):
    transform_ops = get_transforms(dataset_name, split)
    dataset = FinegrainedDataset(
        dataset_name=dataset_name,
        data_root=data_root,
        split=split,
        transform=transform_ops
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=(split == 'train')
    )
    return loader, dataset.num_classes 