# 文件: evaluate_pytorch.py

import torch
from networks import ModifiedVGG16
from pruning import PackNetPruning
from dataset import create_dataloader
import os

# --- 1. 配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 要评估的任务 ('cubs', 'cars', 或 'flowers')
TASK_TO_EVALUATE = 'flowers'  # 可修改为 'cubs' 或 'flowers'
TASK_LIST = ['cubs', 'cars', 'flowers']  # 必须和训练时的任务顺序一致

# 模型和掩码的路径
CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "packnet_model.pth")
MASK_PATH = os.path.join(CHECKPOINT_DIR, "packnet_masks.npy")
DATA_ROOT = "data"

# --- 2. 构建模型结构 ---
print("正在初始化模型结构...")
model = ModifiedVGG16().to(device)

# --- 3. 为模型添加所有任务的分类头 ---
print("正在为模型重建所有任务的分类头...")
for task in TASK_LIST:
    _, task_num_classes = create_dataloader(dataset_name=task, data_root=DATA_ROOT, split='test')
    model.add_dataset(task, task_num_classes)

# --- 4. 加载训练好的权重 ---
print("正在加载模型权重和偏置...")
if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=device)
    # 处理意外的键
    state_dict = ckpt['model_state_dict']
    # 移除不在当前模型中的键
    for key in list(state_dict.keys()):
        if key.startswith('classifier.'):
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    dataset2biases = ckpt.get('dataset2biases', {})
    print(f"模型权重已从 {MODEL_PATH} 加载。")
else:
    raise FileNotFoundError(f"未找到模型文件: {MODEL_PATH}")

# --- 5. 加载剪枝掩码 ---
print("正在加载掩码...")
if os.path.exists(MASK_PATH):
    masks = PackNetPruning.load_masks(MASK_PATH)
else:
    raise FileNotFoundError(f"未找到掩码文件: {MASK_PATH}")

# --- 6. 准备当前任务的数据加载器 ---
print(f"正在为任务 '{TASK_TO_EVALUATE}' 准备测试数据...")
test_loader, num_classes = create_dataloader(
    dataset_name=TASK_TO_EVALUATE,
    data_root=DATA_ROOT,
    split='test',
    batch_size=32,
    shuffle=False
)

# --- 7. 配置模型以进行特定任务评估 ---
model.set_dataset(TASK_TO_EVALUATE)
print(f"模型已设置为评估 '{TASK_TO_EVALUATE}' 任务，类别数: {num_classes}")

# --- 8. 应用特定任务的掩码 ---
print("正在应用任务掩码和偏置...")
pruner = PackNetPruning(
    model=model.shared,
    previous_masks=masks
)

if TASK_TO_EVALUATE in dataset2biases:
    print(f"正在为任务 '{TASK_TO_EVALUATE}' 恢复偏置...")
    pruner.restore_biases(dataset2biases[TASK_TO_EVALUATE])
else:
    print(f"警告: 未找到任务 '{TASK_TO_EVALUATE}' 的偏置快照。")

task_index = TASK_LIST.index(TASK_TO_EVALUATE) + 2
pruner.apply_mask(task_index)
print(f"已应用任务索引 {task_index} ({TASK_TO_EVALUATE.upper()}) 的掩码。")

# --- 9. 执行评估 ---
def evaluate(dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

print("\n开始评估...")
accuracy = evaluate(test_loader)
print(f"\n{'='*30}")
print(f"任务 '{TASK_TO_EVALUATE.upper()}' 的最终准确率: {accuracy:.2f}%")
print(f"{'='*30}")