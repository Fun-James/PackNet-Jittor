# 文件: evaluate.py (已修正加载顺序)

import jittor as jt
from networks import ModifiedVGG16
from pruning import PackNetPruning
from dataset import create_dataloader
import os

# --- 1. 配置 ---
jt.flags.use_cuda = 1 if jt.has_cuda else 0

# 要评估的任务 ('cubs', 'cars', 或 'flowers')
TASK_TO_EVALUATE = 'cars'  # 您可以修改这里来评估不同任务
TASK_LIST = ['cubs', 'cars', 'flowers'] # 必须和训练时的任务顺序一致

# 模型和掩码的路径
CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "packnet_model.pkl")
MASK_PATH = os.path.join(CHECKPOINT_DIR, "packnet_masks.npy")
DATA_ROOT = "data"

# --- 2. 构建模型结构 ---
print("正在初始化模型结构...")
# 初始化一个空白的VGG16模型
model = ModifiedVGG16()

# --- 3. 为模型添加所有任务的分类头 ---
# PackNet需要为每个任务动态添加分类头。
# 在评估时，我们必须先重建出训练完成时模型的完整结构。
print("正在为模型重建所有任务的分类头...")
for task in TASK_LIST:
    # 获取每个任务的类别数
    _, task_num_classes = create_dataloader(dataset_name=task, data_root=DATA_ROOT, split='test')
    model.add_dataset(task, task_num_classes)

# --- 4. 加载训练好的权重 ---
# 关键修正：在模型结构完全构建好之后，再加载权重
print("正在加载模型权重和偏置...")
if os.path.exists(MODEL_PATH):
    # 加载包含模型权重和偏置字典的检查点
    ckpt = jt.load(MODEL_PATH)
    model.load_state_dict(ckpt['model_state_dict'])
    dataset2biases = ckpt.get('dataset2biases', {}) # 使用 .get() 保证向后兼容
    print(f"模型权重已从 {MODEL_PATH} 加载。")
else:
    raise FileNotFoundError(f"未找到模型文件: {MODEL_PATH}")

# --- 5. 加载剪枝掩码 ---
print("正在加载掩码...")
if os.path.exists(MASK_PATH):
    # 使用pruning.py中的静态方法加载掩码
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
# 设置模型使用当前要评估任务的分类头
model.set_dataset(TASK_TO_EVALUATE)
print(f"模型已设置为评估 '{TASK_TO_EVALUATE}' 任务，类别数: {num_classes}")

# --- 8. 应用特定任务的掩码 ---
print("正在应用任务掩码和偏置...")
pruner = PackNetPruning(
    model=model.shared,
    previous_masks=masks
)

# 恢复偏置 (在应用掩码之前或之后都可以，但逻辑上在这里更清晰)
if TASK_TO_EVALUATE in dataset2biases:
    print(f"正在为任务 '{TASK_TO_EVALUATE}' 恢复偏置...")
    pruner.restore_biases(dataset2biases[TASK_TO_EVALUATE])
else:
    print(f"警告: 未找到任务 '{TASK_TO_EVALUATE}' 的偏置快照。")

# 应用掩码
task_index = TASK_LIST.index(TASK_TO_EVALUATE) + 2
pruner.apply_mask(task_index)
print(f"已应用任务索引 {task_index} ({TASK_TO_EVALUATE.upper()}) 的掩码。")


# --- 9. 执行评估 ---
def evaluate(dataloader):
    """评估模型性能"""
    model.eval()
    correct = 0
    total = 0
    with jt.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            predicted = jt.argmax(outputs, dim=1)[0]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

print("\n开始评估...")
accuracy = evaluate(test_loader)
print(f"\n{'='*30}")
print(f"任务 '{TASK_TO_EVALUATE.upper()}' 的最终准确率: {accuracy:.2f}%")
print(f"{'='*30}")