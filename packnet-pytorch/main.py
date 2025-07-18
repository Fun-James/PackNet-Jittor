import os
import torch
import numpy as np
from networks import ModifiedVGG16
from dataset import create_dataloader
from pruning import PackNetPruning

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class PackNetTrainer:
    """
    PackNet训练器类，PyTorch实现
    """
    def __init__(self, data_root, save_dir="checkpoints"):
        self.data_root = data_root
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.task_list = ['cubs', 'cars', 'flowers']
        self.pruning_ratio = 0.75
        self.initial_lr = 0.001
        self.batch_size = 32
        self.finetune_epochs = 20
        self.post_prune_epochs = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ModifiedVGG16().to(self.device)
        self.previous_masks = self._init_masks()
        self.dataset2biases = {}
        print(f"PackNet训练器初始化完成\n任务序列: {self.task_list}\n剪枝率: {self.pruning_ratio * 100}%")

    def _init_masks(self):
        masks = {}
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                mask = torch.ones_like(module.weight.data)
                masks[module_idx] = mask
        return masks

    def train_epoch(self, dataloader, optimizer, criterion, pruner):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            pruner.make_grads_zero()
            optimizer.step()
            pruner.make_pruned_zero()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if batch_idx % 50 == 0:
                acc = 100.0 * correct / total if total > 0 else 0.0
                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {acc:.2f}%")
        return total_loss / len(dataloader), 100.0 * correct / total

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100.0 * correct / total

    def adjust_learning_rate(self, optimizer, epoch, base_lr):
        lr = base_lr * (0.1 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train_task(self, task_idx, task_name):
        print(f"\n{'='*50}\n训练任务 {task_idx + 1}: {task_name.upper()}\n{'='*50}")
        train_loader, num_classes = create_dataloader(
            dataset_name=task_name,
            data_root=self.data_root,
            split='train',
            batch_size=self.batch_size
        )
        test_loader, _ = create_dataloader(
            dataset_name=task_name,
            data_root=self.data_root,
            split='test',
            batch_size=self.batch_size,
            shuffle=False
        )
        self.model.add_dataset(task_name, num_classes)
        self.model.set_dataset(task_name)
        pruner = PackNetPruning(
            model=self.model.shared,
            prune_perc=self.pruning_ratio,
            previous_masks=self.previous_masks
        )
        pruner.make_finetuning_mask()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.initial_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        criterion = torch.nn.CrossEntropyLoss()
        print(f"\n阶段1: 微调训练 {task_name}")
        best_acc = 0.0
        for epoch in range(self.finetune_epochs):
            current_lr = self.adjust_learning_rate(optimizer, epoch, self.initial_lr)
            if epoch % 10 == 0:
                print(f"当前学习率: {current_lr}")
            print(f"Epoch {epoch + 1}/{self.finetune_epochs}")
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, pruner)
            test_acc = self.evaluate(test_loader)
            print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"  测试准确率: {test_acc:.2f}%")
            best_acc = max(best_acc, test_acc)
        print(f"微调完成，最佳准确率: {best_acc:.2f}%")
        print(f"\n阶段2: 剪枝 {task_name}")
        print("剪枝前评估:")
        pre_prune_acc = self.evaluate(test_loader)
        print(f"  剪枝前准确率: {pre_prune_acc:.2f}%")
        pruner.prune()
        print("剪枝后评估:")
        post_prune_acc = self.evaluate(test_loader)
        print(f"  剪枝后准确率: {post_prune_acc:.2f}%")
        if self.post_prune_epochs > 0:
            print(f"\n阶段3: 剪枝后微调 {task_name}")
            post_prune_base_lr = self.initial_lr * 0.1
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=post_prune_base_lr,
                momentum=0.9,
                weight_decay=5e-4
            )
            for epoch in range(self.post_prune_epochs):
                current_lr = self.adjust_learning_rate(optimizer, epoch, post_prune_base_lr)
                if epoch % 10 == 0:
                    print(f"当前学习率: {current_lr}")
                print(f"微调 Epoch {epoch + 1}/{self.post_prune_epochs}")
                train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, pruner)
                test_acc = self.evaluate(test_loader)
                print(f"  微调准确率: {test_acc:.2f}%")
        self.previous_masks = pruner.current_masks
        print(f"正在为任务 '{task_name}' 保存偏置快照...")
        self.dataset2biases[task_name] = pruner.get_biases()
        final_acc = self.evaluate(test_loader)
        return final_acc

    def save_checkpoint(self):
        checkpoint_path = os.path.join(self.save_dir, "packnet_model.pth")
        mask_path = os.path.join(self.save_dir, "packnet_masks.npy")
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'dataset2biases': self.dataset2biases
        }
        torch.save(ckpt, checkpoint_path)
        PackNetPruning.save_masks(self.previous_masks, mask_path)
        print(f"检查点已保存到: {checkpoint_path}")
        print(f"掩码已保存到: {mask_path}")

    def _initial_prune(self):
        """对ImageNet预训练模型执行初始剪枝"""
        print("\n" + "="*50)
        print("对ImageNet预训练模型（任务1）执行初始剪枝")
        print("="*50)
        
        # 创建剪枝器，此时的previous_masks全为1
        pruner = PackNetPruning(
            model=self.model.shared,
            prune_perc=self.pruning_ratio,
            previous_masks=self.previous_masks
        )
        
        # 直接使用previous_masks作为current_masks
        pruner.current_masks = self.previous_masks
        
        # 执行剪枝
        print("\n开始对ImageNet预训练权重进行剪枝...")
        pruner.prune()
        print("ImageNet预训练模型剪枝完成")
        
        # 更新masks
        self.previous_masks = pruner.current_masks
        print("已更新掩码，为后续任务准备空间")

    def run_experiment(self):
        print("开始PackNet多任务学习实验")
        import time
        start_time = time.time()
        
        # 阶段一：对ImageNet预训练模型进行初始剪枝
        self._initial_prune()
        
        # 阶段二：对后续任务执行完整的训练-剪枝-重训练循环
        task_results = {}
        for task_idx, task_name in enumerate(self.task_list):
            final_acc = self.train_task(task_idx, task_name)
            task_results[task_name] = final_acc
            
        self.save_checkpoint()
        total_time = time.time() - start_time
        print(f"\nPackNet实验完成！\n总耗时: {total_time/60:.2f} 分钟")
        return

if __name__ == "__main__":
    set_seed(42)
    trainer = PackNetTrainer(data_root="data")
    trainer.run_experiment()