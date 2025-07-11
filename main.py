"""
PackNet主训练脚本
按照原论文算法实现PackNet多任务学习
"""

import os
import jittor as jt
import jittor.nn as nn
import jittor.optim as optim
import numpy as np
import time

from dataset import create_dataloader
from pruning import PackNetPruning 
from networks import ModifiedVGG16

jt.flags.use_cuda = 1 if jt.has_cuda else 0

class PackNetTrainer:
    """
    PackNet训练器类，按照原论文算法实现
    """
    
    def __init__(self, data_root, save_dir="checkpoints"):
        self.data_root = data_root
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 任务配置
        self.task_list = ['cubs', 'cars', 'flowers']
        self.pruning_ratio = 0.75  # 原论文使用75%剪枝率
        
        # 训练参数
        self.initial_lr = 0.001
        self.batch_size = 32
        self.finetune_epochs =20
        self.post_prune_epochs = 10
        
        # 创建模型
        self.model = ModifiedVGG16()
        
        # 初始化掩码
        self.previous_masks = self._init_masks()
        self.dataset2biases = {}  # <-- 新增：初始化用于存储各任务偏置的字典
        
        print(f"PackNet训练器初始化完成")
        print(f"任务序列: {self.task_list}")
        print(f"剪枝率: {self.pruning_ratio * 100}%")
        
    def _init_masks(self):
        """初始化掩码，所有权重初始标记为1（属于ImageNet预训练）"""
        masks = {}
        for module_idx, module in enumerate(self.model.modules()):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                mask = jt.ones_like(module.weight)
                masks[module_idx] = mask
        return masks
    
    def train_epoch(self, dataloader, optimizer, criterion, pruner):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            # 前向传播
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.backward(loss)
            
            # 关键改动：将 optimizer 传递给 make_grads_zero
            pruner.make_grads_zero(optimizer)
            
            # 更新参数
            optimizer.step()
            
            # 关键：确保被剪枝的权重保持为0
            pruner.make_pruned_zero()
            
            # 统计
            total_loss += loss.item()
            predicted = jt.argmax(outputs, dim=1)[0]

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                acc = 100.0 * correct / total
                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {acc:.2f}%")
                
        return total_loss / len(dataloader), 100.0 * correct / total
    
    def evaluate(self, dataloader, dataset_idx=None):
        """评估模型性能"""
        self.model.eval()
        correct = 0
        total = 0
        
        with jt.no_grad():
            for images, labels in dataloader:
                outputs = self.model(images)
                predicted = jt.argmax(outputs, dim=1)[0]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return 100.0 * correct / total
    
    def adjust_learning_rate(self, optimizer, epoch, base_lr):
        """每10个epoch将学习率乘以0.1"""
        lr = base_lr * (0.1 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def train_task(self, task_idx, task_name):
        """训练单个任务"""
        print(f"\n{'='*50}")
        print(f"训练任务 {task_idx + 1}: {task_name.upper()}")
        print(f"{'='*50}")
        
        # 1. 加载数据
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
        
        # 2. 添加新任务的分类器
        self.model.add_dataset(task_name, num_classes)
        self.model.set_dataset(task_name)
        
        # 3. 创建剪枝器
        pruner = PackNetPruning(
            model=self.model.shared,  # 只对共享层进行剪枝
            prune_perc=self.pruning_ratio,
            previous_masks=self.previous_masks
        )
        
        # 4. 关键步骤：为新任务准备掩码
        pruner.make_finetuning_mask()
        
        # 5. 设置优化器
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.initial_lr, 
            momentum=0.9, 
            weight_decay=5e-4
        )
        criterion = nn.CrossEntropyLoss()
        
        # 6. 训练阶段
        print(f"\n阶段1: 微调训练 {task_name}")
        best_acc = 0.0
        
        for epoch in range(self.finetune_epochs):
            # 调整学习率
            current_lr = self.adjust_learning_rate(optimizer, epoch, self.initial_lr)
            if epoch % 10 == 0:
                print(f"当前学习率: {current_lr}")
            
            print(f"Epoch {epoch + 1}/{self.finetune_epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, pruner)
            
            # 评估
            test_acc = self.evaluate(test_loader)
            
            print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"  测试准确率: {test_acc:.2f}%")
            
            best_acc = max(best_acc, test_acc)
        
        print(f"微调完成，最佳准确率: {best_acc:.2f}%")
        
        # 7. 剪枝阶段
        print(f"\n阶段2: 剪枝 {task_name}")
        print("剪枝前评估:")
        pre_prune_acc = self.evaluate(test_loader)
        print(f"  剪枝前准确率: {pre_prune_acc:.2f}%")
        
        # 执行剪枝
        pruner.prune()
        
        print("剪枝后评估:")
        post_prune_acc = self.evaluate(test_loader)
        print(f"  剪枝后准确率: {post_prune_acc:.2f}%")
        
        # 8. 剪枝后微调（可选）
        if self.post_prune_epochs > 0:
            print(f"\n阶段3: 剪枝后微调 {task_name}")
            
            # 为剪枝后微调创建新的优化器，使用更小的初始学习率
            post_prune_base_lr = self.initial_lr * 0.1
            optimizer = optim.SGD(
                self.model.parameters(), 
                lr=post_prune_base_lr,
                momentum=0.9, 
                weight_decay=5e-4
            )
            
            for epoch in range(self.post_prune_epochs):
                # 调整学习率
                current_lr = self.adjust_learning_rate(optimizer, epoch, post_prune_base_lr)
                if epoch % 10 == 0:
                    print(f"当前学习率: {current_lr}")
                
                print(f"微调 Epoch {epoch + 1}/{self.post_prune_epochs}")
                
                train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, pruner)
                test_acc = self.evaluate(test_loader)
                
                print(f"  微调准确率: {test_acc:.2f}%")
        
        # 9. 更新掩码
        self.previous_masks = pruner.current_masks
        
        # 10. 保存当前任务的偏置快照
        print(f"正在为任务 '{task_name}' 保存偏置快照...")
        self.dataset2biases[task_name] = pruner.get_biases()

        # 11. 最终评估
        final_acc = self.evaluate(test_loader)
        
        return final_acc
    
    def evaluate_all_tasks(self):
        """评估所有任务的性能"""
        print(f"\n{'='*50}")
        print("最终评估：所有任务性能")
        print(f"{'='*50}")
        
        results = {}
        
        for task_idx, task_name in enumerate(self.task_list):
            print(f"\n评估任务: {task_name.upper()}")
            
            # 加载测试数据
            test_loader, _ = create_dataloader(
                dataset_name=task_name, 
                data_root=self.data_root, 
                split='test', 
                batch_size=self.batch_size, 
                shuffle=False
            )
            
            # 设置模型到对应任务
            self.model.set_dataset(task_name)
            
            # 创建剪枝器并应用对应任务的掩码
            pruner = PackNetPruning(
                model=self.model.shared,
                previous_masks=self.previous_masks
            )
            pruner.apply_mask(task_idx + 1)  # 任务索引从1开始
            
            # 评估
            accuracy = self.evaluate(test_loader)
            results[task_name] = accuracy
            
            print(f"  {task_name} 最终准确率: {accuracy:.2f}%")
        
        # 打印汇总结果
        print(f"\n{'='*50}")
        print("PackNet实验结果汇总")
        print(f"{'-'*30}")
        for task_name, acc in results.items():
            print(f"{task_name.upper():<10s} | {acc:.2f}%")
        print(f"{'-'*30}")
        
        return results
    
    def save_checkpoint(self):
        """保存检查点"""
        checkpoint_path = os.path.join(self.save_dir, "packnet_model.pkl")
        mask_path = os.path.join(self.save_dir, "packnet_masks.npy")

        # 将模型和偏置都保存到 .pkl 文件中
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'dataset2biases': self.dataset2biases
        }
        jt.save(ckpt, checkpoint_path)

        # 保存掩码
        PackNetPruning.save_masks(self.previous_masks, mask_path)

        print(f"检查点已保存到: {checkpoint_path}")
        print(f"掩码已保存到: {mask_path}")
    
    def run_experiment(self):
        """运行完整的PackNet实验"""
        print("开始PackNet多任务学习实验")
        start_time = time.time()
        
        task_results = {}
        
        # 逐个训练任务
        for task_idx, task_name in enumerate(self.task_list):
            final_acc = self.train_task(task_idx, task_name)
            task_results[task_name] = final_acc
        
        # 保存检查点
        self.save_checkpoint()
        
        total_time = time.time() - start_time
        print(f"\nPackNet实验完成！")
        print(f"总耗时: {total_time/60:.2f} 分钟")
        
        return 


if __name__ == "__main__":
    # 设置随机种子
    jt.set_global_seed(42)
    np.random.seed(42)
    
    # 创建训练器并运行实验
    trainer = PackNetTrainer(data_root="data")
    results = trainer.run_experiment()