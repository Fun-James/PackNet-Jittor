# pruning.py

import jittor as jt
import numpy as np
import os

class PackNetPruning:
    """
    PackNet剪枝功能的核心实现类
    使用数字标记的掩码系统（0=剪枝，1=任务1，2=任务2，...）
    """

    def __init__(self, model, prune_perc=0.5, previous_masks=None):
        self.model = model
        self.prune_perc = prune_perc
        self.previous_masks = previous_masks or {}
        
        # 计算当前数据集索引
        if self.previous_masks:
            valid_key = list(self.previous_masks.keys())[0]
            self.current_dataset_idx = int(self.previous_masks[valid_key].max())
        else:
            self.current_dataset_idx = 0
        
        self.current_masks = None

    def get_prunable_modules(self):
        """获取所有可剪枝的模块"""
        prunable_modules = []
        # 直接使用传入的 model (应为 model.shared)
        for module_idx, module in enumerate(self.model.modules()):
            if isinstance(module, (jt.nn.Conv2d, jt.nn.Linear)):
                prunable_modules.append((module_idx, module))
        return prunable_modules

    def make_finetuning_mask(self):
        """
        核心函数：为新任务准备掩码
        将之前剪枝的权重（标记为0）重新激活为新任务可用
        """
        print(f"为新任务准备掩码，新任务索引: {self.current_dataset_idx + 1}")
        
        if not self.previous_masks:
            # 如果没有之前的掩码，初始化全部为1
            self.current_dataset_idx = 1
            self.current_masks = {}
            for module_idx, module in self.get_prunable_modules():
                mask = jt.ones_like(module.weight)
                self.current_masks[module_idx] = mask
        else:
            # 增加数据集索引
            self.current_dataset_idx += 1
            
            # 将之前剪枝的权重（值为0）重新激活为新任务可用
            self.current_masks = {}
            for module_idx, module in self.get_prunable_modules():
                if module_idx in self.previous_masks:
                    mask = self.previous_masks[module_idx].clone()
                    # 关键：将剪枝的权重重新激活为新任务可用
                    mask[mask == 0] = self.current_dataset_idx
                    self.current_masks[module_idx] = mask
                else:
                    # 新层全部分配给新任务
                    mask = jt.ones_like(module.weight) * self.current_dataset_idx
                    self.current_masks[module_idx] = mask

    def prune(self):
        """
        对当前任务的权重进行剪枝
        只对属于当前任务的权重进行剪枝
        """
        print(f'开始剪枝，任务索引: {self.current_dataset_idx}')
        print(f'剪枝比例: {self.prune_perc * 100:.1f}%')
        
        if not self.current_masks:
            print("错误：没有当前掩码，请先调用make_finetuning_mask()")
            return
        
        for module_idx, module in self.get_prunable_modules():
            if module_idx in self.current_masks:
                mask = self.current_masks[module_idx]
                weight = module.weight
                
                # 只考虑属于当前任务的权重
                current_task_mask = (mask == self.current_dataset_idx)
                current_task_weights = weight[current_task_mask]
                
                if current_task_weights.numel() == 0:
                    # print(f"模块 {module_idx}: 没有属于当前任务的权重") # 可以注释掉以减少输出
                    continue
                
                # 计算剪枝阈值
                abs_weights = jt.abs(current_task_weights)
                num_to_prune = int(self.prune_perc * current_task_weights.numel())
                
                if num_to_prune > 0:
                    # 使用kthvalue找到阈值
                    flat_weights = abs_weights.flatten()
                    sorted_weights, _ = jt.sort(flat_weights)
                    
                    # 避免越界
                    if num_to_prune > len(sorted_weights):
                        num_to_prune = len(sorted_weights)
                    
                    if num_to_prune > 0:
                        cutoff_value = sorted_weights[num_to_prune - 1]
                        
                        # 创建移除掩码
                        remove_mask = (jt.abs(weight) <= cutoff_value) & current_task_mask
                        
                        # 更新掩码：被剪枝的权重标记为0
                        mask[remove_mask] = 0
                        
                        # 物理清零被剪枝的权重
                        weight[remove_mask] = 0.0
                        
                        # 统计信息
                        num_pruned = remove_mask.sum().item()
                        total_params = current_task_weights.numel()
                        print(f'模块 {module_idx}: 剪枝 {num_pruned}/{total_params} '
                              f'({100 * num_pruned / total_params:.2f}%)')
                
                # 更新当前掩码
                self.current_masks[module_idx] = mask

    def make_grads_zero(self,optimizer):
        """
        将不属于当前任务的权重的梯度设为0
        """
        if not self.current_masks:
            return
        
        for module_idx, module in self.get_prunable_modules():
            if module_idx in self.current_masks:
                mask = self.current_masks[module_idx]
                
                # 关键改动 2：使用 .opt_grad(optimizer) 来访问梯度
                grad = module.weight.opt_grad(optimizer)
                
                if grad is not None:
                    # 只有属于当前任务的权重才能有梯度
                    current_task_mask = (mask == self.current_dataset_idx)
                    
                    # 关键改动：使用 .logic_not() 来反转掩码
                    grad[current_task_mask == False] = 0

    def make_pruned_zero(self):
        """
        确保被剪枝的权重保持为0
        """
        if not self.current_masks:
            return
        
        for module_idx, module in self.get_prunable_modules():
            if module_idx in self.current_masks:
                mask = self.current_masks[module_idx]
                # 被剪枝的权重（标记为0）保持为0
                module.weight[mask == 0] = 0.0

    def apply_mask(self, dataset_idx):
        """
        应用特定任务的掩码，用于评估时恢复特定任务的权重
        """
        for module_idx, module in self.get_prunable_modules():
            if module_idx in self.current_masks:
                mask = self.current_masks[module_idx]
                weight = module.weight
                
                # 清零不属于指定任务的权重
                weight[mask == 0] = 0.0  # 被剪枝的权重
                weight[mask > dataset_idx] = 0.0  # 后续任务的权重

    @staticmethod
    def save_masks(masks, filepath):
        """保存掩码到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        masks_np = {k: v.numpy() for k, v in masks.items()}
        np.save(filepath, masks_np)
        print(f"掩码已保存到: {filepath}")

    @staticmethod
    def load_masks(filepath):
        """从文件加载掩码"""
        if not os.path.exists(filepath):
            return {}
        masks_np = np.load(filepath, allow_pickle=True).item()
        masks = {k: jt.array(v) for k, v in masks_np.items()}
        print(f"掩码已从文件加载: {filepath}")
        return masks