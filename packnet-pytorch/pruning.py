import torch
import numpy as np
import os

class PackNetPruning:
    """
    PackNet剪枝功能的核心实现类，使用数字标记的掩码系统（0=剪枝，1=任务1，2=任务2，...）
    """
    def __init__(self, model, prune_perc=0.5, previous_masks=None):
        self.model = model
        self.prune_perc = prune_perc
        self.previous_masks = previous_masks or {}
        if self.previous_masks:
            valid_key = list(self.previous_masks.keys())[0]
            self.current_dataset_idx = int(self.previous_masks[valid_key].max())
        else:
            self.current_dataset_idx = 0
        self.current_masks = None

    def get_prunable_modules(self):
        prunable_modules = []
        for module_idx, module in enumerate(self.model.modules()):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                prunable_modules.append((module_idx, module))
        return prunable_modules

    def make_finetuning_mask(self):
        print(f"为新任务准备掩码，新任务索引: {self.current_dataset_idx + 1}")
        if not self.previous_masks:
            self.current_dataset_idx = 1
            self.current_masks = {}
            for module_idx, module in self.get_prunable_modules():
                mask = torch.ones_like(module.weight.data)
                self.current_masks[module_idx] = mask
        else:
            self.current_dataset_idx += 1
            self.current_masks = {}
            for module_idx, module in self.get_prunable_modules():
                if module_idx in self.previous_masks:
                    mask = self.previous_masks[module_idx].clone()
                    mask[mask == 0] = self.current_dataset_idx
                    self.current_masks[module_idx] = mask
                else:
                    mask = torch.ones_like(module.weight.data) * self.current_dataset_idx
                    self.current_masks[module_idx] = mask

    def prune(self):
        print(f'开始剪枝，任务索引: {self.current_dataset_idx}')
        print(f'剪枝比例: {self.prune_perc * 100:.1f}%')
        if not self.current_masks:
            print("错误：没有当前掩码，请先调用make_finetuning_mask()")
            return
        for module_idx, module in self.get_prunable_modules():
            if module_idx in self.current_masks:
                mask = self.current_masks[module_idx]
                weight = module.weight.data
                current_task_mask = (mask == self.current_dataset_idx)
                current_task_weights = weight[current_task_mask]
                if current_task_weights.numel() == 0:
                    continue
                abs_weights = current_task_weights.abs()
                num_to_prune = int(self.prune_perc * current_task_weights.numel())
                if num_to_prune > 0:
                    sorted_weights, _ = torch.sort(abs_weights.flatten())
                    if num_to_prune > len(sorted_weights):
                        num_to_prune = len(sorted_weights)
                    if num_to_prune > 0:
                        cutoff_value = sorted_weights[num_to_prune - 1]
                        remove_mask = (weight.abs() <= cutoff_value) & current_task_mask
                        mask[remove_mask] = 0
                        weight[remove_mask] = 0.0
                        num_pruned = remove_mask.sum().item()
                        total_params = current_task_weights.numel()
                        print(f'模块 {module_idx}: 剪枝 {num_pruned}/{total_params} ({100 * num_pruned / total_params:.2f}%)')
                self.current_masks[module_idx] = mask

    def make_grads_zero(self):
        if not self.current_masks:
            return
        for module_idx, module in self.get_prunable_modules():
            if module_idx in self.current_masks:
                mask = self.current_masks[module_idx]
                if module.weight.grad is not None:
                    current_task_mask = (mask == self.current_dataset_idx)
                    module.weight.grad[~current_task_mask] = 0

    def make_pruned_zero(self):
        if not self.current_masks:
            return
        for module_idx, module in self.get_prunable_modules():
            if module_idx in self.current_masks:
                mask = self.current_masks[module_idx]
                module.weight.data[mask == 0] = 0.0

    def apply_mask(self, dataset_idx):
        masks_to_apply = self.current_masks if self.current_masks is not None else self.previous_masks
        for module_idx, module in self.get_prunable_modules():
            if module_idx in masks_to_apply:
                mask = masks_to_apply[module_idx]
                weight = module.weight.data
                keep_mask = (mask > 0) & (mask <= dataset_idx)
                weight[~keep_mask] = 0.0

    def get_biases(self):
        biases = {}
        for module_idx, module in self.get_prunable_modules():
            if hasattr(module, 'bias') and module.bias is not None:
                biases[module_idx] = module.bias.data.clone()
        return biases

    def restore_biases(self, biases):
        for module_idx, module in self.get_prunable_modules():
            if module_idx in biases:
                module.bias.data.copy_(biases[module_idx])

    @staticmethod
    def save_masks(masks, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        masks_np = {k: v.cpu().numpy() for k, v in masks.items()}
        np.save(filepath, masks_np)
        print(f"掩码已保存到: {filepath}")

    @staticmethod
    def load_masks(filepath):
        if not os.path.exists(filepath):
            return {}
        masks_np = np.load(filepath, allow_pickle=True).item()
        masks = {k: torch.tensor(v) for k, v in masks_np.items()}
        print(f"掩码已从文件加载: {filepath}")
        return masks 