import torch
import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict


class BalancedBatchSampler(Sampler):
    """
    平衡批次采样器
    确保每个batch中每个类别选择相同数量的样本
    与PyTorch DataLoader兼容的版本
    """
    
    def __init__(self, dataset, n_classes_per_batch, n_samples_per_class, shuffle=True):
        """
        Args:
            dataset: 数据集
            n_classes_per_batch: 每个batch选择的类别数量
            n_samples_per_class: 每个类别选择的样本数量
            shuffle: 是否打乱顺序
        """
        self.dataset = dataset
        self.n_classes_per_batch = n_classes_per_batch
        self.n_samples_per_class = n_samples_per_class
        self.shuffle = shuffle
        
        # 获取所有类别和对应的样本索引
        self.class_to_indices = defaultdict(list)
        
        # 适配PRCV数据集的数据结构
        # PRCV返回: (pid, image_id, vis_path, nir_path, cp_path, sk_path, caption)
        # 其中pid是第一个元素
        for idx, item in enumerate(dataset):
            if isinstance(item, tuple) and len(item) >= 1:
                # 第一个元素是pid (person ID)
                label = item[0]
                self.class_to_indices[label].append(idx)
            else:
                # 如果不是元组，尝试其他格式
                print(f"Warning: Unexpected dataset item format at index {idx}: {type(item)}")
                continue
        
        self.classes = list(self.class_to_indices.keys())
        self.n_classes = len(self.classes)
        
        # 计算每个epoch的batch数量
        self.batches_per_epoch = self.n_classes // self.n_classes_per_batch
        
        # 确保每个类别在每个epoch中都被选择
        if self.n_classes % self.n_classes_per_batch != 0:
            print(f"Warning: {self.n_classes} classes cannot be evenly divided by {self.n_classes_per_batch}")
            print(f"Will use {self.batches_per_epoch} batches per epoch")
        
        print(f"Dataset info: {self.n_classes} classes, {len(dataset)} samples")
        print(f"Classes per batch: {self.n_classes_per_batch}, Samples per class: {self.n_samples_per_class}")
        print(f"Batches per epoch: {self.batches_per_epoch}")
        
        # 预生成所有epoch的索引序列
        self._generate_epoch_indices()
    
    def _generate_epoch_indices(self):
        """预生成一个epoch的所有索引序列"""
        self.epoch_indices = []
        
        # 为每个batch选择类别和样本
        for batch_idx in range(self.batches_per_epoch):
            batch_indices = []
            
            # 选择这个batch的类别
            start_class_idx = batch_idx * self.n_classes_per_batch
            end_class_idx = min(start_class_idx + self.n_classes_per_batch, self.n_classes)
            batch_classes = self.classes[start_class_idx:end_class_idx]
            
            # 为每个类别选择样本
            for class_id in batch_classes:
                class_indices = self.class_to_indices[class_id]
                
                # 如果这个类别的样本不够，重复采样
                if len(class_indices) < self.n_samples_per_class:
                    # 重复采样
                    selected_indices = np.random.choice(
                        class_indices, 
                        size=self.n_samples_per_class, 
                        replace=True
                    )
                else:
                    # 随机选择不重复的样本
                    selected_indices = np.random.choice(
                        class_indices, 
                        size=self.n_samples_per_class, 
                        replace=False
                    )
                
                batch_indices.extend(selected_indices)
            
            self.epoch_indices.extend(batch_indices)
    
    def __iter__(self):
        """返回一个epoch的索引序列"""
        if self.shuffle:
            # 重新生成epoch索引并打乱
            self._generate_epoch_indices()
            np.random.shuffle(self.epoch_indices)
        
        return iter(self.epoch_indices)
    
    def __len__(self):
        """返回一个epoch的总样本数"""
        return len(self.epoch_indices)


class PRCVBalancedSampler(BalancedBatchSampler):
    """
    PRCV数据集专用的平衡采样器
    每个batch选择8个类别，每个类别选择4张图片
    """
    
    def __init__(self, dataset, n_classes_per_batch=8, n_samples_per_class=4, shuffle=True):
        super().__init__(dataset, n_classes_per_batch, n_samples_per_class, shuffle)
        
        # 验证参数
        if n_classes_per_batch * n_samples_per_class != 32:
            print(f"Warning: batch_size should be {n_classes_per_batch * n_samples_per_class} for balanced sampling")
        
        print(f"PRCV Balanced Sampler initialized:")
        print(f"  - Classes per batch: {n_classes_per_batch}")
        print(f"  - Samples per class: {n_samples_per_class}")
        print(f"  - Total samples per batch: {n_classes_per_batch * n_samples_per_class}")
        print(f"  - Total classes: {self.n_classes}")
        print(f"  - Batches per epoch: {self.batches_per_epoch}")
        print(f"  - Total samples per epoch: {len(self.epoch_indices)}")
    
    def get_batch_info(self):
        """获取批次信息统计"""
        class_selection_count = defaultdict(int)
        
        for idx in self.epoch_indices:
            item = self.dataset[idx]
            if isinstance(item, tuple) and len(item) >= 1:
                label = item[0]  # 第一个元素是pid
                class_selection_count[label] += 1
        
        print(f"\nBatch sampling statistics:")
        print(f"  - Total samples per epoch: {len(self.epoch_indices)}")
        print(f"  - Average selections per class: {np.mean(list(class_selection_count.values())):.2f}")
        print(f"  - Min selections per class: {min(class_selection_count.values())}")
        print(f"  - Max selections per class: {max(class_selection_count.values())}")
        
        return class_selection_count
