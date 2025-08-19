# PRCV Multi-Modal Person Re-identification

本项目是基于UNIReID-main修改的多模态人员重识别系统，支持红外、彩铅、素描、文本四种模态检索RGB图像。

## 功能特性

### 支持的模态
- **NIR (Near-Infrared)**: 近红外图像
- **CP (Colored Pencil)**: 彩铅图像  
- **SK (Sketch)**: 素描图像
- **TEXT**: 文本描述
- **VIS (RGB)**: RGB图像（目标模态）

### 检索任务
1. **单模态检索**: 支持任意单模态独立检索
2. **多模态组合检索**: 支持任意多模态组合检索（双模态、三模态、四模态）

## 数据集结构

```
/data/taoxuefeng/PRCV/
├── train/
│   ├── nir/          # 红外图像
│   ├── cp/           # 彩铅图像
│   ├── sk/           # 素描图像
│   ├── vis/          # RGB图像
│   └── text_annos.json  # 文本标注
└── val/
    ├── gallery/      # 测试集RGB图像
    ├── nir/          # 测试集红外图像
    ├── cp/           # 测试集彩铅图像
    ├── sk/           # 测试集素描图像
    └── val_queries.json  # 测试查询
```

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA 10.2+
- 其他依赖见requirements.txt

## 快速开始

### 自动环境检查
```bash
# 运行快速启动脚本
python quick_start.py

# 或者使用bash脚本（如果可用）
./quick_start.sh
```

这个脚本会自动：
- 检查Python和PyTorch环境
- 验证CUDA可用性
- 创建必要的输出目录
- 检查数据集路径
- 提供交互式训练选项

## 使用方法

### 1. 训练模型

```bash
# 使用配置文件训练（推荐）
python train_prcv_simple.py --config_file configs/prcv_config.yaml

# 或者直接指定参数
python train_prcv_simple.py --batch_size 32 --num_epoch 100 --lr 0.0001

# 如果遇到权限问题，也可以使用原始脚本但指定本地输出目录
python train_prcv.py --output_dir ./outputs --batch_size 32 --num_epoch 100 --lr 0.0001
```

### 2. 测试模型

```bash
# 使用配置文件测试（推荐）
python test_prcv_simple.py --config_file configs/prcv_config.yaml --checkpoint path/to/checkpoint.pth

# 或者直接指定参数
python test_prcv_simple.py --checkpoint path/to/checkpoint.pth --output_dir ./results

# 如果遇到权限问题，也可以使用原始脚本但指定本地输出目录
python test_prcv.py --output_dir ./outputs --checkpoint path/to/checkpoint.pth
```

### 3. 配置文件说明

主要配置参数：

- `dataset_name`: 数据集名称（PRCV）
- `root_dir`: 数据集根目录
- `pretrain_choice`: CLIP预训练模型选择
- `fusion_way`: 多模态融合方式（concat/attention/weighted_sum）
- `batch_size`: 批次大小
- `num_epoch`: 训练轮数
- `lr`: 学习率

## 模型架构

### 1. 特征提取器
- 使用CLIP预训练模型作为backbone
- 红外、彩铅、素描共享视觉编码器参数
- 文本使用独立的文本编码器

### 2. 多模态融合模块
- **Concat**: 特征拼接后投影
- **Attention**: 自注意力机制融合
- **Weighted Sum**: 可学习权重融合

### 3. 损失函数
- 对比学习损失（ITC Loss）
- ID分类损失
- 各模态独立损失

## 输出结果

测试完成后会生成`ranking_results.txt`文件，包含：

```
query_idx    query_type       ranking_list_idx
0            onemodal_NIR      [5,9,856,126,324,1768,…]
1            onemodal_NIR      [823,456,16,94,532,…]
...
37196        twomodal_CP_SK    [2522,3019,25,3472,…]
...
```

## 性能指标

- **R1, R5, R10**: Top-1, Top-5, Top-10准确率
- **mAP**: 平均精度均值
- **CMC**: 累积匹配特征曲线

## 注意事项

1. 确保数据集路径正确
2. 训练时建议使用GPU加速
3. 可根据硬件调整batch_size和num_workers
4. 多模态融合方式可根据任务需求选择

## 故障排除

### 常见问题

1. **CUDA内存不足**: 减小batch_size
2. **数据集加载失败**: 检查数据集路径和格式
3. **模型加载失败**: 检查checkpoint文件路径
4. **权限错误**: 使用`train_prcv_simple.py`和`test_prcv_simple.py`脚本

### 权限问题解决方案

如果遇到类似错误：
```
PermissionError: [Errno 13] Permission denied: '/data1'
```

**解决方案1**: 使用简化的脚本（推荐）
```bash
python train_prcv_simple.py --config_file configs/prcv_config.yaml
python test_prcv_simple.py --config_file configs/prcv_config.yaml --checkpoint path/to/checkpoint.pth
```

**解决方案2**: 指定本地输出目录
```bash
python train_prcv.py --output_dir ./outputs --config_file configs/prcv_config.yaml
python test_prcv.py --output_dir ./outputs --config_file configs/prcv_config.yaml --checkpoint path/to/checkpoint.pth
```

### 调试建议

1. 先在小数据集上测试
2. 检查日志输出
3. 验证数据预处理步骤
4. 确保输出目录有写入权限

### 数据集测试

在开始训练之前，建议先测试数据集：

```bash
# 检查数据集路径和结构
python check_data_paths.py

# 测试数据集修复后的功能
python test_prcv_fix.py

# 测试训练流程
python test_training_flow.py

# 运行完整代码测试
python test_code.py
```

## 扩展功能

### 添加新模态
1. 在`MultiModalDataset`中添加新模态处理
2. 在`MultiModalFusion`中添加融合逻辑
3. 更新配置文件

### 自定义损失函数
1. 在`MultiModalReID.forward()`中添加新损失
2. 更新训练循环中的损失计算

## 引用

如果使用本项目，请引用原始UNIReID论文：

```bibtex
@article{unireid,
  title={Towards Modality-Agnostic Person Re-identification with Descriptive Query},
  author={...},
  journal={...},
  year={...}
}
```
