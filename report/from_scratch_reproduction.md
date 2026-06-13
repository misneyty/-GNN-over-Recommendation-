# 从零训练完整复现说明

## 1. 目的

最终流程已经改为从随机初始化开始运行，不再读取以下任何历史权重：

```text
heterognn_structural_100epoch.pt
heterognn_100epoch.pt
heterognn_final_60epoch.pt
```

老师只需要准备官方 `data_file` 数据并执行一条命令，即可完成训练、校准、独立评估和 submission 生成。

## 2. 复现命令

```powershell
python main.py `
  --full-pipeline `
  --data-dir data_file `
  --device cuda `
  --batch-size 65536 `
  --snapshot-epochs 60,80,100 `
  --seed 0
```

不指定输出路径时默认生成：

```text
outputs/checkpoints/heterognn_from_scratch_full.pt
outputs/submissions/Submission_heterognn_from_scratch_full.csv
```

## 3. 已完成的完整实测

本项目已在本地删除历史权重依赖后完整运行 100 epoch。实测结果：

| 指标 | 结果 |
| --- | ---: |
| 调优集 F1 | 0.963276 |
| 独立评估 F1 | **0.963075** |
| 独立评估 Accuracy | 0.963098 |
| 独立评估 AUC | 0.994125 |
| 阈值 | 0.463 |

该分数比使用历史 checkpoint 得到的 `0.963584` 低 `0.000509`，但它是能够由当前提交代码从头复现的正式结果。

完整训练耗时约 31 分钟，其中 100 epoch HeteroGNN 训练约 28 分钟。

## 4. 完整流程

`--full-pipeline` 会依次执行：

1. 使用 `seed=0` 划分作者-论文训练边和验证边；
2. 从随机初始化创建 HeteroGNN；
3. 每个 epoch 使用固定种子重新采样负边；
4. 使用固定的 NumPy 排列和 PyTorch Generator 确定 batch 顺序；
5. 连续训练 100 epoch；
6. 在第 60、80、100 epoch 保存内存快照；
7. 使用三个快照分别计算作者-论文概率；
8. 从训练边、作者关系、论文引用和 `feature.pkl` 构造 89 个可解释特征；
9. 使用 40% 验证样本拟合候选校准器；
10. 使用 30% 验证样本选择分类阈值；
11. 使用前 70% 样本重新拟合最终校准器；
12. 在封存的最后 30% 上报告独立 F1；
13. 对测试边预测并生成 submission。

## 5. 为什么使用训练快照

60、80、100 epoch 快照来自同一次从零训练，不是外部预训练模型。

不同训练阶段关注的信息略有差异：

- 60 epoch 保留更平滑的关系模式；
- 80 epoch 提供中间训练阶段信息；
- 100 epoch 提供最终拟合结果。

校准器综合三个分数，可以降低单个训练阶段的偶然误差，同时只需要进行一次完整训练。

## 6. 可复现性设置

`src/utils.py` 中的 `configure_reproducibility(...)` 固定：

- Python hash seed；
- Python `random`；
- NumPy；
- PyTorch CPU；
- PyTorch CUDA；
- cuDNN deterministic；
- cuDNN benchmark；
- CUBLAS workspace；
- PyTorch deterministic algorithms。

每个 epoch 的 batch 顺序由 `seed * 1000 + epoch` 单独确定，不受其他代码消耗随机数的影响。

语义历史中位数使用 CPU 上的确定性下中位数计算，避免 CUDA median 的非确定性警告。

## 7. 训练配置

```text
model: HeteroGNN
hidden dimension: 64
layers: 2
epochs: 100
snapshot epochs: 60, 80, 100
batch size: 65536
optimizer: Adam
learning rate: 0.001
weight decay: 0.00001
validation ratio: 0.1
seed: 0
```

校准器：

```text
HistGradientBoostingClassifier
learning_rate: 0.025
max_iter: 600
max_leaf_nodes: 63
min_samples_leaf: 40
l2_regularization: 4.0
n_iter_no_change: 10
```

## 8. 环境

本次完整实测环境：

```text
Python: 3.9.24
PyTorch: 2.5.1+cu121
CUDA runtime: 12.1
NumPy: 2.0.2
Pandas: 2.3.3
SciPy: 1.13.1
scikit-learn: 1.6.1
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

`requirements.txt` 已固定主要 Python 依赖版本。不同型号 GPU 或不同 CUDA 构建可能产生很小的浮点差异，但不会再依赖仓库外或预先生成的模型权重。

复现本次 CUDA 12.1 环境可使用：

```powershell
pip install -r requirements-cu121.txt
```

## 9. 主要代码改动

### `src/full_pipeline.py`

新增完整流程，包括：

- 从零训练；
- 固定 batch 顺序；
- 快照保存；
- 快照集成打分；
- 特征构造；
- 校准、阈值和独立评估；
- checkpoint 与 submission 保存；
- 训练参数和软件环境元数据保存。

### `main.py`

新增：

```text
--full-pipeline
--snapshot-epochs
```

老师无需调用内部脚本。

### `src/utils.py`

新增 `configure_reproducibility(...)`，统一配置确定性运行。

### `src/structural_features.py`

将 CUDA median 改为 CPU 上的确定性下中位数，保持原特征定义。

### `requirements.txt`

固定本次实测使用的软件版本。

## 10. 输出检查

实测 submission：

```text
outputs/submissions/Submission_heterognn_from_scratch_full.csv
```

包含 `2,047,262` 行预测，与测试边数量完全一致；列名为：

```text
Index,Predicted
```

其中 `Predicted` 只包含 `0` 和 `1`。
