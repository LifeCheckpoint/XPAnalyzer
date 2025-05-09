# XPAnalyzer

基于 Transformer 等模型的 XP 分析相关算法

## Tag Transformer

通过遮盖训练对角色标签局部共现关系建模，捕捉标签组非线性共现关系

- **标签预测**: 通过已有标签预测 `[MASK]` 标签或其它标签。
- **遮挡分析**: 解释性工作，分析输入标签对特定标签预测概率的影响。
- **Tags 嵌入可视化**: 使用 `t-SNE` 对 Tags 嵌入进行可视化。

## 安装

1. `git clone https://github.com/LifeCheckpoint/XPAnalyzer`。
2. 如未安装 `uv`，可通过以下命令安装：`pip install uv`
3. 安装依赖：`uv sync`

## 使用

### 配置

项目的主要配置位于 `config/` 目录下，可通过新建配置文件并在 `config/configs.py` 调整配置选择

配置支持调整训练、评估、测试等参数

### 模型

选择配置后，训练（以及测试用）模型放置于 `saves/experiment_name/` 下，`experiment_name` 通过配置设置

### 训练 Tag Transformer

```bash
uv run python train.py
```

或

```bash
python train.py
```

### 运行测试和工具

```bash
uv run python test.py
```

或

```bash
python test.py
```

可通过修改 `test.py` 的 `steps` 字典来选择运行特定的测试或工具。

## 数据集

[角色-tags对应数据集](https://github.com/Zzzzzzyt/moegirl-dataset/blob/main/moegirl/preprocess/data_min.json)