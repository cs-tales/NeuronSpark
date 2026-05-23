# 复现说明

## 环境配置

```bash
pip install -r requirements.txt
```

## 训练

```bash
python train.py
```

## 生成提交

```bash
python infer.py
```

## 注意事项

- 数据集请从比赛平台下载，放在 `data/` 目录下
- 模型权重将在训练后自动保存至 `checkpoints/` 目录
- 提交文件将输出至 `submissions/` 目录
