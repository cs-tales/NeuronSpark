# [OS物语] — NeuronSpark 2026 Writeup

成员：郑源羽、卢宇扬、王彦凯
提交日期：2026/05/29

## 2026-00

### 1. 解题概述

用一个神经网络训练图像分类模型。

思路：暂不考虑使用OCR技术，基于公开模型ResNet-18进行训练。

预期得分：500
实际得分：500

### 2. 关键改进

使用 transform 进行数据增强和预处理。
基于 ResNet-18 模型进行训练。

### 3. 验证与复现

```bash
cd ./src/2026-00
conda create -n 2026-00 python=3.10
conda activate 2026-00
pip install -r requirements.txt
python my_train.py
python my_infer.py
```

### 4. AI使用说明

使用的 AI 工具：Deepseek

使用 AI 解释了数据预处理的方法，进行了一些 bug 解释。

### 5. 证据截图

位于 `./screenshots/2026-00/` 目录下。

### 6. 代码包

位于 `./src/2026-00/` 目录下。

`my_train.py` 是模型训练文件。
`my_infer.py` 是模型推理并输出结果的文件。
`requirements.txt` 是依赖包列表。