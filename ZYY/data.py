import torch
import torch.nn as nn
import librosa
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MultiLabelBinarizer
from sklearn.metrics import mean_squared_error
from tqdm import tqdm # 推荐用来显示进度条 (pip install tqdm)
import os
import glob
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import TensorDataset, DataLoader
# 首先是将原始音频数据读取进来，并转化为 (720, 96000) 的 numpy 数组
#模块一：DATA
def data_loader(file_path):
    # 按照 16000Hz 采样率加载音频文件
    y, sr = librosa.load(file_path, sr=16000)
    
    # 目标时长为 6s，对应采样点数为 6 * 16000 = 96000
    target_samples = 6 * 16000
    
    # 将音频固定为 96000 长度（不足6秒补零，超过6秒截断）
    audio_vector = librosa.util.fix_length(y, size=target_samples)
    
    # 【新增】使用 librosa 提取 Log-Mel 频谱特征
    # n_mels=128 表示使用 128 个梅尔频带，这将决定我们图像的高度
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    # 将功率谱转换为分贝 (Log 刻度)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel

# def load_train_dataset(audio_dir):
#     """
#     遍历指定音频目录，抽取该目录下的所有.wav文件，并将其转化为 (N, 96000) 的 numpy 数组。
#     """
#     # 匹配目录下所有wav文件路径
#     audio_paths = glob.glob(os.path.join(audio_dir, "*.wav"))
    
#     X_data = []
#     file_names = []
    
#     print(f"正在从 {audio_dir} 提取并构造音频数据阵列...")
#     for path in tqdm(audio_paths):
#         audio_vec = data_loader(path)
#         X_data.append(audio_vec)
#         file_names.append(os.path.basename(path))
        
#     X_data = np.array(X_data)
    
#     # 因为直接读取文件夹没有标签，所以返回所有的文件名 (用来方便之后回传给csv获取标签)
#     print(f"音频数据加载完成，形状为: {X_data.shape}")
#     return X_data, file_names

#
def prepare_dataset(csv_path="train.csv"):
    #读取csv文件
    df = pd.read_csv(csv_path)
    #对表格（DataFrame）中名为 labels 的那一列数据进行格式清洗和切分，把原本用竖线 | 隔开的字符串，变成一个真正的 Python 列表（List）。
    df['labels_list'] = df['labels'].apply(lambda x: str(x).split('|'))
    #将标签列表转换为 Multi-Hot 编码的矩阵 (大小为: 样本数 x 9)
    mlb = MultiLabelBinarizer()
    y_data = mlb.fit_transform(df['labels_list'])
     # 打印转换出的具体类别列表，确认是否是 9 个
    print(f"识别到的类别列表 ({len(mlb.classes_)}个):", mlb.classes_)

    #严格按照csv顺序读取相应的音频特征数据，确保每个样本的特征和标签一一对应
    X_data = []
    #提取log-mel频谱特征，并构造特征矩阵 X (大小为: 样本数 x 64 x 188)
    for path in tqdm(df['audio']):
        # 这里用 os.path.normpath 可以自动处理 Windows 的斜杠问题
        safe_path = os.path.normpath(path)
        features = data_loader(safe_path)
        X_data.append(features)
    X_data = np.array(X_data)
    X_data = np.expand_dims(X_data,axis=1)#增加一个维度，变成 (N, 1, 64, 188)，适合输入 CNN 模型
    print(f"数据处理完毕！特征 X 形状: {X_data.shape}, 标签 y 形状: {y_data.shape}")
    
    # 返回处理好的特征矩阵、标签矩阵，以及 binarizer (后面推理测试集时还会用到)
    return X_data, y_data, mlb,df


#######################################################
# #模块二 TRAIN
# import torch.nn.functional as F
# class AudioCNN(nn.Module):
#     def __init__(self, num_classes=9):
#         super(AudioCNN, self).__init__()
#         #FIRST CONV LAYER
#         self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)

        
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        

#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # 自适应池化层，强制将高度和宽度固定为 1x1 
#         # (这样就不用自己算经过三次 MaxPool 之后特征图剩余具体的 HxW 大小)
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # 全连接分类层
#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x):
#         # x.shape = (Batch, 1, 64, 188)
        
#         x = self.pool1(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool2(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
#         # 结果变为 (Batch, 128, 1, 1)
#         x = self.adaptive_pool(x)
        
#         # 展平为 (Batch, 128)
#         x = torch.flatten(x, 1)
        
#         # 输出为 (Batch, 9)
#         out = self.fc(x)
#         # 注意：这里千万不要加 Sigmoid，因为我们要用 BCEWithLogitsLoss
#         return out    
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
class AudioResNet(nn.Module):
    def __init__(self, num_classes=9):
        super(AudioResNet, self).__init__()
        
        # 1. 挂载标准的 ResNet34，并加装预训练权重 (提升巨大！)
        # 使用 weights='DEFAULT' 以获得在 ImageNet 上的预训练能力
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # 2. 魔改第一层：将接纳 3 通道 RGB 的卷积层，换成接纳 1 通道 Log-Mel 的卷积层
        # 我们保留原有的其它参数 (64个输出, 7x7卷积核, stride和padding)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1, # <--- 核心：改成 1 通道 
            out_channels=original_conv1.out_channels, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=original_conv1.bias
        )
        
        # 3. 魔改最后一层分类器：把原本 1000 分类的输出，爆改成我们的 9 分类
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # x.shape = (Batch, 1, 64, 188)
        # 直接把图喂给魔改版的 ResNet 消化
        return self.resnet(x)

# def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
#     # 检测是否有GPU可用
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用的设备: {device}")
#     model.to(device)
    
#     # 核心：多标签分类一定要用这个损失函数
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
    
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0.0
        
#         for batch_X, batch_y in train_loader:
#             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
#             # 清空导数缓存
#             optimizer.zero_grad()
            
#             # 前向传播
#             outputs = model(batch_X)
            
#             # 算损失
#             loss = criterion(outputs, batch_y)
            
#             # 反向传播更新
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
            
#         print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {total_loss/len(train_loader):.4f}")
        
#         # TODO: 这里还可以补充一段测试 val_loader 准确率的代码，你要试试先跑到这里吗？

#     return model

from sklearn.metrics import f1_score # 记得在文件上面 import 这个，用来算分

# 【修改】给函数增加了 pos_weight 参数的接收
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, pos_weight=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")
    model.to(device)
    
    # 核心：引入算好的正样本倾向权重
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_f1 = 0.0 # 记录最高分
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        # --- 训练阶段 ---
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- 验证阶段 (Early Stopping 核心逻辑) ---
        model.eval()
        val_preds = []
        val_trues = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                
                # 转换出 0/1 结果用来计算本地 F1 分数
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int) 
                
                val_preds.append(preds)
                val_trues.append(batch_y.cpu().numpy())
                
        val_preds = np.vstack(val_preds)
        val_trues = np.vstack(val_trues)
        
        # 使用 sklearn 的 f1_score 计算宏平均分，提前洞察比赛成绩
        current_val_f1 = f1_score(val_trues, val_preds, average='macro', zero_division=0)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Macro-F1: {current_val_f1:.4f}")
        
        # --- 如果更强了，就保存当前的最强形态 ---
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  🌟 发现新高分！已保存为 best_model.pth")

    print(f"\n训练结束！最好验证集 Macro-F1: {best_val_f1:.4f}")
    
    # 训练完后，把刚才表现最好的最强权重重新加载回模型身上，再 return 回去
    model.load_state_dict(torch.load("best_model.pth"))
    return model

if __name__ == "__main__":
    #提取全量数据
    X_all,y_all,mlb,df = prepare_dataset("./train.csv")
    #按照场景site划分防止过拟合
    groups = df['site'].values
    # 划分训练集和测试集,其中 test_size=0.2 表示 20% 的数据作为测试集，n_splits=1 表示只进行一次划分，random_state=42 是随机种子，确保每次运行结果一致。
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, val_idx = next(gss.split(X_all, y_all, groups=groups))
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val     = X_all[val_idx], y_all[val_idx]
    print(f"训练集数量: {len(X_train)}, 验证集数量: {len(X_val)}")

    # 将 numpy 数组转换为 PyTorch 的 Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# === 【新增】计算类别分布规律，生成 pos_weight ===
    # 统计 y_train 中每个类别的正样本数量
    # y_train 形状为 (Num_train, 9)
    pos_counts = y_train.sum(axis=0)
    total_counts = len(y_train)
    neg_counts = total_counts - pos_counts
    
    # 为了防止某个类别由于数量极少导致除以0报错，加上一个微小的 epsilon (1e-5)
    pos_weight_np = neg_counts / (pos_counts + 1e-5)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32)
    print(f"为应对类别不平衡，已计算出 pos_weight: \n{pos_weight.numpy()}")

    # 封装进DataLoader
    batch_size = 16
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === 将以下两行替换掉你原来的 for 循环测试代码 ===
    print("\n开始训练模型...")
    model = AudioResNet(num_classes=9)
    trained_model = train_model(model, train_loader, val_loader, epochs=30, lr=1e-4, pos_weight=pos_weight)
    
    # 训练结束后保存模型权重
    torch.save(trained_model.state_dict(), "model_weights.pth")
    print("模型权重已保存至 model_weights.pth")