
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 1. 读取CSV
df = pd.read_csv("./train.csv")

# 2. 划分训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# 3. 标签映射
labels = ["invoice","receipt","schedule","poster","lab_note",
          "notice","handwritten_note","form","meeting_minutes","grade_report"]
label_to_idx = {l: i for i, l in enumerate(labels)}

# 4. 数据集类
class DocDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"./{row['image']}"
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = label_to_idx[row["label"]]
        return img, label

# 5. 数据增强与预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = DocDataset(train_df, train_transform)
val_dataset = DocDataset(val_df, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 6. 加载预训练模型
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(512, 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 7. 训练
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {acc:.2f}%")

# 8. 保存模型
torch.save(model.state_dict(), "baseline.pth")
print("模型已保存为 baseline.pth")
