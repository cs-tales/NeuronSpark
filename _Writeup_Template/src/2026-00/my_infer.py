import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch
import torch.nn as nn

test_df = pd.read_csv("./test.csv")

labels = ["invoice","receipt","schedule","poster","lab_note",
          "notice","handwritten_note","form","meeting_minutes","grade_report"]

class TestDataset(Dataset):
    def __init__(self, df, transform):
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
        return img, row["id"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = TestDataset(test_df, transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.resnet18()
model.fc = nn.Linear(512, 10)
model.load_state_dict(torch.load("baseline.pth", map_location="cpu"))
model.eval()

results = []
with torch.no_grad():
    for images, ids in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for img_id, pred in zip(ids, predicted):
            results.append({"id": img_id, "label": labels[pred]})

out_df = pd.DataFrame(results)
out_df.to_csv("results.csv", index=False)
print("results.csv 已生成，共", len(out_df), "条预测")
