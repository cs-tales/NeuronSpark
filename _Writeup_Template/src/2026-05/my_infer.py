from my_train import MyModel, load_and_preprocess
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

LABELS = ['alarm', 'ambient', 'applause', 'door', 'footsteps',
          'glass_break', 'keyboard', 'rain', 'vehicle']

ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

class TestDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path: str, audio_dir: str):
        super().__init__()
        self.csv_path = csv_path
        self.audio_dir = audio_dir
        df = pd.read_csv(csv_path)

        audio_paths = df["audio"].astype(str).tolist()
        audio_feats = np.stack([
            load_and_preprocess(self._resolve_audio_path(p))[0] for p in audio_paths
        ], axis=0)

        df = pd.get_dummies(df, columns=["site", "device"])

        self.ids = df["id"].tolist()

        df.drop(columns=["audio", "id", "duration_sec"], inplace=True)
        x_meta = df.to_numpy(dtype=np.float32)
        self.x_audio = torch.from_numpy(audio_feats)
        self.x_meta = torch.from_numpy(x_meta)

        print(type(self.x_audio), self.x_audio.shape)
        print(type(self.x_meta), self.x_meta.shape)

    def _resolve_audio_path(self, audio_path: str) -> str:
        if os.path.isabs(audio_path) or os.path.exists(audio_path):
            return audio_path
        return os.path.join(self.audio_dir, audio_path)

    def __len__(self):
        return len(self.x_audio)
    
    def __getitem__(self, idx):
        return self.x_audio[idx], self.x_meta[idx]
    

def main_infer():
    # 加载测试集数据
    test_dataset = TestDataset('./test.csv', './audio/test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(
        meta_dim=test_dataset.x_meta.shape[1],
        embed_dim=test_dataset.x_audio.shape[1]
    ).to(device)
    model.load_state_dict(torch.load("./best_model.pth"))

    # 预测
    model.eval()
    preds = []
    with torch.no_grad():
        for x_audio, x_meta in test_loader:
            logits = model(x_audio.to(device), x_meta.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            for row in probs:
                labels = [LABELS[i] for i, s in enumerate(row) if s >= 0.1]
                if not labels:
                    labels = [LABELS[int(np.argmax(row))]]
                if "ambient" in labels and len(labels) > 1:
                    labels = ["ambient"]
                preds.append("|".join(labels))

    out = pd.DataFrame({"id": test_dataset.ids, "labels": preds})
    out.to_csv("results.csv", index=False)

if __name__ == '__main__':
    main_infer()