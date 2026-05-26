import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import librosa
import os
from torch.utils.data import Dataset, DataLoader, random_split
from models import Cnn14

# ./models.py 和 ./pytorch_utils.py 拷贝自 https://github.com/qiuqiangkong/audioset_tagging_cnn
# ./Cnn14_mAP=0.431.pth 下载自 https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth

PANN_CKPT_PATH = "./Cnn14_mAP=0.431.pth"
PANN_SAMPLE_RATE = 32000
PANN_WINDOW_SIZE = 1024
PANN_HOP_SIZE = 320
PANN_MEL_BINS = 64
PANN_FMIN = 50
PANN_FMAX = 14000
PANN_CLASSES = 527

TARGET_SR = PANN_SAMPLE_RATE
TARGET_DUR = 6.0           # 6 秒
TARGET_LEN = int(TARGET_SR * TARGET_DUR)   # 96000 采样点

LABELS = ['alarm', 'ambient', 'applause', 'door', 'footsteps',
          'glass_break', 'keyboard', 'rain', 'vehicle']

BATCH_SIZE = 64
EPOCH = 50
LEARNING_RATE = 1e-3

RANDOM_SEED = 42

_PANN_MODEL = None
_PANN_DEVICE = None


def _get_panns_model(device: torch.device) -> torch.nn.Module:
    global _PANN_MODEL, _PANN_DEVICE
    if _PANN_MODEL is None or _PANN_DEVICE != device:
        if not os.path.exists(PANN_CKPT_PATH):
            raise FileNotFoundError(f"Checkpoint not found: {PANN_CKPT_PATH}")
        _PANN_MODEL = Cnn14(
            sample_rate=PANN_SAMPLE_RATE,
            window_size=PANN_WINDOW_SIZE,
            hop_size=PANN_HOP_SIZE,
            mel_bins=PANN_MEL_BINS,
            fmin=PANN_FMIN,
            fmax=PANN_FMAX,
            classes_num=PANN_CLASSES,
        )
        checkpoint = torch.load(PANN_CKPT_PATH, map_location=device)
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
        if isinstance(state_dict, dict):
            clean_state = {}
            for key, value in state_dict.items():
                clean_key = key[7:] if key.startswith("module.") else key
                clean_state[clean_key] = value
            _PANN_MODEL.load_state_dict(clean_state, strict=False)
        _PANN_MODEL.to(device)
        _PANN_MODEL.eval()
        _PANN_DEVICE = device
    return _PANN_MODEL


def _extract_panns_embedding(waveform: torch.Tensor, device: torch.device) -> torch.Tensor:
    model = _get_panns_model(device)
    with torch.no_grad():
        outputs = model(waveform)
    if isinstance(outputs, tuple):
        _, embedding = outputs
        return embedding
    if isinstance(outputs, dict):
        embedding = outputs.get("embedding")
        if embedding is None:
            embedding = outputs.get("embeddings")
        if embedding is None:
            raise ValueError("PANNs output does not include embedding.")
        return embedding
    raise ValueError("Unexpected PANNs output type.")


def load_and_preprocess(audio_path, pre_emphasis_coef=0.97):
    # 1. 读取：强制 32k 单声道
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    # print(type(y), type(sr), y.shape, sr)

    # 2. 长度对齐：不足补零，超出截断
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    else:
        y = y[:TARGET_LEN]

    # 3. 幅度归一化到 [-1, 1]
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # 4. 预加重：y[t] = x[t] - α * x[t-1]
    # y_pe = np.append(y[0], y[1:] - pre_emphasis_coef * y[:-1])
    # print(type(y_pe), y_pe.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    waveform = torch.from_numpy(y.astype(np.float32, copy=False)).unsqueeze(0).to(device)
    embedding = _extract_panns_embedding(waveform, device)

    return embedding.squeeze(0).cpu().numpy().astype(np.float32, copy=False), sr


def _resolve_audio_path(audio_path: str, audio_dir: str) -> str:
    if os.path.isabs(audio_path) or os.path.exists(audio_path):
        return audio_path
    return os.path.join(audio_dir, audio_path)

class MyDataset(Dataset):
    
    def __init__(self, csv_path: str, audio_dir: str):
        super().__init__()
        self.csv_path = csv_path
        self.audio_dir = audio_dir
        df = pd.read_csv(csv_path)

        audio_paths = df["audio"].astype(str).tolist()
        audio_feats = np.stack([
            load_and_preprocess(_resolve_audio_path(p, audio_dir))[0] for p in audio_paths
        ], axis=0)

        df = pd.get_dummies(df, columns=["site", "device"])
        y = df["labels"].str.get_dummies(sep="|").to_numpy(dtype=np.float32)
        print(df["labels"].str.get_dummies(sep="|").head())

        df.drop(columns=["audio", "id", "labels", "duration_sec"], inplace=True)
        x_meta = df.to_numpy(dtype=np.float32)
        self.x_audio = torch.from_numpy(audio_feats)
        self.x_meta = torch.from_numpy(x_meta)
        self.y = torch.from_numpy(y)

        print(type(self.x_audio), self.x_audio.shape)
        print(type(self.x_meta), self.x_meta.shape)
        print(type(self.y), self.y.shape)
        print(y[:5])

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x_audio[idx], self.x_meta[idx], self.y[idx]
    

class MyModel(nn.Module):

    def __init__(
        self,
        meta_dim: int,
        embed_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        output_dim: int = 9
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_audio, x_meta):
        h = torch.cat([x_audio, x_meta], dim=1)
        return self.net(h)


def main_train():
    
    dataset = MyDataset('./train.csv', './audio/train')
    meta_dim = dataset.x_meta.shape[1]
    embed_dim = dataset.x_audio.shape[1]
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel(meta_dim=meta_dim, embed_dim=embed_dim).to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')

    for epoch in range(EPOCH):
        model.train()
        for x_audio, x_meta, y in train_loader:
            x_audio = x_audio.to(device)
            x_meta = x_meta.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x_audio, x_meta)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

        model.eval()
        for x_audio, x_meta, y in val_loader:
            x_audio = x_audio.to(device)
            x_meta = x_meta.to(device)
            y = y.to(device)
            with torch.no_grad():
                y_hat = model(x_audio, x_meta)
                val_loss = loss_fn(y_hat, y)
                
        print(f"Epoch {epoch+1}/{EPOCH}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved Best Model")

if __name__ == '__main__':

    torch.manual_seed(RANDOM_SEED)
    # ds = MyDataset('./train.csv', './audio/train')
    main_train()
 