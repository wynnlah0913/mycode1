import os
import json
import torch
from torch.utils.data import Dataset, DataLoader


class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, label_map_path, seq_len=600):
        """
        Args:
            data_dir (str): JSON 資料所在資料夾
            label_map_path (str): labels.json 檔案路徑
            seq_len (int): 固定序列長度（不足補零，超過截斷）
        """
        self.data_dir = data_dir
        self.seq_len = seq_len

        # 載入 label map
        with open(label_map_path, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)

        # 收集所有 JSON 檔案路徑
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        json_path = self.files[idx]

        with open(json_path, "r", encoding="utf-8") as f:
            sample = json.load(f)

        label = sample["label"]
        data = sample["data"]  # list of frames，每個 frame 是一個節點向量

        # 轉成 tensor
        x = torch.tensor(data, dtype=torch.float32)  # [seq_len_raw, feature_dim]

        # 固定長度處理（padding 或截斷）
        if x.shape[0] < self.seq_len:
            pad_len = self.seq_len - x.shape[0]
            pad = torch.zeros(pad_len, x.shape[1])
            x = torch.cat([x, pad], dim=0)
        else:
            x = x[:self.seq_len, :]

        # 取得 label id
        y = self.label_map[label]

        return x, y


def get_dataloader(data_dir, label_map_path, batch_size=8, shuffle=True, seq_len=600):
    dataset = SignLanguageDataset(data_dir, label_map_path, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
