import argparse
import json
import torch

from sign_lstm_attention import SignLSTMWithAttention


def load_label_map(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    # 反轉 {詞: id} -> {id: 詞}
    id2label = {v: k for k, v in label_map.items()}
    return id2label


def prepare_input(json_path, seq_len=600):
    with open(json_path, "r", encoding="utf-8") as f:
        sample = json.load(f)

    data = torch.tensor(sample["data"], dtype=torch.float32)  # [seq_len_raw, feature_dim]

    if data.shape[0] < seq_len:
        pad_len = seq_len - data.shape[0]
        pad = torch.zeros(pad_len, data.shape[1])
        data = torch.cat([data, pad], dim=0)
    else:
        data = data[:seq_len, :]

    return data.unsqueeze(0)  # [1, seq_len, feature_dim]


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入 label map
    id2label = load_label_map(args.labels)
    num_classes = len(id2label)

    # 建立模型
    model = SignLSTMWithAttention(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes
    ).to(device)

    # 載入訓練好的權重
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 準備輸入
    x = prepare_input(args.json_file, seq_len=args.seq_len).to(device)

    with torch.no_grad():
        outputs, attn_weights = model(x)
        _, predicted = torch.max(outputs, 1)
        predicted_label = id2label[predicted.item()]

    print(f"✅ 預測結果: {predicted_label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Sign Language Recognition")
    parser.add_argument("--json_file", type=str, required=True, help="要辨識的 JSON 檔案")
    parser.add_argument("--labels", type=str, default="./labels.json", help="labels.json 路徑")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/sign_model.pt", help="訓練好的模型檔案")
    parser.add_argument("--input_size", type=int, default=330, help="每個 frame 的特徵維度")
    parser.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM 層數")
    parser.add_argument("--seq_len", type=int, default=600, help="固定序列長度")

    args = parser.parse_args()
    inference(args)