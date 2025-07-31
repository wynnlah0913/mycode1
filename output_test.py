import torch
import torch.nn as nn
import numpy as np

# 讀取模型與字典
model_path = "char_model.pth"
state_dict, char_to_int = torch.load(model_path)
int_to_char = {i: c for c, i in char_to_int.items()}
n_vocab = len(char_to_int)

# 模型架構（要跟訓練時一模一樣）
class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(self.dropout(x))
        return x

# 初始化模型與權重
model = CharModel()
model.load_state_dict(state_dict)
model.eval()

# 讀取語料做 prompt（用於選取起始文字）
with open("chinese.txt", "r", encoding="utf-8") as f:
    raw_text = f.read().replace("\n", "")
seq_length = 100
start = np.random.randint(0, len(raw_text) - seq_length)
prompt = raw_text[start:start + seq_length]
pattern = [char_to_int[c] for c in prompt]

# 顯示 prompt
print("輸入起始文本：")
print(prompt)
print("\n生成結果：")

# 生成文字
generated = ""
with torch.no_grad():
    for _ in range(300):  # 你可以改成 1000、500 等等
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        prediction = model(x)
        index = int(prediction.argmax())
        result = int_to_char[index]
        generated += result

        pattern.append(index)
        pattern = pattern[1:]

print(generated)
print("\n✅ 完成文字生成")
