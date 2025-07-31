import torch
import torch.nn as nn
import numpy as np

# 超參數
seq_length = 100
hidden_size = 256
num_layers = 1
batch_size = 64
num_epochs = 3000
lr = 0.003

# 載入並處理中文語料
with open("chinese.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
raw_text = raw_text.replace("\n", "")
chars = sorted(list(set(raw_text)))
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for c, i in char_to_int.items()}
n_vocab = len(chars)

# 建立訓練資料
dataX = []
dataY = []
for i in range(0, len(raw_text) - seq_length):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print(f"訓練樣本數: {n_patterns}")

# reshape & normalize
X = np.reshape(dataX, (n_patterns, seq_length, 1)) / float(n_vocab)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(dataY, dtype=torch.long)

# 建立模型
class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, n_vocab)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 最後時間步
        x = self.linear(self.dropout(x))
        return x

model = CharModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 訓練模型
for epoch in range(num_epochs):
    permutation = torch.randperm(X.size(0))
    loss_epoch = 0
    for i in range(0, X.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X[indices], y[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {loss_epoch:.4f}")

# 儲存模型和字典
torch.save((model.state_dict(), char_to_int), "char_model.pth")
print("模型訓練完成並已儲存為 char_model.pth")
