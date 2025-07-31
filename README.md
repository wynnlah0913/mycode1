# mycode1
# 中文文字生成模型（Character-Level Chinese Text Generator）

本專案是一個基於 PyTorch 實作的 **中文字符級文字生成模型**，使用 LSTM 架構，透過現代白話文語料進行訓練，可生成類似風格的連貫文本。

---

## 📦 專案內容

- `train_char_model.py`：訓練模型腳本，載入語料後訓練 LSTM 模型。
- `generate.py`：使用訓練好的模型產生新文字。
- `chinese.txt`：訓練用語料，約 3 萬字現代白話文，已清洗並格式化。
- `char_model.pth`：訓練完成後儲存的模型參數與字典（自動生成）。
- `README.md`：本說明文件。

---

## 🚀 快速開始

### 1️⃣ 安裝環境

請先安裝 PyTorch：

```bash
pip install torch
