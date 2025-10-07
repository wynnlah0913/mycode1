import os
import re

# 設定要處理的資料夾路徑
folder_path = r'D:/video_downloaded/five'  # 請改成你的資料夾路徑

# 正則表達式：擷取【】中包含的中文字（排除英文與數字）
pattern = re.compile(r"【([^】]+)】")

for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)
    
    if not os.path.isfile(old_path):
        continue  # 跳過資料夾或非檔案

    match = pattern.search(filename)
    if match:
        chinese_text = match.group(1)
        _, ext = os.path.splitext(filename)  # 取得副檔名
        new_filename = f"{chinese_text}{ext}"  # 例如：轉成 "測試檔案.txt"
        new_path = os.path.join(folder_path, new_filename)
        
        # 若檔名衝突可加以處理
        if not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"已重新命名：{filename} → {new_filename}")
        else:
            print(f"跳過：{new_filename} 已存在")
    else:
        print(f"未找到括號中文字：{filename}")
