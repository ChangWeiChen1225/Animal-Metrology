import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_animal_dataset(base_dir):
    # 取得所有轉換好的標註檔 (.txt)
    # 以 labels 為基準，確保每一張進去訓練的圖都有對應的標註
    all_labels = glob(f"{base_dir}/labels/*.txt")
    if not all_labels:
        print(f"在 {base_dir}/labels 找不到任何 .txt 標註檔")
        return

    # 切分數據 (80% 訓練, 20% 驗證)
    train_labels, val_labels = train_test_split(all_labels, test_size=0.2, random_state=42)

    # 定義處理邏輯
    for split, label_paths in [('train', train_labels), ('val', val_labels)]:
        img_dest = f"{base_dir}/{split}/images"
        label_dest = f"{base_dir}/{split}/labels"
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(label_dest, exist_ok=True)

        print(f"正在準備 {split} 資料集...")
        for lp in tqdm(label_paths):
            file_base = os.path.basename(lp).rsplit('.', 1)[0]
            img_name = f"{file_base}.jpg"
            img_src = f"{base_dir}/images/images/{img_name}"

            # 確保圖片存在才進行複製
            if os.path.exists(img_src):
                # 複製標註
                shutil.copy(lp, f"{label_dest}/{file_base}.txt")
                # 複製圖片
                shutil.copy(img_src, f"{img_dest}/{img_name}")

    print(f"訓練集路徑: {base_dir}/train (包含 {len(train_labels)} 組圖片與標註)")
    print(f"驗證集路徑: {base_dir}/val (包含 {len(val_labels)} 組圖片與標註)")

# 執行切割
# split_animal_dataset('/content/animal_pose_data')
