from ultralytics import YOLO
import torch
import os
import shutil
import glob
from training_split_data import split_animal_dataset
from training_data_pose_to_yolo import convert_animal_pose_to_yolo
from training_data_loader import download_training_data_json

# 下載數據 (假設已上傳 kaggle.json)
download_training_data_json()


# 執行轉換
animal_pose_data_dir = "./animal_pose_data/labels"
os.makedirs(animal_pose_data_dir, exist_ok=True)
convert_animal_pose_to_yolo(
    './animal_pose_data/keypoints.json',
    './animal_pose_data/images/images',
    '/content/animal_pose_data/labels'
)


# 執行切割
split_animal_dataset('/content/animal_pose_data')

yaml_content = """
path: /content/animal_pose_data  # 資料集根目錄
train: train/images
val: val/images

# 關鍵點設定：20 個點 (包含左右眼、耳、鼻、四肢關節等)
kpt_shape: [20, 3]
names:
  0: animal
"""

with open("animal_pose.yaml", "w") as f:
    f.write(yaml_content)

# 確定裝置
device_to_use = 0 if torch.cuda.is_available() else 'cpu'

# 指定上次訓練最後生成的權重路徑
last_weight = './runs/pose/animal_metrology/eye_detection/weights/last.pt'


if os.path.exists(last_weight):
    print(f"找到上次訓練紀錄，準備從 {last_weight} 接續...")

    # 強制修改權重檔內的目標 Epoch 數
    # 加入 weights_only=False 以繞過 PyTorch 2.6 的安全檢查
    ckpt = torch.load(last_weight, map_location='cpu', weights_only=False)

    # 修改目標 Epoch
    if 'train_args' in ckpt:
        ckpt['train_args']['epochs'] = 100
        print(f"[*] 已將原始訓練參數中的 epochs 改為 100")

    # 有些版本會存在不同的 key 裡
    if 'args' in ckpt:
        ckpt['args']['epochs'] = 100

    # 儲存回檔案
    torch.save(ckpt, last_weight)
    print("權重檔修改完成並已儲存。")
    
    # 直接載入 last.pt
    model = YOLO(last_weight)
    
    # 執行訓練，只需傳入 resume=True
    # 原本的 data, epochs, imgsz 等參數會自動從 last.pt 內部的快取讀取
    results = model.train(resume=True)
else:
    print("找不到 last.pt。")

    model = YOLO('yolov8n-pose.pt')

    results = model.train(
        data='animal_pose.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=device_to_use,  # 套用自動判斷的結果
        project='animal_metrology',
        name='eye_detection',
        save=True,           # 確保會儲存 Checkpoint
        save_period=5,       # 每 1 個 epoch 檢查一次是否要更新 best.pt (預設即為 1)
        exist_ok=True
    )


# 訓練完成後執行
# 設定來源與目標路徑
source_dir = './runs/pose/animal_metrology/eye_detection/weights'
target_dir = './weights'

# 1. 確保目標資料夾存在
os.makedirs(target_dir, exist_ok=True)

# 2. 取得 source_dir 內所有 .pt 檔案的路徑
pt_files = glob.glob(os.path.join(source_dir, '*.pt'))

if not pt_files:
    print(f"在 {source_dir} 中找不到任何 .pt 檔案。")
else:
    print(f"準備處理 {len(pt_files)} 個權重檔案...")
    for file_path in pt_files:
        # 取得檔名 (例如: epoch50.pt)
        file_name = os.path.basename(file_path)
        
        # 設定目標完整路徑
        dest_path = os.path.join(target_dir, file_name)
        
        # 執行移動 (保留原檔用 shutil.copy)
        shutil.copy(file_path, dest_path)

    print(f"權重整理完成，目前 {target_dir} 內共有 {len(os.listdir(target_dir))} 個檔案")
