import json
import os
from PIL import Image
from tqdm import tqdm

def convert_animal_pose_to_yolo(json_path, img_dir, output_labels_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    os.makedirs(output_labels_dir, exist_ok=True)

    # 建立 ID 到檔名的映射
    # data['images'] 是 {'1': '2007_000063.jpg', ...}
    img_id_to_name = data['images']
    # print(img_id_to_name)

    # 處理標註
    print("轉換標註格式...")
    for ann in tqdm(data['annotations']):
        # print(ann)
        img_id = str(ann['image_id'])
        if img_id not in img_id_to_name:
          print(img_id)
          continue

        file_name = img_id_to_name[img_id]
        img_path = os.path.join(img_dir, file_name)

        if not os.path.exists(img_path):
          print(img_path)
          continue

        # 獲取圖片寬高 (JSON 沒提供)
        with Image.open(img_path) as im:
            img_w, img_h = im.size
            # print(img_w, img_h)

        # BBox 轉換
        bx, by, bw, bh = ann['bbox']
        # print(bx, by, bw, bh)
        x_center = (bx + bw/2) / img_w
        y_center = (by + bh/2) / img_h
        w_norm = bw / img_w
        h_norm = bh / img_h

        # 關鍵點轉換 (20個點，取全部)
        kpts = ann['keypoints']
        # print(kpts)
        yolo_kpts = []
        for kp in kpts:
            # kp 會是 [x, y, v]
            kx = kp[0] / img_w
            ky = kp[1] / img_h
            kv = kp[2]
            yolo_kpts.extend([kx, ky, kv])

        # 寫入 YOLO 格式檔案
        label_path = os.path.join(output_labels_dir, file_name.rsplit('.', 1)[0] + ".txt")
        with open(label_path, 'a') as f_out:
            # class_id 設為 0
            line = f"0 {x_center} {y_center} {w_norm} {h_norm} " + " ".join(map(str, yolo_kpts))
            f_out.write(line + "\n")

# 執行轉換
# animal_pose_data_dir = "./animal_pose_data/labels"
# os.makedirs(animal_pose_data_dir, exist_ok=True)
# convert_animal_pose_to_yolo(
#     './animal_pose_data/keypoints.json',
#     './animal_pose_data/images/images',
#     '/content/animal_pose_data/labels'
# )
