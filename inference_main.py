import os
import pandas as pd
import cv2
import numpy as np
import requests
from inference_data_loader import get_target_images
from inference_model_engine import DetectionEngine
from inference_metrology_tool import calculate_dist

def run_project():
    print("正在篩選 COCO 符合條件之圖片")
    image_urls = get_target_images(target_cat='cat', min_count=2)

    if not image_urls:
        print("未找到符合條件的圖片。")
        return

    # 初始化引擎 (內部包含 Pose 模型與 Seg 模型)
    engine = DetectionEngine()
    all_measurements = []

    # 建立 results/raw results 資料夾
    raw_dir = "raw_images"
    output_dir = "results"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for idx, url in enumerate(image_urls):
        # print(f"正在處理第 {idx+1} 張圖片: {url}")
        # 下載圖片並存在 raw_images 
        response = requests.get(url, stream=True)
        temp_path = os.path.join(raw_dir, f"raw_{idx}.jpg")
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        

        # 階段一：執行 Pose 偵測 (取得 Box 與 眼睛 Keypoints)
        pose_res = engine.detect_pose(temp_path)
        img = pose_res.orig_img.copy()
        
        keypoints_all = pose_res.keypoints.xy.cpu().numpy()
        boxes_all = pose_res.boxes.xyxy.cpu().numpy() # 取得所有 Bounding Boxes

        # 階段二：針對每個 Box 進行局部區域分割 (Segmentation)
        # 將 Box 傳給 engine，讓它只在該區域做 Seg
        for i, (box, kpts) in enumerate(zip(boxes_all, keypoints_all)):
            if len(kpts) < 2: continue

            # 局部輪廓處理
            # 這裡調用你構思的邏輯：利用 Box 資訊來優化 Segmentation
            mask_data = engine.get_segment_in_box(temp_path, box)
            
            if mask_data is not None:
                points = np.int32([mask_data])
                # 繪製該動物的專屬輪廓
                cv2.polylines(img, points, isClosed=True, color=(255, 0, 255), thickness=2)

            # 關鍵點量測
            left_eye, right_eye = kpts[0], kpts[1]
            d_eyes = calculate_dist(left_eye, right_eye)

            # 視覺化眼睛
            p1, p2 = tuple(map(int, left_eye)), tuple(map(int, right_eye))
            cv2.circle(img, p1, 4, (0, 0, 255), -1)
            cv2.circle(img, p2, 4, (0, 0, 255), -1)
            cv2.line(img, p1, p2, (0, 255, 0), 2)

            all_measurements.append({
                "image_idx": idx,
                "animal_id": i,
                "eye_dist_px": round(d_eyes, 2)
            })

        # 跨動物測量
        if len(keypoints_all) >= 2:
            re1, re2 = keypoints_all[0][1], keypoints_all[1][1]
            dist_between = calculate_dist(re1, re2)
            cv2.line(img, tuple(map(int, re1)), tuple(map(int, re2)), (0, 255, 255), 2)
            cv2.putText(img, f"Cross-Dist: {dist_between:.1f}px", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        save_path = os.path.join(output_dir, f"result_{idx}.jpg")
        cv2.imwrite(save_path, img)
        print(f"影像已儲存至: {save_path}")

    # 存檔
    csv_path = os.path.join(output_dir, "measurement_results.csv")
    pd.DataFrame(all_measurements).to_csv(csv_path, index=False)
    print(f"執行成功，數據與影像皆已存入 {output_dir} 資料夾")

if __name__ == "__main__":
    run_project()
