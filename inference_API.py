import os
import pandas as pd
import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import List

# 導入你的自定義模組
from inference_data_loader import get_target_images
from inference_model_engine import DetectionEngine
from inference_metrology_tool import calculate_dist

# --- 1. 安全安全性配置 (測試帳號) ---
security = HTTPBasic()
USER_DB = {
    "guest_user": "animal_test_2026"
}

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    user = USER_DB.get(credentials.username)
    if user is None or user != credentials.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="帳號或密碼錯誤",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# --- 2. 初始化 API 伺服器 ---
app = FastAPI(
    title="Animal Metrology API ",
    description="基於級聯式深度學習任務的自動化動物量測系統",
    version="1.0.0"
)

# 初始化引擎 (全域共用避免重複載入模型)
engine = DetectionEngine()

# --- 3. 核心量測邏輯 (封裝原本的 run_project) ---
@app.post("/v1/measure/batch", tags=["Measurement"])
def run_batch_measurement(target_cat: str = 'cat', min_count: int = 2, username: str = Depends(authenticate)):
    """
    啟動批次處理任務：篩選 COCO 圖片 -> 偵測 Pose -> 分割 Seg -> 量測數據 -> 存檔
    """
    print(f"[*] 使用者 {username} 啟動了批次處理任務...")
    
    # 步驟 1: 篩選圖片
    image_urls = get_target_images(target_cat=target_cat, min_count=min_count)
    if not image_urls:
        return {"status": "error", "message": "未找到符合條件的圖片。"}

    all_measurements = []
    raw_dir = "raw_images"
    output_dir = "results"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 步驟 2: 逐張處理
    for idx, url in enumerate(image_urls):
        try:
            # 下載圖片
            response = requests.get(url, timeout=10)
            temp_path = os.path.join(raw_dir, f"raw_{idx}.jpg")
            with open(temp_path, 'wb') as f:
                f.write(response.content)

            # 階段一：Pose 偵測
            pose_res = engine.detect_pose(temp_path)
            img = pose_res.orig_img.copy()
            keypoints_all = pose_res.keypoints.xy.cpu().numpy()
            boxes_all = pose_res.boxes.xyxy.cpu().numpy()

            # 階段二：區域分割與量測
            for i, (box, kpts) in enumerate(zip(boxes_all, keypoints_all)):
                if len(kpts) < 2 or np.all(kpts[0] == 0): continue

                # 局部輪廓
                mask_data = engine.get_segment_in_box(temp_path, box)
                if mask_data is not None:
                    points = np.int32([mask_data])
                    cv2.polylines(img, points, isClosed=True, color=(255, 0, 255), thickness=2)

                # 關鍵點量測
                left_eye, right_eye = kpts[0], kpts[1]
                d_eyes = calculate_dist(left_eye, right_eye)

                # 視覺化
                p1, p2 = tuple(map(int, left_eye)), tuple(map(int, right_eye))
                cv2.circle(img, p1, 4, (0, 0, 255), -1)
                cv2.circle(img, p2, 4, (0, 0, 255), -1)
                cv2.line(img, p1, p2, (0, 255, 0), 2)

                all_measurements.append({
                    "image_idx": idx,
                    "animal_id": i,
                    "eye_dist_px": round(float(d_eyes), 2)
                })

            # 跨動物測量
            if len(keypoints_all) >= 2:
                re1, re2 = keypoints_all[0][1], keypoints_all[1][1]
                dist_between = calculate_dist(re1, re2)
                cv2.line(img, tuple(map(int, re1)), tuple(map(int, re2)), (0, 255, 255), 2)

            # 存檔
            save_path = os.path.join(output_dir, f"result_{idx}.jpg")
            cv2.imwrite(save_path, img)

        except Exception as e:
            print(f"處理失敗 {url}: {e}")
            continue

    # 步驟 3: 導出報表
    csv_path = os.path.join(output_dir, "measurement_results.csv")
    df = pd.DataFrame(all_measurements)
    df.to_csv(csv_path, index=False)

    return {
        "status": "success",
        "processed_images": len(image_urls),
        "total_animals_detected": len(all_measurements),
        "report_path": csv_path
    }

# --- 4. 啟動入口 ---
if __name__ == "__main__":
    # 提供兩種啟動模式：若環境有設定 API 模式則啟動伺服器，否則執行一次性任務
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        print("[*] 正在啟動 API 模式 (Swagger UI: http://localhost:8000/docs)")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # 如果直接執行，就跑一次預設任務 (原本 run_project 的行為)
        print("[*] 正在執行單次批次處理任務...")
        # 這裡可以直接呼叫內部的函式邏輯
        # 為了演示方便，我們用本地調用方式
        class MockCreds: username = "local_admin"
        run_batch_measurement(username="local_admin")