import numpy as np
from ultralytics import YOLO

class DetectionEngine:
    """
    整合 Pose 與 Segmentation 的雙模型偵測引擎
    1. 使用自定義的 Animal-Pose 模型抓取眼睛關鍵點與邊界框 (Box)
    2. 使用預訓練的 Segmentation 模型，並根據 Box 進行區域篩選以獲得輪廓
    """
    def __init__(self, pose_weight="./weights/best.pt", seg_weight="yolo11n-seg.pt"):
        # 載入自定義的動物姿勢模型 (用於量測)
        self.pose_model = YOLO(pose_weight)
        # 載入官方分割模型 (用於標出輪廓)
        self.seg_model = YOLO(seg_weight)

    def detect_pose(self, img_path):
        """
        執行 Pose 推論，回傳第一個結果物件
        包含：boxes, keypoints, orig_img 等
        """
        results = self.pose_model(img_path, verbose=False)
        return results[0]

    def get_segment_in_box(self, img_path, box):
        """
        利用 Box 資訊作為引導，過濾出該區域內的 Segmentation Mask。
        1. 執行全圖分割。
        2. 計算每個 Mask 的幾何中心點。
        3. 判斷中心點是否落在指定的 Box 範圍內，若是則回傳該 Mask 的座標點位。
        """
        # 執行分割模型推論
        results = self.seg_model(img_path, verbose=False)[0]
        
        if results.masks is not None:
            # 遍歷所有的 Mask 點位 (xy 格式)
            for mask in results.masks.xy:
                # 計算該輪廓的中心點 (平均值)
                mean_x, mean_y = np.mean(mask, axis=0)
                
                # 判斷中心點是否在 Pose 模型給出的 Box 內 [x1, y1, x2, y2]
                if box[0] <= mean_x <= box[2] and box[1] <= mean_y <= box[3]:
                    return mask
                    
        return None
