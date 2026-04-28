import numpy as np

def calculate_dist(p1, p2):
    """歐幾里得距離公式: sqrt((x2-x1)^2 + (y2-y1)^2)"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def verify_measurement(pixel_dist, ground_truth=None):
    # 驗證邏輯：若有標註數據則計算 L2 Error
    pass
