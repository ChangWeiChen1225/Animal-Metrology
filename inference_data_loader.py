import os
import requests
import zipfile
import shutil
from pycocotools.coco import COCO

def download_coco_json():
    # 下載數據 (假設已上傳 kaggle.json)
    if not os.path.exists('./coco-val.zip'):
      os.system('kaggle datasets download -d tructrungnguyen/coco-val')

    # 解壓縮
    with zipfile.ZipFile('./coco-val.zip', 'r') as zip_ref:
      zip_ref.extractall('.')

def get_target_images(target_cat='cat', min_count=2):
    download_coco_json()
    coco = COCO("./data/annotations/instances_val2017.json")

    # 取得目標類別與「人」的類別 ID
    target_cat_id = coco.getCatIds(catNms=[target_cat])
    person_cat_id = coco.getCatIds(catNms=['person']) # 取得人的 ID
    
    # 先取得所有包含目標類別的圖片 ID
    imgIds = coco.getImgIds(catIds=target_cat_id)

    valid_urls = []
    print(f"正在從 {len(imgIds)} 張圖片中篩選不包含『人』且有至少 {min_count} 隻『{target_cat}』的圖片...")

    for imgId in imgIds:
        # 檢查這張圖裡面的所有標註
        all_annIds = coco.getAnnIds(imgIds=imgId)
        all_anns = coco.loadAnns(all_annIds)
        
        # 取得這張圖中所有出現過的類別 ID
        present_cat_ids = set([ann['category_id'] for ann in all_anns])
        
        # 檢查是否包含人？ (如果包含就跳過)
        if person_cat_id[0] in present_cat_ids:
            continue
            
        # 檢查目標類別（貓）的數量是否達標？
        target_annIds = coco.getAnnIds(imgIds=imgId, catIds=target_cat_id)
        if len(target_annIds) >= min_count:
            img_info = coco.loadImgs(imgId)[0]
            valid_urls.append(img_info['coco_url'])
            
            # 測試用：10 張
            if len(valid_urls) >= 10: 
                break 

    print(f"篩選完成，共找到 {len(valid_urls)} 張符合條件的圖片。")
    return valid_urls
