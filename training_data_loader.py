import os
import zipfile

# 下載數據 (假設已上傳 kaggle.json)
def download_training_data_json():
  if not os.path.exists('./animal-pose-dataset.zip'):
    os.system('kaggle datasets download -d bloodaxe/animal-pose-dataset')

  # 解壓縮
  with zipfile.ZipFile('./animal-pose-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('./animal_pose_data')
