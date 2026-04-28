# 1. 使用 Ultralytics 官方提供的底層鏡像 (已包含 Python, PyTorch, CUDA 等環境)
FROM ultralytics/ultralytics:latest

# 2. 設定容器內的工作目錄
WORKDIR /app

# 3. 先複製 requirements.txt 並安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 複製專案其餘檔案進入容器
COPY . .

# 5. 建立結果輸出的資料夾 
RUN mkdir -p results raw_images

# 6. 設定容器啟動時預設執行的指令
CMD ["python", "main.py"]