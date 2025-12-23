import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
env_path = os.path.join(project_root, ".env")
import uvicorn #type:ignore
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Security #type:ignore 
from pydantic import BaseModel #type:ignore
from typing import List, Optional
import json
import time
from app.core.filter import ImageFilter
from app.core.config import API_KEYS #type:ignore
from fastapi.security.api_key import APIKeyHeader  #type:ignore
from starlette.status import HTTP_403_FORBIDDEN  #type:ignore
from dotenv import load_dotenv #type:ignore
load_dotenv(env_path)

MINIO_CONFIG = {
    "endpoint": os.getenv("MINIO_ENDPOINT"), # IP và Port MinIO server
    "access_key": os.getenv("MINIO_ACCESS_KEY"), # User đăng nhập
    "secret_key": os.getenv("MINIO_SECRET_KEY"), # password
    "bucket_name": "filter-images-bucket", # Tên bucket muốn lưu
    "secure": False # Đặt True nếu link là https:// , False nếu là http://
}
MODEL_PATH = os.getenv("MODEL_PATH")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "api_request_log"
COLLECTION_NAME = "api_unlabeled_images"
TARGET_CLASSES = ["smartphone", "pen", "note paper","t-shirt","smartwatch","glasses","bracelet","dishwasher","cabinet","sofa","box cutter","shoes","table","scissor","paper"]
CLASS_MAPPING = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
    15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush',
    80: 'bracelet', 81: 'glasses', 82: 't-shirt', 83: 'sofa', 84: 'table',
    85: 'wardrobe', 86: 'cabinet', 87: 'tablet', 88: 'pen', 89: 'shoes',
    90: 'dishwasher', 91: 'cupboard', 92: 'lemon', 93: 'bread', 94: 'paper',
    95: 'sanwich', 96: 'phonebattery', 97: 'smartphone', 98: 'smartwatch', 99: 'phonecases',
    100: 'note paper', 101: 'box cutter', 102: 'butterfly clip', 103: 'paper clip', 104: 'hole puncher',
    105: 'straight ruler', 106: 'scissor', 107: 'calculator', 108: 'lap', 109: 'paper hole punch',
    110: 'paper flip', 111: 'carbon paper', 112: 'envelope', 113: 'butterfly flip', 114: 'paper cutting table',
    115: 'pencil', 116: 'pa', 117: 'paper cutting paper', 118: 'paper note', 119: 'giấy note',
    120: 'mini stapler', 121: 'calendar', 122: 'eraser', 123: 'calender', 124: 'wall calendar',
    125: 'cellq'
}
description_text = """
**API đánh nhãn và lọc ảnh tự động sử dụng YOLO.**

---
### Hướng dẫn sử dụng

Hệ thống yêu cầu **API Key** để bảo mật. Vui lòng làm theo các bước sau để test:

1. Bấm nút **Authorize** (biểu tượng ổ khóa ) ở góc phải.
2. Nhập API Key của bạn vào ô `value`.
3. Bấm **Authorize** -> **Close**.
4. Chọn endpoint `/v1/filter` bên dưới -> Bấm **Try it out** để upload ảnh.
"""
app = FastAPI(title="Image Filter API",description=description_text,version="1.0.0",contact={"name": "Minh Admin","email": "minhchitran12345678910@gmail.com",})
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

# Hàm kiểm tra Key
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header in API_KEYS:
        return API_KEYS[api_key_header] # Trả về tên người dùng
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, 
            detail="❌ Sai API Key hoặc chưa có quyền truy cập!"
        )
# Biến toàn cục để lưu instance của filter
filter_tool = None 

@app.on_event("startup") # Khi server khởi động chạy hàm này ngay lập tức
def startup_event():
    """Hàm chạy 1 lần khi server khởi động để load Model"""
    global filter_tool
    max_retries = 10  # Thử tối đa 10 lần
    for i in range(max_retries):
        try:
            print(f"🔄 Đang thử kết nối Database và Load Model (Lần {i+1}/{max_retries})...")
            filter_tool = ImageFilter(
                model_path=MODEL_PATH,
                mongo_uri=MONGO_URI,
                db_name=DB_NAME,
                collection_name=COLLECTION_NAME,
                target_classes=TARGET_CLASSES,
                minio_config=MINIO_CONFIG,
                enable_filter=True,
                device=0,
                class_mapping=CLASS_MAPPING
            )
            print("✅ KẾT NỐI THÀNH CÔNG! AI Service đã sẵn sàng.")
            break # Thoát vòng lặp nếu thành công
            
        except Exception as e:
            print(f"⚠️ Lỗi khởi tạo (Lần {i+1}): {e}")
            print("⏳ Đợi 5 giây rồi thử lại...")
            time.sleep(5) # Ngủ 5 giây chờ Mongo khởi động xong

@app.get("/") # Kích hoạt khi người dùng vào link với endpoint "/"
def health_check():
    return {"status": "ok", "service": "Image Filter API"}

@app.post("/v1/filter")
async def filter_image(
    file: UploadFile = File(...), 
    source: Optional[str] = Form("unknown"),
    user_name: str = Depends(get_api_key)
):
    """
    Endpoint nhận file ảnh (Upload) và trả về kết quả lọc.
    
    - **file**: File ảnh upload (binary)
    - **source**: Nguồn gốc ảnh (tùy chọn, ví dụ: 'team_marketing', 'crawler_bot')
    """
    if not filter_tool:
        raise HTTPException(status_code=503, detail="AI Service chưa sẵn sàng")

    # Đọc dữ liệu bytes từ file upload
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Lỗi đọc file")
    print(f"🕵️‍♂️ Request từ: {user_name} (Source: {source})")

    # Tạo metadata để log
    metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
        "api_source": source,
        "user": user_name
    }

    # Gọi Tool Filter
    is_valid, labels, details, action_result = filter_tool.process(image_bytes, metadata=metadata)

    # Trả kết quả JSON
    conf_list = [d.get('confidence', 0) for d in details] if details else []
    # action_result = "KEEP" if is_valid else "UNPROCESSED"
    return {
        "filename": file.filename,
        "is_valid": is_valid,
        "action": action_result,
        "detected_labels": labels, # List tên các vật thể
        "detections": details,     # Chứa full info: Box, Name, Confidence
        "confidence": conf_list,   
        "processed_by": "Server của Minh",
        "user": user_name
    }

if __name__ == "__main__":
    # Chạy server ở port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)