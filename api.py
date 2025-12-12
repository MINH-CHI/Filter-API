import os
import uvicorn #type:ignore
from fastapi import FastAPI, File, UploadFile, HTTPException, Form #type:ignore 
from pydantic import BaseModel #type:ignore
from typing import List, Optional
import json
import time
from filter import ImageFilter

MODEL_PATH = os.getenv("MODEL_PATH", "finetuned_nc126_best_mAP.onnx")
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
app = FastAPI(title="Image Filter API",description="API đánh nhãn và lọc ảnh tự động sử dụng YOLO.",version="1.0.0")

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
    source: Optional[str] = Form("unknown")
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

    # Tạo metadata để log
    metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
        "api_source": source
    }

    # Gọi Tool Filter
    is_valid, labels = filter_tool.process(image_bytes, metadata=metadata)

    # Trả kết quả JSON
    return {
        "filename": file.filename,
        "is_valid": is_valid,
        "detected_labels": labels,
        "action": "KEEP" if is_valid else "DISCARD"
    }

if __name__ == "__main__":
    # Chạy server ở port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)