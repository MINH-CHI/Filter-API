import os
import uvicorn #type:ignore
from fastapi import FastAPI, File, UploadFile, HTTPException, Form #type:ignore 
from pydantic import BaseModel #type:ignore
from typing import List, Optional
import json
import time
from filter import ImageFilter

MODEL_PATH = os.getenv("MODEL_PATH")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "api_request_log" 
COLLECTION_NAME = "api_unlabeled_images"
TARGET_CLASSES = ["smartphone", "pen", "note paper","t-shirt","smartwatch","glasses","bracelet","dishwasher","cabinet","sofa","box cutter","shoes","table","scissor","paper"]

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
                device=0
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