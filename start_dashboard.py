import subprocess
import re
import sys
import time
import pymongo
from datetime import datetime
import os
from dotenv import load_dotenv

MONGO_URI = os.getenv("MONGO_URI") 
DB_NAME = "api_request_log"
CONFIG_COLLECTION = "system_config"
def get_cloudflare_url():
    print("🚀 Đang khởi động Cloudflare Tunnel...")
    
    # Chạy lệnh cloudflared dưới nền (Subprocess)
    cmd = ["cloudflared.exe", "tunnel", "--url", "http://127.0.0.1:8000"]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True, # Đọc output dưới dạng text
        bufsize=1  # Đọc từng dòng (Line buffered)
    )

    url = None
    
    # Đọc từng dòng log của Cloudflare để tìm URL
    try:
        while True:
            # Cloudflare in link ra stderr
            line = process.stderr.readline()
            if not line:
                break
                
            # Tìm dòng chứa .trycloudflare.com
            if ".trycloudflare.com" in line:
                # Dùng Regex bắt cái link
                match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                if match:
                    url = match.group(0)
                    print(f"✅ Đã bắt được URL mới: {url}")
                    break
    except KeyboardInterrupt:
        process.terminate()
        return None

    return url, process

def save_url_to_mongo(url):
    """Lưu URL vào MongoDB để Streamlit Cloud đọc được"""
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        coll = db[CONFIG_COLLECTION]
        
        # Upsert: Nếu chưa có thì tạo mới, có rồi thì cập nhật
        coll.update_one(
            {"config_key": "active_api_url"}, # Điều kiện tìm
            {
                "$set": {
                    "value": url,
                    "updated_at": datetime.now(),
                    "updated_by": "start_app_script"
                }
            },
            upsert=True
        )
        print(f"☁️ Đã đẩy URL lên MongoDB thành công!")
    except Exception as e:
        print(f"❌ Lỗi không lưu được vào Mongo: {e}")

if __name__ == "__main__":
    if not MONGO_URI:
        print("❌ Lỗi: Chưa cấu hình MONGO_URI trong file .env local!")
        sys.exit(1)

    result = get_cloudflare_url()
    if not result:
        sys.exit(1)
        
    url, cf_process = result
    
    # 1. Ghi lên Cloud Database
    save_url_to_mongo(url)
    
    print("\n--- 🌐 HỆ THỐNG ĐÃ ONLINE ---")
    print(f"API URL: {url}")
    print("Streamlit Cloud bây giờ có thể tự động nhận diện URL này.")
    print("⏳ Đang giữ kết nối Cloudflare... (Không được tắt cửa sổ này)")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cf_process.terminate()
        print("Đã tắt.")