import subprocess
import re
import sys
import secrets
import string
import time
import pymongo #type: ignore
from datetime import datetime
import os
from dotenv import load_dotenv #type: ignore
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI") 
DB_NAME = "api_request_log"
CONFIG_COLLECTION = "system_config"
def ensure_api_keys_exist():
    """
    Kiểm tra file secrets_config.py. 
    Nếu chưa có -> Tạo mới.
    Nếu có rồi -> Bỏ qua (để tránh đổi key của người dùng).
    """
    file_name = "secrets_config.py"
    
    if os.path.exists(file_name):
        print(f"✅ Đã tìm thấy file '{file_name}'. Giữ nguyên Key cũ.")
        return

    print(f"⚠️ Chưa thấy file '{file_name}'. Đang tạo Key mới...")
    
    # Logic tạo key
    def generate_key(prefix="sk", length=32):
        alphabet = string.ascii_letters + string.digits
        random_string = ''.join(secrets.choice(alphabet) for _ in range(length))
        return f"{prefix}_{random_string}"

    users = [
        ("Sếp khánh", "Data_team"),
        ("Anh Khôi", "Data_team"),
        ("Vương", "AI_team"),
        ("Mạnh", "AI_team"),
        ("Minh","Data_team")
    ]

    file_content = "API_KEYS = {\n"
    print("\n--- 🔑 DANH SÁCH KEY VỪA TẠO ---")
    for name, prefix in users:
        key = generate_key(prefix=prefix)
        file_content += f'    "{key}": "{name}",\n'
        print(f"👤 {name}: {key}")
    file_content += "}\n"
    print("--------------------------------\n")

    with open(file_name, "w", encoding="utf-8") as f:
        f.write(file_content)
    
    print(f"💾 Đã lưu key vào '{file_name}'. Nhớ chạy build lại Docker nhé!")
def get_cloudflare_url():
    print("🚀 Đang khởi động Cloudflare Tunnel...")
    
    # Chạy lệnh cloudflared dưới nền (Subprocess) (Port API server)
    cmd = ["cloudflared.exe", "tunnel", "--url", "http://127.0.0.1:8501"]
    
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
    ensure_api_keys_exist()
    result = get_cloudflare_url()
    if not result:
        sys.exit(1)
        
    url, cf_process = result
    # Ghi lên Cloud Database
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