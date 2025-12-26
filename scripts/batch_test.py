import os
import sys
import io
import requests # type:ignore
import pandas as pd # type:ignore
import time
import random
from datetime import datetime
from tqdm import tqdm # type:ignore
from pymongo import MongoClient # type:ignore
from dotenv import load_dotenv # type:ignore
from google.auth.transport.requests import Request # type:ignore
from google.oauth2.credentials import Credentials # type:ignore
from google_auth_oauthlib.flow import InstalledAppFlow # type:ignore
from googleapiclient.discovery import build # type:ignore
from googleapiclient.http import MediaIoBaseDownload # type:ignore
import concurrent.futures
from threading import Lock
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
CREDENTIALS_DIR = os.path.join(project_root, "credentials")
TOKEN_FILE = os.path.join(CREDENTIALS_DIR, 'token.json')
CLIENT_SECRETS_FILE = os.path.join(CREDENTIALS_DIR, 'client_secrets.json')
ENV_PATH = os.path.join(project_root, ".env")
load_dotenv(ENV_PATH)
# API_URL = "https://wrap-caroline-neutral-goat.trycloudflare.com/v1/filter"
API_KEY = os.getenv("API_KEY")
DRIVE_BASE_FOLDER_NAME = "DATA"
DRIVE_SUB_FOLDER_NAME = "object_detection"
DRIVE_VPP_FOLDER_NAME = "classes-do-gia-dung"
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "api_request_log" 
COLLECTION_NAME = "api_unlabeled_images"
CONFIG_COLLECTION = "system_config"

drive_lock = Lock()
print_lock = Lock()
# 1. Nhóm Model Đã Học Tốt (STRONG)
# Gồm các class có > 170 mẫu
STRONG_CLASSES = [
    "bed", 
    "table",       # Lưu ý: Model học "dining table", hy vọng nó nhận ra "table" chung chung
    "cabinet", 
    "dishwasher",
    "scissors"
]

# 2. Nhóm Model Học Ít/Yếu (WEAK)
# Gồm các class < 150 mẫu (Dễ bị nhận diện sai hoặc conf thấp)
WEAK_CLASSES = [
    "shelf", 
    "sofa",        # Chỉ có 82 mẫu -> Khả năng fail cao
    "toaster"      # Chỉ có 52 mẫu -> Rất yếu
]

# Tỷ lệ lấy mẫu (Bạn có thể chỉnh lại tùy số lượng ảnh thực tế trong folder)
RATIOS = {
    "STRONG": 0.6,  # Lấy 50% từ nhóm Giường, Bàn, Tủ...
    "WEAK": 0.4    # Lấy 30% từ nhóm Sofa, Toaster...
    # "UNKNOWN": 0.2  # Lấy 20% từ nhóm Quạt, Chổi (để test lọc nhiễu)
}
def get_mongo_client():
    return MongoClient(MONGO_URI)
def get_active_api_url():
    """
    Hàm mới: Tự động lấy URL Cloudflare mới nhất từ MongoDB.
    Giúp bạn không phải copy-paste link thủ công mỗi lần chạy lại Tunnel.
    """
    try:
        client = get_mongo_client()
        db = client[DB_NAME]
        config = db[CONFIG_COLLECTION].find_one({"config_key": "active_api_url"})
        
        if config and "value" in config:
            url = config["value"]
            print(f"🔗 Đã lấy API URL từ Mongo: {url}")
            # Đảm bảo URL kết thúc bằng /v1/filter
            return f"{url}/v1/filter"
        else:
            print("⚠️ Không tìm thấy URL trong Mongo. Dùng URL mặc định.")
            return "http://127.0.0.1:8000/v1/filter" # Fallback về Localhost
            
    except Exception as e:
        print(f"⚠️ Lỗi lấy URL từ Mongo: {e}. Dùng Localhost.")
        return "http://127.0.0.1:8000/v1/filter"
def get_drive_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, ["https://www.googleapis.com/auth/drive"])
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, ["https://www.googleapis.com/auth/drive"])
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def list_all_files_in_folder(service, folder_id):
    all_files = []
    page_token = None
    while True:
        try:
            response = service.files().list(q=f"'{folder_id}' in parents and trashed=false",
                fields='nextPageToken, files(id, name)', pageSize=1000, pageToken=page_token).execute()
            all_files.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break
        except Exception as e:
            print(f"  [LỖI] Lỗi khi liệt kê file (pagination): {e}")
            break
    print(f"  Đã tìm thấy tổng cộng {len(all_files)} files (ảnh + metadata).")
    return all_files

def find_folder_id_by_name(service, folder_name, parent_id):
    """Tìm ID của folder con dựa vào tên và ID cha"""
    try:
        query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        if not files:
            print(f"❌ [LỖI] Không tìm thấy folder '{folder_name}' trong parent '{parent_id}'")
            return None
        return files[0]['id']
    except Exception as e:
        print(f"❌ [LỖI API] Khi tìm folder {folder_name}: {e}")
        return None
def download_file_bytes(service, file_id):
    """Tải file về RAM dưới dạng bytes"""
    try:
        with drive_lock:
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            fh.seek(0)
            return fh.getvalue() # Trả về bytes
    except Exception:
        return None

def build_task_list(service):
    tasks = []
    print("🔄 Đang định vị thư mục mục tiêu...")

    data_id = find_folder_id_by_name(service, DRIVE_BASE_FOLDER_NAME, "1PlH4I4MMHal4oMFf6aqFnUC8-sOwO60A")
    if not data_id: 
        return []

    obj_det_id = find_folder_id_by_name(service, DRIVE_SUB_FOLDER_NAME, data_id)
    if not obj_det_id: 
        return []

    target_root_id = find_folder_id_by_name(service, DRIVE_VPP_FOLDER_NAME, obj_det_id)
    if not target_root_id: 
        return []

    print(f"✅ Đã vào tới folder đích: {DRIVE_VPP_FOLDER_NAME} (ID: {target_root_id})")
    print("🔄 Đang quét các class con...")

    class_folders = []
    page_token = None
    while True:
        res = service.files().list(
            q=f"'{target_root_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()
        class_folders.extend(res.get('files', []))
        page_token = res.get('nextPageToken')
        if not page_token: break

    print(f"📂 Tìm thấy {len(class_folders)} class (nhãn). Đang quét ảnh...")

    for folder in class_folders:
        label_name = folder['name'] # Tên folder chính là nhãn thực tế (Actual Label)
        folder_id = folder['id']
        
        # Lấy danh sách file ảnh trong folder class này
        all_files = list_all_files_in_folder(service, folder_id)
        
        count = 0
        for f in all_files:
            if f['name'].lower().endswith(('.png', '.jpg', '.jpeg')):
                tasks.append({
                    "file_id": f['id'],
                    "filename": f['name'],
                    "actual_label": label_name,    
                    "category_type": DRIVE_VPP_FOLDER_NAME
                })
                count += 1

    return tasks
def get_processed_filenames(collection):
    """Lấy danh sách các filename đã có status='Done' trong DB"""
    print("🔍 Đang kiểm tra lịch sử trong MongoDB...")
    query = {
        "source": "batch_test",
        "status": "Done"
    }
    # Chỉ lấy trường filename để tiết kiệm RAM
    records = collection.find(query, {"filename": 1})
    processed_set = set(doc['filename'] for doc in records)
    print(f"📚 Tìm thấy {len(processed_set)} ảnh đã xử lý xong trước đó.")
    return processed_set
def process_single_task(task, api_url, service):
    """Hàm này chạy song song trên mỗi luồng"""
    filename = task['filename']
    actual = task['actual_label']
    
    # Tạo Client Mongo riêng cho luồng này (Thread-safe)
    local_client = MongoClient(MONGO_URI)
    local_collection = local_client[DB_NAME][COLLECTION_NAME]

    result_record = {
        "timestamp": datetime.now(),
        "filename": filename,
        "actual_label": actual,
        "type": task['category_type'],
        "group_type": task.get('group_type', 'FULL_DATASET'),
        # "group_type": "FULL_DATASET",
        "status": "Processing",
        "source": "batch_client_result"
    }

    try:
        # Tải ảnh
        img_bytes = download_file_bytes(service, task['file_id'])
        
        if img_bytes:
            # Gọi API (Phần này chạy song song, không cần Lock)
            files = {"file": (filename, img_bytes, 'image/jpeg')}
            data = {"source": "batch_test"}
            headers = {"x-api-key": API_KEY}
            
            resp = requests.post(api_url, files=files, data=data, headers=headers, timeout=60)
            
            if resp.status_code == 200:
                res_json = resp.json()
                detected_labels = res_json.get("detected_labels", [])
                action = res_json.get("action", "UNKNOWN")
                detections = res_json.get("detections", [])
                
                bboxes = []
                all_confs = []
                target_confs = []
                actual_norm = str(actual).lower().strip()
                if detections:
                    for d in detections:
                        conf = d.get('confidence', 0)
                        label = str(d.get('object', '')).lower()
                        
                        # Lưu box
                        if 'box' in d: 
                            bboxes.append(str(d['box']))
                        
                        all_confs.append(conf)

                        # Kiểm tra xem object này có khớp với Actual Label không?
                        # Ví dụ: actual="table" khớp với label="dining table"
                        if actual_norm in label:
                            target_confs.append(conf)
                if target_confs:
                    # Model CÓ nhìn thấy vật thể đúng
                    # Lấy max của đúng vật thể đó (VD: Lấy 0.15 của Table, bỏ qua 0.95 của Person)
                    final_conf = max(target_confs)
                elif all_confs:
                    # Model KHÔNG thấy vật thể đúng
                    # Lấy max của vật thể gây nhiễu nhất (để biết model đang nhìn nhầm ra cái gì mạnh nhất)
                    final_conf = max(all_confs) 
                else:
                    final_conf = 0.0
        
                bbox_str = " | ".join(bboxes) if bboxes else ""
                pred_str = ", ".join(detected_labels) if detected_labels else "None"
                
                is_correct = False
                if task['category_type'].lower() == "unknown":
                    is_correct = (not detected_labels) or (action == "UNPROCESSED")
                else:
                    actual_norm = str(actual).lower().strip()
                    for lbl in detected_labels:
                        if actual_norm in str(lbl).lower():
                            is_correct = True
                            break
                
                result_record.update({
                    "predicted_label": pred_str,
                    "confidence": final_conf,
                    "bounding_box": bbox_str,
                    "action": action,
                    "is_correct": is_correct,
                    "detected_labels": detected_labels,
                    "status": "Done"
                })
            else:
                result_record["status"] = f"API Error {resp.status_code}"
        else:
            result_record["status"] = "Download Failed"

    except Exception as e:
        result_record["status"] = f"Code Error: {str(e)}"
    
    finally:
        # Ghi vào DB và đóng kết nối
        local_collection.insert_one(result_record)
        local_client.close()
def filter_and_sample_tasks(all_tasks, processed_files):
    print("\n⚖️  Đang lấy mẫu dữ liệu...")
    
    # Loại bỏ các file đã chạy rồi
    pending_tasks = [t for t in all_tasks if t['filename'] not in processed_files]
    
    class_buckets = {}
    for task in pending_tasks:
        label = task['actual_label']
        if label not in class_buckets:
            class_buckets[label] = []
        class_buckets[label].append(task)
    
    available_classes = list(class_buckets.keys())
    num_classes = len(available_classes)
    
    if num_classes == 0:
        print("⚠️ Không tìm thấy class nào hoặc tất cả ảnh đã được xử lý!")
        return []

    # Tính toán số lượng cần lấy cho mỗi class
    # Ví dụ: 1000 / 8 = 125. Dư 0.
    quota_per_class = 1000 // num_classes
    remainder = 1000 % num_classes # Số dư (để xử lý nếu chia không hết)

    print(f"📊 Tìm thấy {num_classes} classes. Mục tiêu tổng: {1000} ảnh.")
    print(f"👉 Trung bình mỗi class sẽ lấy khoảng: {quota_per_class} ảnh.")

    final_tasks = []
    
    # Lấy mẫu ngẫu nhiên
    for i, (label, tasks) in enumerate(class_buckets.items()):
        total_in_class = len(tasks)
        
        # Tính số lượng cần lấy cho class này
        # Cộng thêm 1 vào các class đầu tiên nếu phép chia có dư
        n_take = quota_per_class + (1 if i < remainder else 0)
        
        # Đảm bảo không lấy quá số lượng hiện có 
        n_take = min(n_take, total_in_class)
        
        if n_take > 0:
            # Random sample
            selected = random.sample(tasks, n_take)
            
            # Gán nhãn nhóm để tiện theo dõi sau này (Optional)
            for t in selected:
                t['group_type'] = "EVEN_TEST_1K" 
            
            final_tasks.extend(selected)
            print(f"  ✅ Class '{label}': Đã chọn {len(selected)}/{total_in_class} ảnh")
        else:
            print(f"  ⚠️ Class '{label}': Không còn ảnh nào chưa xử lý.")

    # Xáo trộn lần cuối để khi chạy đa luồng các class được xử lý xen kẽ
    random.shuffle(final_tasks)
    
    print(f"🚀 TỔNG CỘNG: Đã chọn được {len(final_tasks)} ảnh để chạy test.")
    return final_tasks
def run_test():
    print("🚀 Bắt đầu Test (Multi-thread)...")
    
    api_url = get_active_api_url()
    
    # Init Service (1 lần duy nhất)
    service = get_drive_service()
    if not service:
        print("❌ Không kết nối được Google Drive")
        return

    # Lấy danh sách task và lọc trùng
    tasks = build_task_list(service)
    
    client = get_mongo_client()
    processed_files = get_processed_filenames(client[DB_NAME][COLLECTION_NAME])
    client.close()
    
    # tasks_to_run = [t for t in tasks if t['filename'] not in processed_files]
    tasks_to_run = filter_and_sample_tasks(tasks, processed_files)
    total_tasks = len(tasks_to_run)
    print(f"📋 Tổng số ảnh cần test: {total_tasks}")
    
    if total_tasks == 0:
        print("✅ Đã xử lý hết. Không còn gì để chạy.")
        return

    # Max Workers = 5 để không bị Google chặn rate limit
    MAX_WORKERS = 5 
    print(f"⚡ Đang chạy với {MAX_WORKERS} luồng song song...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tất cả công việc vào Pool
        futures = []
        for task in tasks_to_run:
            futures.append(executor.submit(process_single_task, task, api_url, service))
        
        # Dùng tqdm để hiện thanh loading
        for _ in tqdm(concurrent.futures.as_completed(futures), total=total_tasks, desc="Processing Images"):
            pass

    print("\n✅ Đã hoàn thành test đa luồng.")
if __name__ == "__main__":
    run_test()