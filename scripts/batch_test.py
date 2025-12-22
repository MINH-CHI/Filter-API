import os
import sys
import io
import requests # type:ignore
import pandas as pd # type:ignore
import time
from datetime import datetime
from tqdm import tqdm # type:ignore
from pymongo import MongoClient # type:ignore
from dotenv import load_dotenv # type:ignore
from google.auth.transport.requests import Request # type:ignore
from google.oauth2.credentials import Credentials # type:ignore
from google_auth_oauthlib.flow import InstalledAppFlow # type:ignore
from googleapiclient.discovery import build # type:ignore
from googleapiclient.http import MediaIoBaseDownload # type:ignore

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
TOKEN_FILE = 'token.json'
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "api_request_log" 
COLLECTION_NAME = "api_unlabeled_images"
CONFIG_COLLECTION = "system_config"

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
        "source": "batch_script_runner",
        "status": "Done"
    }
    # Chỉ lấy trường filename để tiết kiệm RAM
    records = collection.find(query, {"filename": 1})
    processed_set = set(doc['filename'] for doc in records)
    print(f"📚 Tìm thấy {len(processed_set)} ảnh đã xử lý xong trước đó.")
    return processed_set
def run_test():
    print("🚀 Bắt đầu Script Batch Test...")
    
    api_url = get_active_api_url()
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    # Tùy chọn: Xóa dữ liệu cũ
    # collection.delete_many({"source": "batch_script_runner"})
    
    service = get_drive_service()
    if not service:
        print("Không thể kết nối được Google Drive")
        return

    tasks = build_task_list(service)
    print(f"📋 Tổng số ảnh cần test: {len(tasks)}")
    processed_files = get_processed_filenames(collection)
    tasks_to_run = []
    for t in tasks:
        if t['filename'] not in processed_files:
            tasks_to_run.append(t)

    for i, task in enumerate(tqdm(tasks_to_run, desc="Processing")):
        filename = task['filename']
        actual = task['actual_label']
        
        result_record = {
            "timestamp": datetime.now(),
            "filename": filename,
            "actual_label": actual,
            "type": task['category_type'],
            "status": "Processing",
            "source": "batch_script_runner"
        }

        img_bytes = download_file_bytes(service, task['file_id'])
        
        if img_bytes:
            try:
                files = {"file": (filename, img_bytes, 'image/jpeg')}
                data = {"source": "batch_test"}
                headers = {"x-api-key": API_KEY}
                
                # Gọi API
                resp = requests.post(api_url, files=files, data=data, headers=headers, timeout=30)
                
                if resp.status_code == 200:
                    res_json = resp.json()
                    
                    # Lấy Detected Labels & Action
                    detected_labels = res_json.get("detected_labels", [])
                    action = res_json.get("action", "UNKNOWN")
                    
                    # Xử lý Detections & Bounding Box
                    detections = res_json.get("detections", []) # Lấy list chi tiết
                    
                    final_conf = 0.0
                    bboxes = []

                    if detections:
                        # Lấy Max Confidence
                        final_conf = max([d.get('confidence', 0) for d in detections])
                        
                        # Trích xuất Bounding Boxes
                        for d in detections:
                            if 'box' in d:
                                bboxes.append(str(d['box']))
                    
                    # Tạo chuỗi Box để hiển thị trên Dashboard
                    bbox_str = " | ".join(bboxes) if bboxes else ""
                    pred_str = ", ".join(detected_labels) if detected_labels else "None"
                    
                    # Logic kiểm tra đúng sai
                    is_correct = False
                    if task['category_type'].lower() == "unknown":
                        is_correct = (not detected_labels) or (action == "DISCARD")
                    else:
                        actual_norm = str(actual).lower().strip()
                        for lbl in detected_labels:
                            if actual_norm in str(lbl).lower():
                                is_correct = True
                                break
                    
                    # Update MongoDB Record
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
                    print(f"\nAPI Error: {resp.text}")

            except Exception as e:
                result_record["status"] = f"Code Error: {str(e)}"
        else:
            result_record["status"] = "Download Failed"

        # Lưu vào Mongo
        collection.insert_one(result_record)
        time.sleep(5)

    print("✅ Đã hoàn thành test.")
if __name__ == "__main__":
    run_test()