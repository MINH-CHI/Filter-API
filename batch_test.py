import os
import io
import requests # type:ignore
import pandas as pd # type:ignore
import time
from datetime import datetime
from tqdm import tqdm # type:ignore
from pymongo import MongoClient # type:ignore
from google.auth.transport.requests import Request # type:ignore
from google.oauth2.credentials import Credentials # type:ignore
from google_auth_oauthlib.flow import InstalledAppFlow # type:ignore
from googleapiclient.discovery import build # type:ignore
from googleapiclient.http import MediaIoBaseDownload # type:ignore

API_URL = "https://types-prep-visible-hat.trycloudflare.com/v1/filter"
API_KEY = "Data_team_kOH17bVPOEf7kPd6y0YNICNSnZyT5neg"
DRIVE_BASE_FOLDER_NAME = "DATA"
DRIVE_SUB_FOLDER_NAME = "object_detection"
DRIVE_VPP_FOLDER_NAME = "classes-do-gia-dung"
OUTPUT_FILE = "drive_test_results.xlsx"
TOKEN_FILE = 'token.json' 
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "api_request_log" 
COLLECTION_NAME = "api_unlabeled_images"

def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]
def get_drive_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, ["https://www.googleapis.com/auth/drive"])
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secrets.json', ["https://www.googleapis.com/auth/drive"])
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

def run_test():
    print("🚀 Bắt đầu Script Batch Test...")
    
    # 1. Kết nối Mongo & Xóa dữ liệu cũ (để Dashboard sạch sẽ)
    collection = get_mongo_collection()
    # print("🧹 Đang dọn dẹp dữ liệu test cũ trên MongoDB...")
    # collection.delete_many({}) # Xóa sạch collection test cũ
    
    # 2Kết nối Drive
    service = get_drive_service()
    if not service: 
        return

    # Quét task
    print("🔄 Đang quét ảnh từ Drive...")
    tasks = build_task_list(service)
    print(f"📋 Tìm thấy {len(tasks)} ảnh.")

    # 4. Chạy vòng lặp xử lý
    for i, task in enumerate(tqdm(tasks, desc="Processing")):
        filename = task['filename']
        actual = task['actual_label']
        
        # Tải ảnh
        img_bytes = download_file_bytes(service, task['file_id'])
        
        result_record = {
            "timestamp": datetime.now(),
            "filename": filename,
            "actual_label": actual,
            "type": task['category_type'],
            "status": "Processing..."
        }

        if img_bytes:
            try:
                files = {"file": (filename, img_bytes, 'image/jpeg')}
                data = {"source": "batch_script_runner"}
                headers = {"x-api-key": API_KEY}
                
                # Gọi API
                resp = requests.post(API_URL, files=files, data=data, headers=headers)
                
                if resp.status_code == 200:
                    res_json = resp.json()
                    detections = res_json.get("detections", [])
                    
                    if detections:
                        best = sorted(detections, key=lambda x: x['confidence'], reverse=True)[0]
                        pred = best['object']
                        conf = best['confidence']
                    else:
                        pred = "None"
                        conf = 0.0
                    
                    # Logic Check
                    is_correct = False
                    if task['category_type'] == "unknown":
                        is_correct = (pred == "None")
                    else:
                        is_correct = str(actual).lower() in str(pred).lower()

                    # Cập nhật record
                    result_record.update({
                        "predicted_label": pred,
                        "confidence": conf,
                        "is_correct": is_correct,
                        "status": "Done"
                    })
                else:
                    result_record["status"] = f"API Error {resp.status_code}"
            except Exception as e:
                result_record["status"] = f"Error: {str(e)}"
        else:
            result_record["status"] = "Download Failed"

        # QUAN TRỌNG: Ghi ngay vào MongoDB
        collection.insert_one(result_record)
        
        # Sleep nhẹ
        time.sleep(0.5)

    print("✅ Đã hoàn thành test.")
if __name__ == "__main__":
    run_test()