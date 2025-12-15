import os
import io
import requests # type:ignore
import pandas as pd # type:ignore
import time
from datetime import datetime
from tqdm import tqdm # type:ignore
from pymongo import MongoClient # type:ignore
from dotenv import load_dotenv
from google.auth.transport.requests import Request # type:ignore
from google.oauth2.credentials import Credentials # type:ignore
from google_auth_oauthlib.flow import InstalledAppFlow # type:ignore
from googleapiclient.discovery import build # type:ignore
from googleapiclient.http import MediaIoBaseDownload # type:ignore
load_dotenv()
API_URL = "https://dir-doe-pursuant-price.trycloudflare.com/v1/filter"
API_KEY = os.getenv("API_KEY")
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
    
    collection = get_mongo_collection()
    # Tùy chọn: Xóa dữ liệu cũ nếu muốn chạy lại từ đầu
    # collection.delete_many({"source": "batch_script_runner"})

    service = get_drive_service()
    if not service: return

    tasks = build_task_list(service)
    print(f"📋 Tổng số ảnh cần test: {len(tasks)}")

    # Giảm thời gian sleep xuống 2s để chạy nhanh hơn (nếu API chịu tải được)
    SLEEP_TIME = 2 

    for i, task in enumerate(tqdm(tasks, desc="Đang xử lý")):
        filename = task['filename']
        actual = task['actual_label']
        
        # Chuẩn bị Record mặc định
        result_record = {
            "timestamp": datetime.now(),
            "filename": filename,
            "actual_label": actual,
            "type": task['category_type'],
            "status": "Processing",
            "source": "batch_script_runner" # Đánh dấu để dễ lọc trên Dashboard
        }

        # Tải ảnh từ Drive
        img_bytes = download_file_bytes(service, task['file_id'])
        
        if img_bytes:
            try:
                files = {"file": (filename, img_bytes, 'image/jpeg')}
                data = {"source": "batch_test"} # Gửi thêm metadata nếu cần
                headers = {"x-api-key": API_KEY}
                
                # 3. GỌI API
                resp = requests.post(API_URL, files=files, data=data, headers=headers, timeout=30)
                
                if resp.status_code == 200:
                    res_json = resp.json()
                    
                    # Lấy Labels
                    detected_labels = res_json.get("detected_labels", [])
                    if isinstance(detected_labels, str): # Phòng hờ API trả về string thay vì list
                        detected_labels = [detected_labels]
                    
                    # Lấy Confidence (Xử lý an toàn cho cả số và mảng)
                    raw_conf = res_json.get("confidence", 0.0)
                    final_conf = 0.0
                    
                    if isinstance(raw_conf, list):
                        # Nếu là list, lấy max hoặc trung bình (ở đây lấy max)
                        if len(raw_conf) > 0:
                            final_conf = max([float(c) for c in raw_conf if isinstance(c, (int, float))])
                    else:
                        try:
                            final_conf = float(raw_conf)
                        except:
                            final_conf = 0.0

                    action = res_json.get("action", "UNKNOWN")

                    # Nối mảng thành chuỗi để lưu vào DB
                    pred_str = ", ".join(detected_labels) if detected_labels else "None"
                    
                    is_correct = False
                    # Folder là "unknown" thì đúng nếu AI không detect ra gì (hoặc DISCARD)
                    if task['category_type'].lower() == "unknown":
                        is_correct = (not detected_labels) or (action == "DISCARD")
                    else:
                        #  Kiểm tra nhãn thực tế có nằm trong danh sách AI detect không
                        # Chuẩn hóa về lower case để so sánh
                        actual_norm = str(actual).lower().strip()
                        
                        # Check xem actual có xuất hiện trong bất kỳ nhãn nào detected không
                        for lbl in detected_labels:
                            if actual_norm in str(lbl).lower():
                                is_correct = True
                                break
                    
                    # Update kết quả vào Record
                    result_record.update({
                        "predicted_label": pred_str, # Lưu chuỗi các nhãn
                        "confidence": final_conf,
                        "action": action,
                        "is_correct": is_correct,
                        "detected_labels": detected_labels, # Lưu cả mảng gốc nếu cần query sau này
                        "status": "Done"
                    })
                    
                else:
                    result_record["status"] = f"API Error {resp.status_code}"
                    print(f"\nAPI Error: {resp.text}")

            except Exception as e:
                result_record["status"] = f"Code Error: {str(e)}"
        else:
            result_record["status"] = "Download Failed"

        # 5. Insert vào Mongo
        collection.insert_one(result_record)
        time.sleep(10)

    print("✅ Đã hoàn thành test.")

    print("✅ Đã hoàn thành test.")
if __name__ == "__main__":
    run_test()