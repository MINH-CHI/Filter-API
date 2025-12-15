import os
import io
import requests # type:ignore
import pandas as pd # type:ignore
import time
from tqdm import tqdm # type:ignore
import concurrent.futures
from google.auth.transport.requests import Request # type:ignore
from google.oauth2.credentials import Credentials # type:ignore
from google_auth_oauthlib.flow import InstalledAppFlow # type:ignore
from googleapiclient.discovery import build # type:ignore
from googleapiclient.http import MediaIoBaseDownload # type:ignore

API_URL = "https://cave-reconstruction-invention-somewhat.trycloudflare.com/v1/filter"
API_KEY = "Data_team_kOH17bVPOEf7kPd6y0YNICNSnZyT5neg"
DATASET_FOLDER_ID = "1PlH4I4MMHal4oMFf6aqFnUC8-sOwO60A" 
DRIVE_BASE_FOLDER_NAME = "DATA"
DRIVE_SUB_FOLDER_NAME = "object_detection"
DRIVE_VPP_FOLDER_NAME = "classes-do-gia-dung"
OUTPUT_FILE = "drive_test_results.xlsx"
TOKEN_FILE = 'token.json' 

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

def build_task_list(service, root_id):
    tasks = []
    print("🔄 Đang định vị thư mục mục tiêu...")

    data_id = find_folder_id_by_name(service, DRIVE_BASE_FOLDER_NAME, root_id)
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

def process_single_task(service, task):
    """Download ảnh từ Drive -> Gửi API -> Trả kết quả"""
    file_id = task['file_id']
    filename = task['filename']
    
    # Download ảnh từ Drive
    image_bytes = download_file_bytes(service, file_id)
    
    if not image_bytes:
        return {**task, "error": "Download Failed"}

    # Gửi API
    try:
        # Request lib cần tuple (filename, bytes, content_type) để upload từ memory
        files = {"file": (filename, image_bytes, 'image/jpeg')} 
        data = {"source": "drive_batch_test"}
        headers = {"x-api-key": API_KEY}
        
        response = requests.post(API_URL, files=files, data=data, headers=headers)
        
        if response.status_code == 200:
            res = response.json()
            detections = res.get("detections", [])
            
            if detections:
                best_det = sorted(detections, key=lambda x: x['confidence'], reverse=True)[0]
                pred_label = best_det['object']
                conf = best_det['confidence']
            else:
                pred_label = "None"
                conf = 0.0
            
            is_correct = False
            if task['category_type'] == "unknown":
                is_correct = (pred_label == "None")
            else:
                # So sánh tương đối (vd: 'smartphone' in 'black smartphone')
                is_correct = str(task['actual_label']).lower() in str(pred_label).lower()

            return {
                "filename": filename,
                "type": task['category_type'],
                "actual_label": task['actual_label'],
                "predicted_label": pred_label,
                "confidence": conf,
                "action": res.get("action"),
                "is_correct": is_correct,
                "file_id": file_id
            }
        else:
            return {**task, "error": f"API {response.status_code}"}
            
    except Exception as e:
        return {**task, "error": str(e)}

def run_test():
    # Khởi tạo Drive Service
    service = get_drive_service()
    if not service:
        print("❌ Không thể kết nối Google Drive")
        return

    # Quét toàn bộ file cần test
    tasks = build_task_list(service, DATASET_FOLDER_ID)
    print(f"🚀 Tìm thấy tổng cộng {len(tasks)} ảnh. Bắt đầu test tuần tự...")

    results = []
    
    # 2. Chạy Tuần tự
    for i, task in enumerate(tqdm(tasks)):
        try:
            # Gọi hàm xử lý trực tiếp
            res = process_single_task(service, task)
            results.append(res)
            time.sleep(3) 
            
        except KeyboardInterrupt:
            print("\n🛑 Người dùng dừng chương trình. Đang lưu kết quả tạm thời...")
            break
        except Exception as e:
            print(f"\n⚠️ Lỗi bất ngờ tại file {task['filename']}: {e}")
            # Vẫn lưu lại lỗi để biết file nào hỏng
            results.append({**task, "error": str(e)})

    # Xuất Excel
    # if results:
    #     df = pd.DataFrame(results)
    #     # Sắp xếp cho đẹp
    #     if 'type' in df.columns and 'actual_label' in df.columns:
    #         df = df.sort_values(by=['type', 'actual_label'])
            
    #     df.to_excel(OUTPUT_FILE, index=False)
        
    #     # Thống kê nhanh
    #     if 'is_correct' in df.columns:
    #         # Lọc bỏ các dòng lỗi trước khi tính toán
    #         valid_results = df[df['is_correct'].notnull()] 
    #         if not valid_results.empty:
    #             acc = valid_results['is_correct'].mean() * 100
    #             print(f"\n📊 Accuracy sơ bộ: {acc:.2f}% (trên {len(valid_results)} ảnh thành công)")
            
    #     print(f"✅ Đã lưu kết quả tại: {OUTPUT_FILE}")
    # else:
    #     print("⚠️ Không có kết quả nào được xử lý.")

if __name__ == "__main__":
    run_test()