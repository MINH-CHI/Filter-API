import os
import io
import requests
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# --- CẤU HÌNH ---
API_URL = "https://translation-published-visiting-nearest.trycloudflare.com/v1/filter"
API_KEY = "Data_team_kOH17bVPOEf7kPd6y0YNICNSnZyT5neg"
DATASET_FOLDER_ID = "1PlH4I4MMHal4oMFf6aqFnUC8-sOwO60A" # <--- ID folder gốc trên Drive
DRIVE_BASE_FOLDER_NAME = "DATA"
DRIVE_SUB_FOLDER_NAME = "object_detection"
DRIVE_VPP_FOLDER_NAME = "classes-do-gia-dung"
OUTPUT_FILE = "drive_test_results.xlsx"
TOKEN_FILE = 'token.json' # File lưu token đăng nhập Drive

# --- 1. CÁC HÀM HELPER GOOGLE DRIVE ---
def get_drive_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, ["https://www.googleapis.com/auth/drive"])
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', ["https://www.googleapis.com/auth/drive"])
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

# --- 2. HÀM CRAWL CẤU TRÚC FOLDER ---
def build_task_list(service, root_id):
    tasks = []
    print("🔄 Đang định vị thư mục mục tiêu...")

    # BƯỚC 1: ĐI THEO ĐƯỜNG DẪN CỤ THỂ
    # Level 1: DATA
    data_id = find_folder_id_by_name(service, DRIVE_BASE_FOLDER_NAME, root_id)
    if not data_id: return []

    # Level 2: object_detection
    obj_det_id = find_folder_id_by_name(service, DRIVE_SUB_FOLDER_NAME, data_id)
    if not obj_det_id: return []

    # Level 3: classes-do-gia-dung (Đây là folder chứa các class con)
    target_root_id = find_folder_id_by_name(service, DRIVE_VPP_FOLDER_NAME, obj_det_id)
    if not target_root_id: return []

    print(f"✅ Đã vào tới folder đích: {DRIVE_VPP_FOLDER_NAME} (ID: {target_root_id})")
    print("🔄 Đang quét các class con...")

    # BƯỚC 2: LIỆT KÊ CÁC FOLDER CLASS (Vd: noi-com, quat, bep-ga...)
    # Lấy tất cả folder nằm trong 'classes-do-gia-dung'
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

    # BƯỚC 3: DUYỆT TỪNG CLASS ĐỂ LẤY ẢNH
    for folder in class_folders:
        label_name = folder['name'] # Tên folder chính là nhãn thực tế (Actual Label)
        folder_id = folder['id']
        
        # Lấy danh sách file ảnh trong folder class này
        # (Sử dụng lại hàm list_all_files_in_folder bạn đã viết, nhưng nhớ filter ảnh)
        all_files = list_all_files_in_folder(service, folder_id) # Hàm này của bạn ở trên
        
        count = 0
        for f in all_files:
            if f['name'].lower().endswith(('.png', '.jpg', '.jpeg')):
                tasks.append({
                    "file_id": f['id'],
                    "filename": f['name'],
                    "actual_label": label_name,    # Vd: noi-com
                    "category_type": DRIVE_VPP_FOLDER_NAME # Vd: classes-do-gia-dung
                })
                count += 1
        # print(f"  -> Class '{label_name}': {count} ảnh")

    return tasks

# --- 3. HÀM TEST (WORKER) ---
def process_single_task(service, task):
    """Download ảnh từ Drive -> Gửi API -> Trả kết quả"""
    file_id = task['file_id']
    filename = task['filename']
    
    # 1. Download ảnh từ Drive
    image_bytes = download_file_bytes(service, file_id)
    
    if not image_bytes:
        return {**task, "error": "Download Failed"}

    # 2. Gửi API
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
                "file_id": file_id # Giữ lại ID để dễ truy vết nếu cần
            }
        else:
            return {**task, "error": f"API {response.status_code}"}
            
    except Exception as e:
        return {**task, "error": str(e)}

# --- 4. MAIN ---
def run_test():
    # Khởi tạo Drive Service (Làm 1 lần ở main thread)
    # Lưu ý: Google API Client object không thread-safe hoàn toàn, 
    # nhưng build object nhẹ nên có thể pass hoặc init mới trong thread. 
    # Tốt nhất pass service object vào nhưng cẩn thận. 
    # Ở đây để đơn giản ta sẽ dùng chung service object (thường ổn với read-only).
    service = get_drive_service()
    if not service:
        print("❌ Không thể kết nối Google Drive")
        return

    # 1. Quét toàn bộ file cần test
    tasks = build_task_list(service, DATASET_FOLDER_ID)
    print(f"🚀 Tìm thấy tổng cộng {len(tasks)} ảnh. Bắt đầu test...")

    results = []
    
    # 2. Chạy Multi-thread
    # Lưu ý: Giảm max_workers xuống khoảng 4-5 để tránh bị Google chặn (Rate Limit)
    # Vì mỗi worker sẽ gọi download API.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit task
        future_to_task = {
            executor.submit(process_single_task, service, task): task 
            for task in tasks
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks)):
            res = future.result()
            results.append(res)

    # 3. Xuất Excel
    if results:
        df = pd.DataFrame(results)
        # Sắp xếp cho đẹp
        if 'type' in df.columns and 'actual_label' in df.columns:
            df = df.sort_values(by=['type', 'actual_label'])
            
        df.to_excel(OUTPUT_FILE, index=False)
        
        # Thống kê nhanh
        if 'is_correct' in df.columns:
            acc = df['is_correct'].mean() * 100
            print(f"\n📊 Accuracy sơ bộ: {acc:.2f}%")
            
        print(f"✅ Đã lưu kết quả tại: {OUTPUT_FILE}")
    else:
        print("⚠️ Không có kết quả nào được xử lý.")

if __name__ == "__main__":
    run_test()