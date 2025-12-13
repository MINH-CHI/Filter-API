import os
import requests #type:ignore
import pandas as pd #type:ignore
from tqdm import tqdm  #type:ignore
import concurrent.futures

# --- CẤU HÌNH ---
API_URL = "https://translation-published-visiting-nearest.trycloudflare.com/v1/filter"
API_KEY = "Data_team_kOH17bVPOEf7kPd6y0YNICNSnZyT5neg"
DATASET_DIR = "dataset_test_1000"
OUTPUT_FILE = "test_results_1000.xlsx"

def test_single_image(file_path, actual_label, category_type):
    """Gửi 1 ảnh và lấy kết quả"""
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"source": "batch_test"}
            headers = {"x-api-key": API_KEY}
            
            response = requests.post(API_URL, files=files, data=data, headers=headers)
            
            if response.status_code == 200:
                res = response.json()
                
                # Lấy detection có confidence cao nhất
                detections = res.get("detections", [])
                if detections:
                    # Sắp xếp giảm dần theo confidence
                    best_det = sorted(detections, key=lambda x: x['confidence'], reverse=True)[0]
                    pred_label = best_det['object']
                    confidence = best_det['confidence']
                else:
                    pred_label = "None"
                    confidence = 0.0
                
                return {
                    "filename": os.path.basename(file_path),
                    "type": category_type,      # valid / imbalance / unknown
                    "actual_label": actual_label, # Nhãn thực tế (tên thư mục)
                    "predicted_label": pred_label,
                    "confidence": confidence,
                    "action": res.get("action"),
                    "is_correct": str(actual_label) in str(pred_label) if category_type != "unknown" else (pred_label == "None")
                }
            else:
                return {"filename": os.path.basename(file_path), "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"filename": os.path.basename(file_path), "error": str(e)}

def run_test():
    tasks = []
    results = []
    
    # Duyệt qua các thư mục
    # Giả sử cấu trúc: dataset/valid/smartphone/anh1.jpg
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                
                # Phân tích đường dẫn để lấy label
                parts = file_path.split(os.sep)
                # parts[-2] là tên thư mục chứa ảnh (label thực tế)
                actual_label = parts[-2] 
                # parts[-3] là loại (valid/unknown...)
                category_type = parts[-3] if len(parts) >= 3 else "unknown"
                
                tasks.append((file_path, actual_label, category_type))

    print(f"🚀 Bắt đầu test {len(tasks)} ảnh...")
    
    # Chạy đa luồng (Multi-thread) cho nhanh (10 ảnh cùng lúc)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(test_single_image, t[0], t[1], t[2]): t for t in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
            res = future.result()
            results.append(res)

    # Xuất ra Excel
    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ Đã xong! Kết quả lưu tại {OUTPUT_FILE}")

if __name__ == "__main__":
    run_test()