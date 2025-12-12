import requests #type:ignore
import io
API_URL = "https://filter-api-ywwi.onrender.com/v1/filter" 

def check_image_with_ai(image_bytes, source_name="crawler_bot_1"):
    """
    Hàm gửi ảnh sang API để check xem có nên giữ lại không.
    Trả về: True (Giữ) / False (Bỏ)
    """
    try:
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        data = {'source': source_name}
        
        # Gọi API
        response = requests.post(API_URL, files=files, data=data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            
            # Kiểm tra quyết định của AI
            if result['action'] == 'KEEP':
                print(f"AI duyệt: {result['detected_labels']}")
                return True, result['detected_labels'] # Giữ lại
            else:
                print(f"AI loại bỏ: Ảnh rác hoặc không đúng chủ đề.")
                return False, []
        else:
            print(f"Lỗi API server: {response.status_code}")
            return True, [] # Mặc định giữ lại nếu API lỗi (để về lọc tay sau)
            
    except Exception as e:
        print(f"Không kết nối được AI API: {e}")
        return True, [] # Mặc định giữ lại nếu mất mạng