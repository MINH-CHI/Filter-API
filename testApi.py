import io
import os
import cv2
import sys
import requests
import traceback

from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from PIL import Image
from dotenv import load_dotenv

class ImageProcessor():
    def __init__(self):
        pass
    
    def read_image(self, image_path):
        try:
            return Image.open(image_path)
        except Exception as e:
            print(f"[ERROR] - Lỗi khi đọc image: {e}")
            traceback.print_exc()
            
    def read_binary_image(self, response):
        try:
            return Image.open(io.BytesIO(response))
        except Exception as e:
            print(f"[ERROR] - Lỗi khi đọc ảnh nhị phân: {e}")
            
    def read_image_for_compute(self, image_path):
        try:
            return cv2.imread(image_path, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[ERROR] - Lỗi khi đọc dữ liệu bằng cv2 để tính toán: {e}")
            
    def save_image(self, image, output_path, filename=None):
        try:
            os.makedirs(output_path, exist_ok=True)
            
            if filename:
                output_path = os.path.join(output_path, filename)
            
            image.save(output_path)
            return
        except Exception as e:
            print(f"[ERROR] - Lỗi khi lưu ảnh ra thư mục {output_path}, lỗi {e}")
            traceback.print_exc()
            
    def save_array_image(self, image, output_path, filename=None):
        try:
            os.makedirs(output_path, exist_ok=True)

            if filename:
                output_path = os.path.join(output_path, filename)
            
            success = cv2.imwrite(output_path, image)
            if not success:
                print(f"[WARNING] - Không thể lưu ảnh vào {output_path}")
        except Exception as e:
            print(f"[ERROR] - Lỗi khi lưu trữ ảnh ra bằng cv2 ra thư mục {output_path}, lỗi {e}")        
        
    def get_image_from_url(self, url):
        try:
            # Tạo session có retry
            session = requests.Session()
            retry_strategy = Retry(
                total=5,                      # Thử lại tối đa 5 lần
                backoff_factor=1,             # Delay giữa các lần retry: 1s, 2s, 4s,...
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["GET"]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            # Chuẩn hóa URL
            full_url = f"https:{url}" if url.startswith("//") else url
            print(f"[INFO] - Nhận phản hồi từ url: {full_url}")

            # Gửi request với stream + timeout
            with session.get(full_url, stream=True, timeout=15) as response:
                response.raise_for_status()

                # Đọc nội dung từng chunk để tránh lỗi IncompleteRead
                bytes_content = io.BytesIO()
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # bỏ qua keep-alive chunk
                        bytes_content.write(chunk)

                bytes_content.seek(0)
                return bytes_content

        except requests.exceptions.ChunkedEncodingError as e:
            print(f"[WARN] - Mất kết nối giữa chừng khi tải ảnh: {url} | {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] - Lỗi request khi tải ảnh: {url} | {e}")
            return None
        except Exception as e:
            print(f"[ERROR] - Lỗi không xác định khi tải item về từ url: {url} | {e}")
            traceback.print_exc()
            return None
    def check_image_via_api(self, image_buffer, api_url, target_list=None, source="script_client"):
        """
        Gửi ảnh (dạng BytesIO) lên API để kiểm tra.
        
        Args:
            image_buffer (io.BytesIO): Dữ liệu ảnh trong RAM.
            api_url (str): Đường dẫn API (vd: http://localhost:8000/v1/filter).
            target_list (str, optional): Chuỗi các class mong muốn (vd: "pen, laptop").
            source (str): Nguồn gửi request.
            
        Returns:
            dict: Kết quả JSON từ API hoặc None nếu lỗi.
        """
        if not image_buffer:
            print("[WARN] - Không có dữ liệu ảnh để check API.")
            return None

        try:
            # Đảm bảo con trỏ file ở đầu
            image_buffer.seek(0)
            
            # Chuẩn bị payload
            files = {
                'file': ('image_check.jpg', image_buffer, 'image/jpeg')
            }
            data = {
                'source': source
            }
            
            # Nếu người dùng muốn lọc theo danh sách riêng
            if target_list:
                data['target_list'] = target_list

            print(f"[API] - Đang gửi ảnh tới {api_url}...")
            response = requests.post(api_url, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                # Reset con trỏ buffer về đầu sau khi gửi xong (để các hàm sau còn dùng được)
                image_buffer.seek(0) 
                return result
            else:
                print(f"[API ERROR] - Status: {response.status_code} | Msg: {response.text}")
                image_buffer.seek(0)
                return None
                
        except requests.exceptions.ConnectionError:
            print(f"[API ERROR] - Không thể kết nối tới server API ({api_url}). Server đã bật chưa?")
            return None
        except Exception as e:
            print(f"[API ERROR] - Lỗi không xác định: {e}")
            if image_buffer: image_buffer.seek(0)
            return None
if __name__ == "__main__":
    API_ENDPOINT = "http://localhost:8000/v1/filter" # Đảm bảo server API đang chạy port 8000
    OUTPUT_FOLDER = "./ket_qua_download"
    
    # Khởi tạo Processor
    processor = ImageProcessor()
    
    # URL test
    test_url = "https://img.lazcdn.com/g/p/3bca6aed13ab1ce9afd4ac0df71a23d0.jpg"
    
    # Tải ảnh về RAM
    img_buffer = processor.get_image_from_url(test_url)
    
    if img_buffer:
        # GỌI API ĐỂ LỌC
        api_result = processor.check_image_via_api(
            img_buffer, 
            API_ENDPOINT, 
            target_list="laptop, mouse" 
        )
        
        # Xử lý kết quả
        if api_result:
            print(f"\n--- KẾT QUẢ API ---")
            print(f"Hợp lệ: {api_result.get('is_valid')}")
            print(f"Nhãn tìm thấy: {api_result.get('detected_labels')}")
            print(f"Hành động: {api_result.get('action')}")
            print("-" * 20)
            
            if api_result.get('is_valid'):
                print("Ảnh đạt chuẩn. Đang lưu...")
                # Lưu file
                processor.save_image(img_buffer, OUTPUT_FOLDER, filename="keo.jpg")
            else:
                print("Ảnh bị loại (Không đúng class yêu cầu hoặc model chưa học).")
        else:
            print("Có lỗi khi gọi API (Server chưa bật hoặc lỗi mạng).")