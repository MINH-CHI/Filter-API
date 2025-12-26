import torch #type: ignore
import os
import mimetypes
import cv2 #type: ignore
import numpy as np #type: ignore
from pymongo import MongoClient, errors  #type: ignore
from bson.binary import Binary #type: ignore
from ultralytics import YOLO #type: ignore
from datetime import datetime
import io
from minio import Minio #type: ignore
from minio.error import S3Error #type: ignore
class ImageFilter:
    def __init__(self, model_path, mongo_uri, db_name, collection_name,target_classes, minio_config, image_handler = None, log_handler = None, enable_filter = True, device=0,class_mapping=None):
        self.class_mapping = class_mapping  
        self.enable_filter = enable_filter
        self.device = device
        self.target_classes = set(target_classes)
        self.stats = {label: 0 for label in target_classes} # Thống kê
        
        if image_handler:
            self.image_handler = image_handler
            print("[INFOR] Đang sử dụng loại lưu trữ ảnh từ user")
        elif minio_config:
            self.minio_client = None
            self.bucket_name = minio_config.get("bucket_name", "images")
            try:
                self.minio_client = Minio(
                    endpoint= minio_config['endpoint'], # minio:9000 / localhost:9000
                    access_key=minio_config["access_key"],
                    secret_key=minio_config["secret_key"],
                    secure=minio_config["secure"] # True nếu dùng HTTPS
                )
                
                # Kiểm tra xem bucket có tồn tại không, nếu không thì tạo
                if not self.minio_client.bucket_exists(self.bucket_name):
                    self.minio_client.make_bucket(self.bucket_name)
                    print(f"[MINIO] Đã tạo bucket mới: {self.bucket_name}")
                else:
                    print(f"[MINIO] Đã kết nối bucket: {self.bucket_name}")
                    
            except Exception as e:
                print(f"[ERROR] ❌ Không thể kết nối MinIO: {e}")
                self.minio_client = None
            self.image_handler = self._upload_to_minio
            print("[INFOR] Đang sử dụng loại lưu trữ ảnh là MinIO")
        else:
            self.image_handler = lambda data, name : None
            print("[Warning] Không có cấu hình lưu ảnh")
        
        if log_handler:
            self.log_handler = log_handler
            print("[INFOR] Đang dùng database do user tùy chỉnh")
        elif mongo_uri and db_name and collection_name:
            self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS = 5000)
            self.db = self.mongo_client[db_name]
            self.collection = self.db[collection_name]
            self.mongo_client.server_info()
            print(f"[INFOR] Đã kết nối tới Mongo ở database: {db_name}")
            self.log_handler = self._log_to_mongo
            print("[INFO] Sử dụng Default MongoDB Handler.")
        else:
            self.log_handler = lambda **kwargs: print(f"[LOG] {kwargs.get('action')}")
            print("[WARNING] Không có cấu hình Database. Log chỉ in ra console.")
            
        if self.enable_filter:
            # Check nhanh xem máy host có nhận GPU không
            if self.device == 0 and not torch.cuda.is_available():
                print("[WARNING] Đã chọn GPU nhưng Torch không tìm thấy CUDA -> Không được dùng")
                return
            else:
                print(f"Đang sử dụng thiết bị: {self.device}")
            print(f"[INFO] Đang load model từ {model_path}...")# Tải model
            self.model = YOLO(model_path,task="detect")
        else:
            print("Filter đang tắt. Mọi ảnh sẽ được chấp nhận.")

    def _bytes_to_image(self, image_bytes): 
        if not isinstance(image_bytes, (bytes, bytearray)):
            print(f"[Error] Dữ liệu đầu vào không phải là bytes. Nhận được kiểu: {type(image_bytes)}")
            return None

        nparr = np.frombuffer(image_bytes, np.uint8) 
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("[Warning] Dữ liệu bytes không phải là file ảnh hợp lệ hoặc bị hỏng.")
        return img
    def _upload_to_minio(self, image_bytes, full_path_name):
        """
        Upload ảnh lên MinIO.
        :param full_path_name: Tên file có thể kèm folder (VD: 'retrain/image.jpg')
        """
        if not self.minio_client:
            return None
        
        try:
            # Tách thư mục và tên file riêng biệt
            # VD: "retrain/anh.jpg" -> folder="retrain", filename="anh.jpg"
            folder, filename = os.path.split(full_path_name)
            
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # folder + timestamp + filename
            # VD: "retrain" + "20251223_anh.jpg" -> "retrain/20251223_anh.jpg"
            final_object_name = f"{timestamp_str}_{filename}"
            if folder:
                final_object_name = f"{folder}/{final_object_name}"
            
            # Tự động đoán Content-Type (image/jpeg hay image/png)
            content_type, _ = mimetypes.guess_type(filename)
            if not content_type:
                content_type = "image/jpeg" # Fallback nếu không đoán được

            # Upload
            data_stream = io.BytesIO(image_bytes)
            data_length = len(image_bytes)
            
            self.minio_client.put_object(
                self.bucket_name,
                final_object_name, # Tên object đã chuẩn hóa
                data_stream,
                data_length,
                content_type=content_type 
            )
            
            # Trả về đường dẫn để lưu vào MongoDB
            return final_object_name
            
        except Exception as e:
            print(f"❌ [MINIO ERROR] Upload thất bại: {e}")
            return None

    def process(self, input_data, metadata=None,custom_targets=None):
        # check cờ tắt/bật
        if not self.enable_filter:
            return True, [], [], "BYPASSED"

        # Decode ảnh
        img_numpy = self._bytes_to_image(input_data)
        if img_numpy is None:
            self._log_to_mongo(metadata,detected_labels=[], is_valid=False, action="UNPROCESSED", reason="Invalid Image Data (Decode Failed)")
            return False, [], "Image decode failed", "ERROR"
        
        # Inference
        results = self.model(img_numpy, conf=0, verbose=False)
        detailed_info = []
        detected_labels = set()
        
        # Lấy danh sách tên class phát hiện được
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                box_coords = box.xyxy[0].tolist() # box.xyxy trả về tensor [x1, y1, x2, y2], cần chuyển sang list
                box_coords = [round(x, 1) for x in box_coords]
                
                label_name = str(cls_id)
                if self.class_mapping:
                    label_name = self.class_mapping.get(cls_id, str(cls_id))
                elif hasattr(self.model, 'names'):
                    label_name = self.model.names.get(cls_id, str(cls_id))
                detailed_info.append({
                    "object": label_name,
                    "confidence": round(conf, 2),
                    "box": box_coords
                })
                
                detected_labels.add(label_name)
        
        # Model trả về None (Không phát hiện gì hoặc confidence thấp)
        action_result = ""
        is_valid_result = False # Default
        reason_msg = ""
        if not detected_labels:
            action_result = "UNPROCESSED"
            reason_msg = "No Objects Detected"
        else:
            if custom_targets:
                targets_to_check = set(custom_targets)
            else:
                targets_to_check = self.target_classes

            intersect = detected_labels.intersection(targets_to_check)
            is_valid_result = bool(intersect)
            
            if is_valid_result:
                action_result = "KEEP"
                reason_msg = "Found Target Classes"
            else:
                action_result = "SKIP"
                reason_msg = "Objects detected but NOT in Target"
                
        minio_path = None        
        if input_data:
            filename = metadata.get("filename", "unknown.jpg") if metadata else "unknown.jpg"
            save_name = ""
            if action_result == "KEEP":
                # Thêm prefix "dataset/" vào trước tên file
                save_name = f"dataset/{filename}" 
                # minio_path = self._upload_to_minio(input_data, save_name)
                
            # Ảnh model mù (UNPROCESSED)
            elif action_result == "UNPROCESSED":
                # Thêm prefix "retrain/" để gom riêng ra
                save_name = f"retrain/{filename}"
                # minio_path = self._upload_to_minio(input_data, save_name)
            # Model đã được học rồi nên skip
            else:
                pass
            
            if save_name:
                minio_path = self.image_handler(input_data,save_name)
        # Ghi log mọi case vào MongoDB
        self.log_handler(
            metadata=metadata,
            detected_labels=list(detected_labels),
            detections_detail=detailed_info,
            is_valid=is_valid_result,
            action=action_result,
            reason=reason_msg,
            minio_object_name=minio_path
        )
        
        # Update thống kê (chỉ cộng nếu là target)
        if is_valid_result:
            for label in detected_labels.intersection(self.target_classes):
                if label in self.stats: 
                    self.stats[label] += 1

        return is_valid_result, list(detected_labels), detailed_info , action_result

    def _log_to_mongo(self, metadata, action, detected_labels=None, detections_detail=None,is_valid=False, reason=None, minio_object_name=None):
        """
        Ghi log chi tiết mọi request vào MongoDB để phục vụ Dashboard.
        """
        try:
            # Xử lý Metadata
            meta = metadata or {}
            
            # Lôi thông tin quan trọng ra ngoài
            user_name = meta.get("user", "Anonymous")
            source = meta.get("api_source", "unknown")
            filename = meta.get("filename", "unknown")

            # Tạo document cấu trúc phẳng
            doc = {
                "timestamp": datetime.now(),       # Thời gian server nhận ảnh
                "user": user_name,                 
                "source": source,
                "filename": filename,
                "is_valid": is_valid,              # Kết quả logic: True/False
                "action": action,                  # Hành động: KEEP/DISCARD -> Mở rộng thêm như UNPROCESSED để xử lý sau
                "detected_labels": detected_labels if detected_labels else [],
                "detections_detail": detections_detail if detections_detail else [],
                "reason": reason,
                "minio_image_path": minio_object_name, # Lưu tên file trên MinIO
                "storage_type": "minio" if minio_object_name else "none",        
                "raw_metadata": meta
            }

            # Insert vào Database
            self.collection.insert_one(doc)
            print(f"[MONGO] ✅ Đã lưu log user: {user_name} | Labels: {detected_labels}")

        except Exception as e:
            print(f"[ERROR] ❌ Lỗi khi lưu MongoDB: {e}")

    def get_stats(self):
        """Trả về thống kê số lượng đã thu thập"""
        return self.stats