import torch #type: ignore
import cv2 #type: ignore
import numpy as np #type: ignore
from pymongo import MongoClient, errors  #type: ignore
from bson.binary import Binary #type: ignore
from ultralytics import YOLO #type: ignore
from datetime import datetime
from bson.binary import Binary
class ImageFilter:
    def __init__(self, model_path, mongo_uri, db_name, collection_name,target_classes, enable_filter = True, device=0,class_mapping=None):
        self.class_mapping = class_mapping  
        self.enable_filter = enable_filter
        self.device = device
        self.target_classes = set(target_classes)
        self.stats = {label: 0 for label in target_classes} # Thống kê
        
        if self.enable_filter:
            # Check nhanh xem máy có nhận GPU không
            if self.device == 0 and not torch.cuda.is_available():
                print("[WARNING] Đã chọn GPU nhưng Torch không tìm thấy CUDA -> sẽ tự động chuyển về CPU.")
                self.device = 'cpu'
            else:
                print(f"Đang sử dụng thiết bị: {self.device}")

        if self.enable_filter:
            print(f"[INFO] Đang load model từ {model_path}...")# Tải model
            self.model = YOLO(model_path,task="detect")

            print(" [ÌNO] Đang kết nối MongoDB...") # kết nối mongo
            try:
                self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
                self.db = self.client[db_name]
                self.collection = self.db[collection_name]
                # Test kết nối
                self.client.server_info()
            except errors.ServerSelectionTimeoutError as err:
                print("[ERROR] Không thể kết nối MongoDB")
                raise err
        else:
            print("Filter đang tắt. Mọi ảnh sẽ được chấp nhận.")

    def _bytes_to_image(self, image_bytes):
        if not isinstance(image_bytes, (bytes, bytearray)):
            print(f"[Error] Dữ liệu đầu vào không phải là bytes. Nhận được kiểu: {type(image_bytes)}")
            return None

        if not image_bytes or len(image_bytes) == 0:
            print("[WARNING] Dữ liệu bytes bị rỗng (Empty).")
            return None

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                print("[Warning] Dữ liệu bytes không phải là file ảnh hợp lệ hoặc bị hỏng.")
                return None
                
            return img

        except Exception as e:
            print(f"[Error] Có lỗi ngoại lệ khi xử lý ảnh. Chi tiết: {e}")
            return None

    def process(self, image_bytes, metadata=None,custom_targets=None):
        # check cờ tắt/bật
        if not self.enable_filter:
            return True, []

        # Decode ảnh
        img = self._bytes_to_image(image_bytes)
        if img is None:
            return False, [] # ảnh lỗi -> bỏ

        # Inference
        results = self.model(img, conf=0.65, verbose=False)
        
        detected_labels = []
        
        # Lấy danh sách tên class phát hiện được
        for result in results:
            for box in result.boxes:
                # Lấy ID và Confidence
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                label_name = str(cls_id) # Mặc định là số (string)
                
                if self.class_mapping:
                    # Ưu tiên 1: Tra cứu từ Class Mapping (Sửa lỗi ONNX)
                    label_name = self.class_mapping.get(cls_id, str(cls_id))
                elif hasattr(self.model, 'names'):
                    # Ưu tiên 2: Lấy từ nội tại Model (Nếu dùng .pt)
                    label_name = self.model.names.get(cls_id, str(cls_id))
                
                detected_labels.append(label_name)
        
        unique_labels = set(detected_labels)

        
        # Model trả về None (Không phát hiện gì hoặc confidence thấp)
        if not unique_labels:
            self._log_to_mongo(
            metadata=metadata, 
            image_bytes=image_bytes, 
            detected_labels=[], 
            is_valid=False, 
            action="DISCARD", 
            reason="Model returned None"
        )
            return False, [] # Bỏ sau khi đã ném vào DB
        
        if custom_targets and len(custom_targets) > 0:
            targets_to_check = set(custom_targets)
        else:
            targets_to_check = self.target_classes
        # Có nhãn, kiểm tra xem có nằm trong các class requirement không ?
        intersect = unique_labels.intersection(targets_to_check)

        if intersect:
            # Có ít nhất 1 nhãn mục tiêu -> GIỮ LẠI
            is_valid_result = bool(intersect) # True nếu có giao nhau
            action_result = "KEEP" if is_valid_result else "DISCARD"
            
            # Gọi hàm log cho trường hợp đã detect ra (Dù KEEP hay DISCARD đều log hết)
            self._log_to_mongo(
                metadata=metadata,
                image_bytes=image_bytes,
                detected_labels=list(unique_labels),
                is_valid=is_valid_result,
                action=action_result,
                reason="Filtered by Target Classes"
            )

            return is_valid_result, list(unique_labels)
        else:
            # Có nhãn nhưng không thuộc các class requirement -> bỏ
            return False, list(unique_labels)

    def _log_to_mongo(self, metadata, image_bytes, detected_labels=None, is_valid=False, action="DISCARD", reason=None):
        """
        Ghi log chi tiết mọi request vào MongoDB để phục vụ Dashboard.
        """
        try:
            # Xử lý Metadata (Tránh lỗi nếu metadata là None)
            meta = metadata or {}
            
            # Lôi thông tin quan trọng ra ngoài (QUAN TRỌNG)
            user_name = meta.get("user", "Anonymous")
            source = meta.get("api_source", "unknown")
            filename = meta.get("filename", "unknown")

            # Tạo document cấu trúc phẳng (Flat Structure)
            doc = {
                "timestamp": datetime.now(),       # Thời gian server nhận ảnh
                "user": user_name,                 # <--- Đây là cái bạn đang cần hiển thị
                "source": source,
                "filename": filename,
                "is_valid": is_valid,              # Kết quả logic: True/False
                "action": action,                  # Hành động: KEEP/DISCARD
                "detected_labels": detected_labels if detected_labels else [],
                "reason": reason,                  # Ghi chú thêm (nếu có)
                
                # Lưu ảnh nhỏ (Thumbnail) hoặc ảnh gốc
                # Lưu ý: MongoDB giới hạn 16MB/doc. Nếu ảnh quá to nên resize trước khi lưu.
                "image_data": Binary(image_bytes) if image_bytes else None,
                
                # Lưu lại toàn bộ metadata gốc vào 1 góc để debug sau này
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
    
    # Hướng dẫn sử dụng
    # filter = ImageFilter(
    # model_path="yolov8n.pt",       # file weights (.pt)
    # mongo_uri="mongodb://localhost:27017/",
    # db_name="minh_db",
    # collection_name="unknown_images",
    # target_classes=TARGET_CLASSES,
    # enable_filter=True            
    # )
    
    # meta_info = {"source_url": image_url, "scraper_id": "bot_01"} 
    
    # is_valid, labels = filter.process(image_bytes, metadata=meta_info) => trả về (True/False, [danh sách nhãn phát hiện])