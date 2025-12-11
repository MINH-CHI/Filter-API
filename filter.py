import cv2
import torch
import numpy as np
from pymongo import MongoClient, errors 
from datetime import datetime
from bson.binary import Binary
from ultralytics import YOLO

class ImageFilter:
    def __init__(self, model_path, mongo_uri, db_name, collection_name,target_classes, enable_filter = True, device=0):
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
            self.model = YOLO(model_path)

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
        results = self.model(img, verbose=False, conf=0.5,device=self.device)
        
        detected_labels = []
        
        # Lấy danh sách tên class phát hiện được
        for result in results:
            for cls_id in result.boxes.cls:
                label_name = self.model.names[int(cls_id)]
                detected_labels.append(label_name)
        
        unique_labels = set(detected_labels)

        
        # Model trả về None (Không phát hiện gì hoặc confidence thấp)
        if not unique_labels:
            self._log_to_mongo(metadata, image_bytes, reason="Model returned None")
            return False, [] # Bỏ sau khi đã ném vào DB
        
        if custom_targets and len(custom_targets) > 0:
            targets_to_check = set(custom_targets)
        else:
            targets_to_check = self.target_classes
        # Có nhãn, kiểm tra xem có nằm trong các class requirement không ?
        intersect = unique_labels.intersection(self.target_classes)

        if intersect:
            # Có ít nhất 1 nhãn mục tiêu -> GIỮ LẠI
            for label in intersect:
                self.stats[label] += 1
            return True, list(unique_labels)
        else:
            # Có nhãn nhưng không thuộc các class requirement -> bỏ
            return False, list(unique_labels)

    def _log_to_mongo(self, metadata, image_bytes,reason):
        """Lưu metadata của ảnh và ảnh có nhãn chưa học vào MongoDB để xử lý sau"""
        try:
            doc = {
                "reason": reason,
                "metadata": metadata or {},
                "status": "pending_review",
                "image_data": Binary(image_bytes)
            }
            self.collection.insert_one(doc)
            print("[ÌNOR] Đã lưu case None vào MongoDB")
        except Exception as e:
            print(f"[ERROR] Lỗi khi lưu case vòa MongoDB: {e}")

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