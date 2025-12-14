import torch #type: ignore
import cv2 #type: ignore
import numpy as np #type: ignore
from pymongo import MongoClient, errors  #type: ignore
from bson.binary import Binary #type: ignore
from ultralytics import YOLO #type: ignore
from datetime import datetime
from bson.binary import Binary #type: ignore
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
                self.device = 'cpu' # ÉP dùng GPU
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

    def _bytes_to_image(self, input_data): # Check đối tượng của OpenCV
        if isinstance(input_data, np.ndarray):
            _, buffer = cv2.imencode('.jpg', input_data)
            return buffer.tobytes()
        elif isinstance(input_data, (bytes, bytearray)):
            if len(input_data) == 0:
                print("[Warning] Dữ liệu bytes đầu vào bị rỗng.")
                return None
            return input_data

        else:
            print(f"[Error] Định dạng đầu vào không hỗ trợ: {type(input_data)}. Cần: numpy.ndarray hoặc bytes.")
            return None

    def process(self, input_data, metadata=None,custom_targets=None):
        # check cờ tắt/bật
        if not self.enable_filter:
            return True, [], []

        # Decode ảnh
        img = self._bytes_to_image(input_data)
        if img is None:
            return False, [], [] # ảnh lỗi -> bỏ

        # Inference
        results = self.model(img, conf=0.8, verbose=False)        
        detailed_info = []
        detected_labels = set()
        
        # Lấy danh sách tên class phát hiện được
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                label_name = str(cls_id)
                if self.class_mapping:
                    label_name = self.class_mapping.get(cls_id, str(cls_id))
                elif hasattr(self.model, 'names'):
                    label_name = self.model.names.get(cls_id, str(cls_id))
                detailed_info.append({
                    "object": label_name,
                    "confidence": round(conf, 2)
                })
                
                detected_labels.add(label_name)
        
        # unique_labels = set(detected_labels)

        
        # Model trả về None (Không phát hiện gì hoặc confidence thấp)
        if not detected_labels:
            self._log_to_mongo(
            metadata=metadata, 
            input_data=input_data, 
            detected_labels=[], 
            is_valid=False, 
            action="DISCARD", 
            reason="Model returned None"
        )
            return False, [], [] # Bỏ sau khi đã ném vào DB
        
        if custom_targets and len(custom_targets) > 0:
            targets_to_check = set(custom_targets)
        else:
            targets_to_check = self.target_classes
        # Có nhãn, kiểm tra xem có nằm trong các class requirement không ?
        intersect = detected_labels.intersection(targets_to_check)

        is_valid_result = bool(intersect) # True nếu có giao nhau, False nếu không
        action_result = "KEEP" if is_valid_result else "DISCARD"
        
        # Ghi log (Dù là chó, mèo hay điện thoại đều được ghi lại hết)
        self._log_to_mongo(
            metadata=metadata,
            input_data=input_data,
            detected_labels=list(detected_labels),
            detections_detail=detailed_info,
            is_valid=is_valid_result,
            action=action_result,
            reason="Filtered by Target Classes"
        )
        
        # Update thống kê (chỉ cộng nếu là target)
        if is_valid_result:
            for label in intersect:
                if label in self.stats:
                    self.stats[label] += 1

        return is_valid_result, list(detected_labels), detailed_info

    def _log_to_mongo(self, metadata, input_data, detected_labels=None, detections_detail=None,is_valid=False, action="DISCARD", reason=None):
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
                "action": action,                  # Hành động: KEEP/DISCARD
                "detected_labels": detected_labels if detected_labels else [],
                "detections_detail": detections_detail if detections_detail else [],
                "reason": reason,                  # Ghi chú thêm (nếu có)
                
                # Lưu ảnh nhỏ (Thumbnail) hoặc ảnh gốc
                "image_data": Binary(input_data) if input_data else None,
                
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