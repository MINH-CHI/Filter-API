import streamlit as st #type:ignore
import requests #type:ignore
import pandas as pd #type:ignore
import plotly.express as px #type:ignore
from pymongo import MongoClient #type:ignore
import traceback
import os
import io
import sys
from minio import Minio
from dotenv import load_dotenv #type: ignore
dashboard_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(dashboard_dir)
project_root = os.path.dirname(app_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
env_path = os.path.join(project_root, ".env")
from PIL import Image, ImageDraw, ImageFont #type:ignore
import time
from datetime import datetime, timedelta, time as dt_time
load_dotenv(env_path)
def load_config(key, default_value = None):
    # if key in st.secrets:
    #     return st.secrets[key]
    return os.getenv(key, default_value)
st.set_page_config(page_title="AI Image Filter Dashboard", layout="wide", page_icon="🕵️")
# load_dotenv(env_path)
# MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "api_request_log"
COLLECTION_NAME = "test_confidence_0.1" 
CONFIG_COLLECTION = "system_config"

API_KEY = load_config("API_KEY", "default-secret-key")
MONGO_URI = load_config("MONGO_URI")
DB_NAME = load_config("DB_NAME", "api_request_log")
CONFIG_COLLECTION = load_config("CONFIG_COLLECTION", "system_config")
# MINIO_ENDPOINT = load_config("MINIO_ENDPOINT","localhost:9000")
MINIO_ACCESS_KEY = load_config("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = load_config("MINIO_SECRET_KEY")
MINIO_BUCKET = load_config("MINIO_BUCKET_NAME")
MINIO_SECURE = load_config("MINIO_SECURE", False)
@st.cache_resource # Kết nối 1 lần
def init_mongo_client():
    """Khởi tạo kết nối MongoDB và cache lại để dùng chung."""
    if not MONGO_URI:
        return None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        return client
    except Exception as e:
        return None
@st.cache_resource
def init_minio_client():
    client = Minio(
        endpoint="localhost:9000", # localhost:9000
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    return client
def annotate_image(image_source, detections):
    if isinstance(image_source, bytes):
        image = Image.open(io.BytesIO(image_source)).convert('RGB')
    else:
        image = image_source.copy().convert("RGB")
    
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for det in detections:
        box = det.get("box") # Format: [x1, y1, x2, y2]
        label = det.get("object", "obj")
        conf = det.get("confidence", 0.0)
        
        if box and len(box) == 4:
            # Vẽ khung
            draw.rectangle(box, outline="red", width=3)
            
            # Vẽ nhãn nền đỏ chữ trắng
            text = f"{label} {conf:.2f}"
            text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
            draw.rectangle(text_bbox, fill="red")
            draw.text((box[0], box[1]), text, fill="white", font=font)
            
    return image
def get_api_url_from_mongo():
    """Lấy API URL mới nhất từ MongoDB"""
    try:
        client = init_mongo_client()
        db = client[DB_NAME]
        coll = db[CONFIG_COLLECTION]
        
        doc = coll.find_one({"config_key":"active_api_url"})
        if doc and "value" in doc:
            return doc["value"]
    except Exception as e :
        st.error(f"Lỗi đọc MongoDB: {e}", icon="⚠️")
    return None

cloud_url = get_api_url_from_mongo()
BASE_URL = cloud_url if cloud_url else "http://127.0.0.1:8000"
if BASE_URL.endswith("/"): 
    BASE_URL = BASE_URL[:-1]
API_URL = f"{BASE_URL}/v1/filter"
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2593/2593491.png", width=50)
    st.title("Menu")
    st.success("✅ Đã xác thực tự động")
    st.divider()
    st.info(f"API: `{API_URL}`")
    st.caption("v1.3 - Auto Auth")

st.title("🕵️ Hệ thống Kiểm soát & Lọc Ảnh AI")

def annotate_image(image_source, detections):
    """Hàm vẽ bounding box lên ảnh"""
    if isinstance(image_source, bytes):
        image = Image.open(io.BytesIO(image_source)).convert('RGB')
    else:
        image = image_source.copy().convert("RGB")
    
    draw = ImageDraw.Draw(image)
    try: font = ImageFont.truetype("arial.ttf", 20)
    except: font = ImageFont.load_default()

    for det in detections:
        box = det.get("box") # [x1, y1, x2, y2]
        label = det.get("object", "obj")
        conf = det.get("confidence", 0.0)
        
        if box and len(box) == 4:
            draw.rectangle(box, outline="red", width=3)
            text = f"{label} {conf:.2f}"
            text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
            draw.rectangle(text_bbox, fill="red")
            draw.text((box[0], box[1]), text, fill="white", font=font)
    return image
def load_logs(start_date, end_date):
    client = init_mongo_client()
    if not client:
        return None
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    start_dt = datetime.combine(start_date, dt_time.min) 
    end_dt = datetime.combine(end_date, dt_time.max)
    query = {
        "timestamp": {
            "$gte": start_dt, # Greater than or equal (Lớn hơn hoặc bằng)
            "$lte": end_dt    # Less than or equal (Nhỏ hơn hoặc bằng)
        }
    }
    try:
        data = list(collection.find(query).sort("timestamp", -1).limit(2000))
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Lỗi truy vấn Log: {e}")
        return pd.DataFrame()
def load_test_results(limit=100):
    client = init_mongo_client()
    if not client:
        return pd.DataFrame()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME] 
    query = {"source": "batch_client_result"}
    try:
        # Lấy 500 record mới nhất
        data = list(collection.find(query).sort("timestamp", -1).limit(limit))
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()

tab1, tab2 = st.tabs(["🚀 Demo & Visualize", "🧪 Phân tích Test"])

with tab1:
    st.header("Test Model & Vẽ Bounding Box")
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload ảnh", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="Ảnh gốc", use_container_width=True)
            if st.button("🔍 Quét & Vẽ Box", type="primary"):
                with st.spinner('Đang xử lý...'):
                    try:
                        uploaded_file.seek(0)
                        files = {'file': uploaded_file}
                        data = {'source': 'streamlit_demo'}
                        headers = {'x-api-key': API_KEY}
                        response = requests.post(API_URL, files=files, data=data, headers=headers)
                        if response.status_code == 200:
                            result = response.json()
                            with col2:
                                st.subheader("Kết quả AI")
                                detections = result.get('detections', [])
                                annotated_img = annotate_image(original_image, detections)
                                st.image(annotated_img, caption=f"Result ({len(detections)} objects)", use_container_width=True)
                                st.json(result)
                        else:
                            st.error(f"Lỗi API: {response.text}")
                    except Exception as e:
                        st.error(f"Lỗi: {e}")

with tab2:
    st.header("🧪 Kết quả Batch Test")
    
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        limit_rows = st.slider("Số lượng ảnh hiển thị (Mới nhất)", 5, 100, 20)
    with c2:
        if st.button("🔄 Refresh Data"): st.rerun()
    with c3:
        if st.button("🗑️ Xóa Log Test", type="primary"):
            client = init_mongo_client()
            if client:
                client[DB_NAME][COLLECTION_NAME].delete_many({"source": "batch_client_result"})
                st.toast("Đã xóa dữ liệu test!", icon="🧹")
                time.sleep(1)
                st.rerun()

    # Load Data
    df_test = load_test_results(limit=limit_rows)
    minio_client = init_minio_client()

    if df_test.empty:
        st.warning("⚠️ Không có dữ liệu Test. Hãy chạy `python batch_test.py`.")
    else:
        # Metrics tổng quan
        total = len(df_test)
        correct = df_test['is_correct'].sum() if 'is_correct' in df_test.columns else 0
        acc = (correct/total*100) if total > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Số mẫu", total)
        m2.metric("Độ chính xác", f"{acc:.1f}%")
        m3.metric("KEEP Ratio", f"{len(df_test[df_test.get('action')=='KEEP'])}/{total}")
        
        st.divider()
        st.subheader(f"📋 Danh sách chi tiết ({len(df_test)} ảnh gần nhất)")

        h1, h2, h3 = st.columns([1, 2, 2])
        h1.markdown("**1. Tên File & Info**")
        h2.markdown("**2. Ảnh Gốc (MinIO)**")
        h3.markdown("**3. Ảnh Sau Xử Lý (AI Bbox)**")
        st.markdown("---")

        for idx, row in df_test.iterrows():
            # Layout 3 cột cho mỗi dòng
            c_info, c_orig, c_proc = st.columns([1, 2, 2])
            
            # --- Cột 1: Thông tin ---
            with c_info:
                st.markdown(f"📄 **{row.get('filename', 'N/A')}**")
                
                # Màu sắc trạng thái
                act = row.get('action', 'UNKNOWN')
                color = "green" if act == "KEEP" else "orange" if act == "SKIP" else "red"
                st.markdown(f"Action: :{color}[**{act}**]")
                
                st.caption(f"Label gốc: {row.get('actual_label')}")
                st.caption(f"Label đoán: {row.get('predicted_label')}")
                st.caption(f"Conf: {row.get('confidence', 0):.2f}")
                st.caption(f"Time: {row.get('timestamp').strftime('%H:%M:%S')}")

            # --- Chuẩn bị ảnh ---
            minio_path = row.get("minio_image_path")
            img_bytes = None
            
            # --- Cột 2: Ảnh Gốc từ MinIO ---
            with c_orig:
                if minio_path and minio_client:
                    try:
                        # Fetch từ MinIO
                        resp = minio_client.get_object(MINIO_BUCKET, minio_path)
                        img_bytes = resp.read()
                        resp.close()
                        resp.release_conn()
                        
                        st.image(img_bytes, caption="Original from MinIO", use_container_width=True)
                    except Exception as e:
                        st.error(f"Lỗi MinIO: {e}")
                        st.caption(f"Path: {minio_path}")
                else:
                    st.warning("Không có MinIO Path")

            # --- Cột 3: Ảnh Đã Vẽ Box ---
            with c_proc:
                if img_bytes:
                    detections = row.get("detections_detail", [])
                    # Nếu record không có detections_detail, thử lấy detections (tuỳ format db của bạn)
                    if not detections: detections = row.get("detections", [])
                    
                    try:
                        annotated = annotate_image(img_bytes, detections)
                        st.image(annotated, caption=f"Processed ({len(detections)} box)", use_container_width=True)
                    except Exception as e:
                        st.error("Lỗi vẽ ảnh")
                else:
                    st.info("Chưa có ảnh gốc để vẽ")
            
            # Kẻ dòng phân cách giữa các hàng
            st.markdown("---")