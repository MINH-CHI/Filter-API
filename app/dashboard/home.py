import streamlit as st #type:ignore
import requests #type:ignore
import pandas as pd #type:ignore
import plotly.express as px #type:ignore
from pymongo import MongoClient #type:ignore
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

def load_config(key, default_value = None):
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default_value)
st.set_page_config(page_title="AI Image Filter Dashboard", layout="wide", page_icon="🕵️")
# load_dotenv(env_path)
# MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "api_request_log"
COLLECTION_NAME = "api_unlabeled_images" 
CONFIG_COLLECTION = "system_config"

MONGO_URI = load_config("MONGO_URI")
DB_NAME = load_config("DB_NAME", "api_request_log")
CONFIG_COLLECTION = load_config("CONFIG_COLLECTION", "system_config")
MINIO_ENDPOINT = load_config("MINIO_ENDPOINT")
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
        endpoint=MINIO_ENDPOINT, # localhost:9000
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
    st.title("Cấu hình")
    
    # Input API Key
    api_key = st.text_input("🔑 Nhập API Key", type="password", help="Nhập key từ file secrets_config.py")
    st.divider()
    st.header("📅 Bộ lọc thời gian")
    today = datetime.now()
    default_start = today - timedelta(days=3)
    
    start_date = st.date_input("Từ ngày", value=default_start)
    end_date = st.date_input("Đến ngày", value=today)
    
    if start_date > end_date:
        st.error("Ngày bắt đầu phải nhỏ hơn ngày kết thúc!")
        
    st.info(f"API URL: `{API_URL}`")

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
def load_test_results():
    client = init_mongo_client()
    if not client:
        return pd.DataFrame()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME] 
    query = {"source": "batch_script_runner"}
    try:
        # Lấy 500 record mới nhất
        data = list(collection.find(query).sort("timestamp", -1).limit(500))
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()
st.title("🕵️ Hệ thống Kiểm soát & Lọc Ảnh AI")
if not api_key:
    st.warning("⚠️ Vui lòng nhập **API Key** ở thanh bên trái (Sidebar) để bắt đầu sử dụng.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🚀 Demo & Visualize", "📸 Giám sát Live (Lazy Load)", "🧪 Phân tích Batch Test"])

with tab1:
    st.header("Test Model & Vẽ Bounding Box")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload ảnh", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            # Hiển thị ảnh gốc trước
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="Ảnh gốc", use_container_width=True)
            
            if st.button("🔍 Quét & Vẽ Box", type="primary"):
                with st.spinner('Đang xử lý...'):
                    try:
                        uploaded_file.seek(0)
                        files = {'file': uploaded_file}
                        data = {'source': 'streamlit_demo'}
                        headers = {'x-api-key': api_key}
                        
                        response = requests.post(API_URL, files=files, data=data, headers=headers)
                        
                        if response.status_code == 200:
                            result = response.json()
                            with col2:
                                st.subheader("Kết quả AI")
                                action = result.get('action', 'UNKNOWN')
                                detections = result.get('detections', [])
                                
                                # --- VẼ BOX LÊN ẢNH ---
                                annotated_img = annotate_image(original_image, detections)
                                st.image(annotated_img, caption=f"Ảnh đã xử lý ({len(detections)} objects)", use_container_width=True)
                                
                                # Hiển thị Action Label
                                if action == 'KEEP': st.success(f"✅ HỢP LỆ (KEEP)")
                                elif action == 'SKIP': st.warning(f"🟡 SKIP (Đúng nhưng không lấy)")
                                else: st.error(f"❌ LOẠI BỎ ({action})")
                                
                                st.json(result) # Show JSON raw để debug
                        else:
                            st.error(f"Lỗi API: {response.text}")
                    except Exception as e:
                        st.error(f"Lỗi: {e}")
with tab2:
    st.header("📸 Giám sát Dữ liệu Thực tế (Pagination)")
    
    # 1. Load dữ liệu Metadata từ Mongo
    client = init_mongo_client()
    minio = init_minio_client()
    
    if client:
        db = client[DB_NAME]
        coll = db[COLLECTION_NAME]
        
        # Query filter
        start_dt = datetime.combine(start_date, dt_time.min)
        end_dt = datetime.combine(end_date, dt_time.max)
        query = {
            "timestamp": {"$gte": start_dt, "$lte": end_dt},
            "minio_image_path": {"$ne": None} # Chỉ lấy record có ảnh trên MinIO
        }
        
        # Đếm tổng số lượng để phân trang
        total_docs = coll.count_documents(query)
        
        # Cấu hình Pagination (Lazy Load giả lập)
        PAGE_SIZE = 8 # Số ảnh mỗi lần load
        if "page_number" not in st.session_state:
            st.session_state.page_number = 0
            
        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
        with col_nav1:
            if st.button("⬅️ Trang trước") and st.session_state.page_number > 0:
                st.session_state.page_number -= 1
                st.rerun()
        with col_nav3:
            if st.button("Trang sau ➡️") and (st.session_state.page_number + 1) * PAGE_SIZE < total_docs:
                st.session_state.page_number += 1
                st.rerun()
        with col_nav2:
            st.write(f"Đang hiển thị trang **{st.session_state.page_number + 1}** / {((total_docs // PAGE_SIZE) + 1)} (Tổng: {total_docs} ảnh)")

        # Lấy data theo trang (Skip & Limit)
        cursor = coll.find(query).sort("timestamp", -1).skip(st.session_state.page_number * PAGE_SIZE).limit(PAGE_SIZE)
        logs = list(cursor)
        
        if not logs:
            st.info("Không có dữ liệu trong khoảng thời gian này.")
        else:
            # Hiển thị Grid 4 cột
            cols = st.columns(4)
            for idx, log in enumerate(logs):
                with cols[idx % 4]:
                    minio_path = log.get("minio_image_path")
                    detections = log.get("detections_detail", [])
                    action = log.get("action", "UNKNOWN")
                    
                    # Logic màu sắc Status
                    status_color = "green" if action == "KEEP" else "orange" if action == "SKIP" else "red"
                    st.markdown(f":{status_color}[**{action}**] - {log['timestamp'].strftime('%H:%M:%S')}")
                    
                    # Tải ảnh từ MinIO & Vẽ Box
                    if minio and minio_path:
                        try:
                            response = minio.get_object(MINIO_BUCKET, minio_path)
                            img_data = response.read()
                            response.close()
                            response.release_conn()
                            
                            # Vẽ box
                            final_img = annotate_image(img_data, detections)
                            st.image(final_img, use_container_width=True)
                        except Exception as e:
                            st.error(f"Lỗi tải ảnh: {e}")
                    else:
                        st.warning("MinIO chưa kết nối")
with tab3:
    st.header("🧪 Giám sát Batch Test (Real-time)")
    st.markdown("""
    > **Trạng thái:** Hiển thị kết quả từ `batch_test.py`.
    > **Logic màu sắc:** 🟢 **KEEP** (Lấy) | 🟡 **SKIP** (không lấy vì đã được học) | 🔴 **UNPROCESSED** (Không thấy gì)
    """)

    col_re1, col_re2, col_re3 = st.columns([1, 1, 4])
    with col_re1:
        auto_refresh_tab3 = st.toggle("🔴 Auto-Refresh", value=True, key="tab3_live")
    with col_re2:
        if st.button("🗑️ Xóa Log Test", type="primary", key="btn_clear_test"):
            client = init_mongo_client()
            if client:
                # Xóa đúng nguồn dữ liệu test
                client[DB_NAME][COLLECTION_NAME].delete_many({"source": "batch_script_runner"})
                st.toast("Đã xóa sạch dữ liệu test cũ!", icon="🧹")
                time.sleep(1)
                st.rerun()
    with col_re3:
        if st.button("🔄 Làm mới", key="btn_reload_tab3"):
            st.rerun()

    df_test = load_test_results()

    if df_test.empty:
        st.warning("⚠️ Chưa tìm thấy dữ liệu Test. Hãy chạy lệnh `python batch_test.py` ở terminal.")
    else:
        expected_cols = ['is_correct', 'action', 'predicted_label', 'actual_label', 'confidence', 'filename', 'bounding_box']
        for c in expected_cols:
            if c not in df_test.columns: df_test[c] = None

        total_test = len(df_test)
        
        correct_count = df_test['is_correct'].sum()
        acc_val = (correct_count / total_test * 100) if total_test > 0 else 0.0
        
        keep_count = len(df_test[df_test['action'] == 'KEEP'])
        skip_count = len(df_test[df_test['action'] == 'SKIP']) 
        
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Số mẫu đã Test", total_test)
        k2.metric("Độ chính xác", f"{acc_val:.1f}%")
        k3.metric("🟢 KEEP", keep_count)
        k4.metric("🟡 SKIP", skip_count)
        k5.metric("Trạng thái", df_test.iloc[0]['status'] if 'status' in df_test.columns else "N/A")

        st.divider()

        # Chia cột: Bảng chiếm 70%, Biểu đồ chiếm 30%
        c1, c2 = st.columns([7, 3])
        
        with c1:
            st.subheader("📋 Chi tiết từng ảnh")
            
            def highlight_row_by_action(row):
                status = row.get("action", "")
                
                # Logic màu sắc: KEEP=Xanh, SKIP=Vàng, UNPROCESSED=Đỏ
                if status == "KEEP":
                    return ['background-color: #d4edda; color: #155724'] * len(row) # Xanh lá
                elif status == "SKIP":
                    return ['background-color: #fff3cd; color: #856404'] * len(row) # 🟡 Vàng cam
                elif status == "UNPROCESSED":
                    return ['background-color: #f8d7da; color: #721c24'] * len(row) # Đỏ
                return [''] * len(row)

            display_cols = ['timestamp', 'filename', 'actual_label', 'predicted_label', 'bounding_box', 'confidence', 'action', 'is_correct']
            
            df_display = df_test[[c for c in display_cols if c in df_test.columns]].copy()
            
            # Sử dụng style.apply thay vì applymap để tô màu cả dòng
            st.dataframe(
                df_display.style.apply(highlight_row_by_action, axis=1), 
                use_container_width=True,
                height=500
            )

        with c2:
            st.subheader("📊 Thống kê")
            
            # Chart 1: Độ chính xác
            st.caption("Độ chính xác (Model Predict)")
            res_counts = df_test['is_correct'].value_counts().reset_index()
            res_counts.columns = ['Kết quả', 'Số lượng']
            res_counts['Kết quả'] = res_counts['Kết quả'].map({True: 'ĐÚNG', False: 'SAI'})
            
            fig_acc = px.pie(res_counts, names='Kết quả', values='Số lượng', 
                            color='Kết quả', 
                            color_discrete_map={'ĐÚNG':'#28a745', 'SAI':'#dc3545'},
                            hole=0.4)
            fig_acc.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=200)
            st.plotly_chart(fig_acc, use_container_width=True)
            
            st.divider()

            # CHART PHÂN BỐ ACTION (KEEP/SKIP/UNPROCESSED)
            st.caption("Tỷ lệ Xử lý (Action)")
            if 'action' in df_test.columns:
                action_counts = df_test['action'].value_counts().reset_index()
                action_counts.columns = ['Hành động', 'Số lượng']
                
                # Map màu chuẩn
                color_map_action = {
                    "KEEP": "#28a745",       # Xanh
                    "SKIP": "#ffc107",       # Vàng
                    "UNPROCESSED": "#dc3545" # Đỏ
                }
                
                fig_action = px.pie(action_counts, names='Hành động', values='Số lượng',
                                    color='Hành động',
                                    color_discrete_map=color_map_action,
                                    hole=0.4)
                fig_action.update_layout(showlegend=True, margin=dict(t=0, b=0, l=0, r=0), height=200)
                st.plotly_chart(fig_action, use_container_width=True)

    if auto_refresh_tab3:
        time.sleep(15)
        st.rerun()