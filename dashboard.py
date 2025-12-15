import streamlit as st #type:ignore
import requests #type:ignore
import pandas as pd #type:ignore
import plotly.express as px #type:ignore
import pymongo #type:ignore
import os
from dotenv import load_dotenv #type: ignore
import time
from PIL import Image #type:ignore
from datetime import datetime, timedelta, time
load_dotenv()
st.set_page_config(page_title="AI Image Filter Dashboard", layout="wide", page_icon="🕵️")

# Cấu hình kết nối API local
# API_URL = "http://localhost:8000/v1/filter"

# default_api_url = "http://api:8000/v1/filter"
# API_URL = os.getenv("API_URL", "http://localhost:8000/v1/filter")
# Cấu hình kết nối MongoDB (Cho Tab Thống kê)
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "api_request_log"
COLLECTION_NAME = "api_unlabeled_images" 
CONFIG_COLLECTION = "system_config"
@st.cache_data(ttl=60) # Cache 60 giây để đỡ gọi DB nhiều
def get_api_url_from_mongo():
    """Lấy API URL mới nhất từ MongoDB"""
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        db = client[DB_NAME]
        coll = db[CONFIG_COLLECTION]
        
        doc = coll.find_one({"config_key": "active_api_url"})
        if doc and "value" in doc:
            return doc["value"]
    except Exception:
        pass
    return None
cloud_url = get_api_url_from_mongo()

if cloud_url:
    BASE_URL = cloud_url
    st.sidebar.success(f"🟢 Đã kết nối API: {BASE_URL.split('//')[1]}")
else:
    # Cấu hình mặc định hoặc Local
    BASE_URL = "http://localhost:8000"
    st.sidebar.warning("⚠️ Không tìm thấy URL từ Mongo, đang dùng Default.")

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
@st.cache_resource
def init_mongo_connection():
    try:
        if not MONGO_URI:
            return None
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.server_info() # Check kết nối
        return client
    except Exception as e:
        print(f"Mongo Error: {e}")
        return None

def load_logs(start_date, end_date):
    client = init_mongo_connection()
    if not client:
        return None
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    start_dt = datetime.combine(start_date, time.min) 
    end_dt = datetime.combine(end_date, time.max)
    query = {
        "timestamp": {
            "$gte": start_dt, # Greater than or equal (Lớn hơn hoặc bằng)
            "$lte": end_dt    # Less than or equal (Nhỏ hơn hoặc bằng)
        }
    }
    # Lấy 1000 record mới nhất
    data = list(collection.find().sort("timestamp", -1))
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Chuẩn hóa thời gian
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    return df

st.title("🕵️ Hệ thống Kiểm soát & Lọc Ảnh AI")
if not api_key:
    st.warning("⚠️ Vui lòng nhập **API Key** ở thanh bên trái (Sidebar) để bắt đầu sử dụng.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🚀 Demo Lọc Ảnh", "📊 Giám sát Live", "🧪 Phân tích Batch Test"])

with tab1:
    st.header("Test Model AI")
    st.write("Upload ảnh để kiểm tra xem AI nhận diện và bộ lọc hoạt động như thế nào.")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Chọn ảnh (JPG, PNG)", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_file:
            # Hiển thị ảnh
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh gốc", use_container_width=True)
            
            # Nút gọi API
            if st.button("🔍 Quét ngay", type="primary"):
                with st.spinner('Đang gửi request kèm API Key...'):
                    try:
                        # Reset file pointer
                        uploaded_file.seek(0)
                        
                        files = {'file': uploaded_file}
                        data = {'source': 'streamlit_dashboard'}
                        
                        # Headers với API Key
                        headers = {'x-api-key': api_key}
                        
                        # GỌI API
                        response = requests.post(API_URL, files=files, data=data, headers=headers)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            with col2:
                                st.subheader("Kết quả AI:")
                                
                                # Logic hiển thị mới dựa trên 'action'
                                action = result.get('action', 'UNKNOWN')
                                
                                if action == 'KEEP':
                                    st.success(f"✅ HỢP LỆ (KEEP)")
                                    st.balloons()
                                elif action == 'DISCARD':
                                    st.error(f"❌ LOẠI BỎ (DISCARD)")
                                else:
                                    st.warning(f"⚠️ {action}")
                                
                                st.write(f"**Người dùng:** {result.get('user', 'Unknown')}")
                                st.write("**Kết quả chi tiết:**")
                                
                                detections = result.get('detections', [])
                                
                                if detections:
                                    # Nếu có thông tin confidence
                                    for item in detections:
                                        name = item.get('object', 'Unknown')
                                        conf = item.get('confidence', 0)
                                        st.write(f"- 🎯 **{name}**: `{conf * 100:.1f}%`")
                                        st.progress(conf) 
                                else:
                                    st.write(result.get('detected_labels', []))
                                
                                with st.expander("Xem JSON phản hồi"):
                                    st.json(result)
                                    
                        elif response.status_code == 403:
                            st.error("⛔ BỊ TỪ CHỐI! API Key không đúng hoặc không có quyền.")
                        else:
                            st.error(f"Lỗi API ({response.status_code}): {response.text}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("⚠️ Không thể kết nối tới API! Server có đang bật không?")
                    except Exception as e:
                        st.error(f"Lỗi không xác định: {e}")

with tab2:
    st.header("Thống kê dữ liệu Log")
    
    col_ctrl1, col_ctrl2 = st.columns([1, 4])
    with col_ctrl1:
        auto_refresh = st.toggle("🔴 Live (5s)", value=False)
    with col_ctrl2:
        if st.button("🔄 Làm mới"): st.rerun()
        
    df = load_logs(start_date, end_date)
    
    if df is None:
        st.error("❌ Lỗi kết nối MongoDB")
    elif df.empty:
        st.info(f"📭 Không có dữ liệu nào từ ngày {start_date} đến {end_date}.")
    else:
        # Chuẩn hóa cột
        for col in ['action', 'detected_labels', 'user']:
            if col not in df.columns: df[col] = None
            
        # Metrics
        total = len(df)
        kept = len(df[df['action'] == 'KEEP'])
        discarded = len(df[df['action'] == 'DISCARD'])
        rate = (kept/total*100) if total else 0
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng Request (Range)", total)
        m2.metric("✅ Clean", kept)
        m3.metric("🗑️ Spam", discarded)
        m4.metric("Tỷ lệ sạch", f"{rate:.1f}%")
        
        st.divider()
        
        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Tỷ lệ Lọc")
            fig = px.pie(df, names='action', color='action', 
                         color_discrete_map={'KEEP':'green', 'DISCARD':'red', 'UNKNOWN':'gray'})
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("Top Users")
            if 'user' in df.columns:
                u_counts = df['user'].value_counts().reset_index()
                u_counts.columns = ['User', 'Count']
                fig_u = px.bar(u_counts, x='User', y='Count', color='User', text_auto=True)
                st.plotly_chart(fig_u, use_container_width=True)

        # Top Objects
        st.subheader("🔍 Top Vật thể phát hiện")
        exploded = df.explode('detected_labels').dropna(subset=['detected_labels'])
        if not exploded.empty:
            top_obj = exploded['detected_labels'].value_counts().head(15).reset_index()
            top_obj.columns = ['Object', 'Count']
            fig_bar = px.bar(top_obj, x='Count', y='Object', orientation='h', 
                             text_auto=True, color='Count')
            fig_bar.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # Dataframe
        st.subheader("📄 Chi tiết Log")
        display_cols = ['timestamp', 'user', 'filename', 'action', 'detected_labels']
        st.dataframe(df[[c for c in display_cols if c in df.columns]], use_container_width=True)
    if auto_refresh:
        time.sleep(5) # Đợi 5 giây
        st.rerun()
with tab3:
    st.header("🧪 Giám sát Batch Test (Real-time)")
    st.info("💡 Hướng dẫn: Chạy script `python batch_test.py` ở terminal. Dashboard này sẽ tự cập nhật kết quả tại đây.")

    # Nút refresh thủ công và Auto-refresh
    col_re1, col_re2 = st.columns([1, 5])
    with col_re1:
        auto_refresh = st.toggle("🔴 Auto-Refresh (2s)", value=True)
    with col_re2:
        if st.button("🔄 Làm mới ngay"): st.rerun()

    # Load dữ liệu từ Mongo
    client = init_mongo_connection()
    
    # Lấy tất cả kết quả test mới nhất
    # (Có thể thêm limit nếu dữ liệu quá lớn)
    data = list(client[DB_NAME][COLLECTION_NAME].find().sort("timestamp", -1))
    
    df_live = pd.DataFrame(data)

    if df_live.empty:
        st.warning("⚠️ Chưa có dữ liệu test nào đang chạy. Hãy chạy script `batch_test.py`.")
    else:
        # Tính toán Metrics
        total = len(df_live)
        processed = len(df_live[df_live['status'] == 'Done'])
        
        # Metrics Accuracy
        if 'is_correct' in df_live.columns:
            valid_df = df_live[df_live['status'] == 'Done']
            correct = valid_df['is_correct'].sum()
            acc = (correct / len(valid_df) * 100) if not valid_df.empty else 0.0
        else:
            acc = 0.0

        # Hiển thị KPI
        k1, k2, k3 = st.columns(3)
        k1.metric("Số lượng ảnh đã xử lý", f"{total}", delta="Live Update")
        k2.metric("Độ chính xác (Accuracy)", f"{acc:.2f}%")
        k3.metric("Trạng thái mới nhất", df_live.iloc[0]['status'] if not df_live.empty else "N/A")

        # Hiển thị thanh tiến trình (Giả lập hoặc tính nếu biết tổng)
        st.progress(min(processed % 100, 100) / 100) 

        # Layout 2 cột: Bảng dữ liệu & Biểu đồ
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("📋 Bảng kết quả chi tiết")
            display_cols = ['timestamp', 'filename', 'actual_label', 'predicted_label', 'confidence', 'is_correct']
            # Hiển thị cột có sẵn
            cols_to_show = [c for c in display_cols if c in df_live.columns]
            st.dataframe(df_live[cols_to_show], height=400, use_container_width=True)

        with c2:
            st.subheader("📊 Biểu đồ nhanh")
            if 'is_correct' in df_live.columns and not df_live.empty:
                counts = df_live['is_correct'].value_counts().reset_index()
                counts.columns = ['Result', 'Count']
                counts['Result'] = counts['Result'].map({True: 'Đúng', False: 'Sai'})
                
                fig = px.pie(counts, names='Result', values='Count', 
                             color='Result', color_discrete_map={'Đúng':'green', 'Sai':'red'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plot Confidence
                if 'confidence' in df_live.columns:
                    fig2 = px.scatter(df_live, x='predicted_label', y='confidence', 
                                      color='is_correct', title="Confidence Distribution")
                    st.plotly_chart(fig2, use_container_width=True)

    # Logic Auto Refresh
    if auto_refresh:
        time.sleep(2) # Refresh mỗi 2 giây
        st.rerun()