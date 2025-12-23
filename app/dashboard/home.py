import streamlit as st #type:ignore
import requests #type:ignore
import pandas as pd #type:ignore
import plotly.express as px #type:ignore
import pymongo #type:ignore
import os
import sys
dashboard_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(dashboard_dir)
project_root = os.path.dirname(app_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
env_path = os.path.join(project_root, ".env")
from dotenv import load_dotenv #type: ignore
from PIL import Image #type:ignore
import time
from datetime import datetime, timedelta, time as dt_time
load_dotenv(env_path)
st.set_page_config(page_title="AI Image Filter Dashboard", layout="wide", page_icon="🕵️")

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "api_request_log"
COLLECTION_NAME = "api_unlabeled_images" 
CONFIG_COLLECTION = "system_config"
@st.cache_resource
def init_mongo_client():
    """Khởi tạo kết nối MongoDB và cache lại để dùng chung."""
    if not MONGO_URI:
        return None
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.server_info()  # Trigger kiểm tra kết nối
        return client
    except Exception as e:
        st.toast(f"❌ Lỗi kết nối MongoDB: {e}", icon="🔥")
        return None

def get_api_url_from_mongo():
    """Lấy API URL mới nhất từ MongoDB"""
    try:
        client = init_mongo_client()
        db = client[DB_NAME]
        coll = db[CONFIG_COLLECTION]
        
        doc = coll.find_one({"config_key": "active_api_url"})
        if doc and "value" in doc:
            return doc["value"]
    except Exception as e :
        print(f"Có lỗi {repr(e)}")
        pass
    return None

cloud_url = get_api_url_from_mongo()
BASE_URL = ""
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
        auto_refresh_tab2 = st.toggle("🔴 Live (5s)", value=False)
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
    if auto_refresh_tab2:
        time.sleep(13)
        st.rerun()
with tab3:
    st.header("🧪 Giám sát Batch Test (Real-time)")
    st.markdown("""
    > **Trạng thái:** Hiển thị kết quả từ `batch_test.py`.
    > **Cập nhật:** Đã hiển thị cột **Bounding Box**.
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
        
        # Accuracy
        correct_count = df_test['is_correct'].sum()
        acc_val = (correct_count / total_test * 100) if total_test > 0 else 0.0
        
        # Keep Rate
        keep_count = len(df_test[df_test['action'] == 'KEEP'])
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Số mẫu đã Test", total_test)
        k2.metric("Độ chính xác (Accuracy)", f"{acc_val:.1f}%")
        k3.metric("Số ảnh Hợp lệ (KEEP)", keep_count)
        k4.metric("Trạng thái mới nhất", df_test.iloc[0]['status'] if 'status' in df_test.columns else "N/A")

        st.divider()

        # Chia cột: Bảng chiếm 70%, Biểu đồ tròn chiếm 30%
        c1, c2 = st.columns([7, 3])
        
        with c1:
            st.subheader("📋 Chi tiết từng ảnh")
            
            # Hàm tô màu
            def highlight_correct(val):
                return f'background-color: {"#d4edda" if val else "#f8d7da"}' # Xanh/Đỏ nhạt

            display_cols = ['timestamp', 'filename', 'actual_label', 'predicted_label', 'bounding_box', 'confidence', 'action', 'is_correct']
            
            # Format lại DataFrame
            df_display = df_test[[c for c in display_cols if c in df_test.columns]].copy()
            
            st.dataframe(
                df_display.style.applymap(highlight_correct, subset=['is_correct']),
                use_container_width=True,
                height=500
            )

        with c2:
            st.subheader("📊 Tỷ lệ Chính xác")
            
            # Hiển thị Pie Chart Accuracy (Đúng/Sai)
            res_counts = df_test['is_correct'].value_counts().reset_index()
            res_counts.columns = ['Kết quả', 'Số lượng']
            res_counts['Kết quả'] = res_counts['Kết quả'].map({True: 'ĐÚNG', False: 'SAI'})
            
            fig_acc = px.pie(res_counts, names='Kết quả', values='Số lượng', 
                           color='Kết quả', 
                           color_discrete_map={'ĐÚNG':'#28a745', 'SAI':'#dc3545'},
                           hole=0.4)
            st.plotly_chart(fig_acc, use_container_width=True)

    if auto_refresh_tab3:
        time.sleep(15) # Refresh
        st.rerun()