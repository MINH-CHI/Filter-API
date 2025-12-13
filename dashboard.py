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
    
    # 🔐 INPUT API KEY TẠI ĐÂY
    api_key = st.text_input("🔑 Nhập API Key", type="password", help="Nhập key từ file secrets_config.py")
    st.divider()
    st.header("📅 Bộ lọc thời gian")
    # Mặc định chọn 3 ngày gần nhất cho nhẹ
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
                        
                        # 🔐 THÊM HEADER AUTHENTICATION
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
        
    # --- GỌI HÀM VỚI THAM SỐ NGÀY TỪ SIDEBAR ---
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
    st.header("🧪 Đánh giá Hiệu năng Model (1000 Samples)")
    st.markdown("""
    Upload file kết quả từ script `batch_test.py` để phân tích độ tin cậy (Confidence) và các trường hợp sai sót.
    """)

    # 1. Nguồn dữ liệu: Tự tìm file hoặc Upload
    uploaded_file = st.file_uploader("Chọn file Excel kết quả (test_results_1000.xlsx)", type=['xlsx'])
    
    # Tự động tìm file nếu có sẵn ở server
    default_file = "test_results_1000.xlsx"
    df_batch = None
    
    if uploaded_file:
        df_batch = pd.read_excel(uploaded_file)
        st.success(f"Đã tải file: {uploaded_file.name}")
    elif os.path.exists(default_file):
        st.info(f"Đã tìm thấy file `{default_file}` trên server. Đang load...")
        df_batch = pd.read_excel(default_file)
    
    # 2. Hiển thị Dashboard phân tích
    if df_batch is not None:
        # --- CẤU HÌNH NGƯỠNG PASS ---
        col_conf1, col_conf2 = st.columns([1, 3])
        with col_conf1:
            threshold = st.slider("Ngưỡng Pass Confidence", 0.0, 1.0, 0.90, 0.05)
        
        # Thêm cột đánh giá dựa trên ngưỡng slider
        df_batch['Pass_Threshold'] = df_batch['confidence'] >= threshold
        
        # Lọc dữ liệu
        total_samples = len(df_batch)
        passed_samples = len(df_batch[df_batch['Pass_Threshold'] == True])
        failed_samples = total_samples - passed_samples
        pass_rate = (passed_samples / total_samples) * 100
        
        # KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tổng mẫu test", total_samples)
        k2.metric(f"Đạt chuẩn (Conf >= {threshold})", passed_samples)
        k3.metric("Dưới chuẩn (Cần review)", failed_samples, delta_color="inverse")
        k4.metric("Tỷ lệ Pass", f"{pass_rate:.1f}%")
        
        st.divider()
        
        # --- BIỂU ĐỒ 1: PHÂN PHỐI CONFIDENCE (QUAN TRỌNG NHẤT) ---
        st.subheader("1. Biểu đồ Phân phối Độ tin cậy (Confidence Distribution)")
        st.caption("Biểu đồ này cho biết Model đang 'tự tin' hay 'lưỡng lự'. Càng lệch về bên phải (1.0) càng tốt.")
        
        fig_hist = px.histogram(
            df_batch, 
            x="confidence", 
            color="type", # Phân màu theo Valid/Imbalance/Unknown
            nbins=50, 
            marginal="box", # Thêm biểu đồ box plot ở trên
            hover_data=df_batch.columns,
            color_discrete_map={"valid": "green", "imbalance": "orange", "unknown": "red"}
        )
        # Vẽ đường kẻ đỏ ngưỡng threshold
        fig_hist.add_vline(x=threshold, line_width=3, line_dash="dash", line_color="red")
        st.plotly_chart(fig_hist, use_container_width=True)

        # --- BIỂU ĐỒ 2: CHI TIẾT THEO LOẠI ---
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("2. Tỷ lệ Pass theo nhóm dữ liệu")
            # Group by Type và tính tỷ lệ pass
            pass_by_type = df_batch.groupby('type')['Pass_Threshold'].mean().reset_index()
            pass_by_type['Pass_Threshold'] = pass_by_type['Pass_Threshold'] * 100
            
            fig_bar = px.bar(
                pass_by_type, x='type', y='Pass_Threshold', 
                color='type', 
                text_auto='.1f',
                title="Tỷ lệ đạt chuẩn (%) theo từng loại dữ liệu"
            )
            fig_bar.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c2:
            st.subheader("3. Scatter Plot: Confidence vs. Labels")
            # Giúp nhìn nhanh class nào hay bị điểm thấp
            fig_scatter = px.scatter(
                df_batch, x="predicted_label", y="confidence", color="type",
                hover_data=['filename', 'actual_label'],
                title="Độ tin cậy của từng Class dự đoán"
            )
            fig_scatter.add_hline(y=threshold, line_dash="dash", line_color="red")
            st.plotly_chart(fig_scatter, use_container_width=True)

        # --- DANH SÁCH CẦN REVIEW (Failed Cases) ---
        st.subheader("⚠️ Danh sách các ca cần đánh giá lại (Fail Cases)")
        st.write(f"Dưới đây là các ảnh có Confidence < {threshold}. Bạn hãy kiểm tra xem tại sao.")
        
        # Lọc ra các ca fail
        failed_df = df_batch[df_batch['Pass_Threshold'] == False].sort_values(by="confidence")
        
        # Hiển thị bảng tương tác
        st.dataframe(
            failed_df[['filename', 'type', 'actual_label', 'predicted_label', 'confidence']], 
            use_container_width=True
        )
        
        with st.expander("💡 Gợi ý xử lý"):
            st.markdown("""
            * **Nếu Type = 'imbalance' và Conf thấp:** Model chưa học đủ góc độ này -> **Gửi Team AI train thêm.**
            * **Nếu Type = 'valid' và Conf thấp:** Ảnh có thể bị mờ, nhiễu hoặc Model nhận diện kém -> **Cần kiểm tra kỹ.**
            * **Nếu Type = 'unknown' mà Conf CAO (False Positive):** Nguy hiểm! Model đang nhận nhầm rác thành vật thể -> **Cần chỉnh lại Threshold hoặc train thêm class background.**
            """)

    else:
        st.warning("⚠️ Chưa có dữ liệu. Hãy chạy script `batch_test.py` trước, sau đó upload file Excel vào đây.")