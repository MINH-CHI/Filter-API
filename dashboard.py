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
import batch_test
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
    st.header("🧪 Đánh giá Hiệu năng Model")
    
    # Chọn chế độ: Upload file cũ hay Chạy Live mới
    mode = st.radio("Chọn chế độ:", ["📂 Phân tích file Excel cũ", "🚀 Chạy Test Live từ Google Drive"], horizontal=True)

    if mode == "📂 Phân tích file Excel cũ":
        uploaded_file = st.file_uploader("Upload test_results.xlsx", type=['xlsx'])
        default_file = "test_results_1000.xlsx"
        df_batch = None
        if uploaded_file:
            df_batch = pd.read_excel(uploaded_file)
            st.dataframe(df_batch.head())
        elif os.path.exists(default_file):
            st.info(f"Đã tìm thấy file mặc định `{default_file}`.")
            df_batch = pd.read_excel(default_file)
            
        if df_batch is not None:
            col_conf1, col_conf2 = st.columns([1, 3])
            with col_conf1:
                threshold = st.slider("Ngưỡng Pass Confidence", 0.0, 1.0, 0.90, 0.05)
            
            df_batch['Pass_Threshold'] = df_batch['confidence'] >= threshold
            
            total_samples = len(df_batch)
            passed_samples = len(df_batch[df_batch['Pass_Threshold'] == True])
            failed_samples = total_samples - passed_samples
            pass_rate = (passed_samples / total_samples) * 100
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Tổng mẫu test", total_samples)
            k2.metric(f"Đạt chuẩn (Conf >= {threshold})", passed_samples)
            k3.metric("Dưới chuẩn", failed_samples, delta_color="inverse")
            k4.metric("Tỷ lệ Pass", f"{pass_rate:.1f}%")
            
            st.divider()
            
            st.subheader("1. Biểu đồ Phân phối Độ tin cậy")
            fig_hist = px.histogram(
                df_batch, x="confidence", color="type", nbins=50, marginal="box",
                hover_data=df_batch.columns,
                color_discrete_map={"valid": "green", "imbalance": "orange", "unknown": "red"}
            )
            fig_hist.add_vline(x=threshold, line_width=3, line_dash="dash", line_color="red")
            st.plotly_chart(fig_hist, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("2. Tỷ lệ Pass theo nhóm")
                pass_by_type = df_batch.groupby('type')['Pass_Threshold'].mean().reset_index()
                pass_by_type['Pass_Threshold'] = pass_by_type['Pass_Threshold'] * 100
                fig_bar = px.bar(pass_by_type, x='type', y='Pass_Threshold', color='type', text_auto='.1f')
                fig_bar.update_yaxes(range=[0, 100])
                st.plotly_chart(fig_bar, use_container_width=True)
                
            with c2:
                st.subheader("3. Scatter Plot: Confidence vs Labels")
                fig_scatter = px.scatter(
                    df_batch, x="predicted_label", y="confidence", color="type",
                    hover_data=['filename', 'actual_label']
                )
                fig_scatter.add_hline(y=threshold, line_dash="dash", line_color="red")
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.subheader("⚠️ Danh sách Fail Cases")
            failed_df = df_batch[df_batch['Pass_Threshold'] == False].sort_values(by="confidence")
            st.dataframe(failed_df[['filename', 'type', 'actual_label', 'predicted_label', 'confidence']], use_container_width=True)
            
            with st.expander("💡 Gợi ý xử lý"):
                st.markdown("""
                * **Imbalance & Conf thấp:** Train thêm góc độ này.
                * **Valid & Conf thấp:** Kiểm tra chất lượng ảnh.
                * **Unknown & Conf cao:** Coi chừng False Positive.
                """)
        else:
            st.warning("⚠️ Chưa có dữ liệu Excel để phân tích.")

    elif mode == "🚀 Chạy Test Live từ Google Drive":
        st.info("Chế độ này sẽ kết nối Google Drive, tải ảnh và gửi lên API theo thời gian thực.")
        
        # Session State để lưu kết quả Live
        if "live_results" not in st.session_state:
            st.session_state.live_results = []
        if "is_testing" not in st.session_state:
            st.session_state.is_testing = False

        col_btn, col_metric = st.columns([1, 4])
        
        with col_btn:
            if st.button("▶️ BẮT ĐẦU TEST", type="primary", disabled=st.session_state.is_testing):
                st.session_state.is_testing = True
                st.session_state.live_results = [] # Reset
                st.rerun()

        # Hiển thị kết quả Real-time
        placeholder_bar = st.empty()
        placeholder_status = st.empty()
        placeholder_df = st.empty()

        # Logic chạy Test
        if st.session_state.is_testing:
            # Gọi hàm từ batch_test để lấy service
            service = batch_test.get_drive_service()
            
            if not service:
                st.error("❌ Không thể kết nối Google Drive. Kiểm tra file `token.json` hoặc `client_secrets.json`.")
                st.session_state.is_testing = False
            else:
                with st.spinner("Đang quét danh sách ảnh từ Drive..."):
                    # Gọi hàm từ batch_test để lấy danh sách file
                    tasks = batch_test.build_task_list(service)
                
                if not tasks:
                    st.warning("⚠️ Không tìm thấy ảnh nào trong folder quy định.")
                    st.session_state.is_testing = False
                else:
                    placeholder_status.info(f"🚀 Tìm thấy {len(tasks)} ảnh. Đang xử lý...")
                    progress_bar = placeholder_bar.progress(0)

                    # Xử lý hình ảnh
                    for i, task in enumerate(tasks):
                        # Gọi hàm xử lý từng task từ module riêng
                        result = batch_test.process_single_task(
                            service=service, 
                            task=task, 
                            api_key=api_key, 
                            api_url=API_URL
                        )
                        
                        # Cập nhật kết quả vào Session State
                        st.session_state.live_results.append(result)
                        
                        # Cập nhật UI
                        df_live = pd.DataFrame(st.session_state.live_results)
                        placeholder_df.dataframe(df_live, height=400, use_container_width=True)
                        progress_bar.progress((i + 1) / len(tasks))
                        
                        # Sleep nhẹ để không spam server quá gắt
                        time.sleep(0.1) 

                    st.success("✅ Đã hoàn thành Batch Test!")
                    st.session_state.is_testing = False
                    
                    # Nút tải xuống kết quả
                    if st.session_state.live_results:
                        df_final = pd.DataFrame(st.session_state.live_results)
                        csv = df_final.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Tải kết quả CSV", csv, "live_test_results.csv", "text/csv")

        # Hiển thị lại bảng nếu đã chạy xong (để không bị mất khi thao tác khác)
        elif st.session_state.live_results:
            df_live = pd.DataFrame(st.session_state.live_results)
            st.dataframe(df_live, height=400, use_container_width=True)
            
            # Tính toán nhanh Accuracy
            if "Is Correct" in df_live.columns:
                valid = df_live[df_live["Status"] == "Success"]
                if not valid.empty:
                    acc = valid["Is Correct"].mean() * 100
                    st.metric("Độ chính xác hiện tại", f"{acc:.2f}%", f"{len(valid)} mẫu")