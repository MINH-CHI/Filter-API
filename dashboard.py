import streamlit as st #type:ignore
import requests #type:ignore
import pandas as pd #type:ignore
import plotly.express as px #type:ignore
import pymongo #type:ignore
import os
import time
from PIL import Image #type:ignore

st.set_page_config(page_title="AI Image Filter Dashboard", layout="wide", page_icon="🕵️")

# Cấu hình kết nối API local
# API_URL = "http://localhost:8000/v1/filter"

default_api_url = "http://api:8000/v1/filter"
API_URL = os.getenv("API_URL", "http://localhost:8000/v1/filter")
# Cấu hình kết nối MongoDB (Cho Tab Thống kê)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:password123@localhost:27017")
DB_NAME = "api_request_log"
COLLECTION_NAME = "consumer_logs" 

@st.cache_resource
def init_mongo_connection():
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.server_info() # Check kết nối
        return client
    except Exception:
        return None

def load_logs():
    client = init_mongo_connection()
    if not client:
        return None
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    # Lấy 1000 record mới nhất
    data = list(collection.find().sort("timestamp", -1).limit(1000))
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

st.title("🕵️ Hệ thống Kiểm soát & Lọc Ảnh AI")


tab1, tab2 = st.tabs(["🚀 Dùng thử (Demo)", "📊 Thống kê (Analytics)"])

with tab1:
    st.header("Test Model AI (Gọi API)")
    st.write("Upload ảnh để kiểm tra xem AI có nhận diện đúng không.")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Chọn ảnh (JPG, PNG)", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_file:
            # Hiển thị ảnh vừa chọn
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh gốc", use_container_width=True)
            
            # Nút gọi API
            if st.button("🔍 Quét ngay", type="primary"):
                with st.spinner('Đang gửi sang API xử lý...'):
                    try:
                        # Reset con trỏ file về đầu để đọc bytes
                        uploaded_file.seek(0)
                        files = {'file': uploaded_file}
                        data = {'source': 'streamlit_demo'}
                        
                        # GỌI API
                        response = requests.post(API_URL, files=files, data=data)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Hiển thị kết quả bên cột 2
                            with col2:
                                st.subheader("Kết quả từ API:")
                                if result['action'] == 'KEEP':
                                    st.success(f"✅ HỢP LỆ (KEEP)")
                                    st.balloons()
                                else:
                                    st.error(f"❌ LOẠI BỎ (DISCARD)")
                                
                                st.write("**Vật thể phát hiện:**")
                                st.write(result.get('detected_labels', []))
                                
                                with st.expander("Xem JSON thô"):
                                    st.json(result)
                        else:
                            st.error(f"Lỗi API ({response.status_code}): {response.text}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("⚠️ Không thể kết nối tới API! Bạn đã bật server 'main.py' chưa?")
                    except Exception as e:
                        st.error(f"Lỗi: {e}")

with tab2:
    st.header("Thống kê dữ liệu Log")
    col_control_1, col_control_2 = st.columns([1, 4])
    
    with col_control_1:
        # Nút gạt bật tắt chế độ tự động
        auto_refresh = st.toggle("🔴 Chế độ Live (5s)", value=False)
        
    with col_control_2:
        if st.button("🔄 Làm mới ngay lập tức"):
            st.rerun()
        
    df = load_logs()
    
    if df is None:
        st.warning("⚠️ Không thể kết nối MongoDB. Hãy kiểm tra lại chuỗi kết nối MONGO_URI trong code.")
    elif df.empty:
        st.info("Chưa có dữ liệu log nào trong Database.")
    else:
        # KPI Cards
        total = len(df)
        kept = len(df[df['status'] == 'KEEP']) if 'status' in df.columns else 0
        keep_rate = (kept/total * 100) if total > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng ảnh đã quét", total)
        c2.metric("Ảnh hợp lệ (KEEP)", kept)
        c3.metric("Tỷ lệ đạt chuẩn", f"{keep_rate:.1f}%")
        
        st.divider()
        
        # Biểu đồ
        chart1, chart2 = st.columns(2)
        
        with chart1:
            if 'status' in df.columns:
                st.subheader("Tỷ lệ Sàng lọc")
                fig_pie = px.pie(df, names='status', color='status', 
                                color_discrete_map={'KEEP':'green', 'DISCARD':'red'})
                st.plotly_chart(fig_pie, use_container_width=True)
                
        with chart2:
            if 'detected_classes' in df.columns:
                st.subheader("Top vật thể phát hiện")
                exploded_df = df.explode('detected_classes').dropna(subset=['detected_classes'])
                if not exploded_df.empty:
                    top_classes = exploded_df['detected_classes'].value_counts().head(10).reset_index()
                    top_classes.columns = ['Class', 'Count']
                    fig_bar = px.bar(top_classes, x='Count', y='Class', orientation='h', text_auto=True)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.write("Chưa có dữ liệu vật thể.")
                    
        # Bảng dữ liệu
        st.subheader("Lịch sử chi tiết")
        st.dataframe(df, use_container_width=True)
    if auto_refresh:
        time.sleep(5) # Đợi 5 giây
        st.rerun()