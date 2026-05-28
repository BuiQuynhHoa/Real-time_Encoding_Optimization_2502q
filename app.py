import streamlit as st

# =====================================================================
# CẤU HÌNH TRANG & GIAO DIỆN CHUNG
# =====================================================================
st.set_page_config(
    page_title="Real-Time Encoding Optimization | 2502q",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p { color: #cce0ff; margin: 0.3rem 0 0; font-size: 0.95rem; }
    div[data-testid="metric-container"] {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>Real-Time Encoding Optimization</h1>
    <p>Môn: Nén và Mã hóa Đa phương tiện &nbsp;|&nbsp; Mã code: 2502q &nbsp; &nbsp; Nhóm: Quỳnh Hoa & Phương Nga</p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# THANH ĐIỀU HƯỚNG TABS
# =====================================================================
tab1, tab2, tab3 = st.tabs([
    "Module 1 - File Encoding",
    "Module 2 - Live Webcam",
    "Hướng dẫn & Về dự án",
])

# =====================================================================
# TAB 1: MODULE 1 
# =====================================================================
with tab1:
    try:
        from module1_file_encoding import render as render_module1
        render_module1()
    except ImportError as e:
        st.error("Chưa import được Module 1 (Quỳnh Hoa chưa hoàn thiện file module1_file_encoding.py).")
    except Exception as e:
        st.error(f"Lỗi Module 1: {e}")

# =====================================================================
# TAB 2: MODULE 2 
# =====================================================================
with tab2:
    try:
        from module2_webcam_live import render as render_module2
        render_module2()
    except Exception as e:
        st.error(f"Lỗi Module 2: {e}")
        st.exception(e)

# =====================================================================
# TAB 3: THÔNG TIN DỰ ÁN
# =====================================================================
with tab3:
    st.header("Hướng dẫn Sử dụng & Về Dự án")
    col_about, col_guide = st.columns(2)
    with col_about:
        st.subheader("Về dự án")
        st.markdown("""
        **Project:** Real-Time Encoding Optimization
        **Công nghệ sử dụng:** FFmpeg + libx264, OpenCV, Streamlit, Plotly, Python threading.
        """)
        st.subheader("Đóng góp nhóm")
        st.markdown("""
        | Thành viên | Module | Nhiệm vụ chính |
        |---|---|---|
        | **Quỳnh Hoa** | Module 1 | File encoding, benchmark, biểu đồ, báo cáo lý thuyết |
        | **Phương Nga** | Module 2 | Webcam live, threading, dashboard, README, system design |
        """)