import os
import tempfile
import threading
import time
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    H264_PRESETS,
    PRESET_TO_JPEG_QUALITY,
    check_ffmpeg,
    encode_video,
    get_video_info,
    simulate_frame_compression,
)

# =====================================================================
# THREADING: WEBCAM CAPTURE WORKER
# =====================================================================
class WebcamWorker:
    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)     
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.frame = None
        self.lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        if not self.cap.isOpened():
            raise RuntimeError("Không thể mở webcam! Kiểm tra kết nối camera.")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap.isOpened():
            self.cap.release()

    def get_frame(self) -> np.ndarray | None:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def _capture_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

# =====================================================================
# KHỞI TẠO SESSION STATE
# =====================================================================
def _init_session():
    defaults = {
        'm2_running': False,
        'm2_frame_count': 0,
        'm2_start_time': 0.0,
        'm2_fps_history': [],
        'm2_lat_history': [],
        'm2_size_history': [],
        'm2_worker': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# =====================================================================
# HÀM RENDER CHÍNH (GIAO DIỆN STREAMLIT)
# =====================================================================
def render():
    _init_session()
    
    st.header("Module 2 - Webcam Live & Real-Time Dashboard")
    st.markdown("""
    - **Video gốc** vs **Video đã nén** hiển thị song song
    - **FPS** và **Latency** cập nhật theo thời gian thực
    - **Preset** thay đổi ngay lập tức → quan sát artifact xuất hiện
    """)
    st.divider()

    # --- PHẦN 1: LIVE WEBCAM MÔ PHỎNG ---
    st.subheader("Phần 1 – Live Webcam với Mô phỏng Preset")
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        camera_index = st.number_input("Index Camera", min_value=0, max_value=5, value=0)
    with col_cfg2:
        live_preset = st.selectbox("Preset H.264 (mô phỏng)", options=H264_PRESETS, index=H264_PRESETS.index('medium'))
        st.session_state['m2_preset'] = live_preset
    with col_cfg3:
        jpeg_q = PRESET_TO_JPEG_QUALITY.get(live_preset, 70)
        st.metric("JPEG Quality tương ứng", f"{jpeg_q}/100", 
                  delta="chất lượng tốt" if jpeg_q >= 70 else "chất lượng thấp", 
                  delta_color="normal" if jpeg_q >= 70 else "inverse")

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col1:
        start_btn = st.button("Bắt đầu Webcam", disabled=st.session_state['m2_running'], type="primary", use_container_width=True)
    with btn_col2:
        stop_btn = st.button("Dừng", disabled=not st.session_state['m2_running'], use_container_width=True)
    with btn_col3:
        if st.session_state['m2_running']:
            st.success("LIVE - Webcam đang chạy")
        else:
            st.info("Nhấn 'Bắt đầu Webcam' để bắt đầu")

    if start_btn:
        try:
            worker = WebcamWorker(camera_index=int(camera_index))
            worker.start()
            st.session_state['m2_worker'] = worker
            st.session_state['m2_running'] = True
            st.session_state['m2_frame_count'] = 0
            st.session_state['m2_start_time'] = time.time()
            st.session_state['m2_fps_history'] = []
            st.session_state['m2_lat_history'] = []
            st.session_state['m2_size_history'] = []
            st.rerun()
        except Exception as e:
            st.error(f"Không mở được webcam: {e}")

    if stop_btn:
        if st.session_state['m2_worker']:
            st.session_state['m2_worker'].stop()
            st.session_state['m2_worker'] = None
        st.session_state['m2_running'] = False
        st.rerun()

    vid_col1, vid_col2 = st.columns(2)
    orig_ph = vid_col1.empty()
    enc_ph = vid_col2.empty()

    st.markdown("---")
    metrics_row = st.columns(4)
    fps_ph = metrics_row[0].empty()
    lat_ph = metrics_row[1].empty()
    size_ph = metrics_row[2].empty()
    psnr_ph = metrics_row[3].empty()
    chart_ph = st.empty()

    if st.session_state['m2_running']:
        worker = st.session_state['m2_worker']
        if worker is None:
            st.session_state['m2_running'] = False
            st.rerun()
            return

        frame = worker.get_frame()
        if frame is not None:
            preset = st.session_state.get('m2_preset', 'medium')
            t_encode_start = time.time()
            enc_frame, size_kb, psnr_est = simulate_frame_compression(frame, preset)
            latency_ms = (time.time() - t_encode_start) * 1000

            st.session_state['m2_frame_count'] += 1
            elapsed = time.time() - st.session_state['m2_start_time']
            fps = st.session_state['m2_frame_count'] / elapsed if elapsed > 0 else 0

            MAX_HIST = 60
            st.session_state['m2_fps_history'].append(round(fps, 1))
            st.session_state['m2_lat_history'].append(round(latency_ms, 1))
            st.session_state['m2_size_history'].append(round(size_kb, 1))

            for key in ('m2_fps_history', 'm2_lat_history', 'm2_size_history'):
                if len(st.session_state[key]) > MAX_HIST:
                    st.session_state[key] = st.session_state[key][-MAX_HIST:]

            orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            enc_rgb = cv2.cvtColor(enc_frame, cv2.COLOR_BGR2RGB)

            orig_ph.image(orig_rgb, caption="Video gốc (không nén)", use_container_width=True)
            enc_ph.image(enc_rgb, caption=f"Sau nén - Preset: {preset.upper()}", use_container_width=True)

            fps_ph.metric("FPS", f"{fps:.1f}", delta=f"+{fps - 24:.1f}" if fps > 24 else f"{fps - 24:.1f}", delta_color="normal")
            lat_ph.metric("Latency", f"{latency_ms:.1f} ms", delta="real-time" if latency_ms < 33 else "lag", delta_color="normal" if latency_ms < 33 else "inverse")
            size_ph.metric("Frame Size", f"{size_kb:.1f} KB")
            psnr_ph.metric("PSNR Est.", f"~{psnr_est:.0f} dB")

            if len(st.session_state['m2_fps_history']) > 5:
                hist_df = pd.DataFrame({
                    'Frame': range(len(st.session_state['m2_fps_history'])),
                    'FPS': st.session_state['m2_fps_history'],
                    'Latency (ms)': st.session_state['m2_lat_history'],
                })
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist_df['Frame'], y=hist_df['FPS'], name='FPS', mode='lines', line=dict(color='limegreen', width=1.5)))
                fig.add_trace(go.Scatter(x=hist_df['Frame'], y=hist_df['Latency (ms)'], name='Latency (ms)', mode='lines', line=dict(color='tomato', width=1.5, dash='dot'), yaxis='y2'))
                fig.update_layout(
                    title=f"Live Metrics - Preset: {preset.upper()} Frame #{st.session_state['m2_frame_count']}",
                    height=200, margin=dict(t=30, b=20, l=40, r=40),
                    yaxis=dict(title='FPS'), yaxis2=dict(title='Latency', overlaying='y', side='right'),
                    legend=dict(orientation='h', y=1.1)
                )
                chart_ph.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Không lấy được frame từ webcam...")

        time.sleep(0.05)
        st.rerun()

    st.divider()

    # --- PHẦN 2: RECORD & ENCODE THỰC TẾ ---
    st.subheader("Phần 2 – So sánh H.264 Thực tế (Record → Encode → So sánh)")
    if not check_ffmpeg():
        st.error("FFmpeg chưa cài → không thể encode H.264 thực tế.")
        return

    col_rec1, col_rec2 = st.columns(2)
    with col_rec1:
        record_seconds = st.slider("Thời gian quay (giây)", 3, 15, 5)
    with col_rec2:
        record_presets = st.multiselect("Preset H.264 để so sánh", H264_PRESETS, default=['ultrafast', 'medium', 'slow'])

    record_btn = st.button("Quay & Encode", type="primary")

    if record_btn:
        if st.session_state['m2_running']:
            st.warning("Dừng Live Webcam trước khi quay clip!")
        elif not record_presets:
            st.warning("Chọn ít nhất 1 preset!")
        else:
            _record_and_encode(camera_index=int(camera_index), duration_sec=record_seconds, presets=record_presets, crf=23)

    if 'm2_record_results' in st.session_state:
        _display_record_results()

# =====================================================================
# HÀM XỬ LÝ RECORD & ENCODE
# =====================================================================
def _record_and_encode(camera_index: int, duration_sec: int, presets: list, crf: int = 23):
    tmp_dir = tempfile.mkdtemp(prefix="m2_rec_")
    raw_path = os.path.join(tmp_dir, 'raw_capture.mp4')

    progress = st.progress(0.0)
    status = st.empty()
    status.info(f"Đang quay... {duration_sec} giây")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error("Không mở được webcam!")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(raw_path, fourcc, fps, (w, h))

    start = time.time()
    frame_list = []
    while (time.time() - start) < duration_sec:
        ret, frame = cap.read()
        if ret:
            writer.write(frame)
            frame_list.append(frame)
        elapsed = time.time() - start
        progress.progress(min(elapsed / duration_sec, 0.3))

    cap.release()
    writer.release()
    status.info(f"Đã quay {len(frame_list)} frames. Đang encode...")

    from utils import get_size_mb, get_compression_ratio, get_encoding_fps, calc_psnr, calc_ssim
    results = []
    original_mb = get_size_mb(raw_path)

    for i, preset in enumerate(presets):
        out_path = os.path.join(tmp_dir, f'h264_{preset}_crf{crf}.mp4')
        status.info(f"Encoding preset: {preset} ({i+1}/{len(presets)})...")
        try:
            enc_time = encode_video(raw_path, out_path, preset, crf)
            enc_mb = get_size_mb(out_path)
            enc_fps = get_encoding_fps(raw_path, enc_time)
            comp_ratio = get_compression_ratio(original_mb, enc_mb)

            enc_info = get_video_info(out_path)
            bitrate_kbps = enc_info['bitrate'] / 1000 if enc_info else 0
            psnr = calc_psnr(raw_path, out_path, max_sec=10)
            ssim = calc_ssim(raw_path, out_path, max_sec=10)

            results.append({
                'Preset': preset, 'CRF': crf, 'Encoding FPS': enc_fps,
                'Encoding Time (s)': round(enc_time, 2),
                'Original Size (MB)': round(original_mb, 2),
                'Encoded Size (MB)': round(enc_mb, 2),
                'Compression Ratio': round(comp_ratio, 2),
                'Bitrate (kbps)': round(bitrate_kbps, 1),
                'PSNR (dB)': round(psnr, 2) if psnr else None,
                'SSIM': round(ssim, 4) if ssim else None,
                '_out_path': out_path,
            })
        except Exception as e:
            st.error(f"Lỗi preset {preset}: {e}")

        progress.progress(0.3 + 0.7 * (i+1) / len(presets))

    st.session_state['m2_record_results'] = pd.DataFrame(results)
    st.session_state['m2_raw_path'] = raw_path
    status.success("Hoàn thành! Xem kết quả bên dưới.")
    st.rerun()

# =====================================================================
# HÀM HIỂN THỊ KẾT QUẢ RECORD
# =====================================================================
def _display_record_results():
    df = st.session_state['m2_record_results']
    st.markdown("#### Kết quả H.264 Thực tế")
    display_cols = [c for c in df.columns if not c.startswith('_')]
    st.dataframe(df[display_cols].set_index('Preset'), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df, x='Preset', y='Encoding FPS', title='Encoding FPS', color='Encoding FPS', color_continuous_scale='RdYlGn', text=df['Encoding FPS'].apply(lambda x: f'{x:.1f}'))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(df, x='Preset', y='Encoded Size (MB)', title='Kích thước File (MB)', color='Encoded Size (MB)', color_continuous_scale='Reds_r', text=df['Encoded Size (MB)'].apply(lambda x: f'{x:.2f}'))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    df_q = df.copy()
    df_q['PSNR (dB)'] = pd.to_numeric(df_q['PSNR (dB)'], errors='coerce')
    if df_q['PSNR (dB)'].notna().any():
        fig = px.line(df_q, x='Preset', y='PSNR (dB)', title='PSNR (dB) - H.264 Thực tế', markers=True)
        st.plotly_chart(fig, use_container_width=True)

    csv = df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button("Tải CSV kết quả", csv, "webcam_results.csv", "text/csv")