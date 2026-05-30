"""
module2_webcam_live.py — Module 2: Webcam Live & Real-Time Dashboard
======================================================================
Môn:     Nén và Mã hóa Đa phương tiện
Mã code: 2502q

Chức năng:
  Phần 1 — Live Webcam Proxy Simulation:
    - Capture webcam qua daemon thread (WebcamWorker)
    - JPEG compression làm proxy intra-frame H.264 artifact
    - Hiển thị song song: video gốc | video đã nén
    - Real-time dashboard: FPS tức thời, Latency, Frame Size, PSNR ước tính

  Phần 2 — Record & Encode (H.264 thực):
    - Quay clip ngắn từ webcam (3–15 giây)
    - Encode thực bằng libx264 với nhiều preset
    - Tính PSNR, SSIM, Compression Ratio, Bitrate thực
    - So sánh preset qua biểu đồ và bảng số liệu

Kiến trúc threading:
  - WebcamWorker chạy trên daemon thread riêng (background)
  - Main thread (Streamlit) đọc frame qua threading.Lock
  - Tránh blocking UI do hardware I/O latency
  - Xem Section 5.3 của báo cáo để biết chi tiết thiết kế

Ghi chú tương thích hệ điều hành:
  - Linux   : dùng cv2.CAP_V4L2 để giảm latency buffer
  - Windows : dùng cv2.CAP_DSHOW để tránh hang
  - macOS   : dùng cv2.CAP_AVFOUNDATION
  - Fallback: cv2.CAP_ANY nếu platform không nhận diện được
  Phát hiện tự động qua platform.system().
"""

import os
import platform
import tempfile
import threading
import time

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils import (
    H264_PRESETS,
    PRESET_TO_JPEG_QUALITY,
    calc_psnr,
    calc_ssim,
    check_ffmpeg,
    encode_video,
    get_compression_ratio,
    get_encoding_fps,
    get_size_mb,
    get_video_info,
    simulate_frame_compression,
)


# =====================================================================
# HELPER: CHỌN BACKEND CAMERA THEO HỆ ĐIỀU HÀNH
# =====================================================================

def _get_camera_backend() -> int:
    """
    Trả về backend OpenCV phù hợp với hệ điều hành hiện tại.

    FIX so với phiên bản cũ:
      cv2.CAP_V4L2 là Video4Linux2 API — chỉ tồn tại trên Linux.
      Dùng nó trên Windows/macOS gây crash với AttributeError hoặc
      cv2.error ngay khi khởi tạo VideoCapture.

    Backend được chọn:
      Linux   → cv2.CAP_V4L2  (thấp latency, bỏ qua frame buffer cũ)
      Windows → cv2.CAP_DSHOW (DirectShow — native Windows API)
      macOS   → cv2.CAP_AVFOUNDATION (AVFoundation — native macOS API)
      Khác    → cv2.CAP_ANY   (để OpenCV tự chọn)

    Returns:
        int — giá trị backend constant của OpenCV.
    """
    os_name = platform.system()
    if os_name == 'Linux':
        return cv2.CAP_V4L2
    elif os_name == 'Windows':
        return cv2.CAP_DSHOW
    elif os_name == 'Darwin':   # macOS
        return cv2.CAP_AVFOUNDATION
    else:
        return cv2.CAP_ANY


# =====================================================================
# THREADING: WEBCAM CAPTURE WORKER
# =====================================================================

class WebcamWorker:
    """
    Background daemon thread để capture frame từ webcam liên tục.

    Mục đích:
        Tách biệt hardware I/O (webcam polling) khỏi UI rendering thread.
        Streamlit re-executes toàn bộ script mỗi khi có interaction;
        nếu cv2.VideoCapture().read() nằm trong main thread, mỗi rerun
        sẽ block cho đến khi frame mới sẵn sàng — gây jitter nghiêm trọng.

    Thiết kế:
        - Daemon thread: tự động kết thúc khi main thread đóng
        - threading.Lock: bảo vệ shared frame buffer khỏi race condition
        - frame.copy() trong get_frame(): tránh tearing do concurrent write
        - session_state persistence: giữ kết nối camera qua Streamlit reruns

    Tham chiếu: Section 5.3 của báo cáo — Asynchronous Multithreading Design.
    """

    def __init__(self, camera_index: int = 0):
        """
        Khởi tạo WebcamWorker.

        Args:
            camera_index: Index của camera (0 = camera đầu tiên,
                         thường là webcam tích hợp hoặc USB đầu tiên).
        """
        backend = _get_camera_backend()
        self.cap = cv2.VideoCapture(camera_index, backend)

        # Cấu hình capture parameters
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Giảm buffer size để luôn lấy frame mới nhất (tránh lag)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # MJPEG thường cho throughput cao hơn YUYV trên nhiều webcam
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass   # Một số camera không hỗ trợ MJPG — bỏ qua, dùng default

        self.frame    = None           # Buffer lưu frame mới nhất
        self.lock     = threading.Lock()
        self._running = False
        self._thread  = None

    def start(self) -> None:
        """Kiểm tra camera và khởi động daemon thread."""
        if not self.cap.isOpened():
            raise RuntimeError(
                'Không thể mở webcam! '
                'Kiểm tra: (1) camera đã kết nối chưa, '
                '(2) camera_index đúng chưa, '
                '(3) camera có đang dùng bởi app khác không.'
            )
        self._running = True
        self._thread  = threading.Thread(
            target=self._capture_loop,
            daemon=True,          # tự kết thúc khi main thread đóng
        )
        self._thread.start()

    def stop(self) -> None:
        """Dừng capture loop và giải phóng camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)   # chờ tối đa 2 giây
        if self.cap.isOpened():
            self.cap.release()

    def get_frame(self) -> 'np.ndarray | None':
        """
        Lấy bản copy của frame mới nhất (thread-safe).

        Trả về copy thay vì reference để decoupling hoàn toàn giữa
        UI render thread và background write thread, tránh partial
        frame corruption hay visual tearing.
        """
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def _capture_loop(self) -> None:
        """
        Vòng lặp capture chạy trên daemon thread.

        Liên tục đọc frame từ camera và cập nhật self.frame.
        Dùng Lock để đảm bảo thread-safe khi main thread đọc đồng thời.
        Sleep ngắn khi read() thất bại để tránh busy-waiting tiêu tốn CPU.
        """
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)   # 10ms backoff khi camera tạm thời fail


# =====================================================================
# KHỞI TẠO SESSION STATE
# =====================================================================

def _init_session() -> None:
    """Khởi tạo các key session_state với giá trị mặc định."""
    defaults = {
        'm2_running':      False,
        'm2_frame_count':  0,
        'm2_start_time':   0.0,
        'm2_fps_history':  [],     # history FPS để vẽ chart
        'm2_lat_history':  [],     # history latency (ms) per frame
        'm2_size_history': [],     # history frame size (KB)
        'm2_worker':       None,
        'm2_preset':       'medium',
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =====================================================================
# HÀM TÍNH FPS TỨC THỜI (INSTANTANEOUS)
# =====================================================================

def _compute_instantaneous_fps(lat_history: list[float], window: int = 10) -> float:
    """
    Tính FPS tức thời từ moving average của latency gần nhất.

    FIX so với phiên bản cũ:
      Phiên bản cũ tính FPS = total_frames / total_elapsed_time.
      Đây là average FPS từ đầu session — hội tụ chậm, không phản ánh
      performance hiện tại khi user thay đổi preset.

      Phiên bản mới: FPS_inst = 1000 / mean(latency[-window:])
      Cho biết FPS tức thời dựa trên compression latency 10 frame gần nhất.
      Phản hồi ngay khi preset thay đổi.

    Args:
        lat_history: Danh sách latency per frame (ms)
        window:      Số frame gần nhất để tính moving average

    Returns:
        FPS tức thời (float). Trả 0.0 nếu history rỗng.
    """
    if not lat_history:
        return 0.0
    recent = lat_history[-window:] if len(lat_history) >= window else lat_history
    avg_lat_ms = sum(recent) / len(recent)
    return round(1000.0 / avg_lat_ms, 1) if avg_lat_ms > 0 else 0.0


# =====================================================================
# HÀM RENDER CHÍNH (ENTRY POINT TỪ app.py)
# =====================================================================

def render() -> None:
    """
    Render toàn bộ giao diện Module 2.

    Cấu trúc:
      Phần 1 — Live Webcam Proxy: Capture → JPEG compress → display song song
      Phần 2 — Record & Encode: Quay clip → H.264 thực → metrics chính xác
    """
    _init_session()

    st.header('Module 2 — Webcam Live & Real-Time Dashboard')
    st.markdown("""
    **Phần 1:** Hiển thị real-time video gốc vs video nén song song — thay đổi preset để thấy ngay artifact.

    **Phần 2:** Quay clip ngắn → encode H.264 thực tế → đo PSNR/SSIM/Bitrate/FPS chính xác.
    """)

    st.divider()

    # ──────────────────────────────────────────────────────────────────
    # PHẦN 1 — LIVE WEBCAM VỚI MÔ PHỎNG PRESET
    # ──────────────────────────────────────────────────────────────────
    st.subheader('Phần 1 — Live Webcam với Mô phỏng Preset H.264')

    # Giải thích proxy approach
    with st.expander('ℹ️ Ghi chú về phương pháp mô phỏng (đọc trước khi dùng)', expanded=False):
        st.markdown("""
        **Module 2 Phần 1 dùng JPEG compression làm proxy mô phỏng H.264**, vì:
        - H.264 encoding thực cần nhiều frame liên tiếp (GOP) và mất 100ms–vài giây/frame
        - JPEG hoạt động per-frame, in-memory, đạt 1–30ms/frame

        **Proxy mô phỏng đúng:**
        - ✅ Intra-frame spatial degradation (blockiness, texture loss)
        - ✅ Thứ tự compression ratio (ultrafast = frame to, veryslow = frame nhỏ)

        **Proxy KHÔNG mô phỏng được:**
        - ❌ Inter-frame temporal prediction (P-frames, B-frames)
        - ❌ Motion compensation artifacts
        - ❌ Rate control behavior (CRF/CBR/VBR)

        **→ Để đo PSNR/SSIM/Bitrate chính xác: dùng Phần 2.**
        """)

    # Cấu hình
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

    with col_cfg1:
        camera_index = st.number_input(
            'Index Camera',
            min_value=0, max_value=9, value=0,
            help='0 = camera đầu tiên (thường là webcam tích hợp).',
        )

    with col_cfg2:
        live_preset = st.selectbox(
            'Preset H.264 (mô phỏng)',
            options=H264_PRESETS,
            index=H264_PRESETS.index('medium'),
            help='Thay đổi preset để thấy artifact thay đổi ngay lập tức.',
        )
        st.session_state['m2_preset'] = live_preset

    with col_cfg3:
        jpeg_q   = PRESET_TO_JPEG_QUALITY.get(live_preset, 72)
        psnr_est = 25 + 0.2 * jpeg_q
        st.metric(
            'JPEG Quality tương ứng',
            f'{jpeg_q}/100',
            delta=f'PSNR est. ~{psnr_est:.0f} dB',
            delta_color='normal' if jpeg_q >= 62 else 'inverse',
            help='Quality factor trong bảng ánh xạ empirical. Xem Section 4.3.',
        )

    # Nút điều khiển
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col1:
        start_btn = st.button(
            'Bắt đầu Webcam',
            disabled=st.session_state['m2_running'],
            type='primary',
            use_container_width=True,
        )
    with btn_col2:
        stop_btn = st.button(
            'Dừng',
            disabled=not st.session_state['m2_running'],
            use_container_width=True,
        )
    with btn_col3:
        if st.session_state['m2_running']:
            st.success('🔴 LIVE — Webcam đang chạy')
        else:
            st.info("Nhấn 'Bắt đầu Webcam' để bắt đầu")

    # Xử lý Start
    if start_btn:
        try:
            worker = WebcamWorker(camera_index=int(camera_index))
            worker.start()
            st.session_state['m2_worker']      = worker
            st.session_state['m2_running']     = True
            st.session_state['m2_frame_count'] = 0
            st.session_state['m2_start_time']  = time.time()
            st.session_state['m2_fps_history'] = []
            st.session_state['m2_lat_history'] = []
            st.session_state['m2_size_history'] = []
            st.rerun()
        except RuntimeError as e:
            st.error(f'Lỗi mở webcam: {e}')
        except Exception as e:
            st.error(f'Lỗi không xác định: {e}')

    # Xử lý Stop
    if stop_btn:
        worker = st.session_state.get('m2_worker')
        if worker is not None:
            worker.stop()
            st.session_state['m2_worker']  = None
        st.session_state['m2_running'] = False
        st.rerun()

    # Placeholder cho video frames
    vid_col1, vid_col2 = st.columns(2)
    orig_ph = vid_col1.empty()
    enc_ph  = vid_col2.empty()

    # Label tiêu đề cho 2 cột
    vid_col1.caption('**Video gốc** — Không nén')
    vid_col2.caption(f'**Video nén** — Preset: `{live_preset.upper()}` (JPEG Q={PRESET_TO_JPEG_QUALITY.get(live_preset,72)})')

    st.markdown('---')

    # Placeholder cho metrics
    metrics_row = st.columns(4)
    fps_ph  = metrics_row[0].empty()
    lat_ph  = metrics_row[1].empty()
    size_ph = metrics_row[2].empty()
    psnr_ph = metrics_row[3].empty()

    # Placeholder cho chart
    chart_ph = st.empty()

    # ── VÒNG LẶP CHÍNH ──────────────────────────────────────────────
    if st.session_state['m2_running']:
        worker = st.session_state.get('m2_worker')

        if worker is None:
            st.session_state['m2_running'] = False
            st.rerun()
            return

        frame = worker.get_frame()

        if frame is not None:
            preset = st.session_state.get('m2_preset', 'medium')

            # ── Đo latency: chỉ tính thời gian compression cycle ──────
            t_start = time.perf_counter()
            enc_frame, size_kb, psnr_est = simulate_frame_compression(frame, preset)
            latency_ms = (time.perf_counter() - t_start) * 1000

            # ── Cập nhật history ──────────────────────────────────────
            st.session_state['m2_frame_count'] += 1
            MAX_HIST = 60   # giữ tối đa 60 điểm

            st.session_state['m2_lat_history'].append(round(latency_ms, 2))
            st.session_state['m2_size_history'].append(round(size_kb, 1))

            for key in ('m2_lat_history', 'm2_size_history'):
                if len(st.session_state[key]) > MAX_HIST:
                    st.session_state[key] = st.session_state[key][-MAX_HIST:]

            # ── FPS tức thời (instantaneous) từ latency 10 frame gần nhất
            fps_inst = _compute_instantaneous_fps(
                st.session_state['m2_lat_history'], window=10
            )
            st.session_state['m2_fps_history'].append(round(fps_inst, 1))
            if len(st.session_state['m2_fps_history']) > MAX_HIST:
                st.session_state['m2_fps_history'] = \
                    st.session_state['m2_fps_history'][-MAX_HIST:]

            # ── Hiển thị frames ──────────────────────────────────────
            orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            enc_rgb  = cv2.cvtColor(enc_frame, cv2.COLOR_BGR2RGB)
            orig_ph.image(orig_rgb, use_container_width=True)
            enc_ph.image(enc_rgb,  use_container_width=True)

            # ── Hiển thị metrics ─────────────────────────────────────
            fps_ph.metric(
                'FPS (tức thời)',
                f'{fps_inst:.1f}',
                delta='real-time ✓' if fps_inst >= 30 else f'< 30fps',
                delta_color='normal' if fps_inst >= 30 else 'inverse',
                help='FPS tính từ moving average latency 10 frame gần nhất.',
            )
            lat_ph.metric(
                'Latency',
                f'{latency_ms:.1f} ms',
                delta='< 33ms ✓' if latency_ms < 33 else 'lag',
                delta_color='normal' if latency_ms < 33 else 'inverse',
                help='Thời gian của bước JPEG compression. Không bao gồm camera capture và UI render.',
            )
            size_ph.metric(
                'Frame Size',
                f'{size_kb:.1f} KB',
                help='Kích thước JPEG buffer sau nén (proxy cho H.264 bitrate).',
            )
            psnr_ph.metric(
                'PSNR ước tính',
                f'~{psnr_est:.0f} dB',
                help='Linear approx: 25 + 0.2×Q. Chỉ dùng cho qualitative feedback. Sai số ±3-5dB.',
            )

            # ── Live chart: FPS và Latency ───────────────────────────
            if len(st.session_state['m2_fps_history']) > 3:
                hist_df = pd.DataFrame({
                    'Frame':       range(len(st.session_state['m2_fps_history'])),
                    'FPS':         st.session_state['m2_fps_history'],
                    'Latency (ms)': st.session_state['m2_lat_history']
                                    [-len(st.session_state['m2_fps_history']):],
                })

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist_df['Frame'], y=hist_df['FPS'],
                    name='FPS (tức thời)',
                    mode='lines',
                    line=dict(color='limegreen', width=1.5),
                ))
                fig.add_trace(go.Scatter(
                    x=hist_df['Frame'], y=hist_df['Latency (ms)'],
                    name='Latency (ms)',
                    mode='lines',
                    line=dict(color='tomato', width=1.5, dash='dot'),
                    yaxis='y2',
                ))
                # Đường ngưỡng real-time
                fig.add_hline(
                    y=30, line_dash='dash', line_color='yellow',
                    annotation_text='30 FPS threshold',
                    annotation_position='top right',
                )

                fig.update_layout(
                    title=(
                        f'Live Metrics — Preset: {preset.upper()}  '
                        f'(Frame #{st.session_state["m2_frame_count"]})'
                    ),
                    height=220,
                    margin=dict(t=40, b=20, l=40, r=60),
                    yaxis=dict(title='FPS', rangemode='nonnegative'),
                    yaxis2=dict(
                        title='Latency (ms)',
                        overlaying='y', side='right',
                        rangemode='nonnegative',
                    ),
                    legend=dict(orientation='h', y=1.15),
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                chart_ph.plotly_chart(fig, use_container_width=True)

        else:
            st.warning('Chưa nhận được frame từ webcam. Đợi 1–2 giây...')

        # Delay ngắn rồi rerun để duy trì display loop
        time.sleep(0.05)
        st.rerun()

    st.divider()

    # ──────────────────────────────────────────────────────────────────
    # PHẦN 2 — RECORD & ENCODE THỰC TẾ (H.264 libx264)
    # ──────────────────────────────────────────────────────────────────
    st.subheader('Phần 2 — So sánh H.264 Thực tế (Record → Encode → Metrics)')

    st.markdown("""
    Quay một đoạn clip ngắn từ webcam, sau đó encode bằng **libx264 thực tế**
    với các preset được chọn. Kết quả cho PSNR, SSIM, Bitrate và Encoding FPS
    **chính xác** — không phải ước tính proxy như Phần 1.
    """)

    if not check_ffmpeg():
        st.error('FFmpeg chưa cài → không thể encode H.264 thực tế. Xem README.md.')
        return

    col_rec1, col_rec2 = st.columns(2)
    with col_rec1:
        record_seconds = st.slider('Thời gian quay (giây)', min_value=3, max_value=15, value=5)
    with col_rec2:
        record_presets = st.multiselect(
            'Preset H.264 để so sánh',
            options=H264_PRESETS,
            default=['ultrafast', 'medium', 'slow'],
            help='Chọn 2–4 preset để thấy sự khác biệt rõ nhất.',
        )

    record_btn = st.button('Quay & Encode', type='primary', use_container_width=True)

    if record_btn:
        if st.session_state['m2_running']:
            st.warning('Vui lòng dừng Live Webcam (Phần 1) trước khi quay clip!')
        elif not record_presets:
            st.warning('Chọn ít nhất 1 preset!')
        else:
            _record_and_encode(
                camera_index=int(camera_index),
                duration_sec=record_seconds,
                presets=record_presets,
                crf=23,
            )

    if 'm2_record_results' in st.session_state:
        _display_record_results()


# =====================================================================
# HÀM XỬ LÝ RECORD & ENCODE
# =====================================================================

def _record_and_encode(
    camera_index: int,
    duration_sec: int,
    presets:      list[str],
    crf:          int = 23,
) -> None:
    """
    Quay clip từ webcam và encode bằng libx264 với các preset được chọn.

    Pipeline:
      1. Mở camera, capture frames trong duration_sec giây
      2. Lưu raw clip bằng OpenCV VideoWriter (mp4v codec)
      3. Với mỗi preset: encode_video() → tính metrics đầy đủ
      4. Lưu kết quả vào session_state để _display_record_results() hiển thị

    Ghi chú về reference video:
      Raw clip được lưu bằng mp4v (lossy MPEG-4 Part 2), không phải YUV lossless.
      PSNR/SSIM tính so với mp4v clip này (không phải raw pixel gốc).
      Đây là limitation được ghi nhận trong Section 7 của báo cáo.
      Giá trị PSNR có thể cao hơn thực tế do transcoding effect.
    """
    tmp_dir  = tempfile.mkdtemp(prefix='m2_rec_')
    raw_path = os.path.join(tmp_dir, 'raw_capture.mp4')

    progress = st.progress(0.0, text='Chuẩn bị...')
    status   = st.empty()

    # ── Bước 1: Quay clip ────────────────────────────────────────────
    status.info(f'Đang quay... {duration_sec} giây. Nhìn vào camera.')

    backend = _get_camera_backend()
    cap = cv2.VideoCapture(int(camera_index), backend)

    if not cap.isOpened():
        st.error('Không mở được webcam! Kiểm tra kết nối camera.')
        return

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(raw_path, fourcc, fps, (w, h))

    start       = time.time()
    frame_list  = []

    while (time.time() - start) < duration_sec:
        ret, frame = cap.read()
        if ret:
            writer.write(frame)
            frame_list.append(frame)
        elapsed = time.time() - start
        progress.progress(
            min(elapsed / duration_sec * 0.3, 0.3),
            text=f'Quay... {elapsed:.1f}/{duration_sec}s ({len(frame_list)} frames)',
        )

    cap.release()
    writer.release()

    if not frame_list:
        st.error('Không capture được frame nào. Kiểm tra camera.')
        return

    status.info(f'Đã quay {len(frame_list)} frames ({len(frame_list)/fps:.1f}s). Đang encode...')

    # ── Bước 2: Encode với từng preset ──────────────────────────────
    original_mb = get_size_mb(raw_path)
    results     = []

    for i, preset in enumerate(presets):
        out_path = os.path.join(tmp_dir, f'h264_{preset}_crf{crf}.mp4')
        status.info(f'Encoding `{preset}` ({i + 1}/{len(presets)})...')

        try:
            enc_time     = encode_video(raw_path, out_path, preset, crf)
            enc_mb       = get_size_mb(out_path)
            enc_fps_val  = get_encoding_fps(raw_path, enc_time)
            comp_ratio   = get_compression_ratio(original_mb, enc_mb)

            enc_info     = get_video_info(out_path)
            bitrate_kbps = enc_info['bitrate'] / 1000 if enc_info else 0

            psnr = calc_psnr(raw_path, out_path, max_sec=10)
            ssim = calc_ssim(raw_path, out_path, max_sec=10)

            results.append({
                'Preset':             preset,
                'CRF':                crf,
                'Encoding FPS':       enc_fps_val,
                'Encoding Time (s)':  round(enc_time, 2),
                'Original Size (MB)': round(original_mb, 2),
                'Encoded Size (MB)':  round(enc_mb, 2),
                'Compression Ratio':  round(comp_ratio, 2),
                'Bitrate (kbps)':     round(bitrate_kbps, 1),
                'PSNR (dB)':          round(psnr, 2) if psnr is not None else None,
                'SSIM':               round(ssim, 4) if ssim is not None else None,
                '_out_path':          out_path,
            })

        except Exception as e:
            st.error(f'Lỗi preset `{preset}`: {e}')
            results.append({
                'Preset': preset, 'Error': str(e),
                'Encoding FPS': 0, 'PSNR (dB)': None, 'SSIM': None,
            })

        progress.progress(
            0.3 + 0.7 * (i + 1) / len(presets),
            text=f'Hoàn thành {i + 1}/{len(presets)} presets',
        )

    st.session_state['m2_record_results'] = pd.DataFrame(results)
    st.session_state['m2_raw_path']        = raw_path
    status.success('Hoàn thành! Xem kết quả bên dưới.')
    st.rerun()


# =====================================================================
# HÀM HIỂN THỊ KẾT QUẢ RECORD
# =====================================================================

def _display_record_results() -> None:
    """
    Hiển thị kết quả H.264 thực tế từ Part 2.

    Bao gồm:
      - Bảng số liệu đầy đủ (tất cả metrics)
      - Biểu đồ Encoding FPS (bar chart)
      - Biểu đồ File Size so sánh (bar chart)
      - Biểu đồ PSNR theo preset (line chart)
      - Biểu đồ SSIM theo preset (line chart)
      - Nút tải CSV
    """
    df = st.session_state['m2_record_results']

    st.markdown('#### Kết quả H.264 Thực tế (libx264)')

    # Ghi chú về reference
    st.caption(
        '**Ghi chú:** PSNR/SSIM được tính so với clip raw mp4v (không phải YUV lossless). '
        'Xem Section 7 — Limitations của báo cáo.'
    )

    # Bảng tổng hợp
    display_cols = [c for c in df.columns if not c.startswith('_') and c != 'Error']
    st.dataframe(
        df[display_cols].set_index('Preset'),
        use_container_width=True,
    )

    # Nút download CSV
    csv = df[display_cols].to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label='Tải CSV kết quả',
        data=csv,
        file_name='webcam_h264_results.csv',
        mime='text/csv',
    )

    st.markdown('---')

    # ── Biểu đồ 1 & 2 ─────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df, x='Preset', y='Encoding FPS',
            title='Encoding FPS theo Preset (cao hơn = nhanh hơn)',
            color='Encoding FPS',
            color_continuous_scale='RdYlGn',
            text=df['Encoding FPS'].apply(lambda x: f'{x:.1f}'),
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            df, x='Preset', y='Encoded Size (MB)',
            title='Kích thước File Đã Nén (MB)',
            color='Encoded Size (MB)',
            color_continuous_scale='Reds_r',
            text=df['Encoded Size (MB)'].apply(lambda x: f'{x:.2f}'),
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Biểu đồ PSNR & SSIM ───────────────────────────────────────
    df_q = df.copy()
    df_q['PSNR (dB)'] = pd.to_numeric(df_q['PSNR (dB)'], errors='coerce')
    df_q['SSIM']      = pd.to_numeric(df_q['SSIM'],       errors='coerce')

    has_psnr = df_q['PSNR (dB)'].notna().any()
    has_ssim = df_q['SSIM'].notna().any()

    if has_psnr or has_ssim:
        fig = go.Figure()

        if has_psnr:
            fig.add_trace(go.Scatter(
                x=df_q['Preset'], y=df_q['PSNR (dB)'],
                name='PSNR (dB)', mode='lines+markers',
                line=dict(color='royalblue', width=2),
                marker=dict(size=8),
                yaxis='y1',
            ))
            # Ngưỡng tham chiếu PSNR
            fig.add_hline(
                y=35, line_dash='dash', line_color='orange',
                annotation_text='35 dB (acceptable threshold)',
                annotation_position='top left',
            )

        if has_ssim:
            fig.add_trace(go.Scatter(
                x=df_q['Preset'], y=df_q['SSIM'],
                name='SSIM', mode='lines+markers',
                line=dict(color='tomato', width=2, dash='dash'),
                marker=dict(size=8),
                yaxis='y2',
            ))

        fig.update_layout(
            title='PSNR và SSIM theo Preset (H.264 thực tế, CRF=23)',
            yaxis=dict(
                title='PSNR (dB)',
                side='left',
                rangemode='tozero',
            ),
            yaxis2=dict(
                title='SSIM (0–1)',
                side='right',
                overlaying='y',
                range=[0.85, 1.0],
            ),
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tóm tắt quan sát
        if has_psnr and len(df_q['PSNR (dB)'].dropna()) >= 2:
            psnr_min = df_q['PSNR (dB)'].min()
            psnr_max = df_q['PSNR (dB)'].max()
            psnr_range = psnr_max - psnr_min
            st.info(
                f'**Quan sát PSNR:** Dao động {psnr_range:.2f} dB giữa các preset '
                f'(từ {psnr_min:.2f} dB → {psnr_max:.2f} dB). '
                f'Biến thiên nhỏ xác nhận CRF=23 đang duy trì quality target '
                f'nhất quán bất kể preset — đúng với lý thuyết CRF rate control.'
            )
    else:
        st.info('PSNR/SSIM chưa có dữ liệu. Kiểm tra FFmpeg đã cài đủ chưa.')