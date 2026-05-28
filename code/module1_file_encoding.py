"""
module1_file_encoding.py — Module 1: Mã hóa File & Đánh giá Offline
======================================================================
Môn:     Nén và Mã hóa Đa phương tiện
Mã code: 2502q

Chức năng:
  - Upload file video
  - Chọn preset H.264 và CRF bằng thanh trượt
  - Chạy benchmark mã hóa tự động
  - Tính toán đầy đủ: FPS, File Size, Compression Ratio, Bitrate, PSNR, SSIM
  - Vẽ 4 biểu đồ Plotly so sánh preset
  - Xuất kết quả CSV

Được gọi từ app.py qua hàm render().
"""

import os
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    H264_PRESETS,
    calc_psnr,
    calc_ssim,
    check_ffmpeg,
    encode_video,
    get_encoding_fps,
    get_size_mb,
    get_video_info,
    run_benchmark,
)


# ─────────────────────────────────────────────────────────────────────────────
# HÀM CHÍNH
# ─────────────────────────────────────────────────────────────────────────────

def render():
    """Render toàn bộ giao diện Module 1."""

    st.header("Module 1 — Mã hóa File Video & Đánh giá Offline")
    st.markdown("""
    Module này mã hóa file video với nhiều **Preset H.264** và so sánh:

    | Metric | Ý nghĩa |
    |---|---|
    | **Encoding FPS** | Tốc độ encoder (frame/giây) — càng cao càng nhanh |
    | **Compression Ratio** | Bao nhiêu lần nhỏ hơn so với gốc |
    | **Bitrate (kbps)** | Dung lượng dữ liệu mỗi giây video |
    | **PSNR (dB)** | Chất lượng hình ảnh — cao hơn = tốt hơn (≥35dB là tốt) |
    | **SSIM** | Độ tương đồng cấu trúc (0–1) — gần 1 = giống gốc |
    """)

    # ── Kiểm tra FFmpeg ─────────────────────────────────────────────────────
    if not check_ffmpeg():
        st.error("**FFmpeg chưa được cài đặt!** Xem hướng dẫn cài đặt trong README.md")
        with st.expander("Hướng dẫn cài FFmpeg nhanh"):
            st.code("""
# Windows (dùng Chocolatey):
choco install ffmpeg

# macOS (dùng Homebrew):
brew install ffmpeg

# Ubuntu/Debian:
sudo apt update && sudo apt install -y ffmpeg

# Kiểm tra:
ffmpeg -version
            """, language="bash")
        return

    st.success("FFmpeg đã sẵn sàng")
    st.divider()

    # ── BƯỚC 1: Upload video ─────────────────────────────────────────────────
    st.subheader("Bước 1 — Upload Video")

    uploaded = st.file_uploader(
        "Chọn file video (MP4, AVI, MOV, MKV...)",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'],
        help="Khuyến nghị dùng video 720p hoặc 1080p, thời lượng 10–60 giây để test nhanh.",
    )

    if uploaded is None:
        # Hiển thị gợi ý video mẫu
        st.info("""
          Upload một file video để bắt đầu.

        **Gợi ý dataset mẫu (miễn phí):**
        - [Pexels Videos](https://www.pexels.com/videos/) — video stock HD miễn phí
        - [Big Buck Bunny](https://peach.blender.org/download/) — video test phổ biến
        - [Blender Demo Files](https://www.blender.org/download/demo-files/)
        """)
        return

    # Lưu file upload vào thư mục tạm
    tmp_dir    = tempfile.mkdtemp(prefix="enc_")
    input_path = os.path.join(tmp_dir, uploaded.name)
    with open(input_path, 'wb') as f:
        f.write(uploaded.getbuffer())

    # Lưu path vào session để dùng lại sau
    st.session_state['m1_input_path'] = input_path
    st.session_state['m1_tmp_dir']    = tmp_dir

    # Hiển thị thông tin video gốc
    info = get_video_info(input_path)
    if info:
        st.subheader("Thông tin Video Gốc")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Độ phân giải", f"{info['width']}×{info['height']}")
        c2.metric("FPS phát lại", f"{info['fps']}")
        c3.metric("Thời lượng",   f"{info['duration']:.1f} s")
        c4.metric("Kích thước",   f"{get_size_mb(input_path):.2f} MB")
        c5.metric("Codec gốc",    info['codec'].upper())
    else:
        st.warning("Không đọc được thông tin video. Hãy thử file khác.")

    st.divider()

    # ── BƯỚC 2: Cấu hình mã hóa ─────────────────────────────────────────────
    st.subheader("Bước 2 — Cấu hình Mã hóa H.264")

    col_left, col_right = st.columns(2)

    with col_left:
        selected_presets = st.multiselect(
            "Chọn Preset để so sánh",
            options=H264_PRESETS,
            default=['ultrafast', 'medium', 'slow'],
            help=(
                "**ultrafast** → nhanh nhất, file to, chất lượng thấp\n\n"
                "**veryslow**  → chậm nhất, file nhỏ, chất lượng cao\n\n"
                "Chọn 3–5 preset để thấy rõ sự khác biệt."
            ),
        )

    with col_right:
        crf_value = st.slider(
            "CRF — Constant Rate Factor",
            min_value=0, max_value=51, value=23, step=1,
            help=(
                "**0**  → lossless (file rất to)\n\n"
                "**18** → chất lượng rất cao (mắt không nhận ra sự khác biệt)\n\n"
                "**23** → mặc định FFmpeg — cân bằng tốt\n\n"
                "**28** → chấp nhận được, file nhỏ\n\n"
                "**51** → tệ nhất"
            ),
        )
        # Hiển thị mức độ chất lượng tương ứng
        if crf_value <= 18:
            st.success(f"CRF {crf_value} — Chất lượng rất cao (file to)")
        elif crf_value <= 26:
            st.info(f"CRF {crf_value} — Chất lượng tốt (cân bằng)")
        elif crf_value <= 35:
            st.warning(f"CRF {crf_value} — Chất lượng trung bình (file nhỏ)")
        else:
            st.error(f"CRF {crf_value} — Chất lượng thấp (nhiều artifact)")

    if not selected_presets:
        st.warning("Hãy chọn ít nhất 1 preset!")
        return

    st.divider()

    # ── BƯỚC 3: Chạy benchmark ───────────────────────────────────────────────
    st.subheader("Bước 3 — Chạy Benchmark")

    # Ước tính thời gian chạy
    n = len(selected_presets)
    st.caption(f"Sẽ mã hóa **{n} preset**. Mỗi preset mất khoảng 10–60 giây tùy video và máy tính.")

    run_btn = st.button("Bắt đầu Benchmark", type="primary", use_container_width=True)

    if run_btn:
        output_dir  = os.path.join(tmp_dir, 'outputs')
        progress    = st.progress(0.0)
        status      = st.empty()
        results_raw = []

        original_mb = get_size_mb(input_path)

        for step, preset in enumerate(selected_presets):
            status.markdown(f"**Đang xử lý:** `{preset}` ({step+1}/{n})...")
            out_path = os.path.join(output_dir, f'enc_{preset}_crf{crf_value}.mp4')
            os.makedirs(output_dir, exist_ok=True)

            try:
                # Mã hóa
                enc_time = encode_video(input_path, out_path, preset, crf_value)

                # Metrics cơ bản
                enc_mb     = get_size_mb(out_path)
                enc_fps    = get_encoding_fps(input_path, enc_time)
                comp_ratio = original_mb / enc_mb if enc_mb > 0 else 0

                # Bitrate
                enc_info     = get_video_info(out_path)
                bitrate_kbps = enc_info['bitrate'] / 1000 if enc_info else 0

                # PSNR & SSIM (tính trên 20s đầu để tiết kiệm thời gian)
                status.markdown(f"Đang tính PSNR/SSIM cho `{preset}`...")
                psnr = calc_psnr(input_path, out_path, max_sec=20)
                ssim = calc_ssim(input_path, out_path, max_sec=20)

                results_raw.append({
                    'Preset':             preset,
                    'CRF':                crf_value,
                    'Encoding FPS':       enc_fps,
                    'Encoding Time (s)':  round(enc_time, 2),
                    'Original Size (MB)': round(original_mb, 2),
                    'Encoded Size (MB)':  round(enc_mb, 2),
                    'Compression Ratio':  round(comp_ratio, 2),
                    'Bitrate (kbps)':     round(bitrate_kbps, 1),
                    'PSNR (dB)':          round(psnr, 2) if psnr else None,
                    'SSIM':               round(ssim, 4) if ssim else None,
                    '_output_path':       out_path,
                })

            except Exception as e:
                st.error(f"Lỗi preset `{preset}`: {e}")
                results_raw.append({'Preset': preset, 'Error': str(e)})

            progress.progress((step + 1) / n)

        status.success("Benchmark hoàn thành!")

        if results_raw:
            df = pd.DataFrame(results_raw)
            st.session_state['m1_results'] = df

    # ── BƯỚC 4: Hiển thị kết quả ─────────────────────────────────────────────
    if 'm1_results' not in st.session_state:
        return

    df = st.session_state['m1_results']
    st.divider()
    st.subheader("Bước 4 — Kết quả")

    # Bảng tổng hợp (không hiện cột nội bộ _output_path)
    display_cols = [c for c in df.columns if not c.startswith('_')]
    st.dataframe(df[display_cols].set_index('Preset'), use_container_width=True)

    # Nút tải CSV
    csv_bytes = df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label    = "Tải CSV kết quả",
        data     = csv_bytes,
        file_name= f"benchmark_crf{df['CRF'].iloc[0] if 'CRF' in df else 23}.csv",
        mime     = "text/csv",
    )

    st.divider()

    # ── BƯỚC 5: Biểu đồ ──────────────────────────────────────────────────────
    st.subheader("Bước 5 — Biểu đồ So sánh")

    tab_fps, tab_size, tab_quality, tab_bitrate = st.tabs([
        "⚡ Encoding FPS", "Kích thước File", "PSNR & SSIM", "Bitrate",
    ])

    # ── Tab 1: Encoding FPS ──────────────────────────────────────────────────
    with tab_fps:
        st.markdown("**FPS mã hóa** cho biết encoder xử lý bao nhiêu frame/giây. Cao hơn = nhanh hơn.")
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

        # Encoding time
        fig2 = px.line(
            df, x='Preset', y='Encoding Time (s)',
            title='Thời gian mã hóa theo Preset (thấp hơn = nhanh hơn)',
            markers=True,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("**ultrafast** nhanh hơn nhiều so với **slow** nhưng hiệu quả nén kém hơn.")

    # ── Tab 2: File Size ─────────────────────────────────────────────────────
    with tab_size:
        st.markdown("So sánh kích thước file gốc vs file đã nén.")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Gốc', x=df['Preset'], y=df['Original Size (MB)'],
            marker_color='steelblue', opacity=0.6,
        ))
        fig.add_trace(go.Bar(
            name='Đã nén', x=df['Preset'], y=df['Encoded Size (MB)'],
            marker_color='tomato',
            text=df['Encoded Size (MB)'].apply(lambda x: f'{x:.2f} MB'),
            textposition='outside',
        ))
        fig.update_layout(title='Kích thước File (MB) — Gốc vs Đã nén', barmode='group')
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(
            df, x='Preset', y='Compression Ratio',
            title='Tỉ lệ nén theo Preset (cao hơn = nén nhiều hơn)',
            markers=True,
        )
        fig2.update_traces(line_color='coral', line_width=2)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Preset chậm hơn (slow, veryslow) nén hiệu quả hơn → file nhỏ hơn với cùng CRF.")

    # ── Tab 3: PSNR & SSIM ───────────────────────────────────────────────────
    with tab_quality:
        st.markdown("**PSNR** và **SSIM** đo chất lượng hình ảnh sau mã hóa so với gốc.")

        df_q = df.copy()
        df_q['PSNR (dB)'] = pd.to_numeric(df_q['PSNR (dB)'], errors='coerce')
        df_q['SSIM']       = pd.to_numeric(df_q['SSIM'],       errors='coerce')

        has_psnr = df_q['PSNR (dB)'].notna().any()
        has_ssim = df_q['SSIM'].notna().any()

        if has_psnr or has_ssim:
            fig = go.Figure()
            if has_psnr:
                fig.add_trace(go.Scatter(
                    x=df_q['Preset'], y=df_q['PSNR (dB)'],
                    name='PSNR (dB)', mode='lines+markers',
                    line=dict(color='royalblue', width=2),
                    yaxis='y1',
                ))
            if has_ssim:
                fig.add_trace(go.Scatter(
                    x=df_q['Preset'], y=df_q['SSIM'],
                    name='SSIM', mode='lines+markers',
                    line=dict(color='tomato', width=2, dash='dash'),
                    yaxis='y2',
                ))
            fig.update_layout(
                title='PSNR và SSIM theo Preset',
                yaxis =dict(title='PSNR (dB)',  side='left'),
                yaxis2=dict(title='SSIM (0–1)', side='right', overlaying='y'),
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Thêm vùng tham chiếu cho PSNR
            col_a, col_b = st.columns(2)
            col_a.info("**PSNR tham chiếu:**\n\n> 40 dB: Rất tốt\n\n30–40 dB: Tốt\n\n< 30 dB: Kém")
            col_b.info("**SSIM tham chiếu:**\n\n> 0.95: Rất tốt\n\n0.85–0.95: Tốt\n\n< 0.85: Kém")
        else:
            st.info("PSNR/SSIM chưa được tính. Hãy chạy lại benchmark.")

    # ── Tab 4: Bitrate ───────────────────────────────────────────────────────
    with tab_bitrate:
        st.markdown("**Bitrate** (kbps) = lượng dữ liệu video mỗi giây. Thấp hơn = nén hiệu quả hơn.")
        fig = px.bar(
            df, x='Preset', y='Bitrate (kbps)',
            title='Bitrate (kbps) theo Preset',
            color='Bitrate (kbps)',
            color_continuous_scale='Blues_r',
            text=df['Bitrate (kbps)'].apply(lambda x: f'{x:.0f}'),
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Với cùng CRF, preset chậm hơn → bitrate thấp hơn → nén hiệu quả hơn.")

    # ── BƯỚC 6: So sánh chất lượng hình ảnh (Qualitative) ───────────────────
    st.divider()
    st.subheader("Bước 6 — So sánh Hình ảnh (Qualitative)")
    st.markdown("Trích xuất 1 frame từ video gốc và video đã nén để so sánh bằng mắt.")

    if '_output_path' in df.columns:
        # Lọc preset có file output
        valid_rows = df[df['_output_path'].notna()]
        if not valid_rows.empty:
            preset_compare = st.selectbox(
                "Chọn preset để so sánh với bản gốc:",
                valid_rows['Preset'].tolist(),
            )
            row = valid_rows[valid_rows['Preset'] == preset_compare].iloc[0]

            col_orig, col_enc = st.columns(2)
            try:
                # Trích frame ở giây thứ 2
                orig_frame = _extract_frame(input_path, second=2)
                enc_frame  = _extract_frame(row['_output_path'], second=2)

                col_orig.image(orig_frame, caption=f"Gốc ({get_size_mb(input_path):.2f} MB)", use_container_width=True)
                col_enc.image(
                    enc_frame,
                    caption=f"Preset: {preset_compare} — {row['Encoded Size (MB)']:.2f} MB",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Không trích được frame: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# HÀM PHỤ
# ─────────────────────────────────────────────────────────────────────────────

def _extract_frame(video_path: str, second: float = 2.0) -> np.ndarray | None:
    """
    Trích xuất 1 frame từ video tại thời điểm `second` (giây).
    Returns: ảnh RGB numpy array hoặc None.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * second))
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None
