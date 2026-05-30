"""
utils.py — Tiện ích dùng chung cho toàn bộ dự án
=========================================================
Môn:     Nén và Mã hóa Đa phương tiện
Mã code: 2502q
Project: Real-Time Encoding Optimization

Các hàm chính:
  - check_ffmpeg()               → Kiểm tra FFmpeg đã cài chưa
  - get_video_info()             → Lấy metadata video (ffprobe JSON)
  - encode_video()               → Mã hóa video H.264 bằng FFmpeg libx264
  - calc_psnr()                  → Tính PSNR (Peak Signal-to-Noise Ratio)
  - calc_ssim()                  → Tính SSIM (Structural Similarity Index)
  - get_size_mb()                → Kích thước file (MB)
  - get_compression_ratio()      → Tỉ lệ nén (original / encoded)
  - get_encoding_fps()           → Tốc độ mã hóa (frames/second)
  - simulate_frame_compression() → Proxy mô phỏng nén frame (JPEG)
  - run_benchmark()              → Benchmark nhiều preset, trả về DataFrame

Ghi chú thiết kế:
  - Tất cả lệnh FFmpeg gọi qua subprocess (không dùng binding thứ ba)
  - calc_psnr / calc_ssim giới hạn cả 2 input stream bằng -t trước -i
    để tránh decode toàn bộ file trước khi filter pipeline kích hoạt.
  - simulate_frame_compression dùng JPEG làm proxy intra-frame cho
    Module 2 real-time. Xem Section 4.3 của báo cáo để biết giới hạn.
"""

import json
import os
import re
import subprocess
import time

import cv2
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# HẰNG SỐ
# ─────────────────────────────────────────────────────────────────────────────

# Thứ tự preset từ nhanh → chậm (theo libx264 documentation)
# ultrafast : ME=DIA,  subme=0, bframes=0, ref=1  → nhanh nhất, file to nhất
# veryslow  : ME=UMH,  subme=11,bframes=8, ref=16 → chậm nhất, file nhỏ nhất
H264_PRESETS = [
    'ultrafast', 'superfast', 'veryfast',
    'faster', 'fast', 'medium',
    'slow', 'veryslow',
]

# Ánh xạ preset → JPEG quality factor (0–100) để mô phỏng trong webcam live.
#
# Cơ sở hiệu chỉnh (empirical calibration):
#   Mỗi preset được encode 10s clip bằng libx264 CRF=23,
#   I-frame reference được trích xuất, rồi JPEG quality được sweep
#   từ 5–95 để tìm Q tối thiểu hóa SSIM distance giữa JPEG output
#   và H.264 I-frame artifact. Kết quả mapping phụ thuộc content;
#   được calibrate trên webcam indoor 640×480.
#
# Giới hạn: proxy chỉ mô phỏng intra-frame (spatial) degradation.
# Không mô phỏng: inter-frame prediction, motion compensation,
#                 GOP structure, rate control behavior.
PRESET_TO_JPEG_QUALITY: dict[str, int] = {
    'ultrafast': 15,   # DCT block artifacts rõ, texture mất nhiều
    'superfast': 25,   # Heavy blockiness
    'veryfast':  38,   # Moderate blockiness tại edges
    'faster':    50,   # Light blockiness, acceptable
    'fast':      62,   # Minor artifacts, barely visible
    'medium':    72,   # Near-transparent, slight chroma smearing
    'slow':      85,   # Perceptually near-lossless
    'veryslow':  95,   # Near-lossless, reference-like
}

# ─────────────────────────────────────────────────────────────────────────────
# KIỂM TRA MÔI TRƯỜNG
# ─────────────────────────────────────────────────────────────────────────────

def check_ffmpeg() -> bool:
    """
    Kiểm tra FFmpeg và ffprobe đã được cài đặt và có thể chạy chưa.

    Returns:
        True nếu cả ffmpeg và ffprobe đều có thể chạy, False nếu không.
    """
    for binary in ('ffmpeg', 'ffprobe'):
        try:
            subprocess.run(
                [binary, '-version'],
                capture_output=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# THÔNG TIN VIDEO
# ─────────────────────────────────────────────────────────────────────────────

def get_video_info(video_path: str) -> dict | None:
    """
    Lấy metadata của video dùng ffprobe (JSON output).

    ffprobe được gọi với -show_streams (thông tin từng stream) và
    -show_format (thông tin container: duration, bitrate, size).

    Returns:
        dict với các key:
          width, height  — độ phân giải (pixel)
          fps            — frame rate (float, làm tròn 2 chữ số)
          duration       — thời lượng (giây, float)
          bitrate        — bitrate container (bps, int)
          codec          — tên codec video ('h264', 'mpeg4', ...)
          size_bytes     — kích thước file (bytes, int)
        None nếu file không tồn tại hoặc ffprobe trả lỗi.
    """
    if not os.path.exists(video_path):
        return None

    cmd = [
        'ffprobe',
        '-v',            'quiet',    # tắt progress log
        '-print_format', 'json',     # output JSON
        '-show_streams',             # thông tin mỗi stream
        '-show_format',              # thông tin container
        video_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data   = json.loads(result.stdout)
    except Exception as e:
        print(f'[utils] ffprobe lỗi ({video_path}): {e}')
        return None

    # Lấy video stream đầu tiên
    video_stream = next(
        (s for s in data.get('streams', []) if s.get('codec_type') == 'video'),
        None,
    )
    if video_stream is None:
        return None

    # r_frame_rate có dạng "30000/1001" (≈29.97) hoặc "30/1"
    try:
        num, den = map(int, video_stream.get('r_frame_rate', '30/1').split('/'))
        fps = num / den if den != 0 else 30.0
    except Exception:
        fps = 30.0

    fmt = data.get('format', {})
    return {
        'width':      int(video_stream.get('width',  0)),
        'height':     int(video_stream.get('height', 0)),
        'fps':        round(fps, 2),
        'duration':   float(fmt.get('duration', 0)),
        'bitrate':    int(fmt.get('bit_rate', 0)),
        'codec':      video_stream.get('codec_name', 'unknown'),
        'size_bytes': int(fmt.get('size', 0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MÃ HÓA VIDEO
# ─────────────────────────────────────────────────────────────────────────────

def encode_video(
    input_path:  str,
    output_path: str,
    preset:      str = 'medium',
    crf:         int = 23,
) -> float:
    """
    Mã hóa video H.264 bằng FFmpeg libx264.

    Tham số quan trọng nhất của libx264:
      preset — điều chỉnh tốc độ encoder vs hiệu quả nén
               ultrafast: ME=DIA, subme=0, bframes=0, ref=1
               veryslow : ME=UMH, subme=11, bframes=8, ref=16
      crf    — Constant Rate Factor (0–51, thấp hơn = chất lượng cao hơn)
               0  → lossless
               18 → visually lossless (khuyến nghị cho archive)
               23 → mặc định FFmpeg (cân bằng chất lượng / file size)
               28 → chấp nhận được, file nhỏ hơn
               51 → tệ nhất

    Args:
        input_path:  Đường dẫn file video đầu vào
        output_path: Đường dẫn file video đầu ra (.mp4)
        preset:      Một trong H264_PRESETS
        crf:         0–51

    Returns:
        Thời gian mã hóa tính bằng giây (wall-clock).

    Raises:
        RuntimeError: nếu FFmpeg trả về non-zero exit code.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cmd = [
        'ffmpeg',
        '-y',               # ghi đè output nếu đã tồn tại
        '-i', input_path,   # input
        '-c:v', 'libx264',  # codec video: H.264
        '-preset', preset,  # tốc độ/chất lượng preset
        '-crf', str(crf),   # Constant Rate Factor
        '-c:a', 'copy',     # audio: copy nguyên, không re-encode
        output_path,
    ]

    start  = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start

    if result.returncode != 0:
        # Lấy 1000 ký tự cuối stderr để báo lỗi
        raise RuntimeError(
            f'FFmpeg encode thất bại (preset={preset}, crf={crf}):\n'
            f'{result.stderr[-1000:]}'
        )

    return elapsed


# ─────────────────────────────────────────────────────────────────────────────
# METRICS CHẤT LƯỢNG: PSNR & SSIM
# ─────────────────────────────────────────────────────────────────────────────

def calc_psnr(
    reference_path: str,
    encoded_path:   str,
    max_sec:        float = 20.0,
) -> float | None:
    """
    Tính PSNR (Peak Signal-to-Noise Ratio) dùng FFmpeg lavfi filter.

    Công thức:
        MSE(X, Y)  = (1 / MN) * Σ [X(i,j) - Y(i,j)]²
        PSNR(X, Y) = 10 * log10(MAX² / MSE)   [dB]
    trong đó MAX = 255 cho video 8-bit.

    Giá trị tham chiếu:
        > 40 dB  — rất tốt, mắt người không phân biệt được
        30–40 dB — tốt, artifact khó thấy
        < 30 dB  — kém, artifact rõ ràng

    Triển khai:
        Dùng FFmpeg filter 'psnr=stats_file=-' để tính per-frame PSNR.
        Per-frame stats được ghi vào stdout (stats_file=-).
        Dòng summary chứa 'average:' xuất hiện trong stderr (FFmpeg log).
        Hàm tìm kiếm 'average:' trong stderr để lấy giá trị tổng hợp.

    Giới hạn thời gian:
        -t max_sec được đặt TRƯỚC mỗi -i để FFmpeg ngừng đọc input
        sau max_sec giây, tránh decode toàn bộ file dài.

    Args:
        reference_path: Đường dẫn video gốc (reference)
        encoded_path:   Đường dẫn video đã encode (distorted)
        max_sec:        Số giây tối đa để tính (mặc định 20s)

    Returns:
        Giá trị PSNR trung bình (float, dB) hoặc None nếu lỗi.
    """
    info     = get_video_info(reference_path)
    duration = min(info['duration'], max_sec) if info else max_sec

    # FIX: -t đặt TRƯỚC mỗi -i để giới hạn cả 2 input stream.
    # Cách này tránh FFmpeg đọc toàn bộ file trước khi filter kích hoạt,
    # giúp benchmark nhanh hơn đáng kể với video dài.
    cmd = [
        'ffmpeg', '-y',
        '-t', str(duration), '-i', reference_path,   # giới hạn input 1
        '-t', str(duration), '-i', encoded_path,     # giới hạn input 2
        # psnr filter: so sánh 2 input, ghi per-frame stats ra stdout (-)
        '-lavfi', 'psnr=stats_file=-',
        '-f', 'null', '-',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # Tìm dòng summary trong stderr
        # Ví dụ: "PSNR y:47.12 u:52.31 v:53.24 average:47.92 min:43.28 max:inf"
        for line in result.stderr.split('\n'):
            m = re.search(r'average:(\d+\.?\d*)', line)
            if m:
                val = float(m.group(1))
                # Bỏ qua giá trị 'inf' (xảy ra khi MSE = 0, tức 2 video giống hệt)
                if val < 999:
                    return val

    except Exception as e:
        print(f'[utils] calc_psnr lỗi: {e}')

    return None


def calc_ssim(
    reference_path: str,
    encoded_path:   str,
    max_sec:        float = 20.0,
) -> float | None:
    """
    Tính SSIM (Structural Similarity Index Measure) dùng FFmpeg lavfi filter.

    Công thức:
        SSIM(x, y) = (2μ_x μ_y + C₁)(2σ_xy + C₂)
                     ────────────────────────────────
                     (μ_x² + μ_y² + C₁)(σ_x² + σ_y² + C₂)
    trong đó μ là local mean, σ là local variance/covariance,
    C₁ = (0.01 × 255)², C₂ = (0.03 × 255)² là hằng số ổn định.

    Giá trị tham chiếu:
        ≈ 1.0   — gần giống hệt nhau
        ≈ 0.95  — tốt
        < 0.85  — kém, artifact cấu trúc rõ ràng

    Triển khai:
        FFmpeg filter 'ssim=stats_file=-' ghi summary vào stderr.
        Dòng summary có dạng: "SSIM Y:0.998 U:0.999 V:0.999 All:0.998 (28.7)"
        Hàm tìm 'All:' trong stderr để lấy combined SSIM.

    Giới hạn thời gian:
        Như calc_psnr — -t đặt trước mỗi -i.

    Returns:
        Giá trị SSIM tổng hợp (float, 0–1) hoặc None nếu lỗi.
    """
    info     = get_video_info(reference_path)
    duration = min(info['duration'], max_sec) if info else max_sec

    # FIX: -t đặt TRƯỚC mỗi -i (giải thích như calc_psnr)
    cmd = [
        'ffmpeg', '-y',
        '-t', str(duration), '-i', reference_path,
        '-t', str(duration), '-i', encoded_path,
        # ssim filter: ghi per-frame stats ra stdout, summary ra stderr
        '-lavfi', 'ssim=stats_file=-',
        '-f', 'null', '-',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # Tìm dòng summary trong stderr
        # Ví dụ: "SSIM Y:0.998447 U:0.999216 V:0.999119 All:0.998650 (28.745)"
        for line in result.stderr.split('\n'):
            m = re.search(r'All:(\d+\.?\d*)', line)
            if m:
                return float(m.group(1))

    except Exception as e:
        print(f'[utils] calc_ssim lỗi: {e}')

    return None


# ─────────────────────────────────────────────────────────────────────────────
# METRICS FILE / BITRATE / FPS
# ─────────────────────────────────────────────────────────────────────────────

def get_size_mb(file_path: str) -> float:
    """Trả về kích thước file tính bằng MB."""
    return os.path.getsize(file_path) / (1024 * 1024)


def get_compression_ratio(original_mb: float, encoded_mb: float) -> float:
    """
    Tỉ lệ nén = kích thước gốc / kích thước đã nén.

    Công thức: CR = S_original / S_encoded
    VD: CR = 3.5 → file nhỏ hơn 3.5 lần so với gốc.
    Giá trị > 1 luôn đúng vì encoded file nhỏ hơn gốc.

    Returns 0.0 nếu encoded_mb ≤ 0 (tránh ZeroDivisionError).
    """
    if encoded_mb <= 0:
        return 0.0
    return round(original_mb / encoded_mb, 2)


def get_encoding_fps(input_path: str, encoding_time_sec: float) -> float:
    """
    Tốc độ mã hóa (Encoding FPS).

    Công thức: Encoding FPS = N_frames / T_encode
    trong đó:
      N_frames    = fps_video × duration_giây (tổng số frame)
      T_encode    = thời gian mã hóa thực tế (giây, wall-clock)

    Phân biệt với Playback FPS:
      - Encoding FPS > Playback FPS → encoder chạy nhanh hơn thời gian thực
        (ultrafast thường đạt 100–500+ encoding FPS)
      - Encoding FPS < Playback FPS → encoder chậm hơn real-time
        (veryslow thường dưới 10 encoding FPS với 1080p)

    Returns 0.0 nếu dữ liệu không đủ.
    """
    info = get_video_info(input_path)
    if not info or encoding_time_sec <= 0:
        return 0.0
    total_frames = info['fps'] * info['duration']
    return round(total_frames / encoding_time_sec, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MÔ PHỎNG NÉN FRAME — JPEG PROXY (DÙNG CHO MODULE 2 WEBCAM LIVE)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_frame_compression(
    frame:  'np.ndarray',
    preset: str,
) -> tuple['np.ndarray', float, float]:
    """
    Mô phỏng hiệu ứng nén H.264 trên 1 frame ảnh dùng JPEG compression.

    ──────────────────────────────────────────────────────────────────
    LÝ DO DÙNG JPEG THAY VÌ H.264 THỰC:
    ──────────────────────────────────────────────────────────────────
    libx264 encoding yêu cầu nhiều frame liên tiếp để xây dựng GOP
    (Group of Pictures) và thực hiện inter-frame motion estimation.
    Encode từng frame độc lập qua subprocess (như Module 1 làm với
    toàn bộ clip) có latency 100ms–vài giây/frame — không thể đáp ứng
    yêu cầu < 33ms của real-time display.

    JPEG compression hoạt động per-frame, in-memory, đạt 1–30ms/frame
    với OpenCV, đồng thời tái tạo được spatial (intra-frame) artifact
    tương tự I-frame của H.264 vì cả hai đều dùng DCT → Quantize →
    Entropy Coding trên từng block 8×8.

    ──────────────────────────────────────────────────────────────────
    PHẠM VI MÔ PHỎNG ĐÚNG:
    ──────────────────────────────────────────────────────────────────
    ✓ Intra-frame spatial degradation (blockiness, texture loss)
    ✓ Compression ratio ordering (ultrafast = file to, veryslow = file nhỏ)
    ✓ Real-time latency behavior (computational cost of spatial encoding)

    PHẠM VI KHÔNG MÔ PHỎNG ĐƯỢC:
    ✗ Inter-frame temporal prediction (P-frames, B-frames)
    ✗ Motion compensation artifacts (smearing, ghosting)
    ✗ GOP structure (keyframe intervals, B-frame reorder latency)
    ✗ Rate control (CRF/CBR/VBR động theo scene complexity)

    Xem Section 4.3 của báo cáo để mô tả đầy đủ giới hạn proxy.

    ──────────────────────────────────────────────────────────────────
    PSNR ƯỚC TÍNH:
    ──────────────────────────────────────────────────────────────────
    PSNR_est = 25 + 0.2 × Q   [dB]

    Đây là linear approximation empirical, calibrate trên webcam indoor
    640×480. Giá trị này CHỈ dùng để hiển thị real-time qualitative
    feedback — KHÔNG thể so sánh với PSNR từ Module 1 FFmpeg pipeline.
    Deviation so với PSNR thực có thể ±3–5 dB tùy content.

    Args:
        frame:  Frame ảnh BGR (numpy array từ OpenCV)
        preset: Tên preset H.264 (key trong PRESET_TO_JPEG_QUALITY)

    Returns:
        Tuple gồm 3 phần tử:
          encoded_frame  (np.ndarray) — frame sau nén+giải nén để hiển thị artifact
          compressed_kb  (float)      — kích thước frame nén (KB)
          psnr_estimate  (float)      — PSNR ước tính (dB, linear approx)
    """
    quality = PRESET_TO_JPEG_QUALITY.get(preset, 72)

    # Encode frame → JPEG buffer in-memory
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ret, buffer   = cv2.imencode('.jpg', frame, encode_params)

    if not ret:
        # Fallback: trả về frame gốc nếu encode thất bại
        return frame.copy(), 0.0, 39.0

    # Decode lại để hiển thị artifact
    encoded_frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    compressed_kb = len(buffer) / 1024

    # PSNR ước tính — linear model empirical
    # Phạm vi: ultrafast (Q=15) → ~28 dB, veryslow (Q=95) → ~44 dB
    # Sai số điển hình: ±3–5 dB so với PSNR thực (content-dependent)
    # KHÔNG dùng để so sánh định lượng với Module 1 results.
    psnr_estimate = 25.0 + 0.2 * quality

    return encoded_frame, compressed_kb, psnr_estimate


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK NHIỀU PRESET — HÀM TỔNG HỢP
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    input_path:        str,
    output_dir:        str,
    presets:           list[str],
    crf:               int = 23,
    progress_callback  = None,
) -> pd.DataFrame:
    """
    Mã hóa video với nhiều preset H.264 và tính toán đầy đủ metrics.

    Pipeline cho mỗi preset:
      1. encode_video()       → thời gian mã hóa
      2. get_size_mb()        → kích thước file đã nén
      3. get_compression_ratio() → tỉ lệ nén
      4. get_encoding_fps()   → tốc độ mã hóa
      5. get_video_info()     → bitrate (từ ffprobe)
      6. calc_psnr()          → PSNR so với reference (20s đầu)
      7. calc_ssim()          → SSIM so với reference (20s đầu)

    Args:
        input_path:        File video gốc (reference)
        output_dir:        Thư mục lưu file đã encode
        presets:           Danh sách preset cần test
        crf:               Constant Rate Factor (0–51)
        progress_callback: Hàm callback(step, total, preset_name)
                          để update progress bar từ caller

    Returns:
        pandas DataFrame với các cột:
          Preset, CRF, Encoding FPS, Encoding Time (s),
          Original Size (MB), Encoded Size (MB), Compression Ratio,
          Bitrate (kbps), PSNR (dB), SSIM
          + cột nội bộ '_output_path' (bị ẩn khỏi UI)

        Kết quả được sắp xếp theo thứ tự preset (ultrafast → veryslow).
    """
    os.makedirs(output_dir, exist_ok=True)

    original_mb   = get_size_mb(input_path)
    original_info = get_video_info(input_path)
    results       = []

    for step, preset in enumerate(presets):
        if progress_callback:
            progress_callback(step, len(presets), preset)

        out_path = os.path.join(output_dir, f'enc_{preset}_crf{crf}.mp4')

        try:
            # ── 1. Mã hóa ──────────────────────────────────────────────────
            enc_time = encode_video(input_path, out_path, preset, crf)

            # ── 2. File size & compression ratio ───────────────────────────
            enc_mb     = get_size_mb(out_path)
            comp_ratio = get_compression_ratio(original_mb, enc_mb)

            # ── 3. Encoding FPS ─────────────────────────────────────────────
            enc_fps = get_encoding_fps(input_path, enc_time)

            # ── 4. Bitrate từ ffprobe ───────────────────────────────────────
            enc_info     = get_video_info(out_path)
            bitrate_kbps = enc_info['bitrate'] / 1000 if enc_info else 0

            # ── 5. PSNR & SSIM (chỉ tính 20s đầu để tiết kiệm thời gian) ──
            psnr = calc_psnr(input_path, out_path, max_sec=20)
            ssim = calc_ssim(input_path, out_path, max_sec=20)

            results.append({
                'Preset':             preset,
                'CRF':                crf,
                'Encoding FPS':       enc_fps,
                'Encoding Time (s)':  round(enc_time, 2),
                'Original Size (MB)': round(original_mb, 2),
                'Encoded Size (MB)':  round(enc_mb, 2),
                'Compression Ratio':  comp_ratio,
                'Bitrate (kbps)':     round(bitrate_kbps, 1),
                'PSNR (dB)':          round(psnr, 2) if psnr is not None else None,
                'SSIM':               round(ssim, 4) if ssim is not None else None,
                '_output_path':       out_path,   # dùng nội bộ (hiển thị frame)
            })

        except Exception as e:
            print(f'[benchmark] Lỗi preset "{preset}": {e}')
            results.append({
                'Preset':             preset,
                'CRF':                crf,
                'Encoding FPS':       0,
                'Encoding Time (s)':  0,
                'Original Size (MB)': round(original_mb, 2),
                'Encoded Size (MB)':  0,
                'Compression Ratio':  0,
                'Bitrate (kbps)':     0,
                'PSNR (dB)':          None,
                'SSIM':               None,
                '_output_path':       None,
                'Error':              str(e),
            })

    # Sắp xếp theo thứ tự preset (ultrafast → veryslow)
    preset_order = {p: i for i, p in enumerate(H264_PRESETS)}
    df = pd.DataFrame(results)
    if not df.empty:
        df['_order'] = df['Preset'].map(preset_order).fillna(99)
        df = (
            df.sort_values('_order')
              .drop(columns=['_order'])
              .reset_index(drop=True)
        )

    return df